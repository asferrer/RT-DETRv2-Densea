#!/usr/bin/env python
"""
TensorRT inference script for RT-DETRv2 models.

Features:
- Handles image, video, and camera/stream inputs automatically.
- Optimized for TensorRT engines (.engine files), especially with FP16.
- Supports engines built with dynamic shapes (dynamic batch size).
- Allocates necessary host (pinned) and device (GPU) memory using PyCUDA.
- Performs asynchronous inference using CUDA streams.
- Includes preprocessing and postprocessing steps.
    - Preprocessing: Resizes, normalizes (0-1), transposes to CHW.
    - Postprocessing: Filters detections by confidence threshold.
      (NOTE: Assumes the engine outputs absolute xyxy coordinates based on previous debugging).
- Minimalist bounding box drawing using OpenCV.
- Optional real-time display (`--show`) or saving results to file (default).
- Basic benchmarking (warm-up, latency, FPS).
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Important: Initializes CUDA context and handles cleanup
import time
from tqdm import tqdm
import colorsys

# TensorRT Logger Severity
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Default maximum number of detections expected from the model output.
# Used for allocating output buffers if the engine profile information is unreliable.
MAX_OUTPUT_DETECTIONS = 300

# Default input dimensions (adjust if your model differs)
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def generate_color_map(num_classes: int) -> list:
    """Generates a list of distinct BGR colors for drawing."""
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        # Convert to BGR for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

def draw_detections_cv2(image, labels, boxes, scores, label_map=None, color_map=None):
    """
    Draws bounding boxes and labels on an image using OpenCV with a minimalist style.
    """
    if label_map is None: label_map = {}
    if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0:
        return image # Return original image if no detections
    img_h, img_w = image.shape[:2]
    for i, box in enumerate(boxes):
        label_id = int(labels[i])
        score = scores[i]
        x1, y1, x2, y2 = map(int, box) # Ensure coordinates are integers
        if color_map and 0 <= label_id < len(color_map):
            color = color_map[label_id]
        else:
            color = (0, 255, 0) # Default to green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        class_name = label_map.get(label_id, str(label_id))
        text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tag_x = x1
        tag_y = y1 - text_h - baseline - 3 # Default position above
        if tag_y < 0: # If tag goes above image, place it inside the box
            tag_y = y1 + baseline + 1
        cv2.rectangle(image, (tag_x, tag_y), (tag_x + text_w + 2, tag_y + text_h + baseline), color, -1)
        cv2.putText(image, text, (tag_x + 1, tag_y + text_h), font, font_scale, (0,0,0), thickness)
    return image

def auto_detect_mode(input_source: str) -> str:
    """
    Automatically detects the input mode (image, video, camera/stream).
    """
    # ... (Function content remains the same) ...
    if not isinstance(input_source, str):
         try: int(input_source); return "camera"
         except (ValueError, TypeError): raise TypeError(f"Input must be path (str) or camera index (int), got {type(input_source)}")
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']; video_exts = ['.mp4', '.avi', '.mkv', '.mov']
    if os.path.exists(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext in image_exts: return "image"
        elif ext in video_exts: return "video"
        else:
            cap_test = cv2.VideoCapture(input_source)
            if cap_test.isOpened(): cap_test.release(); return "video"
            else: print(f"Warning: Unknown extension '{ext}', treating as image."); return "image"
    else:
        try: int(input_source); return "camera"
        except ValueError:
            if input_source.startswith(("rtsp://", "http://", "https://")): return "camera"
            else: raise ValueError(f"Input source not found or not recognized: {input_source}")

def preprocess_image(image_bgr, target_height, target_width):
    """
    Preprocesses an image (BGR NumPy array) for RT-DETR inference.
    """
    original_h, original_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if original_h > target_height else cv2.INTER_LINEAR
    img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=interpolation)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    batch_images = np.expand_dims(img_chw, axis=0)
    batch_sizes = np.array([[original_w, original_h]], dtype=np.int64)
    return np.ascontiguousarray(batch_images), np.ascontiguousarray(batch_sizes)

def postprocess(labels, boxes, scores, original_hw, confidence_thr, input_shape=(640, 640)):
    """
    Postprocesses the raw outputs from the RT-DETR model.
    NOTE: Assumes 'boxes' are already absolute xyxy coordinates.
    """
    keep = scores > confidence_thr
    num_after_filter = np.sum(keep)
    if num_after_filter == 0: return np.array([]), np.array([]), np.array([])
    filtered_labels = labels[keep]
    filtered_scores = scores[keep]
    filtered_boxes_abs = boxes[keep]
    return filtered_labels.astype(int), filtered_boxes_abs.astype(int), filtered_scores


def load_engine(engine_file_path):
    """Loads a TensorRT engine from file."""
    if not os.path.exists(engine_file_path): raise FileNotFoundError(f"TensorRT engine file not found: {engine_file_path}")
    print(f"Loading engine from: {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None: raise RuntimeError("Failed to load TensorRT engine.")
    print("TensorRT engine loaded successfully."); return engine

def allocate_buffers(engine, batch_size=1):
    """
    Allocates host (pinned) and device (GPU) buffers for TRT engine bindings.
    """
    inputs = []; outputs = []; bindings = []; stream = cuda.Stream()
    print(f"Allocating buffers for max batch size {batch_size} (based on profile or assumption)...")
    for binding_idx in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(binding_idx)
        shape_from_engine = engine.get_tensor_shape(binding_name)
        dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        mode = engine.get_tensor_mode(binding_name)
        calculated_shape = None
        if mode == trt.TensorIOMode.INPUT:
            if -1 in shape_from_engine:
                try:
                    profile_shapes = engine.get_tensor_profile_shape(binding_name, 0)
                    shape = profile_shapes[2]; shape = tuple([batch_size if dim == -1 else dim for dim in shape])
                    calculated_shape = shape
                    print(f"  Binding '{binding_name}' (Input, Dynamic): Using max shape -> {shape}")
                except Exception as e: print(f"Error getting profile shape for input '{binding_name}': {e}"); raise
            else: calculated_shape = shape_from_engine; print(f"  Binding '{binding_name}' (Input, Static): Using shape {shape_from_engine}")
        else: # OUTPUT
            print(f"  Binding '{binding_name}' (Output): Shape from engine: {shape_from_engine}")
            if binding_name == "labels" or binding_name == "scores": shape = (batch_size, MAX_OUTPUT_DETECTIONS)
            elif binding_name == "boxes": shape = (batch_size, MAX_OUTPUT_DETECTIONS, 4)
            else: print(f"  Warning: Unknown output '{binding_name}'. Using shape from engine: {shape_from_engine}"); shape = shape_from_engine
            if -1 in shape: shape = tuple([batch_size if dim == -1 else dim for dim in shape])
            calculated_shape = shape; print(f"    Allocating based on assumed max detections: {calculated_shape}")
        size = trt.volume(calculated_shape) if calculated_shape else 0
        buffer_shape_info = calculated_shape
        if size <= 0: print(f"  Warning: Calc size <= 0 for '{binding_name}'. Allocating min size (1)."); size = 1; buffer_shape_info = shape_from_engine
        nbytes = size * np.dtype(dtype).itemsize
        try: host_mem = cuda.pagelocked_empty(size, dtype=dtype); device_mem = cuda.mem_alloc(nbytes)
        except Exception as e: print(f"Error allocating memory for {binding_name} (size {size}, {nbytes} bytes): {e}"); raise
        bindings.append(int(device_mem))
        buffer_info = {'host': host_mem, 'device': device_mem, 'shape': buffer_shape_info, 'dtype': dtype, 'name': binding_name, 'allocated_size': size}
        if mode == trt.TensorIOMode.INPUT: inputs.append(buffer_info)
        else: outputs.append(buffer_info)
        print(f"    Allocated {nbytes / (1024*1024):.2f} MB for '{binding_name}' (size: {size})")
    if not inputs: print("Warning: No input buffers allocated.");
    if not outputs: print("Warning: No output buffers allocated.");
    print("Buffer allocation complete."); return inputs, outputs, bindings, stream

def infer_tensorrt(context, engine, inputs, outputs, stream, batch_input_image, batch_orig_size):
    """
    Performs asynchronous inference using TensorRT context and pre-allocated buffers.
    Uses execute_async_v3.
    """
    images_buffer = next((inp for inp in inputs if inp['name'] == "images"), None)
    sizes_buffer = next((inp for inp in inputs if inp['name'] == "orig_target_sizes"), None)
    if not images_buffer or not sizes_buffer: raise ValueError("Input buffers not found.")
    try:
        image_data_size = batch_input_image.size; size_data_size = batch_orig_size.size
        images_buffer['host'].ravel()[:image_data_size] = batch_input_image.ravel()
        sizes_buffer['host'].ravel()[:size_data_size] = batch_orig_size.ravel()
    except Exception as e: print(f"Error copying to host buffers: {e}"); raise
    cuda.memcpy_htod_async(images_buffer['device'], images_buffer['host'], stream)
    cuda.memcpy_htod_async(sizes_buffer['device'], sizes_buffer['host'], stream)
    if not context.set_input_shape("images", batch_input_image.shape): print(f"Error set_input_shape 'images'"); return None
    if not context.set_input_shape("orig_target_sizes", batch_orig_size.shape): print(f"Error set_input_shape 'orig_target_sizes'"); return None
    for buf in inputs + outputs:
        if not context.set_tensor_address(buf['name'], buf['device']): print(f"Error set_tensor_address '{buf['name']}'"); return None
    if not context.execute_async_v3(stream_handle=stream.handle): print("Error execute_async_v3"); return None
    for out in outputs: cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    results = {}
    for out in outputs:
         try: output_shape = context.get_tensor_shape(out['name'])
         except Exception: output_shape = context.get_binding_shape(engine.get_binding_index(out['name']))
         valid_output_shape = True
         if any(dim < 0 for dim in output_shape):
              print(f"Warning: Invalid output shape {output_shape} for '{out['name']}'. Using alloc shape."); output_shape = out['shape']
              if any(dim < 0 for dim in output_shape): print(f"Error: Fallback shape invalid for '{out['name']}'."); results[out['name']] = None; valid_output_shape = False
         if valid_output_shape:
             expected_size = trt.volume(output_shape); allocated_size = out['allocated_size']
             if expected_size == 0 and allocated_size >= 1: print(f"Warning: Output '{out['name']}' shape {output_shape}. Flat buffer."); results[out['name']] = out['host'][:1]
             elif allocated_size >= expected_size: results[out['name']] = out['host'][:expected_size].reshape(output_shape)
             else: print(f"Error: Output buffer size mismatch '{out['name']}'. Alloc:{allocated_size}, Expected:{expected_size}"); results[out['name']] = None
    return results

def main(args):
    """Loads engine, allocates buffers, runs inference loop, handles output."""

    # --- Define Class Map and Colors Here ---
    # Example class map (COCO simplified) - Replace with your actual class map
    CLASS_MAP = {
        0: "Bottle", 1: "Can", 2: "Fishing_Net", 3: "Glove", 4: "Mask",
        5: "Metal_Debris", 6: "Plastic_Debris", 7: "Tire",
    }
    NUM_CLASSES = len(CLASS_MAP)
    label_map = CLASS_MAP # Use this variable now defined in main scope
    # Generate colors based on the number of classes
    cv_color_map = generate_color_map(NUM_CLASSES) # Use this variable now defined in main scope

    # --- Setup ---
    if not os.path.exists(args.engine_file): raise FileNotFoundError(f"TensorRT engine file not found: {args.engine_file}")
    mode = auto_detect_mode(args.input); print(f"Detected mode: {mode}")
    if not args.show: os.makedirs(args.save_dir, exist_ok=True); print(f"Output directory: {args.save_dir}")

    # --- Load Engine & Allocate Buffers ---
    engine = None; context = None; inputs = outputs = bindings = stream = None
    try:
        engine = load_engine(args.engine_file)
        context = engine.create_execution_context()
        if context is None: raise RuntimeError("Failed to create execution context.")
        inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size=1)
    except Exception as e: print(f"Error during setup: {e}"); sys.exit(1) # Exit if setup fails

    # --- Prepare Output Name ---
    try:
        input_name = os.path.splitext(os.path.basename(args.input))[0] if isinstance(args.input, str) else f"camera_{args.input}"
    except Exception: input_name = "output"
    engine_name = os.path.splitext(os.path.basename(args.engine_file))[0]
    output_base_name = f"{input_name}_{engine_name}"

    # --- Benchmarking and Display Setup ---
    warmup_runs = 3; inference_times = []; window_name = "TensorRT Inference"

    # --- Process Image ---
    if mode == "image":
        try:
            frame = cv2.imread(args.input)
            if frame is None: raise FileNotFoundError(f"Could not read image: {args.input}")
            print(f"Image loaded: {args.input}")
            input_image_batch, input_size_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
            print(f"Performing {warmup_runs} warm-up runs...")
            for _ in range(warmup_runs): _ = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
            print("Warm-up complete.")
            print("Performing inference...")
            start_time = time.perf_counter()
            trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
            end_time = time.perf_counter(); inference_times.append(end_time - start_time)
            if trt_outputs is None: raise RuntimeError("TensorRT inference failed.")
            labels_out = trt_outputs.get("labels"); boxes_out = trt_outputs.get("boxes"); scores_out = trt_outputs.get("scores")
            if labels_out is None or boxes_out is None or scores_out is None: raise ValueError("Inconsistent/missing output tensor.")
            if labels_out.ndim > 0 and labels_out.shape[0] == 1: labels_out = labels_out[0]
            if boxes_out.ndim > 0 and boxes_out.shape[0] == 1: boxes_out = boxes_out[0]
            if scores_out.ndim > 0 and scores_out.shape[0] == 1: scores_out = scores_out[0]
            original_wh = input_size_batch[0]
            labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, original_wh, args.threshold)
            print(f"Postprocessing complete. Final detections: {len(labels)}")
            if len(labels) == 0 and scores_out.size > 0: max_score = np.max(scores_out) if scores_out.size > 0 else 0; print(f"NOTE: No detections above threshold {args.threshold}. Max score: {max_score:.4f}.")
            elif len(labels) < 5 and len(labels) < len(scores_out): print(f"NOTE: Few detections ({len(labels)}). Lower --threshold (current: {args.threshold})")
            annotated_frame = draw_detections_cv2(frame.copy(), labels, boxes, scores, label_map, cv_color_map)
            if args.show:
                print("Displaying result. Press 'q' to close.")
                cv2.imshow(window_name, annotated_frame);
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                out_filename = f"{output_base_name}.jpg"; out_path = os.path.join(args.save_dir, out_filename)
                if cv2.imwrite(out_path, annotated_frame): print(f"Inferred image saved to: {out_path}")
                else: print(f"Error saving image to {out_path}")
        except Exception as e: print(f"Error processing image: {e}")

    # --- Process Video/Camera ---
    elif mode in ["video", "camera"]:
        cap = None; out_writer = None
        try:
            if mode == "video": cap = cv2.VideoCapture(args.input)
            else: cap = cv2.VideoCapture(int(args.input)) if args.input.isdigit() else cv2.VideoCapture(args.input)
            if not cap or not cap.isOpened(): raise IOError(f"Cannot open source: {args.input}")
            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30; print(f"Warning: Could not get FPS, defaulting to {fps}")
            print(f"Source opened: {args.input} ({w}x{h} @ {fps:.2f} FPS)")
            if not args.show:
                out_filename = f"{output_base_name}.mp4"; out_path = os.path.join(args.save_dir, out_filename)
                out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not out_writer.isOpened(): raise IOError(f"Cannot open VideoWriter for {out_path}")
                print(f"Saving output video to: {out_path}")
            else: cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            ret, frame = cap.read();
            if not ret: raise IOError("Cannot read first frame.")
            print(f"Performing {warmup_runs} warm-up runs...")
            input_image_batch, input_size_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
            for _ in range(warmup_runs): _ = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
            print("Warm-up complete.")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "video" and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
            pbar = tqdm(total=total_frames if total_frames > 0 and not args.show else None, desc="Processing", unit="frame", disable=args.show)
            frame_count = 0
            while True:
                current_frame = frame
                input_image_batch, input_size_batch = preprocess_image(current_frame, INPUT_HEIGHT, INPUT_WIDTH)
                start_time = time.perf_counter()
                trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
                end_time = time.perf_counter(); inference_times.append(end_time - start_time)
                annotated_frame = current_frame.copy() # Default to original if inference fails
                if trt_outputs is not None:
                    try:
                        labels_out = trt_outputs.get("labels"); boxes_out = trt_outputs.get("boxes"); scores_out = trt_outputs.get("scores")
                        if labels_out is not None and boxes_out is not None and scores_out is not None:
                            if labels_out.ndim > 0 and labels_out.shape[0] == 1: labels_out = labels_out[0]
                            if boxes_out.ndim > 0 and boxes_out.shape[0] == 1: boxes_out = boxes_out[0]
                            if scores_out.ndim > 0 and scores_out.shape[0] == 1: scores_out = scores_out[0]
                            original_wh = input_size_batch[0]
                            labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, original_wh, args.threshold)
                            annotated_frame = draw_detections_cv2(annotated_frame, labels, boxes, scores, label_map, cv_color_map)
                        # else: print(f"Warning: Inconsistent outputs frame {frame_count}.") # Optional warning
                    except Exception as e: print(f"Error postprocessing/drawing frame {frame_count}: {e}")
                # else: print(f"Warning: Inference failed frame {frame_count}.") # Optional warning

                if args.show:
                    cv2.imshow(window_name, annotated_frame);
                    if cv2.waitKey(1) & 0xFF == ord('q'): print("Exiting..."); break
                else:
                    if out_writer: out_writer.write(annotated_frame); pbar.update(1)
                frame_count += 1; ret, frame = cap.read()
                if not ret: break
            if not args.show: pbar.close()
        except Exception as e: print(f"Error during video/camera processing: {e}")
        finally:
            if cap and cap.isOpened(): cap.release()
            if out_writer and out_writer.isOpened(): out_writer.release()
            if not args.show: print(f"\nVideo processing finished.")

    # --- Cleanup and Stats ---
    cv2.destroyAllWindows()
    try:
        # Free PyCUDA buffers first
        if inputs:
            for inp in inputs: inp['device'].free()
        if outputs:
            for out in outputs: out['device'].free()
        # Then delete TRT objects (context before engine)
        if 'context' in locals() and context: del context
        if 'engine' in locals() and engine: del engine
        print("TensorRT/PyCUDA resources released.")
    except Exception as e: print(f"Error during resource cleanup: {e}")

    # --- Display Benchmarking Stats ---
    if inference_times:
        valid_inference_times = inference_times # Use all recorded times
        avg_latency_ms=np.mean(valid_inference_times)*1000; std_latency_ms=np.std(valid_inference_times)*1000
        min_latency_ms=np.min(valid_inference_times)*1000; max_latency_ms=np.max(valid_inference_times)*1000
        avg_fps=1.0/np.mean(valid_inference_times)
        print("\n--- Inference Time Stats (TensorRT context.execute_async_v3) ---")
        print(f"Frames processed: {len(valid_inference_times)}")
        print(f"Avg Latency: {avg_latency_ms:.2f} ms"); print(f"Min Latency: {min_latency_ms:.2f} ms")
        print(f"Max Latency: {max_latency_ms:.2f} ms"); print(f"Std Dev:     {std_latency_ms:.2f} ms")
        print(f"Avg FPS:     {avg_fps:.2f}")
    else: print("\nNo inference runs recorded for benchmarking.")

    print("Script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TensorRT inference script for RT-DETRv2 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--engine_file', type=str, required=True, help="Path to the TensorRT engine file (.engine).")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to input image/video file or camera index/URL.")
    parser.add_argument('--save_dir', type=str, default='inference_results_trt',
                        help="Directory to save output files (if --show is not used).")
    parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device hint (TensorRT primarily uses GPU if engine is built for it).")
    parser.add_argument('--show', action='store_true',
                        help="Display inference results in real-time instead of saving to file.")

    # Dependency check
    try:
        import tensorrt; import pycuda.driver; import pycuda.autoinit
    except ImportError as e: print(f"Error: Missing dependency - {e}"); sys.exit(1)

    args = parser.parse_args()
    main(args)
