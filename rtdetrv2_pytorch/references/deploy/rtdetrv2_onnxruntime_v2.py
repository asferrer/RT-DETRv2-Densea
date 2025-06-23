#!/usr/bin/env python
"""
ONNX inference script for RT-DETRv2 models.

Features:
- Handles image, video, and camera/stream inputs automatically.
- Minimalist bounding box drawing using OpenCV.
- Optional real-time display (`--show`) or saving results to file (default).
- Basic benchmarking (warm-up, latency, FPS).
"""

import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import time
from tqdm import tqdm
import colorsys

# --- Helper Functions ---

def generate_color_map(num_classes: int) -> list:
    """Generates a list of distinct BGR colors for drawing."""
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

COCO_CLASSES = {
    0: "Bottle",
    1: "Can",
    2: "Fishing_Net",
    3: "Glove",
    4: "Mask",
    5: "Metal_Debris",
    6: "Plastic_Debris",
    7: "Tire",
}
NUM_CLASSES = 8 # Ensure this matches your dictionary

label_map = COCO_CLASSES
cv_color_map = generate_color_map(NUM_CLASSES)

def draw_detections_cv2(image, labels, boxes, scores, label_map=None, color_map=None):
    """
    Draws bounding boxes and labels on the image using OpenCV
    with a more minimalist style.
    """
    if label_map is None: label_map = {}
    if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0:
        return image

    img_h, img_w = image.shape[:2] # Get image dimensions for adjustments

    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # Ensure they are integers
        class_id = int(label)

        # Get color from the map (list of BGR tuples)
        if color_map and 0 <= class_id < len(color_map):
            color = color_map[class_id]
        else:
            color = (0, 255, 0) # Default green

        # --- Draw thinner Bounding Box ---
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1) # Thickness 1

        # --- Prepare and Draw Minimalist Label ---
        class_name = label_map.get(class_id, str(class_id))
        text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 # Slightly smaller font scale
        thickness = 1

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate label position (top-left by default)
        tag_x = x1
        tag_y = y1 - text_h - baseline - 3 # Y position of the label (with padding)

        # Adjust if the label goes off the top
        if tag_y < 0:
            tag_y = y1 + baseline + 1 # Put it inside, below the top edge of the box

        # Calculate final text position within the label
        text_x = tag_x + 1
        text_y = tag_y + text_h

        # Draw the small rectangle (label background)
        cv2.rectangle(image, (tag_x, tag_y), (tag_x + text_w + 2, tag_y + text_h + baseline), color, -1) # Filled

        # Draw the text on the label
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,0,0), thickness) # Black text

    return image

def auto_detect_mode(input_source: str) -> str:
    """Detects if the input is an image, video, or camera/stream."""
    if not isinstance(input_source, str):
         try:
             int(input_source)
             return "camera"
         except (ValueError, TypeError):
              raise TypeError(f"Input must be a path (str) or camera index (int), received {type(input_source)}")

    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_exts = ['.mp4', '.avi', '.mkv', '.mov']

    if os.path.exists(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext in image_exts:
            return "image"
        elif ext in video_exts:
            return "video"
        else:
            # Try opening as video even with unknown extension
            cap_test = cv2.VideoCapture(input_source)
            if cap_test.isOpened():
                cap_test.release()
                return "video"
            else:
                 print(f"Warning: Unknown extension '{ext}', treating as image.")
                 return "image"
    else:
        try:
            # Try interpreting as a numeric camera index
            int(input_source)
            return "camera"
        except ValueError:
            # If not a numeric index, check if it's a URL
            if input_source.startswith("rtsp://") or \
               input_source.startswith("http://") or \
               input_source.startswith("https://"):
                return "camera"
            else:
                # If not an existing path, index, or URL, it's invalid
                raise ValueError(f"Unrecognized or non-existent input source: {input_source}")


# --- Pre/Post-processing Functions ---
def preprocess_image(image_bgr, target_height, target_width):
    """Preprocesses a BGR image (OpenCV) for RT-DETR."""
    original_h, original_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if original_h > target_height else cv2.INTER_LINEAR
    img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=interpolation)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    batch_images = np.expand_dims(img_chw, axis=0)
    # Use [W, H] format for orig_target_sizes
    batch_sizes = np.array([[original_w, original_h]], dtype=np.int64)
    return np.ascontiguousarray(batch_images), np.ascontiguousarray(batch_sizes)

# postprocess only filters, assumes boxes are already absolute xyxy
def postprocess(labels, boxes, scores, original_hw, confidence_thr, input_shape=(640, 640)):
    """
    Postprocesses the RT-DETR outputs for one image (multiple detections).
    CURRENT ASSUMPTION: 'boxes' are already ABSOLUTE xyxy and correctly scaled.
    Only applies confidence filtering.
    """
    # Filter by confidence
    keep = scores > confidence_thr
    num_after_filter = np.sum(keep)

    if num_after_filter == 0:
        return np.array([]), np.array([]), np.array([])

    filtered_labels = labels[keep]
    filtered_scores = scores[keep]
    filtered_boxes_abs = boxes[keep]

    # Return filtered results
    return filtered_labels.astype(int), filtered_boxes_abs.astype(int), filtered_scores


# --- Main Function ---
def main(args):
    # --- Validate and Configure ---
    if not os.path.exists(args.onnx_model):
        raise FileNotFoundError(f"ONNX file not found at: {args.onnx_model}")

    mode = auto_detect_mode(args.input)
    print(f"Detected mode: {mode}")
    # Only create directory if we are saving
    if not args.show:
        os.makedirs(args.save_dir, exist_ok=True)

    # --- Configure ONNX Runtime ---
    print("Configuring ONNX Runtime session...")
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if args.device == 'cpu':
         print("Forcing use of CPUExecutionProvider.")
         providers = ['CPUExecutionProvider']
    elif 'CUDAExecutionProvider' not in available_providers:
        print("WARNING: CUDAExecutionProvider not found. Using CPU.")
        providers = ['CPUExecutionProvider']

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(args.onnx_model, sess_options=sess_options, providers=providers)
        session_providers = session.get_providers()
        print(f"Session created using: {session_providers}")
        if 'CUDAExecutionProvider' in session_providers:
             print(f"CUDA provider active on device: {ort.get_device()}")
        input_details = session.get_inputs()
        output_details = session.get_outputs()
        input_names = [inp.name for inp in input_details]
        output_names = [out.name for out in output_details]
        print(f"Expected input names: {input_names}")
        print(f"Expected output names: {output_names}")
    except Exception as e:
        print(f"Error creating ONNX Runtime session: {e}")
        exit()

    # --- Processing based on mode ---
    input_height = 640
    input_width = 640
    input_shape = (input_height, input_width)

    try:
        if isinstance(args.input, str):
             input_basename = os.path.splitext(os.path.basename(args.input))[0]
        else:
             input_basename = f"camera_{args.input}"
    except Exception as e:
         print(f"Warning: Could not generate input base name ({e}), using 'output'.")
         input_basename = "output"

    onnx_model_name = os.path.splitext(os.path.basename(args.onnx_model))[0]
    output_base_name = f"{input_basename}_{onnx_model_name}"

    warmup_runs = 3 # Reduce slightly for visualization
    inference_times = []
    window_name = "ONNX Inference" # Window name

    if mode == "image":
        frame = cv2.imread(args.input)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")
        print(f"Image loaded: {args.input}")

        input_image_batch, input_size_batch = preprocess_image(frame, input_height, input_width)
        ort_inputs = {input_names[0]: input_image_batch, input_names[1]: input_size_batch}

        print(f"Performing {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs):
            _ = session.run(output_names, ort_inputs)
        print("Warm-up completed.")

        print("Performing inference...")
        start_time = time.perf_counter()
        ort_outputs = session.run(output_names, ort_inputs)
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)

        try:
            # Extract outputs by name for robustness
            labels_output = ort_outputs[output_names.index("labels")]
            boxes_output = ort_outputs[output_names.index("boxes")]
            scores_output = ort_outputs[output_names.index("scores")]
            # Remove batch dimension if present (assuming batch size 1)
            if labels_output.shape[0] == 1: labels_output = labels_output[0]
            if boxes_output.shape[0] == 1: boxes_output = boxes_output[0]
            if scores_output.shape[0] == 1: scores_output = scores_output[0]
        except (ValueError, IndexError, Exception) as e:
             print(f"Error extracting/processing model outputs: {e}")
             print("Check if output names ('labels', 'boxes', 'scores') match your ONNX model.")
             exit()

        original_wh_for_postprocess = input_size_batch[0] # Should be [W, H]
        labels, boxes, scores = postprocess(
            labels_output, boxes_output, scores_output,
            original_wh_for_postprocess, args.threshold, input_shape
        )
        print(f"Postprocessing completed. Final detections: {len(labels)}")

        annotated_frame = draw_detections_cv2(frame.copy(), labels, boxes, scores, label_map, cv_color_map)

        # --- Show or Save ---
        if args.show:
            print("Displaying result. Press 'q' to close.")
            cv2.imshow(window_name, annotated_frame)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            out_filename = f"{output_base_name}.jpg"
            out_path = os.path.join(args.save_dir, out_filename)
            try:
                cv2.imwrite(out_path, annotated_frame)
                print(f"Inferred image saved to: {out_path}")
            except Exception as e:
                print(f"Error saving image to {out_path}: {e}")


    elif mode in ["video", "camera"]:
        cap = None
        out_writer = None # Initialize to None
        if mode == "video":
            cap = cv2.VideoCapture(args.input)
            source_description = f"Video: {args.input}"
        else: # camera mode
            try:
                cam_input = int(args.input)
                cap = cv2.VideoCapture(cam_input)
                source_description = f"Camera index: {cam_input}"
            except ValueError:
                # Treat as URL or other string identifier
                cap = cv2.VideoCapture(args.input)
                source_description = f"Camera/Stream: {args.input}"

        if not cap or not cap.isOpened():
            raise IOError(f"Error: could not open source: {args.input}")
        print(f"Source opened: {source_description}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25 # Use default FPS if unavailable
        print(f"Video/stream properties: {width}x{height} @ {fps:.2f} FPS")

        # Configure VideoWriter ONLY if not using --show
        if not args.show:
            out_filename = f"{output_base_name}.mp4"
            out_path = os.path.join(args.save_dir, out_filename)
            try:
                out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                if not out_writer.isOpened():
                     raise IOError(f"Could not open VideoWriter for {out_path}")
                print(f"Saving output video to: {out_path}")
            except Exception as e:
                print(f"Error initializing VideoWriter: {e}")
                if cap and cap.isOpened(): cap.release()
                exit()
        else:
            # Create window if we are showing
             cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


        # --- Warm-up before loop ---
        ret, frame = cap.read()
        if not ret:
             raise IOError("Could not read the first frame from the source.")
        print(f"Performing {warmup_runs} warm-up runs...")
        input_image_batch, input_size_batch = preprocess_image(frame, input_height, input_width)
        ort_inputs = {input_names[0]: input_image_batch, input_names[1]: input_size_batch}
        for _ in range(warmup_runs):
            _ = session.run(output_names, ort_inputs)
        print("Warm-up completed.")

        # --- Processing Loop ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "video" and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        pbar = tqdm(total=total_frames if total_frames > 0 and not args.show else None, # Disable pbar if showing real-time
                    desc="Processing frames", unit="frame", disable=args.show)

        frame_count = 0
        low_detection_warning_printed = False
        while True:
            # Use the frame read at the end of the previous iteration (or the first frame)
            current_frame = frame

            # Preprocess
            input_image_batch, input_size_batch = preprocess_image(current_frame, input_height, input_width)
            ort_inputs = {input_names[0]: input_image_batch, input_names[1]: input_size_batch}

            # Inference
            start_time = time.perf_counter()
            ort_outputs = session.run(output_names, ort_inputs)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            # Postprocess
            try:
                labels_output = ort_outputs[output_names.index("labels")]
                boxes_output = ort_outputs[output_names.index("boxes")]
                scores_output = ort_outputs[output_names.index("scores")]
                if labels_output.shape[0] == 1: labels_output = labels_output[0]
                if boxes_output.shape[0] == 1: boxes_output = boxes_output[0]
                if scores_output.shape[0] == 1: scores_output = scores_output[0]
            except (ValueError, IndexError, Exception) as e:
                print(f"\nError extracting/processing outputs on frame {frame_count}: {e}")
                # Try to read next frame and continue
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                if not args.show: pbar.update(1)
                continue # Skip drawing/saving for this frame

            original_wh_for_postprocess = input_size_batch[0] # Should be [W, H]
            labels, boxes, scores = postprocess(
                labels_output, boxes_output, scores_output,
                original_wh_for_postprocess, args.threshold, input_shape
            )

            # Print warning if few detections (potential threshold issue)
            if len(labels) < 2 and not low_detection_warning_printed:
                 # Check if the model actually output more detections before filtering
                 if len(scores_output) > len(labels):
                     print(f"\n[Warning] Few detections ({len(labels)}) with threshold {args.threshold}. Consider using '--threshold' with a lower value (e.g., 0.2).")
                     low_detection_warning_printed = True # Print only once

            # Draw detections
            annotated_frame = draw_detections_cv2(current_frame.copy(), labels, boxes, scores, label_map, cv_color_map)

            # --- Show or Save ---
            if args.show:
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
            else:
                if out_writer: out_writer.write(annotated_frame)
                pbar.update(1)

            frame_count += 1
            # Read the next frame for the *next* iteration
            ret, frame = cap.read()
            if not ret:
                 break # End of video or stream

        # --- Cleanup after loop ---
        if not args.show: pbar.close()
        if cap and cap.isOpened(): cap.release()
        if out_writer and out_writer.isOpened(): out_writer.release()
        if not args.show: print(f"\nVideo processed and saved.")


    # --- Cleanup and Statistics ---
    cv2.destroyAllWindows() # Close window if it was opened

    # --- Display Time Statistics ---
    valid_inference_times = inference_times # Already contains only successful run times
    if valid_inference_times:
        avg_latency_ms = np.mean(valid_inference_times) * 1000
        std_latency_ms = np.std(valid_inference_times) * 1000
        min_latency_ms = np.min(valid_inference_times) * 1000
        max_latency_ms = np.max(valid_inference_times) * 1000
        avg_fps = 1.0 / np.mean(valid_inference_times)

        print("\n--- Inference Time Statistics (session.run) ---")
        print(f"Number of frames processed: {len(valid_inference_times)}")
        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Minimum Latency: {min_latency_ms:.2f} ms")
        print(f"Maximum Latency: {max_latency_ms:.2f} ms")
        print(f"Standard Deviation: {std_latency_ms:.2f} ms")
        print(f"Average FPS:      {avg_fps:.2f}")
    else:
        print("\nNot enough inferences were performed to calculate statistics.")


    print("Process completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ONNX inference script for RT-DETRv2 (image/video/camera) with saving or display."
    )
    parser.add_argument('--onnx_model', type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to image/video or camera index/URL.")
    parser.add_argument('--save_dir', type=str, default='inference_results_onnx',
                        help="Directory to save results (ignored if --show is used).")
    parser.add_argument('--threshold', type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device to use ('cuda' or 'cpu'). 'cuda' will try GPU if available.")
    parser.add_argument('--show', action='store_true',
                        help="Show inference output in real-time instead of saving to file.")

    args = parser.parse_args()
    main(args)
