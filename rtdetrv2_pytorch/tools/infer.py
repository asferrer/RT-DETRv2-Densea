#!/usr/bin/env python
"""
Script de inferencia para RT-DETR-V2 que detecta automáticamente el modo de entrada 
(en imagen, video o cámara/stream) en función del input recibido.
No se muestra la salida en tiempo real, sino que se guarda automáticamente la inferencia 
(con las detecciones pintadas) usando un nombre que combina el nombre del input y el nombre
del archivo de configuración. Si en algún frame no se detecta nada, se guarda el frame original.
"""

import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import colorsys
from tqdm import tqdm

# Se añade el directorio padre para poder importar YAMLConfig y otros módulos internos.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

########################
# Funciones auxiliares de visualización
########################

def generate_color_map_from_label_map(label_map: dict) -> dict:
    keys = sorted(label_map.keys())
    n = len(keys)
    color_map = {}
    for i, key in enumerate(keys):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        color_hex = '#{0:02x}{1:02x}{2:02x}'.format(int(r*255), int(g*255), int(b*255))
        color_map[key] = color_hex
    return color_map

def get_label_map(dataset) -> dict:
    label_map = {}
    try:
        categories = dataset.coco.dataset.get('categories', [])
        for cat in categories:
            label_map[cat['id']] = cat['name']
    except Exception as e:
        print(f"No se pudo obtener el mapeo de etiquetas: {e}")
    return label_map

def draw_inferences_pil(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                         label_map=None, color_map=None, threshold: float = 0.5, radius: int = 2):
    """
    Dibuja las inferencias sobre una imagen PIL de forma minimalista. 
    Si no se detecta nada (todos los scores ≤ threshold), se devuelve la imagen original.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    valid_indices = scores > threshold
    # Si no hay detecciones válidas, se devuelve la imagen original
    if valid_indices.sum() == 0:
        return image

    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    padding = 2
    for j, box in enumerate(valid_boxes):
        x0, y0, x1, y1 = map(int, box)
        pred_id = valid_labels[j].item() + 1  # Se suma 1 si el modelo devuelve índices 0-indexados
        color = color_map.get(pred_id, "black") if color_map is not None else "black"
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=color, width=1)
        if label_map is not None:
            label_text = label_map.get(pred_id, str(pred_id))
        else:
            label_text = str(pred_id)
        text = f"{label_text} {valid_scores[j].item():.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if y0 - text_height - padding >= 0:
            text_position = (x0 + padding, y0 - text_height - padding)
        else:
            text_position = (x0 + padding, y1 + padding)
        draw.text(text_position, text, font=font, fill=color, stroke_width=1, stroke_fill="black")
    return image

########################
# Funciones de conversión entre PIL y OpenCV
########################

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

########################
# Clase ModelWrapper
########################

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs  # Se espera (labels, boxes, scores)

########################
# Inferencia en un frame
########################

def inference_on_frame(frame: np.ndarray, model, transform, device, label_map, color_map, threshold=0.5):
    """
    Realiza inferencia sobre un frame. Se convierte el frame a PIL, se aplica el transform para inferencia,
    y se dibujan las detecciones sobre la imagen PIL (usando orig_size del frame original).
    Se devuelve el frame anotado en el mismo tamaño original.
    """
    pil_img = cv2_to_pil(frame)
    # El transform puede incluir un resize para cumplir con el input del modelo.
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    # Se utiliza el tamaño original para reescalar las cajas.
    orig_size = torch.tensor([pil_img.width, pil_img.height])[None].to(device)
    with torch.no_grad():
        labels, boxes, scores = model(input_tensor, orig_size)
    # Dibujar sobre la imagen original
    annotated_pil = draw_inferences_pil(pil_img, labels[0], boxes[0], scores[0],
                                         label_map=label_map, color_map=color_map, threshold=threshold)
    # Asegurarse de que la imagen anotada tenga el mismo tamaño original
    annotated_pil = annotated_pil.resize((pil_img.width, pil_img.height))
    annotated_frame = pil_to_cv2(annotated_pil)
    return annotated_frame

########################
# Detección automática de modo
########################

def auto_detect_mode(input_source: str) -> str:
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_exts = ['.mp4', '.avi', '.mkv', '.mov']
    if os.path.exists(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext in image_exts:
            return "image"
        elif ext in video_exts:
            return "video"
        else:
            return "image"
    else:
        try:
            int(input_source)
            return "camera"
        except ValueError:
            if input_source.startswith("rtsp://") or input_source.startswith("http://") or input_source.startswith("https://"):
                return "camera"
            else:
                return "camera"

########################
# Función principal
########################

def main():
    parser = argparse.ArgumentParser(
        description="Script de inferencia para imágenes, video o cámara (RTSP/webcam) que guarda automáticamente el resultado."
    )
    parser.add_argument('-c', '--config', type=str, required=True, help="Ruta al archivo YAML de configuración.")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Ruta al checkpoint del modelo.")
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Dispositivo: cpu o cuda.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Ruta al archivo de imagen/video o índice/URL para cámara (RTSP/webcam).")
    parser.add_argument('--save_dir', type=str, default='inference_results',
                        help="Directorio base para guardar los resultados.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Umbral para filtrar detecciones.")
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Detectar automáticamente el modo
    mode = auto_detect_mode(args.input)
    print(f"Modo detectado: {mode}")
    
    # Cargar configuración y modelo
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location=device)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    # Cargar el modelo de configuración (por ejemplo, cfg.model) y también instanciar ModelWrapper
    model_instance = cfg.model
    model_instance.load_state_dict(state)
    model = ModelWrapper(cfg).to(device)
    model.eval()
    
    # Obtener label map y color map
    label_map = get_label_map(cfg.val_dataloader.dataset)
    color_map = generate_color_map_from_label_map(label_map)
    
    # Definir transformación de entrada (para cumplir con el input del modelo)
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # Crear directorio para guardar resultados
    os.makedirs(args.save_dir, exist_ok=True)
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    if mode == "image":
        pil_img = Image.open(args.input).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        orig_size = torch.tensor([pil_img.width, pil_img.height])[None].to(device)
        with torch.no_grad():
            labels, boxes, scores = model(input_tensor, orig_size)
        annotated_img = draw_inferences_pil(pil_img, labels[0], boxes[0], scores[0],
                                             label_map=label_map, color_map=color_map, threshold=args.threshold)
        out_filename = f"{input_basename}_{config_name}.jpg"
        out_path = os.path.join(args.save_dir, out_filename)
        annotated_img.save(out_path)
        print(f"Imagen inferida guardada en: {out_path}")
    elif mode in ["video", "camera"]:
        if mode == "video":
            cap = cv2.VideoCapture(args.input)
        else:
            try:
                cam_index = int(args.input)
                cap = cv2.VideoCapture(cam_index)
            except ValueError:
                cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print("Error: no se pudo abrir la fuente de video/cámara.")
            return
        # Obtener tamaño original del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25
        if mode == "video":
            out_filename = f"{input_basename}_{config_name}.mp4"
        else:
            out_filename = f"camera_{config_name}.mp4"
        out_path = os.path.join(args.save_dir, out_filename)
        # Configurar VideoWriter con el tamaño original
        out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "video" else None
        pbar = tqdm(total=total_frames, desc="Procesando frames", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Realizar inferencia y obtener el frame anotado
            annotated_frame = inference_on_frame(frame, model, transform, device, label_map, color_map, threshold=args.threshold)
            # Forzar el tamaño a (width, height) para coincidir con VideoWriter
            annotated_frame = cv2.resize(annotated_frame, (width, height))
            out_writer.write(annotated_frame)
            pbar.update(1)
        pbar.close()
        cap.release()
        out_writer.release()
        print(f"Video inferido guardado en: {out_path}")

if __name__ == '__main__':
    main()
