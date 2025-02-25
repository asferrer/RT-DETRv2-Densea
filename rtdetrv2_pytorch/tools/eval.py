
#!/usr/bin/env python
"""
Script de evaluación para RT-DETR-V2 que procesa el dataloader de validación una única vez,
guardando las imágenes inferidas (si se activa --draw) y acumulando las métricas adicionales
(para calcular la matriz de confusión y el reporte de clasificación) a partir de la misma inferencia.
Se aplican optimizaciones de eficiencia: uso de AMP, half precision y liberación periódica de caché.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm  # Barra de progreso
import colorsys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar directorio padre para importar YAMLConfig y otros módulos internos
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

def draw_inferences(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                     label_map=None, color_map=None, threshold: float = 0.5, 
                     save_path: str = None, radius: int = 2):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    valid_indices = scores > threshold
    if valid_indices.sum() == 0:
        # Si no se detecta nada, devuelve la imagen original
        return image
    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    padding = 2  
    for j, box in enumerate(valid_boxes):
        x0, y0, x1, y1 = map(int, box)
        pred_id = valid_labels[j].item() + 1  # Ajuste para 0-indexados
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
    
    if save_path is not None:
        image.save(save_path)
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
        return outputs  # (labels, boxes, scores)

########################
# Función para obtener subcarpeta según imagen
########################

def get_subfolder_for_image(file_path: str, image_dirs: list) -> str:
    file_path = os.path.abspath(file_path)
    for img_dir in image_dirs:
        img_dir = os.path.abspath(img_dir)
        if file_path.startswith(img_dir):
            return os.path.basename(img_dir)
        elif os.path.basename(img_dir) in file_path:
            return os.path.basename(img_dir)
    return "unknown"

########################
# Funciones para métricas adicionales (integradas en una única pasada)
########################

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def match_detections_in_image(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    preds = sorted(zip(pred_boxes, pred_labels, pred_scores), key=lambda x: x[2], reverse=True)
    matched_pairs = []
    unmatched_pred = []
    assigned_gt = set()
    for p_box, p_label, p_score in preds:
        best_iou = 0
        best_gt_idx = None
        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in assigned_gt:
                continue
            current_iou = iou(p_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx
        if best_iou >= iou_thresh and best_gt_idx is not None:
            assigned_gt.add(best_gt_idx)
            matched_pairs.append((gt_labels[best_gt_idx], p_label))
        else:
            unmatched_pred.append(p_label)
    unmatched_gt = [gt_labels[i] for i in range(len(gt_labels)) if i not in assigned_gt]
    return matched_pairs, unmatched_gt, unmatched_pred

########################
# Función de evaluación integrada (una sola pasada)
########################

def evaluate_on_dataset(args, cfg, device):
    model = ModelWrapper(cfg).to(device)
    model.eval()
    dataloader = cfg.val_dataloader
    evaluator = None
    if hasattr(cfg, 'evaluator'):
        evaluator = cfg.evaluator
    image_dirs = cfg.val_dataloader.dataset.image_dirs if isinstance(cfg.val_dataloader.dataset.image_dirs, list) else [cfg.val_dataloader.dataset.image_dirs]
    label_map = get_label_map(cfg.val_dataloader.dataset)
    color_map = generate_color_map_from_label_map(label_map)
    os.makedirs(args.base_eval_dir, exist_ok=True)
    
    # Listas para métricas adicionales (acumulación en la misma pasada)
    all_y_true = []
    all_y_pred = []
    batch_count = 0
    
    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluando batches", unit="batch")):
        if isinstance(batch, dict):
            images = batch['img']
            targets = batch.get('targets', None)
            if 'orig_size' in batch:
                orig_sizes = batch['orig_size']
            else:
                B = images.shape[0]
                orig_sizes = torch.tensor([[640, 640]] * B).to(device)
            file_names = batch.get('file_name', None)
        elif isinstance(batch, (list, tuple)):
            images, targets = batch
            B = images.shape[0]
            orig_sizes = torch.tensor([[640, 640]] * B).to(device)
            file_names = None
        else:
            raise ValueError("Formato de batch no reconocido.")
        images = images.to(device)
        with torch.no_grad():
            if orig_sizes.dim() == 1:
                orig_sizes = orig_sizes.unsqueeze(0)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images, orig_sizes)
            else:
                outputs = model(images, orig_sizes)
        labels_batch, boxes_batch, scores_batch = outputs
        
        # Acumular métricas adicionales para cada imagen del batch
        for i in range(B):
            gt_boxes = []
            gt_labels = []
            if targets is not None and isinstance(targets[i], dict) and "annotations" in targets[i]:
                for ann in targets[i]["annotations"]:
                    if "bbox" in ann and "category_id" in ann:
                        gt_boxes.append(ann["bbox"])
                        gt_labels.append(ann["category_id"])
            pred_boxes = boxes_batch[i].cpu().tolist()
            pred_labels = [int(x.item())+1 for x in labels_batch[i].cpu()]
            pred_scores = scores_batch[i].cpu().tolist()
            # Si no hay ground truth ni predicciones, se omite esta imagen
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            matched, unmatched_gt, unmatched_pred = match_detections_in_image(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5
            )
            for gt_lab, pred_lab in matched:
                all_y_true.append(gt_lab)
                all_y_pred.append(pred_lab)
            for gt_lab in unmatched_gt:
                all_y_true.append(gt_lab)
                all_y_pred.append(0)
            for pred_lab in unmatched_pred:
                all_y_true.append(0)
                all_y_pred.append(pred_lab)
        
        # Si se desea guardar imágenes con inferencias
        if args.draw:
            if file_names is not None:
                for i, fname in enumerate(file_names):
                    try:
                        im = Image.open(fname).convert('RGB')
                    except Exception as e:
                        print(f"Error al cargar la imagen {fname}: {e}")
                        continue
                    out_fname = os.path.join(args.base_eval_dir, "predicted_images",f"result_{os.path.basename(fname)}")
                    draw_inferences(im, labels_batch[i], boxes_batch[i], scores_batch[i],
                                    label_map=label_map, color_map=color_map, threshold=0.5, save_path=out_fname)
            else:
                for i in range(images.shape[0]):
                    fname = targets[i].get("file_name", None) if isinstance(targets[i], dict) else None
                    subfolder = get_subfolder_for_image(fname, image_dirs) if fname is not None else "unknown"
                    out_fname = os.path.join(args.base_eval_dir, "predicted_images", f"result_batch_{idx}_{i}.jpg")
                    im = T.ToPILImage()(images[i].cpu())
                    draw_inferences(im, labels_batch[i], boxes_batch[i], scores_batch[i],
                                    label_map=label_map, color_map=color_map, threshold=0.5, save_path=out_fname)
        batch_count += 1
        if device.type == "cuda" and batch_count % 20 == 0:
            torch.cuda.empty_cache()
    
    # Calcular y guardar métricas adicionales en una sola pasada
    all_labels = [0] + sorted(label_map.keys())
    cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)
    report = classification_report(
        all_y_true, all_y_pred, labels=all_labels,
        target_names=["background"] + [label_map[k] for k in sorted(label_map.keys())]
    )
    
    # Guardar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["background"] + [label_map[k] for k in sorted(label_map.keys())],
                yticklabels=["background"] + [label_map[k] for k in sorted(label_map.keys())])
    plt.xlabel("Predicción")
    plt.ylabel("Ground Truth")
    plt.title("Matriz de Confusión")
    cm_path = os.path.join(args.base_eval_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"\nMatriz de Confusión guardada en: {cm_path}")
    
    # Guardar reporte de clasificación
    plt.figure(figsize=(8, 8))
    plt.text(0.01, 0.05, report, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.title("Reporte de Clasificación")
    report_path = os.path.join(args.base_eval_dir, "classification_report.png")
    plt.tight_layout()
    plt.savefig(report_path)
    plt.close()
    print(f"\nReporte de Clasificación guardado en: {report_path}")

########################
# Funciones de evaluación "clásica"
########################

def evaluate_on_image(args, cfg, device):
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_tensor = transform(im_pil)[None].to(device)
    model = ModelWrapper(cfg).to(device)
    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(im_tensor, orig_size)
        else:
            outputs = model(im_tensor, orig_size)
        labels, boxes, scores = outputs
    label_map = get_label_map(cfg.val_dataloader.dataset)
    color_map = generate_color_map_from_label_map(label_map)
    if args.draw:
        im_copy = im_pil.copy()
        out_path = os.path.join(args.base_eval_dir, "result_single.jpg")
        draw_inferences(im_copy, labels[0], boxes[0], scores[0],
                        label_map=label_map, color_map=color_map, threshold=0.5, save_path=out_path)
    else:
        for j in range(len(labels[0])):
            if scores[0][j] > 0.5:
                pred_id = labels[0][j].item() + 1
                label_text = label_map.get(pred_id, str(pred_id))
                print(f"Label: {label_text}, Score: {scores[0][j].item():.2f}")

########################
# Función principal
########################

def main():
    parser = argparse.ArgumentParser(
        description="Script de evaluación del modelo con opción de visualización de inferencias y métricas adicionales en una única pasada."
    )
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Ruta al archivo de configuración YAML.")
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help="Ruta al checkpoint del modelo.")
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help="Dispositivo de cómputo (e.g., cpu o cuda).")
    parser.add_argument('--im_file', type=str, default=None,
                        help="Ruta a una imagen individual para inferencia (modo imagen única).")
    parser.add_argument('--draw', action='store_true',
                        help="Si se especifica, dibuja y guarda las imágenes con inferencias.")
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help="Directorio base para guardar las imágenes y métricas.")
    args = parser.parse_args()
    device = torch.device(args.device)
    
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.base_eval_dir = os.path.join(args.output_dir, config_name)
    os.makedirs(args.base_eval_dir, exist_ok=True)
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location=device)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    model_instance = cfg.model
    model_instance.load_state_dict(state)
    
    # Si se activa --draw, se informará dónde se guardarán las imágenes
    if args.draw:
        print(f"Las imágenes inferidas se guardarán en: {args.base_eval_dir}")
        predicted_images_dir = os.path.join(args.base_eval_dir, "predicted_images")
        os.makedirs(predicted_images_dir, exist_ok=True)
    
    if args.im_file is not None:
        print("Evaluación sobre imagen individual...")
        evaluate_on_image(args, cfg, device)
    else:
        print("Evaluación sobre dataset de validación...")
        evaluate_on_dataset(args, cfg, device)
        
if __name__ == '__main__':
    main()
