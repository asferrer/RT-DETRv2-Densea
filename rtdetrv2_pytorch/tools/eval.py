#!/usr/bin/env python
"""
Script de evaluación para RT-DETR-V2 que procesa el dataloader de validación en una única pasada,
guardando las imágenes inferidas y acumulando las métricas adicionales (matriz de confusión, reporte
de clasificación y reporte de falsos positivos y falsos negativos). Además, si se activa el flag --debug 
y las detecciones difieren de la ground truth, se genera una imagen comparativa (izquierda: predicción; 
derecha: ground truth).

Versión actualizada:
- Añadida la capacidad de generar un archivo de predicciones en formato COCO (`coco_predictions.json`).
- Añadida la evaluación de métricas estándar COCO (mAP) usando `pycocotools` si está instalado.
- Se activa con el flag --coco_eval.
"""

import os
import sys
import cv2
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import colorsys
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Bloque para la importación y evaluación COCO
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_TOOLS_AVAILABLE = True
except ImportError:
    COCO_TOOLS_AVAILABLE = False


# Agregar directorio padre para importar YAMLConfig y otros módulos internos
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

########################
# Funciones auxiliares de visualización y ajuste
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
            # Aseguramos que la clave sea un entero
            label_map[int(cat['id'])] = cat['name']
    except Exception as e:
        print(f"No se pudo obtener el mapeo de etiquetas: {e}")
    return label_map

def compute_label_offset(pred_labels_tensor: torch.Tensor, label_map: dict) -> int:
    """
    Determina dinámicamente el offset a aplicar a las etiquetas predichas para que sean coherentes
    con el mapeo de etiquetas. Si el modelo ya produce etiquetas iniciando en 0, se devuelve 0.
    """
    if pred_labels_tensor.numel() > 0:
        # Utilizamos el valor mínimo de las predicciones para comparar
        raw_pred_min = int(pred_labels_tensor.min().item())
        min_label = min(label_map.keys()) if label_map else 0
        offset = min_label - raw_pred_min
        return offset
    return 0

def draw_inferences(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                     label_map=None, color_map=None, threshold: float = 0.5, 
                     label_offset: int = 0, save_path: str = None, radius: int = 2):
    """Dibuja las detecciones del modelo sobre la imagen."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    valid_indices = scores > threshold
    if valid_indices.sum() == 0:
        return image
    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    padding = 2  
    for j, box in enumerate(valid_boxes):
        x0, y0, x1, y1 = map(int, box)
        # Se aplica el offset dinámico
        pred_id = valid_labels[j].item() + label_offset
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

def draw_ground_truth(image: Image.Image, annotations: list, label_map=None, color_map=None, radius: int = 2):
    """
    Dibuja las anotaciones ground truth sobre la imagen.
    Se espera que 'annotations' sea una lista de diccionarios con al menos 'bbox' y 'category_id'.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    padding = 2
    for ann in annotations:
        if "bbox" not in ann or "category_id" not in ann:
            continue
        box = ann["bbox"]
        # Se asume que bbox está en formato [xmin, ymin, xmax, ymax]
        x0, y0, x1, y1 = map(int, box)
        cat_id = int(ann["category_id"])
        color = color_map.get(cat_id, "red") if color_map is not None else "red"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if label_map is not None:
            label_text = label_map.get(cat_id, str(cat_id))
        else:
            label_text = str(cat_id)
        text = f"GT: {label_text}"
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
# Función de evaluación "clásica" para imagen
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
        outputs = model(im_tensor, orig_size)
        labels, boxes, scores = outputs

    label_map = get_label_map(cfg.val_dataloader.dataset)
    color_map = generate_color_map_from_label_map(label_map)
    
    # Con el modelo considerando etiquetas iniciando en 0, el offset debería ser 0
    label_offset = compute_label_offset(labels[0], label_map)
    
    if args.draw:
        im_copy = im_pil.copy()
        out_path = os.path.join(args.base_eval_dir, "result_single.jpg")
        draw_inferences(im_copy, labels[0], boxes[0], scores[0],
                        label_map=label_map, color_map=color_map, threshold=0.5, 
                        label_offset=label_offset, save_path=out_path)
    else:
        for j in range(len(labels[0])):
            if scores[0][j] > 0.5:
                pred_id = labels[0][j].item() + label_offset
                label_text = label_map.get(pred_id, str(pred_id))
                print(f"Label: {label_text}, Score: {scores[0][j].item():.2f}")

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
# Función de Debug: Comparativa entre predicción y ground truth
########################

def save_debug_comparison(original_image: Image.Image, pred_labels: torch.Tensor, pred_boxes: torch.Tensor, 
                            pred_scores: torch.Tensor, gt_annotations: list, label_map, color_map, 
                            threshold: float, label_offset: int, out_path: str):
    """
    Genera una imagen comparativa para debug:
      - Izquierda: Imagen con inferencias predichas por el modelo.
      - Derecha: Imagen con las anotaciones ground truth dibujadas.
    Guarda la imagen comparativa en 'out_path'.
    """
    pred_img = original_image.copy()
    gt_img = original_image.copy()
    
    pred_img = draw_inferences(pred_img, pred_labels, pred_boxes, pred_scores,
                               label_map=label_map, color_map=color_map, threshold=threshold, 
                               label_offset=label_offset)
    
    gt_img = draw_ground_truth(gt_img, gt_annotations, label_map=label_map, color_map=color_map)
    
    width, height = original_image.size
    combined = Image.new('RGB', (width*2, height))
    combined.paste(pred_img, (0,0))
    combined.paste(gt_img, (width,0))
    combined.save(out_path)

########################
# Función para generar la gráfica de distribución de instancias por clase en el dataset de validación
# con el valor exacto y también mostrar las rutas de las imágenes en consola.
########################

def plot_val_dataset_distribution(cfg, args):
    """
    Genera una gráfica de barras con el número de instancias por clase
    en el dataset de validación, mostrando además el valor exacto sobre cada barra.
    También imprime en consola las rutas donde se ubican las imágenes de validación.
    """
    dataset = cfg.val_dataloader.dataset
    label_map = get_label_map(dataset)

    # Imprimir en consola las rutas del dataset de validación
    val_image_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
    print(f"\nEl dataset de validación se localiza en las siguientes rutas:\n{val_image_dirs}")

    # Contar cuántas anotaciones hay por cada categoría
    cat_counts = Counter()
    ann_list = dataset.coco.dataset.get('annotations', [])
    for ann in ann_list:
        cat_counts[ann['category_id']] += 1

    # Ordenar por ID de categoría
    cat_ids = sorted(cat_counts.keys())
    cat_names = [label_map.get(cat_id, str(cat_id)) for cat_id in cat_ids]
    counts = [cat_counts[cat_id] for cat_id in cat_ids]

    # Graficar
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cat_names, counts, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Clases')
    plt.ylabel('Número de instancias')
    plt.title('Distribución de instancias por clase (Validación)')

    # Mostrar valor exacto sobre cada barra
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(counts) if counts else 0,
            str(height),
            ha='center',
            va='bottom',
            fontsize=9
        )

    dist_path = os.path.join(args.base_eval_dir, "dataset_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    print(f"Distribución de instancias por clase guardada en: {dist_path}")

########################
# Función para evaluación COCO
########################
def run_coco_evaluation(coco_gt, coco_pred_json_path, output_dir):
    """
    Ejecuta la evaluación estándar de COCO (mAP) usando pycocotools.
    """
    if not COCO_TOOLS_AVAILABLE:
        print("\n--- Evaluación COCO Omitida ---")
        print("La librería `pycocotools` no está instalada.")
        print("Para calcular el mAP, instálala con: pip install pycocotools")
        print(f"El archivo de predicciones ha sido guardado en: {coco_pred_json_path}")
        return

    print("\n--- Iniciando Evaluación Estándar COCO (mAP) ---")
    
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = []

    coco_dt = coco_gt.loadRes(coco_pred_json_path)

    # Inicializar COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Ejecutar evaluación
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Guardar resultados en un archivo de texto
    report_path = os.path.join(output_dir, "coco_evaluation_report.txt")
    original_stdout = sys.stdout
    with open(report_path, 'w') as f:
        sys.stdout = f
        coco_eval.summarize()
    sys.stdout = original_stdout  # Restaurar stdout
    print(f"Reporte de evaluación COCO guardado en: {report_path}")
    print("--------------------------------------------------")

########################
# Función de evaluación integrada (una sola pasada)
########################

def evaluate_on_dataset(args, cfg, device):
    model = ModelWrapper(cfg).to(device)
    model.eval()
    dataloader = cfg.val_dataloader
    dataset = dataloader.dataset
    label_map = get_label_map(dataset)
    color_map = generate_color_map_from_label_map(label_map)
    os.makedirs(args.base_eval_dir, exist_ok=True)
    plot_val_dataset_distribution(cfg, args)

    # Listas para acumulación global de detecciones emparejadas (para la matriz de confusión y reporte de clasificación)
    all_y_true = []
    all_y_pred = []

    # Contadores para falsos positivos (FP) y falsos negativos (FN) por detección
    fp_counter = Counter()
    fn_counter = Counter()
    
    # Lista para guardar predicciones en formato COCO
    coco_predictions = []

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
        elif isinstance(batch, (list, tuple)):
            images, targets = batch
            B = images.shape[0]
            orig_sizes = torch.tensor([[640, 640]] * B).to(device)
        else:
            raise ValueError("Formato de batch no reconocido.")
        images = images.to(device)
        with torch.no_grad():
            if orig_sizes.dim() == 1:
                orig_sizes = orig_sizes.unsqueeze(0)
            outputs = model(images, orig_sizes)
        labels_batch, boxes_batch, scores_batch = outputs
        
        # Procesamiento de cada imagen en el batch
        for i in range(len(images)):
            gt_boxes = []
            gt_labels = []
            gt_annotations = []
            image_id = None
            
            if targets is not None and isinstance(targets[i], dict):
                # Extraer image_id para la evaluación COCO
                if 'image_id' in targets[i]:
                    image_id = targets[i]['image_id'].item()
                
                if "annotations" in targets[i]:
                    for ann in targets[i]["annotations"]:
                        if "bbox" in ann and "category_id" in ann:
                            gt_annotations.append(ann)
                            gt_boxes.append(ann["bbox"])
                            gt_labels.append(int(ann["category_id"]))
                elif "boxes" in targets[i] and "labels" in targets[i]:
                    if hasattr(targets[i]["boxes"], "tolist"):
                        gt_boxes = targets[i]["boxes"].tolist()
                    else:
                        gt_boxes = targets[i]["boxes"]
                    if isinstance(targets[i]["labels"], torch.Tensor):
                        gt_labels = targets[i]["labels"].tolist()
                    else:
                        gt_labels = targets[i]["labels"]
                    gt_annotations = [{"bbox": box, "category_id": lab} for box, lab in zip(gt_boxes, gt_labels)]
            
            # Calcular el offset dinámico para las predicciones de esta imagen
            raw_pred_labels_tensor = labels_batch[i].cpu()
            label_offset = compute_label_offset(raw_pred_labels_tensor, label_map)
            
            pred_boxes = boxes_batch[i].cpu()
            pred_scores = scores_batch[i].cpu()

            # Acumulación de resultados para COCO eval
            if args.coco_eval and image_id is not None:
                for j in range(len(raw_pred_labels_tensor)):
                    box = pred_boxes[j].tolist()
                    x1, y1, x2, y2 = box
                    coco_box = [x1, y1, x2 - x1, y2 - y1] # Convertir a [x, y, w, h]
                    
                    pred_cat_id = raw_pred_labels_tensor[j].item() + label_offset
                    
                    coco_predictions.append({
                        'image_id': image_id,
                        'category_id': pred_cat_id,
                        'bbox': coco_box,
                        'score': pred_scores[j].item()
                    })

            raw_pred_labels = [int(x.item()) for x in raw_pred_labels_tensor]
            pred_labels_offset = [p + label_offset for p in raw_pred_labels]
            pred_boxes_list = pred_boxes.tolist()
            pred_scores_list = pred_scores.tolist()

            # Filtrar las predicciones según el umbral para las métricas personalizadas
            filtered_indices = [j for j, s in enumerate(pred_scores_list) if s > args.detection_threshold]
            filtered_pred_boxes = [pred_boxes_list[j] for j in filtered_indices]
            filtered_pred_labels = [pred_labels_offset[j] for j in filtered_indices]
            filtered_pred_scores = [pred_scores_list[j] for j in filtered_indices]
            
            if len(gt_boxes) == 0 and len(filtered_pred_boxes) == 0:
                continue

            matched, unmatched_gt, unmatched_pred = match_detections_in_image(
                gt_boxes, gt_labels, filtered_pred_boxes, filtered_pred_labels, filtered_pred_scores, iou_thresh=args.iou_threshold
            )

            # Acumular solo los emparejamientos para la matriz de confusión (excluyendo casos de no detección)
            for gt_lab, pred_lab in matched:
                all_y_true.append(gt_lab)
                all_y_pred.append(pred_lab)
            
            # Para cada GT sin match, se considera un falso negativo (se actualiza el contador)
            for gt_lab in unmatched_gt:
                fn_counter[gt_lab] += 1
                all_y_true.append(gt_lab)
                all_y_pred.append(-1)
            
            # Para cada predicción sin match, se considera un falso positivo
            for pred_lab in unmatched_pred:
                fp_counter[pred_lab] += 1
                all_y_true.append(-1)
                all_y_pred.append(pred_lab)
            
            if args.debug and targets is not None and isinstance(targets[i], dict):
                num_pred = sum(s > args.detection_threshold for s in pred_scores_list)
                if len(gt_annotations) != num_pred:
                    pil_img = T.ToPILImage()(images[i].cpu())
                    debug_out = os.path.join(args.base_eval_dir, f"debug_batch{idx}_img{i}.jpg")
                    save_debug_comparison(pil_img, raw_pred_labels_tensor, pred_boxes, pred_scores,
                                          gt_annotations, label_map, color_map, args.detection_threshold, label_offset, debug_out)
        # Si se desea guardar imágenes con inferencias
        if args.draw:
            for i in range(images.shape[0]):
                out_fname = os.path.join(args.base_eval_dir, "predicted_images", f"result_batch_{idx}_{i}.jpg")
                im = T.ToPILImage()(images[i].cpu())
                raw_pred_labels_tensor = labels_batch[i].cpu()
                label_offset = compute_label_offset(raw_pred_labels_tensor, label_map)
                draw_inferences(im, raw_pred_labels_tensor, boxes_batch[i].cpu(), scores_batch[i].cpu(),
                                label_map=label_map, color_map=color_map, threshold=0.5, 
                                label_offset=label_offset, save_path=out_fname)
        batch_count += 1
        if device.type == "cuda" and batch_count % 20 == 0:
            torch.cuda.empty_cache()
    
    # Métricas personalizadas existentes
    print("\n--- Métricas Personalizadas ---")
    all_labels = [-1] + sorted(label_map.keys())
    target_names = ["Background"] + [label_map[k] for k in sorted(label_map.keys())]
    
    # Prevenir error si no hay detecciones en absoluto
    if all_y_true and all_y_pred:
        cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)
        report = classification_report(all_y_true, all_y_pred, labels=all_labels,
                                       target_names=target_names, zero_division=0)
                                       
        num_classes = len(target_names)
        fig_size = max(8, 0.5 * num_classes)
        plt.figure(figsize=(fig_size, fig_size))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                         xticklabels=target_names,
                         yticklabels=target_names)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Matriz de Confusión")
        cm_path = os.path.join(args.base_eval_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"Matriz de Confusión guardada en: {cm_path}")

        plt.figure(figsize=(8, 8))
        plt.text(0.01, 0.05, report, {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.title("Reporte de Clasificación (Detecciones emparejadas)")
        report_path = os.path.join(args.base_eval_dir, "classification_report.png")
        plt.tight_layout()
        plt.savefig(report_path)
        plt.close()
        print(f"Reporte de Clasificación guardado en: {report_path}")
    else:
        print("No se generaron detecciones o ground truths, se omiten las métricas de clasificación.")

    fp_fn_report = "Reporte de Falsos Positivos y Falsos Negativos (No Detecciones)\n"
    fp_fn_report += "Categoría\tFalsos Positivos\tFalsos Negativos\n"
    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]
        fp = fp_counter.get(cat_id, 0)
        fn = fn_counter.get(cat_id, 0)
        fp_fn_report += f"{cat_name} ({cat_id})\t{fp}\t{fn}\n"
    
    fp_fn_report_path = os.path.join(args.base_eval_dir, "fp_fn_report.txt")
    with open(fp_fn_report_path, "w") as f:
        f.write(fp_fn_report)
    print(f"Reporte de Falsos Positivos y Falsos Negativos guardado en: {fp_fn_report_path}")
    
    classes = [label_map[k] for k in sorted(label_map.keys())]
    fp_counts = [fp_counter.get(k, 0) for k in sorted(label_map.keys())]
    fn_counts = [fn_counter.get(k, 0) for k in sorted(label_map.keys())]
    
    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, fp_counts, width, label='Falsos Positivos', color='salmon')
    plt.bar(x + width/2, fn_counts, width, label='Falsos Negativos', color='lightblue')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.xlabel("Clases")
    plt.ylabel("Cantidad")
    plt.title("Falsos Positivos y Falsos Negativos por Clase (No Detecciones)")
    plt.legend()
    fp_fn_bar_path = os.path.join(args.base_eval_dir, "fp_fn_bar_chart.png")
    plt.tight_layout()
    plt.savefig(fp_fn_bar_path)
    plt.close()
    print(f"Gráfica de FP y FN guardada en: {fp_fn_bar_path}")

    # Ejecutar evaluación COCO si se solicita
    if args.coco_eval:
        pred_json_path = os.path.join(args.base_eval_dir, "coco_predictions.json")
        with open(pred_json_path, 'w') as f:
            json.dump(coco_predictions, f)
        
        # El objeto 'dataset.coco' es la instancia de la API de COCO para el ground truth
        run_coco_evaluation(dataset.coco, pred_json_path, args.base_eval_dir)

########################
# Función principal
########################

def main():
    parser = argparse.ArgumentParser(
        description="Script de evaluación del modelo con opción de visualización, métricas adicionales, reporte de falsos positivos y falsos negativos, y función de debug comparativo."
    )
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Ruta al archivo de configuración YAML.")
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help="Ruta al checkpoint del modelo.")
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help="Dispositivo de cómputo (e.g., cpu o cuda).")
    parser.add_argument('-dt', '--detection_threshold', type=float, default=0.5,
                        help="Threshold de detección como true positive")
    parser.add_argument('-iout', '--iou_threshold', type=float, default=0.5,
                        help="Threshold de IoU para asignación de predicción")
    parser.add_argument('--im_file', type=str, default=None,
                        help="Ruta a una imagen individual para inferencia (modo imagen única).")
    parser.add_argument('--draw', action='store_true',
                        help="Si se especifica, dibuja y guarda las imágenes con inferencias.")
    parser.add_argument('--debug', action='store_true',
                        help="Si se especifica, guarda imágenes de debug comparativas entre predicción y ground truth.")
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help="Directorio base para guardar las imágenes y métricas.")
    parser.add_argument('--coco_eval', action='store_true',
                        help="Si se especifica, ejecuta la evaluación estándar de COCO (mAP).")
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
    model_instance.load_state_dict(state, strict=False)
    
    if args.draw:
        predicted_images_dir = os.path.join(args.base_eval_dir, "predicted_images")
        os.makedirs(predicted_images_dir, exist_ok=True)
        print(f"Las imágenes inferidas se guardarán en: {predicted_images_dir}")
    
    if args.im_file is not None:
        print("Evaluación sobre imagen individual...")
        evaluate_on_image(args, cfg, device)
    else:
        print("Evaluación sobre dataset de validación...")
        evaluate_on_dataset(args, cfg, device)

if __name__ == '__main__':
    main()