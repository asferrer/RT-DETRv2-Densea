#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T

from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Añadir la ruta raíz del proyecto (ajusta según tu estructura)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Importar módulos internos
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import CocoDetection  # fallback para dataset
from src.data.dataset.coco_utils import convert_to_coco_api
from src.data.dataset.coco_eval import CocoEvaluator

##############################################################################
# Diccionario con IDs -> nombres de clase para RT-DETRv2 (ajusta según tu dataset)
##############################################################################
rt_detr_category2name = {
    1:  "Can",
    2:  "Squared_Can",
    3:  "Wood",
    4:  "Bottle",
    5:  "Plastic_Bag",
    6:  "Glove",
    7:  "Fishing_Net",
    8:  "Tire",
    9:  "Packaging_Bag",
    10: "WashingMachine",
    11: "Metal_Chain",
    12: "Rope",
    13: "Towel",
    14: "Plastic_Debris",
    15: "Metal_Debris",
    16: "Pipe",
    17: "Shoe",
    18: "Car_Bumper",
    19: "Basket",
    20: "Mask",
}

###########################################################################
# Custom Compose y funciones de transformación para soportar la firma:
#   (img, target, dataset)
###########################################################################
class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target, dataset):
        for t in self.transforms:
            img, target, dataset = t(img, target, dataset)
        return img, target, dataset

def transform_resize(size):
    def _resize(img, target, dataset):
        # Guardar el tamaño original (en formato [alto, ancho])
        if "orig_size" not in target:
            target["orig_size"] = torch.tensor(img.size[::-1])
        orig_size = img.size  # (ancho, alto)
        img = img.resize(tuple(size), Image.BILINEAR)
        # Actualizar bounding boxes si existen (se asume formato [x_min, y_min, x_max, y_max])
        if "boxes" in target:
            scale_x = size[0] / orig_size[0]
            scale_y = size[1] / orig_size[1]
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                scale = torch.tensor([scale_x, scale_y, scale_x, scale_y],
                                     dtype=boxes.dtype, device=boxes.device)
                target["boxes"] = boxes * scale
        return img, target, dataset
    return _resize

def transform_convert_pil(dtype='float32', scale=True):
    def _convert(img, target, dataset):
        # Convertir imagen PIL a tensor (escala a [0,1])
        img = T.ToTensor()(img)
        return img, target, dataset
    return _convert

def build_transforms(transform_cfg):
    """
    Construye la pipeline de transformaciones a partir de la configuración YAML.
    Se crean funciones personalizadas que reciben (img, target, dataset).
    """
    transforms_list = []
    ops = transform_cfg.get("ops", [])
    for op in ops:
        op_type = op.get("type")
        if op_type == "Resize":
            size = op.get("size", [640, 640])
            transforms_list.append(transform_resize(size))
        elif op_type == "ConvertPILImage":
            dtype = op.get("dtype", "float32")
            scale = op.get("scale", True)
            transforms_list.append(transform_convert_pil(dtype, scale))
        else:
            logging.warning(f"Transform op '{op_type}' no implementado. Se omitirá.")
    if not transforms_list:
        transforms_list = [transform_resize([640, 640]), transform_convert_pil()]
    return CustomCompose(transforms_list)

###########################################################################
# Función de collate personalizada para evitar que se intente apilar
# campos con número variable de cajas (por ejemplo, 'boxes')
###########################################################################
def custom_collate(batch):
    # Se apilan las imágenes (suponiendo que tienen el mismo tamaño)
    images = torch.stack([item[0] for item in batch], dim=0)
    # Se dejan las anotaciones como lista (cada elemento es un diccionario)
    targets = [item[1] for item in batch]
    return images, targets

###########################################################################
# Funciones auxiliares: parse_args, load_model, prepare_dataset, etc.
###########################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de modelo RT-DETRv2 (detección y clasificación).")
    parser.add_argument("-c", "--config", required=True, help="Ruta al archivo de configuración .yaml")
    parser.add_argument("-m", "--model", required=True, help="Ruta al checkpoint .pth entrenado")
    parser.add_argument("-o", "--output", required=True, help="Directorio para guardar resultados (se puede sobreescribir con config.yaml)")
    parser.add_argument("-d", "--device", default="cuda", help="Dispositivo para evaluación: 'cpu' o 'cuda'")
    return parser.parse_args()

def load_model(model_path, device, cfg):
    # Cargar el checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"]
    else:
        state_dict = checkpoint["model"]

    # Cargar pesos en el modelo (se asume que cfg.model ya está construido en la configuración)
    cfg.model.load_state_dict(state_dict)
    
    # Definir un wrapper para RT-DETRv2 que use deploy() en modelo y postprocesador
    class RTDETRv2Wrapper(nn.Module):
        def __init__(self, model, postprocessor):
            super(RTDETRv2Wrapper, self).__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model_wrapper = RTDETRv2Wrapper(cfg.model, cfg.postprocessor)
    return model_wrapper.to(device)

def prepare_dataset(config):
    """
    Prepara el dataset de validación utilizando la sección 'val_dataloader' de la configuración.
    Se contempla que el dataset pueda ser del tipo 'MultiDirCocoDetection' o el por defecto 'CocoDetection'.
    """
    val_cfg = config.yaml_cfg["val_dataloader"]
    ds_cfg = val_cfg["dataset"]

    # Construir la pipeline de transformaciones a partir de la configuración YAML
    if "transforms" in ds_cfg:
        transforms_compose = build_transforms(ds_cfg["transforms"])
    else:
        transforms_compose = CustomCompose([transform_resize([640, 640]), transform_convert_pil()])

    dataset_type = ds_cfg.get("type", "CocoDetection")
    if dataset_type == "MultiDirCocoDetection":
        # Importar la clase correspondiente (asegúrate de que esté definida en el módulo correcto)
        from src.data.dataset.coco_dataset import MultiDirCocoDetection
        dataset = MultiDirCocoDetection(
            image_dir=ds_cfg["image_dir"],
            anno_path=ds_cfg["anno_path"],
            transforms=transforms_compose
        )
    else:
        dataset = CocoDetection(
            img_folder=ds_cfg["img_folder"],
            ann_file=ds_cfg["ann_file"],
            transforms=transforms_compose
        )
    return dataset

def plot_confusion_matrix(conf_matrix, classes, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matriz_confusion.png"))
    plt.close()

def plot_metrics_counts(conf_matrix, output_dir):
    fp = (conf_matrix.sum(axis=0) - np.diag(conf_matrix)).sum()
    fn = (conf_matrix.sum(axis=1) - np.diag(conf_matrix)).sum()
    tp = np.diag(conf_matrix).sum()
    tn = conf_matrix.sum() - (fp + fn + tp)

    metrics_counts = {"FP": int(fp), "FN": int(fn), "TP": int(tp), "TN": int(tn)}
    plt.figure(figsize=(5, 4))
    plt.bar(metrics_counts.keys(), metrics_counts.values(),
            color=["red", "orange", "green", "blue"])
    plt.title("Conteo de FP, FN, TP, TN")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conteo_metricas.png"))
    plt.close()

def draw_images_with_predictions(image, labels, boxes, scores, output_path, threshold=0.6):
    """
    Dibuja sobre la imagen las predicciones que superan el umbral indicado.
    """
    draw = ImageDraw.Draw(image)
    mask = scores > threshold
    valid_labels = labels[mask]
    valid_boxes = boxes[mask]
    valid_scores = scores[mask]

    for j, box in enumerate(valid_boxes):
        class_id = valid_labels[j].item()
        class_name = rt_detr_category2name.get(class_id, f"ID:{class_id}")
        conf_val = round(valid_scores[j].item(), 2)
        draw.rectangle(list(box), outline='red', width=2)
        draw.text((box[0], box[1]), text=f"{class_name} {conf_val}", fill='blue')
    
    image.save(output_path)

def evaluate_model(model, data_loader, coco_evaluator, device, output_dir):
    model.eval()
    all_targets = []
    all_predictions = []
    drawn_images = []
    max_draw = 10
    draw_count = 0

    for samples, targets in data_loader:
        samples = samples.to(device)
        # Recoger y mover a dispositivo los tamaños originales para el postprocesamiento
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)

        with torch.no_grad():
            outputs = model(samples, orig_sizes)

        # Asegurar que el campo "boxes" sea un tensor con forma [N, 4]
        for output in outputs:
            # Si no es tensor, lo convertimos
            if not isinstance(output["boxes"], torch.Tensor):
                output["boxes"] = torch.tensor(output["boxes"])
            # Si es 1D, agregar dimensión de batch
            if output["boxes"].ndim == 1:
                output["boxes"] = output["boxes"].unsqueeze(0)
            # Si es 2D pero la segunda dimensión no es 4, intentar remodelar
            elif output["boxes"].ndim == 2 and output["boxes"].shape[1] != 4:
                try:
                    output["boxes"] = output["boxes"].view(-1, 4)
                except Exception as e:
                    logging.error(f"No se pudo remodelar 'boxes': {output['boxes'].shape}, error: {e}")

        # Actualizar el evaluador COCO
        results = {t["image_id"].item(): output for t, output in zip(targets, outputs)}
        coco_evaluator.update(results)

        # Recolectar etiquetas para las métricas de clasificación
        for target, output in zip(targets, outputs):
            gt_labels = target["labels"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()

            all_targets.extend(gt_labels)
            all_predictions.extend(pred_labels)

            # Guardar algunas imágenes para visualización si se dispone de 'file_name'
            if draw_count < max_draw and "file_name" in target:
                img_path = os.path.join(data_loader.dataset.image_dir[0] if hasattr(data_loader.dataset, "image_dir") else "", target["file_name"])
                if os.path.exists(img_path):
                    img_pil = Image.open(img_path).convert("RGB")
                    drawn_images.append({
                        "image": img_pil,
                        "labels": output["labels"].cpu(),
                        "boxes": output["boxes"].cpu(),
                        "scores": output["scores"].cpu(),
                        "file_name": target["file_name"]
                    })
                    draw_count += 1

    # Dibujar las imágenes con las predicciones
    for item in drawn_images:
        output_image_path = os.path.join(output_dir, f"resultado_{item['file_name']}")
        draw_images_with_predictions(
            item["image"],
            item["labels"],
            item["boxes"],
            item["scores"],
            output_image_path
        )

    # Finalizar la evaluación COCO
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return all_targets, all_predictions

def main():
    args = parse_args()

    # Configurar logging para seguimiento
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Iniciando evaluación de RT-DETRv2")

    # Cargar configuración (se procesa el YAML con sus includes)
    config = YAMLConfig(args.config)
    # Se prioriza la clave 'output_dir' definida en el YAML (si existe)
    output_dir = config.yaml_cfg.get("output_dir", args.output)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Cargar el modelo RT-DETRv2
    model = load_model(args.model, device, config)
    logging.info("Modelo cargado correctamente.")

    # Preparar el dataset y DataLoader de validación
    dataset = prepare_dataset(config)
    val_cfg = config.yaml_cfg["val_dataloader"]
    batch_size = val_cfg.get("total_batch_size", 1)
    shuffle = val_cfg.get("shuffle", False)
    num_workers = val_cfg.get("num_workers", 0)
    drop_last = val_cfg.get("drop_last", False)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate  # Usamos la función de collate personalizada
    )
    logging.info("Dataset y DataLoader preparados.")

    # Configurar el evaluador COCO
    iou_types = config.yaml_cfg.get("evaluator", {}).get("iou_types", ["bbox"])
    coco_api = convert_to_coco_api(dataset)
    coco_evaluator = CocoEvaluator(coco_api, iou_types)
    
    # Lanzar la evaluación del modelo
    all_targets, all_predictions = evaluate_model(model, data_loader, coco_evaluator, device, output_dir)
    logging.info("Evaluación del modelo completada.")

    # Calcular las métricas de clasificación
    logging.info("Calculando matriz de confusión y reporte de clasificación...")
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    unique_labels = sorted(set(all_targets).union(all_predictions))
    if hasattr(dataset, "category2name"):
        class_names = [dataset.category2name.get(cid, f"ID:{cid}") for cid in unique_labels]
    else:
        class_names = [rt_detr_category2name.get(cid, f"ID:{cid}") for cid in unique_labels]

    conf_matrix_reindexed = conf_matrix[np.ix_(unique_labels, unique_labels)]

    plot_confusion_matrix(conf_matrix_reindexed, class_names, output_dir)
    plot_metrics_counts(conf_matrix_reindexed, output_dir)

    report_dict = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    report_path = os.path.join(output_dir, "reporte_clasificacion.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

    logging.info(f"Evaluación completada. Resultados guardados en '{output_dir}'.")

if __name__ == "__main__":
    main()
