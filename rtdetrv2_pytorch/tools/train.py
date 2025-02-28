"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import argparse
import numpy as np
import albumentations as A
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Agregar el directorio padre al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS

########################################
# Funciones de análisis y aumento personalizado
########################################

def compute_class_distribution(dataset):
    """
    Recorre el dataset y devuelve un diccionario con la frecuencia de cada clase.
    Se asume que cada muestra retorna una tupla (imagen, target) y que en target
    existe una clave 'labels' con la lista de etiquetas.
    """
    class_counts = {}
    for img, target in dataset:
        labels = target.get('labels', [])
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def get_underrepresented_classes(class_distribution, threshold_ratio=0.5):
    """
    Devuelve una lista de clases cuya frecuencia es menor que el umbral,
    definido como un ratio (por defecto 0.5) del valor medio de la distribución.
    """
    total = sum(class_distribution.values())
    num_classes = len(class_distribution)
    mean_count = total / num_classes if num_classes > 0 else 0
    threshold = mean_count * threshold_ratio
    underrepresented = [cls for cls, count in class_distribution.items() if count < threshold]
    return underrepresented

# Definir un pipeline de aumentos de datos para detección de residuos en fondo marino,
# utilizando las operaciones definidas en el fichero de configuración.
# Se incluye un pipeline que también procesa las bounding boxes.
underrep_augmentation_pipeline = A.Compose([
    # RandomPhotometricDistort: se simula con ColorJitter
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # RandomGaussianBlur
    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.5), p=0.3),
    # RandomNoise
    A.GaussNoise(var_limit=(0.00002, 0.00003), mean=0, p=0.3),
    # RandomRotation
    A.Rotate(limit=15, p=0.4),
    # RandomZoomOut se simula con ShiftScaleRotate con escala negativa
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0), rotate_limit=0, p=0.3),
    # RandomIoUCrop: se utiliza RandomSizedBBoxSafeCrop
    A.RandomSizedBBoxSafeCrop(height=640, width=640, erosion_rate=0.0, p=0.8),
    # RandomHorizontalFlip
    A.HorizontalFlip(p=0.5),
    # Resize a 640x640
    A.Resize(640, 640)
    # Se omiten las operaciones de sanitización y conversión que se manejan en otro bloque
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def custom_augment(image, target, underrepresented_classes, augmentation_pipeline):
    """
    Aplica el pipeline de aumentos si la muestra contiene alguna clase subrepresentada.
    Se asume que target es un diccionario con 'labels' y opcionalmente 'bboxes'.
    """
    labels = target.get('labels', [])
    if any(label in underrepresented_classes for label in labels):
        # Convertir la imagen a numpy array si no lo es
        image_np = np.array(image) if not isinstance(image, np.ndarray) else image
        # Extraer bounding boxes (se espera formato pascal_voc: [x_min, y_min, x_max, y_max])
        bboxes = target.get('bboxes', [])
        augmented = augmentation_pipeline(image=image_np, bboxes=bboxes, labels=labels)
        augmented_image = augmented['image']
        target['bboxes'] = augmented.get('bboxes', bboxes)
        return augmented_image, target
    else:
        return image, target

########################################
# Función principal de entrenamiento
########################################

def main(args) -> None:
    """Función principal para entrenar RT-DETRv2 con aumento de datos condicional."""
    # Configuración distribuida
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # Se permite únicamente una opción entre from_scratch, resume o tuning
    assert not all([args.tuning, args.resume]), \
        'Solo se admite from_scratch o resume o tuning a la vez'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg:', cfg.__dict__)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # Análisis y procesamiento del dataset de entrenamiento
    dataset_train = cfg.train_dataloader.dataset

    # Calcular la distribución de clases
    # class_distribution = compute_class_distribution(dataset_train)
    # print("Distribución de clases:", class_distribution)

    # Identificar clases subrepresentadas (p.ej., aquellas con menos de la mitad del promedio)
    # underrepresented_classes = get_underrepresented_classes(class_distribution)
    # print("Clases subrepresentadas:", underrepresented_classes)

    # Integra la función de aumento personalizado en el dataset.
    # Se asume que cada muestra es una tupla (imagen, target)
    #dataset_train.transform = lambda sample: custom_augment(*sample, underrepresented_classes, underrep_augmentation_pipeline)

    print("Analizando el dataset de entrenamiento...")
    # Opcional: guardar un análisis visual del dataset de entrenamiento
    dataset_train.analyze_dataset(
        save_path=f"{cfg.output_dir}/train_dataset_analysis.png",
        split_name="Entrenamiento"
    )

    # Análisis del dataset de validación (sin aplicar aumento condicional)
    print("Analizando el dataset de validación...")
    dataset_val = cfg.val_dataloader.dataset
    dataset_val.analyze_dataset(
        save_path=f"{cfg.output_dir}/val_dataset_analysis.png",
        split_name="Validación"
    )

    # Crear el solver a partir de la tarea definida en el YAML
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argumentos prioritarios
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='Reanudar desde checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='Afinar desde checkpoint')
    parser.add_argument('-d', '--device', type=str, help='Dispositivo (GPU, CPU, etc.)')
    parser.add_argument('--seed', type=int, help='Semilla para reproducibilidad')
    parser.add_argument('--use-amp', action='store_true', help='Entrenamiento con precisión mixta automática')
    parser.add_argument('--output-dir', type=str, help='Directorio de salida')
    parser.add_argument('--summary-dir', type=str, help='Directorio para resúmenes de tensorboard')
    parser.add_argument('--test-only', action='store_true', default=False, help='Solo ejecutar validación')

    # Actualización de configuración YAML
    parser.add_argument('-u', '--update', nargs='+', help='Actualizar configuración YAML')

    # Argumentos del entorno
    parser.add_argument('--print-method', type=str, default='builtin', help='Método de impresión')
    parser.add_argument('--print-rank', type=int, default=0, help='ID de impresión por rango')
    parser.add_argument('--local-rank', type=int, help='ID de rango local')

    args = parser.parse_args()
    main(args)
