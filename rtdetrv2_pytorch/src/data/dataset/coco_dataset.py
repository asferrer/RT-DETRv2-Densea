"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import json
import torch
import torch.utils.data
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import torchvision
torchvision.disable_beta_transforms_warning()

from PIL import Image 
from pycocotools import mask as coco_mask

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['CocoDetection']

@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self._initialize_categories()

    def _initialize_categories(self):
        """
        Lee las categorías directamente del archivo de anotaciones COCO.
        Crea los mapeos necesarios para category_id -> nombre y otros.
        """
        self.categories = self.coco.dataset['categories']
        self.category2name = {cat['id']: cat['name'] for cat in self.categories}
        self.mscoco_category2label = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.mscoco_label2category = {i: cat['id'] for i, cat in enumerate(self.categories)}

        # Log de las categorías inicializadas
        print(f"Categorías inicializadas: {len(self.categories)} categorías cargadas.")
        print("Mapping: ", self.category2name)
    
    @classmethod
    def get_category_mappings(cls, coco_instance):
        """
        Returns the category mappings (category2name, mscoco_category2label, mscoco_label2category).

        Args:
            coco_instance (CocoDetection): An instance of CocoDetection.

        Returns:
            dict: A dictionary containing the category mappings.
        """
        return {
            "category2name": coco_instance.category2name,
            "mscoco_category2label": coco_instance.mscoco_category2label,
            "mscoco_label2category": coco_instance.mscoco_label2category,
        }

    def analyze_dataset(self, save_path=None, split_name="Dataset"):
        """
        Analyzes the dataset and generates a distribution graph for the number of instances per class.

        Args:
            save_path (str): Path to save the plot (optional).
            split_name (str): Name of the dataset split (e.g., "Train", "Val").
        """
        # Count categories directly from annotations
        class_counts = Counter()
        for ann in self.coco.dataset['annotations']:
            class_counts[ann['category_id']] += 1

        # Map IDs to class names
        class_counts_named = {self.category2name[k]: v for k, v in class_counts.items()}

        # Prepare data for Seaborn
        data = {
            "Clase": list(class_counts_named.keys()),
            "Número de Instancias": list(class_counts_named.values())
        }

        # Convert data into a DataFrame for compatibility
        import pandas as pd
        df = pd.DataFrame(data)

        # Generate plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x="Número de Instancias", 
            y="Clase", 
            data=df, 
            palette="viridis", 
            hue="Clase", 
            dodge=False, 
            legend=False  # Ensure no legend is displayed
        )
        plt.title(f"Distribución de Instancias por Clase en {split_name}", fontsize=16)
        plt.xlabel("Número de Instancias", fontsize=14)
        plt.ylabel("Clases", fontsize=14)

        # Annotate each bar with the count value
        for i, (count, y) in enumerate(zip(df["Número de Instancias"], df["Clase"])):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=12)

        # Save or show the plot
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()

        return class_counts_named
    
    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=self.mscoco_category2label)
            # image, target = self.prepare(image, target, category2label=self.mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        
        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
            
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target

mscoco_category2name = {
    0: "Can",
    1: "Squared_Can",
    2: "Wood",
    3: "Bottle",
    4: "Plastic_Bag",
    5: "Glove",
    6: "Fishing_Net",
    7: "Tire",
    8: "Packaging_Bag",
    9: "WashingMachine",
    10: "Metal_Chain",
    11: "Rope",
    12: "Towel",
    13: "Plastic_Debris",
    14: "Metal_Debris",
    15: "Pipe",
    16: "Shoe",
    17: "Car_Bumper",
    18: "Basket",
    19: "Mask"
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

@register()
class MultiDirCocoDetection(CocoDetection):
    """
    Variante que permite especificar múltiples carpetas `image_dirs` o `image_dir` en lugar
    de un único `img_folder`. Así, para cada imagen, buscamos en cada carpeta
    hasta encontrarla.
    """
    def __init__(
        self, 
        image_dir=None, 
        image_dirs=None, 
        ann_file=None, 
        anno_path=None, 
        img_folder=None,  # Añadido para aceptar img_folder y evitar errores
        transforms=None, 
        return_masks=False, 
        remap_mscoco_category=False, 
        **kwargs
    ):
        """
        Args:
            image_dir (list[str], optional): Lista de directorios de imágenes (compatibilidad con configuraciones anteriores).
            image_dirs (list[str], optional): Nueva lista de directorios de imágenes.
            ann_file (str, optional): Ruta al archivo JSON de anotaciones COCO.
            anno_path (str, optional): Ruta al archivo JSON de anotaciones COCO (compatibilidad con configuraciones anteriores).
            img_folder (str, optional): Carpeta de imágenes (ignorada en esta clase).
            transforms: Transformaciones a aplicar a las imágenes y objetivos.
            return_masks (bool): Si es True, retorna máscaras.
            remap_mscoco_category (bool): Si es True, remapea las categorías de COCO a 0..N-1.
            kwargs: Otros argumentos adicionales.
        """
        # Eliminar 'name' y 'img_folder' de kwargs para evitar duplicados
        kwargs.pop('name', None)
        kwargs.pop('img_folder', None)

        # Asignar ann_file desde ann_file o anno_path
        if ann_file is None:
            ann_file = anno_path
        if ann_file is None:
            raise ValueError("Debes proporcionar 'ann_file' o 'anno_path'.")

        # Combinar image_dirs y image_dir
        image_dirs_combined = []
        if image_dirs:
            image_dirs_combined += image_dirs
        if image_dir:
            if not isinstance(image_dir, list):
                raise TypeError("'image_dir' debe ser una lista de directorios.")
            image_dirs_combined += image_dir

        if not image_dirs_combined:
            raise ValueError("Debes proporcionar al menos una carpeta en 'image_dirs' o 'image_dir'.")

        # Inicializar la clase padre con img_folder vacío
        super().__init__(
            img_folder="", 
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
            **kwargs
        )

        self.image_dirs = image_dirs_combined

    def load_item(self, idx):
        """
        Sobrescribimos el método load_item para buscar imágenes en múltiples directorios.
        """
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(img_id)
        target_anno = coco.loadAnns(ann_ids)

        # Obtener el nombre del archivo relativo desde las anotaciones
        path = coco.loadImgs(img_id)[0]['file_name']

        # Buscar la imagen en cada directorio
        found_path = None
        for d in self.image_dirs:
            candidate = os.path.join(d, path)
            if os.path.exists(candidate):
                found_path = candidate
                break
        if found_path is None:
            raise FileNotFoundError(f"No se encontró la imagen '{path}' en los directorios {self.image_dirs}")

        image = Image.open(found_path).convert("RGB")
        target = {'image_id': img_id, 'annotations': target_anno}

        # Aplicar la preparación (ConvertCocoPolysToMask, etc.)
        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=self.mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        # Añadir el índice
        target['idx'] = torch.tensor([idx])

        # Convertir boxes y masks a tensores si están presentes
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'], key='boxes', spatial_size=image.size[::-1]
            )
        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

def analyze_annotations(coco, save_path=None, split_name="Dataset"):
    """
    Analiza las anotaciones COCO y genera un gráfico con el número de instancias por clase usando Seaborn.

    Args:
        coco: Instancia COCO con las anotaciones cargadas.
        save_path (str): Ruta para guardar la gráfica generada (opcional).
        split_name (str): Nombre del split analizado (e.g., "Train", "Val").

    Returns:
        dict: Conteo de instancias por clase.
    """
    # Contar categorías directamente desde las anotaciones
    class_counts = Counter()
    for ann in coco.dataset['annotations']:
        class_counts[ann['category_id']] += 1

    # Mapear IDs a nombres de clase
    id_to_name = {cat['id']: cat['name'] for cat in coco.dataset['categories']}
    class_counts_named = {id_to_name[k]: v for k, v in class_counts.items()}

    # Preparar datos para Seaborn
    data = {
        "Clase": list(class_counts_named.keys()),
        "Número de Instancias": list(class_counts_named.values())
    }

    # Crear gráfico
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Número de Instancias", y="Clase", data=data, palette="viridis")
    plt.title(f"Distribución de Instancias por Clase en {split_name}", fontsize=16)
    plt.xlabel("Número de Instancias", fontsize=14)
    plt.ylabel("Clases", fontsize=14)

    # Guardar o mostrar el gráfico
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()

    return class_counts_named