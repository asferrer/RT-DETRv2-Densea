"""
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
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, Mask

from PIL import Image 
from pycocotools import mask as coco_mask

from ._dataset import DetDataset
from ...core import register

__all__ = ['CocoDetection', 'MultiDirCocoDetection', 'mscoco_category2name', 'mscoco_category2label', 'mscoco_label2category', 'analyze_annotations']

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
        # El atributo 'epoch' se establece a través de set_epoch para evitar conflictos
    
    def set_epoch(self, epoch):
        """ This is called by the solver to set the epoch.
            We set the internal `_epoch` attribute to bypass the read-only property.
        """
        self._epoch = epoch

    def _initialize_categories(self):
        """
        Lee las categorías directamente del archivo de anotaciones COCO.
        """
        self.categories = self.coco.dataset['categories']
        self.category2name = {cat['id']: cat['name'] for cat in self.categories}
        self.mscoco_category2label = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.mscoco_label2category = {i: cat['id'] for i, cat in enumerate(self.categories)}
        print(f"Categorías inicializadas: {len(self.categories)} categorías cargadas.")

    @classmethod
    def get_category_mappings(cls, coco_instance):
        return {
            "category2name": coco_instance.category2name,
            "mscoco_category2label": coco_instance.mscoco_category2label,
            "mscoco_label2category": coco_instance.mscoco_label2category,
        }

    def analyze_dataset(self, save_path=None, split_name="Dataset"):
        class_counts = Counter()
        for ann in self.coco.dataset['annotations']:
            class_counts[ann['category_id']] += 1
        class_counts_named = {self.category2name.get(k, k): v for k, v in class_counts.items()}
        import pandas as pd
        df = pd.DataFrame({"Clase": list(class_counts_named.keys()), "Número de Instancias": list(class_counts_named.values())})
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x="Número de Instancias", y="Clase", data=df, palette="viridis", hue="Clase", dodge=False, legend=False)
        plt.title(f"Distribución de Instancias por Clase en {split_name}", fontsize=16)
        plt.xlabel("Número de Instancias", fontsize=14)
        plt.ylabel("Clases", fontsize=14)
        for i, (count, y) in enumerate(zip(df["Número de Instancias"], df["Clase"])):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=12)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()
        return class_counts_named

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        
        if target is None:
            return self.__getitem__((idx + 1) % len(self))

        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        
        return img, target

    def load_item(self, idx):
        image, target_anns = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target_anns}
        image, target = self.prepare(image, target, remap=self.remap_mscoco_category, category2label=self.mscoco_category2label)
        
        if 'boxes' not in target or target['boxes'].shape[0] == 0:
            return image, None

        target['boxes'] = BoundingBoxes(
            target['boxes'],
            format="XYXY",
            canvas_size=image.size[::-1]
        )
        if 'masks' in target:
            target['masks'] = Mask(target['masks'])
        
        target['idx'] = torch.tensor([idx])
        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
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
        image_id = torch.tensor([target["image_id"]])
        anno = [obj for obj in target["annotations"] if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        
        if not anno:
            return image, {"image_id": image_id, "boxes": torch.zeros(0, 4, dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.int64)}

        boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label')
        labels = [category2label.get(obj["category_id"], -1) if kwargs.get('remap') else obj["category_id"] for obj in anno]
        labels = torch.tensor(labels, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) & (labels != -1)
        
        target_out = {"image_id": image_id, "boxes": boxes[keep], "labels": labels[keep]}
        
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            target_out["masks"] = convert_coco_poly_to_mask(segmentations, h, w)[keep]

        if anno and "keypoints" in anno[0]:
            keypoints = torch.as_tensor([obj["keypoints"] for obj in anno], dtype=torch.float32)
            if keypoints.shape[0] > 0:
                target_out["keypoints"] = keypoints.view(keypoints.shape[0], -1, 3)[keep]

        target_out["area"] = torch.tensor([obj["area"] for obj in anno])[keep]
        target_out["iscrowd"] = torch.tensor([obj.get("iscrowd", 0) for obj in anno])[keep]
        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
    
        return image, target_out

mscoco_category2name = {
    0: "Can", 1: "Squared_Can", 2: "Wood", 3: "Bottle", 4: "Plastic_Bag",
    5: "Glove", 6: "Fishing_Net", 7: "Tire", 8: "Packaging_Bag", 9: "WashingMachine",
    10: "Metal_Chain", 11: "Rope", 12: "Towel", 13: "Plastic_Debris", 14: "Metal_Debris",
    15: "Pipe", 16: "Shoe", 17: "Car_Bumper", 18: "Basket", 19: "Mask"
}
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

@register()
class MultiDirCocoDetection(CocoDetection):
    def __init__(self, image_dirs=None, ann_file=None, transforms=None, **kwargs):
        # Para mantener la compatibilidad, aceptamos ambos nombres de argumento
        if 'anno_path' in kwargs:
            ann_file = kwargs.pop('anno_path')
        if 'image_dir' in kwargs:
            image_dirs = kwargs.pop('image_dir')

        if not ann_file: raise ValueError("Debes proporcionar 'ann_file' o 'anno_path'.")
        if not image_dirs: raise ValueError("Debes proporcionar 'image_dirs' o 'image_dir'.")

        super().__init__(img_folder="", ann_file=ann_file, transforms=transforms, **kwargs)
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]

    def load_item(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        target_anno = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        path = coco.loadImgs(img_id)[0]['file_name']

        found_path = next((os.path.join(d, path) for d in self.image_dirs if os.path.exists(os.path.join(d, path))), None)
        if not found_path:
            raise FileNotFoundError(f"No se encontró la imagen '{path}' en los directorios {self.image_dirs}")

        image = Image.open(found_path).convert("RGB")
        target = {'image_id': img_id, 'annotations': target_anno}

        image, target = self.prepare(image, target, remap=self.remap_mscoco_category, category2label=self.mscoco_category2label)

        if 'boxes' not in target or target['boxes'].shape[0] == 0:
            return image, None

        target['boxes'] = BoundingBoxes(target['boxes'], format="XYXY", canvas_size=image.size[::-1])
        if 'masks' in target:
            target['masks'] = Mask(target['masks'])

        target['idx'] = torch.tensor([idx])
        return image, target

def analyze_annotations(coco, save_path=None, split_name="Dataset"):
    class_counts = Counter(ann['category_id'] for ann in coco.dataset['annotations'])
    id_to_name = {cat['id']: cat['name'] for cat in coco.dataset['categories']}
    class_counts_named = {id_to_name.get(k, k): v for k, v in class_counts.items()}
    
    import pandas as pd
    data = pd.DataFrame(class_counts_named.items(), columns=["Clase", "Número de Instancias"])
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Número de Instancias", y="Clase", data=data, palette="viridis")
    plt.title(f"Distribución de Instancias por Clase en {split_name}", fontsize=16)
    plt.xlabel("Número de Instancias", fontsize=14)
    plt.ylabel("Clases", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()
    return class_counts_named