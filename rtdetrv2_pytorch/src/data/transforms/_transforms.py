""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import PIL

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms.v2 import functional as F

from typing import Any, Dict

from ...core import register

@register()
class RandomPhotometricDistort(nn.Module):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__()
        self.transform = T.ColorJitter(*args, **kwargs)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            img = self.transform(img)
        return img, target

@register()
class RandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.2):
        super().__init__()
        self.transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            img = self.transform(img)
        return img, target

@register()
class RandomNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01, p=0.2):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor) and img.is_floating_point():
                noise = torch.normal(mean=self.mean, std=self.std, size=img.shape, device=img.device)
                img = img + noise
        return img, target

@register()
class RandomRotation(nn.Module):
    def __init__(self, degrees, p=0.5, **kwargs):
        super().__init__()
        self.transform = T.RandomRotation(degrees=degrees, **kwargs)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return self.transform(img, target)
        return img, target

@register()
class RandomZoomOut(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.RandomZoomOut(*args, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class RandomIoUCrop(nn.Module):
    def __init__(self, *args, p=0.8, **kwargs):
        super().__init__()
        self.transform = T.RandomIoUCrop(*args, **kwargs)
        self.p = p
    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return self.transform(img, target)
        return img, target

@register()
class RandomHorizontalFlip(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.RandomHorizontalFlip(*args, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class Resize(nn.Module):
    def __init__(self, size, **kwargs):
        super().__init__()
        self.transform = T.Resize(size=size, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class SanitizeBoundingBoxes(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.SanitizeBoundingBoxes(*args, **kwargs)
    def forward(self, img, target):
        return self.transform(img, target)

@register()
class ConvertBoxes(nn.Module):
    def __init__(self, fmt='cxcywh', normalize=True):
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def forward(self, img, target):
        # Primero, nos aseguramos de que las cajas existan y sean del tipo correcto.
        if 'boxes' in target and isinstance(target['boxes'], BoundingBoxes):
            boxes = target['boxes']
            
            # 1. Convertir formato
            boxes = F.convert_bounding_box_format(boxes, new_format=self.fmt)
            
            # 2. Normalizar manualmente para evitar la llamada ambigua
            if self.normalize:
                # Obtenemos el tamaño del lienzo desde el propio objeto BoundingBoxes
                h, w = boxes.canvas_size
                if self.fmt == 'cxcywh':
                    # Normalizar cx, x, w
                    boxes[:, 0] /= w
                    boxes[:, 2] /= w
                    # Normalizar cy, y, h
                    boxes[:, 1] /= h
                    boxes[:, 3] /= h
                else: # asume xyxy
                    boxes[:, 0::2] /= w
                    boxes[:, 1::2] /= h

            target['boxes'] = boxes
        return img, target
# ------------------------------------------

@register()
class ToImage(T.ToImage):
    pass
        
@register()
class ToDtype(T.ToDtype):
    def __init__(self, dtype, scale=False):
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        super().__init__(dtype=torch_dtype, scale=scale)