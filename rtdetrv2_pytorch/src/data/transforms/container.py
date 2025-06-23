""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

from ._transforms import EmptyTransform
from ...core import register, GLOBAL_CONFIG

print("Registered transformations:", list(GLOBAL_CONFIG.keys()))

@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    
                    if name in GLOBAL_CONFIG:
                        module_info = GLOBAL_CONFIG[name]
                        transform_class = getattr(module_info['_pymodule'], module_info['_name'])
                    elif hasattr(T, name):
                        transform_class = getattr(T, name)
                    else:
                        raise NameError(f"Transformation '{name}' not found in GLOBAL_CONFIG or torchvision.transforms.v2")

                    if 'dtype' in op and isinstance(op['dtype'], str):
                        if 'torch.' in op['dtype']:
                            op['dtype'] = getattr(torch, op['dtype'].split('.')[-1])
                        else:
                            op['dtype'] = getattr(torch, op['dtype'])
                    
                    transform = transform_class(**op)
                    transforms.append(transform)
                    op['type'] = name
                    
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError('La operación de transformación no es válida')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)

        if policy is None:
            policy = {'name': 'default'}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]
    
    def default_forward(self, *inputs: Any) -> Any:
        img, target = inputs[:2]
        dataset_obj = inputs[2] if len(inputs) > 2 else None

        for transform in self.transforms:
            transform_class_name = type(transform).__name__
            
            if transform_class_name in ['ConvertPILImage', 'RandomPhotometricDistort', 'RandomGaussianBlur', 'RandomNoise']:
                img = transform(img)
            else:
                img, target = transform(img, target)
        
        data_to_transform = (img, target)
        if dataset_obj is not None:
            return *data_to_transform, dataset_obj
        else:
            return data_to_transform

    def stop_epoch_forward(self, *inputs: Any):
        img, target = inputs[:2]
        dataset = inputs[2]
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        for transform in self.transforms:
            transform_class_name = type(transform).__name__
            if transform_class_name in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                if transform_class_name in ['ConvertPILImage', 'RandomPhotometricDistort', 'RandomGaussianBlur', 'RandomNoise']:
                    img = transform(img)
                else:
                    img, target = transform(img, target)
        
        data_to_transform = (img, target)
        return *data_to_transform, dataset


    def stop_sample_forward(self, *inputs: Any):
        img, target = inputs[:2]
        dataset = inputs[2]
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            transform_class_name = type(transform).__name__
            if transform_class_name in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                if transform_class_name in ['ConvertPILImage', 'RandomPhotometricDistort', 'RandomGaussianBlur', 'RandomNoise']:
                    img = transform(img)
                else:
                    img, target = transform(img, target)

        self.global_samples += 1
        
        data_to_transform = (img, target)
        return *data_to_transform, dataset