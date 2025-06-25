""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from typing import Any

from ...core import register, GLOBAL_CONFIG


@register()
class Compose(nn.Module):
    def __init__(self, ops, policy=None):
        super().__init__()
        
        transforms = []
        if ops:
            for op in ops:
                op_copy = op.copy()
                name = op_copy.pop('type')
                
                if name in GLOBAL_CONFIG:
                    module_info = GLOBAL_CONFIG[name]
                    transform_class = getattr(module_info['_pymodule'], module_info['_name'])
                elif hasattr(T, name):
                    transform_class = getattr(T, name)
                else:
                    raise NameError(f"Transformación '{name}' no encontrada.")

                if 'dtype' in op_copy and isinstance(op_copy['dtype'], str):
                    op_copy['dtype'] = getattr(torch, op_copy['dtype'].split('.')[-1])

                transforms.append(transform_class(**op_copy))
        
        self.transforms = nn.ModuleList(transforms)
        self.policy = policy

    def forward(self, *inputs: Any) -> Any:
        img, target = inputs[0], inputs[1]

        # print("\n" + "="*80)
        # print("--- INICIO PIPELINE DE TRANSFORMACIONES ---")
        # print(f"Entrada inicial -> img: {type(img)}, target: {type(target)}")
        # print("="*80)

        for i, t in enumerate(self.transforms):
            transform_name = type(t).__name__
            # print(f"\n[Paso {i+1}/{len(self.transforms)}] Aplicando '{transform_name}'...")
            # print(f"  - Antes  -> img: {type(img)}, target: {type(target)}")
            
            try:
                img, target = t(img, target)
                # print(f"  - Después -> img: {type(img)}, target: {type(target)}")
                # print(f"  -> OK: '{transform_name}' aplicada con éxito.")
            except Exception as e:
                # print("\n" + "!"*80)
                # print(f"!!! ERROR al aplicar la transformación '{transform_name}' !!!")
                # print(f"!!! Tipo de error: {type(e).__name__} - {e} !!!")
                # print("!"*80)
                raise e

        # print("\n" + "="*80)
        # print("--- FIN PIPELINE DE TRANSFORMACIONES ---")
        # print("="*80 + "\n")

        if len(inputs) > 2:
            return img, target, inputs[2]
        else:
            return img, target