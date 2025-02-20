#!/usr/bin/env python
"""
Script para evaluar el modelo configurado y, opcionalmente, visualizar (dibujar)
las inferencias (cajas, etiquetas y scores) sobre las imágenes.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

# Se añade el directorio padre para poder importar YAMLConfig y otros módulos internos.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

def draw_inferences(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.6, save_path: str = None):
    """
    Dibuja las inferencias sobre una imagen (objetos PIL). Para cada detección con score mayor
    al umbral se dibuja un rectángulo en rojo y se escribe la etiqueta junto al score en azul.
    
    Args:
        image (PIL.Image): Imagen original sobre la que dibujar.
        labels (torch.Tensor): Tensor con las etiquetas de detección.
        boxes (torch.Tensor): Tensor con las coordenadas de las cajas [xmin, ymin, xmax, ymax].
        scores (torch.Tensor): Tensor con las puntuaciones de detección.
        threshold (float): Umbral para filtrar detecciones.
        save_path (str): Ruta para guardar la imagen con inferencias. Si es None, no se guarda.
    """
    draw = ImageDraw.Draw(image)
    
    # Se filtran las detecciones que superan el umbral
    valid_indices = scores > threshold
    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    
    for j, box in enumerate(valid_boxes):
        # Se transforma el tensor a lista de coordenadas
        b = list(box)
        # Dibujar el rectángulo (se puede ajustar el grosor si es necesario)
        draw.rectangle(b, outline='red', width=2)
        # Dibujar la etiqueta y el score sobre la imagen
        text = f"{valid_labels[j].item()} {valid_scores[j].item():.2f}"
        draw.text((b[0], b[1]), text, fill='blue')
        print(text)
    
    if save_path is not None:
        image.save(save_path)
        print(f"Imagen guardada en: {save_path}")

class ModelWrapper(nn.Module):
    """
    Esta clase envuelve al modelo y al postprocesador. Se utiliza el método deploy()
    de cada componente para obtener la versión lista para inferencia.
    """
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs  # Se espera una tupla: (labels, boxes, scores)

def evaluate_on_image(args, cfg, device):
    """
    Realiza inferencia sobre una única imagen.
    
    Args:
        args: Argumentos parseados (de argparse).
        cfg: Configuración cargada (YAMLConfig).
        device: Dispositivo de cómputo (cpu o cuda).
    """
    # Se carga la imagen y se conserva su tamaño original.
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    
    # Se define la transformación: se reescala a 640x640 y se convierte a tensor.
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_tensor = transform(im_pil)[None].to(device)
    
    # Se construye el modelo para inferencia.
    model = ModelWrapper(cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(im_tensor, orig_size)
        # Se espera que outputs sea una tupla: (labels, boxes, scores)
        labels, boxes, scores = outputs
    
    # Si se solicita la visualización, se dibujan las inferencias sobre la imagen
    if args.draw:
        # Se crea una copia para no modificar la imagen original.
        im_copy = im_pil.copy()
        out_path = os.path.join(args.output_dir, "result_single.jpg")
        draw_inferences(im_copy, labels[0], boxes[0], scores[0], threshold=0.6, save_path=out_path)
    else:
        # Solo se muestran los resultados por consola.
        for j in range(len(labels[0])):
            if scores[0][j] > 0.6:
                print(f"Label: {labels[0][j].item()}, Score: {scores[0][j].item():.2f}")

def evaluate_on_dataset(args, cfg, device):
    """
    Realiza la evaluación sobre el conjunto de validación (dataset).
    Se itera sobre el dataloader definido en la configuración, se realizan las inferencias
    y se actualiza el evaluador (por ejemplo, CocoEvaluator) si está disponible.
    
    Además, si se solicita la visualización, para cada imagen se intentará cargar la imagen
    original (usando la información del batch) y se guardará una copia con las inferencias.
    
    Args:
        args: Argumentos parseados.
        cfg: Configuración cargada (YAMLConfig).
        device: Dispositivo de cómputo.
    """
    # Se construye el modelo para evaluación.
    model = ModelWrapper(cfg).to(device)
    model.eval()
    
    # Se obtiene el dataloader de validación a partir de la configuración.
    dataloader = cfg.val_dataloader
    evaluator = None
    if hasattr(cfg, 'evaluator'):
        evaluator = cfg.evaluator
    
    if args.draw:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Se itera sobre el dataloader
    for idx, batch in enumerate(dataloader):
        # Se asume que el batch es un diccionario con al menos la llave 'img'.
        # Opcionalmente, se esperan las llaves 'targets' (ground truth) y 'file_name' (ruta original de la imagen).
        if isinstance(batch, dict):
            images = batch['img']
            targets = batch.get('targets', None)
            # Se verifica si el batch contiene el tamaño original de la imagen.
            if 'orig_size' in batch:
                orig_sizes = batch['orig_size']
            else:
                # En caso contrario, se asume un tamaño fijo (aunque lo ideal es contar con la información original).
                B = images.shape[0]
                orig_sizes = torch.tensor([[640, 640]] * B).to(device)
            file_names = batch.get('file_name', None)
        elif isinstance(batch, (list, tuple)):
            # Se asume que el batch es (images, targets)
            images, targets = batch
            B = images.shape[0]
            orig_sizes = torch.tensor([[640, 640]] * B).to(device)
            file_names = None
        else:
            raise ValueError("Formato de batch no reconocido.")
        
        images = images.to(device)
        
        with torch.no_grad():
            # Nos aseguramos de que orig_sizes tenga forma [batch_size, 2]
            if orig_sizes.dim() == 1:
                orig_sizes = orig_sizes.unsqueeze(0)
            outputs = model(images, orig_sizes)
            # Se espera que outputs sea una tupla: (labels, boxes, scores)
        labels_batch, boxes_batch, scores_batch = outputs
        
        # Si se dispone de un evaluador y de las anotaciones ground truth, se actualiza
        if evaluator is not None and targets is not None:
            # Se itera sobre cada imagen del batch y se construye la predicción en el formato esperado.
            batch_predictions = {}
            for i in range(len(labels_batch)):
                # Extraer image_id del ground truth si está presente; de lo contrario usar el índice.
                if "image_id" in targets[i]:
                    image_id = int(targets[i]["image_id"].item()) if isinstance(targets[i]["image_id"], torch.Tensor) else int(targets[i]["image_id"])
                else:
                    image_id = i
                
                batch_predictions[image_id] = {
                    "boxes": boxes_batch[i].cpu(),
                    "labels": labels_batch[i].cpu(),
                    "scores": scores_batch[i].cpu(),
                }
            evaluator.update(batch_predictions)
        else:
            print("No se dispone de evaluador o anotaciones ground truth en este batch.")
        
        # Si se solicita la visualización y se cuenta con la ruta original de la imagen, se dibujan las inferencias.
        if args.draw and file_names is not None:
            for i, fname in enumerate(file_names):
                try:
                    im = Image.open(fname).convert('RGB')
                except Exception as e:
                    print(f"Error al cargar la imagen {fname}: {e}")
                    continue
                out_fname = os.path.join(args.output_dir, f"result_{os.path.basename(fname)}")
                draw_inferences(im, labels_batch[i], boxes_batch[i], scores_batch[i], threshold=0.6, save_path=out_fname)
        elif args.draw:
            print("No se encontraron nombres de archivo en el batch para dibujar las inferencias. Se dibujarán usando las imágenes del batch.")
            # Asumir que las imágenes ya se tienen en el batch:
            for i in range(images.shape[0]):
                # Convertir la imagen tensor a PIL si es necesario (suponiendo que el tensor está en [C,H,W])
                im = T.ToPILImage()(images[i].cpu())
                out_fname = os.path.join(args.output_dir, f"result_batch_{idx}_{i}.jpg")
                draw_inferences(im, labels_batch[i], boxes_batch[i], scores_batch[i], threshold=0.6, save_path=out_fname)

    
    # Una vez evaluado todo el dataset, se calculan y muestran las métricas (si el evaluador está definido)
    if evaluator is not None:
        # Sincronizar (si es necesario en tu entorno distribuido)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
    else:
        print("Evaluación completada. No se dispone de un evaluador para calcular métricas.")

def main():
    parser = argparse.ArgumentParser(
        description="Script de evaluación del modelo con opción de visualización de inferencias."
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
                        help="Si se especifica, dibuja las inferencias sobre las imágenes.")
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help="Directorio para guardar las imágenes con inferencias dibujadas.")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Se carga la configuración desde el YAML (se pueden pasar parámetros adicionales si fuera necesario)
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Se carga el checkpoint y se extrae el state_dict del modelo.
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    # Se carga el state_dict en el modelo (el modelo se creará al acceder a cfg.model)
    model_instance = cfg.model
    model_instance.load_state_dict(state)
    
    # Se muestra en consola que se usará el directorio para guardar resultados (en caso de dibujo)
    if args.draw:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Las imágenes con inferencias se guardarán en: {args.output_dir}")
    
    # Modo de evaluación: si se proporciona una imagen individual se evalúa sobre ella;
    # de lo contrario se evalúa sobre el dataset de validación.
    if args.im_file is not None:
        print("Evaluación sobre imagen individual...")
        evaluate_on_image(args, cfg, device)
    else:
        print("Evaluación sobre dataset de validación...")
        evaluate_on_dataset(args, cfg, device)

if __name__ == '__main__':
    main()
