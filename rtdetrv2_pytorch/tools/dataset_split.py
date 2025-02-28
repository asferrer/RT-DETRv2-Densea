#!/usr/bin/env python3
import json
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def plot_class_distribution(split_name, coco_subset, cat_id_to_name, categories, output_dir):
    """
    Calcula y guarda un gráfico de barras con el número total de instancias 
    (bounding boxes) por clase para el split dado.
    
    Cada barra muestra, sobre ella, el número exacto de instancias para esa clase.
    El gráfico se almacena en el directorio de salida con un nombre que indica el split.
    """
    # Inicializar el conteo por cada categoría (utilizando el nombre de la clase)
    counts = {cat["name"]: 0 for cat in categories}
    for ann in coco_subset["annotations"]:
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, "Desconocido")
        counts[cat_name] += 1

    names = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color='skyblue')
    ax.set_title(f"Distribución de instancias por clase en {split_name}")
    ax.set_xlabel("Clases")
    ax.set_ylabel("Número de instancias")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Guardar el gráfico en el directorio de output
    plot_filename = os.path.join(output_dir, f"{split_name.lower()}_distribution.png")
    plt.savefig(plot_filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Script para dividir un dataset COCO (detección de objetos) en train, validación y test, "
                    "filtrando clases con pocas instancias y balanceando automáticamente el volumen de cada clase. "
                    "Se incluyen imágenes sin anotaciones para robustecer el entrenamiento."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta al archivo COCO JSON (con imágenes, anotaciones y categorías).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio de salida donde se guardarán train.json, val.json, test.json y las gráficas.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Fracción de imágenes para el conjunto de test (default: 0.15).")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fracción de imágenes para el conjunto de validación (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para aleatorización (default: 42).")
    parser.add_argument("--min_instances", type=int, default=5,
                        help="Número mínimo de instancias (bounding boxes) requeridas por clase para ser considerada (default: 5).")
    args = parser.parse_args()

    # Cargar el archivo COCO JSON
    with open(args.input, 'r') as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # --- Filtrado por mínimo de instancias ---
    # Contar el número total de bounding boxes por cada clase
    instance_counts = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        instance_counts[cat_id] = instance_counts.get(cat_id, 0) + 1

    # Determinar las clases permitidas según el mínimo requerido
    allowed_class_ids = {cat_id for cat_id, count in instance_counts.items() if count >= args.min_instances}
    
    removed_classes = [cat for cat in categories if cat["id"] not in allowed_class_ids]
    if removed_classes:
        print("Se eliminarán las siguientes clases por no cumplir con el mínimo de instancias:")
        for cat in removed_classes:
            print(f" - {cat['name']} (id: {cat['id']}, instancias: {instance_counts.get(cat['id'], 0)})")
    
    # Actualizar la lista de categorías permitidas
    allowed_categories = [cat for cat in categories if cat["id"] in allowed_class_ids]

    # Filtrar anotaciones: conservar solo aquellas de las clases permitidas
    annotations_filtered = [ann for ann in annotations if ann["category_id"] in allowed_class_ids]

    # Incluir todas las imágenes, incluso aquellas sin anotaciones, para robustecer el entrenamiento
    images_filtered = images

    print(f"Total imágenes: {len(images_filtered)}")
    print(f"Total clases permitidas: {len(allowed_categories)}")

    # --- Preparar la matriz multi-etiqueta ---
    # Se construye una matriz binaria en la que cada fila representa una imagen y cada columna una clase permitida
    image_ids = [img["id"] for img in images_filtered]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    n_images = len(images_filtered)

    allowed_cat_ids = [cat["id"] for cat in allowed_categories]
    allowed_cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(allowed_cat_ids)}
    n_categories = len(allowed_categories)

    Y = np.zeros((n_images, n_categories), dtype=int)
    for ann in annotations_filtered:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in id_to_index and cat_id in allowed_cat_id_to_index:
            i = id_to_index[img_id]
            j = allowed_cat_id_to_index[cat_id]
            Y[i, j] = 1

    # --- Incluir imágenes sin anotaciones ---
    # Se añade una columna dummy para identificar aquellas imágenes que no tienen ninguna anotación (en las clases permitidas)
    Y_dummy = np.concatenate([Y, np.zeros((n_images, 1), dtype=int)], axis=1)
    for i in range(n_images):
        if np.sum(Y[i]) == 0:
            Y_dummy[i, -1] = 1

    # --- Realizar el split multi-etiqueta ---
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    X_dummy = np.zeros((n_images, 1))
    train_val_idx, test_idx = next(msss.split(X_dummy, Y_dummy))

    # Separar train y validación del conjunto train_val
    val_fraction = args.val_size / (1 - args.test_size)
    X_dummy_tv = np.zeros((len(train_val_idx), 1))
    Y_tv = Y_dummy[train_val_idx]
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=args.seed)
    train_idx_rel, val_idx_rel = next(msss_val.split(X_dummy_tv, Y_tv))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    # Convertir índices a conjuntos de image IDs para cada split
    train_ids = {image_ids[i] for i in train_idx}
    val_ids = {image_ids[i] for i in val_idx}
    test_ids = {image_ids[i] for i in test_idx}

    # Función auxiliar para construir un subset COCO a partir de un set de image IDs
    def build_coco_subset(image_id_set, base_coco):
        subset = {}
        for key in base_coco:
            if key == "images":
                subset["images"] = [img for img in base_coco["images"] if img["id"] in image_id_set]
            elif key == "annotations":
                subset["annotations"] = [ann for ann in base_coco["annotations"] if ann["image_id"] in image_id_set]
            else:
                subset[key] = base_coco[key]
        return subset

    # Actualizar el diccionario COCO filtrado
    coco_filtered = dict(coco)
    coco_filtered["images"] = images_filtered
    coco_filtered["annotations"] = annotations_filtered
    coco_filtered["categories"] = allowed_categories

    # Construir splits
    coco_train = build_coco_subset(train_ids, coco_filtered)
    coco_val = build_coco_subset(val_ids, coco_filtered)
    coco_test = build_coco_subset(test_ids, coco_filtered)

    # Guardar los archivos JSON de cada split
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(coco_train, f, indent=4)
    with open(os.path.join(args.output_dir, "val.json"), "w") as f:
        json.dump(coco_val, f, indent=4)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(coco_test, f, indent=4)

    print("División completada:")
    print(f"  Train: {len(train_ids)} imágenes")
    print(f"  Validación: {len(val_ids)} imágenes")
    print(f"  Test: {len(test_ids)} imágenes")

    # Crear diccionario para mapear category_id a nombre (para la visualización)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in allowed_categories}

    # Guardar las gráficas de distribución de instancias por clase para cada split
    plot_class_distribution("Train", coco_train, cat_id_to_name, allowed_categories, args.output_dir)
    plot_class_distribution("Validación", coco_val, cat_id_to_name, allowed_categories, args.output_dir)
    plot_class_distribution("Test", coco_test, cat_id_to_name, allowed_categories, args.output_dir)

if __name__ == "__main__":
    main()
