import os
import random
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm

# Parámetros de entrada
annotation_file = '/app/RT-DETR/dataset/annotations_combined_fixed.json'  # Ruta al archivo JSON de anotaciones
# Lista de directorios donde se buscarán las imágenes
images_dirs = [
    '/app/RT-DETR/dataset/cleansea_dataset/CocoFormatDataset/train_coco/JPEGImages',
    '/app/RT-DETR/dataset/cleansea_dataset/CocoFormatDataset/test_coco/JPEGImages',
    '/app/RT-DETR/dataset/Neural_Ocean/train',
    '/app/RT-DETR/dataset/Neural_Ocean/valid',
    '/app/RT-DETR/dataset/Neural_Ocean/test',
    '/app/RT-DETR/dataset/Ocean_garbage/train',
    '/app/RT-DETR/dataset/Ocean_garbage/valid',
    '/app/RT-DETR/dataset/Ocean_garbage/test',
    '/app/RT-DETR/dataset/synthetic_dataset/coco/JPEGImages',
    # Agrega más directorios si es necesario
]
output_dir = 'review_dataset'  # Directorio de salida para guardar las imágenes anotadas

# Crear directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar anotaciones COCO
coco = COCO(annotation_file)

# Crear un diccionario para buscar metadata de imágenes por su nombre
imagenes_coco = {img['file_name']: img for img in coco.loadImgs(coco.getImgIds())}

# Diccionario para mapear id de categoría a nombre
categories = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

# Función para buscar la ruta de una imagen en los directorios proporcionados
def buscar_imagen(nombre_archivo):
    for dir_path in images_dirs:
        ruta = os.path.join(dir_path, nombre_archivo)
        if os.path.exists(ruta):
            return ruta
    return None

# Función auxiliar para identificar archivos de imagen por su extensión
def es_imagen(archivo):
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return os.path.splitext(archivo)[1].lower() in extensiones

# Procesar cada directorio
for dir_path in tqdm(images_dirs, desc="Procesando directorios"):
    # Listar archivos de imagen en el directorio
    archivos = [os.path.join(dir_path,archivo) for archivo in os.listdir(dir_path) if es_imagen(archivo)]
    # Seleccionar 5 imágenes aleatoriamente (o todas si hay menos)
    imagenes_seleccionadas = random.sample(archivos, min(5, len(archivos)))
    
    for nombre_archivo in imagenes_seleccionadas:
        # Buscar la imagen usando la función buscar_imagen
        ruta_imagen = buscar_imagen(nombre_archivo)
        if ruta_imagen is None:
            print(f"Advertencia: No se encontró la imagen {nombre_archivo}")
            continue

        # Verificar que la imagen esté en el dataset de anotaciones COCO
        if nombre_archivo not in imagenes_coco:
            print(f"Advertencia: La imagen {nombre_archivo} no se encuentra en las anotaciones COCO")
            continue

        # Obtener metadata de la imagen y su id
        meta = imagenes_coco[nombre_archivo]
        img_id = meta['id']

        # Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Advertencia: No se pudo cargar la imagen {ruta_imagen}")
            continue

        # Obtener todas las anotaciones de la imagen
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        if not anns:
            print(f"Advertencia: La imagen {nombre_archivo} no tiene anotaciones.")
            continue

        # Dibujar las anotaciones (bounding boxes y nombre de la categoría)
        for ann in anns:
            x, y, w, h = ann['bbox']
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(imagen, pt1, pt2, (0, 255, 0), 2)
            cat_name = cat_id_to_name.get(ann['category_id'], 'N/A')
            cv2.putText(imagen, cat_name, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Obtener el conjunto de categorías presentes en la imagen
        categorias_presentes = set()
        for ann in anns:
            cat_name = cat_id_to_name.get(ann['category_id'], 'N/A')
            categorias_presentes.add(cat_name)

        # Guardar la imagen anotada en cada carpeta de la clase correspondiente
        for cat in categorias_presentes:
            salida_dir_cat = os.path.join(output_dir, cat)
            if not os.path.exists(salida_dir_cat):
                os.makedirs(salida_dir_cat)
            # Se utiliza el nombre base de la imagen para la salida
            ruta_salida = os.path.join(salida_dir_cat, os.path.basename(nombre_archivo))
            cv2.imwrite(ruta_salida, imagen)