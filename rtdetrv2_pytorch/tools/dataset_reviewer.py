import os
import random
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm

# Parámetros de entrada
annotation_file = '/app/RT-DETR/dataset/annotations_cleansea+synthetic_v2.json'  # Ruta al archivo JSON de anotaciones
# Lista de directorios donde se buscarán las imágenes
images_dirs = [
    '/app/RT-DETR/dataset/cleansea_dataset/CocoFormatDataset/train_coco/JPEGImages',
    '/app/RT-DETR/dataset/synthetic_dataset_v2/images',
]
output_dir = 'review_dataset_synthetic'  # Directorio de salida para guardar las imágenes anotadas

# Crear directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar anotaciones COCO
coco = COCO(annotation_file)

# Crear un diccionario para buscar metadata de imágenes por su nombre (solo el nombre del archivo)
imagenes_coco = {img['file_name']: img for img in coco.loadImgs(coco.getImgIds())}

# Diccionario para mapear id de categoría a nombre
categories = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

# Función auxiliar para identificar archivos de imagen por su extensión
def es_imagen(archivo):
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return os.path.splitext(archivo)[1].lower() in extensiones

# Procesar cada directorio
for dir_path in tqdm(images_dirs, desc="Procesando directorios"):
    # Diccionario para agrupar imágenes por categoría: {nombre_categoria: [archivo1, archivo2, ...]}
    imagenes_por_categoria = {}
    
    # Listar archivos de imagen en el directorio
    for archivo in os.listdir(dir_path):
        if not es_imagen(archivo):
            continue
        
        # Verificar que la imagen exista en las anotaciones COCO
        if archivo not in imagenes_coco:
            print(f"Advertencia: La imagen {archivo} no se encuentra en las anotaciones COCO")
            continue
        
        # Obtener metadata y anotaciones de la imagen
        meta = imagenes_coco[archivo]
        img_id = meta['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        if not anns:
            print(f"Advertencia: La imagen {archivo} no tiene anotaciones.")
            continue
        
        # Determinar las categorías presentes en la imagen
        categorias_presentes = set()
        for ann in anns:
            cat_name = cat_id_to_name.get(ann['category_id'], 'N/A')
            categorias_presentes.add(cat_name)
        
        # Agregar la imagen a cada categoría en la que aparece
        for cat in categorias_presentes:
            if cat not in imagenes_por_categoria:
                imagenes_por_categoria[cat] = []
            if archivo not in imagenes_por_categoria[cat]:
                imagenes_por_categoria[cat].append(archivo)
    
    # Para cada categoría, se seleccionan 5 imágenes aleatorias y se procesan
    for cat, lista_imagenes in imagenes_por_categoria.items():
        muestras = random.sample(lista_imagenes, min(5, len(lista_imagenes)))
        for archivo in muestras:
            ruta_imagen = os.path.join(dir_path, archivo)
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"Advertencia: No se pudo cargar la imagen {ruta_imagen}")
                continue
            
            # Obtener las anotaciones de la imagen y dibujarlas
            meta = imagenes_coco[archivo]
            img_id = meta['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                x, y, w, h = ann['bbox']
                pt1 = (int(x), int(y))
                pt2 = (int(x + w), int(y + h))
                cv2.rectangle(imagen, pt1, pt2, (0, 255, 0), 2)
                cat_name_ann = cat_id_to_name.get(ann['category_id'], 'N/A')
                cv2.putText(imagen, cat_name_ann, (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Guardar la imagen anotada en la carpeta correspondiente a la categoría
            salida_dir_cat = os.path.join(output_dir, cat)
            if not os.path.exists(salida_dir_cat):
                os.makedirs(salida_dir_cat)
            ruta_salida = os.path.join(salida_dir_cat, archivo)
            cv2.imwrite(ruta_salida, imagen)
