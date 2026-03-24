"""
Auto-anotación de imágenes de rumas usando modelo YOLO entrenado.
Genera archivos .txt en formato YOLO para subir a Roboflow.

Uso:
    python auto_annotate.py

Salida:
    rumas-18-anotado/
    ├── classes.txt
    ├── imagen.jpg
    ├── imagen.txt
    └── ...
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Modelo entrenado para detección de rumas
MODEL_PATH = r"D:\SUBASTA\minera alma\minera\MinaRumas\models\deteccion_rumas\deteccion_ruma.pt"

# Directorio con las imágenes originales
IMAGES_DIR = Path(r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\rumas-18")

# Directorio de salida
OUTPUT_DIR = Path(r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\rumas-18-anotado-95")

# Umbral de confianza (más bajo = más detecciones para corregir en Roboflow)
# Cambiado a 0.05 para forzar al modelo a detectar el 95%+ de las rumas
CONFIDENCE_THRESHOLD = 0.05

# Extensiones de imagen válidas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Clases (del modelo entrenado)
CLASSES = ["ruma"]


def main():
    print("=" * 60)
    print("  AUTO-ANOTACIÓN DE RUMAS PARA ROBOFLOW")
    print("=" * 60)

    # Verificar que el modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: No se encontró el modelo en:\n   {MODEL_PATH}")
        return

    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Escribir classes.txt
    classes_file = OUTPUT_DIR / "classes.txt"
    with open(classes_file, "w") as f:
        for cls in CLASSES:
            f.write(f"{cls}\n")
    print(f"\n✅ classes.txt creado con clases: {CLASSES}")

    # Obtener lista de imágenes
    image_files = sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"\n❌ ERROR: No se encontraron imágenes en:\n   {IMAGES_DIR}")
        return

    print(f"\n📷 Imágenes encontradas: {len(image_files)}")
    print(f"📁 Salida: {OUTPUT_DIR}")
    print(f"🎯 Umbral de confianza: {CONFIDENCE_THRESHOLD}")
    print(f"\n🔄 Cargando modelo...")

    # Cargar modelo
    model = YOLO(MODEL_PATH)
    print(f"✅ Modelo cargado: {MODEL_PATH}")
    print(f"   Clases del modelo: {model.names}")

    # Estadísticas
    total_images = len(image_files)
    total_detections = 0
    images_with_detections = 0
    images_without_detections = 0

    print(f"\n🚀 Procesando {total_images} imágenes...\n")

    for i, img_path in enumerate(image_files, 1):
        # Ejecutar inferencia
        results = model(str(img_path), conf=CONFIDENCE_THRESHOLD, imgsz=1280, verbose=False)
        result = results[0]

        # Obtener dimensiones de la imagen
        img_h, img_w = result.orig_shape

        # Copiar imagen al directorio de salida
        dst_img = OUTPUT_DIR / img_path.name
        shutil.copy2(img_path, dst_img)

        # Generar archivo .txt con anotaciones YOLO
        txt_name = img_path.stem + ".txt"
        txt_path = OUTPUT_DIR / txt_name

        boxes = result.boxes
        num_detections = len(boxes)
        total_detections += num_detections

        if num_detections > 0:
            images_with_detections += 1
        else:
            images_without_detections += 1

        with open(txt_path, "w") as f:
            for box in boxes:
                # Obtener coordenadas xyxy
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Convertir a formato YOLO (normalizado)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                # Todas las detecciones se mapean a clase 0 (ruma)
                # ya que el modelo fue entrenado específicamente para rumas
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Progreso
        status = f"✅ {num_detections} det." if num_detections > 0 else "⚠️  0 det."
        if i % 10 == 0 or i == total_images:
            print(f"  [{i:3d}/{total_images}] {img_path.name} → {status}")

    # Resumen final
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  📷 Total imágenes procesadas:  {total_images}")
    print(f"  🎯 Total detecciones:          {total_detections}")
    print(f"  ✅ Imágenes con detecciones:    {images_with_detections}")
    print(f"  ⚠️  Imágenes sin detecciones:   {images_without_detections}")
    print(f"  📊 Promedio det/imagen:         {total_detections / total_images:.1f}")
    print(f"\n  📁 Archivos generados en:")
    print(f"     {OUTPUT_DIR}")
    print(f"\n  📤 Listo para subir a Roboflow!")
    print("=" * 60)


if __name__ == "__main__":
    main()
