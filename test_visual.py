"""
Visualización de prueba: dibuja los bounding boxes sobre una imagen para verificar
que las anotaciones YOLO son correctas.
"""
import cv2
import os

# Imagen y anotación de prueba
IMG_PATH = r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\rumas-18-anotado\belen_titan_01_2026-03-17_162846.jpg"
TXT_PATH = r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\rumas-18-anotado\belen_titan_01_2026-03-17_162846.txt"
OUTPUT_PATH = r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\test_visual.jpg"

# Leer imagen
img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]
print(f"Imagen: {w}x{h}")

# Leer anotaciones
count = 0
with open(TXT_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "ruma", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        count += 1

print(f"Bounding boxes dibujados: {count}")
cv2.imwrite(OUTPUT_PATH, img)
print(f"Imagen guardada en: {OUTPUT_PATH}")
