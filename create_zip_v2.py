"""
Crear ZIP v2 para Roboflow, incluyendo data.yaml
"""
import zipfile
from pathlib import Path

INPUT_DIR = Path(r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\rumas-18-anotado")
ZIP_PATH = Path(r"D:\SUBASTA\deteccion-rumas-belen-para-roboflow\rumas-18-v2.zip")

yaml_content = """train: train/images
val: train/images

nc: 1
names: ['ruma']
"""

print("Creando ZIP v2 para Roboflow...")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    # Agregar data.yaml a la raiz
    zf.writestr("data.yaml", yaml_content)
    
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix.lower() == ".jpg":
            zf.write(f, f"train/images/{f.name}")
        elif f.suffix.lower() == ".txt" and f.name != "classes.txt":
            content = f.read_text().replace("\r\n", "\n").strip() + "\n"
            zf.writestr(f"train/labels/{f.name}", content)

print(f"ZIP creado: {ZIP_PATH}")
