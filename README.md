# belen-rumas

Scripts para autoanotar imagenes de rumas con YOLO y generar ZIPs en lotes de 50 listos para importar a Roboflow.

## Incluye

- `create_zip.py`: toma una o varias carpetas de imagenes, autoanota con YOLO y crea un ZIP por lote.
- `run_jetson.sh`: wrapper para correr el flujo en una Jetson.
- `models/model_detection.pt`: modelo listo para usar.

## Estructura de salida

Cada lote se genera como:

```text
batch_001/
  classes.txt
  data.yaml
  train/
    images/
    labels/
batch_001.zip
```

## Preparar la Jetson

1. Clona el repo:

```bash
git clone https://github.com/Luis-gath/belen-rumas.git
cd belen-rumas
```

2. Instala dependencias Python.

Nota: en Jetson, `torch` debe venir de la instalacion compatible con tu JetPack. Este repo solo fija `ultralytics`.

```bash
python3 -m pip install -r requirements.txt
chmod +x run_jetson.sh
```

## Ejecutar

Ejemplo con las cuatro fechas:

```bash
./run_jetson.sh \
  /ruta/capture_rumas/2026-03-17 \
  /ruta/capture_rumas/2026-03-18 \
  /ruta/capture_rumas/2026-03-19 \
  /ruta/capture_rumas/2026-03-20
```

Variables utiles:

```bash
export DEVICE=0
export PREDICT_BATCH=4
export IMGSZ=960
export BATCH_SIZE=50
export CONF=0.05
export OUTPUT_ROOT=/ruta/salida/roboflow_batches
./run_jetson.sh /ruta/2026-03-17 /ruta/2026-03-18
```

## Uso directo de Python

```bash
python3 create_zip.py \
  --model-path ./models/model_detection.pt \
  --output-root ./output/roboflow_batches \
  --device 0 \
  --predict-batch 4 \
  --imgsz 960 \
  --batch-size 50 \
  --conf 0.05 \
  /ruta/capture_rumas/2026-03-17 \
  /ruta/capture_rumas/2026-03-18
```

## Notas

- Por defecto el lote es de `50` imagenes.
- Si la Jetson se queda corta de memoria, baja `PREDICT_BATCH` a `2` y `IMGSZ` a `640`.
- Las carpetas de imagenes y ZIPs generados no se versionan en Git.
