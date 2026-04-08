# Entrenamiento en Google Colab

Usa `train_colab.ipynb` para entrenar en Colab sin depender de la Jetson.

## Flujo recomendado

1. Sube o abre `train_colab.ipynb` en Google Colab.
2. Activa GPU: `Runtime > Change runtime type > T4/L4/A100 GPU`.
3. En Colab, guarda tu API key como secret con nombre `ROBOFLOW_API_KEY`.
4. Ejecuta las celdas en orden.

El notebook guarda todo en Google Drive:

```text
MyDrive/belen_yolo/
  datasets/roboflow_belen_v5/
  training_runs/
  models/
```

Si el dataset ya existe en Drive y tiene `data.yaml`, no lo vuelve a descargar.
Para forzar una nueva descarga, cambia en el notebook:

```python
FORCE_DOWNLOAD = True
```

## Configuracion inicial

El notebook viene con una configuracion conservadora:

```python
MODEL = "yolo11l.pt"
IMGSZ = 1024
BATCH = 2
WORKERS = 2
EPOCHS = 100
```

Si te toca A100 o L4 y ves memoria libre, puedes subir `BATCH` a `4` u `8`.
Si te toca T4 y aparece OOM, baja `BATCH` a `1`.

## Reanudar

Si Colab se desconecta, vuelve a ejecutar las celdas de instalacion, Drive y configuracion.
Luego usa la celda de resume del notebook. El checkpoint esperado queda en:

```text
MyDrive/belen_yolo/training_runs/rumas_yolo11l_colab/weights/last.pt
```
