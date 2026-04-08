#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/roboflow_batches}"
DEVICE="${DEVICE:-0}"
PREDICT_BATCH="${PREDICT_BATCH:-4}"
IMGSZ="${IMGSZ:-1280}"
BATCH_SIZE="${BATCH_SIZE:-50}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/models/model_detection.pt}"

COMMAND="${1:-}"

if [[ "$COMMAND" == "train" ]]; then
    echo "=== Iniciando Entrenamiento Jetson Orin ==="
    shift
    python3 "$SCRIPT_DIR/train_jetson.py" "$@"
elif [[ "$COMMAND" == "annotate" ]] || [[ -z "$COMMAND" ]]; then
    echo "=== Auto-Anotación y Creación de Zips (Roboflow) ==="
    if [[ "$COMMAND" == "annotate" ]]; then shift; fi
    if [ "$#" -eq 0 ]; then
        if [ -d "$SCRIPT_DIR/capture_ruma" ]; then
            set -- "$SCRIPT_DIR/capture_ruma"
        elif [ -d "$SCRIPT_DIR/rumas-18" ]; then
            set -- "$SCRIPT_DIR/rumas-18"
        else
            echo "Error: No se indicaron directorios y no se encontró 'capture_ruma' ni 'rumas-18' en $SCRIPT_DIR"
            exit 1
        fi
    fi
    mkdir -p "$(dirname "$OUTPUT_ROOT")"
    python3 "$SCRIPT_DIR/create_zip.py" \
      --model-path "$MODEL_PATH" \
      --output-root "$OUTPUT_ROOT" \
      --device "$DEVICE" \
      --predict-batch "$PREDICT_BATCH" \
      --imgsz "$IMGSZ" \
      --batch-size "$BATCH_SIZE" \
      --conf "$CONF" \
      --iou "$IOU" \
      "$@"
else
    echo "Uso:"
    echo "  ./run_jetson.sh annotate [carpetas...]  # Para generar anotaciones ZIP (Por defecto)"
    echo "  ./run_jetson.sh train [argumentos_py]   # Para entrenar modelo personalizado"
    exit 1
fi
