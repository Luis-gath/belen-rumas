#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/roboflow_batches}"
DEVICE="${DEVICE:-0}"
PREDICT_BATCH="${PREDICT_BATCH:-4}"
IMGSZ="${IMGSZ:-960}"
BATCH_SIZE="${BATCH_SIZE:-50}"
CONF="${CONF:-0.05}"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/models/model_detection.pt}"

if [ "$#" -lt 1 ]; then
  echo "Uso:"
  echo "  ./run_jetson.sh /ruta/capture_rumas/2026-03-17 [/ruta/2026-03-18 ...]"
  exit 1
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
  "$@"

