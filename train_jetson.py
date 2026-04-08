"""
Entrenamiento de modelo YOLO personalizado para detección de rumas.
Optimizado para Jetson Orin AGX con GPU máxima precisión.

Uso:
    python3 train_jetson.py                        # Entrenamiento completo
    python3 train_jetson.py --epochs 50            # Menos épocas para prueba
    python3 train_jetson.py --model yolo11m.pt     # Usar modelo Medium
    python3 train_jetson.py --resume                # Reanudar entrenamiento anterior

Prerequisitos en Jetson:
    pip install ultralytics

El dataset esperado en ./dataset/ debe tener esta estructura:
    dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

Para generar el dataset desde cero con auto-anotación, ejecuta:
    ./run_jetson.sh annotate
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# ── Compatibilidad stdout en Windows / Jetson ─────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent

# ── Configuración por defecto ─────────────────────────────────────────────────
# Rutas relativas al repositorio (funciona en Jetson y en Windows)
DEFAULT_DATASET_DIR   = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR    = REPO_ROOT / "training_runs"
DEFAULT_MODEL_BASE    = "yolo11l.pt"   # Large = máxima precisión en Orin
DEFAULT_EPOCHS        = 100
DEFAULT_IMGSZ         = 1280           # Alta resolución → mejor detección de rumas
DEFAULT_BATCH         = 8             # Ajustar si la Orin queda sin memoria (bajar a 4)
DEFAULT_PATIENCE      = 20            # Early-stopping: para si no mejora en 20 épocas
DEFAULT_LR0           = 0.01
DEFAULT_LRF           = 0.001
DEFAULT_WORKERS       = 4
DEFAULT_PROJECT_NAME  = "rumas_custom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo YOLO personalizado para detección de rumas en Jetson Orin."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Ruta a la carpeta con data.yaml. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_BASE,
        help=(
            f"Modelo base YOLO a usar (yolo11n.pt/m/l/x). "
            f"Si existe en models/, lo usa como punto de partida. "
            f"Default: {DEFAULT_MODEL_BASE}"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Número de épocas de entrenamiento. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Resolución de entrenamiento. Default: {DEFAULT_IMGSZ}",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size. Reduce a 4 si la GPU se queda sin memoria. Default: {DEFAULT_BATCH}",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Épocas sin mejora para early-stop. Default: {DEFAULT_PATIENCE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta donde se guardan los runs de entrenamiento. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar el último entrenamiento interrumpido.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo: 0 (GPU), cpu. Si no se indica, usa GPU automáticamente.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"DataLoader workers. Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help=f"Nombre del proyecto de entrenamiento. Default: {DEFAULT_PROJECT_NAME}",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Activar augmentación avanzada (mosaic, mixup, flipud). Default: True",
    )
    parser.add_argument(
        "--copy-best",
        action="store_true",
        default=True,
        help="Copiar automáticamente el best.pt a models/ al terminar. Default: True",
    )
    return parser.parse_args()


def check_gpu() -> str:
    """Verifica disponibilidad de GPU y devuelve el device a usar."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ GPU detectada: {gpu_name} ({mem_gb:.1f} GB)")
            return "0"
        else:
            print("  ⚠️  No se detectó GPU. Usando CPU (será lento).")
            return "cpu"
    except ImportError:
        print("  ❌ PyTorch no está instalado. Ejecuta: pip install torch torchvision")
        sys.exit(1)


def resolve_model(model_arg: str) -> str:
    """
    Resuelve la ruta del modelo base.
    Prioridad:
      1. Si existe en models/ del repo → lo usa como fine-tuning
      2. Si es un nombre de modelo YOLO estándar → ultralytics lo descarga
    """
    local_path = REPO_ROOT / "models" / model_arg
    if local_path.exists():
        print(f"  ✅ Usando modelo local: {local_path}")
        return str(local_path)

    # Buscar cualquier .pt en models/ como alternativa
    models_dir = REPO_ROOT / "models"
    available_pts = sorted(models_dir.glob("*.pt"))
    if available_pts:
        best_local = available_pts[-1]
        print(f"  ℹ️  '{model_arg}' no encontrado. Usando modelo local: {best_local.name}")
        return str(best_local)

    print(f"  ℹ️  Usando modelo base público: {model_arg} (se descargará si es necesario)")
    return model_arg


def validate_dataset(dataset_dir: Path, data_yaml: Path) -> None:
    """Verifica que el dataset tenga la estructura correcta."""
    errors: list[str] = []

    if not dataset_dir.exists():
        errors.append(f"La carpeta del dataset no existe: {dataset_dir}")
    if not data_yaml.exists():
        errors.append(f"No se encontró data.yaml en: {data_yaml}")

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            path = dataset_dir / split / sub
            if not path.exists():
                errors.append(f"Falta la carpeta: {path}")

    if errors:
        print("\n❌ Errores en el dataset:")
        for err in errors:
            print(f"   • {err}")
        print(
            "\n💡 Sugerencia: Primero genera el dataset con:\n"
            "   ./run_jetson.sh annotate\n"
            "   Luego sube a Roboflow, descarga con split train/val, y colócalo en ./dataset/"
        )
        sys.exit(1)

    # Contar imágenes
    train_imgs = list((dataset_dir / "train" / "images").glob("*.*"))
    val_imgs   = list((dataset_dir / "val" / "images").glob("*.*"))
    print(f"  ✅ Dataset válido: {len(train_imgs)} imágenes entrenamiento, {len(val_imgs)} validación")


def copy_best_model(run_dir: Path, models_dir: Path, run_name: str) -> None:
    """Copia best.pt a la carpeta models/ con nombre descriptivo."""
    best_src = run_dir / "weights" / "best.pt"
    if not best_src.exists():
        print(f"  ⚠️  No se encontró best.pt en {best_src}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"model_detection_{run_name}_{timestamp}.pt"
    dest_path = models_dir / dest_name
    shutil.copy2(best_src, dest_path)

    # Actualizar también model_detection.pt (el que usan los otros scripts)
    default_model = models_dir / "model_detection.pt"
    shutil.copy2(best_src, default_model)

    print(f"\n  ✅ Modelo guardado en:  models/{dest_name}")
    print(f"  ✅ También actualizado: models/model_detection.pt")
    print(f"\n  🚀 Listo para producción. Prueba con:")
    print(f"     python3 test_visual.py --model models/model_detection.pt")


def main() -> None:
    args = parse_args()

    print("=" * 68)
    print("  ENTRENAMIENTO YOLO PARA DETECCIÓN DE RUMAS — JETSON ORIN")
    print("=" * 68)

    # ── 1. Verificar GPU ──────────────────────────────────────────────────────
    print("\n🔍 Verificando hardware...")
    device = args.device if args.device else check_gpu()

    # ── 2. Resolver modelo base ───────────────────────────────────────────────
    print("\n🔍 Resolviendo modelo base...")
    model_path = resolve_model(args.model)

    # ── 3. Validar dataset ────────────────────────────────────────────────────
    data_yaml = args.dataset / "data.yaml"

    if not args.resume:
        print(f"\n🔍 Validando dataset en: {args.dataset}")
        validate_dataset(args.dataset, data_yaml)
    else:
        print("\n♻️  Modo RESUME: reanudando entrenamiento anterior...")

    # ── 4. Configurar output ──────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    # ── 5. Mostrar configuración ──────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("  CONFIGURACIÓN DE ENTRENAMIENTO")
    print("-" * 68)
    print(f"  Modelo base:    {model_path}")
    print(f"  Dataset:        {data_yaml}")
    print(f"  Épocas:         {args.epochs}")
    print(f"  Resolución:     {args.imgsz}px")
    print(f"  Batch size:     {args.batch}")
    print(f"  Early-stop:     {args.patience} épocas")
    print(f"  Device:         {device}")
    print(f"  Workers:        {args.workers}")
    print(f"  Runs output:    {args.output}")
    print("-" * 68)

    if not args.resume:
        print("\n⏳ Esto puede tardar 1-3 horas en la Jetson Orin. ¡Paciencia! 💪")
        print("   Presiona Ctrl+C para pausar y reanudar con --resume\n")

    # ── 6. Cargar e iniciar entrenamiento ─────────────────────────────────────
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        print("❌ ultralytics no está instalado. Ejecuta: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)

    train_kwargs = {
        "data":       str(data_yaml),
        "epochs":     args.epochs,
        "imgsz":      args.imgsz,
        "batch":      args.batch,
        "patience":   args.patience,
        "device":     device,
        "workers":    args.workers,
        "project":    str(args.output),
        "name":       args.name,
        "exist_ok":   args.resume,
        "resume":     args.resume,
        # ── Augmentación para mayor precisión con pocos datos ──────────────
        "mosaic":     1.0,       # Combina 4 imágenes (clave con pocos datos)
        "mixup":      0.1,       # Mezcla imágenes suave
        "flipud":     0.5,       # Flip vertical
        "fliplr":     0.5,       # Flip horizontal
        "degrees":    15.0,      # Rotación ±15°
        "translate":  0.1,       # Traslación 10%
        "scale":      0.5,       # Escala ±50%
        "shear":      2.0,       # Cizalla leve
        "perspective":0.0,       # Perspectiva (desactivada para rumas)
        "hsv_h":      0.015,     # Variación de tono
        "hsv_s":      0.7,       # Variación de saturación
        "hsv_v":      0.4,       # Variación de brillo
        # ── Hiperparámetros de aprendizaje ────────────────────────────────
        "lr0":        DEFAULT_LR0,
        "lrf":        DEFAULT_LRF,
        "warmup_epochs": 3,
        "cos_lr":     True,      # Cosine LR scheduler
        "optimizer":  "AdamW",   # AdamW converge mejor que SGD con pocos datos
        "weight_decay": 0.0005,
        "label_smoothing": 0.1,  # Reduce sobreajuste
        "plots":      True,      # Genera gráficos de curvas de entrenamiento
        "save":       True,
        "save_period": 10,       # Guardar checkpoint cada 10 épocas
        "val":        True,
        "verbose":    True,
    }

    results = model.train(**train_kwargs)

    # ── 7. Copiar el mejor modelo ─────────────────────────────────────────────
    if args.copy_best:
        run_dir = Path(results.save_dir)
        models_dir = REPO_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        print("\n📦 Guardando mejor modelo...")
        copy_best_model(run_dir, models_dir, args.name)

    print("\n" + "=" * 68)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print(f"  📁 Resultados en: {args.output}/{args.name}")
    print(f"  🏆 Mejor modelo:  models/model_detection.pt")
    print("=" * 68)


if __name__ == "__main__":
    main()
