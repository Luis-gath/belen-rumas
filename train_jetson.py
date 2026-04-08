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
import zipfile
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
DEFAULT_BATCH         = 4             # Seguro para Jetson; subir a 8 solo si no hay OOM
DEFAULT_PATIENCE      = 20            # Early-stopping: para si no mejora en 20 épocas
DEFAULT_LR0           = 0.01
DEFAULT_LRF           = 0.001
DEFAULT_WORKERS       = 1             # Evita OOM por prefetch de DataLoader en memoria unificada
DEFAULT_PROJECT_NAME  = "rumas_custom"

# ── Configuración Roboflow (Opcional, para descarga automática) ───────────────
# Pega tus datos aquí si quieres descargar el dataset directamente desde el código.
ROBOFLOW_API_KEY      = ""            # Pega tu Private API Key aquí (ej: "xyz123abc")
ROBOFLOW_WORKSPACE    = "asdcain-gzzzu" # Tu espacio de trabajo
ROBOFLOW_PROJECT      = "belen"       # El nombre de tu proyecto
ROBOFLOW_VERSION      = 5             # La versión que quieres descargar
ROBOFLOW_FORMAT       = "auto"        # auto prueba formatos YOLO compatibles
ROBOFLOW_AUTO_FORMATS = ("yolov8", "yolov5pytorch", "yolov11", "yolov5")


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
            f"Los nombres yolo*.pt se descargan desde Ultralytics. "
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
        help=f"Batch size. Reduce a 2 si la Jetson termina con 'Killed'. Default: {DEFAULT_BATCH}",
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
        help=f"DataLoader workers. Usa 0-1 en Jetson si el proceso termina con 'Killed'. Default: {DEFAULT_WORKERS}",
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
    # ── Argumentos Roboflow ───────────────────────────────────────────────────
    parser.add_argument(
        "--rf-key",
        type=str,
        default=ROBOFLOW_API_KEY,
        help="Roboflow Private API Key para descargar el dataset automáticamente.",
    )
    parser.add_argument(
        "--rf-workspace",
        type=str,
        default=ROBOFLOW_WORKSPACE,
        help="Nombre de tu espacio de trabajo en Roboflow.",
    )
    parser.add_argument(
        "--rf-project",
        type=str,
        default=ROBOFLOW_PROJECT,
        help="Nombre de tu proyecto en Roboflow.",
    )
    parser.add_argument(
        "--rf-version",
        type=int,
        default=ROBOFLOW_VERSION,
        help="Número de versión del dataset en Roboflow.",
    )
    parser.add_argument(
        "--rf-format",
        type=str,
        default=ROBOFLOW_FORMAT,
        help=(
            "Formato de exportación Roboflow. Usa 'auto' para probar formatos YOLO "
            f"compatibles: {', '.join(ROBOFLOW_AUTO_FORMATS)}. Default: {ROBOFLOW_FORMAT}"
        ),
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
      1. Si es un nombre oficial yolo*.pt → Ultralytics lo descarga
      2. Si es una ruta explicita existente → la usa como fine-tuning
      3. Si no existe → falla de forma explicita; no usa otro modelo local silenciosamente
    """
    model_name = Path(model_arg).name.lower()
    is_plain_name = model_name == model_arg.lower() and "/" not in model_arg and "\\" not in model_arg
    if is_plain_name and model_name.startswith("yolo") and model_name.endswith(".pt"):
        print(f"  ✅ Usando modelo oficial Ultralytics: {model_arg}")
        return model_arg

    if is_plain_name:
        print(f"  ❌ No usaré el modelo local por nombre: {model_arg}")
        print("     Para máxima precisión usa: --model yolo11l.pt")
        print("     Si realmente quieres un .pt local, pásalo como ruta explícita: --model models/archivo.pt")
        sys.exit(1)

    direct_path = Path(model_arg)
    if direct_path.exists():
        print(f"  ✅ Usando modelo local indicado explícitamente: {direct_path}")
        return str(direct_path)

    print(f"  ❌ No se encontró el modelo solicitado: {model_arg}")
    print("     Ejemplo recomendado para máxima precisión controlada: --model yolo11l.pt")
    sys.exit(1)


def find_dataset_yaml(dataset_dir: Path) -> Path | None:
    """Busca el YAML del dataset priorizando data.yaml."""
    yamls = list(dataset_dir.rglob("*.yaml")) + list(dataset_dir.rglob("*.yml"))

    # Excluir posibles archivos ocultos o irrelevantes, buscar el que se parezca a data.yaml
    yamls = [y for y in yamls if "data" in y.name.lower() or "dataset" in y.name.lower()]

    if not yamls:
        return None

    yamls.sort(
        key=lambda y: (
            y.name.lower() != "data.yaml",
            len(y.relative_to(dataset_dir).parts),
            str(y).lower(),
        )
    )
    return yamls[0]


def validate_dataset(dataset_dir: Path) -> Path:
    """Busca el data.yaml, verifica la estructura y devuelve el directorio real del dataset."""
    true_yaml = find_dataset_yaml(dataset_dir)

    if true_yaml is None:
        print(f"\n❌ Errores en el dataset:")
        print(f"   • No se encontró un archivo data.yaml dentro de: {dataset_dir}")
        print(
            "\n💡 Sugerencia: Elimina la carpeta y vuelve a intentar para forzar la descarga.\n"
        )
        sys.exit(1)

    true_dir = true_yaml.parent
    
    # Roboflow a veces usa "valid" en lugar de "val"
    val_name = "valid" if (true_dir / "valid").exists() else "val"
    
    errors: list[str] = []
    for split in ("train", val_name):
        for sub in ("images", "labels"):
            path = true_dir / split / sub
            if not path.exists():
                errors.append(f"Falta la carpeta: {path}")

    if errors:
        print("\n❌ Errores en la estructura del dataset:")
        for err in errors:
            print(f"   • {err}")
        sys.exit(1)

    # Contar imágenes
    train_imgs = list((true_dir / "train" / "images").glob("*.*"))
    val_imgs   = list((true_dir / val_name / "images").glob("*.*"))
    print(f"  ✅ Dataset encontrado y válido en: {true_dir}")
    print(f"  ✅ Imágenes: {len(train_imgs)} en train, {len(val_imgs)} en val/valid")
    
    return true_dir


def roboflow_formats_to_try(requested_format: str) -> list[str]:
    """Devuelve los formatos Roboflow a probar."""
    requested = requested_format.strip()
    if not requested or requested.lower() == "auto":
        return list(ROBOFLOW_AUTO_FORMATS)
    return [requested]


def preview_directory(root: Path, max_entries: int = 20) -> str:
    """Muestra una vista corta de la carpeta para diagnosticar descargas vacias."""
    if not root.exists():
        return f"      {root} no existe"

    entries = sorted(root.rglob("*"), key=lambda path: str(path).lower())
    if not entries:
        return "      (carpeta vacia)"

    lines = []
    for path in entries[:max_entries]:
        suffix = "/" if path.is_dir() else ""
        lines.append(f"      - {path.relative_to(root)}{suffix}")
    if len(entries) > max_entries:
        lines.append(f"      ... {len(entries) - max_entries} archivos/carpetas mas")
    return "\n".join(lines)


def extract_zip_artifacts(root: Path) -> None:
    """Extrae ZIPs que Roboflow haya dejado sin desempaquetar."""
    if not root.exists():
        return

    for zip_path in sorted(root.rglob("*.zip"), key=lambda path: str(path).lower()):
        extract_dir = zip_path.with_suffix("")
        if find_dataset_yaml(extract_dir):
            continue

        print(f"  ℹ️  Se encontró un ZIP sin extraer. Extrayendo: {zip_path.name}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version: int,
    download_dir: Path,
    model_format: str,
) -> Path:
    """Descarga el dataset desde Roboflow usando la librería oficial."""
    try:
        from roboflow import Roboflow  # noqa: PLC0415
    except ImportError:
        print("❌ Para usar integración con Roboflow debes instalar la librería:")
        print("   pip install roboflow")
        sys.exit(1)

    print("\n🌐 Conectando con Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version_obj = project.version(version)
    failures: list[str] = []

    for model_format_candidate in roboflow_formats_to_try(model_format):
        print(f"  ↳ Descargando formato Roboflow: {model_format_candidate}")

        # Limpiar carpeta si ya existe para evitar que Roboflow reutilice una descarga incompleta.
        # Importante: version.download() retorna sin descargar si location ya existe
        # y overwrite=False, incluso si la carpeta esta vacia.
        if download_dir.exists():
            shutil.rmtree(download_dir)

        download_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            try:
                dataset = version_obj.download(
                    model_format_candidate,
                    location=str(download_dir),
                    overwrite=True,
                )
            except TypeError:
                # Compatibilidad con SDKs antiguos que no aceptan overwrite.
                dataset = version_obj.download(model_format_candidate, location=str(download_dir))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{model_format_candidate}: {exc}")
            continue

        dataset_path = Path(getattr(dataset, "location", download_dir))
        search_roots = [dataset_path]
        if download_dir != dataset_path:
            search_roots.append(download_dir)

        for root in search_roots:
            extract_zip_artifacts(root)

        for root in search_roots:
            dataset_yaml = find_dataset_yaml(root)
            if dataset_yaml is not None:
                print(f"✅ Dataset de Roboflow descargado en: {dataset_path}")
                print(f"  ✅ YAML encontrado: {dataset_yaml}")
                return dataset_yaml.parent

        failures.append(
            f"{model_format_candidate}: descarga sin data.yaml. Contenido de {download_dir}:\n"
            f"{preview_directory(download_dir)}"
        )

    print("\n❌ Roboflow descargó una carpeta, pero no encontré data.yaml.")
    print("   Intentos realizados:")
    for failure in failures:
        print(f"   • {failure}")
    print(
        "\n💡 En Roboflow, genera/descarga la versión como YOLOv8 PyTorch o YOLOv5 PyTorch. "
        "YOLO11 usa la misma estructura de dataset para entrenar con Ultralytics."
    )
    sys.exit(1)


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

    # ── 2.5 Descarga desde Roboflow (opcional) ────────────────────────────────
    if args.rf_key and args.rf_workspace and args.rf_project:
        args.dataset = download_roboflow_dataset(
            api_key=args.rf_key,
            workspace=args.rf_workspace,
            project_name=args.rf_project,
            version=args.rf_version,
            download_dir=REPO_ROOT / f"roboflow_{args.rf_project}_v{args.rf_version}",
            model_format=args.rf_format,
        )

    # ── 3. Validar dataset ────────────────────────────────────────────────────
    if not args.resume:
        print(f"\n🔍 Validando dataset en: {args.dataset}")
        args.dataset = validate_dataset(args.dataset)
        data_yaml = find_dataset_yaml(args.dataset)
        if data_yaml is None:
            print(f"❌ No se encontró data.yaml en: {args.dataset}")
            sys.exit(1)
    else:
        data_yaml = find_dataset_yaml(args.dataset) or (args.dataset / "data.yaml")
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
        "cache":      False,      # No cargar imagenes en RAM; Jetson usa memoria unificada
        "project":    str(args.output),
        "name":       args.name,
        "exist_ok":   args.resume,
        "resume":     args.resume,
        # ── Augmentación para mayor precisión con pocos datos ──────────────
        "mosaic":     1.0,       # Combina 4 imágenes (clave con pocos datos)
        "mixup":      0.1,       # Mezcla imágenes suave
        "close_mosaic": 10,      # Reduce uso de memoria al final del entrenamiento
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
        "amp":        True,      # Menos memoria que FP32; Ultralytics ya valida AMP antes de entrenar
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
