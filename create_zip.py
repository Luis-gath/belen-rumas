"""
Prepara lotes para Roboflow a partir de varias carpetas de imagenes.

Flujo:
1. Lee imagenes desde varias carpetas.
2. Ejecuta autoanotacion YOLO con un modelo .pt.
3. Agrupa en lotes de N imagenes.
4. Genera una carpeta y un ZIP por lote con formato YOLO para Roboflow.

Uso:
    python create_zip.py
    python create_zip.py --limit 10
    python create_zip.py --batch-size 50 --conf 0.05
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parent
REPO_MODEL_PATH = REPO_ROOT / "models" / "model_detection.pt"
DEFAULT_INPUT_DIRS = []
if (REPO_ROOT / "capture_ruma").exists():
    DEFAULT_INPUT_DIRS = [REPO_ROOT / "capture_ruma"]
elif (REPO_ROOT / "rumas-18").exists():
    DEFAULT_INPUT_DIRS = [REPO_ROOT / "rumas-18"]

DEFAULT_MODEL_PATH = REPO_MODEL_PATH
DEFAULT_BATCH_SIZE = 50
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_PREDICT_BATCH = 8
DEFAULT_IOU = 0.45
DEFAULT_DEDUPE_IOU = 0.50
DEFAULT_DEDUPE_OVERLAP = 0.80


def default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / f"roboflow_batches_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoanota imagenes y crea lotes ZIP para importar en Roboflow."
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        type=Path,
        help="Carpetas con imagenes. Si no se indican, se usan las 4 fechas configuradas.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta del modelo YOLO (.pt). Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Carpeta base donde se escribiran los lotes y ZIPs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Imagenes por lote. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Umbral de confianza para inferencia. Default: {DEFAULT_CONFIDENCE}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Tamano de inferencia YOLO. Default: {DEFAULT_IMAGE_SIZE}",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IoU para NMS de YOLO. Mas bajo = menos cajas duplicadas. Default: {DEFAULT_IOU}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo YOLO. Ej: cpu, 0, cuda:0. Si no se indica, Ultralytics decide.",
    )
    parser.add_argument(
        "--predict-batch",
        type=int,
        default=DEFAULT_PREDICT_BATCH,
        help=(
            "Batch interno de inferencia YOLO. Reduce este valor si el equipo se pone pesado. "
            f"Default: {DEFAULT_PREDICT_BATCH}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo las primeras N imagenes. Util para prueba.",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=DEFAULT_DEDUPE_IOU,
        help=(
            "IoU del filtro extra para quitar duplicados despues de YOLO. "
            f"Default: {DEFAULT_DEDUPE_IOU}"
        ),
    )
    parser.add_argument(
        "--dedupe-overlap",
        type=float,
        default=DEFAULT_DEDUPE_OVERLAP,
        help=(
            "Solapamiento sobre la caja menor para quitar cajas anidadas del mismo objeto. "
            f"Default: {DEFAULT_DEDUPE_OVERLAP}"
        ),
    )
    return parser.parse_args()


def collect_images(input_dirs: list[Path]) -> list[Path]:
    image_files: list[Path] = []

    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"La ruta no es carpeta: {input_dir}")

        files = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{input_dir} -> {len(files)} imagenes")
        image_files.extend(files)

    return image_files


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def write_classes_file(target: Path, class_names: list[str]) -> None:
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(class_names) + "\n")


def write_data_yaml(target: Path, class_names: list[str]) -> None:
    names = ", ".join(f"'{name}'" for name in class_names)
    content = (
        "train: train/images\n"
        "val: train/images\n\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names}]\n"
    )
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def zip_directory(directory: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(directory).as_posix())


def compute_intersection(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    inter_area = compute_intersection(box_a, box_b)
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_overlap_on_smaller(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    inter_area = compute_intersection(box_a, box_b)
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return 0.0
    return inter_area / smaller


def extract_filtered_detections(result, dedupe_iou: float, dedupe_overlap: float) -> list[tuple[int, float, float, float, float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    candidates: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        candidates.append((cls_id, conf, (x1, y1, x2, y2)))

    candidates.sort(key=lambda item: item[1], reverse=True)

    filtered: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for candidate in candidates:
        cls_id, _, box_xyxy = candidate
        is_duplicate = False

        for kept_cls_id, _, kept_box in filtered:
            if kept_cls_id != cls_id:
                continue

            if compute_iou(box_xyxy, kept_box) >= dedupe_iou:
                is_duplicate = True
                break

            if compute_overlap_on_smaller(box_xyxy, kept_box) >= dedupe_overlap:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(candidate)

    return [(cls_id, *box_xyxy) for cls_id, _, box_xyxy in filtered]


def write_yolo_label(
    label_path: Path, result, dedupe_iou: float, dedupe_overlap: float
) -> int:
    detections = extract_filtered_detections(result, dedupe_iou, dedupe_overlap)

    with label_path.open("w", encoding="utf-8", newline="\n") as handle:
        if not detections:
            return 0

        img_h, img_w = result.orig_shape

        for cls_id, x1, y1, x2, y2 in detections:
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            handle.write(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )

    return len(detections)


def ensure_empty_output_root(output_root: Path) -> None:
    if output_root.exists():
        existing_items = list(output_root.iterdir())
        if existing_items:
            raise FileExistsError(
                f"La carpeta de salida ya existe y no esta vacia: {output_root}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_dirs = args.input_dirs or DEFAULT_INPUT_DIRS

    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que 0")
    if args.predict_batch <= 0:
        raise ValueError("--predict-batch debe ser mayor que 0")

    if not args.model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {args.model_path}")

    ensure_empty_output_root(args.output_root)

    print("=" * 72)
    print("PREPARACION DE LOTES ROBOFLOW")
    print("=" * 72)
    print(f"Modelo:       {args.model_path}")
    print(f"Salida:       {args.output_root}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Confianza:    {args.conf}")
    print(f"IoU NMS:      {args.iou}")
    print(f"Image size:   {args.imgsz}")
    print(f"Device:       {args.device or 'auto'}")
    print(f"Pred batch:   {args.predict_batch}")
    print(f"Dedupe IoU:   {args.dedupe_iou}")
    print(f"Dedupe Over.: {args.dedupe_overlap}")
    print("-" * 72)
    print("Entradas:")
    for input_dir in input_dirs:
        print(f"  - {input_dir}")
    print("-" * 72)

    image_files = collect_images(input_dirs)
    if args.limit is not None:
        image_files = image_files[: args.limit]
        print(f"Limit aplicado: {len(image_files)} imagenes")

    if not image_files:
        raise RuntimeError("No se encontraron imagenes para procesar")

    total_batches = math.ceil(len(image_files) / args.batch_size)
    print(f"Total imagenes: {len(image_files)}")
    print(f"Total lotes:    {total_batches}")
    print("-" * 72)

    print("Cargando modelo...")
    model = YOLO(str(args.model_path))
    class_names = [model.names[index] for index in sorted(model.names)]
    print(f"Clases detectadas: {class_names}")
    print("-" * 72)

    manifest_path = args.output_root / "manifest.csv"
    total_detections = 0
    images_with_detections = 0

    batches = chunked(image_files, args.batch_size)

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "batch_id",
                "image_name",
                "source_path",
                "label_path",
                "zip_path",
                "detections",
            ]
        )

        for batch_index, batch_images in enumerate(batches, start=1):
            batch_name = f"batch_{batch_index:03d}"
            batch_dir = args.output_root / batch_name
            train_dir = batch_dir / "train"
            images_dir = train_dir / "images"
            labels_dir = train_dir / "labels"
            zip_path = args.output_root / f"{batch_name}.zip"

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            write_classes_file(batch_dir / "classes.txt", class_names)
            write_data_yaml(batch_dir / "data.yaml", class_names)

            print(
                f"[{batch_index:02d}/{total_batches:02d}] "
                f"Procesando {len(batch_images)} imagenes -> {batch_name}"
            )

            results = model.predict(
                source=[str(path) for path in batch_images],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                batch=min(args.predict_batch, len(batch_images)),
                device=args.device,
                verbose=False,
            )

            if len(results) != len(batch_images):
                raise RuntimeError(
                    f"El modelo devolvio {len(results)} resultados para "
                    f"{len(batch_images)} imagenes"
                )

            batch_detections = 0
            batch_images_with_detections = 0

            for image_path, result in zip(batch_images, results):
                target_image_path = images_dir / image_path.name
                target_label_path = labels_dir / f"{image_path.stem}.txt"

                if target_image_path.exists():
                    raise FileExistsError(
                        f"Nombre de imagen duplicado dentro del lote: {target_image_path.name}"
                    )

                shutil.copy2(image_path, target_image_path)
                detections = write_yolo_label(
                    target_label_path,
                    result,
                    dedupe_iou=args.dedupe_iou,
                    dedupe_overlap=args.dedupe_overlap,
                )

                batch_detections += detections
                total_detections += detections
                if detections > 0:
                    batch_images_with_detections += 1
                    images_with_detections += 1

                writer.writerow(
                    [
                        batch_name,
                        image_path.name,
                        str(image_path),
                        str(target_label_path),
                        str(zip_path),
                        detections,
                    ]
                )

            zip_directory(batch_dir, zip_path)
            print(
                f"    ZIP listo: {zip_path.name} | "
                f"imagenes con deteccion: {batch_images_with_detections}/{len(batch_images)} | "
                f"detecciones: {batch_detections}"
            )

    print("-" * 72)
    print("Proceso terminado")
    print(f"Lotes creados:              {len(batches)}")
    print(f"Imagenes procesadas:        {len(image_files)}")
    print(f"Imagenes con detecciones:   {images_with_detections}")
    print(f"Imagenes sin detecciones:   {len(image_files) - images_with_detections}")
    print(f"Total detecciones:          {total_detections}")
    print(f"Manifest:                   {manifest_path}")
    print(f"Salida final:               {args.output_root}")


if __name__ == "__main__":
    main()
