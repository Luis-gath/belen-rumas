"""
Genera vistas previas con bounding boxes a partir de un dataset preanotado.

Uso:
    python preview_annotations.py --dataset-root ./output/preanotado_dataset_conf025_dedupe_v1
    python preview_annotations.py --dataset-root ./output/preanotado_dataset_conf025_dedupe_v1 --count 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dibuja previews de etiquetas YOLO sobre imagenes preanotadas."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Carpeta de salida generada por create_zip.py",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Cantidad de previews a generar. Default: 2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida para previews. Default: <dataset-root>/previews",
    )
    return parser.parse_args()


def find_batch_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("batch_")
    )


def collect_image_label_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for batch_dir in find_batch_dirs(dataset_root):
        images_dir = batch_dir / "train" / "images"
        labels_dir = batch_dir / "train" / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))

    return pairs


def draw_labels(image_path: Path, label_path: Path, output_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    height, width = image.shape[:2]
    count = 0

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, x_center, y_center, box_width, box_height = parts

            x_center_px = float(x_center) * width
            y_center_px = float(y_center) * height
            box_width_px = float(box_width) * width
            box_height_px = float(box_height) * height

            x1 = int(x_center_px - box_width_px / 2)
            y1 = int(y_center_px - box_height_px / 2)
            x2 = int(x_center_px + box_width_px / 2)
            y2 = int(y_center_px + box_height_px / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                "ruma",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            count += 1

    cv2.imwrite(str(output_path), image)
    return count


def main() -> None:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count debe ser mayor que 0")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"No existe la carpeta: {args.dataset_root}")

    output_dir = args.output_dir or (args.dataset_root / "previews")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_image_label_pairs(args.dataset_root)
    if not pairs:
        raise RuntimeError("No se encontraron pares de imagen/label para previsualizar")

    selected_pairs = pairs[: args.count]

    print(f"Dataset:     {args.dataset_root}")
    print(f"Previews:    {output_dir}")
    print(f"Seleccion:   {len(selected_pairs)} de {len(pairs)} imagenes")

    for index, (image_path, label_path) in enumerate(selected_pairs, start=1):
        output_path = output_dir / f"preview_{index:02d}_{image_path.name}"
        box_count = draw_labels(image_path, label_path, output_path)
        print(f"{output_path.name} -> {box_count} cajas")


if __name__ == "__main__":
    main()
