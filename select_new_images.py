"""
Selecciona imagenes nuevas desde capture_rumas excluyendo las ya procesadas.

Uso tipico:
    python select_new_images.py ^
      --source-root "D:\\SUBASTA\\minera-belen\\iakol_minera_belen_titan_alma_peru\\capture_rumas" ^
      --used-root "D:\\SUBASTA\\deteccion-rumas-belen-para-roboflow" ^
      --limit 100

Luego usa la carpeta selected_images/ con create_zip.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    r"D:\SUBASTA\minera-belen\iakol_minera_belen_titan_alma_peru\capture_rumas"
)

TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{6})")
CAMERA_RE = re.compile(r"belen_titan_(\d+)", re.IGNORECASE)


def default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / f"seleccion_nuevas_roboflow_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona imagenes no usadas desde capture_rumas para nuevo etiquetado."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Carpeta raiz de capturas. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--used-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Carpeta(s) donde buscar imagenes ya procesadas. "
            "Se puede repetir. Default: repo actual."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Carpeta donde copiar la seleccion y guardar manifest. Default: timestamp.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Cantidad de imagenes nuevas a seleccionar. Default: 100.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Filtrar por camara, ej. --camera 01 --camera 03. "
            "Si no se indica, usa todas."
        ),
    )
    parser.add_argument(
        "--balance-by-camera",
        action="store_true",
        help="Selecciona en round-robin por camara para evitar sesgo fuerte hacia una sola camara.",
    )
    parser.add_argument(
        "--rf-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="API key de Roboflow. Si no se indica, usa la variable ROBOFLOW_API_KEY.",
    )
    parser.add_argument(
        "--rf-workspace",
        type=str,
        default=None,
        help="Workspace ID de Roboflow para excluir exactamente el dataset actual.",
    )
    parser.add_argument(
        "--rf-project",
        type=str,
        default=None,
        help="Project ID de Roboflow para excluir exactamente el dataset actual.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra resumen; no copia imagenes.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"No existe: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_stem(path: Path) -> str:
    return path.stem.casefold()


def normalize_filename_stem(filename: str) -> str:
    return Path(filename).stem.casefold()


def parse_capture_time(path: Path) -> datetime:
    match = TIMESTAMP_RE.search(path.stem)
    if match:
        return datetime.strptime("_".join(match.groups()), "%Y-%m-%d_%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def parse_camera(path: Path) -> str:
    match = CAMERA_RE.search(path.stem)
    if match:
        return match.group(1).zfill(2)
    return "unknown"


def build_used_stems(used_roots: list[Path], source_root: Path) -> set[str]:
    used: set[str] = set()
    source_resolved = source_root.resolve()

    for used_root in used_roots:
        for image in iter_images(used_root):
            # No marques como usada la fuente completa si alguien la pasa por accidente.
            try:
                image.resolve().relative_to(source_resolved)
            except ValueError:
                used.add(normalize_stem(image))

    return used


def fetch_roboflow_dataset_images(api_key: str, workspace: str, project: str) -> list[dict[str, str]]:
    url = f"https://api.roboflow.com/{workspace}/search/v1?api_key={api_key}"
    continuation_token: str | None = None
    rows: list[dict[str, str]] = []

    while True:
        payload: dict[str, object] = {
            "query": f"dataset:{project}",
            "pageSize": 250,
            "fields": ["filename", "split"],
        }
        if continuation_token:
            payload["continuationToken"] = continuation_token

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)

        for item in data.get("results", []):
            project_data = item.get("projectData", {}).get(project, {})
            rows.append(
                {
                    "id": item.get("id", ""),
                    "filename": item.get("filename", ""),
                    "split": project_data.get("split", ""),
                }
            )

        continuation_token = data.get("continuationToken")
        if not continuation_token:
            break

    return rows


def select_balanced_candidates(
    candidates: list[tuple[datetime, str, str, Path]],
    limit: int,
) -> list[tuple[datetime, str, str, Path]]:
    by_camera: dict[str, list[tuple[datetime, str, str, Path]]] = {}
    for candidate in candidates:
        by_camera.setdefault(candidate[1], []).append(candidate)

    for camera_candidates in by_camera.values():
        camera_candidates.sort()

    selected: list[tuple[datetime, str, str, Path]] = []
    while len(selected) < limit:
        active_cameras = [
            camera for camera, camera_candidates in by_camera.items() if camera_candidates
        ]
        if not active_cameras:
            break

        active_cameras.sort(key=lambda camera: by_camera[camera][0])
        progressed = False
        for camera in active_cameras:
            if len(selected) >= limit:
                break
            selected.append(by_camera[camera].pop(0))
            progressed = True
        if not progressed:
            break

    return selected


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit debe ser mayor que 0")

    if args.used_root is None:
        args.used_root = [] if (args.rf_workspace and args.rf_project) else [REPO_ROOT]

    output_root = args.output_root or default_output_root()
    selected_dir = output_root / "selected_images"
    manifest_path = output_root / "selected_images.csv"

    source_images = iter_images(args.source_root)
    source_by_stem = {normalize_stem(path): path for path in source_images}
    used_stems = build_used_stems(args.used_root, args.source_root)
    camera_filter = {camera.zfill(2) for camera in args.camera}
    roboflow_rows: list[dict[str, str]] = []

    if args.rf_workspace or args.rf_project:
        if not (args.rf_workspace and args.rf_project and args.rf_key):
            raise ValueError(
                "Para usar Roboflow debes indicar --rf-workspace, --rf-project y --rf-key "
                "(o variable ROBOFLOW_API_KEY)."
            )
        roboflow_rows = fetch_roboflow_dataset_images(
            api_key=args.rf_key,
            workspace=args.rf_workspace,
            project=args.rf_project,
        )
        used_stems.update(
            normalize_filename_stem(row["filename"])
            for row in roboflow_rows
            if row.get("filename")
        )

    candidates = []
    excluded_used = 0
    excluded_camera = 0

    for image in source_images:
        camera = parse_camera(image)
        if camera_filter and camera not in camera_filter:
            excluded_camera += 1
            continue
        if normalize_stem(image) in used_stems:
            excluded_used += 1
            continue
        candidates.append((parse_capture_time(image), camera, image.name.casefold(), image))

    candidates.sort()
    if args.balance_by_camera:
        selected = select_balanced_candidates(candidates, args.limit)
    else:
        selected = candidates[: args.limit]

    print("=" * 72)
    print("SELECCION DE IMAGENES NUEVAS")
    print("=" * 72)
    print(f"Fuente:                   {args.source_root}")
    if args.used_root:
        print("Raices usadas:")
        for used_root in args.used_root:
            print(f"  - {used_root}")
    if roboflow_rows:
        print(f"Roboflow actual:          {args.rf_workspace}/{args.rf_project}")
        print(f"Imagenes en Roboflow:     {len(roboflow_rows)}")
    print(f"Imagenes en fuente:        {len(source_images)}")
    print(f"Stems ya usados:           {len(used_stems)}")
    print(f"Excluidas por usadas:      {excluded_used}")
    if camera_filter:
        print(f"Filtro camara:             {', '.join(sorted(camera_filter))}")
        print(f"Excluidas por camara:      {excluded_camera}")
    print(f"Candidatas nuevas:         {len(candidates)}")
    print(f"Seleccionadas:             {len(selected)}")
    print(f"Balance por camara:        {'si' if args.balance_by_camera else 'no'}")
    print(f"Salida:                    {output_root}")

    if selected:
        selected_cameras = Counter(camera for _, camera, _, _ in selected)
        selected_dates = Counter(captured_at.date().isoformat() for captured_at, _, _, _ in selected)
        print("Camaras seleccionadas:")
        for camera, count in sorted(selected_cameras.items()):
            print(f"  - {camera}: {count}")
        print("Fechas seleccionadas:")
        for date, count in sorted(selected_dates.items()):
            print(f"  - {date}: {count}")

    output_root.mkdir(parents=True, exist_ok=True)
    if roboflow_rows:
        roboflow_manifest_path = output_root / "roboflow_current_dataset.csv"
        with roboflow_manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "filename", "split", "source_found", "source_path"])
            matched = 0
            missing = 0
            for row in roboflow_rows:
                filename = row["filename"]
                source_path = source_by_stem.get(normalize_filename_stem(filename))
                source_found = source_path is not None
                matched += int(source_found)
                missing += int(not source_found)
                writer.writerow(
                    [
                        row["id"],
                        filename,
                        row["split"],
                        "yes" if source_found else "no",
                        str(source_path) if source_path else "",
                    ]
                )
        print(f"Roboflow CSV:             {roboflow_manifest_path}")
        print(f"Roboflow encontrados en fuente: {matched}")
        print(f"Roboflow faltantes en fuente:   {missing}")

    if args.dry_run:
        print("Dry run: no se copiaron imagenes.")
        return

    selected_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_name",
                "camera",
                "captured_at",
                "source_path",
                "selected_path",
                "size_bytes",
            ]
        )

        for captured_at, camera, _, source_path in selected:
            target_path = selected_dir / source_path.name
            if target_path.exists():
                raise FileExistsError(f"Nombre duplicado en seleccion: {target_path}")
            shutil.copy2(source_path, target_path)
            writer.writerow(
                [
                    source_path.name,
                    camera,
                    captured_at.isoformat(sep=" "),
                    str(source_path),
                    str(target_path),
                    source_path.stat().st_size,
                ]
            )

    print(f"Imagenes copiadas en:      {selected_dir}")
    print(f"Manifest:                  {manifest_path}")


if __name__ == "__main__":
    main()
