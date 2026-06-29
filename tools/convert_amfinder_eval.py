"""Convierte AMFinder CNN1 (Zenodo 5118948) al formato CSV de MicorizaeVision.

Solo evaluacion: Y/N/X -> AMColonised/Uncolonised/Background (+ DSE/Hybrid/Unreadable=0).
No modifica entrenamiento ni manifests productivos.

Salida (por defecto):
  Data/am/am/amfinder_eval/
    <stem>.jpg                              (hardlink o copia desde AM-train)
    <stem>_2021-07-22T00_00_00.000000_cnn_1_annotations.csv
  Data/am/am/amfinder_eval/amfinder_eval_catalog.json

Uso:
  python tools/convert_amfinder_eval.py
  python tools/convert_amfinder_eval.py --source Data/AM-train --limit 5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CSV_HEADER = [
    "row",
    "col",
    "AMColonised",
    "Uncolonised",
    "Background",
    "Unreadable",
    "DSE",
    "Hybrid",
]
ANNOTATION_TS = "2021-07-22T00_00_00.000000"
YNX_TO_COL = {
    "Y": "AMColonised",
    "N": "Uncolonised",
    "X": "Background",
}


def _parse_settings(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # algunos settings.json usan comillas simples
        text = text.replace("'", '"')
        return json.loads(text)


def _is_active(val: object) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    if s in ("1", "1.0", "True", "true"):
        return True
    try:
        return float(s) >= 0.5
    except ValueError:
        return False


def _row_to_amfinder_csv(row: dict[str, str]) -> dict[str, int]:
    out = {c: 0 for c in CSV_HEADER[2:]}
    active = [k for k in "YNX" if _is_active(row.get(k))]
    if len(active) == 1:
        out[YNX_TO_COL[active[0]]] = 1
    elif len(active) > 1:
        # prioridad Background > N > Y (fondo domina en bordes)
        for k in ("X", "N", "Y"):
            if k in active:
                out[YNX_TO_COL[k]] = 1
                break
    else:
        out["Unreadable"] = 1
    return {
        "row": int(row["row"]),
        "col": int(row["col"]),
        **out,
    }


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def convert_pair(
    jpg: Path,
    zip_path: Path,
    out_dir: Path,
    *,
    link_images: bool = True,
) -> dict:
    stem = jpg.stem
    with zipfile.ZipFile(zip_path) as zf:
        if "col.tsv" not in zf.namelist():
            raise ValueError(f"sin col.tsv en {zip_path.name}")
        settings = _parse_settings(zf.read("settings.json")) if "settings.json" in zf.namelist() else {}
        tile_edge = int(settings.get("tile_edge", 0))
        rows = list(csv.DictReader(io.StringIO(zf.read("col.tsv").decode("utf-8")), delimiter="\t"))

    converted = [_row_to_amfinder_csv(r) for r in rows]
    counts = {k: 0 for k in ("AMColonised", "Uncolonised", "Background", "Unreadable")}
    for r in converted:
        for col in counts:
            if r[col]:
                counts[col] += 1

    csv_name = f"{stem}_{ANNOTATION_TS}_cnn_1_annotations.csv"
    csv_path = out_dir / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(converted)

    dst_jpg = out_dir / jpg.name
    img_mode = "skip"
    if link_images and jpg.exists():
        img_mode = _link_or_copy(jpg, dst_jpg)

    rel_image = dst_jpg.relative_to(ROOT).as_posix()
    return {
        "stem": stem,
        "source_jpg": jpg.relative_to(ROOT).as_posix(),
        "source_zip": zip_path.relative_to(ROOT).as_posix(),
        "image_path": rel_image,
        "annotation_path": csv_path.relative_to(ROOT).as_posix(),
        "tile_edge": tile_edge,
        "n_tiles": len(converted),
        "counts": counts,
        "image_link_mode": img_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AMFinder CNN1 -> formato CSV MicorizaeVision (eval)")
    ap.add_argument("--source", type=Path, default=ROOT / "Data" / "AM-train")
    ap.add_argument("--out", type=Path, default=ROOT / "Data" / "am" / "am" / "amfinder_eval")
    ap.add_argument("--limit", type=int, default=0, help="0 = todos los pares CNN1")
    ap.add_argument("--copy-images", action="store_true", help="copiar JPG (default: hardlink)")
    args = ap.parse_args()

    source = args.source.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(source.glob("CNN1_*.jpg"))
    if args.limit > 0:
        jpgs = jpgs[: args.limit]

    catalog: list[dict] = []
    errors: list[dict] = []
    for jpg in jpgs:
        zp = source / f"{jpg.stem}.zip"
        if not zp.exists():
            errors.append({"jpg": str(jpg), "error": "zip faltante"})
            continue
        try:
            entry = convert_pair(
                jpg,
                zp,
                out_dir,
                link_images=not args.copy_images,
            )
            catalog.append(entry)
            print(
                f"OK {jpg.name}  tiles={entry['n_tiles']}  edge={entry['tile_edge']}  "
                f"M+={entry['counts']['AMColonised']} M-={entry['counts']['Uncolonised']} "
                f"Bg={entry['counts']['Background']}"
            )
        except Exception as e:  # noqa: BLE001
            errors.append({"jpg": str(jpg), "error": str(e)})
            print(f"FAIL {jpg.name}: {e}")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.relative_to(ROOT)),
        "out_dir": str(out_dir.relative_to(ROOT)),
        "mapping": {"Y": "AMColonised/Mplus", "N": "Uncolonised/Mminus", "X": "Background"},
        "n_ok": len(catalog),
        "n_errors": len(errors),
        "pairs": catalog,
        "errors": errors,
    }
    catalog_path = out_dir / "amfinder_eval_catalog.json"
    catalog_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nCatalogo -> {catalog_path}")
    print(f"Convertidos: {len(catalog)}/{len(jpgs)}  errores: {len(errors)}")


if __name__ == "__main__":
    main()
