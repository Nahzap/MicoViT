"""Prepara carpetas amfinder_train / amfinder_external desde el catalogo CNN1.

Uso:
  python tools/prepare_amfinder_split.py
  python tools/prepare_amfinder_split.py --config configs/amfinder_split.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Split AMFinder train vs external_test")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "amfinder_split.yaml",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_stems = set(cfg.get("train_stems") or [])
    catalog_path = ROOT / str(cfg["catalog"])
    train_dir = ROOT / str(cfg["train_dir"])
    external_dir = ROOT / str(cfg["external_dir"])

    meta = json.loads(catalog_path.read_text(encoding="utf-8"))
    pairs = list(meta.get("pairs") or [])
    if not pairs:
        print("Catalogo vacio.", file=sys.stderr)
        return 1

    train_entries: list[dict] = []
    external_entries: list[dict] = []
    tile_sizes: dict[str, int] = {}

    for entry in pairs:
        stem = str(entry["stem"])
        img_src = ROOT / str(entry["image_path"]).replace("\\", "/")
        ann_src = ROOT / str(entry["annotation_path"]).replace("\\", "/")
        tile_edge = int(entry.get("tile_edge", 252))
        tile_sizes[stem] = tile_edge

        if stem in train_stems:
            out_dir = train_dir
            bucket = train_entries
            split = "train"
            subset = "amfinder_train"
        else:
            out_dir = external_dir
            bucket = external_entries
            split = "external_test"
            subset = "amfinder_external"

        img_dst = out_dir / img_src.name
        ann_dst = out_dir / ann_src.name
        img_mode = _link_or_copy(img_src, img_dst)
        ann_mode = _link_or_copy(ann_src, ann_dst)

        rel_img = str(img_dst.relative_to(ROOT / "Data")).replace("\\", "/")
        bucket.append(
            {
                "stem": stem,
                "image_path": rel_img,
                "annotation_path": str(ann_dst.relative_to(ROOT)).replace("\\", "/"),
                "tile_edge": tile_edge,
                "split": split,
                "subset": subset,
                "image_link_mode": img_mode,
                "annotation_link_mode": ann_mode,
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.relative_to(ROOT)).replace("\\", "/"),
        "n_train": len(train_entries),
        "n_external": len(external_entries),
        "train_stems": sorted(train_stems),
        "tile_sizes": tile_sizes,
        "train": train_entries,
        "external": external_entries,
    }
    manifest_path = ROOT / "Data" / "am" / "am" / "amfinder_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Train: {len(train_entries)} imgs -> {train_dir}")
    print(f"External: {len(external_entries)} imgs -> {external_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
