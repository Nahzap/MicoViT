"""Construye `manifests/tiles_index.{parquet|csv}` enriqueciendo el manifest
de labels con bbox absoluto y un `tile_id` estable.

No crea ninguna salida bitmap: el cache real de tiles se materializa
on-demand desde `tile_cutter.crop_tile`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from ..common.io import read_table, write_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..gate_data_splits import _load_amfinder_tile_sizes, tile_size_for_image

log = get_logger("phase_b_tiling")


def _load_tile_sizes(configs_dir: Path) -> dict:
    """Devuelve {'AM': 252, 'ERM': 126, '__default__': 252}."""
    with open(configs_dir / "datasets.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    tile_cfg = cfg.get("tile", {}) or {}
    sizes = dict(tile_cfg.get("pixel_size_by_lineage", {}) or {})
    sizes["__default__"] = int(tile_cfg.get("default", tile_cfg.get("pixel_size", 252)))
    return sizes


def tile_size_for_lineage(lineage: str, sizes: dict) -> int:
    return int(sizes.get(lineage, sizes.get("__default__", 252)))


def build_tiles_index(
    manifests_dir: Optional[Path] = None,
    tile_size: Optional[int] = None,
    out_dir: Optional[Path] = None,
    *,
    apply_multidensity: bool = False,
    multidensity_tiers: tuple[str, ...] = ("dense", "coarse"),
    base_tile_size: int = 252,
    dense_tile_size: int = 189,
    coarse_tile_size: int = 336,
) -> Path:
    paths = get_paths()
    manifests_dir = manifests_dir or paths.manifests
    out_dir = out_dir or paths.manifests
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _load_tile_sizes(paths.configs)
    if tile_size:
        sizes = {k: int(tile_size) for k in sizes}
    log.info(f"Cargando manifest_labels desde {manifests_dir} (tile_size_by_lineage={sizes})")
    df = read_table(manifests_dir / "manifest_labels")

    needed = {"image_path", "row", "col", "stage1", "stage2", "lineage", "subset", "split"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"manifest_labels no tiene columnas requeridas: {missing}")

    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    amf_sizes = _load_amfinder_tile_sizes(paths.root)
    stems = {}
    if (manifests_dir / "manifest_images.parquet").exists() or (
        manifests_dir / "manifest_images.csv"
    ).exists():
        imgs = read_table(manifests_dir / "manifest_images")
        if "image_path" in imgs.columns and "stem" in imgs.columns:
            stems = dict(zip(imgs["image_path"].astype(str), imgs["stem"].astype(str)))

    def _resolve_tile_size(row: pd.Series) -> int:
        if tile_size is not None:
            return int(tile_size)
        ip = str(row["image_path"])
        stem = stems.get(ip, Path(ip).stem)
        return tile_size_for_image(
            ip, stem, str(row["lineage"]), tile_size_for_lineage(str(row["lineage"]), sizes), amf_sizes
        )

    df["tile_size"] = df.apply(_resolve_tile_size, axis=1).astype(int)
    df["density_tier"] = "base"
    df["x0"] = (df["col"] * df["tile_size"]).astype(int)
    df["y0"] = (df["row"] * df["tile_size"]).astype(int)
    df["x1"] = df["x0"] + df["tile_size"]
    df["y1"] = df["y0"] + df["tile_size"]
    df["tile_id"] = (
        df["image_path"].astype(str)
        + "@"
        + df["density_tier"].astype(str)
        + "@r"
        + df["row"].astype(str)
        + "c"
        + df["col"].astype(str)
    )

    if apply_multidensity:
        from .multi_density import expand_multidensity_am_train

        n_before = len(df)
        images_df = read_table(manifests_dir / "manifest_images")
        df = expand_multidensity_am_train(
            df,
            images_df,
            base_size=base_tile_size,
            dense_size=dense_tile_size,
            coarse_size=coarse_tile_size,
            tiers=multidensity_tiers,
        )
        log.info(f"Multi-densidad: +{len(df) - n_before:,} tiles (tiers={multidensity_tiers})")

    cols_order = [
        "tile_id",
        "image_path",
        "row",
        "col",
        "x0",
        "y0",
        "x1",
        "y1",
        "tile_size",
        "density_tier",
        "stage1",
        "stage2",
        "aux_class",
        "lineage",
        "subset",
        "split",
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]

    out_path = write_table(df, out_dir / "tiles_index")

    by_stage1 = df["stage1"].value_counts().to_dict()
    log.info(f"[green]tiles_index OK[/green] -> {out_path.name} ({len(df)} tiles)")
    log.info(f"  Stage1 globals: {by_stage1}")

    return out_path
