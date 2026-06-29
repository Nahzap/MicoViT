"""Splits de datos para gate AM v4 (multi-densidad + AMFinder)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .common.io import read_table
from .common.paths import get_paths


def _load_amfinder_tile_sizes(root: Path) -> dict[str, int]:
    manifest = root / "Data" / "am" / "am" / "amfinder_split_manifest.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in (data.get("tile_sizes") or {}).items()}


def tile_size_for_image(image_path: str, stem: str, lineage: str, default: int, amf_sizes: dict) -> int:
    name = Path(image_path).stem
    if name in amf_sizes:
        return amf_sizes[name]
    if stem in amf_sizes:
        return amf_sizes[stem]
    return default


def split_gate_am_v4(
    tiles_index_path: Optional[Path] = None,
    *,
    lineages: Iterable[str] = ("AM",),
    exclude_unreadable: bool = True,
    train_density_tiers: Iterable[str] | None = ("base",),
    val_density_tier: str = "base",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Train (nativo multi-densidad + amfinder_train), val (am_test base), external (amfinder_external)."""
    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))

    if lineages:
        df = df[df["lineage"].isin(list(lineages))].copy()
    if exclude_unreadable:
        df = df[df["stage1"] != "Unreadable"].copy()
    if "density_tier" not in df.columns:
        df["density_tier"] = "base"
    if train_density_tiers is not None:
        allowed_tiers = {str(t) for t in train_density_tiers}
        df_train_pool = df[df["density_tier"].astype(str).isin(allowed_tiers)].copy()
    else:
        df_train_pool = df

    train_df = df_train_pool[
        (df_train_pool["subset"].isin(["am_train", "amfinder_train"]))
        & (df_train_pool["split"].isin(["train"]))
    ].reset_index(drop=True)

    val_df = df[
        (df["subset"] == "am_test")
        & (df["split"] == "test")
        & (df["density_tier"].astype(str) == val_density_tier)
    ].reset_index(drop=True)

    external_df = df[
        (df["subset"] == "amfinder_external") & (df["split"] == "external_test")
    ].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError(f"Split v4 vacio: train={len(train_df)} val={len(val_df)}")

    from .gate_domain_buckets import attach_domain_buckets

    train_df = attach_domain_buckets(train_df)
    val_df = attach_domain_buckets(val_df)
    if len(external_df):
        external_df = attach_domain_buckets(external_df)

    info = {
        "split_mode": "gate_v4",
        "n_train_images": int(train_df["image_path"].nunique()),
        "n_val_images": int(val_df["image_path"].nunique()),
        "n_external_images": int(external_df["image_path"].nunique()) if len(external_df) else 0,
        "n_train_tiles": int(len(train_df)),
        "n_val_tiles": int(len(val_df)),
        "n_external_tiles": int(len(external_df)),
        "train_pos": int((train_df["stage1"] == "Mplus").sum()),
        "val_pos": int((val_df["stage1"] == "Mplus").sum()),
        "train_mminus": int((train_df["stage1"] == "Mminus").sum()),
        "val_mminus": int((val_df["stage1"] == "Mminus").sum()),
        "train_bg": int((train_df["stage1"] == "Background").sum()),
        "val_bg": int((val_df["stage1"] == "Background").sum()),
        "train_by_subset": train_df.groupby("subset").size().to_dict(),
        "train_by_density": train_df.groupby("density_tier").size().to_dict(),
        "train_by_domain": train_df.groupby("domain_bucket").size().to_dict(),
    }
    return train_df, val_df, external_df, info


def cache_tiles_for_gate_v4(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Union para compilar cache DINO (train completo + val nativo base)."""
    return (
        pd.concat([train_df, val_df], ignore_index=True)
        .drop_duplicates(subset=["image_path", "row", "col", "tile_size"], keep="first")
        .reset_index(drop=True)
    )
