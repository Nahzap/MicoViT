"""Buckets de dominio/escala para tiles gate AM (subcentros domain-aware)."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DOMAIN_BUCKETS: tuple[str, ...] = (
    "am_native",
    "amfinder_40",
    "amfinder_126",
    "amfinder_256",
)

_AM_NATIVE_SUBSETS = frozenset({"am_train", "am_test"})


def tile_edge_from_row(row: pd.Series) -> int:
    if "tile_edge" in row.index and pd.notna(row.get("tile_edge")):
        return int(row["tile_edge"])
    if "tile_size" in row.index and pd.notna(row.get("tile_size")):
        return int(row["tile_size"])
    return 252


def domain_bucket_for_row(row: pd.Series) -> str:
    """Asigna bucket de dominio a un tile del manifest."""
    subset = str(row.get("subset", "")).strip()
    if subset in _AM_NATIVE_SUBSETS:
        return "am_native"
    edge = tile_edge_from_row(row)
    if edge <= 45:
        return "amfinder_40"
    if edge <= 180:
        return "amfinder_126"
    return "amfinder_256"


def attach_domain_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas `tile_edge` y `domain_bucket` si faltan."""
    out = df.copy()
    if "tile_edge" not in out.columns and "tile_size" in out.columns:
        out["tile_edge"] = out["tile_size"].astype(int)
    if "domain_bucket" not in out.columns:
        out["domain_bucket"] = out.apply(domain_bucket_for_row, axis=1)
    return out


def domain_buckets_from_batch(batch: object) -> list[str] | None:
    """Lee dominios por tile desde GPUImageBatch (si están disponibles)."""
    domains = getattr(batch, "domain_buckets", None)
    if domains is None:
        return None
    return list(domains)


def summarize_domain_metrics(
    assignments: pd.DataFrame,
    *,
    label_col: str = "gold_class",
) -> list[dict]:
    """Recall por clase y dominio para reportes."""
    if assignments.empty or "domain_bucket" not in assignments.columns:
        return []
    rows: list[dict] = []
    for (dom, cls), grp in assignments.groupby(["domain_bucket", label_col], dropna=False):
        rows.append(
            {
                "domain_bucket": str(dom),
                "class": str(cls),
                "n_tiles": int(len(grp)),
                "recall": float(grp["correct"].mean()) if len(grp) else 0.0,
                "margin_mean": float(grp["subcenter_margin"].mean()) if len(grp) else 0.0,
            }
        )
    return rows


def merge_tile_metadata(
    assignments: pd.DataFrame,
    tiles_df: pd.DataFrame,
    *,
    extra_cols: Iterable[str] = ("domain_bucket", "subset", "tile_edge", "tile_size"),
) -> pd.DataFrame:
    """Enriquece asignaciones con metadatos del manifest."""
    cols = ["image_path", "row", "col"]
    avail = [c for c in extra_cols if c in tiles_df.columns]
    if not avail:
        return assignments
    meta = tiles_df[cols + avail].drop_duplicates(subset=cols)
    out = assignments.merge(meta, on=cols, how="left")
    if "domain_bucket" not in out.columns and "tile_edge" in out.columns:
        out["domain_bucket"] = out.apply(domain_bucket_for_row, axis=1)
    return out
