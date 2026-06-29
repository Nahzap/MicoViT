"""Rejillas multi-densidad para am_train (base 252 px)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

TIE_ORDER = ("Background", "Mminus", "Mplus", "Unreadable")


def _tie_break(counts: Counter) -> str:
    if not counts:
        return "Background"
    best_n = max(counts.values())
    for cls in TIE_ORDER:
        if counts.get(cls, 0) == best_n:
            return cls
    return "Background"


def _overlap_area(ax0: int, ay0: int, ax1: int, ay1: int, bx0: int, by0: int, bx1: int, by1: int) -> int:
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def _base_lookup(base_df: pd.DataFrame, base_size: int) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for _, r in base_df.iterrows():
        out[(int(r["row"]), int(r["col"]))] = str(r["stage1"])
    return out


def dense_tiles_from_base(
    base_df: pd.DataFrame,
    *,
    image_width: int,
    image_height: int,
    base_size: int = 252,
    dense_size: int = 189,
    density_tier: str = "dense",
) -> pd.DataFrame:
    if base_df.empty:
        return base_df.iloc[0:0].copy()

    lookup = _base_lookup(base_df, base_size)
    n_cols = int(np.ceil(image_width / dense_size))
    n_rows = int(np.ceil(image_height / dense_size))
    template = base_df.iloc[0].to_dict()

    rows: list[dict] = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx = c * dense_size + dense_size // 2
            cy = r * dense_size + dense_size // 2
            br, bc = cy // base_size, cx // base_size
            stage1 = lookup.get((br, bc))
            if stage1 is None:
                br2, bc2 = cy // base_size, cx // base_size
                stage1 = lookup.get((br2, bc2), "Background")
            x0 = c * dense_size
            y0 = r * dense_size
            row = dict(template)
            row.update(
                {
                    "row": r,
                    "col": c,
                    "x0": x0,
                    "y0": y0,
                    "x1": x0 + dense_size,
                    "y1": y0 + dense_size,
                    "tile_size": dense_size,
                    "density_tier": density_tier,
                    "stage1": stage1,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def coarse_tiles_from_base(
    base_df: pd.DataFrame,
    *,
    image_width: int,
    image_height: int,
    base_size: int = 252,
    coarse_size: int = 336,
    density_tier: str = "coarse",
) -> pd.DataFrame:
    if base_df.empty:
        return base_df.iloc[0:0].copy()

    base = base_df.copy()
    n_cols = int(np.ceil(image_width / coarse_size))
    n_rows = int(np.ceil(image_height / coarse_size))
    template = base.iloc[0].to_dict()

    rows: list[dict] = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * coarse_size
            y0 = r * coarse_size
            x1 = x0 + coarse_size
            y1 = y0 + coarse_size
            r0 = max(0, y0 // base_size)
            r1 = min(int(base["row"].max()), (y1 - 1) // base_size)
            c0 = max(0, x0 // base_size)
            c1 = min(int(base["col"].max()), (x1 - 1) // base_size)
            counts: Counter = Counter()
            sub = base[
                (base["row"] >= r0)
                & (base["row"] <= r1)
                & (base["col"] >= c0)
                & (base["col"] <= c1)
            ]
            for _, bt in sub.iterrows():
                area = _overlap_area(
                    int(bt["x0"]),
                    int(bt["y0"]),
                    int(bt["x1"]),
                    int(bt["y1"]),
                    x0,
                    y0,
                    x1,
                    y1,
                )
                if area > 0:
                    counts[str(bt["stage1"])] += area
            if not counts:
                continue
            row = dict(template)
            row.update(
                {
                    "row": r,
                    "col": c,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "tile_size": coarse_size,
                    "density_tier": density_tier,
                    "stage1": _tie_break(counts),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def expand_multidensity_am_train(
    tiles_df: pd.DataFrame,
    images_df: pd.DataFrame,
    *,
    base_size: int = 252,
    dense_size: int = 189,
    coarse_size: int = 336,
    tiers: Iterable[str] = ("dense", "coarse"),
) -> pd.DataFrame:
    if "density_tier" not in tiles_df.columns:
        tiles_df = tiles_df.copy()
        tiles_df["density_tier"] = "base"

    base_train = tiles_df[
        (tiles_df["subset"] == "am_train")
        & (tiles_df["density_tier"] == "base")
        & (tiles_df["tile_size"].astype(int) == base_size)
    ].copy()
    if base_train.empty:
        return tiles_df

    dims = images_df.set_index("image_path")[["width", "height"]].to_dict("index")
    extra_parts: list[pd.DataFrame] = []
    tier_set = set(tiers)

    for image_path, grp in base_train.groupby("image_path", sort=False):
        d = dims.get(image_path) or dims.get(str(image_path))
        if not d or d.get("width") is None:
            continue
        w, h = int(d["width"]), int(d["height"])
        if "dense" in tier_set:
            extra_parts.append(
                dense_tiles_from_base(
                    grp,
                    image_width=w,
                    image_height=h,
                    base_size=base_size,
                    dense_size=dense_size,
                )
            )
        if "coarse" in tier_set:
            extra_parts.append(
                coarse_tiles_from_base(
                    grp,
                    image_width=w,
                    image_height=h,
                    base_size=base_size,
                    coarse_size=coarse_size,
                )
            )

    if not extra_parts:
        return tiles_df

    extra = pd.concat(extra_parts, ignore_index=True)
    extra["tile_id"] = (
        extra["image_path"].astype(str)
        + "@"
        + extra["density_tier"].astype(str)
        + "@r"
        + extra["row"].astype(str)
        + "c"
        + extra["col"].astype(str)
    )
    return pd.concat([tiles_df, extra], ignore_index=True)
