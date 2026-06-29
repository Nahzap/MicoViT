"""Auditoria espacial automatica de etiquetas (sin revisar todas las imagenes a mano).

Comprueba en batch:
- tiles fuera de los limites de la imagen (row/col vs width/height)
- duplicados (image_path, row, col)
- tile_size inconsistente por imagen
- muestra aleatoria de N imagenes con overlay L0+L2 (solo las problematicas o sample)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..common.io import read_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths

log = get_logger("phase_d.gate_audit")


def _load_image_dims() -> pd.DataFrame:
    paths = get_paths()
    try:
        img = read_table(paths.manifests / "manifest_images")
    except FileNotFoundError:
        return pd.DataFrame(columns=["image_path", "width", "height"])
    cols = ["image_path", "width", "height"]
    for c in cols:
        if c not in img.columns:
            return pd.DataFrame(columns=cols)
    return img[cols].drop_duplicates("image_path")


def audit_tiles_geometry(tiles_df: pd.DataFrame) -> dict[str, Any]:
    """Auditoria numerica sobre todo el manifest (rapida, sin abrir JPEGs)."""
    df = tiles_df.copy()
    required = {"image_path", "row", "col", "tile_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tiles_df sin columnas {missing}")

    if "x0" not in df.columns:
        df["x0"] = df["col"].astype(int) * df["tile_size"].astype(int)
        df["y0"] = df["row"].astype(int) * df["tile_size"].astype(int)
        df["x1"] = df["x0"] + df["tile_size"].astype(int)
        df["y1"] = df["y0"] + df["tile_size"].astype(int)

    dup = df.duplicated(subset=["image_path", "row", "col"], keep=False)
    n_dup = int(dup.sum())

    ts_by_img = df.groupby("image_path")["tile_size"].nunique()
    inconsistent_ts = ts_by_img[ts_by_img > 1].index.tolist()

    img_dims = _load_image_dims()
    oob_rows: list[dict] = []
    if not img_dims.empty:
        merged = df.merge(img_dims, on="image_path", how="left")
        has_dims = merged["width"].notna() & merged["height"].notna()
        sub = merged[has_dims]
        oob = (
            (sub["x0"] < 0)
            | (sub["y0"] < 0)
            | (sub["x1"] > sub["width"].astype(int))
            | (sub["y1"] > sub["height"].astype(int))
        )
        if oob.any():
            bad = sub.loc[oob, ["image_path", "row", "col", "x0", "y0", "x1", "y1", "width", "height"]]
            for rel, grp in bad.groupby("image_path"):
                oob_rows.append({"image_path": rel, "n_oob_tiles": int(len(grp))})
    else:
        log.warning("[Gate audit] manifest_images sin width/height; omitiendo chequeo OOB")

    n_oob = sum(r["n_oob_tiles"] for r in oob_rows)
    n_images = df["image_path"].nunique()
    pass_ok = n_dup == 0 and n_oob == 0 and not inconsistent_ts

    report = {
        "pass": pass_ok,
        "n_tiles": int(len(df)),
        "n_images": int(n_images),
        "n_duplicate_tiles": n_dup,
        "n_oob_tiles": n_oob,
        "images_with_oob": oob_rows[:50],
        "images_inconsistent_tile_size": inconsistent_ts[:20],
        "stage1_counts": df["stage1"].value_counts().to_dict() if "stage1" in df.columns else {},
    }
    return report


def render_audit_overlays(
    tiles_df: pd.DataFrame,
    *,
    image_paths: list[str],
    out_dir: Path,
    downscale: int = 4,
) -> list[Path]:
    """Genera L0+L2 solo para las imagenes indicadas (auditoria visual acotada)."""
    from PIL import Image

    from ..layers import LayerContext, compose, downscale_context, save_png

    Image.MAX_IMAGE_PIXELS = None
    paths = get_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for rel in image_paths:
        sub = tiles_df[tiles_df["image_path"] == rel]
        if sub.empty:
            continue
        full = paths.root / rel
        if not full.exists():
            log.warning(f"[Gate audit] imagen no encontrada: {full}")
            continue
        ts = int(sub["tile_size"].iloc[0]) if "tile_size" in sub.columns else 252
        img = np.array(Image.open(full).convert("RGB"))
        ctx = LayerContext(image=img, tile_size=ts, tiles=sub[["row", "col", "stage1"]])
        ctx = downscale_context(ctx, downscale)
        composed = compose(["L0", "L1", "L2"], ctx, alphas=[1.0, 0.85, 0.55])
        stem = Path(rel).stem
        p = save_png(composed, out_dir / f"{stem}__audit_L0_L1_L2.png")
        saved.append(p)
    return saved


def pick_audit_images(report: dict[str, Any], tiles_df: pd.DataFrame, *, sample_n: int, seed: int = 0) -> list[str]:
    """Prioriza imagenes con OOB; completa con muestra aleatoria."""
    priority = [r["image_path"] for r in report.get("images_with_oob", [])]
    rng = np.random.default_rng(seed)
    all_imgs = tiles_df["image_path"].unique().tolist()
    rng.shuffle(all_imgs)
    chosen: list[str] = []
    for p in priority:
        if p not in chosen:
            chosen.append(p)
    for p in all_imgs:
        if len(chosen) >= sample_n:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen[:sample_n]


def run_gate_spatial_audit(
    tiles_df: pd.DataFrame,
    *,
    out_dir: Path,
    sample_n: int = 5,
    downscale: int = 4,
) -> dict[str, Any]:
    """Ejecuta auditoria completa y escribe JSON + PNGs de muestra."""
    report = audit_tiles_geometry(tiles_df)
    sample_imgs = pick_audit_images(report, tiles_df, sample_n=sample_n)
    overlays = render_audit_overlays(
        tiles_df, image_paths=sample_imgs, out_dir=out_dir / "sample_overlays", downscale=downscale
    )
    report["sample_images"] = sample_imgs
    report["overlay_paths"] = [str(p.relative_to(out_dir)) for p in overlays]

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "spatial_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    status = "PASS" if report["pass"] else "FAIL"
    log.info(
        f"[Gate audit] {status} | tiles={report['n_tiles']:,} imgs={report['n_images']} "
        f"dup={report['n_duplicate_tiles']} oob={report['n_oob_tiles']} | "
        f"muestra visual={len(overlays)} PNG -> {out_dir}"
    )
    if not report["pass"]:
        log.warning(
            "[Gate audit] Hay problemas geometricos en el manifest. "
            "Corrige Fase A/B antes de confiar en entrenamiento."
        )
    return report
