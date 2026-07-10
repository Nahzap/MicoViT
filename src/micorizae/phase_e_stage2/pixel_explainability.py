"""Artefactos de explicabilidad MEViT — mapas, narrativa, EXPLICABILIDAD_PIXEL.md."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

from ..common.run_outputs import RunOutputs
from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX
from .pixel_explain_metrics import tile_explain_stats
from .pixel_prior_maps import PriorMaps, compute_prior_maps, dominant_class_name


@dataclass
class TileExplainBundle:
    seg_vit: np.ndarray
    seg_weak: np.ndarray
    probs: np.ndarray
    prior: PriorMaps
    attention: Optional[np.ndarray] = None


def render_scalar_heatmap(field: np.ndarray, cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    gray = (np.clip(field, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap)[:, :, ::-1]


def render_prob_map(probs: np.ndarray, class_idx: int) -> np.ndarray:
    ch = probs[class_idx] if probs.ndim == 3 else probs
    return render_scalar_heatmap(ch.astype(np.float32), cv2.COLORMAP_PLASMA)


def render_disagreement_map(pred: np.ndarray, weak: np.ndarray) -> np.ndarray:
    diff = (pred != weak).astype(np.float32)
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[diff > 0] = (255, 60, 60)
    return out


def build_tile_narrative(
  row: int,
  col: int,
  stats: dict[str, float],
  dominant: str,
  pct_colonized: float,
) -> str:
    return (
        f"Tile (r={row}, c={col}): dominante {dominant} ({pct_colonized:.1f}% colonizado). "
        f"Frangi medio={stats.get('prior_IH_mean', 0):.2f}. "
        f"ViT–weak PPA={stats.get('ppa_tile', 0):.2f}. "
        f"Desacuerdo={stats.get('disagreement_pct', 0):.1f}%. "
        f"ECE local={stats.get('ece_tile', 0):.3f}."
    )


def explain_quant_table(
    tiles_df: pd.DataFrame,
    bundles: dict[tuple[int, int], TileExplainBundle],
    tile_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pct_lookup = {}
    if not tile_table.empty and {"row", "col", "pct_colonized"}.issubset(tile_table.columns):
        for rec in tile_table.itertuples(index=False):
            pct_lookup[(int(rec.row), int(rec.col))] = float(rec.pct_colonized)

    for rec in tiles_df.itertuples(index=False):
        key = (int(rec.row), int(rec.col))
        bundle = bundles.get(key)
        if bundle is None:
            continue
        stats = tile_explain_stats(bundle.seg_vit, bundle.seg_weak, bundle.probs, bundle.prior.evidence)
        dom = dominant_class_name(bundle.seg_vit)
        pct = pct_lookup.get(key, 0.0)
        rows.append(
            {
                "row": key[0],
                "col": key[1],
                "image_path": str(getattr(rec, "image_path", "")),
                "stage2_gold": str(getattr(rec, "stage2", "")) if hasattr(rec, "stage2") else "",
                "pred_dominant": dom,
                "narrative": build_tile_narrative(key[0], key[1], stats, dom, pct),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def write_explicabilidad_pixel_md(
    *,
    run: RunOutputs,
    image_stem: str,
    explain_df: pd.DataFrame,
    summary: dict[str, Any],
    methods_doc: str = "Docs/methods/20260707_211212_MEViT_pixel_morph_methods.md",
) -> Path:
    out = run.reports / "EXPLICABILIDAD_PIXEL.md"
    lines = [
        "# Explicabilidad pixel MEViT — L9",
        "",
        f"**Run:** `{run.run_id}`",
        f"**Imagen:** `{image_stem}`",
        "",
        "## Resumen",
        "",
        f"- PPA imagen (media tiles): **{summary.get('ppa_mean', 0):.3f}**",
        f"- ECE imagen (media tiles): **{summary.get('ece_mean', 0):.4f}**",
        f"- Tiles con desacuerdo >10%: **{summary.get('n_high_disagreement', 0)}**",
        "",
        "## Método",
        "",
        f"Descomposición MEViT según `{methods_doc}`: priors Frangi/circularidad/entropía + softmax ViT.",
        "",
        "## Tiles",
        "",
    ]
    if not explain_df.empty:
        for rec in explain_df.itertuples(index=False):
            lines.append(f"- {rec.narrative}")
    else:
        lines.append("- (sin tiles M+ con explicación)")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_tile_explain_maps(
    stem: str,
    maps_dir: Path,
    bundle: TileExplainBundle,
    *,
    export_probs: bool = True,
    export_priors: bool = True,
    attention_layer: Optional[int] = None,
) -> dict[str, str]:
    from .pixel_attention import render_attention_heatmap

    rel: dict[str, str] = {}
    maps_dir.mkdir(parents=True, exist_ok=True)

    if export_priors:
        p = maps_dir / f"{stem}__L9_prior_frangi.png"
        cv2.imwrite(str(p), cv2.cvtColor(render_scalar_heatmap(bundle.prior.frangi), cv2.COLOR_RGB2BGR))
        rel["prior_frangi"] = p.name
        p2 = maps_dir / f"{stem}__L9_prior_circularity.png"
        cv2.imwrite(str(p2), cv2.cvtColor(render_scalar_heatmap(bundle.prior.circularity), cv2.COLOR_RGB2BGR))
        rel["prior_circularity"] = p2.name
        p3 = maps_dir / f"{stem}__L9_prior_entropy.png"
        cv2.imwrite(str(p3), cv2.cvtColor(render_scalar_heatmap(bundle.prior.entropy), cv2.COLOR_RGB2BGR))
        rel["prior_entropy"] = p3.name

    if export_probs:
        for cname in PIXEL_CLASS_NAMES:
            if cname == "BG":
                continue
            idx = PIXEL_CLASS_TO_IDX[cname]
            p = maps_dir / f"{stem}__L9_prob_{cname}.png"
            cv2.imwrite(str(p), cv2.cvtColor(render_prob_map(bundle.probs, idx), cv2.COLOR_RGB2BGR))
            rel[f"prob_{cname}"] = p.name

    disp = maps_dir / f"{stem}__L9_disagreement.png"
    cv2.imwrite(str(disp), cv2.cvtColor(render_disagreement_map(bundle.seg_vit, bundle.seg_weak), cv2.COLOR_RGB2BGR))
    rel["disagreement"] = disp.name

    if bundle.attention is not None:
        layer = attention_layer if attention_layer is not None else "last"
        pa = maps_dir / f"{stem}__L9_dino_attention_L{layer}.png"
        cv2.imwrite(str(pa), cv2.cvtColor(render_attention_heatmap(bundle.attention), cv2.COLOR_RGB2BGR))
        rel["dino_attention"] = pa.name

    return rel


def stitch_float_map_from_tiles(
    image_shape: tuple[int, int],
    tiles_df: pd.DataFrame,
    tile_maps: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Reensambla mapa float H×W desde patches por tile (promedio en solapes)."""
    h, w = image_shape
    acc = np.zeros((h, w), dtype=np.float32)
    wt = np.zeros((h, w), dtype=np.float32)
    for rec in tiles_df.itertuples(index=False):
        key = (int(rec.row), int(rec.col))
        patch = tile_maps.get(key)
        if patch is None:
            continue
        x0, y0 = int(rec.x0), int(rec.y0)
        x1, y1 = int(rec.x1), int(rec.y1)
        ph, pw = y1 - y0, x1 - x0
        if ph <= 0 or pw <= 0:
            continue
        if patch.shape != (ph, pw):
            patch = cv2.resize(patch.astype(np.float32), (pw, ph), interpolation=cv2.INTER_LINEAR)
        region = acc[y0:y1, x0:x1]
        region += patch.astype(np.float32)
        wt[y0:y1, x0:x1] += 1.0
    wt = np.maximum(wt, 1.0)
    return acc / wt


def image_explain_summary(explain_df: pd.DataFrame) -> dict[str, Any]:
    if explain_df.empty:
        return {"ppa_mean": 0.0, "ece_mean": 0.0, "n_high_disagreement": 0}
    return {
        "ppa_mean": float(explain_df["ppa_tile"].mean()),
        "ece_mean": float(explain_df["ece_tile"].mean()),
        "n_high_disagreement": int((explain_df["disagreement_pct"] > 10.0).sum()),
    }


def write_explain_audit_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
