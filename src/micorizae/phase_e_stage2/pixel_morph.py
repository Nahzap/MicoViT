"""Descomposición morfológica píxel en tiles M+ — Fase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ..phase_i_weakseg.pipeline import (
    CLASS_ARBUSCULE,
    CLASS_COLONY,
    CLASS_HYPHAE,
    CLASS_ROOT,
    CLASS_VESICLE,
    WeakSegParams,
    consolidate_with_precedence,
    render_overlay,
    render_segmentation_color,
    segment_tile,
    smooth_seams,
)
from .pixel_class_map import (
    COLONY_CLASSES,
    MORPH_STRUCTURE_CLASSES,
    NUM_PIXEL_CLASSES,
    PIXEL_CLASS_COLORS,
    PIXEL_CLASS_TO_IDX,
    PIXEL_IDX_TO_CLASS,
    PIXEL_IGNORE_INDEX,
)

# weakseg class id → pixel morph id (5 clases; ROOT weak → BG)
_WEAK_TO_PIXEL = {
    0: PIXEL_CLASS_TO_IDX["BG"],
    CLASS_ROOT: PIXEL_CLASS_TO_IDX["BG"],
    CLASS_COLONY: PIXEL_CLASS_TO_IDX["H"],
    CLASS_HYPHAE: PIXEL_CLASS_TO_IDX["IH"],
    CLASS_VESICLE: PIXEL_CLASS_TO_IDX["V"],
    CLASS_ARBUSCULE: PIXEL_CLASS_TO_IDX["A"],
}


@dataclass
class PixelMorphParams:
    weak: WeakSegParams = field(default_factory=WeakSegParams)
    seam_sigma: float = 0.85
    overlay_alpha: float = 0.48


def weak_seg_to_pixel_map(weak_seg: np.ndarray) -> np.ndarray:
    """Mapa weakseg (0..5) → clases píxel Fase 2 (0..5)."""
    out = np.zeros_like(weak_seg, dtype=np.uint8)
    for src, dst in _WEAK_TO_PIXEL.items():
        out[weak_seg == src] = dst
    return out


def segment_tile_pixel_morph(tile_rgb: np.ndarray, params: Optional[PixelMorphParams] = None) -> np.ndarray:
    """Segmenta un tile RGB (H,W,3) uint8 → mapa píxel uint8 (5 clases).

      - BG (0): fondo blanco y tejido sin tinción (no fúngico).
      - IH/V/A: estructuras discretas (hifas, vesículas rellenas, arbúsculos).
      - H: corteza colonizada con tinción (incluye azul denso/saturado, que es
        colonización intensa, NO vesículas ni ruido).

    En regiones de azul saturado se SUPRIMEN los seeds de estructura (blobs/crestas
    espurios) y el área se etiqueta como H — colonización densa, no speckle.
    """
    p = params or PixelMorphParams()
    masks = segment_tile(tile_rgb, p.weak)
    root = masks["root"] > 0
    hyphae = masks["hyphae"] > 0
    vesicle = masks["vesicle"] > 0
    arbuscule = masks["arbuscule"] > 0
    saturated = masks.get("ambiguous", None)
    saturated = np.zeros_like(root) if saturated is None else saturated.astype(bool)

    # Seeds espurios en azul saturado se descartan → esa zona quedará como H.
    hyphae &= ~saturated
    vesicle &= ~saturated
    arbuscule &= ~saturated
    structure = hyphae | vesicle | arbuscule

    if "stain" in masks:
        stain = masks["stain"]
        thr = float(p.weak.stain_pctl)
        root_vals = stain[root] if root.any() else stain.reshape(-1)
        stain_thr = float(np.percentile(root_vals, thr)) if root_vals.size else 0.0
        stained = (stain > max(stain_thr, 1e-3)) & root
    else:
        stained = root
    # el azul saturado es colonización aunque quede bajo el percentil de tinción
    stained = stained | (saturated & root)

    seg = np.zeros(root.shape, dtype=np.uint8)  # BG (fondo + tejido sin tinción)
    seg[stained & ~structure] = PIXEL_CLASS_TO_IDX["H"]  # corteza colonizada (incl. azul denso)
    seg[hyphae] = PIXEL_CLASS_TO_IDX["IH"]               # precedencia A > V > IH
    seg[vesicle] = PIXEL_CLASS_TO_IDX["V"]
    seg[arbuscule] = PIXEL_CLASS_TO_IDX["A"]

    if p.seam_sigma > 0:
        seg = _smooth_pixel_seg(seg, sigma=p.seam_sigma)
    return seg


def _smooth_pixel_seg(seg: np.ndarray, sigma: float) -> np.ndarray:
    probs = np.stack([(seg == i).astype(np.float32) for i in range(NUM_PIXEL_CLASSES)], axis=0)
    from scipy.ndimage import gaussian_filter

    for i in range(NUM_PIXEL_CLASSES):
        probs[i] = gaussian_filter(probs[i], sigma=sigma)
    return np.argmax(probs, axis=0).astype(np.uint8)


def _place_patch(canvas: np.ndarray, patch: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    h, w = canvas.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    ph, pw = y1 - y0, x1 - x0
    if patch.shape[0] != ph or patch.shape[1] != pw:
        patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_NEAREST)
    canvas[y0:y1, x0:x1] = patch


def stitch_pixel_map_from_tiles(
    image_shape: tuple[int, int],
    tiles_df: pd.DataFrame,
    tile_segments: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Reensambla mapa píxel H×W desde segmentos por (row,col)."""
    h, w = image_shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    for rec in tiles_df.itertuples(index=False):
        key = (int(rec.row), int(rec.col))
        if key not in tile_segments:
            continue
        x0, y0 = int(rec.x0), int(rec.y0)
        x1, y1 = int(rec.x1), int(rec.y1)
        _place_patch(canvas, tile_segments[key], x0, y0, x1, y1)
    return canvas


def morph_entropy(seg: np.ndarray, eps: float = 1e-9) -> float:
    """Entropía morfológica normalizada sobre píxeles ROOT+colonias."""
    mask = seg > PIXEL_CLASS_TO_IDX["BG"]
    if not mask.any():
        return 0.0
    sub = seg[mask]
    counts = np.bincount(sub, minlength=NUM_PIXEL_CLASSES).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > eps]
    h = -(p * np.log(p + eps)).sum()
    k = len(MORPH_STRUCTURE_CLASSES)
    return float(h / np.log(k + eps))


def quantize_segment(seg: np.ndarray, tissue_only: bool = True) -> dict[str, float]:
    """Fracciones de área por clase morfológica."""
    if tissue_only:
        valid = seg > PIXEL_CLASS_TO_IDX["BG"]
        denom = int(valid.sum())
        base = seg[valid] if denom > 0 else seg.reshape(-1)
    else:
        denom = seg.size
        base = seg.reshape(-1)
    if denom == 0:
        return {f"pct_{c}": 0.0 for c in MORPH_STRUCTURE_CLASSES + ("colonized",)}

    counts = np.bincount(base, minlength=NUM_PIXEL_CLASSES).astype(np.float64)
    total = float(denom)

    out: dict[str, float] = {}
    for c in PIXEL_CLASS_COLORS:
        if c == "BG":
            continue
        idx = PIXEL_CLASS_TO_IDX[c]
        out[f"pct_{c}"] = float(counts[idx] / total * 100.0)

    colony_px = sum(counts[PIXEL_CLASS_TO_IDX[c]] for c in COLONY_CLASSES)
    out["pct_colonized"] = float(colony_px / total * 100.0)
    out["morph_entropy"] = morph_entropy(seg)
    return out


def quantize_tiles_table(
    tiles_df: pd.DataFrame,
    tile_segments: dict[tuple[int, int], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for rec in tiles_df.itertuples(index=False):
        key = (int(rec.row), int(rec.col))
        seg = tile_segments.get(key)
        if seg is None:
            continue
        q = quantize_segment(seg)
        rows.append(
            {
                "row": key[0],
                "col": key[1],
                "image_path": str(getattr(rec, "image_path", "")),
                "stage2_gold": str(getattr(rec, "stage2", "")) if hasattr(rec, "stage2") else "",
                **q,
            }
        )
    return pd.DataFrame(rows)


def render_pixel_class_map(seg: np.ndarray) -> np.ndarray:
    color = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for name, rgb in PIXEL_CLASS_COLORS.items():
        color[seg == PIXEL_CLASS_TO_IDX[name]] = np.array(rgb, dtype=np.uint8)
    return color


def render_colony_binary(seg: np.ndarray) -> np.ndarray:
    colony = np.isin(seg, [PIXEL_CLASS_TO_IDX[c] for c in COLONY_CLASSES])
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    out[colony] = (0, 200, 255)
    return out


def render_confidence_heatmap(seg: np.ndarray) -> np.ndarray:
    """Entropía local del mapa duro suavizado (no confianza del modelo)."""
    return render_smoothness_heatmap(seg)


def render_smoothness_heatmap(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    counts = np.zeros((NUM_PIXEL_CLASSES, h, w), dtype=np.float32)
    for c in range(NUM_PIXEL_CLASSES):
        counts[c] = (seg == c).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    k = 5
    smooth = np.stack([gaussian_filter(counts[c], sigma=1.2) for c in range(NUM_PIXEL_CLASSES)], axis=0)
    smooth = smooth / (smooth.sum(axis=0, keepdims=True) + 1e-9)
    ent = -(smooth * np.log(smooth + 1e-9)).sum(axis=0)
    conf = 1.0 - ent / np.log(NUM_PIXEL_CLASSES)
    gray = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def render_diagnostic_overlay(image_rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = render_pixel_class_map(seg)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, color, alpha, 0.0)


def load_tile_rgb_from_image(image_rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return image_rgb[y0:y1, x0:x1].copy()
