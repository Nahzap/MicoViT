"""Detector V — ATLAS LoG → contorno cerrado (única técnica de vesículas).

Único dueño de detección V (por tile y multi-tile). Sin discos Hough sintéticos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from ..atlas_log import (
    detect_vesicle_blobs_atlas,
    detect_vesicle_masks_atlas_for_tiles,
    vesicle_prior_from_mask,
)
from .types import DetectResult

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams

TECHNIQUE = "atlas_log_closed_contour_multitile"
TECHNIQUE_VERSION = 3

# API pública del módulo V para priors (sin re-detectar)
prior_from_mask = vesicle_prior_from_mask


def detect_vesicles(
    density: np.ndarray,
    root: np.ndarray,
    params: "WeakSegParams",
    *,
    frangi_map: Optional[np.ndarray] = None,
) -> DetectResult:
    """Vesículas: semillas LoG multi-escala → contornos cerrados reales (ATLAS)."""
    dens = density.astype(np.float32)
    root_b = root.astype(bool)
    mask = detect_vesicle_blobs_atlas(
        dens,
        root_b,
        min_sigma=float(params.ves_min_sigma),
        max_sigma=float(params.ves_max_sigma),
        num_sigma=int(params.ves_num_sigma),
        threshold=float(params.ves_blob_thr),
        contrast_min=float(params.ves_contrast_min),
        bg_max=float(params.ves_bg_max),
        min_area=int(params.vesicle_min_area),
        max_radius=int(params.ves_max_radius),
        roundness_min=float(params.ves_roundness_min),
        solidity_min=float(params.ves_solidity_min),
        nms_dist_ratio=float(getattr(params, "ves_nms_dist_ratio", 0.55)),
        reject_tubular=True,
        density_pctl=float(getattr(params, "ves_density_pctl", 76.0)),
        max_instances=int(getattr(params, "ves_max_instances", 20)),
        frangi_map=frangi_map,
    )
    score = vesicle_prior_from_mask(mask, root_b)
    return DetectResult(
        mask=mask.astype(bool),
        score=score,
        meta={
            "technique": TECHNIQUE,
            "version": TECHNIQUE_VERSION,
            "n_px": int(mask.sum()),
        },
    )


def detect_vesicle_masks_for_tiles(
    tiles_hwc: list[np.ndarray],
    rows: list[int],
    cols: list[int],
    tile_size: int,
    *,
    params: Optional["WeakSegParams"] = None,
) -> list[np.ndarray]:
    """V multi-tile (ATLAS sobre mosaico). Reemplaza Hough/giant como fuente de labels."""
    from micorizae.morph_core.params import WeakSegParams
    from micorizae.morph_core.stain import stain_maps

    from .bg_root import detect_root

    weak = params if params is not None else WeakSegParams()
    densities: list[np.ndarray] = []
    roots: list[np.ndarray] = []
    for tile in tiles_hwc:
        m = stain_maps(tile)
        dens = m["density"].astype(np.float32)
        root = detect_root(m, weak).mask.astype(bool)
        if not root.any():
            thr = float(np.percentile(dens, 40.0))
            root = dens >= thr
        densities.append(dens)
        roots.append(root)

    return detect_vesicle_masks_atlas_for_tiles(
        densities,
        roots,
        rows,
        cols,
        tile_size,
        min_sigma=float(weak.ves_min_sigma),
        max_sigma=max(float(weak.ves_max_sigma), 0.45 * float(tile_size)),
        num_sigma=max(int(weak.ves_num_sigma), 14),
        threshold=float(weak.ves_blob_thr),
        contrast_min=float(weak.ves_contrast_min),
        min_area=int(weak.vesicle_min_area),
        roundness_min=float(weak.ves_roundness_min),
        solidity_min=float(weak.ves_solidity_min),
        nms_dist_ratio=float(getattr(weak, "ves_nms_dist_ratio", 0.55)),
        density_pctl=float(getattr(weak, "ves_density_pctl", 76.0)),
        max_instances=int(getattr(weak, "ves_max_instances", 40)),
    )


def apply_v_masks_to_label_and_priors(
    label: np.ndarray,
    prior_e: Optional[np.ndarray],
    prior_v: Optional[np.ndarray],
    v_native: np.ndarray,
    *,
    input_size: int,
    v_idx: int = 2,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Fuerza píxeles V (máscara ATLAS native) sobre label + priors."""
    if v_native is None or not np.any(v_native):
        return label, prior_e, prior_v
    g = v_native.astype(np.uint8)
    if g.shape[0] != input_size or g.shape[1] != input_size:
        g = cv2.resize(g, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    g_b = g.astype(bool)
    out_l = label.copy()
    out_l[g_b] = np.uint8(v_idx)
    out_e, out_pv = prior_e, prior_v
    if prior_e is not None:
        out_e = prior_e.copy()
        ch = out_e[v_idx].astype(np.float32)
        ch[g_b] = np.maximum(ch[g_b], 1.0)
        out_e[v_idx] = ch.astype(prior_e.dtype)
    if prior_v is not None:
        out_pv = prior_v.copy().astype(np.float32)
        out_pv[g_b] = np.maximum(out_pv[g_b], 1.0)
        out_pv = out_pv.astype(prior_v.dtype)
    return out_l, out_e, out_pv
