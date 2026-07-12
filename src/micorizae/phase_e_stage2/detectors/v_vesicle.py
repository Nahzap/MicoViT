"""Detector V — ATLAS LoG → contorno cerrado (única técnica de vesículas).

Único dueño de ``filter_vesicle_mask_spherical`` vía ``atlas_log``.
Ningún otro módulo debe re-filtrar esfericidad/circularidad de V.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..atlas_log import detect_vesicle_blobs_atlas, vesicle_prior_from_mask
from .types import DetectResult

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams

TECHNIQUE = "atlas_log_closed_contour"
TECHNIQUE_VERSION = 1

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
