"""Detector IH — Frangi × top-hat (única técnica de hifas)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from skimage.filters import frangi

from .morph_utils import remove_small_components
from .types import DetectResult

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams

TECHNIQUE = "frangi_x_tophat"
TECHNIQUE_VERSION = 1


def detect_hyphae(
    density: np.ndarray,
    root: np.ndarray,
    params: "WeakSegParams",
    *,
    exclude: Optional[np.ndarray] = None,
    frangi_vessel: Optional[np.ndarray] = None,
) -> DetectResult:
    """Hifas: vesselness Frangi × prominencia top-hat (piso absoluto anti-speckle)."""
    dens = density.astype(np.float32)
    root_b = root.astype(bool)
    if frangi_vessel is not None:
        vessel = frangi_vessel.astype(np.float32)
    else:
        vessel = frangi(dens, sigmas=list(params.ih_frangi_sigmas), black_ridges=False).astype(
            np.float32
        )
    d8 = (np.clip(dens, 0, 1) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.ih_tophat_disk, params.ih_tophat_disk)
    )
    tophat = cv2.morphologyEx(d8, cv2.MORPH_TOPHAT, k).astype(np.float32) / 255.0
    valid = vessel[root_b]
    if valid.size == 0:
        z = np.zeros_like(root_b, dtype=bool)
        return DetectResult(
            mask=z,
            score=np.zeros(root_b.shape, dtype=np.float32),
            meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION},
        )
    thr = float(np.percentile(valid, params.frangi_pctl))
    prom_thr = max(
        params.ih_tophat_abs, float(np.percentile(tophat[root_b], params.ih_prom_pctl))
    )
    mask = (vessel >= thr) & (tophat >= prom_thr) & root_b
    if exclude is not None:
        mask = mask & ~np.asarray(exclude, dtype=bool)
    mask = remove_small_components(mask, params.ih_min_area) & root_b

    score = np.zeros(root_b.shape, dtype=np.float32)
    if root_b.any():
        vn = vessel.copy()
        tn = tophat.copy()
        vmax = float(vn[root_b].max()) + 1e-9
        tmax = float(tn[root_b].max()) + 1e-9
        score[root_b] = (vn[root_b] / vmax) * (tn[root_b] / tmax)
        smax = float(score[root_b].max()) + 1e-9
        score[root_b] = score[root_b] / smax

    return DetectResult(
        mask=mask,
        score=score,
        meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION},
    )
