"""Detector BG/root — tejido vs fondo (única técnica: maxc + saturación)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .types import DetectResult

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams

TECHNIQUE = "stain_bg_maxc_sat"
TECHNIQUE_VERSION = 1


def detect_root(
    stain_maps: dict[str, np.ndarray],
    params: "WeakSegParams",
) -> DetectResult:
    """Tejido = no fondo blanco. Fondo: maxc alto y baja saturación."""
    bg = (stain_maps["maxc"] > params.stain_bg_maxc) & (stain_maps["sat"] < params.stain_bg_sat)
    tissue = ~bg
    k = np.ones((3, 3), np.uint8)
    t = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_OPEN, k)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    t = binary_fill_holes(t > 0)
    if float(t.mean()) < params.root_min_cov:
        mask = np.ones(t.shape, dtype=bool)
    else:
        mask = t.astype(bool)
    return DetectResult(
        mask=mask,
        score=None,
        meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION},
    )
