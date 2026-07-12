"""Detector A — Arb-score Gallaud (única técnica de arbúsculos)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from ..arbuscule_score import compute_arbuscule_score
from .morph_utils import remove_small_components
from .types import DetectResult

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams

TECHNIQUE = "arb_score_gallaud"
TECHNIQUE_VERSION = 1


def detect_arbuscules(
    density: np.ndarray,
    root: np.ndarray,
    params: "WeakSegParams",
    *,
    hyphae: Optional[np.ndarray] = None,
    vesicle: Optional[np.ndarray] = None,
    thr: float = 0.35,
) -> DetectResult:
    """Arbúsculos: un solo forward de arb-score (mask + score continuo)."""
    dens = density.astype(np.float32)
    root_b = root.astype(bool)
    hyph = np.zeros_like(root_b) if hyphae is None else np.asarray(hyphae, dtype=bool)
    ves = np.zeros_like(root_b) if vesicle is None else np.asarray(vesicle, dtype=bool)
    try:
        score, arb = compute_arbuscule_score(
            dens, root_b, hyphae=hyph, vesicle=ves, params=params, thr=float(thr)
        )
        mask = arb & root_b
        return DetectResult(
            mask=mask,
            score=score.astype(np.float32),
            meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION},
        )
    except Exception:
        low = gaussian_filter(dens, sigma=3.0)
        fine = np.abs(dens - low)
        fine = gaussian_filter(fine, sigma=1.0)
        valid = root_b & ~hyph & ~ves
        if int(valid.sum()) == 0:
            z = np.zeros_like(root_b, dtype=bool)
            return DetectResult(
                mask=z,
                score=np.zeros(root_b.shape, dtype=np.float32),
                meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION, "fallback": True},
            )
        fine_thr = float(np.percentile(fine[valid], params.arb_fine_pctl))
        dens_thr = float(np.percentile(dens[root_b], params.arb_density_pctl))
        arb = (fine >= fine_thr) & (dens >= dens_thr) & valid
        k = np.ones((3, 3), np.uint8)
        arb = cv2.morphologyEx(arb.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        arb = remove_small_components(arb, params.arb_min_area) & root_b
        score = np.zeros(root_b.shape, dtype=np.float32)
        if valid.any():
            fmax = float(fine[valid].max()) + 1e-9
            score[valid] = (fine[valid] / fmax).astype(np.float32)
        return DetectResult(
            mask=arb,
            score=score,
            meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION, "fallback": True},
        )
