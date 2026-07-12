"""Detector H — colonización por tinción residual (única técnica H).

H = tinción presente en root sin firma estructural IH/V/A.
No detecta estructuras; consume máscaras estructurales ya producidas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .types import DetectResult

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams

TECHNIQUE = "stain_residual_colony"
TECHNIQUE_VERSION = 1


def detect_colony(
    stain: np.ndarray,
    root: np.ndarray,
    params: "WeakSegParams",
    *,
    structure: Optional[np.ndarray] = None,
    saturated: Optional[np.ndarray] = None,
) -> DetectResult:
    """H_dense: stain alto (o saturación ambigua) fuera de IH∪V∪A."""
    root_b = root.astype(bool)
    stain_f = stain.astype(np.float32)
    struct = (
        np.zeros_like(root_b)
        if structure is None
        else np.asarray(structure, dtype=bool)
    )
    sat = (
        np.zeros_like(root_b)
        if saturated is None
        else np.asarray(saturated, dtype=bool)
    )
    if root_b.any():
        thr = float(np.percentile(stain_f[root_b], float(params.stain_pctl)))
    else:
        thr = 0.0
    stained = (stain_f > max(thr, 1e-3)) & root_b
    stained = stained | (sat & root_b)
    mask = stained & ~struct & root_b

    score = np.zeros(root_b.shape, dtype=np.float32)
    if root_b.any():
        vals = stain_f[root_b]
        vmax = float(vals.max()) + 1e-9
        score[root_b] = (stain_f[root_b] / vmax).astype(np.float32)
        score[struct] = 0.0

    return DetectResult(
        mask=mask,
        score=score,
        meta={"technique": TECHNIQUE, "version": TECHNIQUE_VERSION},
    )
