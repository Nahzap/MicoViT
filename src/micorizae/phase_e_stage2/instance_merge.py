"""Fusión de instancias vesiculares post-stitch (unión geométrica de bordes).

SRP: no re-filtra circularidad/esfericidad — eso pertenece solo a ``detectors.v_vesicle``.
Aquí solo se unen fragmentos V separados por bordes de tile (área mínima).
"""

from __future__ import annotations

import cv2
import numpy as np

from .pixel_class_map import PIXEL_CLASS_TO_IDX


def merge_vesicle_instances(
    seg_map: np.ndarray,
    *,
    min_area: int = 30,
    dilate_iters: int = 1,
    circularity_min: float | None = None,  # deprecated; ignorado (SRP)
) -> np.ndarray:
    """Une fragmentos V adyacentes; demote solo por área < min_area → H.

    ``circularity_min`` se acepta por compatibilidad de API pero **no** se usa.
    """
    _ = circularity_min
    v_idx = PIXEL_CLASS_TO_IDX["V"]
    h_idx = PIXEL_CLASS_TO_IDX["H"]
    out = seg_map.copy()
    vmask = (seg_map == v_idx).astype(np.uint8)
    if int(vmask.sum()) < min_area:
        return out

    if dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vmask = cv2.dilate(vmask, k, iterations=int(dilate_iters))

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(vmask, connectivity=8)
    rebuilt = np.zeros_like(vmask)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        rebuilt[labels == i] = 1

    was_v = seg_map == v_idx
    out[was_v & ~rebuilt.astype(bool)] = h_idx
    out[was_v & rebuilt.astype(bool)] = v_idx
    return out
