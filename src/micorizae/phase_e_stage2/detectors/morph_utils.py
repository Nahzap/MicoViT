"""Utilidades morfológicas compartidas entre detectores (sin lógica de clase)."""

from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Elimina CCs con área < min_area (8-conectado)."""
    if min_area <= 1 or not mask.any():
        return mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= lbl == i
    return keep
