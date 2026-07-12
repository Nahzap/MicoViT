"""Detectores grayscale LEGACY — fuera de la ruta de producción Stage2.

No usar en train/infer. Conservados solo para auditoría histórica / tests
explícitos. La ruta canónica es ``stain_aware=True`` + ``detectors.*``.

SRP Stage2 (20260712): una técnica por clase vive en ``phase_e_stage2.detectors``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import frangi
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

if TYPE_CHECKING:
    from micorizae.morph_core.params import WeakSegParams


def _circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if area <= 0.0 or peri <= 1e-6:
        return 0.0
    return float((4.0 * np.pi * area) / (peri * peri))


def tile_root_mask(gray: np.ndarray, p: "WeakSegParams") -> np.ndarray:
    edges = cv2.Canny(gray, p.canny_low, p.canny_high)
    k = np.ones((p.close_kernel, p.close_kernel), dtype=np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    root = binary_fill_holes(closed > 0)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tissue = otsu_inv > 0
    root = root | tissue
    k2 = np.ones((3, 3), dtype=np.uint8)
    root_u8 = cv2.morphologyEx(root.astype(np.uint8), cv2.MORPH_OPEN, k2)
    root_u8 = cv2.morphologyEx(root_u8, cv2.MORPH_CLOSE, k2)
    root = root_u8 > 0
    cov = float(root.mean())
    if cov > p.root_max_cov:
        tighter = (binary_fill_holes(closed > 0) & tissue).astype(np.uint8)
        tighter = cv2.morphologyEx(tighter, cv2.MORPH_OPEN, k2)
        tighter = cv2.morphologyEx(tighter, cv2.MORPH_CLOSE, k2)
        root = tighter > 0
    elif cov < p.root_min_cov:
        root = tissue
    if float(root.mean()) < p.root_min_cov:
        root = np.ones_like(root, dtype=bool)
    return root


def tile_hyphae_mask(gray: np.ndarray, root_mask: np.ndarray, p: "WeakSegParams") -> np.ndarray:
    img = gray.astype(np.float32) / 255.0
    vessel = frangi(img, sigmas=range(1, 4), black_ridges=False)
    valid = vessel[root_mask]
    if valid.size == 0:
        return np.zeros_like(root_mask, dtype=bool)
    thr = float(np.percentile(valid, p.frangi_pctl))
    return (vessel >= thr) & root_mask


def tile_vesicle_mask(gray: np.ndarray, root_mask: np.ndarray, p: "WeakSegParams") -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(gray, dtype=np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < p.vesicle_min_area:
            continue
        if _circularity(c) >= p.vesicle_circularity_min:
            cv2.drawContours(out, [c], -1, color=255, thickness=-1)
    return (out > 0) & root_mask


def tile_arbuscule_mask(gray: np.ndarray, root_mask: np.ndarray, p: "WeakSegParams") -> np.ndarray:
    ent = entropy(img_as_ubyte(gray / 255.0), disk(p.entropy_radius))
    valid = ent[root_mask]
    if valid.size == 0:
        return np.zeros_like(root_mask, dtype=bool)
    thr = float(np.percentile(valid, p.arbuscule_pctl))
    return (ent >= thr) & root_mask


def segment_tile_grayscale(tile_rgb: np.ndarray, params: "WeakSegParams") -> dict[str, np.ndarray]:
    """Ruta grayscale completa (deprecated — no producción)."""
    warnings.warn(
        "legacy_grayscale.segment_tile_grayscale is deprecated; use stain-aware detectors.",
        DeprecationWarning,
        stacklevel=2,
    )
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    root_mask = tile_root_mask(gray, params)
    return {
        "root": root_mask,
        "hyphae": tile_hyphae_mask(gray, root_mask, params),
        "vesicle": tile_vesicle_mask(gray, root_mask, params),
        "arbuscule": tile_arbuscule_mask(gray, root_mask, params),
    }
