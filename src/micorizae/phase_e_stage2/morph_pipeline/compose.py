"""Orquestador morfológico — composición SRP-safe (sin re-detectar clases).

Precedencia: V → IH → A → H → BG.
Smooth de costuras no disuelve máscaras estructurales locked (V/IH/A).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from micorizae.morph_core import WeakSegParams, segment_tile
from ..atlas_log import stain_residual
from ..detectors import detect_colony
from ..pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_TO_IDX


def _smooth_non_structure(
    seg: np.ndarray,
    *,
    sigma: float,
    locked: np.ndarray,
) -> np.ndarray:
    """Gaussian soft over classes, then restore locked structural pixels."""
    from scipy.ndimage import gaussian_filter

    probs = np.stack([(seg == i).astype(np.float32) for i in range(NUM_PIXEL_CLASSES)], axis=0)
    for i in range(NUM_PIXEL_CLASSES):
        probs[i] = gaussian_filter(probs[i], sigma=sigma)
    out = np.argmax(probs, axis=0).astype(np.uint8)
    out[locked] = seg[locked]
    return out


def compose_pixel_map(
    tile_rgb: np.ndarray,
    *,
    weak: Optional[WeakSegParams] = None,
    seam_sigma: float = 0.85,
    masks: Optional[dict[str, np.ndarray]] = None,
    ambiguous_to_h_dense: bool = True,
) -> np.ndarray:
    """Compone mapa píxel 5 clases desde máscaras de detectores (sin re-detectar V).

    ``masks`` debe provenir de ``segment_tile`` (detectores únicos). Este
    orquestador solo suppress IH/A, asigna H, precedence y smooth-safe.
    """
    weak = weak or WeakSegParams()
    if masks is None:
        masks = segment_tile(tile_rgb, weak)

    root = masks["root"] > 0
    hyphae = masks["hyphae"] > 0
    vesicle = masks["vesicle"] > 0  # ya filtrado en módulo V (ATLAS)
    arbuscule = masks["arbuscule"] > 0
    saturated = masks.get("ambiguous")
    saturated = np.zeros_like(root) if saturated is None else saturated.astype(bool)

    dens = masks.get("density")
    if dens is None:
        residual = np.zeros_like(root, dtype=np.float32)
    else:
        residual = stain_residual(dens.astype(np.float32), open_radius=15)
    residual_thr = float(np.percentile(residual[root], 70)) if root.any() else 0.05
    structure_ok = residual >= max(residual_thr, 0.02)

    if ambiguous_to_h_dense:
        suppress = saturated & ~structure_ok
        hyphae = hyphae & ~suppress
        arbuscule = arbuscule & ~suppress
    else:
        hyphae = hyphae & ~saturated
        arbuscule = arbuscule & ~saturated

    structure = hyphae | vesicle | arbuscule
    stain = masks.get("stain")
    if stain is None:
        stain = dens if dens is not None else np.zeros_like(root, dtype=np.float32)

    h_res = detect_colony(
        stain,
        root,
        weak,
        structure=structure,
        saturated=saturated if ambiguous_to_h_dense else None,
    )

    seg = np.zeros(root.shape, dtype=np.uint8)  # BG
    seg[h_res.mask] = PIXEL_CLASS_TO_IDX["H"]
    seg[hyphae] = PIXEL_CLASS_TO_IDX["IH"]
    seg[vesicle] = PIXEL_CLASS_TO_IDX["V"]
    seg[arbuscule] = PIXEL_CLASS_TO_IDX["A"]

    locked = structure
    if seam_sigma > 0:
        seg = _smooth_non_structure(seg, sigma=float(seam_sigma), locked=locked)
    # Autoridad V = solo máscara del detector (sin sangrado del smooth)
    v_idx = PIXEL_CLASS_TO_IDX["V"]
    h_idx = PIXEL_CLASS_TO_IDX["H"]
    if (seg == v_idx).any() or vesicle.any():
        seg[seg == v_idx] = h_idx
        seg[vesicle] = v_idx
    return seg
