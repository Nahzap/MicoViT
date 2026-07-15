"""Pack CPU label+priors por tile — ProcessPool worker sin torch.

Usado en build HDF5 Stage2-Pixel para evitar GIL (ThreadPool no escala en morph).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from micorizae.morph_core import segment_tile
from .pixel_morph import PixelMorphParams, segment_tile_pixel_morph
from .pixel_prior_maps import compute_prior_training_targets

_g_morph: Optional[PixelMorphParams] = None
_g_input_size: int = 224
_g_with_priors: bool = False


def _pin_blas_single_thread() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def init_worker(morph_params: PixelMorphParams, input_size: int, with_priors: bool) -> None:
    global _g_morph, _g_input_size, _g_with_priors
    _g_morph = morph_params
    _g_input_size = int(input_size)
    _g_with_priors = bool(with_priors)
    _pin_blas_single_thread()


def _resize_label(seg: np.ndarray, target_size: int) -> np.ndarray:
    if seg.shape[0] == target_size and seg.shape[1] == target_size:
        return seg.astype(np.uint8)
    return cv2.resize(seg, (target_size, target_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _resize_prior_maps(
    ev: np.ndarray, ves: np.ndarray, input_size: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(ev.shape[-2]), int(ev.shape[-1])
    if h == input_size and w == input_size:
        return ev, ves
    ev_out = np.stack(
        [
            cv2.resize(ev[c], (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            for c in range(ev.shape[0])
        ],
        axis=0,
    )
    ves_out = cv2.resize(ves, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return ev_out, ves_out


def pack_one(tile_hwc_u8: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Un tile HWC uint8 → (label, prior_evidence?, prior_vesicle?)."""
    if _g_morph is None:
        raise RuntimeError("stage2_h5_cpu_pack: worker no inicializado")
    masks = segment_tile(tile_hwc_u8, _g_morph.weak)
    label = _resize_label(segment_tile_pixel_morph(tile_hwc_u8, _g_morph, masks=masks), _g_input_size)
    if not _g_with_priors:
        return label, None, None
    ev, ves = compute_prior_training_targets(tile_hwc_u8, _g_morph, masks=masks)
    ev, ves = _resize_prior_maps(ev, ves, _g_input_size)
    return label, ev.astype(np.float16), ves.astype(np.float16)


def pack_batch(tiles_hwc: list[np.ndarray]) -> list[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    return [pack_one(t) for t in tiles_hwc]


def apply_v_overlays(
    packs: list[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
    v_masks_native: list[np.ndarray],
    *,
    input_size: Optional[int] = None,
) -> list[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Aplica máscaras V ATLAS (native) sobre packs ya etiquetados."""
    from .detectors.v_vesicle import apply_v_masks_to_label_and_priors
    from .pixel_class_map import PIXEL_CLASS_TO_IDX

    if not v_masks_native:
        return packs
    size = int(input_size if input_size is not None else (_g_input_size or 224))
    v_idx = PIXEL_CLASS_TO_IDX["V"]
    out = []
    for pack, vmask in zip(packs, v_masks_native):
        lab, ev, ves = pack
        lab2, ev2, ves2 = apply_v_masks_to_label_and_priors(
            lab, ev, ves, vmask, input_size=size, v_idx=v_idx
        )
        out.append((lab2, ev2, ves2))
    return out


def apply_giant_overlays(
    packs: list[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
    giant_masks_native: list[np.ndarray],
    *,
    input_size: Optional[int] = None,
) -> list[tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """DEPRECATED alias → ``apply_v_overlays``."""
    return apply_v_overlays(packs, giant_masks_native, input_size=input_size)
