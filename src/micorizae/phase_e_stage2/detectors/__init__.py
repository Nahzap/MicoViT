"""Detectores por clase — un módulo = una técnica (SRP Stage2)."""

from __future__ import annotations

from .a_arbuscule import detect_arbuscules
from .bg_root import detect_root
from .h_colony import detect_colony
from .ih_hyphae import detect_hyphae
from .types import DetectResult
from .v_giant_multitile import (
    apply_giant_v_to_label_and_priors,
    detect_giant_vesicle_masks_for_tiles,
)
from .v_vesicle import (
    apply_v_masks_to_label_and_priors,
    detect_vesicle_masks_for_tiles,
    detect_vesicles,
)

__all__ = [
    "DetectResult",
    "detect_root",
    "detect_hyphae",
    "detect_vesicles",
    "detect_vesicle_masks_for_tiles",
    "apply_v_masks_to_label_and_priors",
    "detect_arbuscules",
    "detect_colony",
    "detect_giant_vesicle_masks_for_tiles",
    "apply_giant_v_to_label_and_priors",
]
