"""Detectores por clase — un módulo = una técnica (SRP Stage2)."""

from __future__ import annotations

from .a_arbuscule import detect_arbuscules
from .bg_root import detect_root
from .h_colony import detect_colony
from .ih_hyphae import detect_hyphae
from .types import DetectResult
from .v_vesicle import detect_vesicles

__all__ = [
    "DetectResult",
    "detect_root",
    "detect_hyphae",
    "detect_vesicles",
    "detect_arbuscules",
    "detect_colony",
]
