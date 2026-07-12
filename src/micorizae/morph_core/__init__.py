"""Shared morph contract — params, stain maps, tile segmentation (I↔E).

Breaks the circular dependency: phase_e no longer imports phase_i for
WeakSegParams / stain / segment_tile / class IDs.
"""

from .classes import (
    CLASS_ARBUSCULE,
    CLASS_BG,
    CLASS_COLONY,
    CLASS_COLORS,
    CLASS_HYPHAE,
    CLASS_NAMES,
    CLASS_ROOT,
    CLASS_VESICLE,
    N_CLASSES,
)
from .params import WeakSegParams
from .segment import segment_tile, segment_tile_stain_aware
from .stain import ambiguous_stain_mask, stain_maps

__all__ = [
    "WeakSegParams",
    "stain_maps",
    "segment_tile",
    "segment_tile_stain_aware",
    "ambiguous_stain_mask",
    "CLASS_BG",
    "CLASS_ROOT",
    "CLASS_COLONY",
    "CLASS_HYPHAE",
    "CLASS_VESICLE",
    "CLASS_ARBUSCULE",
    "N_CLASSES",
    "CLASS_COLORS",
    "CLASS_NAMES",
]
