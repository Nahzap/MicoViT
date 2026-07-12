"""Weakseg class IDs shared by morph_core and phase_i (I↔E contract)."""

from __future__ import annotations

CLASS_BG = 0
CLASS_ROOT = 1
CLASS_COLONY = 2
CLASS_HYPHAE = 3
CLASS_VESICLE = 4
CLASS_ARBUSCULE = 5
N_CLASSES = 6

CLASS_COLORS = {
    CLASS_BG: (0, 0, 0),
    CLASS_ROOT: (80, 170, 70),
    CLASS_COLONY: (0, 180, 255),
    CLASS_HYPHAE: (40, 210, 255),
    CLASS_VESICLE: (255, 180, 0),
    CLASS_ARBUSCULE: (240, 60, 210),
}

CLASS_NAMES = {
    CLASS_BG: "BG",
    CLASS_ROOT: "ROOT",
    CLASS_COLONY: "COLONY",
    CLASS_HYPHAE: "IH",
    CLASS_VESICLE: "V",
    CLASS_ARBUSCULE: "A",
}
