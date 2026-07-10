"""Clases morfológicas píxel — Fase 2 (BG + IH / V / A / H)."""

from __future__ import annotations

PIXEL_CLASS_NAMES = ("BG", "IH", "V", "A", "H")
NUM_PIXEL_CLASSES = len(PIXEL_CLASS_NAMES)

# Índice de "ignore": píxeles ambiguos (tinción no resoluble) que NO entran en
# la loss ni en las métricas. Supervisión parcial tipo scribbles/seeds
# (Lin et al. 2016, "ScribbleSup"). uint8 admite 255 en el HDF5.
PIXEL_IGNORE_INDEX = 255

PIXEL_CLASS_TO_IDX = {n: i for i, n in enumerate(PIXEL_CLASS_NAMES)}
PIXEL_IDX_TO_CLASS = {i: n for i, n in enumerate(PIXEL_CLASS_NAMES)}

PIXEL_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "BG": (0, 0, 0),
    "IH": (40, 210, 255),       # cian — hifas intrarradicales
    "V": (255, 180, 0),         # naranja — vesículas
    "A": (240, 60, 210),        # magenta — arbúsculos
    "H": (100, 230, 90),        # verde lima — colonia / espacio hifal (≠ IH)
}

MORPH_STRUCTURE_CLASSES = ("IH", "V", "A", "H")
COLONY_CLASSES = ("IH", "V", "A", "H")
