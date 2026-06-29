"""Extracción de tiles individuales — modo streaming (ventana móvil).

Filosofía: abrir cada imagen UNA vez, recorrer sus tiles como una ventana
deslizante, soltarla, pasar a la siguiente. No mantenemos panorámicas enteras
en RAM (1 GB+ cada una). El usuario que orquesta el barrido es el
`ImageWindowDataset` de `phase_b_tiling.streaming`.

`crop_tile_from_array(arr, row, col, tile_size, ...)` extrae UN tile de una
imagen ya decodificada (el `arr` se mantiene en el scope del caller, así que
sólo se decodifica una vez).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TileBBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def from_rowcol(cls, row: int, col: int, tile_size: int) -> "TileBBox":
        return cls(col * tile_size, row * tile_size, (col + 1) * tile_size, (row + 1) * tile_size)

    @property
    def size(self) -> int:
        return self.x1 - self.x0


def open_image_rgb(image_path: Path) -> np.ndarray:
    """Decodifica una imagen a ndarray RGB. Se usa una sola vez por imagen."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as im:
        return np.array(im.convert("RGB"))


def crop_tile_from_array(
    image_arr: np.ndarray,
    row: int,
    col: int,
    tile_size: int,
    *,
    target_size: Optional[int] = None,
    pad_value: int = 255,
) -> np.ndarray:
    """Extrae UN tile (row,col) desde una imagen ya decodificada en RAM.

    Esta función es lo que se llama dentro del bucle de streaming, ~50 µs por tile
    para AM 252×252 (sólo un slice de numpy).
    """
    bbox = TileBBox.from_rowcol(row, col, tile_size)
    h, w = image_arr.shape[:2]

    x0 = max(0, min(bbox.x0, w))
    y0 = max(0, min(bbox.y0, h))
    x1 = max(0, min(bbox.x1, w))
    y1 = max(0, min(bbox.y1, h))

    if x1 > x0 and y1 > y0:
        sub = image_arr[y0:y1, x0:x1]
    else:
        sub = np.zeros((0, 0, 3), dtype=np.uint8)

    th = sub.shape[0]
    tw = sub.shape[1]
    if (th, tw) != (tile_size, tile_size):
        padded = np.full((tile_size, tile_size, 3), pad_value, dtype=np.uint8)
        padded[:th, :tw] = sub
        sub = padded

    if target_size and target_size != tile_size:
        from PIL import Image

        sub = np.array(Image.fromarray(sub).resize((target_size, target_size), Image.BILINEAR))

    return sub


def crop_tile(
    image_path: Path,
    row: int,
    col: int,
    tile_size: int,
    *,
    target_size: Optional[int] = None,
    pad_value: int = 255,
) -> np.ndarray:
    """Compatibilidad hacia atrás: abre y crop en una sola llamada.

    INEFICIENTE cuando se llama muchas veces sobre la misma imagen — para esos
    casos usar `open_image_rgb` + `crop_tile_from_array` en bucle externo.
    """
    arr = open_image_rgb(image_path)
    return crop_tile_from_array(
        arr, row=row, col=col, tile_size=tile_size, target_size=target_size, pad_value=pad_value
    )
