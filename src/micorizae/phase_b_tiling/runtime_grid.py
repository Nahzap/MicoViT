"""Rejilla de tiles en runtime — experto digital (plan §0.3 / S9).

Construye (row, col, x0, y0, x1, y1, tile_size) desde dimensiones de imagen
sin depender de ``tiles_index`` preexistente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PIL import Image


def image_size(path: Union[str, Path]) -> tuple[int, int]:
    """Devuelve (width, height)."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def build_tile_grid(
    width: int,
    height: int,
    tile_size: int = 252,
    *,
    image_path: Optional[str] = None,
    lineage: str = "AM",
) -> pd.DataFrame:
    """Malla determinista no solapada que cubre toda la imagen.

    Tiles de borde pueden ser parciales (x1/y1 recortados al borde).
    """
    ts = int(tile_size)
    if ts < 8:
        raise ValueError(f"tile_size inválido: {ts}")
    n_cols = int(np.ceil(width / ts))
    n_rows = int(np.ceil(height / ts))
    rows: list[dict] = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * ts
            y0 = r * ts
            x1 = min(x0 + ts, width)
            y1 = min(y0 + ts, height)
            rows.append(
                {
                    "row": r,
                    "col": c,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "tile_size": ts,
                    "image_path": image_path or "",
                    "lineage": lineage,
                    "density_tier": "base",
                }
            )
    return pd.DataFrame(rows)


def build_tile_grid_for_image(
    image_path: Union[str, Path],
    tile_size: int = 252,
    *,
    lineage: str = "AM",
    relative_to: Optional[Path] = None,
) -> pd.DataFrame:
    """Grid runtime para una imagen en disco."""
    path = Path(image_path)
    w, h = image_size(path)
    rel = path.as_posix()
    if relative_to is not None:
        try:
            rel = path.resolve().relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
    return build_tile_grid(w, h, tile_size, image_path=rel, lineage=lineage)
