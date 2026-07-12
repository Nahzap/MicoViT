"""Fase B — Tiling y sincronización espacial.

Entrega E2.a:
    - manifests/tiles_index.{parquet|csv}  con bbox absoluto por tile.
    - utilidades para extraer un tile específico bajo demanda
      (Image.open -> crop, sin cargar la imagen completa en RAM).
"""

from .build_tiles_index import build_tiles_index
from .tile_cutter import crop_tile, crop_tile_from_array, open_image_rgb, TileBBox
from .streaming import ImageWindowDataset, iter_tiles_for_image, plan_image_order
from .runtime_grid import build_tile_grid, build_tile_grid_for_image, image_size

__all__ = [
    "build_tiles_index",
    "crop_tile",
    "crop_tile_from_array",
    "open_image_rgb",
    "TileBBox",
    "ImageWindowDataset",
    "iter_tiles_for_image",
    "plan_image_order",
    "build_tile_grid",
    "build_tile_grid_for_image",
    "image_size",
]
