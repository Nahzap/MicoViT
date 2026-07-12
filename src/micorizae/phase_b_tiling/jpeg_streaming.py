"""Decode JPEG por ventanas (E): pyvips sequential o cv2 acelerado."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .tile_cutter import crop_tile_from_array


def open_image_rgb_fast(image_path: Path) -> np.ndarray:
    """Decode JPEG vía OpenCV (~2-3x vs PIL en panorámicas AM)."""
    import cv2

    buf = np.fromfile(str(image_path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2.imdecode fallo: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pad_tile_u8(tile: np.ndarray, tile_size: int, pad_value: int = 255) -> np.ndarray:
    th, tw = tile.shape[:2]
    if th == tile_size and tw == tile_size:
        return tile
    out = np.full((tile_size, tile_size, 3), pad_value, dtype=np.uint8)
    out[:th, :tw] = tile
    return out


def _batch_tiles_pyvips(
    image_path: Path,
    rowcols: list[tuple[int, int, int]],
    device: torch.device,
    *,
    pad_value: int = 255,
) -> torch.Tensor:
    import pyvips

    img = pyvips.Image.new_from_file(str(image_path), access="sequential")
    w, h = img.width, img.height
    stacked = np.full((len(rowcols), rowcols[0][2], rowcols[0][2], 3), pad_value, dtype=np.uint8)
    ts0 = rowcols[0][2]
    for i, (row, col, ts) in enumerate(rowcols):
        if ts != ts0:
            raise ValueError("batch_tiles_pyvips requiere tile_size uniforme")
        x0 = col * ts
        y0 = row * ts
        cw = min(ts, max(0, w - x0))
        ch = min(ts, max(0, h - y0))
        if cw > 0 and ch > 0:
            region = img.crop(x0, y0, cw, ch)
            arr = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=np.uint8,
                shape=(ch, cw, region.bands),
            )
            if region.bands == 4:
                arr = arr[:, :, :3]
            stacked[i] = _pad_tile_u8(arr, ts, pad_value)
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).to(device, non_blocking=True)


def crop_tile_u8_from_file(
    image_path: Path,
    row: int,
    col: int,
    tile_size: int,
    *,
    pad_value: int = 255,
    image_arr_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Un tile RGB uint8 desde disco. pyvips=crop sin decode completo; si no, cv2+PIL cache."""
    x0 = col * tile_size
    y0 = row * tile_size
    try:
        import pyvips

        img = pyvips.Image.new_from_file(str(image_path), access="sequential")
        w, h = img.width, img.height
        cw = min(tile_size, max(0, w - x0))
        ch = min(tile_size, max(0, h - y0))
        if cw <= 0 or ch <= 0:
            raise ValueError(f"crop fuera de bounds {image_path.name} r{row}c{col}")
        region = img.crop(x0, y0, cw, ch)
        arr = np.ndarray(
            buffer=region.write_to_memory(),
            dtype=np.uint8,
            shape=(ch, cw, region.bands),
        )
        if region.bands == 4:
            arr = arr[:, :, :3]
        return _pad_tile_u8(arr, tile_size, pad_value)
    except (ImportError, OSError, AttributeError, ValueError):
        pass

    cache = image_arr_cache if image_arr_cache is not None else {}
    key = str(image_path.resolve())
    if key not in cache:
        cache[key] = open_image_rgb_fast(image_path)
    return crop_tile_from_array(
        cache[key], row=row, col=col, tile_size=tile_size, pad_value=pad_value
    )


def batch_tiles_streaming_from_file(
    image_path: Path,
    rowcols: list[tuple[int, int, int]],
    device: torch.device,
    *,
    pad_value: int = 255,
    image_arr_cache: dict[str, np.ndarray],
) -> torch.Tensor:
    """Crop tiles sin cargar RGB completo si pyvips está disponible (E)."""
    if not rowcols:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=device)

    key = str(image_path)
    try:
        return _batch_tiles_pyvips(image_path, rowcols, device, pad_value=pad_value)
    except (ImportError, OSError, AttributeError):
        if key not in image_arr_cache:
            image_arr_cache[key] = open_image_rgb_fast(image_path)
        arr = image_arr_cache[key]
        ts = rowcols[0][2]
        stacked = np.full((len(rowcols), ts, ts, 3), pad_value, dtype=np.uint8)
        for i, (row, col, tile_size) in enumerate(rowcols):
            if tile_size != ts:
                raise ValueError("batch_tiles_streaming requiere tile_size uniforme")
            stacked[i] = crop_tile_from_array(
                arr, row=row, col=col, tile_size=tile_size, pad_value=pad_value
            )
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).to(device, non_blocking=True)
