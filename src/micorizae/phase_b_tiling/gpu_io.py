"""I/O 100% CUDA: decode JPEG via nvJPEG + crop por slicing de tensor.

Sin numpy, sin PIL, sin cv2 en el hot path. Toda la imagen vive en VRAM desde
que se decodifica hasta que se libera.

Requiere torch>=1.13 + torchvision compilado con nvJPEG (lo trae el wheel
estándar `torch+cu121`). Verificado en el entorno del proyecto.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_jpeg

from ..common.logging_utils import get_logger

log = get_logger("phase_b.gpu_io")


@dataclass(frozen=True)
class GPUImage:
    """Imagen residente en VRAM. `tensor` es (3, H, W) uint8 en cuda."""

    tensor: torch.Tensor
    image_path: str

    @property
    def height(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[2])

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def vram_mb(self) -> float:
        return self.tensor.element_size() * self.tensor.nelement() / (1024 * 1024)


def decode_jpeg_gpu(image_path: Path | str, device: str | torch.device = "cuda") -> GPUImage:
    """Decodifica un JPEG directamente a VRAM con nvJPEG.

    Lanza RuntimeError si CUDA no está disponible o si nvJPEG falla — sin
    fallback a CPU (filosofía del proyecto: nada toca la CPU en el hot path).
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(f"decode_jpeg_gpu requiere CUDA (recibió device={device})")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. nvJPEG necesita GPU.")

    with open(image_path, "rb") as f:
        raw = f.read()
    # bytearray evita UserWarning de buffer no-writable en torch.frombuffer.
    buf = torch.tensor(bytearray(raw), dtype=torch.uint8)
    img = decode_jpeg(buf, mode=ImageReadMode.RGB, device=device)  # (3,H,W) uint8
    return GPUImage(tensor=img, image_path=str(image_path))


def crop_tile_gpu(
    img: GPUImage,
    row: int,
    col: int,
    tile_size: int,
    *,
    pad_value: int = 255,
) -> torch.Tensor:
    """Extrae un tile (3, tile_size, tile_size) uint8 de una `GPUImage`.

    Slicing puro (vista cuando posible). Si el tile se sale de la imagen, se
    padea con `pad_value` para mantener la geometría cuadrada.
    """
    h, w = img.height, img.width
    y0 = row * tile_size
    x0 = col * tile_size
    y1 = y0 + tile_size
    x1 = x0 + tile_size

    y0c = max(0, min(y0, h))
    x0c = max(0, min(x0, w))
    y1c = max(0, min(y1, h))
    x1c = max(0, min(x1, w))

    if (y1c - y0c == tile_size) and (x1c - x0c == tile_size):
        return img.tensor[:, y0c:y1c, x0c:x1c].contiguous()

    out = torch.full(
        (3, tile_size, tile_size),
        pad_value,
        dtype=torch.uint8,
        device=img.tensor.device,
    )
    if y1c > y0c and x1c > x0c:
        sub = img.tensor[:, y0c:y1c, x0c:x1c]
        out[:, : sub.shape[1], : sub.shape[2]] = sub
    return out


def iter_uniform_tile_rowcol_batches(
    rows: Sequence[int] | np.ndarray,
    cols: Sequence[int] | np.ndarray,
    tile_sizes: Sequence[int] | np.ndarray,
    *,
    batch_size: int,
) -> Iterator[tuple[list[int], list[tuple[int, int, int]]]]:
    """Itera mini-batches con tile_size uniforme, preservando el orden del manifest.

    Multi-densidad mezcla tiers (p. ej. 189/252/336); ``batch_tiles_gpu`` exige un
    solo tamaño por batch. Agrupa tiles consecutivos del mismo ``tile_size``.
    """
    n = len(rows)
    i = 0
    while i < n:
        ts = int(tile_sizes[i])
        indices: list[int] = []
        while i < n and int(tile_sizes[i]) == ts and len(indices) < batch_size:
            indices.append(i)
            i += 1
        rowcols = [(int(rows[j]), int(cols[j]), ts) for j in indices]
        yield indices, rowcols


def batch_tiles_gpu(
    img: GPUImage,
    rowcols: list[tuple[int, int, int]],
    *,
    pad_value: int = 255,
) -> torch.Tensor:
    """Stack de tiles desde la misma `GPUImage`.

    `rowcols`: lista de (row, col, tile_size). Devuelve (N, 3, T, T) uint8 cuda.

    Cuando todos los tiles tienen el mismo `tile_size` y caben dentro de la
    imagen, esto es una operación O(N) de copias pequeñas dentro de VRAM.
    """
    if not rowcols:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=img.tensor.device)

    sizes = {ts for _, _, ts in rowcols}
    if len(sizes) != 1:
        raise ValueError(f"batch_tiles_gpu requiere tile_size uniforme, recibió {sizes}")
    ts = sizes.pop()

    out = torch.full(
        (len(rowcols), 3, ts, ts),
        pad_value,
        dtype=torch.uint8,
        device=img.tensor.device,
    )
    h, w = img.height, img.width
    for i, (row, col, _) in enumerate(rowcols):
        y0 = row * ts
        x0 = col * ts
        y1 = y0 + ts
        x1 = x0 + ts
        y0c = max(0, min(y0, h))
        x0c = max(0, min(x0, w))
        y1c = max(0, min(y1, h))
        x1c = max(0, min(x1, w))
        if y1c > y0c and x1c > x0c:
            sub = img.tensor[:, y0c:y1c, x0c:x1c]
            out[i, :, : sub.shape[1], : sub.shape[2]] = sub
    return out
