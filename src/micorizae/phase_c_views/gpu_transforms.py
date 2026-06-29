"""Fase C — vistas por tile **100% torch CUDA**.

Reescrito sin numpy / sin PIL / sin cv2:
    - resize  -> torch.nn.functional.interpolate
    - blur    -> F.conv2d con kernel gaussiano precomputado en GPU
    - FFT     -> torch.fft.fft2 (nativo CUDA)
    - normalización ImageNet -> tensor ops en GPU

Entradas y salidas son tensores cuda. La normalización a [0,1] y la
restauración del rango log_mag para reconstrucción se mantienen consistentes
con la versión `phase_c_views.transforms` (numpy) para que `freq_reconstruct`
siga siendo válido.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ImageNet stats como buffers reutilizables (se mueven a device on demand)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _gaussian_kernel(radius: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Devuelve kernel (1, 1, K, K) gaussiano normalizado."""
    K = 2 * radius + 1
    coords = torch.arange(K, device=device, dtype=torch.float32) - radius
    g1 = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    k = g1[:, None] @ g1[None, :]
    return k.view(1, 1, K, K)


@dataclass
class FreqRepGPU:
    """Representación frecuencial en GPU (paralela a `FreqRepresentation` de la API numpy)."""

    gray: torch.Tensor          # (N, 1, H, W) float32 cuda en [0,1]
    real: torch.Tensor          # (N, 1, H, W) float32 cuda
    imag: torch.Tensor          # (N, 1, H, W) float32 cuda
    log_mag_norm: torch.Tensor  # (N, 1, H, W) float32 cuda en [0,1]
    cos_phase: torch.Tensor     # (N, 1, H, W) float32 cuda
    sin_phase: torch.Tensor     # (N, 1, H, W) float32 cuda
    dc_mean: torch.Tensor       # (N,) float32 cuda — media removida pre-FFT
    log_mag_min: torch.Tensor   # (N,) float32 cuda
    log_mag_max: torch.Tensor   # (N,) float32 cuda

    @property
    def features_3ch(self) -> torch.Tensor:
        """(N, 3, H, W) listo para el FreqNet — log_mag_norm, cos, sin."""
        return torch.cat([self.log_mag_norm, self.cos_phase, self.sin_phase], dim=1)


def _to_float01(tile_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (N, 3, H, W) o (3, H, W) -> float32 normalizado a [0,1]."""
    if tile_u8.dim() == 3:
        tile_u8 = tile_u8.unsqueeze(0)
    if tile_u8.dtype != torch.uint8:
        raise TypeError(f"tile_u8 dtype debe ser uint8, recibí {tile_u8.dtype}")
    return tile_u8.float() / 255.0


def rgb_view_gpu(
    tiles_u8: torch.Tensor,
    target_size: int = 224,
    normalize_imagenet: bool = True,
) -> torch.Tensor:
    """Resize a `target_size` y opcionalmente normaliza ImageNet.

    Args:
        tiles_u8: (N, 3, H, W) uint8 cuda, o (3, H, W) uint8 cuda.
        target_size: lado del cuadrado de salida.
        normalize_imagenet: si True devuelve (N, 3, target, target) float32 normalizado.
            si False devuelve uint8 (no normalizado) para visualización.

    Returns:
        Tensor cuda de la forma `(N, 3, target_size, target_size)`.
    """
    x = tiles_u8 if tiles_u8.dim() == 4 else tiles_u8.unsqueeze(0)
    if x.shape[-1] != target_size or x.shape[-2] != target_size:
        x = F.interpolate(
            x.float() if x.dtype == torch.uint8 else x,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        x = x.float()
    if normalize_imagenet:
        x = x / 255.0
        mean = _IMAGENET_MEAN.to(x.device)
        std = _IMAGENET_STD.to(x.device)
        return (x - mean) / std
    return x.to(torch.uint8)


def seg_view_gpu(
    tiles_u8: torch.Tensor,
    target_size: int = 320,
    blur_radius: int = 1,
    blur_sigma: float = 1.0,
    normalize_imagenet: bool = True,
) -> torch.Tensor:
    """Resize + leve gaussian blur — entrada típica para U2Net (3, 320, 320).

    El blur se aplica por canal con `F.conv2d` agrupado.
    """
    x = tiles_u8 if tiles_u8.dim() == 4 else tiles_u8.unsqueeze(0)
    if x.shape[-1] != target_size or x.shape[-2] != target_size:
        x = F.interpolate(
            x.float() if x.dtype == torch.uint8 else x,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        x = x.float()

    if blur_radius > 0:
        kernel = _gaussian_kernel(blur_radius, blur_sigma, x.device)
        # kernel (1,1,K,K) -> (3,1,K,K) para conv2d grouped por canal
        kernel = kernel.expand(3, 1, *kernel.shape[-2:])
        x = F.conv2d(x, kernel, padding=blur_radius, groups=3)

    if normalize_imagenet:
        x = x / 255.0
        mean = _IMAGENET_MEAN.to(x.device)
        std = _IMAGENET_STD.to(x.device)
        return (x - mean) / std
    return x.to(torch.uint8)


def freq_view_gpu(tiles_u8: torch.Tensor, target_size: int = 224) -> FreqRepGPU:
    """Espectro complejo + log-mag y (cos, sin) phase — 100% torch CUDA.

    Para cada tile:
        1) convierte a gris ponderado ITU-R BT.601 (0.299R + 0.587G + 0.114B).
        2) resize a target_size.
        3) FFT centrada de `gray - mean`.
        4) log(1 + |F|) normalizado [0,1] por imagen + (cos φ, sin φ).
    """
    x = tiles_u8 if tiles_u8.dim() == 4 else tiles_u8.unsqueeze(0)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0

    if x.shape[-1] != target_size or x.shape[-2] != target_size:
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

    # grayscale por canal (N, 3, H, W) -> (N, 1, H, W)
    w_r, w_g, w_b = 0.299, 0.587, 0.114
    gray = (w_r * x[:, 0:1] + w_g * x[:, 1:2] + w_b * x[:, 2:3]).clamp(0.0, 1.0)

    # remove DC per-image
    dc_mean = gray.mean(dim=(-1, -2), keepdim=True)
    centered = gray - dc_mean

    # FFT centrada — el shift se hace post-FFT
    f = torch.fft.fft2(centered.squeeze(1))               # (N, H, W) complex
    f = torch.fft.fftshift(f, dim=(-2, -1))               # spectro centrado
    real = f.real.unsqueeze(1)                             # (N, 1, H, W)
    imag = f.imag.unsqueeze(1)
    mag = torch.sqrt(real * real + imag * imag)
    log_mag = torch.log1p(mag + 1e-6)                      # log(1+|F|)
    # normalizar por imagen al rango [0, 1]
    flat = log_mag.flatten(1)
    lm_min = flat.min(dim=1).values
    lm_max = flat.max(dim=1).values
    rng = (lm_max - lm_min).clamp(min=1e-6)
    lm_min_v = lm_min.view(-1, 1, 1, 1)
    rng_v = rng.view(-1, 1, 1, 1)
    log_mag_norm = (log_mag - lm_min_v) / rng_v

    # fase wrap-free
    eps = 1e-6
    safe_mag = mag.clamp(min=eps)
    cos_phase = real / safe_mag
    sin_phase = imag / safe_mag

    return FreqRepGPU(
        gray=gray,
        real=real,
        imag=imag,
        log_mag_norm=log_mag_norm,
        cos_phase=cos_phase,
        sin_phase=sin_phase,
        dc_mean=dc_mean.flatten(),
        log_mag_min=lm_min,
        log_mag_max=lm_max,
    )


def freq_reconstruct_gpu(rep: FreqRepGPU) -> torch.Tensor:
    """Reconstrucción exacta desde real+imag — devuelve (N, 1, H, W) float [0,1]."""
    f_centered = torch.complex(rep.real.squeeze(1), rep.imag.squeeze(1))
    f = torch.fft.ifftshift(f_centered, dim=(-2, -1))
    img = torch.fft.ifft2(f).real.unsqueeze(1)
    img = img + rep.dc_mean.view(-1, 1, 1, 1)
    return img.clamp(0.0, 1.0)


@dataclass
class ViewBundleGPU:
    """Resultado de `build_views_gpu`: tres tensores listos para las 3 ramas."""

    rgb: torch.Tensor           # (N, 3, 224, 224) float32 normalizada ImageNet
    seg: torch.Tensor           # (N, 3, 320, 320) float32 normalizada ImageNet
    freq: torch.Tensor          # (N, 3, 224, 224) (log|F|, cosφ, sinφ)
    freq_full: FreqRepGPU       # representación completa para reconstrucción


def build_views_gpu(
    tiles_u8: torch.Tensor,
    *,
    target_size: int = 224,
    seg_target_size: int = 320,
) -> ViewBundleGPU:
    """Construye las 3 vistas para un batch de tiles — todo en VRAM."""
    if tiles_u8.dim() == 3:
        tiles_u8 = tiles_u8.unsqueeze(0)
    rgb = rgb_view_gpu(tiles_u8, target_size=target_size, normalize_imagenet=True)
    seg = seg_view_gpu(tiles_u8, target_size=seg_target_size, blur_radius=1, normalize_imagenet=True)
    freq_full = freq_view_gpu(tiles_u8, target_size=target_size)
    freq3 = freq_full.features_3ch
    return ViewBundleGPU(rgb=rgb, seg=seg, freq=freq3, freq_full=freq_full)
