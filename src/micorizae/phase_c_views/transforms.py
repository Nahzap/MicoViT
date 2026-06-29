"""Transformaciones puras (numpy) que producen las 3 vistas por tile.

Diseño científico explícito (Fase C):
    - vista RGB: cambio de espacio + CLAHE para resaltar tinción azul.
    - vista Seg: resize + leve blur para U2Net.
    - vista Frecuencia: **espectro complejo completo**, NO sólo magnitud.
      - Real e imaginaria de la FFT centrada -> invertible exacta.
      - log|F| + cos(φ) + sin(φ) -> input al FreqNet (3 canales, sin wrap).
      - Phase-only / magnitude-only reconstructions -> evidencia Oppenheim 1981
        de que la fase carga la estructura espacial.

No dependen de torch para mantener Fase C ejecutable sin GPU/PyTorch.
La conversión a tensores se hace en Fase D (datasets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FreqRepresentation:
    """Representación frecuencial reversible de un tile en escala de gris."""

    gray: np.ndarray              # (H, W) float32 en [0,1] — imagen fuente
    real: np.ndarray              # (H, W) float32 — Re(FFT centrada(gray - mean))
    imag: np.ndarray              # (H, W) float32 — Im(FFT centrada(gray - mean))
    log_mag_norm: np.ndarray      # (H, W) float32 en [0,1] — log(1+|F|), normalizada
    phase: np.ndarray             # (H, W) float32 en [-π, π]
    cos_phase: np.ndarray         # (H, W) float32 — cos(φ), input estable al CNN
    sin_phase: np.ndarray         # (H, W) float32 — sin(φ)
    dc_mean: float                # media removida (para reconstrucción exacta)
    log_mag_min: float            # parámetros de normalización log_mag_norm
    log_mag_max: float

    @property
    def features_3ch(self) -> np.ndarray:
        """(3, H, W) float32 listo para PyTorch — entrada estable al FreqNet."""
        return np.stack([self.log_mag_norm, self.cos_phase, self.sin_phase], axis=0)


@dataclass
class ViewBundle:
    rgb: np.ndarray                 # (H, W, 3) uint8 (normalizada o lista para normalizar)
    seg_input: np.ndarray           # (H, W, 3) uint8 listo para U2Net (resize + leve blur)
    freq: FreqRepresentation        # estructura completa, invertible
    target_size: int


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def rgb_view(
    tile: np.ndarray,
    target_size: int = 224,
    clahe: bool = True,
) -> np.ndarray:
    """RGB normalizada — resize + CLAHE en canal L (mejora contraste de tinción azul).

    No aplica normalización ImageNet aquí; eso pertenece al DataLoader.
    """
    from PIL import Image

    pil = Image.fromarray(_ensure_uint8(tile)).resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(pil)

    if clahe:
        try:
            import cv2

            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe_op.apply(l)
            lab = cv2.merge((l, a, b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception:
            pass  # cv2 ausente: devolvemos sin CLAHE

    return arr.astype(np.uint8)


def seg_view(
    tile: np.ndarray,
    target_size: int = 320,
    blur_kernel: int = 3,
) -> np.ndarray:
    """Pre-procesado para U2Net: resize a 320, leve blur para reducir grano del escáner."""
    from PIL import Image, ImageFilter

    pil = Image.fromarray(_ensure_uint8(tile)).resize((target_size, target_size), Image.BILINEAR)
    if blur_kernel and blur_kernel > 1:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_kernel / 2))
    return np.array(pil)


def freq_view(
    tile: np.ndarray,
    target_size: int = 224,
    eps: float = 1e-6,
) -> FreqRepresentation:
    """Espectro complejo completo + representaciones derivadas (invertible).

    NO devuelve sólo `log|F|` (irrecuperable). Devuelve:
        - real, imag de la FFT centrada (reconstrucción exacta).
        - log(1+|F|) normalizada a [0,1] (vista humana).
        - cos(φ), sin(φ) (entrada estable al FreqNet, evita wrap en ±π).

    La media (`dc_mean`) se remueve antes del FFT y se guarda para reconstruir.
    """
    from PIL import Image

    gray_pil = (
        Image.fromarray(_ensure_uint8(tile))
        .convert("L")
        .resize((target_size, target_size), Image.BILINEAR)
    )
    gray = np.asarray(gray_pil, dtype=np.float32) / 255.0
    dc_mean = float(gray.mean())

    f = np.fft.fftshift(np.fft.fft2(gray - dc_mean))
    real = f.real.astype(np.float32)
    imag = f.imag.astype(np.float32)
    mag = np.abs(f)
    phase = np.angle(f).astype(np.float32)

    log_mag = np.log1p(mag + eps).astype(np.float32)
    lm_min = float(log_mag.min())
    lm_max = float(log_mag.max())
    log_mag_norm = (log_mag - lm_min) / max(lm_max - lm_min, eps)

    return FreqRepresentation(
        gray=gray.astype(np.float32),
        real=real,
        imag=imag,
        log_mag_norm=log_mag_norm.astype(np.float32),
        phase=phase,
        cos_phase=np.cos(phase).astype(np.float32),
        sin_phase=np.sin(phase).astype(np.float32),
        dc_mean=dc_mean,
        log_mag_min=lm_min,
        log_mag_max=lm_max,
    )


def freq_reconstruct(rep: FreqRepresentation) -> np.ndarray:
    """Reconstruye la imagen gris desde el espectro complejo COMPLETO (real + imag).

    Es exacta hasta precisión de float32.
    """
    f_centered = rep.real + 1j * rep.imag
    f = np.fft.ifftshift(f_centered)
    img = np.real(np.fft.ifft2(f)) + rep.dc_mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def freq_reconstruct_from_logmag_phase(rep: FreqRepresentation) -> np.ndarray:
    """Reconstruye usando log_mag_norm + (cos φ, sin φ) — la versión 3-canal del FreqNet.

    Útil para verificar que la representación que entra al CNN es funcionalmente
    invertible (modulo error de cuantización log y normalización).
    """
    eps = 1e-6
    rng = max(rep.log_mag_max - rep.log_mag_min, eps)
    log_mag = rep.log_mag_norm * rng + rep.log_mag_min
    mag = np.expm1(log_mag) - eps
    mag = np.clip(mag, 0.0, None)
    phase = np.arctan2(rep.sin_phase, rep.cos_phase)
    f_centered = mag * np.exp(1j * phase)
    f = np.fft.ifftshift(f_centered)
    img = np.real(np.fft.ifft2(f)) + rep.dc_mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def freq_reconstruct_magnitude_only(rep: FreqRepresentation, seed: int = 0) -> np.ndarray:
    """Reconstrucción usando SÓLO la magnitud con fase aleatoria.

    Esto demuestra (Oppenheim & Lim 1981) que sin fase la imagen no se recupera:
    el resultado es ruido con la misma distribución espectral.
    """
    rng = np.random.default_rng(seed)
    random_phase = rng.uniform(-np.pi, np.pi, rep.real.shape).astype(np.float32)
    mag = np.abs(rep.real + 1j * rep.imag)
    f_centered = mag * np.exp(1j * random_phase)
    f = np.fft.ifftshift(f_centered)
    img = np.real(np.fft.ifft2(f)) + rep.dc_mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def freq_reconstruct_phase_only(rep: FreqRepresentation) -> np.ndarray:
    """Reconstrucción usando SÓLO la fase con magnitud unitaria.

    Complemento del experimento anterior: la fase **sola** recupera bordes
    y estructura, demostrando que la mayoría de la información perceptual
    vive en la fase.
    """
    mag = np.ones_like(rep.real)
    phase = np.angle(rep.real + 1j * rep.imag)
    f_centered = mag * np.exp(1j * phase)
    f = np.fft.ifftshift(f_centered)
    img = np.real(np.fft.ifft2(f))
    # esto puede estar centrada en cero; normalizar a [0,1] visualmente
    img -= img.min()
    img /= max(img.max(), 1e-6)
    return (img * 255).astype(np.uint8)


def build_views(
    tile: np.ndarray,
    target_size: int = 224,
    seg_target_size: int = 320,
    clahe: bool = True,
    blur_kernel: int = 3,
) -> ViewBundle:
    return ViewBundle(
        rgb=rgb_view(tile, target_size=target_size, clahe=clahe),
        seg_input=seg_view(tile, target_size=seg_target_size, blur_kernel=blur_kernel),
        freq=freq_view(tile, target_size=target_size),
        target_size=target_size,
    )
