"""Augmentations PSF / stain para Stage2-Pixel train (plan §3.4).

Fundamento: UnMICST 2022 (defocus real > Gaussian); AMFinder hue/intensity;
MICCAI 2025 PSF-Net (blur sintético multi-σ).
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


def parse_psf_sigmas(spec: str = "0.5,1.0,1.5,2.0,2.5") -> list[float]:
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def apply_psf_blur(
    rgb: np.ndarray,
    *,
    sigmas: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5),
    prob: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Aplica Gaussian blur con σ aleatorio (train only)."""
    rng = rng or np.random.default_rng()
    if prob <= 0 or rng.random() > float(prob) or not sigmas:
        return rgb
    sigma = float(rng.choice(list(sigmas)))
    if sigma <= 0:
        return rgb
    k = max(3, int(round(sigma * 4)) | 1)
    return cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma)


def apply_stain_intensity_scale(
    rgb: np.ndarray,
    *,
    lo: float = 0.7,
    hi: float = 1.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Escala contraste hacia el azul (proxy intensidad de tinción)."""
    rng = rng or np.random.default_rng()
    scale = float(rng.uniform(lo, hi))
    out = rgb.astype(np.float32)
    # Empuja canal B y oscurece R (tinción azul típica)
    out[..., 2] = np.clip(out[..., 2] * scale, 0, 255)
    out[..., 0] = np.clip(out[..., 0] / max(scale, 1e-3) * 1.0, 0, 255)
    return out.astype(np.uint8)


def apply_train_augmentations(
    rgb: np.ndarray,
    *,
    psf_sigmas: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5),
    psf_prob: float = 0.35,
    stain_lo: float = 0.7,
    stain_hi: float = 1.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = apply_psf_blur(rgb, sigmas=psf_sigmas, prob=psf_prob, rng=rng)
    if rng.random() < 0.5:
        out = apply_stain_intensity_scale(out, lo=stain_lo, hi=stain_hi, rng=rng)
    return out
