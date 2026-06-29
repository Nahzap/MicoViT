"""Pooling DINO con U2Net solo en tiles Background (E7)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .gate_classes import GATE_CLASS_TO_IDX


def u2net_saliency_batch(
    u2net: torch.nn.Module,
    seg_rgb: torch.Tensor,
    *,
    saliency_size: int,
) -> torch.Tensor:
    """Inferencia U2Net -> mapa (B,H,W) en [0,1]."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        d0, *_ = u2net(seg_rgb)
    sal = F.interpolate(
        d0,
        size=(saliency_size, saliency_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return sal.float()


def bg_only_pooling_mask(
    labels: torch.Tensor,
    saliency: Optional[torch.Tensor],
    *,
    pooling_mode: str = "bg_only",
) -> Optional[torch.Tensor]:
    """Mascara por tile: U2Net solo Background; M+/M- sin mascara (mean-pool)."""
    mode = (pooling_mode or "none").lower()
    if mode != "bg_only" or saliency is None:
        return None if mode in {"none", ""} else saliency
    bg_idx = GATE_CLASS_TO_IDX["Background"]
    bg = labels == bg_idx
    if not bg.any():
        return None
    if bg.all():
        return saliency
    h, w = saliency.shape[-2], saliency.shape[-1]
    full = torch.ones((labels.size(0), h, w), device=saliency.device, dtype=saliency.dtype)
    full[bg] = saliency[bg]
    return full
