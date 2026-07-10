"""Atención DINOv2 para explicabilidad pixel-ViT (reutiliza gate_dino_attention)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from ..phase_d_stage1.gate_dino_attention import (
    DinoAttentionConfig,
    extract_dino_cls_patch_attention,
    parse_attention_layers,
)
from .pixel_vit_model import PixelMorphViT


def attention_config_for_pixel(
    layers_spec: str,
    backbone: torch.nn.Module,
    input_size: int = 224,
) -> DinoAttentionConfig:
    dino = backbone.model if hasattr(backbone, "model") else backbone
    n_blocks = len(dino.blocks)
    return DinoAttentionConfig(
        layer_indices=parse_attention_layers(layers_spec, n_blocks),
        head_reduce="mean",
    )


@torch.inference_mode()
def extract_pixel_vit_attention(
    model: PixelMorphViT,
    rgb_normalized: torch.Tensor,
    *,
    layers_spec: str = "last",
    input_size: int = 224,
) -> np.ndarray:
    """Devuelve mapa (H,W) float32 upsampled de atención CLS→patch (última capa media)."""
    cfg = attention_config_for_pixel(layers_spec, model.encoder, input_size=input_size)
    maps = extract_dino_cls_patch_attention(
        model.encoder,
        rgb_normalized,
        cfg=cfg,
        dino_input_size=input_size,
    )
    last = maps[:, -1, :, :].float().cpu().numpy()
    if last.shape[0] == 1:
        attn = last[0]
    else:
        attn = last.mean(axis=0)
    h, w = rgb_normalized.shape[-2:]
    attn_up = cv2.resize(attn, (w, h), interpolation=cv2.INTER_CUBIC)
    attn_up = attn_up - attn_up.min()
    denom = float(attn_up.max()) + 1e-9
    return (attn_up / denom).astype(np.float32)


def render_attention_heatmap(attn: np.ndarray) -> np.ndarray:
    gray = (np.clip(attn, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)[:, :, ::-1]
