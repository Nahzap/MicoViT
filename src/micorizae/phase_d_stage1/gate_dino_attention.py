"""Extraccion de mapas de atencion DINOv2 ViT para cache gate AM.

Por defecto guarda, por tile y capa, la atencion **CLS -> patch** re-ordenada
como grilla espacial (gh x gw), promediada sobre heads. Formato compacto para
entrenamiento downstream (~8 KB/tile @ 12 capas, 18x18).

Modo opcional `head_reduce=none` guarda todas las heads: (L, H, gh, gw).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from .gate_vision import dino_patch_grid


@dataclass(frozen=True)
class DinoAttentionConfig:
    """Spec de mapas de atencion a persistir."""

    layer_indices: tuple[int, ...]
    head_reduce: str = "mean"  # mean | none
    query_token: str = "cls"  # cls -> atencion del token CLS hacia patches

    def __post_init__(self) -> None:
        if self.head_reduce not in {"mean", "none"}:
            raise ValueError(f"head_reduce debe ser mean o none, recibido {self.head_reduce!r}")
        if self.query_token not in {"cls"}:
            raise ValueError(f"query_token no soportado: {self.query_token!r}")
        if not self.layer_indices:
            raise ValueError("layer_indices no puede estar vacio")

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    def tensor_shape(self, *, num_heads: int, grid_h: int, grid_w: int) -> tuple[int, ...]:
        if self.head_reduce == "mean":
            return (self.num_layers, grid_h, grid_w)
        return (self.num_layers, num_heads, grid_h, grid_w)

    def flat_dim(self, *, num_heads: int, grid_h: int, grid_w: int) -> int:
        shape = self.tensor_shape(num_heads=num_heads, grid_h=grid_h, grid_w=grid_w)
        out = 1
        for d in shape:
            out *= int(d)
        return out


def parse_attention_layers(spec: str, num_blocks: int) -> tuple[int, ...]:
    """Parsea GATE_CACHE_ATTENTION_LAYERS: all | last | 0,5,11."""
    key = str(spec).strip().lower()
    if key == "all":
        return tuple(range(num_blocks))
    if key == "last":
        return (num_blocks - 1,)
    parts = [int(p.strip()) for p in key.split(",") if p.strip()]
    for idx in parts:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"layer index {idx} fuera de rango [0, {num_blocks - 1}]")
    return tuple(parts)


def dino_attention_config_from_strings(
    *,
    layers_spec: str,
    num_blocks: int,
    num_heads: int,
    head_reduce: str = "mean",
    dino_input_size: int = 252,
) -> DinoAttentionConfig:
    gh, gw = dino_patch_grid(dino_input_size)
    _ = gh, gw, num_heads  # validacion implicita via patch grid
    return DinoAttentionConfig(
        layer_indices=parse_attention_layers(layers_spec, num_blocks),
        head_reduce=head_reduce,
    )


def _attention_weights(attn_module: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    """Softmax(QK^T/sqrt(d)) — compatible con Attention y MemEffAttention."""
    B, N, C = x_norm.shape
    num_heads = int(attn_module.num_heads)
    head_dim = C // num_heads
    qkv = attn_module.qkv(x_norm).reshape(B, N, 3, num_heads, head_dim)
    q, k, _v = qkv.unbind(2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    scale = float(getattr(attn_module, "scale", head_dim**-0.5))
    attn = (q @ k.transpose(-2, -1)) * scale
    return attn.softmax(dim=-1)


def _unwrap_dino_model(backbone: nn.Module) -> nn.Module:
    """DINOv2Backbone -> DinoVisionTransformer raw."""
    if hasattr(backbone, "model"):
        return backbone.model
    return backbone


@torch.inference_mode()
def extract_dino_cls_patch_attention(
    backbone: nn.Module,
    x: torch.Tensor,
    *,
    cfg: DinoAttentionConfig,
    dino_input_size: int = 252,
) -> torch.Tensor:
    """Extrae mapas CLS->patch por capa.

    Args:
        backbone: DINOv2Backbone o DinoVisionTransformer.
        x: (B, 3, H, W) float normalizado ImageNet.
        cfg: capas y reduccion de heads.

    Returns:
        float32 tensor (B, L, gh, gw) si head_reduce=mean
        o (B, L, H, gh, gw) si head_reduce=none
    """
    dino = _unwrap_dino_model(backbone)
    gh, gw = dino_patch_grid(dino_input_size)
    n_patches = gh * gw
    layer_set = set(cfg.layer_indices)
    n_reg = int(getattr(dino, "num_register_tokens", 0) or 0)
    patch_start = 1 + n_reg

    x_tokens = dino.prepare_tokens_with_masks(x)
    maps: list[torch.Tensor] = []

    for i, blk in enumerate(dino.blocks):
        x_norm = blk.norm1(x_tokens)
        if i in layer_set:
            attn = _attention_weights(blk.attn, x_norm.float())  # B,H,N,N
            cls_to_patch = attn[:, :, 0, patch_start : patch_start + n_patches]
            if cls_to_patch.shape[-1] != n_patches:
                raise RuntimeError(
                    f"cls_to_patch shape {cls_to_patch.shape} != n_patches={n_patches}"
                )
            spatial = cls_to_patch.reshape(attn.size(0), attn.size(1), gh, gw)
            if cfg.head_reduce == "mean":
                spatial = spatial.mean(dim=1)
            maps.append(spatial)
        x_tokens = blk(x_tokens)

    if len(maps) != cfg.num_layers:
        raise RuntimeError(f"capas extraidas {len(maps)} != esperado {cfg.num_layers}")
    return torch.stack(maps, dim=1)


def attention_maps_to_numpy(maps: torch.Tensor) -> np.ndarray:
    """(B, ...) float32 -> (B, ...) float16 numpy."""
    return maps.detach().float().cpu().numpy().astype(np.float16)


def describe_attention_storage(
    cfg: DinoAttentionConfig,
    *,
    num_heads: int,
    dino_input_size: int,
    n_tiles: int,
) -> dict:
    gh, gw = dino_patch_grid(dino_input_size)
    shape = cfg.tensor_shape(num_heads=num_heads, grid_h=gh, grid_w=gw)
    flat = cfg.flat_dim(num_heads=num_heads, grid_h=gh, grid_w=gw)
    bytes_per = flat * np.dtype(np.float16).itemsize
    return {
        "enabled": True,
        "layers": list(cfg.layer_indices),
        "head_reduce": cfg.head_reduce,
        "query_token": cfg.query_token,
        "grid": [gh, gw],
        "tensor_shape_per_tile": list(shape),
        "flat_dim": flat,
        "n_tiles": n_tiles,
        "size_mb": round(n_tiles * bytes_per / (1024**2), 2),
    }
