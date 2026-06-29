"""Entrada al probe linear / Slice-MS desde cache (embed + attn opcional)."""

from __future__ import annotations

from typing import Optional

import torch

from .gpu_pipeline import GPUImageBatch


def pool_vit_attention(attn: torch.Tensor) -> torch.Tensor:
    """(B, L, gh, gw) -> (B, L) promedio espacial por capa."""
    return attn.mean(dim=(-2, -1))


def probe_features_from_batch(batch: GPUImageBatch) -> torch.Tensor:
    """Concatena embedding DINO con resumen de mapas ViT si existen."""
    if batch.features is None:
        raise ValueError("batch.features es None")
    feats = batch.features
    if batch.vit_attention is not None:
        attn_vec = pool_vit_attention(batch.vit_attention)
        feats = torch.cat([feats, attn_vec], dim=-1)
    return feats


def probe_in_dim_from_attention_meta(attention_meta: Optional[dict], embed_dim: int = 384) -> int:
    """Calcula in_dim del probe: embed + L capas (mean-pool espacial)."""
    if not attention_meta:
        return embed_dim
    shape = attention_meta.get("tensor_shape_per_tile") or ()
    if len(shape) >= 1:
        return embed_dim + int(shape[0])
    flat = int(attention_meta.get("flat_dim", 0))
    if flat <= 0:
        return embed_dim
    gh, gw = attention_meta.get("grid") or [18, 18]
    per_layer = int(gh) * int(gw)
    if per_layer > 0:
        return embed_dim + flat // per_layer
    return embed_dim
