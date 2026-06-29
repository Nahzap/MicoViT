"""Gate Slice-MS con DINO on-the-fly (E6 fine-tune + E7 pooling BG-only)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gate_classes import GATE_CLASS_TO_IDX
from ..gate_bg_pooling import bg_only_pooling_mask
from .slice_encoder import SliceMSEncoder


class GateSliceDinoModel(nn.Module):
    """DINO backbone + Slice MS encoder (entrenamiento end-to-end, sin cache)."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        in_dim: int = 384,
        embed_dim: int = 128,
        num_slices: int = 4,
        num_classes: int = 4,
        pooling_mode: str = "bg_only",
        attn_in_dim: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_mode = (pooling_mode or "none").lower()
        probe_in = in_dim + attn_in_dim
        self.encoder = SliceMSEncoder(in_dim=probe_in, embed_dim=embed_dim, num_slices=num_slices)
        self.gate_head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self._bg_idx = GATE_CLASS_TO_IDX["Background"]

    def encode_features(
        self,
        rgb: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attn_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pool_mask = bg_only_pooling_mask(
            labels, mask, pooling_mode=self.pooling_mode
        ) if labels is not None else mask
        feat = self.backbone(rgb, mask=pool_mask)
        if attn_features is not None:
            feat = torch.cat([feat, attn_features], dim=-1)
        return feat

    def forward_from_features(
        self,
        features: torch.Tensor,
        *,
        return_embed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embed = self.encoder(features)
        logits = self.gate_head(embed)
        if return_embed:
            return logits, embed
        return logits

    def forward(
        self,
        rgb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        labels: Optional[torch.Tensor] = None,
        attn_features: Optional[torch.Tensor] = None,
        return_embed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_features(
            rgb, mask=mask, labels=labels, attn_features=attn_features
        )
        return self.forward_from_features(feat, return_embed=return_embed)


def build_gate_slice_dino(
    backbone: nn.Module,
    *,
    embed_dim: int = 128,
    num_slices: int = 4,
    num_classes: int = 4,
    pooling_mode: str = "bg_only",
    in_dim: int = 384,
    attn_in_dim: int = 0,
) -> GateSliceDinoModel:
    return GateSliceDinoModel(
        backbone,
        in_dim=in_dim,
        embed_dim=embed_dim,
        num_slices=num_slices,
        num_classes=num_classes,
        pooling_mode=pooling_mode,
        attn_in_dim=attn_in_dim,
    )
