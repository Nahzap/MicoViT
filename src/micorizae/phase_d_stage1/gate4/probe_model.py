"""Modelo probe: embedding cache DINO -> Slice MS encoder -> gate 4 clases."""

from __future__ import annotations

import torch
import torch.nn as nn

from .slice_encoder import SliceMSEncoder


class GateSliceProbeModel(nn.Module):
    """Entrena solo proyeccion + cabeza gate (linear probe + Slice MS)."""

    def __init__(
        self,
        in_dim: int = 384,
        embed_dim: int = 128,
        num_slices: int = 4,
        num_classes: int = 4,
        head_hidden: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SliceMSEncoder(in_dim=in_dim, embed_dim=embed_dim, num_slices=num_slices)
        self.num_classes = num_classes
        if head_hidden > 0:
            self.gate_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.gate_head = nn.Linear(embed_dim, num_classes)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward_from_features(
        self,
        features: torch.Tensor,
        *,
        return_embed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embed = self.encode(features)
        logits = self.gate_head(embed)
        if return_embed:
            return logits, embed
        return logits

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError(
            "GateSliceProbeModel usa forward_from_features con cache de embeddings."
        )


def build_gate_slice_probe(
    *,
    in_dim: int = 384,
    embed_dim: int = 128,
    num_slices: int = 4,
    num_classes: int = 4,
) -> GateSliceProbeModel:
    return GateSliceProbeModel(
        in_dim=in_dim,
        embed_dim=embed_dim,
        num_slices=num_slices,
        num_classes=num_classes,
    )
