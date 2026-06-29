"""Proyeccion DINO -> embedding L2-normalizado partido en S slices dimensionales."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceMSEncoder(nn.Module):
    """Embedding L2-normalizado e ∈ R^D partido en S slices dimensionales (chunk)."""

    def __init__(
        self,
        in_dim: int = 384,
        embed_dim: int = 128,
        num_slices: int = 4,
    ):
        super().__init__()
        if embed_dim % num_slices != 0:
            raise ValueError(f"embed_dim {embed_dim} no divisible por num_slices {num_slices}")
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_slices = num_slices
        self.slice_dim = embed_dim // num_slices
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, embed_dim) L2-normalizado."""
        return F.normalize(self.proj(features), dim=-1)

    def slice_chunks(self, embed: torch.Tensor) -> list[torch.Tensor]:
        """Parte e en S sub-vectores, cada uno re-normalizado."""
        parts = embed.chunk(self.num_slices, dim=-1)
        return [F.normalize(p, dim=-1) for p in parts]
