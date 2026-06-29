"""Augmentaciones GPU de tiles para cache M+ (diversidad real en embeddings)."""

from __future__ import annotations

import torch

GATE_TILE_AUG_VARIANTS: tuple[str, ...] = ("hflip", "vflip", "rot90")


def parse_aug_variants(raw: str) -> tuple[str, ...]:
    out = tuple(v.strip() for v in str(raw).split(",") if v.strip())
    for v in out:
        if v not in GATE_TILE_AUG_VARIANTS:
            raise ValueError(f"aug variant desconocida {v!r}; usar {GATE_TILE_AUG_VARIANTS}")
    return out


def apply_gate_tile_aug(tiles: torch.Tensor, variant: str) -> torch.Tensor:
    """tiles: (B, 3, H, W) uint8 cuda."""
    if variant == "hflip":
        return torch.flip(tiles, dims=[-1])
    if variant == "vflip":
        return torch.flip(tiles, dims=[-2])
    if variant == "rot90":
        return torch.rot90(tiles, k=1, dims=[-2, -1])
    raise ValueError(f"variant desconocida: {variant!r}")
