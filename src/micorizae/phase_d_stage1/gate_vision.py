"""Resolución de visión DINO para gate AM (tiles nativos 252 px, resize DINO configurable)."""

from __future__ import annotations

from typing import Any

AM_TILE_SIZE = 252
DINO_PATCH_SIZE = 14
LEGACY_DINO_INPUT = 224


def dino_input_size_from_config(cfg: Any) -> int:
    return int(getattr(cfg, "GATE_DINO_INPUT_SIZE", AM_TILE_SIZE))


def seg_target_size_for(dino_input: int, cfg: Any | None = None) -> int:
    if cfg is not None:
        explicit = getattr(cfg, "GATE_SEG_TARGET_SIZE", None)
        if explicit is not None:
            return int(explicit)
    return int(round(dino_input * 320 / LEGACY_DINO_INPUT))


def dino_patch_grid(dino_input_size: int, patch_size: int = DINO_PATCH_SIZE) -> tuple[int, int]:
    if dino_input_size % patch_size != 0:
        raise ValueError(
            f"GATE_DINO_INPUT_SIZE={dino_input_size} debe ser multiplo de patch_size={patch_size}"
        )
    n = dino_input_size // patch_size
    return n, n


def describe_dino_resolution(dino_input_size: int) -> str:
    gh, gw = dino_patch_grid(dino_input_size)
    legacy_tokens = (LEGACY_DINO_INPUT // DINO_PATCH_SIZE) ** 2
    tokens = gh * gw
    return (
        f"{dino_input_size}px -> grilla {gh}x{gw}={tokens} tokens ViT "
        f"(antes {LEGACY_DINO_INPUT}px -> 16x16={legacy_tokens})"
    )
