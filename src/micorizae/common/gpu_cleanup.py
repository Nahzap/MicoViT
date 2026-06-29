"""Liberacion explicita de VRAM/RAM entre batches GPU."""

from __future__ import annotations

import gc

import torch


def release_cuda_memory(*, gc_collect: bool = False) -> None:
    """Devuelve caches del allocator CUDA; opcionalmente fuerza GC en RAM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if gc_collect:
        gc.collect()


def maybe_gc_collect(batch_idx: int, *, every_n: int) -> None:
    """Ejecuta gc.collect cada N batches (every_n=0 desactiva)."""
    if every_n > 0 and batch_idx % every_n == 0:
        gc.collect()


def maybe_empty_cache(batch_idx: int, *, every_n: int) -> None:
    """Ejecuta empty_cache cada N batches (every_n=0 desactiva)."""
    if every_n > 0 and batch_idx % every_n == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
