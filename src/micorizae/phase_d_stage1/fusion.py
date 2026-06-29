"""Fusión ponderada multi-rama + métrica de consenso (JS divergence).

Master plan §9:
    p_S1(M+) = sum_i w_i * p_i(M+) / sum_i w_i,
    pesos iniciales: w_A=0.45, w_B=0.35, w_C=0.20.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_WEIGHTS = {"A": 0.45, "B": 0.35, "C": 0.20}


def fuse_probabilities(probs_per_branch: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    weights = weights or DEFAULT_WEIGHTS
    branches = [k for k in probs_per_branch if k in weights]
    w = np.array([weights[k] for k in branches], dtype=np.float64)
    w = w / w.sum()
    stack = np.stack([probs_per_branch[k] for k in branches], axis=0)  # (B, N)
    return (w[:, None] * stack).sum(axis=0)


def js_divergence(probs_per_branch: dict[str, np.ndarray], eps: float = 1e-9) -> np.ndarray:
    """JS-divergence binaria entre las K distribuciones de las ramas, por tile.

    Devuelve un vector (N,) en [0, log2(K)] aproximadamente — alto = conflicto.
    """
    keys = list(probs_per_branch.keys())
    K = len(keys)
    p = np.stack([np.clip(probs_per_branch[k], eps, 1 - eps) for k in keys], axis=0)
    # distrib binaria (P, 1-P)
    p_bin = np.stack([p, 1.0 - p], axis=-1)  # (K, N, 2)
    m = p_bin.mean(axis=0)  # (N, 2)
    kl = (p_bin * (np.log(p_bin) - np.log(m[None]))).sum(axis=-1)  # (K, N)
    return kl.mean(axis=0)


def js_divergence_multiclass(probs_per_branch: dict[str, np.ndarray], eps: float = 1e-9) -> np.ndarray:
    """JS-divergence entre distribuciones multiclas (N, C) por rama."""
    keys = list(probs_per_branch.keys())
    p = np.stack([np.clip(probs_per_branch[k], eps, 1.0) for k in keys], axis=0)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), eps, None)
    m = p.mean(axis=0)
    kl_k = (p * (np.log(p) - np.log(m[None, :, :]))).sum(axis=-1)
    return kl_k.mean(axis=0)
