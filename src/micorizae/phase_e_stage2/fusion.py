"""Fusión multiclas e Stage2 + entropía / consenso."""

from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {"A": 0.40, "B": 0.40, "C": 0.20}


def fuse_probabilities_mc(
    probs_per_branch: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Promedio ponderado de distribuciones (K,) por tile -> (N, K)."""
    weights = weights or DEFAULT_WEIGHTS
    branches = [k for k in probs_per_branch if k in weights]
    w = np.array([weights[k] for k in branches], dtype=np.float64)
    w = w / w.sum()
    stack = np.stack([probs_per_branch[k] for k in branches], axis=0)  # (B, N, K)
    return (w[:, None, None] * stack).sum(axis=0)


def entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Entropía Shannon por fila, normalizada a [0,1] por log(K)."""
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    h = -(p * np.log(p)).sum(axis=-1)
    k = probs.shape[-1]
    if k <= 1:
        return np.zeros(probs.shape[0], dtype=np.float64)
    return h / np.log(k)


def js_divergence_mc(probs_per_branch: dict[str, np.ndarray], eps: float = 1e-9) -> np.ndarray:
    """JS divergence multiclas entre ramas, por tile."""
    keys = list(probs_per_branch.keys())
    p = np.stack([np.clip(probs_per_branch[k], eps, 1.0) for k in keys], axis=0)
    p = p / p.sum(axis=-1, keepdims=True)
    m = p.mean(axis=0)
    kl = (p * (np.log(p) - np.log(m[None]))).sum(axis=-1)
    return kl.mean(axis=0)
