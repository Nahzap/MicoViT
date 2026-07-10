"""Métricas G-PX-EX — PPA, ECE, PCA, concordancia prior–pred."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_TO_IDX
from .pixel_prior_maps import prior_argmax_agreement


def pixel_accuracy(pred: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    return prior_argmax_agreement(pred, ref, root=mask)


def expected_calibration_error(
    probs: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    n_bins: int = 10,
    mask: Optional[np.ndarray] = None,
) -> float:
    """ECE sobre P_max vs acierto respecto a ref."""
    if mask is None:
        mask = np.ones(pred.shape, dtype=bool)
    else:
        mask = mask > 0
    if not mask.any():
        return 0.0
    conf = probs.max(axis=0)[mask]
    correct = (pred[mask] == ref[mask]).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(mask.sum())
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not sel.any():
            continue
        ece += sel.sum() / n * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def prior_conditional_accuracy(
    pred: np.ndarray,
    evidence_c: np.ndarray,
    class_idx: int,
    tau: float = 0.5,
    root: Optional[np.ndarray] = None,
) -> float:
    if root is not None:
        base = (evidence_c > tau) & (root > 0)
    else:
        base = evidence_c > tau
    if not base.any():
        return float("nan")
    return float(np.mean(pred[base] == class_idx))


def disagreement_map(pred: np.ndarray, weak: np.ndarray) -> np.ndarray:
    return (pred != weak).astype(np.uint8)


def tile_explain_stats(
    pred: np.ndarray,
    weak: np.ndarray,
    probs: np.ndarray,
    prior: np.ndarray,
) -> dict[str, float]:
    root_mask = (np.max(prior[1:], axis=0) > 0.05).astype(np.uint8)
    ppa = prior_argmax_agreement(pred, weak, root_mask)
    ece = expected_calibration_error(probs, pred, weak, mask=root_mask)
    return {
        "ppa_tile": ppa,
        "ece_tile": ece,
        "prior_IH_mean": float(prior[PIXEL_CLASS_TO_IDX["IH"]][root_mask > 0].mean()) if root_mask.any() else 0.0,
        "prior_V_mean": float(prior[PIXEL_CLASS_TO_IDX["V"]][root_mask > 0].mean()) if root_mask.any() else 0.0,
        "prior_A_mean": float(prior[PIXEL_CLASS_TO_IDX["A"]][root_mask > 0].mean()) if root_mask.any() else 0.0,
        "disagreement_pct": float((pred != weak).sum() / max(pred.size, 1) * 100.0),
    }
