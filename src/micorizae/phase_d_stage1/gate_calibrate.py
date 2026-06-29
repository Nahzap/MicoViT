"""Calibracion post-entrenamiento (temperature scaling + ajuste de sesgos).

Optimiza en holdout train (split test fijo) antes de inferencia final:
- Temperature scaling multiclas
- Sesgos por clase en logits para mejorar Sens M-/M+ sin re-entrenar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .gate_training_protocol import GateTrainProtocol, checkpoint_score, compute_gate_metrics


@dataclass(frozen=True)
class GateCalibration:
    temperature: float = 1.0
    class_bias: tuple[float, ...] = ()  # longitud = n_clases; () = sin sesgo

    def to_dict(self) -> dict:
        return {
            "temperature": float(self.temperature),
            "class_bias": list(self.class_bias),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GateCalibration":
        if not d:
            return cls()
        bias = tuple(float(b) for b in (d.get("class_bias") or ()))
        return cls(
            temperature=float(d.get("temperature", 1.0)),
            class_bias=bias,
        )


def apply_gate_calibration(logits: np.ndarray, calib: GateCalibration) -> np.ndarray:
    if logits.size == 0:
        return logits
    t = max(calib.temperature, 1e-6)
    out = logits.astype(np.float32) / t
    if calib.class_bias:
        bias = np.zeros(out.shape[1], dtype=np.float32)
        n = min(len(calib.class_bias), out.shape[1])
        bias[:n] = np.asarray(calib.class_bias[:n], dtype=np.float32)
        out = out + bias
    return out


def fit_temperature_multiclass(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    init: float = 1.0,
    max_iter: int = 200,
) -> float:
    """Temperature scaling (Guo et al.) para softmax multiclas."""
    if len(labels) == 0:
        return init
    log_t = torch.nn.Parameter(torch.tensor(float(np.log(init))))
    x = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(log_t.exp().item())


def fit_class_biases(
    logits: np.ndarray,
    labels: np.ndarray,
    protocol: GateTrainProtocol,
    *,
    metric: str = "evangelisti_g1",
    grid: float = 0.25,
    span: float = 2.0,
) -> tuple[float, ...]:
    """Busca sesgos en M- y M+ (Bg/Unknown=0) maximizando metrica G1 en val.

    n_clases se infiere de logits.shape[1] (3 o 4); evita el hardcode previo
    que rompia con el modelo gate4 de 4 clases.
    """
    n_cls = int(logits.shape[1]) if logits.ndim == 2 else 3
    if len(labels) == 0:
        return tuple(0.0 for _ in range(n_cls))

    steps = int(round(2 * span / grid)) + 1
    offsets = np.linspace(-span, span, steps)
    idx_mminus, idx_mplus = 1, 2  # Background=0, Unknown=3 quedan fijos en 0
    best_bias = [0.0] * n_cls
    best_score = -1.0

    for b1 in offsets:
        for b2 in offsets:
            bias = np.zeros(n_cls, dtype=np.float32)
            if idx_mminus < n_cls:
                bias[idx_mminus] = float(b1)
            if idx_mplus < n_cls:
                bias[idx_mplus] = float(b2)
            adj = logits + bias
            pred = adj.argmax(axis=1)
            onehot = np.zeros_like(adj)
            onehot[np.arange(len(pred)), pred] = 1.0
            m = compute_gate_metrics(onehot, labels, protocol=protocol)
            score = checkpoint_score(m, metric)
            if score > best_score:
                best_score = score
                best_bias = bias.tolist()
    return tuple(best_bias)


def fit_gate_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    protocol: Optional[GateTrainProtocol] = None,
) -> GateCalibration:
    """Pipeline: T scaling + sesgos M-/M+ (recalibracion post-entrenamiento)."""
    protocol = protocol or GateTrainProtocol()
    if len(labels) == 0:
        return GateCalibration()

    t = fit_temperature_multiclass(logits, labels)
    scaled = logits / max(t, 1e-6)
    bias = fit_class_biases(scaled, labels, protocol, metric=protocol.checkpoint_metric)
    return GateCalibration(temperature=t, class_bias=bias)


def metrics_with_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    calib: GateCalibration,
    protocol: Optional[GateTrainProtocol] = None,
) -> dict:
    adj = apply_gate_calibration(logits, calib)
    return compute_gate_metrics(adj, labels, protocol=protocol)
