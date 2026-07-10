"""Losses diferenciables MEViT — alineación prior morfológico + precedencia suave."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .pixel_class_map import PIXEL_CLASS_TO_IDX


@dataclass
class PriorLossWeights:
    ih: float = 0.05
    v: float = 0.05
    prec: float = 0.02
    a: float = 0.0


@dataclass
class PriorLossBreakdown:
    ih: torch.Tensor
    v: torch.Tensor
    prec: torch.Tensor
    a: torch.Tensor
    total: torch.Tensor


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        return F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def prior_ih_loss(logits: torch.Tensor, frangi_e: torch.Tensor, root_mask: torch.Tensor) -> torch.Tensor:
    """MSE entre softmax IH y evidencia Frangi normalizada."""
    logits = _resize_logits(logits, frangi_e)
    probs = F.softmax(logits, dim=1)
    ih = probs[:, PIXEL_CLASS_TO_IDX["IH"]]
    mask = root_mask > 0.5
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)
    return F.mse_loss(ih[mask], frangi_e[mask])


def prior_a_loss(logits: torch.Tensor, arb_e: torch.Tensor, root_mask: torch.Tensor) -> torch.Tensor:
    """MSE entre softmax A y evidencia de arbúsculo (textura fina × gate densidad) v3.

    Simétrico a ``prior_ih_loss``: ancla el canal A del ViT a la evidencia continua A
    (``prior_evidence[:, A]``), restringido al tejido radicular. Solo activa cuando el
    peso ``a`` > 0 (ver ``combine_prior_losses``); en tejido sin evidencia A la señal es
    ~0, así que no fuerza falsos positivos donde el prior no ve arbúsculo.
    """
    logits = _resize_logits(logits, arb_e)
    probs = F.softmax(logits, dim=1)
    a = probs[:, PIXEL_CLASS_TO_IDX["A"]]
    mask = root_mask > 0.5
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)
    return F.mse_loss(a[mask], arb_e[mask])


def prior_v_loss(logits: torch.Tensor, vesicle_mask: torch.Tensor) -> torch.Tensor:
    logits = _resize_logits(logits, vesicle_mask.unsqueeze(1))
    v_logit = logits[:, PIXEL_CLASS_TO_IDX["V"]]
    if vesicle_mask.sum() < 1:
        return torch.tensor(0.0, device=logits.device)
    return F.binary_cross_entropy_with_logits(v_logit, vesicle_mask.clamp(0, 1))


def precedence_loss(logits: torch.Tensor, vesicle_mask: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """Penaliza softmax V > IH en regiones vesiculares weak."""
    logits = _resize_logits(logits, vesicle_mask.unsqueeze(1))
    probs = F.softmax(logits, dim=1)
    ih = probs[:, PIXEL_CLASS_TO_IDX["IH"]]
    pv = probs[:, PIXEL_CLASS_TO_IDX["V"]]
    pa = probs[:, PIXEL_CLASS_TO_IDX["A"]]
    mask = vesicle_mask > 0.5
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)
    l_v_over_ih = F.relu(pv - ih + delta)[mask].mean()
    l_a_over_v = F.relu(pa - pv + delta)[mask].mean()
    return l_v_over_ih + 0.5 * l_a_over_v


def combine_prior_losses(
    logits: torch.Tensor,
    prior_evidence: torch.Tensor,
    vesicle_mask: torch.Tensor,
    *,
    weights: PriorLossWeights,
) -> PriorLossBreakdown:
    """prior_evidence: (B,5,H,W) — canal IH = Frangi, V = circularidad, etc."""
    root = (prior_evidence.sum(dim=1) > 0).float()
    frangi = prior_evidence[:, PIXEL_CLASS_TO_IDX["IH"]]
    l_ih = prior_ih_loss(logits, frangi, root)
    l_v = prior_v_loss(logits, vesicle_mask)
    l_prec = precedence_loss(logits, vesicle_mask)
    # P3.1: canal A solo se calcula si pesa (>0), para no gastar cómputo cuando está apagado.
    if weights.a > 0.0:
        arb = prior_evidence[:, PIXEL_CLASS_TO_IDX["A"]]
        l_a = prior_a_loss(logits, arb, root)
    else:
        l_a = torch.tensor(0.0, device=logits.device)
    total = weights.ih * l_ih + weights.v * l_v + weights.prec * l_prec + weights.a * l_a
    return PriorLossBreakdown(ih=l_ih, v=l_v, prec=l_prec, a=l_a, total=total)
