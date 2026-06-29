"""Pérdida Slice-MS (académica) y combinada legacy CE+MS."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gate4SliceMSConfig
from .slice_ms_loss import SliceMultiSimilarityLoss


def _build_slice_ms(cfg: Gate4SliceMSConfig) -> SliceMultiSimilarityLoss:
    return SliceMultiSimilarityLoss(
        num_slices=cfg.num_slices,
        alpha=cfg.alpha,
        beta=cfg.beta,
        base=cfg.base,
        hard_mining=cfg.hard_mining,
        mining_margin=cfg.mining_margin,
        confusable_pairs=cfg.confusable_pairs,
        confusable_neg_weight=cfg.confusable_neg_weight,
        confusable_guard_neg_weight=cfg.confusable_guard_neg_weight,
        confusable_base=cfg.confusable_base,
        confusable_band_low=cfg.confusable_band_low,
        confusable_band_high=cfg.confusable_band_high,
        confusable_directed_weights=cfg.confusable_directed_weights,
        band_start_epoch=cfg.band_start_epoch,
    )


class Gate4SliceMSLossOnly(nn.Module):
    """L_total = L_slice_MS únicamente (Wang et al. 2019, variante por slices)."""

    def __init__(self, cfg: Gate4SliceMSConfig):
        super().__init__()
        self.cfg = cfg
        self.slice_ms = _build_slice_ms(cfg)

    def forward(
        self,
        embed: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int = 1,
    ) -> dict[str, torch.Tensor]:
        labels = labels.long()
        ms = self.slice_ms(embed, labels, epoch=epoch)
        zero = embed.sum() * 0.0
        return {"total": ms, "gate": zero.detach(), "slice_ms": ms}


class Gate4CombinedLoss(nn.Module):
    """Legacy: L = L_gate + weight * L_slice_MS. Preferir Gate4SliceMSLossOnly."""

    def __init__(
        self,
        cfg: Gate4SliceMSConfig,
        *,
        gate_loss_fn: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.gate_loss_fn = gate_loss_fn
        self.class_weights = class_weights
        self.slice_ms = _build_slice_ms(cfg)

    def forward(
        self,
        logits: torch.Tensor,
        embed: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int = 1,
        ms_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        labels = labels.long()
        if self.gate_loss_fn is not None:
            gate = self.gate_loss_fn(logits, labels)
        else:
            gate = F.cross_entropy(logits, labels, weight=self.class_weights)

        ms = self.slice_ms(embed, labels, epoch=epoch)
        if ms_only:
            total = ms
        elif self.cfg.warmup_epochs > 0:
            ramp = min(1.0, max(0.0, (epoch - 1) / self.cfg.warmup_epochs))
            ms_w = self.cfg.loss_weight * ramp
            total = gate + ms_w * ms
        else:
            total = gate + self.cfg.loss_weight * ms
        return {"total": total, "gate": gate, "slice_ms": ms}
