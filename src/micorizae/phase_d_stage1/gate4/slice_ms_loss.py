"""Multi-Similarity Loss (Wang et al. CVPR 2019) y variante Slice MS.

Extensiones (Slice-MS pura):
- Minería semi-hard (Wang 2019).
- Banda confundible OR semi-hard (curriculum desde band_start_epoch).
- Pesos dirigidos por par gold→neg (mining asimétrico M-/M+).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSimilarityLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        *,
        hard_mining: bool = False,
        mining_margin: float = 0.1,
        confusable_pairs: tuple[tuple[int, int], ...] = (),
        confusable_neg_weight: float = 1.0,
        confusable_guard_neg_weight: float = 1.0,
        confusable_directed_weights: tuple[tuple[int, int, float], ...] = (),
        confusable_base: float | None = None,
        confusable_band_low: float | None = None,
        confusable_band_high: float | None = None,
        band_start_epoch: int = 1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.hard_mining = hard_mining
        self.mining_margin = mining_margin
        self.confusable_pairs = tuple(tuple(int(x) for x in p) for p in confusable_pairs)
        self.confusable_neg_weight = confusable_neg_weight
        self.confusable_guard_neg_weight = confusable_guard_neg_weight
        self.confusable_directed_weights = confusable_directed_weights
        self.confusable_base = confusable_base
        self.confusable_band_low = confusable_band_low
        self.confusable_band_high = confusable_band_high
        self.band_start_epoch = max(1, int(band_start_epoch))

    def _confusable_neg_mask(self, labels: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        if not self.confusable_pairs:
            return torch.zeros_like(neg_mask)
        li = labels.view(-1, 1)
        lj = labels.view(1, -1)
        mask = torch.zeros_like(neg_mask)
        for a, b in self.confusable_pairs:
            mask |= ((li == a) & (lj == b)) | ((li == b) & (lj == a))
        return mask & neg_mask

    def _build_neg_weight_matrix(
        self, labels: torch.Tensor, neg_mask: torch.Tensor, sim: torch.Tensor
    ) -> torch.Tensor:
        weight_mat = torch.ones_like(sim)
        if not self.confusable_pairs and not self.confusable_directed_weights:
            return weight_mat
        li = labels.view(-1, 1)
        lj = labels.view(1, -1)

        for gold_idx, neg_idx, w in self.confusable_directed_weights:
            if w == 1.0:
                continue
            directed = (li == gold_idx) & (lj == neg_idx) & neg_mask
            weight_mat = torch.where(directed, torch.full_like(sim, w), weight_mat)

        for a, b in self.confusable_pairs:
            pair = ((li == a) & (lj == b)) | ((li == b) & (lj == a))
            pair = pair & neg_mask
            if not pair.any():
                continue
            if self.confusable_directed_weights:
                covered = torch.zeros_like(pair)
                for gold_idx, neg_idx, _ in self.confusable_directed_weights:
                    covered |= (li == gold_idx) & (lj == neg_idx)
                pair = pair & ~covered
            if not pair.any():
                continue
            w = self.confusable_guard_neg_weight if 0 in (a, b) else self.confusable_neg_weight
            if w == 1.0:
                continue
            weight_mat = torch.where(pair, torch.full_like(sim, w), weight_mat)
        return weight_mat

    def _apply_confusable_band(
        self,
        neg_keep: torch.Tensor,
        sim: torch.Tensor,
        labels: torch.Tensor,
        neg_mask: torch.Tensor,
        *,
        epoch: int,
    ) -> torch.Tensor:
        if epoch < self.band_start_epoch:
            return neg_keep
        if (
            not self.confusable_pairs
            or self.confusable_band_low is None
            or self.confusable_band_high is None
        ):
            return neg_keep
        conf_neg = self._confusable_neg_mask(labels, neg_mask)
        in_band = (sim >= self.confusable_band_low) & (sim <= self.confusable_band_high)
        return neg_keep | (conf_neg & in_band)

    def forward(self, embed: torch.Tensor, labels: torch.Tensor, *, epoch: int = 1) -> torch.Tensor:
        if embed.ndim != 2:
            raise ValueError(f"embed debe ser (B,D), recibido {tuple(embed.shape)}")
        labels = labels.view(-1)
        b = embed.size(0)
        if b < 2:
            return embed.sum() * 0.0

        embed = embed.float()
        sim = embed @ embed.t()
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(b, dtype=torch.bool, device=embed.device)
        pos_mask = same & ~eye
        neg_mask = ~same

        if self.hard_mining:
            neg_inf = torch.tensor(float("-inf"), device=embed.device)
            pos_inf = torch.tensor(float("inf"), device=embed.device)
            neg_only = torch.where(neg_mask, sim, neg_inf)
            hardest_neg = neg_only.max(dim=1, keepdim=True).values
            pos_only = torch.where(pos_mask, sim, pos_inf)
            hardest_pos = pos_only.min(dim=1, keepdim=True).values
            pos_keep = pos_mask & (sim < hardest_neg + self.mining_margin)
            neg_keep = neg_mask & (sim > hardest_pos - self.mining_margin)
            no_neg = ~neg_mask.any(dim=1, keepdim=True)
            no_pos = ~pos_mask.any(dim=1, keepdim=True)
            pos_keep = torch.where(no_neg.expand_as(pos_mask), pos_mask, pos_keep)
            neg_keep = torch.where(no_pos.expand_as(neg_mask), neg_mask, neg_keep)
            neg_keep = self._apply_confusable_band(
                neg_keep, sim, labels, neg_mask, epoch=epoch
            )
        else:
            pos_keep = pos_mask
            neg_keep = neg_mask

        pos_loss = torch.zeros(b, device=embed.device)
        neg_loss = torch.zeros(b, device=embed.device)

        if pos_keep.any():
            pos_vals = torch.exp(-self.alpha * (sim - self.base))
            pos_vals = pos_vals.masked_fill(~pos_keep, 0.0)
            pos_loss = (1.0 / self.alpha) * torch.log1p(pos_vals.sum(dim=1))

        if neg_keep.any():
            base_mat = torch.full_like(sim, self.base)
            weight_mat = torch.ones_like(sim)
            if self.confusable_pairs or self.confusable_directed_weights:
                conf = self._confusable_neg_mask(labels, neg_mask)
                if self.confusable_base is not None:
                    base_mat = torch.where(conf, torch.full_like(sim, self.confusable_base), base_mat)
                weight_mat = self._build_neg_weight_matrix(labels, neg_mask, sim)
            neg_vals = weight_mat * torch.exp(self.beta * (sim - base_mat))
            neg_vals = neg_vals.masked_fill(~neg_keep, 0.0)
            neg_loss = (1.0 / self.beta) * torch.log1p(neg_vals.sum(dim=1))

        has_pos = pos_keep.any(dim=1)
        has_neg = neg_keep.any(dim=1)
        active = has_pos & has_neg
        if not active.any():
            return embed.sum() * 0.0
        return (pos_loss[active] + neg_loss[active]).mean()


class SliceMultiSimilarityLoss(nn.Module):
    """L_slice_MS = (1/S) sum_s L_MS(e^(s), y)."""

    def __init__(
        self,
        num_slices: int = 4,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        *,
        hard_mining: bool = False,
        mining_margin: float = 0.1,
        confusable_pairs: tuple[tuple[int, int], ...] = (),
        confusable_neg_weight: float = 1.0,
        confusable_guard_neg_weight: float = 1.0,
        confusable_directed_weights: tuple[tuple[int, int, float], ...] = (),
        confusable_base: float | None = None,
        confusable_band_low: float | None = None,
        confusable_band_high: float | None = None,
        band_start_epoch: int = 1,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.ms = MultiSimilarityLoss(
            alpha=alpha,
            beta=beta,
            base=base,
            hard_mining=hard_mining,
            mining_margin=mining_margin,
            confusable_pairs=confusable_pairs,
            confusable_neg_weight=confusable_neg_weight,
            confusable_guard_neg_weight=confusable_guard_neg_weight,
            confusable_directed_weights=confusable_directed_weights,
            confusable_base=confusable_base,
            confusable_band_low=confusable_band_low,
            confusable_band_high=confusable_band_high,
            band_start_epoch=band_start_epoch,
        )

    def forward(self, embed: torch.Tensor, labels: torch.Tensor, *, epoch: int = 1) -> torch.Tensor:
        if embed.size(-1) % self.num_slices != 0:
            raise ValueError(
                f"embed dim {embed.size(-1)} no divisible por num_slices {self.num_slices}"
            )
        chunks = embed.chunk(self.num_slices, dim=-1)
        chunks = [F.normalize(c, dim=-1) for c in chunks]
        losses = [self.ms(c, labels, epoch=epoch) for c in chunks]
        return torch.stack(losses).mean()
