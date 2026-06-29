"""Ramas Stage2 multiclas e (M+ tiles) — extienden la arquitectura Stage1."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch.nn as nn

from ..phase_d_stage1.models import BranchFrequency, BranchSegmentation, BranchSemantic, count_parameters


class BranchSemanticMC(BranchSemantic):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "dinov2_vits14",
        head_hidden: int = 256,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__(
            backbone_name=backbone_name,
            head_hidden=head_hidden,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )


class BranchSegmentationMC(BranchSegmentation):
    def __init__(
        self,
        num_classes: int,
        weights_path: Optional[Path] = None,
        head_hidden: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__(
            weights_path=weights_path,
            head_hidden=head_hidden,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
        )
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Linear(64, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )


class BranchFrequencyMC(BranchFrequency):
    def __init__(self, num_classes: int, head_hidden: int = 128, dropout: float = 0.2):
        super().__init__(head_hidden=head_hidden, dropout=dropout)
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Linear(256, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )


def build_branch_a_mc(num_classes: int, backbone_name: str = "dinov2_vits14", **kwargs) -> BranchSemanticMC:
    return BranchSemanticMC(num_classes=num_classes, backbone_name=backbone_name, **kwargs)


def build_branch_b_mc(num_classes: int, weights_path: Optional[Path] = None, **kwargs) -> BranchSegmentationMC:
    return BranchSegmentationMC(num_classes=num_classes, weights_path=weights_path, **kwargs)


def build_branch_c_mc(num_classes: int, **kwargs) -> BranchFrequencyMC:
    return BranchFrequencyMC(num_classes=num_classes, **kwargs)


__all__ = [
    "BranchSemanticMC",
    "BranchSegmentationMC",
    "BranchFrequencyMC",
    "build_branch_a_mc",
    "build_branch_b_mc",
    "build_branch_c_mc",
    "count_parameters",
]
