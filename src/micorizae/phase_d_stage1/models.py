"""Tres ramas del gate Stage1 (M+/M-) — todas devuelven logits (B,) sin sigmoid.

Ramas:
    A) BranchSemantic    — backbone configurable (DINOv2 / ConvNeXtV2 / DeiT / ResNet)
                           + cabezal MLP -> 1 logit.
    B) BranchSegmentation — U2NETP encoder (init desde u2netp.pth) -> GAP -> 1 logit.
    C) BranchFrequency   — small CNN sobre tensor (3, 224, 224) = (log|F|, cosφ, sinφ).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SUPPORTED, build_backbone

log = logging.getLogger("micorizae.models")


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------- BRANCH A — Semantic ---------------------------- #

class BranchSemantic(nn.Module):
    """Backbone configurable + cabezal binario o multiclas gate."""

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        head_hidden: int = 256,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        num_classes: int = 1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.backbone, num_features = build_backbone(backbone_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        out_dim = num_classes if num_classes > 1 else 1
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )
        self.num_features = num_features

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.encode(x, mask=mask)
        logits = self.head(feat)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None and hasattr(self.backbone, "forward"):
            try:
                feat = self.backbone(x, mask=mask)
            except TypeError:
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.mean(dim=list(range(2, feat.dim())))
        return feat

    def forward_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.head(feat)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits


# ------------------------ BRANCH B — Segmentation ---------------------------- #

class BranchSegmentation(nn.Module):
    """Encoder de U2NETP (RSU7..RSU4F) -> GAP -> cabezal binario.

    Carga `u2netp.pth` (entrenado por el usuario para microscopía) como
    inicialización transferida. Sólo el encoder se entrena (decoder descartado
    porque el gate Stage1 es per-tile, no per-pixel).
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        head_hidden: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        from .u2net import U2NETP

        self.u2net = U2NETP(in_ch=3, out_ch=1)

        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                self.u2net.load_state_dict(state, strict=True)
                log.info(f"BranchSegmentation: u2netp pesos cargados desde {weights_path}")
            except RuntimeError as e:
                miss = self.u2net.load_state_dict(state, strict=False)
                log.warning(f"BranchSegmentation: load_state_dict no-strict ({e}); {miss}")
        else:
            log.warning("BranchSegmentation: sin pesos U2NETP, init aleatoria.")

        if freeze_encoder:
            for p in self.u2net.parameters():
                p.requires_grad = False

        # Encoder final feat dim: stage5 produce 64 ch en U2NETP.
        out_dim = num_classes if num_classes > 1 else 1
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Linear(64, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Toma el camino del encoder hasta `stage6` y devuelve features 64-ch."""
        u = self.u2net
        hx1 = u.stage1(x);    h = u.pool12(hx1)
        hx2 = u.stage2(h);    h = u.pool23(hx2)
        hx3 = u.stage3(h);    h = u.pool34(hx3)
        hx4 = u.stage4(h);    h = u.pool45(hx4)
        hx5 = u.stage5(h);    h = u.pool56(hx5)
        hx6 = u.stage6(h)
        return hx6  # (B, 64, H/32, W/32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._encode(x)
        logits = self.head(feat)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits


# ------------------------ BRANCH C — FrequencyNet ---------------------------- #

class _ConvBNReLU(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )


class BranchFrequency(nn.Module):
    """CNN compacta sobre el tensor de frecuencia 3-canal.

    Entrada: (B, 3, 224, 224) con (log|F|_norm, cos phase, sin phase).
    Salida:  (B,) logits binarios.

    ~150k params, fácilmente entrenable en CPU/GPU sin pretrained.
    """

    def __init__(self, head_hidden: int = 128, dropout: float = 0.2, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            _ConvBNReLU(3, 32, k=7, s=2, p=3),     # 112
            nn.MaxPool2d(3, 2, 1),                  # 56
        )
        self.layers = nn.Sequential(
            _ConvBNReLU(32, 64),                    # 56
            _ConvBNReLU(64, 64),
            nn.MaxPool2d(2, 2),                     # 28
            _ConvBNReLU(64, 128),
            _ConvBNReLU(128, 128),
            nn.MaxPool2d(2, 2),                     # 14
            _ConvBNReLU(128, 256),
            _ConvBNReLU(256, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        out_dim = num_classes if num_classes > 1 else 1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Linear(256, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        logits = self.head(x)
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits


# --------------------------- Builders públicos ------------------------------- #

def build_branch_a(backbone_name: str = "dinov2_vits14", *, num_classes: int = 1, **kwargs) -> BranchSemantic:
    return BranchSemantic(backbone_name=backbone_name, num_classes=num_classes, **kwargs)


def build_branch_b(weights_path: Optional[Path] = None, *, num_classes: int = 1, **kwargs) -> BranchSegmentation:
    return BranchSegmentation(weights_path=weights_path, num_classes=num_classes, **kwargs)


def build_branch_c(*, num_classes: int = 1, **kwargs) -> BranchFrequency:
    return BranchFrequency(num_classes=num_classes, **kwargs)
