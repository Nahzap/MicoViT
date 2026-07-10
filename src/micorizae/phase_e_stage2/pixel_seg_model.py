"""U2NETP multiclas para segmentación morfológica píxel — Fase 2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..phase_d_stage1.u2net import U2NETP
from .pixel_class_map import NUM_PIXEL_CLASSES

log = logging.getLogger("micorizae.phase_e.pixel_seg")


class PixelMorphU2Net(nn.Module):
    """U2NETP con salida per-pixel (B, C, H, W) logits."""

    def __init__(
        self,
        num_classes: int = NUM_PIXEL_CLASSES,
        weights_path: Optional[Path] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = U2NETP(in_ch=3, out_ch=num_classes)
        if weights_path is not None and Path(weights_path).exists():
            self._load_pretrained(Path(weights_path))

    def _load_pretrained(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        miss = self.net.load_state_dict(state, strict=False)
        log.info(
            "PixelMorphU2Net: init desde %s (strict=False missing=%d unexpected=%d)",
            path,
            len(miss.missing_keys),
            len(miss.unexpected_keys),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d0, *_ = self.net(x)
        return d0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)


def build_pixel_morph_u2net(weights_path: Optional[Path] = None) -> PixelMorphU2Net:
    return PixelMorphU2Net(num_classes=NUM_PIXEL_CLASSES, weights_path=weights_path)
