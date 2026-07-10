"""ViT DINOv2 segmentación morfológica píxel — Fase 2 (Stage2-Pixel)."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..phase_d_stage1.backbones import build_backbone
from .pixel_class_map import NUM_PIXEL_CLASSES

log = logging.getLogger("micorizae.phase_e.pixel_vit")


class _DINOv2SpatialEncoder(nn.Module):
    """Expone mapa de features espaciales desde DINOv2 hub."""

    def __init__(self, backbone_name: str = "dinov2_vits14"):
        super().__init__()
        raw, feat_dim = build_backbone(backbone_name)
        self.model = raw.model if hasattr(raw, "model") else raw
        self.feat_dim = feat_dim
        self.patch_size = int(self.model.patch_embed.patch_size[0])

    def forward(self, x: torch.Tensor, n: int = 1) -> tuple[torch.Tensor, ...]:
        return self.model.get_intermediate_layers(x, n=n, reshape=True)


class _MultiScaleDecoder(nn.Module):
    """Fusiona 4 mapas de features DINOv2 multi-escala y escala progresivamente."""

    def __init__(self, feat_dim: int, num_classes: int, decoder_channels: int = 256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 4, decoder_channels, kernel_size=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        c = decoder_channels
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c // 2, c // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c // 4),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c // 4, c // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c // 8),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c // 8, num_classes, kernel_size=3, padding=1)

    def forward(self, feats: tuple[torch.Tensor, ...], return_feat: bool = False):
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.head(x)
        if return_feat:
            return logits, x
        return logits


class PixelMorphViT(nn.Module):
    """DINOv2 ViT + decoder convolucional (simple o multi-escala) → logits (B, C, H, W)."""

    def __init__(
        self,
        num_classes: int = NUM_PIXEL_CLASSES,
        backbone_name: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        decoder_channels: int = 256,
        decoder_type: str = "multiscale",
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.encoder = _DINOv2SpatialEncoder(backbone_name)
        d = self.encoder.feat_dim
        c = decoder_channels

        if decoder_type == "multiscale":
            self.decode = _MultiScaleDecoder(d, num_classes, c)
            self.feat_dim_out = c // 8
        else:
            self._simple_body = nn.Sequential(
                nn.Conv2d(d, c, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(c // 2, c // 4, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self._simple_head = nn.Conv2d(c // 4, num_classes, kernel_size=1)
            self.decode = nn.Sequential(self._simple_body, self._simple_head)
            self.feat_dim_out = c // 4

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0 and hasattr(self.encoder.model, "blocks"):
                blocks = self.encoder.model.blocks
                for b in blocks[-unfreeze_last_n:]:
                    for p in b.parameters():
                        p.requires_grad = True
                log.info(f"[PixelMorphViT] Descongelados últimos {unfreeze_last_n} bloques del backbone ViT.")

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        dec_feat = None
        if self.decoder_type == "multiscale":
            feats = self.encoder(x, n=4)
            if return_feat:
                logits, dec_feat = self.decode(feats, return_feat=True)
            else:
                logits = self.decode(feats)
        else:
            feat = self.encoder(x, n=1)[0]
            if return_feat:
                dec_feat = self._simple_body(feat)
                logits = self._simple_head(dec_feat)
            else:
                logits = self.decode(feat)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if return_feat:
            return logits, dec_feat
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)


def build_pixel_morph_vit(
    backbone_name: str = "dinov2_vits14",
    freeze_backbone: bool = True,
    decoder_type: str = "multiscale",
    unfreeze_last_n: int = 0,
) -> PixelMorphViT:
    return PixelMorphViT(
        num_classes=NUM_PIXEL_CLASSES,
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        decoder_type=decoder_type,
        unfreeze_last_n=unfreeze_last_n,
    )
