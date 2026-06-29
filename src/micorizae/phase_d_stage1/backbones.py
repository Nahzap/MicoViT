"""Multi-backbone selector para Rama A del gate Stage1.

Soporta 4 backbones validados en metric learning interno:

    - 'dinov2_vits14'              (88 MB, ViT-S/14 self-supervised, 384 feat)
    - 'convnextv2_tiny'            (114 MB, ConvNeXt V2 timm, 768 feat)
    - 'deit_small_patch16_224'     (88 MB, ViT-S supervisado, 384 feat)
    - 'resnet50'                   (102 MB, CNN clásico, 2048 feat)

Todos cacheados localmente en `~/.cache/torch/hub` y `~/.cache/huggingface/hub`.
Si HF está sin red, timm carga desde caché siempre que el snapshot exista.

Devuelve siempre `(backbone_module, num_features)` con un `forward(x) -> (B,D)`.
DINOv2 usa **mean pooling de patch tokens con soporte opcional de máscara
espacial** — wrapper con pooling por máscara para preservar gradientes
espaciales (útil para Grad-CAM en L9).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("micorizae.backbones")


SUPPORTED = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "convnextv2_tiny": 768,
    "deit_small_patch16_224": 384,
    "resnet50": 2048,
    "resnet18": 512,
}


class DINOv2Backbone(nn.Module):
    """Wrapper para DINOv2 con mean pooling de patch tokens (no CLS).

    Wrapper DINOv2 con mean pooling de patch tokens (no CLS).
    Soporta `mask` espacial opcional para pooling restringido a la región del
    tile que contiene tejido (útil para tiles parcialmente vacíos del borde).
    """

    def __init__(self, dino_model: nn.Module):
        super().__init__()
        self.model = dino_model
        self.blocks = dino_model.blocks
        self.patch_embed = dino_model.patch_embed
        self.norm = dino_model.norm
        self.patch_size = dino_model.patch_embed.patch_size[0]  # 14

    def _pool_patch_tokens(
        self, patch_tokens: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return patch_tokens.mean(dim=1)

        B, N, D = patch_tokens.shape
        _, _, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        if h_p * w_p != N:
            h_p = w_p = int(N**0.5)

        m = mask.unsqueeze(1).float()
        if m.shape[-1] != W or m.shape[-2] != H:
            m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        m = F.adaptive_avg_pool2d(m, (h_p, w_p)).reshape(B, -1)
        m = (m > 0.25).float()
        active = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (patch_tokens * m.unsqueeze(-1)).sum(dim=1) / active

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model.get_intermediate_layers(x, n=1, reshape=False)
        patch_tokens = out[0]  # (B, N, D); CLS excluido
        return self._pool_patch_tokens(patch_tokens, x, mask)


def _build_dinov2(name: str) -> tuple[nn.Module, int]:
    """Carga DINOv2 desde el caché local (sin red) usando torch.hub source='local'."""
    import os

    repo_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    try:
        raw = torch.hub.load(repo_dir, name, source="local", pretrained=True, trust_repo=True)
    except Exception as e:
        logger.warning(f"hub local falló ({e}); intento descargar...")
        raw = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    return DINOv2Backbone(raw), SUPPORTED[name]


def _build_timm(model_name: str) -> tuple[nn.Module, int]:
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    return model, model.num_features


def _build_resnet(arch: str) -> tuple[nn.Module, int]:
    import torchvision.models as tvm

    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        num_feat = 512
    elif arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        num_feat = 2048
    else:
        raise ValueError(arch)
    m.fc = nn.Identity()
    return m, num_feat


def build_backbone(name: str) -> tuple[nn.Module, int]:
    name = name.lower()
    if name not in SUPPORTED:
        raise ValueError(f"backbone '{name}' no soportado. Disponibles: {list(SUPPORTED)}")
    if name.startswith("dinov2_"):
        return _build_dinov2(name)
    if name in {"convnextv2_tiny", "deit_small_patch16_224"}:
        hf_name = {
            "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "deit_small_patch16_224": "deit_small_patch16_224.fb_in1k",
        }[name]
        return _build_timm(hf_name)
    if name in {"resnet18", "resnet50"}:
        return _build_resnet(name)
    raise ValueError(name)
