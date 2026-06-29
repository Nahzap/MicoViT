"""Fine-tune parcial DINOv2 + LoRA (E6 / E6b) para gate AM.

Referencias implementadas:
- S1 ExPLoRA (Khanna et al. 2024; Microscopy Foundation Hub 2024-25):
  LoRA rank-bajo en bloques ViT tardios (9-11) con backbone congelado;
  `partial_lora` combina LoRA + unfreeze parcial de ultimos bloques.
- S2 Adaptacion de dominio estilo Cell-DINO (arXiv 2604.10609, 2025):
  fine-tune supervisado corto sobre tiles AM nativos cuando embeddings
  frozen presentan domain_shift (G1 recall M-/M+ < umbral).
  No hay pretrain SSL adicional; LoRA en AM tiles actua como domain adapter.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import torch
import torch.nn as nn

log = logging.getLogger("micorizae.gate_dino_finetune")


class LoRALinear(nn.Module):
    """Adaptador low-rank sobre nn.Linear (PEFT-style)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank debe ser >= 1, recibido {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        in_f, out_f = base.in_features, base.out_features
        self.lora_a = nn.Linear(in_f, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def _dino_blocks(backbone: nn.Module) -> nn.ModuleList:
    if hasattr(backbone, "blocks"):
        return backbone.blocks
    if hasattr(backbone, "model") and hasattr(backbone.model, "blocks"):
        return backbone.model.blocks
    raise AttributeError("backbone DINO sin atributo blocks")


def freeze_dino_backbone(backbone: nn.Module) -> int:
    """Congela todos los parametros del backbone DINO."""
    n = 0
    root = backbone.model if hasattr(backbone, "model") else backbone
    for p in root.parameters():
        p.requires_grad = False
        n += 1
    return n


def unfreeze_dino_last_n_blocks(backbone: nn.Module, last_n: int) -> tuple[int, int]:
    """Descongela solo los ultimos `last_n` bloques ViT (E6)."""
    blocks = _dino_blocks(backbone)
    n_blocks = len(blocks)
    last_n = max(0, min(int(last_n), n_blocks))
    start = n_blocks - last_n
    unfrozen = 0
    for i in range(start, n_blocks):
        for p in blocks[i].parameters():
            p.requires_grad = True
            unfrozen += 1
    log.info(
        f"[DINO finetune] blocks {start}-{n_blocks - 1} entrenables "
        f"({unfrozen} tensors, last_n={last_n})"
    )
    return start, unfrozen


def _parse_block_indices(spec: str, n_blocks: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0:
            idx = n_blocks + idx
        if 0 <= idx < n_blocks:
            out.append(idx)
    return sorted(set(out))


def apply_lora_to_dino_blocks(
    backbone: nn.Module,
    *,
    block_indices: Sequence[int],
    rank: int = 8,
    alpha: float = 16.0,
    target: str = "qkv",
) -> list[LoRALinear]:
    """Inyecta LoRA en attn.qkv de bloques indicados (E6b)."""
    blocks = _dino_blocks(backbone)
    adapters: list[LoRALinear] = []
    target = (target or "qkv").lower()
    for bi in block_indices:
        if bi < 0 or bi >= len(blocks):
            continue
        block = blocks[bi]
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "qkv"):
            log.warning(f"[DINO LoRA] block {bi} sin attn.qkv; omitido")
            continue
        if target not in {"qkv", "qv"}:
            raise ValueError(f"target LoRA no soportado: {target!r}")
        wrapped = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
        attn.qkv = wrapped
        adapters.append(wrapped)
    log.info(
        f"[DINO LoRA] {len(adapters)} adaptadores rank={rank} alpha={alpha} "
        f"blocks={list(block_indices)} target={target}"
    )
    return adapters


def configure_dino_finetune(
    backbone: nn.Module,
    *,
    mode: str = "none",
    last_n_blocks: int = 2,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_blocks: str = "9,10,11",
    lora_target: str = "qkv",
) -> dict:
    """Configura freeze parcial + LoRA segun modo (E6/E6b)."""
    mode = (mode or "none").lower()
    freeze_dino_backbone(backbone)
    info: dict = {"mode": mode, "last_n_blocks": 0, "lora_adapters": 0}
    if mode in {"none", ""}:
        return info
    blocks = _dino_blocks(backbone)
    n_blocks = len(blocks)
    if mode in {"partial", "partial_lora", "full"}:
        if mode == "full":
            last_n = n_blocks
        else:
            last_n = int(last_n_blocks)
        _, unfrozen = unfreeze_dino_last_n_blocks(backbone, last_n)
        info["last_n_blocks"] = last_n
        info["unfrozen_tensors"] = unfrozen
    if mode in {"partial_lora", "lora"} or (mode == "partial" and lora_rank > 0):
        indices = _parse_block_indices(lora_blocks, n_blocks)
        if not indices and last_n_blocks > 0:
            indices = list(range(max(0, n_blocks - int(last_n_blocks)), n_blocks))
        adapters = apply_lora_to_dino_blocks(
            backbone,
            block_indices=indices,
            rank=lora_rank,
            alpha=lora_alpha,
            target=lora_target,
        )
        info["lora_adapters"] = len(adapters)
        info["lora_blocks"] = indices
    return info


def collect_trainable_param_groups(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    backbone_lr_factor: float = 0.1,
) -> list[dict]:
    """Separa encoder/head vs backbone/LoRA para optimizador."""
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name or "gate_head" in name or name.startswith("head."):
            head_params.append(p)
        elif "lora_" in name:
            backbone_params.append(p)
        elif "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    groups: list[dict] = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr * backbone_lr_factor})
    return groups
