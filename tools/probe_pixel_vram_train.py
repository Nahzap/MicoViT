#!/usr/bin/env python3
"""VRAM probe Stage2-Pixel con 5 clases + tensores prior (train real)."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

import config as user_config
from micorizae.phase_e_stage2.pixel_class_map import NUM_PIXEL_CLASSES
from micorizae.phase_e_stage2.pixel_vit_model import build_pixel_morph_vit


def probe(bs: int, *, freeze: bool, amp: bool) -> tuple[bool, float]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = build_pixel_morph_vit(
        backbone_name=str(getattr(user_config, "STAGE2_PIXEL_VIT_MODEL", "dinov2_vits14")),
        freeze_backbone=freeze,
    ).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sz = int(getattr(user_config, "STAGE2_PIXEL_INPUT_SIZE", 224))

    try:
        x = torch.randn(bs, 3, sz, sz, device=device)
        y = torch.randint(0, NUM_PIXEL_CLASSES, (bs, sz, sz), device=device)
        prior_e = torch.randn(bs, NUM_PIXEL_CLASSES, sz, sz, device=device)
        ves = torch.rand(bs, sz, sz, device=device)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            ih = torch.softmax(logits, dim=1)[:, 1]
            loss = loss + F.mse_loss(ih, prior_e[:, 1])
            loss = loss + F.binary_cross_entropy_with_logits(logits[:, 2], ves)
        loss.backward()
        opt.step()
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        ok = peak_gb < 7.0
        del model, opt, x, y, logits, loss, prior_e, ves
        torch.cuda.empty_cache()
        gc.collect()
        return ok, peak_gb
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return False, 99.0
        raise


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA no disponible")
        sys.exit(1)
    freeze = bool(getattr(user_config, "STAGE2_PIXEL_FREEZE_BACKBONE", True))
    amp = bool(getattr(user_config, "STAGE2_PIXEL_USE_AMP", True))
    print(f"freeze_backbone={freeze} amp={amp} n_classes={NUM_PIXEL_CLASSES}")

    best = 1
    for bs in [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64]:
        ok, peak = probe(bs, freeze=freeze, amp=amp)
        flag = "OK" if ok else "OOM"
        print(f"  batch={bs:2d} peak={peak:.2f} GB -> {flag}")
        if ok:
            best = bs
    print(f"RECOMENDADO: batch={best}")


if __name__ == "__main__":
    main()
