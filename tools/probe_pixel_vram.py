#!/usr/bin/env python3
"""Encuentra batch máximo Stage2-Pixel ViT dentro de VRAM dedicada (8 GB)."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

import config as user_config  # noqa: E402
from micorizae.phase_e_stage2.pixel_vit_model import build_pixel_morph_vit  # noqa: E402


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
        x = torch.randn(bs, 3, sz, sz, device=device, dtype=torch.float16 if amp else torch.float32)
        y = torch.randint(0, 5, (bs, sz, sz), device=device)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(x.float() if amp else x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        ok = peak_gb < 7.2  # margen ~800 MB bajo 8 GB dedicada
        del model, opt, x, y, logits, loss
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
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {name} VRAM={total:.1f} GB")

    freeze = bool(getattr(user_config, "STAGE2_PIXEL_FREEZE_BACKBONE", False))
    amp = bool(getattr(user_config, "STAGE2_PIXEL_USE_AMP", True))
    print(f"freeze_backbone={freeze} amp={amp}")

    best = 1
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64]:
        ok, peak = probe(bs, freeze=freeze, amp=amp)
        flag = "OK" if ok else "OOM/shared"
        print(f"  batch={bs:2d} peak={peak:.2f} GB -> {flag}")
        if ok:
            best = bs

    # Si unfrozen no cabe batch>=4, probar frozen
    if best < 4 and not freeze:
        print("\nRe-probe con freeze_backbone=True:")
        for bs in [4, 6, 8, 10, 12, 16]:
            ok, peak = probe(bs, freeze=True, amp=amp)
            flag = "OK" if ok else "OOM/shared"
            print(f"  batch={bs:2d} peak={peak:.2f} GB -> {flag}")
            if ok:
                best = max(best, bs)
        print(f"RECOMENDADO: batch={best} freeze_backbone=True")
    else:
        print(f"RECOMENDADO: batch={best} freeze_backbone={freeze}")


if __name__ == "__main__":
    main()
