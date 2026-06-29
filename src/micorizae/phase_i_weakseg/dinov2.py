"""Puente de entrenamiento DINOv2 para mascaras morfologicas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..common.logging_utils import get_logger
from ..common.run_outputs import RunOutputs

log = get_logger("phase_i_weakseg_dino")


class DinoPatchDataset(Dataset):
    def __init__(self, patches_root: Path):
        self.images = sorted((patches_root / "images").glob("*.png"))
        self.masks = {p.name: p for p in (patches_root / "masks").glob("*.png")}
        self.valid = [p for p in self.images if p.name in self.masks]
        if not self.valid:
            raise ValueError(f"No se encontraron pares imagen/mascara en {patches_root}")

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int):
        img_p = self.valid[idx]
        msk_p = self.masks[img_p.name]
        img = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.float32) / 255.0
        msk = np.asarray(Image.open(msk_p), dtype=np.int64)
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        msk_t = torch.from_numpy(msk).long()
        return img_t, msk_t


@dataclass
class DinoTrainConfig:
    num_classes: int = 6
    model_name: str = "facebook/dinov2-small"
    batch_size: int = 2
    epochs: int = 2
    lr: float = 1e-3
    freeze_backbone: bool = True


def _build_dino_seg_model(model_name: str, num_classes: int, freeze_backbone: bool):
    try:
        from transformers import AutoImageProcessor, Dinov2ForSemanticSegmentation
    except ImportError as e:
        raise ImportError(
            "Falta `transformers`. Instala con: pip install transformers"
        ) from e

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2ForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for n, p in model.named_parameters():
            if "classifier" not in n and "decode_head" not in n:
                p.requires_grad = False
    return proc, model


def train_dinov2_linear_head(
    *,
    patches_root: Path,
    cfg: DinoTrainConfig,
) -> RunOutputs:
    ds = DinoPatchDataset(patches_root)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    _, model = _build_dino_seg_model(cfg.model_name, cfg.num_classes, cfg.freeze_backbone)
    run = RunOutputs.create("weakseg_dinov2_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No hay parametros entrenables en DINOv2 (revisa freeze_backbone).")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    history = []
    for ep in range(cfg.epochs):
        losses = []
        for imgs, masks in dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(pixel_values=imgs)
            logits = out.logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        ep_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": ep + 1, "loss": ep_loss})
        log.info(f"[DINOv2] epoch={ep+1}/{cfg.epochs} loss={ep_loss:.5f}")

    ckpt = run.root / "dinov2_linear_head.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "history": history}, ckpt)
    lines = [
        f"# DINOv2 Fine-Tuning — {run.run_id}",
        "",
        f"- model_name: `{cfg.model_name}`",
        f"- epochs: `{cfg.epochs}`",
        f"- batch_size: `{cfg.batch_size}`",
        f"- lr: `{cfg.lr}`",
        f"- freeze_backbone: `{cfg.freeze_backbone}`",
        f"- n_patches: `{len(ds)}`",
        "",
        f"- checkpoint: `{ckpt}`",
    ]
    (run.reports / "run_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return run

