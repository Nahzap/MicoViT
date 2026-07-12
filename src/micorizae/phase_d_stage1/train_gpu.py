"""DEPRECATED / QUARANTINE — Gen1 Gate ensemble training loop.

Canonical Gate path: ``gate_tile_dino`` + ``gate4``. Do not use for new work.

Bucle de entrenamiento 100% CUDA por rama.

Diferencias con `train.py`:
    - sin DataLoader (CPU bottleneck eliminado)
    - usa `iter_image_batches` que decodifica/transforma en CUDA
    - tqdm muestra progreso por batch con loss/positives en vivo
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..common.logging_utils import get_logger
from .gpu_pipeline import EpochPlan, GPUImageBatch, iter_image_batches, plan_epoch
from .losses import FocalLossBCE

log = get_logger("phase_d.train_gpu")


@dataclass
class GPUTrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    elapsed_s: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_auroc: float = -1.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _compute_metrics(probs: np.ndarray, labels: np.ndarray, thresh: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    pred = (probs >= thresh).astype(np.int32)
    try:
        auroc = float(roc_auc_score(labels, probs))
    except Exception:
        auroc = float("nan")
    return {
        "auroc": auroc,
        "acc": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
    }


def _pick_input(batch: GPUImageBatch, branch: str) -> torch.Tensor:
    if branch == "A":
        return batch.rgb
    if branch == "B":
        return batch.seg
    if branch == "C":
        return batch.freq
    raise ValueError(branch)


def evaluate_branch_gpu(
    model: nn.Module,
    val_plan: EpochPlan,
    branch: str,
    device: torch.device,
    batch_size: int = 16,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
    desc: str = "val",
) -> dict:
    model.eval()
    loss_fn = loss_fn or FocalLossBCE()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n = 0
    pbar = tqdm(
        iter_image_batches(val_plan, batch_size=batch_size, device=device, max_batches=max_batches),
        total=max_batches,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in pbar:
            x = _pick_input(batch, branch)
            y = batch.labels
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    pbar.close()
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    metrics = _compute_metrics(probs, labels) if len(labels) else {"auroc": 0, "acc": 0, "f1": 0}
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train_branch_gpu(
    model: nn.Module,
    train_df,
    val_df,
    branch: str,
    device: torch.device,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    backbone_lr_factor: float = 0.1,
    use_amp: bool = False,
    checkpoint_dir: Optional[Path] = None,
    branch_name: str = "branch",
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    max_neg_per_image: Optional[int] = None,
    grad_clip: float = 5.0,
) -> GPUTrainHistory:
    """Entrena UNA rama usando el pipeline GPU-only."""
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("head", "side", "outconv", "stage1d", "stage2d", "stage3d", "stage4d", "stage5d")):
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_factor})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and device.type == "cuda")
    loss_fn = FocalLossBCE(alpha=0.5, gamma=2.0)

    history = GPUTrainHistory()
    best_state: Optional[dict] = None

    for ep in range(1, epochs + 1):
        train_plan = plan_epoch(
            train_df, max_neg_per_image=max_neg_per_image, seed=ep, shuffle_images=True
        )
        val_plan = plan_epoch(val_df, max_neg_per_image=None, shuffle_images=False, seed=0)

        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        pos_seen = 0
        total = max_train_batches if max_train_batches else None
        pbar = tqdm(
            iter_image_batches(train_plan, batch_size=batch_size, device=device, max_batches=max_train_batches),
            total=total,
            desc=f"[{branch_name}] ep{ep}/{epochs}",
            dynamic_ncols=True,
        )
        for batch in pbar:
            x = _pick_input(batch, branch)
            y = batch.labels
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            pos_seen += int(y.sum().item())
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                avg=f"{running / n:.3f}",
                pos=f"{pos_seen}/{n}",
            )
        pbar.close()
        train_loss = running / max(n, 1)
        val = evaluate_branch_gpu(
            model, val_plan, branch, device, batch_size=batch_size, use_amp=use_amp,
            max_batches=max_val_batches, desc=f"[{branch_name}] val",
        )
        elapsed = time.time() - t0

        history.epochs.append(ep)
        history.train_loss.append(train_loss)
        history.val_loss.append(val["loss"])
        history.val_auroc.append(val["auroc"])
        history.val_acc.append(val["acc"])
        history.val_f1.append(val["f1"])
        history.elapsed_s.append(elapsed)

        improved = val["auroc"] > history.best_val_auroc
        if improved:
            history.best_val_auroc = val["auroc"]
            history.best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(
                    {"model_state_dict": best_state, "epoch": ep, "auroc": val["auroc"]},
                    checkpoint_dir / f"{branch_name}_best.pt",
                )

        flag = "*" if improved else " "
        log.info(
            f"[{branch_name}] ep {ep}/{epochs} {flag} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"AUROC={val['auroc']:.4f} acc={val['acc']:.4f} F1={val['f1']:.4f} "
            f"({elapsed:.1f}s)"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
