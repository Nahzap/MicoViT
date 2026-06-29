"""Entrenamiento GPU Stage2 multiclas e por linaje."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from ..common.logging_utils import get_logger
from .class_map import Stage2ClassMap
from .gpu_pipeline import GPUStage2Batch, EpochPlan, iter_image_batches_stage2, plan_epoch_stage2

log = get_logger("phase_e.train_gpu")


@dataclass
class GPUTrainHistoryS2:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_f1_macro: list[float] = field(default_factory=list)
    elapsed_s: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_f1: float = -1.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _pick_input(batch: GPUStage2Batch, branch: str) -> torch.Tensor:
    if branch == "A":
        return batch.rgb
    if branch == "B":
        return batch.seg
    if branch == "C":
        return batch.freq
    raise ValueError(branch)


def _class_weights(train_df, class_map: Stage2ClassMap, device: torch.device) -> torch.Tensor:
    counts = train_df["stage2_idx"].value_counts().reindex(range(class_map.num_classes), fill_value=0)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w.to_numpy(dtype=np.float32), device=device)


def evaluate_branch_gpu_s2(
    model: nn.Module,
    val_plan: EpochPlan,
    class_map: Stage2ClassMap,
    branch: str,
    device: torch.device,
    batch_size: int = 16,
    loss_fn: Optional[nn.Module] = None,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
    desc: str = "val",
) -> dict:
    model.eval()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    all_pred, all_labels = [], []
    total_loss, n = 0.0, 0
    pbar = tqdm(
        iter_image_batches_stage2(
            val_plan, class_map, batch_size=batch_size, device=device, max_batches=max_batches
        ),
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
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_pred.append(pred)
            all_labels.append(y.cpu().numpy())
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    pbar.close()
    pred = np.concatenate(all_pred) if all_pred else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    if len(labels) == 0:
        return {"loss": 0.0, "acc": 0.0, "f1_macro": 0.0}
    return {
        "loss": total_loss / max(n, 1),
        "acc": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }


def train_branch_gpu_s2(
    model: nn.Module,
    train_df,
    val_df,
    class_map: Stage2ClassMap,
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
    grad_clip: float = 5.0,
) -> GPUTrainHistoryS2:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    class_w = _class_weights(train_df, class_map, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

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

    history = GPUTrainHistoryS2()
    best_state: Optional[dict] = None

    for ep in range(1, epochs + 1):
        train_plan = plan_epoch_stage2(train_df, seed=ep, shuffle_images=True)
        val_plan = plan_epoch_stage2(val_df, seed=0, shuffle_images=False)

        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        pbar = tqdm(
            iter_image_batches_stage2(
                train_plan, class_map, batch_size=batch_size, device=device, max_batches=max_train_batches
            ),
            total=max_train_batches,
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
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{running / n:.3f}")
        pbar.close()

        train_loss = running / max(n, 1)
        val = evaluate_branch_gpu_s2(
            model, val_plan, class_map, branch, device,
            batch_size=batch_size, loss_fn=loss_fn, use_amp=use_amp,
            max_batches=max_val_batches, desc=f"[{branch_name}] val",
        )
        elapsed = time.time() - t0

        history.epochs.append(ep)
        history.train_loss.append(train_loss)
        history.val_loss.append(val["loss"])
        history.val_acc.append(val["acc"])
        history.val_f1_macro.append(val["f1_macro"])
        history.elapsed_s.append(elapsed)

        improved = val["f1_macro"] > history.best_val_f1
        if improved:
            history.best_val_f1 = val["f1_macro"]
            history.best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": ep,
                        "f1_macro": val["f1_macro"],
                        "classes": list(class_map.classes),
                        "lineage": class_map.lineage,
                    },
                    checkpoint_dir / f"{branch_name}_best.pt",
                )

        flag = "*" if improved else " "
        log.info(
            f"[{branch_name}] ep {ep}/{epochs} {flag} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"acc={val['acc']:.4f} F1={val['f1_macro']:.4f} ({elapsed:.1f}s)"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
