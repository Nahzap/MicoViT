"""Entrenamiento e inferencia del gate Stage1 multiclas (Bg / M- / M+)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..common.logging_utils import get_logger
from .fusion import DEFAULT_WEIGHTS, js_divergence_multiclass
from .gate_classes import GATE_CLASS_NAMES, GATE_IDX_TO_CLASS, decode_gate_indices
from .gpu_pipeline import EpochPlan, GPUImageBatch, iter_image_batches, plan_epoch_gate, split_by_image
from .train_gpu import GPUTrainHistory, _pick_input

log = get_logger("phase_d.gate")


def _class_weights_from_df(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = np.array(
        [(df["stage1"] == "Background").sum(), (df["stage1"] == "Mminus").sum(), (df["stage1"] == "Mplus").sum()],
        dtype=np.float64,
    )
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(GATE_CLASS_NAMES) * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _multiclass_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    from .gate_training_protocol import compute_gate_metrics

    return compute_gate_metrics(logits, labels)


@dataclass
class GateEnsembleGPU:
    branch_a: nn.Module
    branch_b: nn.Module
    branch_c: nn.Module
    weights: dict = None
    device: torch.device = torch.device("cuda")

    def to(self, device: torch.device) -> "GateEnsembleGPU":
        self.device = device
        self.branch_a.to(device).eval()
        self.branch_b.to(device).eval()
        self.branch_c.to(device).eval()
        return self


def evaluate_gate_branch_gpu(
    model: nn.Module,
    val_plan: EpochPlan,
    branch: str,
    device: torch.device,
    *,
    batch_size: int = 16,
    class_weights: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    desc: str = "val",
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n = 0
    pbar = tqdm(
        iter_image_batches(
            val_plan, batch_size=batch_size, device=device, label_mode="gate", max_batches=max_batches
        ),
        total=max_batches,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in pbar:
            x = _pick_input(batch, branch)
            y = batch.labels.long()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    pbar.close()
    logits = np.concatenate(all_logits) if all_logits else np.zeros((0, len(GATE_CLASS_NAMES)))
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)
    metrics = _multiclass_metrics(logits, labels) if len(labels) else {"acc": 0, "macro_f1": 0, "per_class_recall": {}}
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train_gate_branch_gpu(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    branch: str,
    device: torch.device,
    *,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    backbone_lr_factor: float = 0.1,
    checkpoint_dir: Optional[Path] = None,
    branch_name: str = "gate_branch_a",
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    max_bg_per_image: int = 50,
    grad_clip: float = 5.0,
) -> GPUTrainHistory:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    class_weights = _class_weights_from_df(train_df, device)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_factor})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    history = GPUTrainHistory()
    best_state: Optional[dict] = None

    for ep in range(1, epochs + 1):
        train_plan = plan_epoch_gate(train_df, max_bg_per_image=max_bg_per_image, seed=ep)
        val_plan = plan_epoch_gate(val_df, max_bg_per_image=None, shuffle_images=False, seed=0)

        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        pbar = tqdm(
            iter_image_batches(
                train_plan,
                batch_size=batch_size,
                device=device,
                label_mode="gate",
                max_batches=max_train_batches,
            ),
            total=max_train_batches,
            desc=f"[{branch_name}] ep{ep}/{epochs}",
            dynamic_ncols=True,
        )
        for batch in pbar:
            x = _pick_input(batch, branch)
            y = batch.labels.long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], grad_clip)
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{running / max(n, 1):.3f}")
        pbar.close()

        train_loss = running / max(n, 1)
        val = evaluate_gate_branch_gpu(
            model,
            val_plan,
            branch,
            device,
            batch_size=batch_size,
            class_weights=class_weights,
            max_batches=max_val_batches,
            desc=f"[{branch_name}] val",
        )
        elapsed = time.time() - t0

        history.epochs.append(ep)
        history.train_loss.append(train_loss)
        history.val_loss.append(val["loss"])
        history.val_auroc.append(val["macro_f1"])
        history.val_acc.append(val["acc"])
        history.val_f1.append(val["macro_f1"])
        history.elapsed_s.append(elapsed)

        improved = val["acc"] > history.best_val_auroc
        if improved:
            history.best_val_auroc = val["acc"]
            history.best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": ep,
                        "acc": val["acc"],
                        "macro_f1": val["macro_f1"],
                        "num_classes": len(GATE_CLASS_NAMES),
                        "gate_mode": True,
                    },
                    checkpoint_dir / f"{branch_name}_best.pt",
                )

        flag = "*" if improved else " "
        log.info(
            f"[{branch_name}] ep {ep}/{epochs} {flag} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"acc={val['acc']:.4f} macro_f1={val['macro_f1']:.4f} "
            f"recall={val.get('per_class_recall', {})} ({elapsed:.1f}s)"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def fuse_gate_probabilities(probs: dict[str, np.ndarray], weights: Optional[dict[str, float]] = None) -> np.ndarray:
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w[k] for k in probs)
    fused = None
    for key, arr in probs.items():
        weight = w.get(key, 0.0) / total_w
        fused = arr * weight if fused is None else fused + arr * weight
    if fused is None:
        raise ValueError("Sin probabilidades para fusionar")
    fused = fused / np.clip(fused.sum(axis=1, keepdims=True), 1e-8, None)
    return fused


@torch.no_grad()
def infer_image_gate_gpu(
    image_path: str | Path,
    ensemble: GateEnsembleGPU,
    tiles_index_path: Optional[Path] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    from ..common.io import read_table
    from ..common.paths import get_paths
    from .gpu_pipeline import plan_epoch_gate

    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = df[df["image_path"] == rel].copy().reset_index(drop=True)
    if sub.empty:
        log.warning(f"No hay tiles para {rel} en el manifest")
        return sub

    plan = plan_epoch_gate(sub, max_bg_per_image=None, shuffle_images=False, shuffle_tiles_within_image=False)
    dev = ensemble.device
    pa_all, pb_all, pc_all = [], [], []
    rows_all, cols_all = [], []

    iterator = tqdm(
        iter_image_batches(plan, batch_size=batch_size, device=dev, label_mode="gate"),
        total=(len(sub) + batch_size - 1) // batch_size,
        desc=f"gate {Path(rel).name}",
        dynamic_ncols=True,
    )
    for batch in iterator:
        la = ensemble.branch_a(batch.rgb)
        lb = ensemble.branch_b(batch.seg)
        lc = ensemble.branch_c(batch.freq)
        pa_all.append(torch.softmax(la.float(), dim=-1).cpu().numpy())
        pb_all.append(torch.softmax(lb.float(), dim=-1).cpu().numpy())
        pc_all.append(torch.softmax(lc.float(), dim=-1).cpu().numpy())
        rows_all.extend(batch.rows.cpu().tolist())
        cols_all.extend(batch.cols.cpu().tolist())

    p_a = np.concatenate(pa_all) if pa_all else np.zeros((0, len(GATE_CLASS_NAMES)))
    p_b = np.concatenate(pb_all) if pb_all else np.zeros((0, len(GATE_CLASS_NAMES)))
    p_c = np.concatenate(pc_all) if pc_all else np.zeros((0, len(GATE_CLASS_NAMES)))
    probs = {"A": p_a, "B": p_b, "C": p_c}
    p_fused = fuse_gate_probabilities(probs, weights=ensemble.weights or DEFAULT_WEIGHTS)
    pred_idx = p_fused.argmax(axis=1)
    consensus = 1.0 - np.clip(js_divergence_multiclass(probs) / np.log(2.0), 0, 1)

    order_df = pd.DataFrame(
        {
            "row": rows_all,
            "col": cols_all,
            "p_bg": p_fused[:, 0],
            "p_mminus": p_fused[:, 1],
            "p_mplus": p_fused[:, 2],
            "p_fused_max": p_fused.max(axis=1),
            "consensus": consensus,
            "gate_pred_idx": pred_idx,
            "stage1_pred": decode_gate_indices(pred_idx),
        }
    )
    merged = sub.merge(order_df, on=["row", "col"], how="left")
    merged["stage1"] = merged["stage1_pred"]
    merged["is_mplus"] = (merged["gate_pred_idx"] == 2).astype(int)
    return merged


def load_gate_am_ensemble(
    *,
    backbone: str = "dinov2_vits14",
    num_classes: int = 3,
    device: torch.device | None = None,
    ckpt_dir: Path | None = None,
    weights_dir: Path | None = None,
) -> GateEnsembleGPU:
    """Carga las 3 ramas del gate AM desde `models/checkpoints/gate_am/`."""
    from ..common.paths import get_paths
    from .models import build_branch_a, build_branch_b, build_branch_c

    paths = get_paths()
    ckpt_dir = ckpt_dir or (paths.root / "models" / "checkpoints" / "gate_am")
    weights_dir = weights_dir or (paths.root / "models" / "weights")
    device = device or torch.device("cuda")

    a = build_branch_a(backbone_name=backbone, num_classes=num_classes)
    b = build_branch_b(weights_path=weights_dir / "u2netp.pth", num_classes=num_classes)
    c = build_branch_c(num_classes=num_classes)

    for model, name in ((a, "gate_branch_a"), (b, "gate_branch_b"), (c, "gate_branch_c")):
        p = ckpt_dir / f"{name}_best.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"Falta checkpoint del gate AM: {p}. Entrena con: python run.py train-gate-am"
            )
        st = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        log.info(f"[Gate] {name} acc={st.get('acc', 0):.4f}")

    return GateEnsembleGPU(
        branch_a=a, branch_b=b, branch_c=c, weights=DEFAULT_WEIGHTS, device=device
    ).to(device)


def build_gate_ensemble_from_trained(
    *,
    branch_a: nn.Module,
    branch_b: nn.Module,
    branch_c: nn.Module,
    device: torch.device,
) -> GateEnsembleGPU:
    """Ensambla ramas ya entrenadas en memoria (post train-gate-am)."""
    return GateEnsembleGPU(
        branch_a=branch_a, branch_b=branch_b, branch_c=branch_c, weights=DEFAULT_WEIGHTS, device=device
    ).to(device)
