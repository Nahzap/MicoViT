"""Inferencia Stage2 sobre tiles M+ (gate Stage1) — 100% CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..common.io import read_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_d_stage1.gpu_pipeline import iter_image_batches, plan_epoch
from .class_map import Stage2ClassMap, load_stage2_class_map
from .fusion import DEFAULT_WEIGHTS, entropy, fuse_probabilities_mc, js_divergence_mc
from .gpu_pipeline import GPUStage2Batch, iter_image_batches_stage2, plan_epoch_stage2

log = get_logger("phase_e.infer_gpu")


@dataclass
class StageTwoEnsembleGPU:
    branch_a: nn.Module
    branch_b: nn.Module
    branch_c: nn.Module
    class_map: Stage2ClassMap
    temp_a: float = 1.0
    temp_b: float = 1.0
    temp_c: float = 1.0
    weights: dict | None = None
    device: torch.device = torch.device("cuda")

    def to(self, device: torch.device) -> "StageTwoEnsembleGPU":
        self.device = device
        self.branch_a.to(device).eval()
        self.branch_b.to(device).eval()
        self.branch_c.to(device).eval()
        return self


def _softmax_np(logits: torch.Tensor) -> np.ndarray:
    p = torch.softmax(logits.float(), dim=-1)
    return p.cpu().numpy()


@torch.no_grad()
def infer_tiles_stage2_gpu(
    tiles_df: pd.DataFrame,
    ensemble: StageTwoEnsembleGPU,
    batch_size: int = 32,
    use_amp: bool = False,
) -> pd.DataFrame:
    """Inferencia Stage2 sobre un sub-DataFrame de tiles (típicamente M+ del gate)."""
    if tiles_df.empty:
        return tiles_df.copy()

    sub = tiles_df.copy().reset_index(drop=True)
    plan = plan_epoch_stage2(sub, shuffle_images=False, shuffle_tiles_within_image=False)
    cmap = ensemble.class_map

    rows_all, cols_all = [], []
    pa_all, pb_all, pc_all = [], [], []

    dev = ensemble.device
    iterator = tqdm(
        iter_image_batches_stage2(plan, cmap, batch_size=batch_size, device=dev),
        total=max(1, (len(sub) + batch_size - 1) // batch_size),
        desc=f"infer_s2 {cmap.lineage}",
        dynamic_ncols=True,
    )
    for batch in iterator:
        with torch.autocast(device_type=dev.type, enabled=use_amp and dev.type == "cuda"):
            la = ensemble.branch_a(batch.rgb) / ensemble.temp_a
            lb = ensemble.branch_b(batch.seg) / ensemble.temp_b
            lc = ensemble.branch_c(batch.freq) / ensemble.temp_c
        pa_all.append(_softmax_np(la))
        pb_all.append(_softmax_np(lb))
        pc_all.append(_softmax_np(lc))
        rows_all.extend(batch.rows.cpu().tolist())
        cols_all.extend(batch.cols.cpu().tolist())

    p_a = np.concatenate(pa_all, axis=0) if pa_all else np.zeros((0, cmap.num_classes))
    p_b = np.concatenate(pb_all, axis=0) if pb_all else np.zeros((0, cmap.num_classes))
    p_c = np.concatenate(pc_all, axis=0) if pc_all else np.zeros((0, cmap.num_classes))
    probs = {"A": p_a, "B": p_b, "C": p_c}
    p_fused = fuse_probabilities_mc(probs, weights=ensemble.weights or DEFAULT_WEIGHTS)
    ent = entropy(p_fused)
    consensus = 1.0 - np.clip(js_divergence_mc(probs) / np.log(2.0), 0, 1)
    pred_idx = p_fused.argmax(axis=-1)

    order_df = pd.DataFrame({
        "row": rows_all,
        "col": cols_all,
        "stage2_pred_idx": pred_idx,
        "stage2_pred": [cmap.idx_to_class[int(i)] for i in pred_idx],
        "stage2_entropy": ent,
        "stage2_consensus": consensus,
    })
    for i, ch in enumerate(cmap.classes):
        order_df[f"p_s2_{ch}"] = p_fused[:, i]

    merged = sub.merge(order_df, on=["row", "col"], how="left")
    return merged


@torch.no_grad()
def infer_image_stage2_gpu(
    image_path: str | Path,
    ensemble: StageTwoEnsembleGPU,
    stage1_df: pd.DataFrame,
    batch_size: int = 32,
    use_amp: bool = False,
) -> pd.DataFrame:
    """Stage2 solo en tiles con `stage1_pred==1` (o `stage1==Mplus` si no hay pred)."""
    paths = get_paths()
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = stage1_df[stage1_df["image_path"] == rel].copy()
    if sub.empty:
        log.warning(f"No hay tiles Stage1 para {rel}")
        return sub

    if "is_mplus" in sub.columns:
        mplus = sub[sub["is_mplus"] == 1].copy()
    elif "stage1_pred" in sub.columns:
        sp = sub["stage1_pred"]
        if sp.dtype == object or str(sp.dtype) == "string":
            mplus = sub[sp == "Mplus"].copy()
        else:
            mplus = sub[sp == 1].copy()
    else:
        mplus = sub[sub["stage1"] == "Mplus"].copy()

    if mplus.empty:
        out = sub.copy()
        for col in ("stage2_pred_idx", "stage2_pred", "stage2_entropy", "stage2_consensus"):
            out[col] = np.nan
        return out

    s2 = infer_tiles_stage2_gpu(mplus, ensemble, batch_size=batch_size, use_amp=use_amp)
    drop_cols = [c for c in s2.columns if c in sub.columns and c not in ("row", "col")]
    out = sub.merge(s2.drop(columns=drop_cols, errors="ignore"), on=["row", "col"], how="left")
    return out
