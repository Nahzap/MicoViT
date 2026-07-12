"""DEPRECATED / QUARANTINE — Gen1 Gate ensemble inference.

Canonical Gate path: ``gate_tile_dino`` + ``gate4``. Do not use for new work.

Inferencia full-image 100% CUDA.

Para una imagen:
    1) decode_jpeg_gpu (nvJPEG) -> GPUImage en VRAM
    2) recorre tiles en mini-batches CUDA via `iter_image_batches`
    3) 3 ramas forward + temperature scaling
    4) fusion ponderada + JS-divergence
    5) DataFrame con (image_path, row, col, p_A, p_B, p_C, p_fused, consensus, stage1_pred)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..common.io import read_table, write_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from .fusion import DEFAULT_WEIGHTS, fuse_probabilities, js_divergence
from .gpu_pipeline import iter_image_batches, plan_epoch

log = get_logger("phase_d.infer_gpu")


@dataclass
class StageOneEnsembleGPU:
    branch_a: nn.Module
    branch_b: nn.Module
    branch_c: nn.Module
    temp_a: float = 1.0
    temp_b: float = 1.0
    temp_c: float = 1.0
    weights: dict = None
    tau_s1: float = 0.65
    device: torch.device = torch.device("cuda")

    def to(self, device: torch.device) -> "StageOneEnsembleGPU":
        self.device = device
        self.branch_a.to(device).eval()
        self.branch_b.to(device).eval()
        self.branch_c.to(device).eval()
        return self


@torch.no_grad()
def infer_image_gpu(
    image_path: str | Path,
    ensemble: StageOneEnsembleGPU,
    tiles_index_path: Optional[Path] = None,
    batch_size: int = 32,
    use_amp: bool = False,
) -> pd.DataFrame:
    """Inferencia completa sobre una imagen, todo en CUDA."""
    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = df[df["image_path"] == rel].copy().reset_index(drop=True)
    if sub.empty:
        log.warning(f"No hay tiles para {rel} en el manifest")
        return sub

    plan = plan_epoch(sub, shuffle_images=False, shuffle_tiles_within_image=False)

    rows_all: list[int] = []
    cols_all: list[int] = []
    pa_all, pb_all, pc_all = [], [], []

    dev = ensemble.device
    iterator = tqdm(
        iter_image_batches(plan, batch_size=batch_size, device=dev),
        total=(len(sub) + batch_size - 1) // batch_size,
        desc=f"infer {Path(rel).name}",
        dynamic_ncols=True,
    )
    for batch in iterator:
        with torch.autocast(device_type=dev.type, enabled=use_amp and dev.type == "cuda"):
            la = ensemble.branch_a(batch.rgb)
            lb = ensemble.branch_b(batch.seg)
            lc = ensemble.branch_c(batch.freq)
        pa = torch.sigmoid(la.float() / ensemble.temp_a).cpu().numpy()
        pb = torch.sigmoid(lb.float() / ensemble.temp_b).cpu().numpy()
        pc = torch.sigmoid(lc.float() / ensemble.temp_c).cpu().numpy()
        pa_all.append(pa); pb_all.append(pb); pc_all.append(pc)
        rows_all.extend(batch.rows.cpu().tolist())
        cols_all.extend(batch.cols.cpu().tolist())

    p_a = np.concatenate(pa_all) if pa_all else np.array([])
    p_b = np.concatenate(pb_all) if pb_all else np.array([])
    p_c = np.concatenate(pc_all) if pc_all else np.array([])
    probs = {"A": p_a, "B": p_b, "C": p_c}
    p_fused = fuse_probabilities(probs, weights=ensemble.weights or DEFAULT_WEIGHTS)
    consensus = 1.0 - np.clip(js_divergence(probs) / np.log(2.0), 0, 1)

    # Re-emparejar con `sub` por (row, col)
    order_df = pd.DataFrame({
        "row": rows_all, "col": cols_all,
        "p_A": p_a, "p_B": p_b, "p_C": p_c,
        "p_fused": p_fused, "consensus": consensus,
    })
    merged = sub.merge(order_df, on=["row", "col"], how="left")
    merged["stage1_pred"] = (merged["p_fused"] >= ensemble.tau_s1).astype(int)
    return merged
