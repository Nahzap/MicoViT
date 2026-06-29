"""Construcción de mapa L4 (segmentación U2Net) para tiles M+.

Genera un mapa continuo en coordenadas de imagen usando solo tiles M+ y
la salida `d0` de U2NETP (rama B). El hot path permanece en CUDA; solo el
mapa final se devuelve como numpy para componer PNGs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import decode_jpeg_gpu
from .gpu_pipeline import iter_image_batches, plan_epoch

log = get_logger("phase_d.segmap")


@torch.no_grad()
def build_segmentation_map_mplus(
    image_path: str | Path,
    tiles_df: pd.DataFrame,
    seg_branch_model: torch.nn.Module,
    *,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve `(seg_prob_map, seg_bin_map)` en tamaño original de imagen.

    Usa únicamente tiles M+ (`stage1_pred==1` o `stage1=='Mplus'`).
    """
    if tiles_df.empty:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    paths = get_paths()
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = tiles_df[tiles_df["image_path"] == rel].copy()
    if sub.empty:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    if "stage1_pred" in sub.columns:
        mplus = sub[sub["stage1_pred"] == 1].copy()
    else:
        mplus = sub[sub["stage1"] == "Mplus"].copy()
    if mplus.empty:
        gimg = decode_jpeg_gpu(image_path, device=device)
        h, w = gimg.height, gimg.width
        del gimg
        torch.cuda.empty_cache()
        z = np.zeros((h, w), dtype=np.float32)
        return z, z

    # Dimensiones reales de la imagen para reensamble espacial.
    gimg = decode_jpeg_gpu(image_path, device=device)
    h, w = gimg.height, gimg.width
    del gimg
    torch.cuda.empty_cache()

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_cnt = np.zeros((h, w), dtype=np.float32)

    tile_size_map = {
        (int(r), int(c)): int(ts)
        for r, c, ts in zip(mplus["row"].values, mplus["col"].values, mplus["tile_size"].values)
    }

    plan = plan_epoch(
        mplus,
        max_neg_per_image=None,
        interleave_by_lineage=False,
        shuffle_images=False,
        shuffle_tiles_within_image=False,
        seed=0,
    )
    n_tiles = int(plan.n_tiles)
    n_batches = (n_tiles + batch_size - 1) // batch_size if n_tiles > 0 else 0
    log.info(f"[L4] reconstruyendo mapa: {n_tiles} tiles M+ en {n_batches} batches (bs={batch_size})")
    seg_branch_model = seg_branch_model.to(device).eval()
    if not hasattr(seg_branch_model, "u2net"):
        raise AttributeError("seg_branch_model no tiene atributo `u2net` para generar máscaras.")
    u2 = seg_branch_model.u2net.eval()

    pbar = tqdm(
        iter_image_batches(plan, batch_size=batch_size, device=device),
        total=n_batches,
        desc="L4 segmap",
        leave=False,
    )
    for batch in pbar:
        d0, *_ = u2(batch.seg)  # (B,1,320,320), sigmoid
        for i in range(d0.shape[0]):
            row = int(batch.rows[i].item())
            col = int(batch.cols[i].item())
            ts = tile_size_map.get((row, col), 252)
            y0 = row * ts
            x0 = col * ts
            if y0 >= h or x0 >= w:
                continue
            y1 = min(y0 + ts, h)
            x1 = min(x0 + ts, w)
            th = y1 - y0
            tw = x1 - x0
            if th <= 0 or tw <= 0:
                continue
            tile_prob = F.interpolate(
                d0[i : i + 1],
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            )[0, 0].detach().float().cpu().numpy()
            prob_sum[y0:y1, x0:x1] += tile_prob
            prob_cnt[y0:y1, x0:x1] += 1.0

    seg_prob = prob_sum / np.maximum(prob_cnt, 1e-6)
    seg_bin = (seg_prob >= threshold).astype(np.float32)
    return seg_prob, seg_bin

