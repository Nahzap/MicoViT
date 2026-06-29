"""Pipeline CUDA Stage2 — solo tiles M+ con subclase anotada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch

from ..common.io import read_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import GPUImage, batch_tiles_gpu, decode_jpeg_gpu
from ..phase_c_views.gpu_transforms import build_views_gpu
from ..phase_d_stage1.gpu_pipeline import EpochPlan, plan_epoch
from .class_map import Stage2ClassMap, load_stage2_class_map

log = get_logger("phase_e.gpu_pipeline")


@dataclass
class GPUStage2Batch:
    rgb: torch.Tensor
    seg: torch.Tensor
    freq: torch.Tensor
    labels: torch.Tensor   # (B,) int64 cuda
    rows: torch.Tensor
    cols: torch.Tensor
    image_path: str


def filter_mplus_stage2(
    tiles_df: pd.DataFrame,
    lineage: str,
    *,
    exclude_unreadable: bool = True,
) -> pd.DataFrame:
    df = tiles_df[tiles_df["lineage"] == lineage].copy()
    df = df[df["stage1"] == "Mplus"].copy()
    df = df[df["stage2"].notna()].copy()
    if exclude_unreadable:
        df = df[df["stage2"] != "Unreadable"].copy()
    return df.reset_index(drop=True)


def split_by_image_mplus(
    lineage: str,
    tiles_index_path: Optional[Path] = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    subsets: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Stage2ClassMap]:
    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))
    df = filter_mplus_stage2(df, lineage)
    if subsets:
        df = df[df["subset"].isin(list(subsets))].copy()
    if df.empty:
        raise ValueError(f"Sin tiles M+ Stage2 para linaje {lineage}")

    cmap = load_stage2_class_map(lineage, only_present_in=df)
    df = df[df["stage2"].isin(cmap.classes)].copy().reset_index(drop=True)
    df["stage2_idx"] = cmap.encode(df["stage2"]).astype(int)

    imgs = df["image_path"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(round(len(imgs) * val_fraction)))
    val_imgs = set(imgs[:n_val].tolist())
    train_df = df[~df["image_path"].isin(val_imgs)].reset_index(drop=True)
    val_df = df[df["image_path"].isin(val_imgs)].reset_index(drop=True)

    info = {
        "lineage": lineage,
        "num_classes": cmap.num_classes,
        "classes": list(cmap.classes),
        "n_train_images": int(train_df["image_path"].nunique()),
        "n_val_images": int(val_df["image_path"].nunique()),
        "n_train_tiles": int(len(train_df)),
        "n_val_tiles": int(len(val_df)),
        "class_counts_train": train_df["stage2"].value_counts().to_dict(),
        "class_counts_val": val_df["stage2"].value_counts().to_dict(),
    }
    return train_df, val_df, info, cmap


def iter_image_batches_stage2(
    plan: EpochPlan,
    class_map: Stage2ClassMap,
    batch_size: int = 16,
    *,
    device: torch.device = torch.device("cuda"),
    target_size: int = 224,
    seg_target_size: int = 320,
    max_batches: Optional[int] = None,
) -> Iterator[GPUStage2Batch]:
    paths = get_paths()
    seen = 0

    for image_rel, sub in plan.items:
        full = paths.root / image_rel
        try:
            gpu_img: GPUImage = decode_jpeg_gpu(full, device=device)
        except Exception as e:
            log.warning(f"[gpu_iter_s2] saltando {image_rel}: {e}")
            continue

        rows = sub["row"].to_numpy(dtype=np.int32)
        cols = sub["col"].to_numpy(dtype=np.int32)
        tile_sizes = sub["tile_size"].to_numpy(dtype=np.int32)
        n = len(sub)

        if "stage2_idx" in sub.columns:
            labels_np = sub["stage2_idx"].to_numpy(dtype=np.int64)
        elif "stage2" in sub.columns and sub["stage2"].notna().all():
            labels_np = class_map.encode(sub["stage2"]).astype(int).to_numpy(dtype=np.int64)
        else:
            labels_np = np.zeros(n, dtype=np.int64)

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            rowcols = [(int(rows[i]), int(cols[i]), int(tile_sizes[i])) for i in range(start, stop)]
            try:
                tiles = batch_tiles_gpu(gpu_img, rowcols)
            except Exception as e:
                log.warning(f"[gpu_iter_s2] error tile batch en {image_rel}: {e}")
                continue

            views = build_views_gpu(tiles, target_size=target_size, seg_target_size=seg_target_size)
            yield GPUStage2Batch(
                rgb=views.rgb,
                seg=views.seg,
                freq=views.freq,
                labels=torch.from_numpy(labels_np[start:stop]).to(device, non_blocking=True),
                rows=torch.from_numpy(rows[start:stop]).to(device, non_blocking=True),
                cols=torch.from_numpy(cols[start:stop]).to(device, non_blocking=True),
                image_path=image_rel,
            )
            seen += 1
            if max_batches and seen >= max_batches:
                del gpu_img
                torch.cuda.empty_cache()
                return

        del gpu_img
        torch.cuda.empty_cache()


def plan_epoch_stage2(
    tiles_df: pd.DataFrame,
    *,
    shuffle_images: bool = True,
    shuffle_tiles_within_image: bool = True,
    seed: int = 0,
) -> EpochPlan:
    return plan_epoch(
        tiles_df,
        max_neg_per_image=None,
        interleave_by_lineage=False,
        shuffle_images=shuffle_images,
        shuffle_tiles_within_image=shuffle_tiles_within_image,
        pos_class="Mplus",
        seed=seed,
    )
