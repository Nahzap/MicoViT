"""Dataset e iteración tiles M+ para entrenamiento píxel Fase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from ..common.io import read_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import batch_tiles_gpu, decode_jpeg_gpu
from ..phase_c_views.gpu_transforms import rgb_view_gpu, seg_view_gpu
from .gpu_pipeline import filter_mplus_stage2
from .pixel_morph import PixelMorphParams, segment_tile_pixel_morph

_log = get_logger("phase_e.pixel_data")


@dataclass
class PixelTileBatch:
    model_input: torch.Tensor   # (B,3,H,W) cuda — ViT RGB o U2Net seg
    labels: torch.Tensor        # (B,H,W) int64 cuda
    rows: torch.Tensor
    cols: torch.Tensor
    image_paths: list[str]
    stage2_gold: list[str] = field(default_factory=list)
    prior_evidence: Optional[torch.Tensor] = None  # (B,5,H,W) cuda — MEViT H5
    prior_vesicle: Optional[torch.Tensor] = None   # (B,H,W) cuda

    @property
    def seg_input(self) -> torch.Tensor:
        """Alias retrocompatible."""
        return self.model_input


def load_mplus_splits(
    lineage: str = "AM",
    val_fraction: float = 0.2,
    seed: int = 42,
    subsets_train: tuple[str, ...] = ("am_train",),
    subsets_test: tuple[str, ...] = ("am_test",),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    paths = get_paths()
    df = read_table(paths.manifests / "tiles_index")
    train_pool = df[df["subset"].isin(subsets_train)].copy()
    train_pool = filter_mplus_stage2(train_pool, lineage, exclude_unreadable=False)
    test_df = df[df["subset"].isin(subsets_test)].copy()
    test_df = filter_mplus_stage2(test_df, lineage, exclude_unreadable=False)

    imgs = train_pool["image_path"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(round(len(imgs) * val_fraction)))
    val_imgs = set(imgs[:n_val].tolist())
    train_df = train_pool[~train_pool["image_path"].isin(val_imgs)].reset_index(drop=True)
    val_df = train_pool[train_pool["image_path"].isin(val_imgs)].reset_index(drop=True)

    info = {
        "n_train_tiles": int(len(train_df)),
        "n_val_tiles": int(len(val_df)),
        "n_test_tiles": int(len(test_df)),
        "n_train_images": int(train_df["image_path"].nunique()),
        "n_val_images": int(val_df["image_path"].nunique()),
        "n_test_images": int(test_df["image_path"].nunique()),
    }
    return train_df, val_df, test_df, info


def _weak_label_batch(
    tiles_u8: torch.Tensor,
    params: PixelMorphParams,
    target_size: int,
) -> torch.Tensor:
    """Genera labels (B,H,W) en CPU desde tiles uint8."""
    b, _, th, tw = tiles_u8.shape
    labels = []
    for i in range(b):
        arr = tiles_u8[i].permute(1, 2, 0).contiguous().cpu().numpy()
        seg = segment_tile_pixel_morph(arr, params)
        if seg.shape[0] != target_size or seg.shape[1] != target_size:
            import cv2

            seg = cv2.resize(seg, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        labels.append(torch.from_numpy(seg.astype(np.int64)))
    return torch.stack(labels, dim=0)


def iter_pixel_batches(
    tiles_df: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
    input_size: int = 224,
    input_mode: str = "vit",
    morph_params: Optional[PixelMorphParams] = None,
    shuffle: bool = True,
    max_batches: Optional[int] = None,
    phase: str = "train",
    log_every_batches: int = 5,
    h5_store: Optional["Stage2PixelH5Store"] = None,
    load_priors: bool = False,
) -> Iterator[PixelTileBatch]:
    if h5_store is not None:
        yield from iter_pixel_batches_h5(
            tiles_df,
            h5_store,
            batch_size=batch_size,
            device=device,
            input_mode=input_mode,
            shuffle=shuffle,
            max_batches=max_batches,
            phase=phase,
            log_every_batches=log_every_batches,
            load_priors=load_priors,
        )
        return
    yield from _iter_pixel_batches_manifest(
        tiles_df,
        batch_size=batch_size,
        device=device,
        input_size=input_size,
        input_mode=input_mode,
        morph_params=morph_params,
        shuffle=shuffle,
        max_batches=max_batches,
        phase=phase,
        log_every_batches=log_every_batches,
    )
def _apply_flip_variant(
    rgb: torch.Tensor, labels: torch.Tensor, variant: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """variant 0=original, 1=H, 2=V, 3=HV."""
    if variant == 1:
        rgb = torch.flip(rgb, dims=[-1])
        labels = torch.flip(labels, dims=[-1])
    elif variant == 2:
        rgb = torch.flip(rgb, dims=[-2])
        labels = torch.flip(labels, dims=[-2])
    elif variant == 3:
        rgb = torch.flip(rgb, dims=[-1, -2])
        labels = torch.flip(labels, dims=[-1, -2])
    return rgb, labels


def _apply_flip_priors(
    prior_e: torch.Tensor,
    prior_v: torch.Tensor,
    variant: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == 0:
        return prior_e, prior_v
    if variant == 1:
        return torch.flip(prior_e, dims=[-1]), torch.flip(prior_v, dims=[-1])
    if variant == 2:
        return torch.flip(prior_e, dims=[-2]), torch.flip(prior_v, dims=[-2])
    return torch.flip(prior_e, dims=[-1, -2]), torch.flip(prior_v, dims=[-1, -2])


def _augment_batch(rgb: torch.Tensor, labels: torch.Tensor, p: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb.shape[0] == 0:
        return rgb, labels
    if torch.rand(1, device=rgb.device).item() < p:
        rgb = torch.flip(rgb, dims=[-1])
        labels = torch.flip(labels, dims=[-1])
    if torch.rand(1, device=rgb.device).item() < p:
        rgb = torch.flip(rgb, dims=[-2])
        labels = torch.flip(labels, dims=[-2])
    if torch.rand(1, device=rgb.device).item() < p:
        k = int(torch.randint(1, 4, (1,), device=rgb.device).item())
        rgb = torch.rot90(rgb, k, [-2, -1])
        labels = torch.rot90(labels, k, [-2, -1])
    return rgb, labels


def expand_train_df_v_flip_oversample(
    train_df: pd.DataFrame,
    h5_store: "Stage2PixelH5Store",
    *,
    min_v_px: int = 30,
    variants: int = 4,
) -> pd.DataFrame:
    """Duplica tiles con vesículas: original + flips H/V/HV (sin duplicar HDF5)."""
    from .pixel_class_map import PIXEL_CLASS_TO_IDX
    from .stage2_pixel_train_report import phase_log

    if train_df.empty or variants <= 1:
        out = train_df.copy()
        out["flip_variant"] = 0
        return out

    v_idx = PIXEL_CLASS_TO_IDX["V"]
    h5_idx = h5_store.indices_for_sub(train_df)
    labels = h5_store.read_labels_at(h5_idx)
    rows: list[dict] = []
    n_rare = 0
    for i in range(len(train_df)):
        row = train_df.iloc[i].to_dict()
        v_count = int((labels[i] == v_idx).sum())
        n_var = variants if v_count >= min_v_px else 1
        if n_var > 1:
            n_rare += 1
        for v in range(n_var):
            r = dict(row)
            r["flip_variant"] = v
            rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    phase_log(
        f"oversample V: {n_rare} tiles V>={min_v_px}px -> {len(out)} filas train "
        f"(x{variants} flips, +{len(out) - len(train_df)} extra)"
    )
    return out


def _build_h5_pixel_batch(
    h5_store: "Stage2PixelH5Store",
    chunk_idx: np.ndarray,
    chunk_df: pd.DataFrame,
    *,
    device: torch.device,
    input_mode: str,
    load_priors: bool,
    do_aug: bool,
    has_flip_col: bool,
) -> PixelTileBatch:
    rgb, labels, prior_e, prior_v = h5_store.read_training_tensors(
        chunk_idx, device, load_priors=load_priors and h5_store.has_priors
    )
    if input_mode != "vit":
        from ..phase_c_views.gpu_transforms import seg_view_gpu

        tiles_u8 = (rgb.clamp(0, 1) * 255).byte()
        model_in = seg_view_gpu(tiles_u8, target_size=rgb.shape[-1], blur_radius=1, normalize_imagenet=True)
    else:
        model_in = rgb
    if has_flip_col:
        fv = chunk_df["flip_variant"].astype(np.int64).values
        for v, dims in ((1, [-1]), (2, [-2]), (3, [-1, -2])):
            sel = np.where(fv == v)[0]
            if sel.size == 0:
                continue
            sel_t = torch.as_tensor(sel, device=model_in.device, dtype=torch.long)
            model_in.index_copy_(0, sel_t, torch.flip(model_in.index_select(0, sel_t), dims=dims))
            labels.index_copy_(0, sel_t, torch.flip(labels.index_select(0, sel_t), dims=dims))
            if prior_e is not None and prior_v is not None:
                prior_e.index_copy_(0, sel_t, torch.flip(prior_e.index_select(0, sel_t), dims=dims))
                prior_v.index_copy_(0, sel_t, torch.flip(prior_v.index_select(0, sel_t), dims=dims))
    elif do_aug:
        model_in, labels = _augment_batch(model_in, labels)
    stage2_gold = chunk_df["stage2"].astype(str).tolist() if "stage2" in chunk_df.columns else []
    return PixelTileBatch(
        model_input=model_in,
        labels=labels,
        rows=torch.tensor(chunk_df["row"].astype(int).tolist(), dtype=torch.long, device=device),
        cols=torch.tensor(chunk_df["col"].astype(int).tolist(), dtype=torch.long, device=device),
        image_paths=chunk_df["image_path"].astype(str).tolist(),
        stage2_gold=stage2_gold,
        prior_evidence=prior_e,
        prior_vesicle=prior_v,
    )


def iter_pixel_batches_h5(
    tiles_df: pd.DataFrame,
    h5_store: Stage2PixelH5Store,
    *,
    batch_size: int,
    device: torch.device,
    input_mode: str = "vit",
    shuffle: bool = True,
    max_batches: Optional[int] = None,
    phase: str = "train",
    log_every_batches: int = 5,
    load_priors: bool = False,
) -> Iterator[PixelTileBatch]:
    import time
    from ..cli import _cfg
    from .stage2_pixel_train_report import phase_log

    if tiles_df.empty:
        return
    df = tiles_df.sample(frac=1.0, random_state=42).reset_index(drop=True) if shuffle else tiles_df.copy()
    n_tiles_total = len(df)
    est_batches = max(1, (n_tiles_total + batch_size - 1) // batch_size)
    if max_batches is not None:
        est_batches = min(est_batches, max_batches)

    ram_cached = getattr(h5_store, "_ram_cache", None) is not None
    src = "RAM cache (sin lzf/batch)" if ram_cached else "HDF5 Stage2-Pixel (cache hit)"
    phase_log(
        f"FASE datos [{phase}] — {src}. "
        f"tiles={n_tiles_total} batch={batch_size} batches~={est_batches} | solo lectura GPU"
    )

    all_idx = h5_store.indices_for_sub(df)
    n_batches = 0
    n_tiles_done = 0
    t0 = time.perf_counter()

    do_aug = phase.startswith("train") and bool(_cfg("STAGE2_PIXEL_AUGMENT_H5", True))
    has_flip_col = "flip_variant" in df.columns
    use_prefetch = bool(_cfg("STAGE2_PIXEL_H5_BATCH_PREFETCH", True)) and phase.startswith("train")

    def _load_slice(start: int, end: int) -> PixelTileBatch:
        return _build_h5_pixel_batch(
            h5_store,
            all_idx[start:end],
            df.iloc[start:end],
            device=device,
            input_mode=input_mode,
            load_priors=load_priors,
            do_aug=do_aug,
            has_flip_col=has_flip_col,
        )

    if use_prefetch:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h5_batch_prefetch")
        try:
            starts = list(range(0, n_tiles_total, batch_size))
            if not starts:
                return
            prefetch_future = pool.submit(_load_slice, starts[0], min(starts[0] + batch_size, n_tiles_total))
            for i, start in enumerate(starts):
                end = min(start + batch_size, n_tiles_total)
                batch = prefetch_future.result()
                if i + 1 < len(starts):
                    nxt = starts[i + 1]
                    prefetch_future = pool.submit(_load_slice, nxt, min(nxt + batch_size, n_tiles_total))
                n_batches += 1
                n_tiles_done += end - start
                if log_every_batches > 0 and (n_batches == 1 or n_batches % log_every_batches == 0):
                    elapsed = time.perf_counter() - t0
                    tiles_per_s = n_tiles_done / max(elapsed, 1e-6)
                    eta_s = (est_batches - n_batches) * (elapsed / max(n_batches, 1))
                    img_tail = Path(batch.image_paths[-1]).name if batch.image_paths else "?"
                    phase_log(
                        f"FASE datos [{phase}] H5 batch {n_batches}/{est_batches} "
                        f"tiles {n_tiles_done}/{n_tiles_total} ({100.0 * n_tiles_done / n_tiles_total:.1f}%) "
                        f"{tiles_per_s:.1f} tiles/s eta={eta_s / 60:.1f}m img={img_tail}"
                    )
                yield batch
                if max_batches is not None and n_batches >= max_batches:
                    return
        finally:
            pool.shutdown(wait=False)
        return

    for start in range(0, n_tiles_total, batch_size):
        end = min(start + batch_size, n_tiles_total)
        batch = _load_slice(start, end)
        n_batches += 1
        n_tiles_done += end - start
        if log_every_batches > 0 and (n_batches == 1 or n_batches % log_every_batches == 0):
            elapsed = time.perf_counter() - t0
            tiles_per_s = n_tiles_done / max(elapsed, 1e-6)
            eta_s = (est_batches - n_batches) * (elapsed / max(n_batches, 1))
            img_tail = Path(batch.image_paths[-1]).name if batch.image_paths else "?"
            phase_log(
                f"FASE datos [{phase}] H5 batch {n_batches}/{est_batches} "
                f"tiles {n_tiles_done}/{n_tiles_total} ({100.0 * n_tiles_done / n_tiles_total:.1f}%) "
                f"{tiles_per_s:.1f} tiles/s eta={eta_s / 60:.1f}m img={img_tail}"
            )
        yield batch
        if max_batches is not None and n_batches >= max_batches:
            return


def _iter_pixel_batches_manifest(
    tiles_df: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
    input_size: int = 224,
    input_mode: str = "vit",
    morph_params: Optional[PixelMorphParams] = None,
    shuffle: bool = True,
    max_batches: Optional[int] = None,
    phase: str = "train",
    log_every_batches: int = 5,
) -> Iterator[PixelTileBatch]:
    import time

    from .stage2_pixel_train_report import phase_log

    if tiles_df.empty:
        return
    morph_params = morph_params or PixelMorphParams()
    df = tiles_df.sample(frac=1.0, random_state=42).reset_index(drop=True) if shuffle else tiles_df.copy()
    paths = get_paths()
    root = paths.root
    by_img = df.groupby("image_path", sort=False)
    n_images = len(by_img)
    n_tiles_total = len(df)
    est_batches = max(1, (n_tiles_total + batch_size - 1) // batch_size)
    if max_batches is not None:
        est_batches = min(est_batches, max_batches)

    phase_log(
        f"FASE datos [{phase}] — tiles M+ desde manifest (NO conforma HDF5). "
        f"tiles={n_tiles_total} imgs={n_images} batch={batch_size} batches~={est_batches} | "
        f"pseudo-labels weak CPU + forward ViT GPU"
    )

    n_batches = 0
    n_tiles_done = 0
    img_idx = 0
    t_batch_start = time.perf_counter()
    buf_rows, buf_cols, buf_paths = [], [], []
    buf_stage2: list[str] = []
    buf_tiles: list[torch.Tensor] = []

    def _flush() -> Optional[PixelTileBatch]:
        nonlocal n_batches, n_tiles_done, t_batch_start
        if not buf_tiles:
            return None
        tiles_u8 = torch.stack(buf_tiles, dim=0)
        if input_mode == "vit":
            model_in = rgb_view_gpu(tiles_u8, target_size=input_size, normalize_imagenet=True)
        else:
            model_in = seg_view_gpu(
                tiles_u8, target_size=input_size, blur_radius=1, normalize_imagenet=True
            )
        labels_cpu = _weak_label_batch(tiles_u8, morph_params, input_size)
        batch = PixelTileBatch(
            model_input=model_in.to(device, non_blocking=True),
            labels=labels_cpu.to(device, non_blocking=True),
            rows=torch.tensor(buf_rows, dtype=torch.long, device=device),
            cols=torch.tensor(buf_cols, dtype=torch.long, device=device),
            image_paths=list(buf_paths),
            stage2_gold=list(buf_stage2),
        )
        n_tiles_done += len(buf_tiles)
        buf_tiles.clear()
        buf_rows.clear()
        buf_cols.clear()
        buf_paths.clear()
        buf_stage2.clear()
        n_batches += 1
        if n_batches == 1 or n_batches % log_every_batches == 0:
            elapsed = time.perf_counter() - t_batch_start
            tiles_per_s = n_tiles_done / max(elapsed, 1e-6)
            eta_s = (est_batches - n_batches) * (elapsed / max(n_batches, 1))
            img_tail = Path(batch.image_paths[-1]).name if batch.image_paths else "?"
            phase_log(
                f"FASE datos [{phase}] batch {n_batches}/{est_batches} "
                f"tiles {n_tiles_done}/{n_tiles_total} ({100.0 * n_tiles_done / n_tiles_total:.1f}%) "
                f"{tiles_per_s:.1f} tiles/s eta={eta_s / 60:.1f}m img={img_tail}"
            )
        return batch

    for img_rel, sub in by_img:
        img_idx += 1
        img_path = root / img_rel
        if not img_path.exists():
            phase_log(f"FASE datos [{phase}] SKIP imagen ausente: {img_rel}")
            continue
        if img_idx == 1 or img_idx % 10 == 0:
            phase_log(
                f"FASE datos [{phase}] JPEG GPU {img_idx}/{n_images}: "
                f"{Path(img_rel).name} ({len(sub)} tiles M+)"
            )
        gimg = decode_jpeg_gpu(img_path, device=device)
        sub = sub.sort_values(["row", "col"]).reset_index(drop=True)
        coords = list(zip(sub["row"].astype(int), sub["col"].astype(int)))
        ts = int(sub["tile_size"].iloc[0]) if "tile_size" in sub.columns else 252
        for chunk_start in range(0, len(coords), batch_size):
            chunk = coords[chunk_start : chunk_start + batch_size]
            if not chunk:
                continue
            rowcols = [(int(r), int(c), ts) for r, c in chunk]
            tiles = batch_tiles_gpu(gimg, rowcols)
            for j, (r, c) in enumerate(chunk):
                buf_tiles.append(tiles[j].cpu())
                buf_rows.append(int(r))
                buf_cols.append(int(c))
                buf_paths.append(str(img_rel))
                s2_val = str(sub["stage2"].iloc[chunk_start + j]) if "stage2" in sub.columns else ""
                buf_stage2.append(s2_val)
                if len(buf_tiles) >= batch_size:
                    out = _flush()
                    if out is not None:
                        yield out
                        if max_batches is not None and n_batches >= max_batches:
                            del gimg
                            return
        del gimg

    out = _flush()
    if out is not None:
        yield out
