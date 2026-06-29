"""Pipeline 100% CUDA para Stage1.

Una sola política: por cada imagen del manifest
    1) decode_jpeg_gpu (nvJPEG) -> GPUImage en VRAM
    2) batch_tiles_gpu(...) crea batches de tiles en VRAM
    3) build_views_gpu -> (rgb, seg, freq) cuda float32
    4) modelo(x) en cuda -> logits cuda
    5) backward+step en cuda
    6) descartar GPUImage (release VRAM)
    7) pasar a la siguiente imagen

Sin DataLoader, sin numpy, sin PIL en el hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch

from ..common.gpu_cleanup import release_cuda_memory
from ..common.io import read_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import (
    GPUImage,
    batch_tiles_gpu,
    decode_jpeg_gpu,
    iter_uniform_tile_rowcol_batches,
)
from ..phase_c_views.gpu_transforms import build_views_gpu

from .gate_classes import encode_gate_indices

log = get_logger("phase_d.gpu_pipeline")

ImageSource = GPUImage | np.ndarray


def _should_cpu_decode(path: Path, *, force: bool, threshold_mb: float) -> bool:
    if force:
        return True
    from .gate_embed_cache import _use_cpu_decode_for_cache

    return _use_cpu_decode_for_cache(path, threshold_mb)


def _load_image_for_tiles(
    full: Path,
    device: torch.device,
    cache: dict[str, ImageSource],
    *,
    force_cpu_decode: bool,
    cpu_decode_above_mb: float,
) -> ImageSource | None:
    key = str(full)
    if key in cache:
        return cache[key]
    try:
        if _should_cpu_decode(full, force=force_cpu_decode, threshold_mb=cpu_decode_above_mb):
            from ..phase_b_tiling.jpeg_streaming import open_image_rgb_fast

            arr = open_image_rgb_fast(full)
            cache[key] = arr
            return arr
        gpu_img = decode_jpeg_gpu(full, device=device)
        cache[key] = gpu_img
        return gpu_img
    except Exception as e:
        log.warning(f"[gpu_iter] saltando {full.name}: {e}")
        return None


def _tiles_from_source(
    source: ImageSource,
    rowcols: list[tuple[int, int, int]],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(source, np.ndarray):
        from .gate_embed_cache import _batch_tiles_cpu_to_gpu

        return _batch_tiles_cpu_to_gpu(source, rowcols, device)
    return batch_tiles_gpu(source, rowcols)


def _batch_meta_from_slice(
    sub: pd.DataFrame,
    start: int,
    stop: int,
) -> tuple[Optional[list[str]], Optional[list[int]]]:
    sub_slice = sub.iloc[start:stop]
    domains = (
        sub_slice["domain_bucket"].astype(str).tolist()
        if "domain_bucket" in sub_slice.columns
        else None
    )
    edges = (
        sub_slice["tile_edge"].astype(int).tolist()
        if "tile_edge" in sub_slice.columns
        else (
            sub_slice["tile_size"].astype(int).tolist()
            if "tile_size" in sub_slice.columns
            else None
        )
    )
    return domains, edges


@dataclass
class GPUImageBatch:
    """Batch heterogéneo construido sobre la misma imagen."""

    rgb: torch.Tensor      # (B, 3, 224, 224) float32 cuda normalizada ImageNet
    seg: torch.Tensor      # (B, 3, 320, 320) float32 cuda normalizada ImageNet
    freq: torch.Tensor     # (B, 3, 224, 224) float32 cuda (log|F|, cosφ, sinφ)
    labels: torch.Tensor   # (B,) float32 cuda binario o int64 cuda multiclas
    rows: torch.Tensor     # (B,) int32 cuda
    cols: torch.Tensor     # (B,) int32 cuda
    image_path: str
    label_mode: str = "binary"
    saliency: Optional[torch.Tensor] = None  # (B, H, W) float32 cuda, U2Net d0
    features: Optional[torch.Tensor] = None  # (B, D) float32 cuda, DINO embed cache
    vit_attention: Optional[torch.Tensor] = None  # (B, L, gh, gw) mapas CLS->patch
    domain_buckets: Optional[list[str]] = None  # bucket dominio/escala por tile
    tile_edges: Optional[list[int]] = None  # tile_edge px por tile (Tier B eval)


@dataclass
class EpochPlan:
    """Lista ordenada (image_path, tiles_df) lista para iterar.

    `tiles_df` ya contiene los tiles SUB-MUESTREADOS y barajados para la época.
    """

    items: list[tuple[str, pd.DataFrame]]
    stratified_batches: Optional[list[pd.DataFrame]] = None

    @property
    def n_tiles(self) -> int:
        if self.stratified_batches is not None:
            return sum(len(df) for df in self.stratified_batches)
        return sum(len(df) for _, df in self.items)

    @property
    def n_images(self) -> int:
        if self.stratified_batches is not None:
            return 0
        return len(self.items)

    def n_batches(self, batch_size: int) -> int:
        """Número de mini-batches GPU al iterar este plan."""
        if batch_size <= 0:
            return 0
        if self.stratified_batches is not None:
            return len(self.stratified_batches)
        return sum((len(df) + batch_size - 1) // batch_size for _, df in self.items)


def plan_epoch(
    tiles_df: pd.DataFrame,
    *,
    max_neg_per_image: Optional[int] = None,
    interleave_by_lineage: bool = True,
    shuffle_images: bool = True,
    shuffle_tiles_within_image: bool = True,
    pos_class: str = "Mplus",
    seed: int = 0,
) -> EpochPlan:
    """Construye el plan determinístico para una época."""
    rng = np.random.default_rng(seed)

    per_image_chunks: dict[str, pd.DataFrame] = {}
    per_lineage: dict[str, list[str]] = {}

    for image_path, grp in tiles_df.groupby("image_path", sort=False):
        if max_neg_per_image is not None:
            pos = grp[grp["stage1"] == pos_class]
            neg = grp[grp["stage1"] != pos_class]
            if len(neg) > max_neg_per_image:
                idx = rng.choice(neg.index.to_numpy(), size=max_neg_per_image, replace=False)
                neg = grp.loc[idx]
            grp = pd.concat([pos, neg])
        if shuffle_tiles_within_image:
            grp = grp.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        per_image_chunks[image_path] = grp
        lin = str(grp["lineage"].iloc[0])
        per_lineage.setdefault(lin, []).append(image_path)

    if shuffle_images:
        for lin in per_lineage:
            rng.shuffle(per_lineage[lin])

    if interleave_by_lineage:
        order: list[str] = []
        while any(per_lineage.values()):
            for lin in list(per_lineage.keys()):
                if per_lineage[lin]:
                    order.append(per_lineage[lin].pop(0))
    else:
        order = [p for lst in per_lineage.values() for p in lst]

    items = [(p, per_image_chunks[p]) for p in order]
    return EpochPlan(items=items)


def _subsample_df(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    return _sample_df(df, n, rng, replace=False)


def _sample_df(df: pd.DataFrame, n: int, rng: np.random.Generator, *, replace: bool = False) -> pd.DataFrame:
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0]
    if len(df) == n and not replace:
        return df
    if len(df) >= n and not replace:
        idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        return df.loc[idx]
    idx = rng.choice(df.index.to_numpy(), size=n, replace=True)
    return df.loc[idx]


def plan_epoch_class_counts(plan: EpochPlan) -> dict[str, int]:
    if plan.stratified_batches:
        from .gate_epoch_sampler import stratified_epoch_class_counts

        return stratified_epoch_class_counts(plan)
    counts: dict[str, int] = {c: 0 for c in ("Background", "Mminus", "Mplus", "Unreadable")}
    for _, df in plan.items:
        for name, n in df["stage1"].value_counts().items():
            counts[str(name)] = counts.get(str(name), 0) + int(n)
    return counts


def plan_epoch_gate(
    tiles_df: pd.DataFrame,
    *,
    max_bg_per_image: Optional[int] = 50,
    balance_mode: str = "cap_bg",
    mplus_oversample_factor: float = 1.0,
    shuffle_images: bool = True,
    shuffle_tiles_within_image: bool = True,
    seed: int = 0,
) -> EpochPlan:
    """Plan de epoca para gate 3-clases (Bg / M- / M+).

    balance_mode:
        cap_bg — conserva todo M+/M-; subsample Background (legacy).
        evangelisti_1to1 — por imagen: |M+| ~= |M- union Background| (solo binario M+ vs resto).
        balanced_3class — por imagen: |M+| ~= |M-| ~= |Bg| (recomendado gate 3 clases).
        balanced_4class — por imagen: |M-| base; |M+| ~= factor*|M-| (oversample);
            Bg ~= |M-| pareado; Unknown capado.
        root_focused — todo M+/M-; Bg capado (foco en objeto, no fondo).
    """
    rng = np.random.default_rng(seed)
    per_image_chunks: dict[str, pd.DataFrame] = {}

    for image_path, grp in tiles_df.groupby("image_path", sort=False):
        mplus = grp[grp["stage1"] == "Mplus"]
        mminus = grp[grp["stage1"] == "Mminus"]
        bg = grp[grp["stage1"] == "Background"]
        unknown = grp[grp["stage1"] == "Unreadable"]

        if balance_mode == "root_focused":
            if max_bg_per_image is not None and len(bg) > max_bg_per_image:
                bg = _subsample_df(bg, max_bg_per_image, rng)
            if max_bg_per_image is not None and len(unknown) > max(1, max_bg_per_image // 2):
                unknown = _subsample_df(unknown, max(1, max_bg_per_image // 2), rng)
        elif balance_mode == "balanced_4class":
            n_base = min(len(mplus), len(mminus))
            if n_base > 0:
                mminus = _sample_df(mminus, n_base, rng, replace=False)
                n_mplus = max(n_base, int(round(n_base * max(mplus_oversample_factor, 1.0))))
                mplus = _sample_df(mplus, n_mplus, rng, replace=len(mplus) < n_mplus)
            # Background DEBE emparejarse con los positivos para ser una tercera
            # clase real. Objetivo: ~= n_base por imagen (anclado a M-).
            target_bg = n_base if n_base > 0 else (max_bg_per_image or len(bg))
            if max_bg_per_image is not None:
                target_bg = min(target_bg, max_bg_per_image) if n_base == 0 else target_bg
            n_bg = min(len(bg), target_bg)
            if n_bg > 0 and len(bg) > n_bg:
                bg = _subsample_df(bg, n_bg, rng)
            unk_cap = max(1, n_base // 2) if n_base > 0 else (max_bg_per_image or 10)
            if len(unknown) > unk_cap:
                unknown = _subsample_df(unknown, unk_cap, rng)
        elif balance_mode == "balanced_3class":
            n_pos = min(len(mplus), len(mminus))
            if n_pos > 0:
                mplus = _subsample_df(mplus, n_pos, rng)
                mminus = _subsample_df(mminus, n_pos, rng)
            n_bg = len(bg)
            if max_bg_per_image is not None:
                n_bg = min(n_bg, max_bg_per_image)
            if n_bg > 0 and len(bg) > n_bg:
                bg = _subsample_df(bg, n_bg, rng)
        elif balance_mode == "evangelisti_1to1":
            n_pos = len(mplus)
            if n_pos > 0:
                neg_pool = pd.concat([mminus, bg])
                if len(neg_pool) > n_pos:
                    idx = rng.choice(neg_pool.index.to_numpy(), size=n_pos, replace=False)
                    neg_pool = neg_pool.loc[idx]
                mminus = neg_pool[neg_pool["stage1"] == "Mminus"]
                bg = neg_pool[neg_pool["stage1"] == "Background"]
            elif max_bg_per_image is not None and len(bg) > max_bg_per_image:
                bg = _subsample_df(bg, max_bg_per_image, rng)
        elif max_bg_per_image is not None and len(bg) > max_bg_per_image:
            bg = _subsample_df(bg, max_bg_per_image, rng)

        grp = pd.concat([mplus, mminus, bg, unknown])
        if shuffle_tiles_within_image and len(grp):
            grp = grp.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        per_image_chunks[image_path] = grp

    order = list(per_image_chunks.keys())
    if shuffle_images:
        rng.shuffle(order)
    return EpochPlan(items=[(p, per_image_chunks[p]) for p in order])


def iter_image_batches(
    plan: EpochPlan,
    batch_size: int = 16,
    *,
    device: torch.device = torch.device("cuda"),
    target_size: int = 252,
    seg_target_size: int = 360,
    pos_class: str = "Mplus",
    label_mode: str = "binary",
    max_batches: Optional[int] = None,
    drop_image_after: bool = True,
    u2net_saliency: Optional[torch.nn.Module] = None,
    saliency_size: int = 252,
    force_cpu_decode: bool = False,
    cpu_decode_above_mb: float = 300.0,
) -> Iterator[GPUImageBatch]:
    """Genera batches recorriendo imagen-por-imagen.

    Decode JPEG en CPU cuando ``force_cpu_decode`` (E6 8GB): la imagen vive en RAM;
    solo los tiles del mini-batch suben a VRAM para DINO/U2Net.
    """
    paths = get_paths()
    seen = 0

    for image_rel, sub in plan.items:
        full = paths.root / image_rel
        image_cache: dict[str, ImageSource] = {}
        source = _load_image_for_tiles(
            full,
            device,
            image_cache,
            force_cpu_decode=force_cpu_decode,
            cpu_decode_above_mb=cpu_decode_above_mb,
        )
        if source is None:
            continue

        rows = sub["row"].to_numpy(dtype=np.int32)
        cols = sub["col"].to_numpy(dtype=np.int32)
        if label_mode == "gate":
            labels_np = encode_gate_indices(sub["stage1"].to_numpy())
            labels_t = torch.from_numpy(labels_np).long()
        else:
            labels_np = (sub["stage1"].to_numpy() == pos_class).astype(np.float32)
            labels_t = torch.from_numpy(labels_np).float()
        tile_sizes = sub["tile_size"].to_numpy(dtype=np.int32)

        for batch_indices, rowcols in iter_uniform_tile_rowcol_batches(
            rows, cols, tile_sizes, batch_size=batch_size
        ):
            if not rowcols:
                continue
            start, stop = batch_indices[0], batch_indices[-1] + 1
            try:
                tiles = _tiles_from_source(source, rowcols, device)
            except Exception as e:
                log.warning(f"[gpu_iter] error tile batch en {image_rel}: {e}")
                continue

            views = build_views_gpu(
                tiles, target_size=target_size, seg_target_size=seg_target_size
            )
            saliency_t: Optional[torch.Tensor] = None
            if u2net_saliency is not None:
                from .gate_bg_pooling import u2net_saliency_batch

                saliency_t = u2net_saliency_batch(
                    u2net_saliency, views.seg, saliency_size=saliency_size
                )
            domains, edges = _batch_meta_from_slice(sub, start, stop)
            idx_t = torch.as_tensor(batch_indices, dtype=torch.long)
            yield GPUImageBatch(
                rgb=views.rgb,
                seg=views.seg,
                freq=views.freq,
                labels=labels_t[idx_t].to(device, non_blocking=True),
                label_mode=label_mode,
                rows=torch.from_numpy(rows[batch_indices]).to(device, non_blocking=True),
                cols=torch.from_numpy(cols[batch_indices]).to(device, non_blocking=True),
                image_path=image_rel,
                saliency=saliency_t,
                domain_buckets=domains,
                tile_edges=edges,
            )
            seen += 1
            if max_batches and seen >= max_batches:
                if drop_image_after:
                    image_cache.clear()
                    if device.type == "cuda":
                        release_cuda_memory()
                return

        if drop_image_after:
            image_cache.clear()
            if device.type == "cuda":
                release_cuda_memory()


def iter_stratified_gate_image_batches(
    plan: EpochPlan,
    *,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    label_mode: str = "gate",
    max_batches: Optional[int] = None,
    target_size: int = 252,
    seg_target_size: int = 360,
    u2net_saliency: Optional[torch.nn.Module] = None,
    saliency_size: int = 252,
    force_cpu_decode: bool = False,
    cpu_decode_above_mb: float = 300.0,
) -> Iterator[GPUImageBatch]:
    """Itera batches pre-armados (g1_stratified) decodificando JPEG on-the-fly."""
    if not plan.stratified_batches:
        return

    paths = get_paths()
    seen = 0
    for batch_idx, sub in enumerate(plan.stratified_batches):
        if sub.empty:
            continue
        rgb_parts: list[torch.Tensor] = []
        seg_parts: list[torch.Tensor] = []
        freq_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        row_parts: list[torch.Tensor] = []
        col_parts: list[torch.Tensor] = []
        sal_parts: list[torch.Tensor] = []
        domains: list[str] = []
        edges: list[int] = []
        image_cache: dict[str, ImageSource] = {}

        for image_rel, grp in sub.groupby("image_path", sort=False):
            full = paths.root / str(image_rel)
            source = _load_image_for_tiles(
                full,
                device,
                image_cache,
                force_cpu_decode=force_cpu_decode,
                cpu_decode_above_mb=cpu_decode_above_mb,
            )
            if source is None:
                continue
            rows = grp["row"].to_numpy(dtype=np.int32)
            cols = grp["col"].to_numpy(dtype=np.int32)
            tile_sizes = grp["tile_size"].to_numpy(dtype=np.int32)
            if label_mode == "gate":
                labels_np = encode_gate_indices(grp["stage1"].to_numpy())
                labels_t = torch.from_numpy(labels_np).long()
            else:
                labels_t = torch.from_numpy(
                    (grp["stage1"].to_numpy() == "Mplus").astype(np.float32)
                ).float()

            for local_indices, rowcols in iter_uniform_tile_rowcol_batches(
                rows, cols, tile_sizes, batch_size=len(grp)
            ):
                if not rowcols:
                    continue
                try:
                    tiles = _tiles_from_source(source, rowcols, device)
                except Exception as e:
                    log.warning(f"[gpu_iter] stratified tiles {image_rel}: {e}")
                    continue
                views = build_views_gpu(
                    tiles, target_size=target_size, seg_target_size=seg_target_size
                )
                saliency_t: Optional[torch.Tensor] = None
                if u2net_saliency is not None:
                    from .gate_bg_pooling import u2net_saliency_batch

                    saliency_t = u2net_saliency_batch(
                        u2net_saliency, views.seg, saliency_size=saliency_size
                    )
                idx_t = torch.as_tensor(local_indices, dtype=torch.long)
                rgb_parts.append(views.rgb)
                seg_parts.append(views.seg)
                freq_parts.append(views.freq)
                label_parts.append(labels_t[idx_t].to(device, non_blocking=True))
                row_parts.append(
                    torch.from_numpy(rows[local_indices]).to(device, non_blocking=True)
                )
                col_parts.append(
                    torch.from_numpy(cols[local_indices]).to(device, non_blocking=True)
                )
                if saliency_t is not None:
                    sal_parts.append(saliency_t)
                if "domain_bucket" in grp.columns:
                    domains.extend(
                        grp.iloc[local_indices]["domain_bucket"].astype(str).tolist()
                    )
                if "tile_edge" in grp.columns:
                    edges.extend(grp.iloc[local_indices]["tile_edge"].astype(int).tolist())
                elif "tile_size" in grp.columns:
                    edges.extend(grp.iloc[local_indices]["tile_size"].astype(int).tolist())

            image_cache.pop(str(full), None)
            if device.type == "cuda":
                release_cuda_memory()

        image_cache.clear()
        if device.type == "cuda":
            release_cuda_memory()

        if not rgb_parts:
            continue
        sal_batch = torch.cat(sal_parts, dim=0) if sal_parts else None
        yield GPUImageBatch(
            rgb=torch.cat(rgb_parts, dim=0),
            seg=torch.cat(seg_parts, dim=0),
            freq=torch.cat(freq_parts, dim=0),
            labels=torch.cat(label_parts, dim=0),
            label_mode=label_mode,
            rows=torch.cat(row_parts, dim=0),
            cols=torch.cat(col_parts, dim=0),
            image_path="__stratified__",
            saliency=sal_batch,
            domain_buckets=domains or None,
            tile_edges=edges or None,
        )
        seen += 1
        if max_batches and seen >= max_batches:
            return


def split_by_image(
    tiles_index_path: Optional[Path] = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    lineages: Optional[Iterable[str]] = None,
    subsets: Optional[Iterable[str]] = None,
    exclude_unreadable: bool = True,
    split_mode: str = "random",
    train_splits: tuple[str, ...] = ("train",),
    val_splits: tuple[str, ...] = ("test",),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split estratificado por imagen.

    `split_mode`:
        - ``random``: holdout aleatorio por imagen dentro del subset filtrado.
        - ``fixed``: train = filas con ``split in train_splits``, val = ``split in val_splits``.
    """
    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))

    if lineages:
        df = df[df["lineage"].isin(list(lineages))].copy()
    if subsets:
        df = df[df["subset"].isin(list(subsets))].copy()
    if exclude_unreadable:
        df = df[df["stage1"] != "Unreadable"].copy()
    if df.empty:
        raise ValueError("No quedan tiles tras los filtros.")

    if split_mode == "fixed" and "split" in df.columns:
        train_df = df[df["split"].isin(train_splits)].reset_index(drop=True)
        val_df = df[df["split"].isin(val_splits)].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Split fijo vacío: train={len(train_df)} val={len(val_df)} "
                f"(train_splits={train_splits}, val_splits={val_splits})"
            )
        train_imgs = train_df["image_path"].unique().tolist()
        val_imgs = val_df["image_path"].unique().tolist()
    else:
        image_lineage = df.groupby("image_path")["lineage"].first().reset_index()
        rng = np.random.default_rng(seed)
        train_imgs, val_imgs = [], []
        for lin, grp in image_lineage.groupby("lineage"):
            imgs = grp["image_path"].to_numpy().copy()
            rng.shuffle(imgs)
            n_val = max(1, int(round(len(imgs) * val_fraction)))
            val_imgs.extend(imgs[:n_val].tolist())
            train_imgs.extend(imgs[n_val:].tolist())

        train_df = df[df["image_path"].isin(train_imgs)].reset_index(drop=True)
        val_df = df[df["image_path"].isin(val_imgs)].reset_index(drop=True)

    info = {
        "split_mode": split_mode,
        "n_train_images": len(train_imgs),
        "n_val_images": len(val_imgs),
        "n_train_tiles": int(len(train_df)),
        "n_val_tiles": int(len(val_df)),
        "train_pos": int((train_df["stage1"] == "Mplus").sum()),
        "val_pos": int((val_df["stage1"] == "Mplus").sum()),
        "train_mminus": int((train_df["stage1"] == "Mminus").sum()),
        "val_mminus": int((val_df["stage1"] == "Mminus").sum()),
        "train_bg": int((train_df["stage1"] == "Background").sum()),
        "val_bg": int((val_df["stage1"] == "Background").sum()),
        "train_unknown": int((train_df["stage1"] == "Unreadable").sum()),
        "val_unknown": int((val_df["stage1"] == "Unreadable").sum()),
    }
    return train_df, val_df, info
