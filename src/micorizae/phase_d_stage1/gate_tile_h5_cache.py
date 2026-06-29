"""Cache HDF5 de tiles AM: materialización de datos (luma/RGB + label).

Principio: conformar HDF5 NO es inferencia. U2Net/DINO no pertenecen aquí.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..common.gpu_cleanup import maybe_empty_cache, maybe_gc_collect, release_cuda_memory
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import (
    batch_tiles_gpu,
    decode_jpeg_gpu,
    iter_uniform_tile_rowcol_batches,
)
from ..phase_b_tiling.jpeg_streaming import batch_tiles_streaming_from_file, open_image_rgb_fast
from ..phase_c_views.gpu_transforms import build_views_gpu
from .gate_classes import encode_gate_indices, GATE_CLASS_TO_IDX
from .gate_embed_cache import (
    _batch_tiles_cpu_to_gpu,
    _jpeg_dimensions,
    _jpeg_rgb_mb,
    _use_cpu_decode_for_cache,
)
from .gpu_pipeline import EpochPlan, GPUImageBatch

log = get_logger("phase_d.gate_h5")

CACHE_VERSION = 2
TILE_H = 224
TILE_W = 224
RGB_SHAPE = (3, TILE_H, TILE_W)
LUMA_SHAPE = (1, TILE_H, TILE_W)
SAL_SHAPE = (TILE_H, TILE_W)
BG_CLASS_IDX = int(GATE_CLASS_TO_IDX["Background"])


def _views_to_h5_array(views, *, grayscale: bool) -> np.ndarray:
    """Tensor normalizado ImageNet -> array fp16 para HDF5."""
    rgb = views.rgb.cpu().numpy().astype(np.float16)
    if not grayscale:
        return rgb
    luma = (
        0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    ).astype(np.float16)
    return luma[:, np.newaxis, :, :]


def _h5_array_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Lee tile HDF5 y devuelve RGB fp32 (B,3,H,W) para DINO/train."""
    data = arr.astype(np.float32)
    if data.ndim == 4 and data.shape[1] == 1:
        return np.repeat(data, 3, axis=1)
    if data.ndim == 3:
        return np.repeat(data[:, np.newaxis, :, :], 3, axis=1)
    return data


def load_frozen_u2net_saliency(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Carga U2NETP congelado (solo cache HDF5 legacy --legacy-h5)."""
    from .u2net import U2NETP

    model = U2NETP(in_ch=3, out_ch=1)
    if Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval().to(device)


def _cache_paths(root: Path) -> tuple[Path, Path, Path]:
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5 = cache_dir / "gate_am_tiles_v1.h5"
    meta = cache_dir / "gate_am_tiles_v1.meta.json"
    lookup = cache_dir / "gate_am_tiles_v1.lookup.parquet"
    return h5, meta, lookup


def _build_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["image_path", "row", "col"]].copy()
    out["h5_idx"] = np.arange(len(df), dtype=np.int64)
    return out[["h5_idx", "image_path", "row", "col"]]


def _manifest_fingerprint(tiles_index_path: Path) -> str:
    st = tiles_index_path.stat()
    payload = f"{tiles_index_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|v{CACHE_VERSION}"
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _format_eta(seconds: float) -> str:
    if seconds < 0:
        return "?"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _h5_compression(name: str) -> str | None:
    mode = (name or "none").lower()
    return None if mode in {"none", "off", ""} else mode


def _probe_h5_resume(
    *,
    progress_path: Path,
    n_tiles: int,
    fingerprint: str,
    h5_path: Path,
    resume_enabled: bool,
) -> int:
    """G: reanudar build interrumpido si progreso y shape HDF5 coinciden."""
    if not resume_enabled or not progress_path.is_file() or not h5_path.is_file():
        return 0
    try:
        with open(progress_path, encoding="utf-8") as f:
            prog = json.load(f)
        if prog.get("status") not in {"running", "failed"}:
            return 0
        if int(prog.get("tiles_total", -1)) != n_tiles:
            return 0
        tiles_done = int(prog.get("tiles_done", 0))
        if tiles_done <= 0:
            return 0
        import h5py

        with h5py.File(h5_path, "r") as hf:
            if hf["rgb"].shape[0] != n_tiles:
                return 0
        log.info(f"[Gate H5] RESUME: {tiles_done:,}/{n_tiles:,} tiles ya escritos")
        return tiles_done
    except Exception as e:
        log.warning(f"[Gate H5] No se pudo reanudar: {e}")
        return 0


def cache_is_valid(
    *,
    n_tiles: int,
    fingerprint: str,
    h5_path: Path,
    meta_path: Path,
    lookup_path: Path,
) -> bool:
    if not (h5_path.exists() and meta_path.exists() and lookup_path.exists()):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("cache_version", 1) != CACHE_VERSION:
            return False
        if meta.get("n_tiles") != n_tiles or meta.get("fingerprint") != fingerprint:
            return False
        import h5py

        with h5py.File(h5_path, "r") as hf:
            key = "luma" if "luma" in hf else "rgb"
            if hf[key].shape[0] != n_tiles:
                return False
        return True
    except Exception:
        return False


def cleanup_incomplete_h5_cache(root: Path) -> list[str]:
    """Elimina HDF5 parcial y progreso de build (no borra cache valido completo)."""
    h5_path, meta_path, lookup_path = _cache_paths(root)
    progress_path = root / "cache" / "gate_am_tiles_v1.build_progress.json"
    if meta_path.exists() and lookup_path.exists() and h5_path.exists():
        return []
    removed: list[str] = []
    for path in (h5_path, meta_path, lookup_path, progress_path):
        if not path.exists():
            continue
        try:
            path.unlink()
            removed.append(str(path))
        except OSError as e:
            log.warning(f"[Gate H5] no se pudo borrar {path.name}: {e}")
    if removed:
        log.info(f"[Gate H5] Cache parcial eliminado ({len(removed)} archivos)")
    return removed


@torch.no_grad()
def build_gate_tile_h5_cache(
    tiles_df: pd.DataFrame,
    u2net: torch.nn.Module | None = None,
    *,
    device: torch.device,
    batch_size: int = 32,
    tiles_index_path: Optional[Path] = None,
    empty_cache_every_n_batches: int = 0,
    gc_collect_every_n_batches: int = 0,
    cpu_decode_above_mb: float = 300.0,
    chunk_tiles: int = 64,
    compression: str | None = "none",
    u2net_bg_only: bool = True,
    skip_u2net: bool = False,
    store_saliency: bool = False,
    grayscale: bool = True,
    streaming_decode: bool = True,
    resume_enabled: bool = True,
) -> Path:
    """Compila tiles AM → HDF5 (luma/RGB fp16 + label; saliency opcional)."""
    import h5py

    paths = get_paths()
    tiles_index_path = tiles_index_path or (paths.manifests / "tiles_index.csv")
    h5_path, meta_path, lookup_path = _cache_paths(paths.root)
    fingerprint = _manifest_fingerprint(tiles_index_path)

    df = tiles_df.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
    groups = list(df.groupby("image_path", sort=False))
    n_images = len(groups)
    n = len(df)
    if cache_is_valid(
        n_tiles=n, fingerprint=fingerprint, h5_path=h5_path, meta_path=meta_path, lookup_path=lookup_path
    ):
        log.info(f"[Gate H5] CACHE HIT -> {h5_path}")
        return h5_path

    if skip_u2net:
        store_saliency = False
    tile_shape = LUMA_SHAPE if grayscale else RGB_SHAPE
    bytes_per_tile = int(np.prod(tile_shape)) * 2
    if store_saliency:
        bytes_per_tile += int(np.prod(SAL_SHAPE)) * 2
    est_gb = n * bytes_per_tile / (1024**3)

    log.info(
        f"[Gate H5] Compilando {n:,} tiles -> {h5_path} "
        f"({'luma' if grayscale else 'rgb'} fp16, saliency={'si' if store_saliency else 'no'}, "
        f"comp={compression or 'none'}, ~{est_gb:.1f} GB)"
    )
    if not skip_u2net:
        if u2net is None:
            raise ValueError("u2net requerido cuando skip_u2net=False")
        u2net.eval().to(device)

    progress_path = paths.root / "cache" / "gate_am_tiles_v1.build_progress.json"
    t0 = time.perf_counter()

    def _publish_h5_progress(*, images_done: int, status: str, detail: str) -> None:
        elapsed = max(0.001, time.perf_counter() - t0)
        pct = round((100.0 * write_idx / max(n, 1)), 2)
        tile_rate = write_idx / elapsed if write_idx > 0 else 0.0
        eta_s = (n - write_idx) / tile_rate if tile_rate > 0 else -1.0
        eta_str = _format_eta(eta_s) if eta_s >= 0 else "?"
        payload = {
            "status": status,
            "images_total": n_images,
            "images_done": images_done,
            "tiles_total": n,
            "tiles_done": write_idx,
            "pct": pct,
            "tiles_per_second": round(tile_rate, 1),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta_s, 1) if eta_s >= 0 else None,
            "eta": eta_str,
            "detail": detail,
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[Gate H5] {pct:.2f}% | {write_idx:,}/{n:,} tiles | "
            f"{tile_rate:.0f} tiles/s | ETA {eta_str} | {detail}",
            flush=True,
        )
        try:
            from .gate_run_live import publish_pipeline_panel

            publish_pipeline_panel(
                pipeline_phase=0,
                phases=[
                    {
                        "id": "hdf5_tiles",
                        "label": "Paso 1: HDF5 tiles (luma+label)",
                        "status": "done" if status == "completed" else "running",
                        "pct": pct,
                        "eta": eta_str,
                        "detail": detail,
                        "tiles_done": write_idx,
                        "tiles_total": n,
                        "tiles_per_second": round(tile_rate, 1),
                    }
                ],
            )
        except Exception:
            pass

    write_idx = 0
    batch_num = 0
    resume_write_idx = _probe_h5_resume(
        progress_path=progress_path,
        n_tiles=n,
        fingerprint=fingerprint,
        h5_path=h5_path,
        resume_enabled=resume_enabled,
    )
    chunk = max(1, min(int(chunk_tiles), n))
    h5_compression = _h5_compression(compression or "none")

    _publish_h5_progress(
        images_done=0,
        status="running",
        detail=f"Iniciando HDF5 ({n_images} imagenes)",
    )

    resume_mode = resume_write_idx > 0
    if resume_mode:
        write_idx = resume_write_idx
        hf_ctx = h5py.File(h5_path, "r+")
    else:
        hf_ctx = h5py.File(h5_path, "w")

    image_arr_cache: dict[str, np.ndarray] = {}
    global_tile_base = 0
    last_progress_t = time.perf_counter()

    with hf_ctx as hf:
        tile_key = "luma" if grayscale else "rgb"
        if not resume_mode:
            d_tiles = hf.create_dataset(
                tile_key,
                shape=(n, *tile_shape),
                dtype=np.float16,
                chunks=(chunk, *tile_shape),
                compression=h5_compression,
            )
            d_sal = None
            if store_saliency:
                d_sal = hf.create_dataset(
                    "saliency",
                    shape=(n, *SAL_SHAPE),
                    dtype=np.float16,
                    chunks=(chunk, *SAL_SHAPE),
                    compression=h5_compression,
                )
            d_lbl = hf.create_dataset("label", shape=(n,), dtype=np.int8)
        else:
            d_tiles = hf[tile_key]
            d_sal = hf["saliency"] if "saliency" in hf and store_saliency else None
            d_lbl = hf["label"]

        for image_idx, (image_rel, grp) in enumerate(
            tqdm(
                groups,
                desc="Gate H5",
                unit="img",
            ),
            start=1,
        ):
            n_img = len(grp)
            if global_tile_base + n_img <= write_idx:
                global_tile_base += n_img
                continue

            full = paths.root / image_rel
            gimg = None
            image_arr: np.ndarray | None = None
            use_cpu = _use_cpu_decode_for_cache(full, cpu_decode_above_mb)
            use_streaming = use_cpu and streaming_decode
            try:
                if use_cpu and not use_streaming:
                    w, h = _jpeg_dimensions(full)
                    est_mb = _jpeg_rgb_mb(w, h)
                    log.info(
                        f"[Gate H5] {Path(image_rel).name}: decode CPU cv2 "
                        f"({est_mb:.0f} MB RGB) -> solo tiles a GPU"
                    )
                    image_arr = open_image_rgb_fast(full)
                elif use_cpu and use_streaming:
                    w, h = _jpeg_dimensions(full)
                    est_mb = _jpeg_rgb_mb(w, h)
                    log.info(
                        f"[Gate H5] {Path(image_rel).name}: decode cv2 "
                        f"({est_mb:.0f} MB RGB, {n_img:,} tiles) -> GPU batch={batch_size}"
                    )
                    t_dec = time.perf_counter()
                    cache_key = str(full)
                    if cache_key not in image_arr_cache:
                        image_arr_cache[cache_key] = open_image_rgb_fast(full)
                    log.info(
                        f"[Gate H5]   decode OK en {time.perf_counter() - t_dec:.1f}s "
                        f"({n_img:,} tiles en imagen)"
                    )
                    _publish_h5_progress(
                        images_done=image_idx - 1,
                        status="running",
                        detail=f"{Path(image_rel).name}: decodificada, procesando tiles...",
                    )
                else:
                    gimg = decode_jpeg_gpu(full, device=device)
            except Exception as e:
                raise RuntimeError(f"[Gate H5] fallo en {image_rel}: {e}") from e

            rows = grp["row"].to_numpy(dtype=np.int32)
            cols = grp["col"].to_numpy(dtype=np.int32)
            labels = encode_gate_indices(grp["stage1"].to_numpy())
            tile_sizes = grp["tile_size"].to_numpy(dtype=np.int32)

            tile_cursor = 0
            for batch_indices, rowcols in iter_uniform_tile_rowcol_batches(
                rows, cols, tile_sizes, batch_size=batch_size
            ):
                batch_len = len(batch_indices)
                batch_global_start = global_tile_base + tile_cursor
                tile_cursor += batch_len
                if batch_global_start + batch_len <= write_idx:
                    continue
                skip_n = max(0, write_idx - batch_global_start)
                if skip_n >= batch_len:
                    continue
                if skip_n:
                    rowcols = rowcols[skip_n:]
                    batch_indices = batch_indices[skip_n:]

                batch_num += 1
                tiles = views = d0 = sal_t = rgb = sal = None
                try:
                    if use_streaming:
                        tiles = batch_tiles_streaming_from_file(
                            full,
                            rowcols,
                            device,
                            image_arr_cache=image_arr_cache,
                        )
                    elif image_arr is not None:
                        tiles = _batch_tiles_cpu_to_gpu(image_arr, rowcols, device)
                    else:
                        assert gimg is not None
                        tiles = batch_tiles_gpu(gimg, rowcols)
                    views = build_views_gpu(tiles)
                    label_batch = labels[batch_indices]
                    b = int(tiles.shape[0])
                    tile_batch = _views_to_h5_array(views, grayscale=grayscale)
                    sal = None
                    if store_saliency:
                        sal = np.zeros((b, *SAL_SHAPE), dtype=np.float16)
                        assert u2net is not None
                        if u2net_bg_only:
                            bg_mask = label_batch == BG_CLASS_IDX
                            if bg_mask.any():
                                bg_seg = views.seg[torch.from_numpy(bg_mask).to(device)]
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    d0, *_ = u2net(bg_seg)
                                sal_t = F.interpolate(
                                    d0.float(), size=(TILE_H, TILE_W), mode="bilinear", align_corners=False
                                ).squeeze(1)
                                sal[bg_mask] = sal_t.cpu().numpy().astype(np.float16)
                                del sal_t, d0
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                d0, *_ = u2net(views.seg)
                            sal_t = F.interpolate(
                                d0.float(), size=(TILE_H, TILE_W), mode="bilinear", align_corners=False
                            ).squeeze(1)
                            sal = sal_t.cpu().numpy().astype(np.float16)
                            del sal_t, d0

                    d_tiles[write_idx : write_idx + b] = tile_batch
                    if d_sal is not None and sal is not None:
                        d_sal[write_idx : write_idx + b] = sal
                    d_lbl[write_idx : write_idx + b] = label_batch
                    write_idx += b
                    del tile_batch
                    if sal is not None:
                        del sal
                finally:
                    if tiles is not None:
                        del tiles, views
                    maybe_empty_cache(batch_num, every_n=empty_cache_every_n_batches)
                    maybe_gc_collect(batch_num, every_n=gc_collect_every_n_batches)
                    now = time.perf_counter()
                    if batch_num % 10 == 0 or (now - last_progress_t) >= 10.0:
                        last_progress_t = now
                        pct_img = round(100.0 * tile_cursor / max(n_img, 1), 1)
                        _publish_h5_progress(
                            images_done=image_idx - 1,
                            status="running",
                            detail=(
                                f"{Path(image_rel).name}: {write_idx:,}/{n:,} tiles "
                                f"(img {pct_img}%)"
                            ),
                        )

            global_tile_base += n_img
            image_arr_cache.pop(str(full), None)
            if gimg is not None:
                del gimg
            if image_arr is not None:
                del image_arr
            release_cuda_memory(gc_collect=True)
            _publish_h5_progress(
                images_done=image_idx,
                status="running",
                detail=f"{Path(image_rel).name}: {write_idx:,}/{n:,} tiles",
            )

        if write_idx != n:
            log.warning(f"[Gate H5] escrito {write_idx} tiles, esperado {n}")

    lookup_df = _build_lookup_table(df)
    lookup_df.to_parquet(lookup_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_version": CACHE_VERSION,
                "n_tiles": n,
                "fingerprint": fingerprint,
                "h5_path": str(h5_path),
                "skip_u2net": bool(skip_u2net),
                "grayscale": bool(grayscale),
                "store_saliency": bool(store_saliency),
                "compression": compression or "none",
            },
            f,
            indent=2,
        )
    log.info(f"[Gate H5] Listo: {write_idx:,} tiles en {h5_path}")
    _publish_h5_progress(
        images_done=n_images,
        status="completed",
        detail=f"HDF5 listo: {write_idx:,} tiles",
    )
    return h5_path


class GateTileH5Store:
    """Lectura por lotes desde HDF5 + tabla de índices."""

    def __init__(self, h5_path: Path, lookup_path: Path):
        import h5py

        self.h5_path = Path(h5_path)
        self.lookup = pd.read_parquet(lookup_path)
        self._lookup_key = self.lookup.set_index(["image_path", "row", "col"])["h5_idx"]
        self._hf = h5py.File(self.h5_path, "r")

    def close(self) -> None:
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __del__(self) -> None:
        self.close()

    def indices_for_sub(self, sub: pd.DataFrame) -> np.ndarray:
        keys = list(zip(sub["image_path"].astype(str), sub["row"].astype(int), sub["col"].astype(int)))
        return np.array([int(self._lookup_key[k]) for k in keys], dtype=np.int64)

    def read_batch(
        self,
        h5_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.asarray(h5_indices, dtype=np.int64)
        if idx.size == 0:
            empty_rgb = torch.empty((0, 3, TILE_H, TILE_W), dtype=torch.float32, device=device)
            empty_sal = torch.empty((0, TILE_H, TILE_W), dtype=torch.float32, device=device)
            return empty_rgb, empty_sal, torch.empty((0,), dtype=torch.int64, device=device)
        order = np.argsort(idx)
        sorted_idx = idx[order]
        uniq, inv_uniq = np.unique(sorted_idx, return_inverse=True)

        key = "luma" if "luma" in self._hf else "rgb"
        rgb_u = torch.from_numpy(_h5_array_to_rgb(self._hf[key][uniq])).to(
            device, non_blocking=True
        )
        rgb = rgb_u[inv_uniq]
        if len(order) > 1:
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            rgb = rgb[inv_order]
        if "saliency" in self._hf:
            sal_u = torch.from_numpy(self._hf["saliency"][uniq].astype(np.float32)).to(
                device, non_blocking=True
            )
            sal = sal_u[inv_uniq]
            if len(order) > 1:
                sal = sal[inv_order]
        else:
            sal = torch.zeros(
                (len(idx), TILE_H, TILE_W),
                dtype=torch.float32,
                device=device,
            )
        lbl_u = torch.from_numpy(self._hf["label"][uniq].astype(np.int64)).to(
            device, non_blocking=True
        )
        lbl = lbl_u[inv_uniq]
        if len(order) > 1:
            lbl = lbl[inv_order]
        return rgb, sal, lbl


def ensure_gate_tile_h5_cache(
    tiles_df: pd.DataFrame,
    u2net: torch.nn.Module,
    device: torch.device,
    *,
    tiles_index_path: Optional[Path] = None,
) -> GateTileH5Store:
    h5_path = build_gate_tile_h5_cache(
        tiles_df, u2net, device=device, tiles_index_path=tiles_index_path
    )
    _, _, lookup_path = _cache_paths(get_paths().root)
    return GateTileH5Store(h5_path, lookup_path)


def iter_h5_gate_batches(
    plan: EpochPlan,
    store: GateTileH5Store,
    *,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    max_batches: Optional[int] = None,
) -> Iterator[GPUImageBatch]:
    """Itera batches desde HDF5 (sin decodificar JPEG ni U2Net)."""
    seen = 0
    for image_rel, sub in plan.items:
        rows = sub["row"].to_numpy(dtype=np.int32)
        cols = sub["col"].to_numpy(dtype=np.int32)
        n = len(sub)
        try:
            all_idx = store.indices_for_sub(sub)
        except KeyError as e:
            log.warning(f"[Gate H5] tile missing in lookup {image_rel}: {e}")
            continue

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            idx = all_idx[start:stop]
            rgb, sal, labels = store.read_batch(idx, device)
            yield GPUImageBatch(
                rgb=rgb,
                seg=rgb,
                freq=rgb,
                labels=labels,
                label_mode="gate",
                rows=torch.from_numpy(rows[start:stop]).to(device, non_blocking=True),
                cols=torch.from_numpy(cols[start:stop]).to(device, non_blocking=True),
                image_path=image_rel,
                saliency=sal,
            )
            seen += 1
            if max_batches and seen >= max_batches:
                return


def iter_h5_stratified_gate_batches(
    plan: EpochPlan,
    store: GateTileH5Store,
    *,
    device: torch.device = torch.device("cuda"),
    max_batches: Optional[int] = None,
    u2net_saliency: Optional[torch.nn.Module] = None,
    saliency_size: int = 252,
) -> Iterator[GPUImageBatch]:
    """Itera batches g1_stratified pre-armados desde HDF5 (sin decodificar JPEG)."""
    if not plan.stratified_batches:
        return

    seen = 0
    for sub in plan.stratified_batches:
        if sub.empty:
            continue
        rows = sub["row"].to_numpy(dtype=np.int32)
        cols = sub["col"].to_numpy(dtype=np.int32)
        try:
            all_idx = store.indices_for_sub(sub)
        except KeyError as e:
            log.warning(f"[Gate H5] stratified tile missing in lookup: {e}")
            continue
        rgb, sal, labels = store.read_batch(all_idx, device)
        if u2net_saliency is not None:
            from .gate_bg_pooling import u2net_saliency_batch

            sal = u2net_saliency_batch(
                u2net_saliency, rgb, saliency_size=saliency_size
            )
        domains: list[str] | None = None
        edges: list[int] | None = None
        if "domain_bucket" in sub.columns:
            domains = sub["domain_bucket"].astype(str).tolist()
        if "tile_edge" in sub.columns:
            edges = sub["tile_edge"].astype(int).tolist()
        elif "tile_size" in sub.columns:
            edges = sub["tile_size"].astype(int).tolist()
        yield GPUImageBatch(
            rgb=rgb,
            seg=rgb,
            freq=rgb,
            labels=labels,
            label_mode="gate",
            rows=torch.from_numpy(rows).to(device, non_blocking=True),
            cols=torch.from_numpy(cols).to(device, non_blocking=True),
            image_path="__stratified_h5__",
            saliency=sal,
            domain_buckets=domains,
            tile_edges=edges,
        )
        seen += 1
        if max_batches and seen >= max_batches:
            return
