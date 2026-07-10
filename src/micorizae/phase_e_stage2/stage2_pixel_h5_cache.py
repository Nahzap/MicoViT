"""Cache HDF5 Stage2-Pixel: RGB normalizado + máscaras morfológicas weak (materialización).

Conformación = pseudo-labels heurísticos (Frangi/vesículas/arbúsculos) + tiles RGB.
Prohibido: ViT, U2Net, DINO u otro modelo aprendido durante el build.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import batch_tiles_gpu, decode_jpeg_gpu
from ..phase_c_views.gpu_transforms import rgb_view_gpu
from .pixel_class_map import NUM_PIXEL_CLASSES
from .pixel_morph import PixelMorphParams, segment_tile_pixel_morph

log = get_logger("phase_e.pixel_h5")

CACHE_VERSION = 3
TILE_H = 224
TILE_W = 224
RGB_SHAPE = (3, TILE_H, TILE_W)
LABEL_SHAPE = (TILE_H, TILE_W)
PRIOR_EVIDENCE_SHAPE = (NUM_PIXEL_CLASSES, TILE_H, TILE_W)
PRIOR_VESICLE_SHAPE = (TILE_H, TILE_W)
_IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _default_h5_workers(workers: Optional[int] = None) -> int:
    import os

    if workers is not None and int(workers) > 0:
        return int(workers)
    return max(1, min(12, (os.cpu_count() or 4) - 2))


def _pin_blas_single_thread() -> None:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _cache_paths(root: Path) -> tuple[Path, Path, Path]:
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / "stage2_pixel_mplus_v1.h5",
        cache_dir / "stage2_pixel_mplus_v1.meta.json",
        cache_dir / "stage2_pixel_mplus_v1.lookup.parquet",
    )


def _manifest_fingerprint(tiles_index_path: Path) -> str:
    st = tiles_index_path.stat()
    payload = f"{tiles_index_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _morph_fingerprint(morph: PixelMorphParams, input_size: int, gate_run_id: str) -> str:
    w = morph.weak
    payload = {
        "input_size": input_size,
        "gate_run_id": gate_run_id,
        "stain_aware": bool(getattr(w, "stain_aware", True)),
        "vesicle_circularity_min": float(w.vesicle_circularity_min),
        "frangi_pctl": float(w.frangi_pctl),
        "arbuscule_pctl": float(w.arbuscule_pctl),
        "ves_bg_max": float(getattr(w, "ves_bg_max", 0.5)),
        "seam_sigma": float(morph.seam_sigma),
        "class_schema": "5c_bg_ih_v_a_h_stain",
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _prior_fingerprint(morph: PixelMorphParams, input_size: int, gate_run_id: str) -> str:
    """Fingerprint separado para priors: incluye la versión de implementación.

    Permite re-materializar priors (Frangi/circularidad/textura) sin re-etiquetar
    labels cuando cambia solo el algoritmo de evidencia (PRIOR_IMPL_VERSION).
    """
    from .pixel_prior_maps import PRIOR_IMPL_VERSION

    base = _morph_fingerprint(morph, input_size, gate_run_id)
    return f"{base}|prior_v{PRIOR_IMPL_VERSION}"


def _build_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["image_path", "row", "col"]].copy()
    out["h5_idx"] = np.arange(len(df), dtype=np.int64)
    return out[["h5_idx", "image_path", "row", "col"]]


def cache_is_valid(
    *,
    n_tiles: int,
    fingerprint: str,
    h5_path: Path,
    meta_path: Path,
    lookup_path: Path,
) -> bool:
    if not (h5_path.is_file() and meta_path.is_file() and lookup_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_version") != CACHE_VERSION:
            return False
        if meta.get("n_tiles") != n_tiles or meta.get("fingerprint") != fingerprint:
            return False
        import h5py

        with h5py.File(h5_path, "r") as hf:
            if hf["rgb"].shape[0] != n_tiles:
                return False
            if hf["label"].shape[0] != n_tiles:
                return False
        return True
    except Exception:
        return False


def _weak_label_tile_u8(tile_u8: np.ndarray, params: PixelMorphParams, target_size: int) -> np.ndarray:
    seg = segment_tile_pixel_morph(tile_u8, params)
    if seg.shape[0] != target_size or seg.shape[1] != target_size:
        import cv2

        seg = cv2.resize(seg, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return seg.astype(np.uint8)


def _build_tile_cpu_pack(
    tile_hwc_u8: np.ndarray,
    morph_params: PixelMorphParams,
    input_size: int,
    *,
    with_priors: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    label = _weak_label_tile_u8(tile_hwc_u8, morph_params, input_size)
    if not with_priors:
        return label, None, None
    from .pixel_prior_maps import compute_prior_training_targets

    ev, ves = compute_prior_training_targets(tile_hwc_u8, morph_params)
    return label, ev.astype(np.float16), ves.astype(np.float16)


def _rgb_h5_chunk_to_u8(rgb_chunk: np.ndarray) -> list[np.ndarray]:
    """rgb_chunk (B,3,H,W) fp16/fp32 ImageNet → list HWC uint8."""
    x = rgb_chunk.astype(np.float32)
    x = x * _IMAGENET_STD_NP + _IMAGENET_MEAN_NP
    x = np.clip(x, 0.0, 1.0)
    out: list[np.ndarray] = []
    for i in range(x.shape[0]):
        out.append((x[i] * 255.0).astype(np.uint8).transpose(1, 2, 0))
    return out


def h5_priors_ready(h5_path: Path, morph_fp: str) -> bool:
    import h5py

    if not h5_path.is_file():
        return False
    try:
        with h5py.File(h5_path, "r") as hf:
            if "prior_evidence" not in hf or "prior_vesicle" not in hf:
                return False
            return str(hf.attrs.get("prior_morph_fingerprint", "")) == morph_fp
    except Exception:
        return False


def ensure_h5_prior_datasets(
    h5_path: Path,
    morph_params: PixelMorphParams,
    *,
    input_size: int = 224,
    gate_run_id: str = "",
    compression: str = "lzf",
    chunk_size: int = 64,
    log_every: int = 64,
    workers: Optional[int] = None,
) -> bool:
    """Materializa prior_evidence + prior_vesicle en HDF5 existente (sin borrar rgb/label).

    Paraleliza el cómputo por tile con un ThreadPoolExecutor: los priors stain-aware
    (Frangi, LoG blob_log, textura) liberan el GIL en scipy.ndimage/opencv, así que
    varios tiles avanzan en paralelo. Sin esto, el bucle es single-thread (~3-4 t/s
    con CPU al ~25%); con N workers escala ~N× hasta saturar el CPU.
    """
    import h5py
    import os
    from concurrent.futures import ThreadPoolExecutor

    from .pixel_prior_maps import compute_prior_training_targets

    paths = get_paths()
    _, meta_path, _ = _cache_paths(paths.root)
    morph_fp = _prior_fingerprint(morph_params, input_size, gate_run_id)
    if h5_priors_ready(h5_path, morph_fp):
        log.info(f"[Stage2-Pixel H5] Priors MEViT CACHE HIT -> {h5_path}")
        print(f"[Stage2-Pixel H5] Priors MEViT CACHE HIT (lectura GPU en train)", flush=True)
        return False

    # 1 hilo BLAS por operación: paralelismo a nivel tile (pool).
    _pin_blas_single_thread()
    n_workers = _default_h5_workers(workers)

    comp = None if (compression or "").lower() in {"", "none", "off"} else compression
    log.info(f"[Stage2-Pixel H5] Materializando priors MEViT en {h5_path} (CPU one-shot)...")
    print(
        "[Stage2-Pixel H5] Materializando priors MEViT stain-aware (Frangi/contraste/textura "
        "sobre density) — one-shot, luego train solo GPU...",
        flush=True,
    )
    log.info(f"[Stage2-Pixel H5] priors con {n_workers} workers (paralelo por tile)")
    print(f"[Stage2-Pixel H5] priors: {n_workers} workers paralelos", flush=True)
    t0 = time.perf_counter()

    def _one(tile_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ev, ves = compute_prior_training_targets(tile_u8, morph_params)
        return ev.astype(np.float16), ves.astype(np.float16)

    with h5py.File(h5_path, "a") as hf, ThreadPoolExecutor(max_workers=n_workers) as pool:
        n = int(hf["rgb"].shape[0])
        if "prior_evidence" in hf:
            del hf["prior_evidence"]
        if "prior_vesicle" in hf:
            del hf["prior_vesicle"]
        hf.create_dataset(
            "prior_evidence",
            shape=(n, *PRIOR_EVIDENCE_SHAPE),
            dtype=np.float16,
            chunks=(min(16, n), *PRIOR_EVIDENCE_SHAPE),
            compression=comp,
        )
        hf.create_dataset(
            "prior_vesicle",
            shape=(n, *PRIOR_VESICLE_SHAPE),
            dtype=np.float16,
            chunks=(min(16, n), *PRIOR_VESICLE_SHAPE),
            compression=comp,
        )
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            rgb_chunk = hf["rgb"][start:end]
            tiles_u8 = _rgb_h5_chunk_to_u8(rgb_chunk)
            ev_np = np.zeros((len(tiles_u8), *PRIOR_EVIDENCE_SHAPE), dtype=np.float16)
            ves_np = np.zeros((len(tiles_u8), *PRIOR_VESICLE_SHAPE), dtype=np.float16)
            for j, (ev, ves) in enumerate(pool.map(_one, tiles_u8)):
                ev_np[j] = ev
                ves_np[j] = ves
            hf["prior_evidence"][start:end] = ev_np
            hf["prior_vesicle"][start:end] = ves_np
            done = end
            if done % log_every == 0 or done == n or start == 0:
                elapsed = max(0.001, time.perf_counter() - t0)
                rate = done / elapsed
                eta = (n - done) / rate if rate > 0 else 0
                print(
                    f"[Stage2-Pixel H5] priors {100.0 * done / n:.1f}% | "
                    f"{done:,}/{n:,} tiles | {rate:.2f} tiles/s | ETA {eta / 60:.1f}m",
                    flush=True,
                )
        hf.attrs["prior_morph_fingerprint"] = morph_fp
        hf.attrs["prior_classes"] = list(["BG", "IH", "V", "A", "H"])

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["prior_morph_fingerprint"] = morph_fp
        meta["prior_materialized"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    log.info(f"[Stage2-Pixel H5] Priors MEViT listos: {n:,} tiles en {elapsed / 60:.1f} min")
    print(f"[Stage2-Pixel H5] Priors MEViT LISTO {n:,} tiles en {elapsed / 60:.1f} min", flush=True)
    return True


def build_stage2_pixel_h5_cache(
    tiles_df: pd.DataFrame,
    morph_params: PixelMorphParams,
    *,
    device: torch.device,
    input_size: int = 224,
    gate_run_id: str = "",
    batch_size: int = 8,
    compression: str = "lzf",
    tiles_index_path: Optional[Path] = None,
    workers: Optional[int] = None,
    store_priors: bool = False,
) -> Path:
    """Materializa tiles M+ → HDF5 (rgb fp16 + label uint8 [+ priors si store_priors])."""
    import h5py
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    paths = get_paths()
    tiles_index_path = tiles_index_path or (paths.manifests / "tiles_index.csv")
    h5_path, meta_path, lookup_path = _cache_paths(paths.root)
    manifest_fp = _manifest_fingerprint(tiles_index_path)
    morph_fp = _morph_fingerprint(morph_params, input_size, gate_run_id)
    prior_fp = _prior_fingerprint(morph_params, input_size, gate_run_id)
    fingerprint = f"{manifest_fp}|{morph_fp}|v{CACHE_VERSION}"

    df = tiles_df.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("Sin tiles M+ para conformar HDF5 Stage2-Pixel")

    if cache_is_valid(
        n_tiles=n,
        fingerprint=fingerprint,
        h5_path=h5_path,
        meta_path=meta_path,
        lookup_path=lookup_path,
    ):
        if not store_priors or h5_priors_ready(h5_path, prior_fp):
            log.info(f"[Stage2-Pixel H5] CACHE HIT -> {h5_path} ({n:,} tiles)")
            print(f"[Stage2-Pixel H5] CACHE HIT -> {h5_path} ({n:,} tiles)", flush=True)
            return h5_path

    _pin_blas_single_thread()
    n_workers = _default_h5_workers(workers)
    comp = None if (compression or "").lower() in {"", "none", "off"} else compression
    est_gb = n * (np.prod(RGB_SHAPE) * 2 + np.prod(LABEL_SHAPE)) / (1024**3)
    if store_priors:
        est_gb += n * (np.prod(PRIOR_EVIDENCE_SHAPE) * 2 + np.prod(PRIOR_VESICLE_SHAPE) * 2) / (1024**3)
    log.info(
        f"[Stage2-Pixel H5] Conformando {n:,} tiles M+ -> {h5_path} "
        f"(rgb+label{'+priors' if store_priors else ''}, {n_workers} workers, comp={comp or 'none'}, ~{est_gb:.2f} GB)"
    )
    print(
        f"[Stage2-Pixel H5] Conformando {n:,} tiles ({n_workers} workers CPU, weak"
        f"{' + priors v3' if store_priors else ''}, sin ViT/U2Net)...",
        flush=True,
    )

    if h5_path.exists():
        h5_path.unlink()
    t0 = time.perf_counter()
    write_idx = 0
    root = paths.root
    groups = list(df.groupby("image_path", sort=False))
    n_images = len(groups)
    cpu_fn = partial(
        _build_tile_cpu_pack,
        morph_params=morph_params,
        input_size=input_size,
        with_priors=store_priors,
    )

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset(
            "rgb",
            shape=(n, *RGB_SHAPE),
            dtype=np.float16,
            chunks=(min(64, n), *RGB_SHAPE),
            compression=comp,
        )
        hf.create_dataset(
            "label",
            shape=(n, *LABEL_SHAPE),
            dtype=np.uint8,
            chunks=(min(64, n), *LABEL_SHAPE),
            compression=comp,
        )
        if store_priors:
            hf.create_dataset(
                "prior_evidence",
                shape=(n, *PRIOR_EVIDENCE_SHAPE),
                dtype=np.float16,
                chunks=(min(16, n), *PRIOR_EVIDENCE_SHAPE),
                compression=comp,
            )
            hf.create_dataset(
                "prior_vesicle",
                shape=(n, *PRIOR_VESICLE_SHAPE),
                dtype=np.float16,
                chunks=(min(16, n), *PRIOR_VESICLE_SHAPE),
                compression=comp,
            )
            hf.attrs["prior_morph_fingerprint"] = prior_fp
            hf.attrs["prior_classes"] = list(["BG", "IH", "V", "A", "H"])

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for img_idx, (img_rel, sub) in enumerate(groups, start=1):
                img_path = root / img_rel
                if not img_path.exists():
                    raise FileNotFoundError(f"[Stage2-Pixel H5] imagen ausente: {img_path}")
                gimg = decode_jpeg_gpu(img_path, device=device)
                sub = sub.sort_values(["row", "col"]).reset_index(drop=True)
                coords = list(zip(sub["row"].astype(int), sub["col"].astype(int)))
                ts = int(sub["tile_size"].iloc[0]) if "tile_size" in sub.columns else 252

                for chunk_start in range(0, len(coords), batch_size):
                    chunk = coords[chunk_start : chunk_start + batch_size]
                    rowcols = [(int(r), int(c), ts) for r, c in chunk]
                    tiles = batch_tiles_gpu(gimg, rowcols)
                    tiles_u8 = torch.stack([tiles[j].cpu() for j in range(len(chunk))], dim=0)
                    rgb_t = rgb_view_gpu(tiles_u8, target_size=input_size, normalize_imagenet=True)
                    rgb_np = rgb_t.cpu().numpy().astype(np.float16)

                    tiles_hwc = [
                        tiles_u8[j].permute(1, 2, 0).contiguous().numpy() for j in range(len(chunk))
                    ]
                    packs = list(pool.map(cpu_fn, tiles_hwc))
                    labels_np = np.stack([p[0] for p in packs], axis=0)
                    b = len(chunk)
                    hf["rgb"][write_idx : write_idx + b] = rgb_np
                    hf["label"][write_idx : write_idx + b] = labels_np
                    if store_priors:
                        hf["prior_evidence"][write_idx : write_idx + b] = np.stack(
                            [p[1] for p in packs], axis=0
                        )
                        hf["prior_vesicle"][write_idx : write_idx + b] = np.stack(
                            [p[2] for p in packs], axis=0
                        )
                    write_idx += b

                    if write_idx == b or write_idx % 40 == 0:
                        elapsed = max(0.001, time.perf_counter() - t0)
                        rate = write_idx / elapsed
                        eta = (n - write_idx) / rate if rate > 0 else 0
                        print(
                            f"[Stage2-Pixel H5] {100.0 * write_idx / n:.1f}% | "
                            f"{write_idx:,}/{n:,} tiles | {rate:.1f} tiles/s | "
                            f"ETA {eta / 60:.1f}m | img {img_idx}/{n_images} {Path(img_rel).name}",
                            flush=True,
                        )

                del gimg
                torch.cuda.empty_cache()

    if write_idx != n:
        raise RuntimeError(f"[Stage2-Pixel H5] escrito {write_idx} tiles, esperado {n}")

    lookup_df = _build_lookup_table(df)
    lookup_df.to_parquet(lookup_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "stage": "stage2_pixel_mplus",
                "n_tiles": n,
                "fingerprint": fingerprint,
                "manifest_fingerprint": manifest_fp,
                "morph_fingerprint": morph_fp,
                "gate_run_id": gate_run_id,
                "input_size": input_size,
                "h5_path": str(h5_path),
                "compression": comp or "none",
                "classes": ["BG", "IH", "V", "A", "H"],
                "prior_morph_fingerprint": prior_fp if store_priors else None,
                "prior_materialized": bool(store_priors),
                "build_workers": n_workers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - t0
    log.info(f"[Stage2-Pixel H5] Listo: {write_idx:,} tiles en {elapsed / 60:.1f} min -> {h5_path}")
    print(f"[Stage2-Pixel H5] LISTO {write_idx:,} tiles en {elapsed / 60:.1f} min", flush=True)
    return h5_path


def _gather_h5_rows(ds, idx: np.ndarray) -> np.ndarray:
    """Lee filas HDF5 con índices arbitrarios (orden único para lectura secuencial)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return ds[idx]
    order = np.argsort(idx)
    sorted_idx = idx[order]
    uniq, inv_uniq = np.unique(sorted_idx, return_inverse=True)
    data_u = ds[uniq]
    data = data_u[inv_uniq]
    if len(order) > 1:
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        data = data[inv_order]
    return data


@dataclass
class H5RamTileCache:
    """Copia descomprimida del HDF5 en RAM — lectura por batch sin lzf."""

    rgb: np.ndarray
    label: np.ndarray
    prior_e: Optional[np.ndarray]
    prior_v: Optional[np.ndarray]

    @classmethod
    def load_from_h5(cls, hf, *, has_priors: bool) -> "H5RamTileCache":
        from .stage2_pixel_train_report import phase_log

        n = int(hf["rgb"].shape[0])
        t0 = time.perf_counter()
        phase_log(f"FASE datos — cargando HDF5 completo a RAM ({n:,} tiles, descomprime lzf una vez)...")
        rgb = np.asarray(hf["rgb"], dtype=np.float16)
        label = np.asarray(hf["label"], dtype=np.uint8)
        prior_e = prior_v = None
        if has_priors:
            prior_e = np.asarray(hf["prior_evidence"], dtype=np.float16)
            prior_v = np.asarray(hf["prior_vesicle"], dtype=np.float16)
        mb = (rgb.nbytes + label.nbytes) / 1e6
        if prior_e is not None and prior_v is not None:
            mb += (prior_e.nbytes + prior_v.nbytes) / 1e6
        elapsed = time.perf_counter() - t0
        phase_log(f"FASE datos — RAM cache lista: {mb:.0f} MB en {elapsed:.1f}s ({n / max(elapsed, 0.1):.0f} tiles/s carga)")
        return cls(rgb=rgb, label=label, prior_e=prior_e, prior_v=prior_v)

    def gather(
        self,
        h5_indices: np.ndarray,
        device: torch.device,
        *,
        load_priors: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        idx = np.asarray(h5_indices, dtype=np.int64)
        if idx.size == 0:
            empty_rgb = torch.empty((0, 3, TILE_H, TILE_W), dtype=torch.float32, device=device)
            empty_lbl = torch.empty((0, TILE_H, TILE_W), dtype=torch.int64, device=device)
            return empty_rgb, empty_lbl, None, None

        rgb = torch.from_numpy(self.rgb[idx]).to(device=device, dtype=torch.float32, non_blocking=True)
        lbl = torch.from_numpy(self.label[idx].astype(np.int64, copy=False)).to(
            device=device, non_blocking=True
        )
        prior_e = prior_v = None
        if load_priors and self.prior_e is not None and self.prior_v is not None:
            prior_e = torch.from_numpy(self.prior_e[idx]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            prior_v = torch.from_numpy(self.prior_v[idx]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        return rgb, lbl, prior_e, prior_v

    def labels_at(self, h5_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(h5_indices, dtype=np.int64)
        if idx.size == 0:
            return np.empty((0, TILE_H, TILE_W), dtype=np.uint8)
        return self.label[idx]


class Stage2PixelH5Store:
    """Lectura por lotes desde HDF5 Stage2-Pixel."""

    def __init__(self, h5_path: Path, lookup_path: Path):
        import h5py

        self.h5_path = Path(h5_path)
        self.lookup = pd.read_parquet(lookup_path)
        self._lookup_key = self.lookup.set_index(["image_path", "row", "col"])["h5_idx"]
        self._hf = h5py.File(self.h5_path, "r")
        self._ram_cache: Optional[H5RamTileCache] = None

    def ensure_ram_cache(self, *, enabled: bool = True) -> None:
        if not enabled or self._ram_cache is not None or self._hf is None:
            return
        self._ram_cache = H5RamTileCache.load_from_h5(self._hf, has_priors=self.has_priors)

    @property
    def has_priors(self) -> bool:
        return self._hf is not None and "prior_evidence" in self._hf and "prior_vesicle" in self._hf

    def read_prior_batch(
        self,
        h5_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_e, prior_v = self.read_training_tensors(h5_indices, device, load_priors=True)
        if prior_e is None or prior_v is None:
            raise RuntimeError("HDF5 sin datasets prior_evidence/prior_vesicle")
        return prior_e, prior_v

    def read_training_tensors(
        self,
        h5_indices: np.ndarray,
        device: torch.device,
        *,
        load_priors: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """rgb, labels, prior_evidence, prior_vesicle — una pasada de índice HDF5."""
        idx = np.asarray(h5_indices, dtype=np.int64)
        if idx.size == 0:
            empty_rgb = torch.empty((0, 3, TILE_H, TILE_W), dtype=torch.float32, device=device)
            empty_lbl = torch.empty((0, TILE_H, TILE_W), dtype=torch.int64, device=device)
            return empty_rgb, empty_lbl, None, None

        if self._ram_cache is not None:
            return self._ram_cache.gather(idx, device, load_priors=load_priors and self.has_priors)

        rgb_np = _gather_h5_rows(self._hf["rgb"], idx).astype(np.float32, copy=False)
        lbl_np = _gather_h5_rows(self._hf["label"], idx).astype(np.int64, copy=False)
        rgb = torch.from_numpy(rgb_np).to(device, non_blocking=True)
        lbl = torch.from_numpy(lbl_np).to(device, non_blocking=True)

        prior_e = prior_v = None
        if load_priors and self.has_priors:
            pe_np = _gather_h5_rows(self._hf["prior_evidence"], idx).astype(np.float32, copy=False)
            pv_np = _gather_h5_rows(self._hf["prior_vesicle"], idx).astype(np.float32, copy=False)
            prior_e = torch.from_numpy(pe_np).to(device, non_blocking=True)
            prior_v = torch.from_numpy(pv_np).to(device, non_blocking=True)

        return rgb, lbl, prior_e, prior_v

    def close(self) -> None:
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __enter__(self) -> "Stage2PixelH5Store":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def indices_for_sub(self, sub: pd.DataFrame) -> np.ndarray:
        keys = list(
            zip(sub["image_path"].astype(str), sub["row"].astype(int), sub["col"].astype(int))
        )
        return np.array([int(self._lookup_key[k]) for k in keys], dtype=np.int64)

    def read_batch(
        self,
        h5_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, lbl, _, _ = self.read_training_tensors(h5_indices, device, load_priors=False)
        return rgb, lbl

    def read_labels_at(self, h5_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(h5_indices, dtype=np.int64)
        if idx.size == 0:
            return np.empty((0, TILE_H, TILE_W), dtype=np.uint8)
        if self._ram_cache is not None:
            return self._ram_cache.labels_at(idx)
        return self._hf["label"][idx]


def ensure_stage2_pixel_h5_cache(
    tiles_df: pd.DataFrame,
    morph_params: PixelMorphParams,
    *,
    device: torch.device,
    input_size: int = 224,
    gate_run_id: str = "",
    batch_size: int = 8,
    compression: str = "lzf",
    force_rebuild: bool = False,
    store_priors: bool = True,
    workers: Optional[int] = None,
) -> Stage2PixelH5Store:
    paths = get_paths()
    h5_path, meta_path, lookup_path = _cache_paths(paths.root)
    if force_rebuild:
        for p in (h5_path, meta_path, lookup_path):
            if p.exists():
                p.unlink()
    build_stage2_pixel_h5_cache(
        tiles_df,
        morph_params,
        device=device,
        input_size=input_size,
        gate_run_id=gate_run_id,
        batch_size=batch_size,
        compression=compression,
        workers=workers,
        store_priors=store_priors,
    )
    if store_priors:
        ensure_h5_prior_datasets(
            h5_path,
            morph_params,
            input_size=input_size,
            gate_run_id=gate_run_id,
            compression=compression,
            workers=workers,
        )
    return Stage2PixelH5Store(h5_path, lookup_path)


def collect_mplus_tiles_for_h5(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["image_path", "row", "col"]).reset_index(drop=True)
    return combined.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
