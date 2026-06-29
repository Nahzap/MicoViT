"""Cache de embeddings DINOv2 por tile AM (v2).

Precomputa vectores DINOv2 congelados (mean-pool sobre tile RGB).
Entrenamiento Slice-MS lee embed + label desde memmap (~150 MB).
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
import torch.nn as nn
import torch.nn.functional as F

from ..common.gpu_cleanup import maybe_empty_cache, maybe_gc_collect, release_cuda_memory
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
from .gate_vision import AM_TILE_SIZE, describe_dino_resolution, dino_patch_grid
from .gpu_pipeline import EpochPlan, GPUImageBatch

log = get_logger("phase_d.gate_embed")

CACHE_VERSION = 5
DEFAULT_EMBED_DIM = 384
PROGRESS_SUFFIX = ".build_progress.json"
DEFAULT_DINO_INPUT = AM_TILE_SIZE
DEFAULT_BATCH_SIZE = 32
DEFAULT_VRAM_BUDGET_MB = 7500.0
MODEL_RESERVE_MB = 1200.0  # DINOv2 + activaciones
MIN_IMAGE_BATCH = 4
DEFAULT_CPU_DECODE_ABOVE_MB = 500.0

_DINO_BLOCKS: dict[str, int] = {
    "dinov2_vits14": 12,
    "dinov2_vitb14": 12,
    "dinov2_vitl14": 24,
    "dinov2_vitg14": 40,
}
_DINO_HEADS: dict[str, int] = {
    "dinov2_vits14": 6,
    "dinov2_vitb14": 12,
    "dinov2_vitl14": 16,
    "dinov2_vitg14": 24,
}


def _dino_num_blocks(backbone_name: str) -> int:
    key = str(backbone_name).strip().lower()
    if key not in _DINO_BLOCKS:
        raise ValueError(f"backbone desconocido para attn: {backbone_name!r}")
    return _DINO_BLOCKS[key]


def _dino_num_heads(backbone_name: str) -> int:
    key = str(backbone_name).strip().lower()
    if key not in _DINO_HEADS:
        raise ValueError(f"backbone desconocido para attn: {backbone_name!r}")
    return _DINO_HEADS[key]


def _jpeg_rgb_mb(width: int, height: int) -> float:
    return width * height * 3 / (1024 * 1024)


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def _use_cpu_decode_for_cache(path: Path, threshold_mb: float) -> bool:
    if threshold_mb <= 0:
        return False
    try:
        w, h = _jpeg_dimensions(path)
        return _jpeg_rgb_mb(w, h) >= threshold_mb
    except Exception as e:
        log.warning(f"[Gate embed] no se pudo leer dimensiones de {path.name}: {e}")
        return False


def _batch_tiles_cpu_to_gpu(
    image_arr: np.ndarray,
    rowcols: list[tuple[int, int, int]],
    device: torch.device,
    *,
    pad_value: int = 255,
) -> torch.Tensor:
    """Crop en RAM + sube solo el mini-batch a VRAM (imagenes gigantes)."""
    from ..phase_b_tiling.tile_cutter import crop_tile_from_array

    if not rowcols:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=device)
    ts = rowcols[0][2]
    stacked = np.full((len(rowcols), ts, ts, 3), pad_value, dtype=np.uint8)
    for i, (row, col, tile_size) in enumerate(rowcols):
        if tile_size != ts:
            raise ValueError("batch_tiles_cpu_to_gpu requiere tile_size uniforme")
        stacked[i] = crop_tile_from_array(
            image_arr, row=row, col=col, tile_size=ts, pad_value=pad_value
        )
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).to(device, non_blocking=True)


def _forward_embed_batch(
    backbone: nn.Module,
    tiles: torch.Tensor,
    *,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    seg_target_size: int = 360,
    attention_cfg: Optional[object] = None,
    labels: Optional[torch.Tensor] = None,
    u2net: Optional[nn.Module] = None,
    pooling_mode: str = "none",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    from .gate_bg_pooling import bg_only_pooling_mask, u2net_saliency_batch
    from .gate_dino_attention import attention_maps_to_numpy, extract_dino_cls_patch_attention

    views = build_views_gpu(
        tiles, target_size=dino_input_size, seg_target_size=seg_target_size
    )
    pool_mask = None
    if pooling_mode == "bg_only" and u2net is not None and labels is not None:
        sal = u2net_saliency_batch(
            u2net, views.seg, saliency_size=dino_input_size
        )
        pool_mask = bg_only_pooling_mask(labels, sal, pooling_mode=pooling_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        feat = backbone(views.rgb, mask=pool_mask)
    feat_np = feat.detach().half().cpu().numpy()
    if attention_cfg is None:
        return feat_np
    attn_maps = extract_dino_cls_patch_attention(
        backbone,
        views.rgb.float(),
        cfg=attention_cfg,
        dino_input_size=dino_input_size,
    )
    attn_np = attention_maps_to_numpy(attn_maps)
    return feat_np, attn_np


def _forward_embed_h5_batch(
    backbone: nn.Module,
    rgb: torch.Tensor,
    saliency: torch.Tensor,
    *,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    attention_cfg: Optional[object] = None,
    labels: Optional[torch.Tensor] = None,
    pooling_mode: str = "none",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """DINO embed desde tiles precomputados en HDF5 (sin JPEG ni U2Net)."""
    import torch.nn.functional as F

    from .gate_bg_pooling import bg_only_pooling_mask
    from .gate_dino_attention import attention_maps_to_numpy, extract_dino_cls_patch_attention

    x = rgb.float()
    if x.shape[-1] != dino_input_size:
        x = F.interpolate(
            x, size=(dino_input_size, dino_input_size), mode="bilinear", align_corners=False
        )
    pool_mask = None
    if pooling_mode == "bg_only" and labels is not None:
        sal = F.interpolate(
            saliency.float().unsqueeze(1),
            size=(dino_input_size, dino_input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        pool_mask = bg_only_pooling_mask(labels, sal, pooling_mode=pooling_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        feat = backbone(x, mask=pool_mask)
    feat_np = feat.detach().half().cpu().numpy()
    if attention_cfg is None:
        return feat_np
    attn_maps = extract_dino_cls_patch_attention(
        backbone,
        x.float(),
        cfg=attention_cfg,
        dino_input_size=dino_input_size,
    )
    attn_np = attention_maps_to_numpy(attn_maps)
    return feat_np, attn_np


def _h5_rgb_to_tiles(rgb: torch.Tensor) -> torch.Tensor:
    """Convierte RGB normalizado HDF5 [B,3,H,W] a tiles uint8 para aug M+."""
    return (rgb.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def _batch_size_for_image(
    base: int,
    gimg: GPUImage,
    *,
    dynamic: bool,
    vram_budget_mb: float = DEFAULT_VRAM_BUDGET_MB,
    model_reserve_mb: float = MODEL_RESERVE_MB,
) -> int:
    """Reduce batch si el JPEG decodificado ocupa mucha VRAM (evita spill a RAM compartida)."""
    if not dynamic:
        return base
    img_mb = gimg.vram_mb
    if img_mb >= 3500:
        eff = max(MIN_IMAGE_BATCH, base // 8)
    elif img_mb >= 2500:
        eff = max(MIN_IMAGE_BATCH, base // 4)
    elif img_mb >= 1500:
        eff = max(8, base // 2)
    elif img_mb >= 700:
        eff = max(12, base // 2)
    else:
        eff = base
    usable = max(256.0, vram_budget_mb - img_mb - model_reserve_mb)
    cap = max(MIN_IMAGE_BATCH, int(usable / 48.0))
    eff = min(base, eff, cap)
    if eff < base:
        _progress_print(
            f"[Gate embed]   VRAM imagen {img_mb:.0f} MB -> batch {base}->{eff} "
            f"(presupuesto {vram_budget_mb:.0f} MB)"
        )
    return max(MIN_IMAGE_BATCH, eff)


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _progress_print(msg: str) -> None:
    """Salida inmediata en terminal (Rich no hace flush durante operaciones largas)."""
    print(msg, flush=True)


def _progress_name(cache_basename: Optional[str] = None) -> str:
    return f"{cache_basename or _gate_cache_basename()}{PROGRESS_SUFFIX}"


def _progress_path_for(
    cpaths: Optional["EmbedCachePaths"] = None, *, cache_basename: Optional[str] = None
) -> Path:
    if cpaths is not None:
        return cpaths.embed.parent / _progress_name(cpaths.embed.stem)
    return get_paths().root / "cache" / _progress_name(cache_basename)


def _write_build_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    try:
        from .gate_run_live import publish_pipeline_panel

        pct = float(payload.get("pct", 0.0) or 0.0)
        status = "done" if payload.get("status") == "completed" else "running"
        detail = (
            payload.get("detail")
            or payload.get("current_image")
            or payload.get("last_image")
            or f"{payload.get('tiles_done', 0)}/{payload.get('tiles_total', 0)} tiles"
        )
        publish_pipeline_panel(
            pipeline_phase=0,
            phases=[
                {
                    "id": "embed_v5",
                    "label": "Paso 2: Embeddings v5 (DINO bg_only)",
                    "status": status,
                    "pct": round(max(0.0, min(100.0, pct)), 2),
                    "eta": payload.get("eta"),
                    "detail": str(detail),
                }
            ],
        )
    except Exception:
        pass


@dataclass(frozen=True)
class EmbedCachePaths:
    embed: Path
    label: Path
    meta: Path
    lookup: Path
    attn: Path


@dataclass(frozen=True)
class EmbedCacheStatus:
    """Estado de la cache de embeddings para prompts en run.py."""

    state: str  # missing | partial | finalize | valid | stale | attn_missing
    n_tiles_expected: int
    n_tiles_meta: Optional[int]
    fingerprint_expected: str
    fingerprint_meta: Optional[str]
    paths: EmbedCachePaths
    embed_mb: float
    message: str


def cache_paths_for_root(root: Path) -> EmbedCachePaths:
    """Rutas publicas de la cache v2 bajo `root/cache/`."""
    return _cache_paths(root)


def _embed_files_complete(
    cpaths: EmbedCachePaths,
    *,
    n_tiles: int,
    embed_dim: int,
    attn_flat_dim: int = 0,
) -> bool:
    if not cpaths.embed.exists() or not cpaths.label.exists():
        return False
    expected_embed = n_tiles * embed_dim * np.dtype(np.float16).itemsize
    expected_label = n_tiles * np.dtype(np.int8).itemsize
    try:
        if cpaths.embed.stat().st_size != expected_embed:
            return False
        if cpaths.label.stat().st_size != expected_label:
            return False
        if attn_flat_dim > 0:
            if not cpaths.attn.exists():
                return False
            expected_attn = n_tiles * attn_flat_dim * np.dtype(np.float16).itemsize
            if cpaths.attn.stat().st_size != expected_attn:
                return False
        return True
    except OSError:
        return False


def _write_lookup_parquet(lookup_df: pd.DataFrame, path: Path) -> None:
    try:
        lookup_df.to_parquet(path, index=False)
    except ImportError as e:
        raise ImportError(
            "pyarrow requerido para gate_am_embeds_v2.lookup.parquet. "
            "Instala dependencias: pip install -r requirements.txt"
        ) from e


def reconstruct_mplus_aug_lookup_rows(
    tiles_df: pd.DataFrame,
    *,
    n_base: int,
    mplus_aug_variants: tuple[str, ...],
    mplus_aug_train_images: set[str],
) -> list[dict]:
    """Reconstruye filas lookup aug (mismo orden que build_gate_embed_cache)."""
    if not mplus_aug_variants:
        return []
    aug_rows: list[dict] = []
    aug_idx = int(n_base)
    df = tiles_df.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
    for image_rel, grp in df.groupby("image_path", sort=False):
        if str(image_rel) not in mplus_aug_train_images:
            continue
        mplus = grp[grp["stage1"].astype(str) == "Mplus"]
        for _, row in mplus.iterrows():
            for vid, _variant in enumerate(mplus_aug_variants, start=1):
                aug_rows.append(
                    {
                        "embed_idx": aug_idx,
                        "image_path": str(image_rel),
                        "row": int(row["row"]),
                        "col": int(row["col"]),
                        "tile_size": int(row["tile_size"]) if "tile_size" in row.index else None,
                        "aug_id": vid,
                    }
                )
                aug_idx += 1
    return aug_rows


def finalize_embed_cache_metadata(
    tiles_df: pd.DataFrame,
    *,
    backbone_name: str = "dinov2_vits14",
    embed_dim: int = DEFAULT_EMBED_DIM,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    seg_target_size: int = 360,
    tiles_index_path: Optional[Path] = None,
    cache_paths: Optional[EmbedCachePaths] = None,
    build_seconds: float = 0.0,
    aug_rows: Optional[list[dict]] = None,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
    n_tiles_total: Optional[int] = None,
    attention_meta: Optional[dict] = None,
    pooling_mode: str = "none",
) -> EmbedCachePaths:
    """Escribe lookup.parquet + meta.json cuando embed/label ya estan completos."""
    paths_root = get_paths()
    tiles_index_path = tiles_index_path or (paths_root.manifests / "tiles_index.csv")
    cpaths = cache_paths or _cache_paths(paths_root.root)
    df = tiles_df.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
    n = len(df)
    n_total = int(n_tiles_total) if n_tiles_total is not None else n
    fingerprint = _manifest_fingerprint(
        tiles_index_path,
        backbone_name,
        dino_input_size,
        mplus_aug_variants=mplus_aug_variants,
        cache_attention=bool(attention_meta),
        attention_layers=str((attention_meta or {}).get("layers_spec", "all")),
        attention_head_reduce=str((attention_meta or {}).get("head_reduce", "mean")),
        pooling_mode=pooling_mode,
    )
    gh, gw = dino_patch_grid(dino_input_size)

    if aug_rows is None and n_total > n and mplus_aug_variants and mplus_aug_train_images:
        aug_rows = reconstruct_mplus_aug_lookup_rows(
            df,
            n_base=n,
            mplus_aug_variants=mplus_aug_variants,
            mplus_aug_train_images=mplus_aug_train_images,
        )
        expected_aug = n_total - n
        if len(aug_rows) != expected_aug:
            log.warning(
                f"[Gate embed] lookup aug reconstruido {len(aug_rows)} filas, esperado {expected_aug}"
            )

    attn_flat = int((attention_meta or {}).get("flat_dim", 0))
    if not _embed_files_complete(
        cpaths, n_tiles=n_total, embed_dim=embed_dim, attn_flat_dim=attn_flat
    ):
        raise RuntimeError(
            "[Gate embed] No se puede finalizar: embed/label/attn incompletos o ausentes."
        )

    _progress_print(f"[Gate embed] Finalizando metadata ({n_total:,} tiles, base={n:,}) -> lookup + meta...")
    lookup_df = _build_lookup_table(df, aug_rows)
    _write_lookup_parquet(lookup_df, cpaths.lookup)
    size_parts = [cpaths.embed, cpaths.label]
    if attn_flat > 0 and cpaths.attn.exists():
        size_parts.append(cpaths.attn)
    size_mb = sum(p.stat().st_size for p in size_parts) / (1024**2)
    meta_payload = {
                "version": CACHE_VERSION,
                "n_tiles": n_total,
                "n_tiles_base": n,
                "mplus_aug_variants": list(mplus_aug_variants),
                "embed_dim": embed_dim,
                "backbone": backbone_name,
                "saliency_mask_pooling": (pooling_mode or "none").lower() == "global",
                "pooling_mode": (pooling_mode or "none").lower(),
                "dino_input_size": dino_input_size,
                "seg_target_size": seg_target_size,
                "patch_grid": [gh, gw],
                "fingerprint": fingerprint,
                "build_seconds": round(build_seconds, 1),
                "size_mb": round(size_mb, 2),
                "embed_path": str(cpaths.embed),
            }
    if attention_meta:
        meta_payload["attention"] = attention_meta
    with open(cpaths.meta, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    progress_path = _progress_path_for(cpaths)
    _write_build_progress(
        progress_path,
        {
            "status": "completed",
            "tiles_total": n_total,
            "tiles_done": n_total,
            "pct": 100.0,
            "size_mb": round(size_mb, 2),
            "finalized": True,
        },
    )
    try:
        progress_path.unlink(missing_ok=True)
    except OSError:
        pass
    _progress_print(f"[Gate embed] Metadata OK -> {cpaths.lookup.name}, {cpaths.meta.name}")
    return cpaths


def _patch_meta_with_attention(
    cpaths: EmbedCachePaths,
    meta: dict,
    *,
    attention_meta: dict,
    fingerprint: str,
    attn_build_seconds: float,
) -> None:
    """Actualiza meta.json existente tras compilar solo mapas de atencion."""
    updated = dict(meta)
    updated["attention"] = attention_meta
    updated["fingerprint"] = fingerprint
    updated["attn_build_seconds"] = round(attn_build_seconds, 1)
    size_parts = [cpaths.embed, cpaths.label]
    if cpaths.attn.exists():
        size_parts.append(cpaths.attn)
    updated["size_mb"] = round(sum(p.stat().st_size for p in size_parts) / (1024**2), 2)
    with open(cpaths.meta, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    progress_path = _progress_path_for(cpaths)
    _write_build_progress(
        progress_path,
        {
            "status": "completed",
            "tiles_total": int(updated.get("n_tiles", 0)),
            "tiles_done": int(updated.get("n_tiles", 0)),
            "pct": 100.0,
            "size_mb": updated.get("size_mb"),
            "attn_only": True,
        },
    )
    try:
        progress_path.unlink(missing_ok=True)
    except OSError:
        pass
    _progress_print(f"[Gate embed] Metadata attn OK -> {cpaths.meta.name}, {cpaths.attn.name}")


def build_gate_attention_cache_only(
    tiles_df: pd.DataFrame,
    backbone: nn.Module,
    *,
    device: torch.device,
    **kwargs,
) -> EmbedCachePaths:
    """Anade mapas ViT CLS->patch sin recomputar embeddings existentes."""
    return build_gate_embed_cache(
        tiles_df,
        backbone,
        device=device,
        attn_only=True,
        cache_attention=True,
        **kwargs,
    )


def expected_embed_tile_count(
    tiles_df: pd.DataFrame,
    *,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
) -> int:
    """Tiles base + vistas augmentadas M+ (solo train)."""
    n = len(tiles_df)
    if not mplus_aug_variants:
        return n
    mplus = tiles_df["stage1"].astype(str) == "Mplus"
    if mplus_aug_train_images is not None:
        mplus &= tiles_df["image_path"].astype(str).isin(mplus_aug_train_images)
    return n + int(mplus.sum()) * len(mplus_aug_variants)


def inspect_embed_cache_status(
    tiles_df: pd.DataFrame,
    *,
    backbone_name: str = "dinov2_vits14",
    embed_dim: int = DEFAULT_EMBED_DIM,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    tiles_index_path: Optional[Path] = None,
    cache_basename: Optional[str] = None,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
    cache_attention: bool = False,
    attention_layers: str = "all",
    attention_head_reduce: str = "mean",
    pooling_mode: str = "none",
) -> EmbedCacheStatus:
    """Detecta si la cache existe, esta incompleta, es valida u obsoleta."""
    paths_root = get_paths()
    tiles_index_path = tiles_index_path or (paths_root.manifests / "tiles_index.csv")
    cpaths = _cache_paths(paths_root.root, cache_basename)
    df = tiles_df.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
    n_base = len(df)
    n_expected = expected_embed_tile_count(
        df,
        mplus_aug_variants=mplus_aug_variants,
        mplus_aug_train_images=mplus_aug_train_images,
    )
    fingerprint = _manifest_fingerprint(
        tiles_index_path,
        backbone_name,
        dino_input_size,
        mplus_aug_variants=mplus_aug_variants,
        cache_attention=cache_attention,
        attention_layers=attention_layers,
        attention_head_reduce=attention_head_reduce,
        pooling_mode=pooling_mode,
    )

    present = {
        "embed": cpaths.embed.exists(),
        "label": cpaths.label.exists(),
        "meta": cpaths.meta.exists(),
        "lookup": cpaths.lookup.exists(),
        "attn": cpaths.attn.exists(),
    }
    any_file = any(present.values())
    all_files = all(present.values())
    embed_mb = cpaths.embed.stat().st_size / (1024 * 1024) if present["embed"] else 0.0

    meta: Optional[dict] = None
    n_meta: Optional[int] = None
    fp_meta: Optional[str] = None
    if present["meta"]:
        try:
            with open(cpaths.meta, encoding="utf-8") as f:
                meta = json.load(f)
            n_meta = meta.get("n_tiles")
            fp_meta = meta.get("fingerprint")
        except Exception:
            meta = None

    if meta and meta.get("saliency_mask_pooling") and meta.get("pooling_mode") != "bg_only":
        return EmbedCacheStatus(
            state="obsolete",
            n_tiles_expected=n_expected,
            n_tiles_meta=n_meta,
            fingerprint_expected=fingerprint,
            fingerprint_meta=fp_meta,
            paths=cpaths,
            embed_mb=embed_mb,
            message="Cache obsoleta (pooling global U2Net). Recompila: python run.py build-gate-cache",
        )
    meta_pool = (meta or {}).get("pooling_mode", "none")
    if meta and meta_pool != (pooling_mode or "none").lower():
        return EmbedCacheStatus(
            state="obsolete",
            n_tiles_expected=n_expected,
            n_tiles_meta=n_meta,
            fingerprint_expected=fingerprint,
            fingerprint_meta=fp_meta,
            paths=cpaths,
            embed_mb=embed_mb,
            message=(
                f"Cache obsoleta (pooling_mode={meta_pool}, esperado {pooling_mode}). "
                "Recompila: python run.py build-gate-cache"
            ),
        )

    if cache_is_valid(n_tiles=n_expected, fingerprint=fingerprint, embed_dim=embed_dim, paths=cpaths):
        return EmbedCacheStatus(
            state="valid",
            n_tiles_expected=n_expected,
            n_tiles_meta=n_meta,
            fingerprint_expected=fingerprint,
            fingerprint_meta=fp_meta,
            paths=cpaths,
            embed_mb=embed_mb,
            message=f"Cache valida: {n_expected:,} tiles ({n_base:,} base), {embed_mb:.1f} MB",
        )

    if (
        cache_attention
        and meta is not None
        and present["embed"]
        and present["label"]
        and present["meta"]
        and present["lookup"]
        and not present["attn"]
        and n_meta is not None
        and int(n_meta) == n_expected
        and not meta.get("attention")
        and str(meta.get("backbone", backbone_name)) == backbone_name
        and int(meta.get("dino_input_size", dino_input_size)) == dino_input_size
        and not meta.get("saliency_mask_pooling", False)
        and _embed_files_complete(cpaths, n_tiles=n_expected, embed_dim=embed_dim)
    ):
        from .gate_dino_attention import dino_attention_config_from_strings, describe_attention_storage

        n_blocks = _dino_num_blocks(backbone_name)
        attn_cfg = dino_attention_config_from_strings(
            layers_spec=attention_layers,
            num_blocks=n_blocks,
            num_heads=_dino_num_heads(backbone_name),
            head_reduce=attention_head_reduce,
            dino_input_size=dino_input_size,
        )
        attn_est = describe_attention_storage(
            attn_cfg,
            num_heads=_dino_num_heads(backbone_name),
            dino_input_size=dino_input_size,
            n_tiles=n_expected,
        )
        return EmbedCacheStatus(
            state="attn_missing",
            n_tiles_expected=n_expected,
            n_tiles_meta=n_meta,
            fingerprint_expected=fingerprint,
            fingerprint_meta=fp_meta,
            paths=cpaths,
            embed_mb=embed_mb,
            message=(
                f"Embeddings OK ({n_expected:,} tiles, {embed_mb:.1f} MB). "
                f"Faltan mapas ViT attn (~{attn_est['size_mb']:.0f} MB, capas={attention_layers})."
            ),
        )

    if not any_file:
        return EmbedCacheStatus(
            state="missing",
            n_tiles_expected=n_expected,
            n_tiles_meta=None,
            fingerprint_expected=fingerprint,
            fingerprint_meta=None,
            paths=cpaths,
            embed_mb=0.0,
            message=f"Sin cache. Se compilaran {n_expected:,} tiles (~150 MB).",
        )

    if (
        _embed_files_complete(cpaths, n_tiles=n_expected, embed_dim=embed_dim)
        and not all_files
        and not (present["meta"] and present["lookup"])
    ):
        return EmbedCacheStatus(
            state="finalize",
            n_tiles_expected=n_expected,
            n_tiles_meta=n_meta,
            fingerprint_expected=fingerprint,
            fingerprint_meta=fp_meta,
            paths=cpaths,
            embed_mb=embed_mb,
            message=(
                f"Embeddings completos ({n_expected:,} tiles, {embed_mb:.1f} MB). "
                "Falta metadata; se finalizara sin recompilar (~segundos)."
            ),
        )

    progress_path = _progress_path_for(cpaths)
    if progress_path.exists() or (present["embed"] and not all_files):
        if progress_path.exists():
            try:
                with open(progress_path, encoding="utf-8") as f:
                    prog = json.load(f)
                if prog.get("status") == "completed" and all_files:
                    progress_path.unlink(missing_ok=True)
                else:
                    return EmbedCacheStatus(
                        state="partial",
                        n_tiles_expected=n_expected,
                        n_tiles_meta=n_meta,
                        fingerprint_expected=fingerprint,
                        fingerprint_meta=fp_meta,
                        paths=cpaths,
                        embed_mb=embed_mb,
                        message=f"Cache incompleta ({embed_mb:.1f} MB). Requiere recompilar.",
                    )
            except Exception:
                return EmbedCacheStatus(
                    state="partial",
                    n_tiles_expected=n_expected,
                    n_tiles_meta=n_meta,
                    fingerprint_expected=fingerprint,
                    fingerprint_meta=fp_meta,
                    paths=cpaths,
                    embed_mb=embed_mb,
                    message=f"Cache incompleta ({embed_mb:.1f} MB). Requiere recompilar.",
                )
        elif present["embed"] and not all_files:
            return EmbedCacheStatus(
                state="partial",
                n_tiles_expected=n_expected,
                n_tiles_meta=n_meta,
                fingerprint_expected=fingerprint,
                fingerprint_meta=fp_meta,
                paths=cpaths,
                embed_mb=embed_mb,
                message=f"Cache incompleta ({embed_mb:.1f} MB). Requiere recompilar.",
            )

    return EmbedCacheStatus(
        state="stale",
        n_tiles_expected=n_expected,
        n_tiles_meta=n_meta,
        fingerprint_expected=fingerprint,
        fingerprint_meta=fp_meta,
        paths=cpaths,
        embed_mb=embed_mb,
        message=(
            f"Cache obsoleta (meta={n_meta} tiles, esperado {n_expected:,}). "
            "Recompilar para alinear con el manifest actual."
        ),
    )


def cache_paths_with_basename(root: Path, basename: str) -> EmbedCachePaths:
    """Rutas de cache con prefijo arbitrario (p. ej. eval AMFinder)."""
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return EmbedCachePaths(
        embed=cache_dir / f"{basename}.embed.npy",
        label=cache_dir / f"{basename}.label.npy",
        meta=cache_dir / f"{basename}.meta.json",
        lookup=cache_dir / f"{basename}.lookup.parquet",
        attn=cache_dir / f"{basename}.attn.npy",
    )


def _gate_cache_basename() -> str:
    try:
        import config as user_config  # type: ignore

        return str(getattr(user_config, "GATE_CACHE_BASENAME", "gate_am_embeds_v3"))
    except Exception:
        return "gate_am_embeds_v3"


def _cache_paths(root: Path, basename: Optional[str] = None) -> EmbedCachePaths:
    return cache_paths_with_basename(root, basename or _gate_cache_basename())


def _manifest_fingerprint(
    tiles_index_path: Path,
    backbone_name: str,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    *,
    mplus_aug_variants: tuple[str, ...] = (),
    cache_attention: bool = False,
    attention_layers: str = "all",
    attention_head_reduce: str = "mean",
    pooling_mode: str = "none",
) -> str:
    st = tiles_index_path.stat()
    aug_tag = ",".join(mplus_aug_variants) if mplus_aug_variants else "none"
    mask_tag = (pooling_mode or "none").lower()
    attn_tag = (
        f"cls:{attention_layers}:{attention_head_reduce}"
        if cache_attention
        else "off"
    )
    payload = (
        f"{tiles_index_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        f"|v{CACHE_VERSION}|{backbone_name}|din{dino_input_size}|mplus_aug={aug_tag}|pool={mask_tag}|attn={attn_tag}"
    )
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def cache_is_valid(
    *,
    n_tiles: int,
    fingerprint: str,
    embed_dim: int,
    paths: EmbedCachePaths,
) -> bool:
    if not all(p.exists() for p in (paths.embed, paths.label, paths.meta, paths.lookup)):
        return False
    try:
        with open(paths.meta, encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("n_tiles") != n_tiles
            or meta.get("fingerprint") != fingerprint
            or meta.get("embed_dim") != embed_dim
        ):
            return False
        if not _embed_files_complete(cpaths=paths, n_tiles=n_tiles, embed_dim=embed_dim):
            return False
        attn_meta = meta.get("attention") or {}
        attn_flat = int(attn_meta.get("flat_dim", 0))
        if attn_flat > 0:
            if not _embed_files_complete(
                cpaths=paths, n_tiles=n_tiles, embed_dim=embed_dim, attn_flat_dim=attn_flat
            ):
                return False
        return True
    except Exception:
        return False


def _clear_cache_files(cpaths: EmbedCachePaths) -> None:
    progress = _progress_path_for(cpaths)
    for p in (cpaths.embed, cpaths.label, cpaths.meta, cpaths.lookup, cpaths.attn, progress):
        try:
            p.unlink(missing_ok=True)  # type: ignore[arg-type]
        except OSError as e:
            log.warning(f"[Gate embed] no se pudo borrar {p.name}: {e}")


def _build_lookup_table(df: pd.DataFrame, aug_rows: Optional[list[dict]] = None) -> pd.DataFrame:
    out = df[["image_path", "row", "col"]].copy()
    out["aug_id"] = 0
    out["embed_idx"] = np.arange(len(df), dtype=np.int64)
    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        out = pd.concat([out, aug_df], ignore_index=True)
    return out[["embed_idx", "image_path", "row", "col", "aug_id"]]


def _flatten_attn_batch(attn_np: np.ndarray) -> np.ndarray:
    return attn_np.reshape(attn_np.shape[0], -1).astype(np.float16, copy=False)


def _resolve_attention_cfg(
    backbone: nn.Module,
    *,
    cache_attention: bool,
    attention_layers: str,
    attention_head_reduce: str,
    dino_input_size: int,
    n_tiles: int = 0,
) -> tuple[Optional[object], Optional[dict]]:
    if not cache_attention:
        return None, None
    from .gate_dino_attention import (
        dino_attention_config_from_strings,
        describe_attention_storage,
    )

    dino = backbone.model if hasattr(backbone, "model") else backbone
    cfg = dino_attention_config_from_strings(
        layers_spec=attention_layers,
        num_blocks=len(dino.blocks),
        num_heads=int(dino.num_heads),
        head_reduce=attention_head_reduce,
        dino_input_size=dino_input_size,
    )
    meta = describe_attention_storage(
        cfg,
        num_heads=int(dino.num_heads),
        dino_input_size=dino_input_size,
        n_tiles=n_tiles,
    )
    meta["layers_spec"] = attention_layers
    meta["head_reduce"] = attention_head_reduce
    return cfg, meta


@torch.no_grad()
def _cuda_warmup(
    backbone: nn.Module,
    *,
    device: torch.device,
    image_path: Path,
    row: int,
    col: int,
    tile_size: int,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    seg_target_size: int = 360,
) -> None:
    """Primer forward DINO compila kernels CUDA; sin esto parece colgado 1-3 min."""
    _progress_print(
        f"[Gate embed] Warmup CUDA DINO @ {dino_input_size}px (1 tile, puede tardar 1-3 min)..."
    )
    t0 = time.perf_counter()
    gimg = decode_jpeg_gpu(image_path, device=device)
    tiles = batch_tiles_gpu(gimg, [(row, col, tile_size)])
    views = build_views_gpu(
        tiles, target_size=dino_input_size, seg_target_size=seg_target_size
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = backbone(views.rgb, mask=None)
    torch.cuda.synchronize()
    del gimg, tiles, views
    release_cuda_memory(gc_collect=True)
    _progress_print(f"[Gate embed] Warmup listo ({time.perf_counter() - t0:.1f}s). Compilando tiles...")


def _write_mplus_aug_embeddings(
    *,
    tiles: torch.Tensor,
    grp: pd.DataFrame,
    start: int,
    stop: int,
    backbone: nn.Module,
    embed_arr: Optional[np.memmap] = None,
    label_arr: Optional[np.memmap] = None,
    aug_write_idx: int,
    aug_rows: list[dict],
    image_rel: str,
    mplus_aug_variants: tuple[str, ...],
    dino_input_size: int,
    seg_target_size: int,
    attention_cfg: Optional[object] = None,
    attn_arr: Optional[np.memmap] = None,
    u2net: Optional[nn.Module] = None,
    pooling_mode: str = "none",
) -> int:
    from .gate_classes import GATE_CLASS_TO_IDX
    from .gate_tile_augment import apply_gate_tile_aug

    mplus_idx = int(GATE_CLASS_TO_IDX["Mplus"])
    stage1_batch = grp["stage1"].to_numpy()[start:stop]
    rows_batch = grp["row"].to_numpy()[start:stop]
    cols_batch = grp["col"].to_numpy()[start:stop]
    for local_i in range(tiles.shape[0]):
        if str(stage1_batch[local_i]) != "Mplus":
            continue
        tile_i = tiles[local_i : local_i + 1]
        row_i = int(rows_batch[local_i])
        col_i = int(cols_batch[local_i])
        tile_sz = int(grp["tile_size"].iloc[start + local_i]) if "tile_size" in grp.columns else None
        for vid, variant in enumerate(mplus_aug_variants, start=1):
            aug_tile = apply_gate_tile_aug(tile_i, variant)
            out = _forward_embed_batch(
                backbone,
                aug_tile,
                dino_input_size=dino_input_size,
                seg_target_size=seg_target_size,
                attention_cfg=attention_cfg,
                labels=torch.tensor([mplus_idx], device=aug_tile.device),
                u2net=u2net,
                pooling_mode=pooling_mode,
            )
            if attention_cfg is None:
                aug_feat = out
            else:
                aug_feat, aug_attn = out
                if attn_arr is not None:
                    attn_arr[aug_write_idx : aug_write_idx + 1] = _flatten_attn_batch(aug_attn)
            if embed_arr is not None:
                embed_arr[aug_write_idx : aug_write_idx + 1] = aug_feat
            if label_arr is not None:
                label_arr[aug_write_idx] = mplus_idx
            aug_rows.append(
                {
                    "embed_idx": aug_write_idx,
                    "image_path": image_rel,
                    "row": row_i,
                    "col": col_i,
                    "tile_size": tile_sz,
                    "aug_id": vid,
                }
            )
            aug_write_idx += 1
            del aug_tile, aug_feat
    return aug_write_idx


@torch.inference_mode()
def build_gate_embed_cache(
    tiles_df: pd.DataFrame,
    backbone: nn.Module,
    *,
    device: torch.device,
    backbone_name: str = "dinov2_vits14",
    embed_dim: int = DEFAULT_EMBED_DIM,
    dino_input_size: int = DEFAULT_DINO_INPUT,
    seg_target_size: int = 360,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tiles_index_path: Optional[Path] = None,
    cache_paths: Optional[EmbedCachePaths] = None,
    force_rebuild: bool = False,
    dynamic_batch: bool = True,
    vram_budget_mb: float = DEFAULT_VRAM_BUDGET_MB,
    cpu_decode_above_mb: float = DEFAULT_CPU_DECODE_ABOVE_MB,
    empty_cache_every_n_batches: int = 0,
    gc_collect_every_n_batches: int = 0,
    memmap_flush_every_n_images: int = 5,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
    cache_attention: bool = False,
    attention_layers: str = "all",
    attention_head_reduce: str = "mean",
    attn_only: bool = False,
    pooling_mode: str = "none",
    h5_store: Optional[object] = None,
    # Compat legacy (config antiguo):
    gpu_cleanup_each_batch: Optional[bool] = None,
) -> EmbedCachePaths:
    """Compila tiles AM -> memmap embed float16 + label int8 (+ attn opcional)."""
    if gpu_cleanup_each_batch is not None and empty_cache_every_n_batches == 0:
        empty_cache_every_n_batches = 1 if gpu_cleanup_each_batch else 0
    if attn_only:
        cache_attention = True
    paths_root = get_paths()
    tiles_index_path = tiles_index_path or (paths_root.manifests / "tiles_index.csv")
    cpaths = cache_paths or _cache_paths(paths_root.root)
    fingerprint = _manifest_fingerprint(
        tiles_index_path,
        backbone_name,
        dino_input_size,
        mplus_aug_variants=mplus_aug_variants,
        cache_attention=cache_attention,
        attention_layers=attention_layers,
        attention_head_reduce=attention_head_reduce,
        pooling_mode=pooling_mode,
    )

    df = tiles_df.sort_values(["image_path", "row", "col"]).reset_index(drop=True)
    n = len(df)
    n_total = expected_embed_tile_count(
        df,
        mplus_aug_variants=mplus_aug_variants,
        mplus_aug_train_images=mplus_aug_train_images,
    )

    existing_meta: Optional[dict] = None
    if attn_only:
        if not cpaths.meta.exists():
            raise RuntimeError("[Gate embed] attn_only requiere meta.json existente.")
        with open(cpaths.meta, encoding="utf-8") as f:
            existing_meta = json.load(f)
        n_total = int(existing_meta["n_tiles"])
        embed_dim = int(existing_meta.get("embed_dim", embed_dim))
        dino_input_size = int(existing_meta.get("dino_input_size", dino_input_size))
        seg_target_size = int(existing_meta.get("seg_target_size", seg_target_size))
        raw_aug = existing_meta.get("mplus_aug_variants") or []
        if raw_aug and not mplus_aug_variants:
            mplus_aug_variants = tuple(str(v) for v in raw_aug)
        if not _embed_files_complete(cpaths, n_tiles=n_total, embed_dim=embed_dim):
            raise RuntimeError("[Gate embed] attn_only: embed/label incompletos.")
        if (
            not force_rebuild
            and existing_meta.get("fingerprint") == fingerprint
            and existing_meta.get("attention")
            and cpaths.attn.exists()
        ):
            attn_flat = int(existing_meta["attention"].get("flat_dim", 0))
            if _embed_files_complete(
                cpaths, n_tiles=n_total, embed_dim=embed_dim, attn_flat_dim=attn_flat
            ):
                log.info(f"[Gate embed] CACHE HIT attn -> {cpaths.attn}")
                return cpaths
    elif (
        not force_rebuild
        and cache_is_valid(n_tiles=n_total, fingerprint=fingerprint, embed_dim=embed_dim, paths=cpaths)
    ):
        log.info(f"[Gate embed] CACHE HIT -> {cpaths.embed}")
        return cpaths

    if attn_only and not cache_attention:
        raise ValueError("cache_attention=False incompatible con attn_only=True")

    attention_cfg = None
    attention_meta = None
    if cache_attention or attn_only:
        attention_cfg, attention_meta = _resolve_attention_cfg(
            backbone,
            cache_attention=True,
            attention_layers=attention_layers,
            attention_head_reduce=attention_head_reduce,
            dino_input_size=dino_input_size,
            n_tiles=int(n_total),
        )

    t0 = time.perf_counter()
    aug_note = f" + {n_total - n:,} aug M+" if n_total > n else ""
    pool_note = "mean-pool DINO" if pooling_mode in {"none", ""} else f"pool={pooling_mode}"
    attn_note = f" + attn {attention_layers}" if attention_cfg is not None else ""
    if attn_only:
        log.info(
            f"[Gate embed] Compilando SOLO atencion ViT {n_total:,} tiles "
            f"@ {describe_dino_resolution(dino_input_size)} [{attn_note}] -> {cpaths.attn}"
        )
    else:
        log.info(
            f"[Gate embed] Compilando {n_total:,} tiles ({n:,} base{aug_note}) "
            f"@ {describe_dino_resolution(dino_input_size)} [{pool_note}{attn_note}] -> {cpaths.embed}"
        )
    if not attn_only:
        _clear_cache_files(cpaths)
    elif cpaths.attn.exists() and force_rebuild:
        try:
            cpaths.attn.unlink()
        except OSError as e:
            log.warning(f"[Gate embed] no se pudo borrar attn previo: {e}")
    backbone.eval().to(device)

    u2net = None
    if pooling_mode == "bg_only" and h5_store is None:
        from .gate_tile_h5_cache import load_frozen_u2net_saliency

        weights = paths_root.root / "models" / "weights" / "u2netp.pth"
        u2net = load_frozen_u2net_saliency(weights, device)
        log.info(f"[Gate embed] E7 pooling bg_only: U2Net cargado ({weights.name})")
    elif pooling_mode == "bg_only" and h5_store is not None:
        log.info("[Gate embed] E7 pooling bg_only: saliencia desde HDF5 (sin U2Net en compilacion)")

    embed_arr: Optional[np.memmap] = None
    label_arr: Optional[np.memmap] = None
    if not attn_only:
        shape_embed = (int(n_total), int(embed_dim))
        shape_label = (int(n_total),)
        embed_arr = np.memmap(cpaths.embed, dtype=np.float16, mode="w+", shape=shape_embed)
        label_arr = np.memmap(cpaths.label, dtype=np.int8, mode="w+", shape=shape_label)
        label_arr[:n] = encode_gate_indices(df["stage1"].to_numpy()).astype(np.int8)
        label_arr.flush()
    attn_arr: Optional[np.memmap] = None
    attn_flat = int((attention_meta or {}).get("flat_dim", 0))
    if attn_flat > 0:
        attn_arr = np.memmap(
            cpaths.attn,
            dtype=np.float16,
            mode="w+",
            shape=(int(n_total), attn_flat),
        )

    write_idx = 0
    aug_write_idx = n
    aug_rows: list[dict] = []
    aug_train = mplus_aug_train_images or set()
    do_aug = bool(mplus_aug_variants) and bool(aug_train)
    groups = list(df.groupby("image_path", sort=False))
    n_images = len(groups)
    progress_path = _progress_path_for(cpaths)
    img_times: list[float] = []
    last_progress_t = t0

    _progress_print(
        f"[Gate embed] Inicio: {n_images} imagenes, {n_total:,} tiles ({n:,} base), batch={batch_size}. "
        f"Monitoreo: cache/{progress_path.name}"
    )
    _write_build_progress(
        progress_path,
        {
            "status": "running",
            "images_total": n_images,
            "images_done": 0,
            "tiles_total": n_total,
            "tiles_done": 0,
            "pct": 0.0,
        },
    )

    first = df.iloc[0]
    _cuda_warmup(
        backbone,
        device=device,
        image_path=paths_root.root / str(first["image_path"]),
        row=int(first["row"]),
        col=int(first["col"]),
        tile_size=int(first["tile_size"]),
        dino_input_size=dino_input_size,
        seg_target_size=seg_target_size,
    )

    for img_idx, (image_rel, grp) in enumerate(groups, start=1):
        img_name = Path(image_rel).name
        n_img = len(grp)
        t_img = time.perf_counter()

        h5_indices: Optional[np.ndarray] = None
        if h5_store is not None:
            try:
                h5_indices = h5_store.indices_for_sub(grp)
            except KeyError:
                h5_indices = None

        if h5_indices is not None:
            img_batch = batch_size
            n_batches = (n_img + img_batch - 1) // img_batch
            _progress_print(
                f"[Gate embed] ({img_idx}/{n_images}) HDF5 {img_name} "
                f"({n_img:,} tiles, acumulado {write_idx:,}/{n:,})"
            )
            rows = grp["row"].to_numpy(dtype=np.int32)
            cols = grp["col"].to_numpy(dtype=np.int32)
            for batch_num, start in enumerate(range(0, n_img, img_batch), start=1):
                stop = min(start + img_batch, n_img)
                idx = h5_indices[start:stop]
                rgb = sal = feat = None
                try:
                    rgb, sal, _ = h5_store.read_batch(idx, device)
                    label_slice = grp["stage1"].to_numpy()[start:stop]
                    label_idx = torch.from_numpy(encode_gate_indices(label_slice)).to(device)
                    out = _forward_embed_h5_batch(
                        backbone,
                        rgb,
                        sal,
                        dino_input_size=dino_input_size,
                        attention_cfg=attention_cfg,
                        labels=label_idx,
                        pooling_mode=pooling_mode,
                    )
                    if attention_cfg is None:
                        feat_np = out
                    else:
                        feat_np, attn_np = out
                    b = int(feat_np.shape[0])
                    if embed_arr is not None:
                        embed_arr[write_idx : write_idx + b] = feat_np
                    if attn_arr is not None and attention_cfg is not None:
                        attn_arr[write_idx : write_idx + b] = _flatten_attn_batch(attn_np)
                    write_idx += b
                    if do_aug and str(image_rel) in aug_train:
                        tiles_aug = _h5_rgb_to_tiles(rgb)
                        aug_write_idx = _write_mplus_aug_embeddings(
                            tiles=tiles_aug,
                            grp=grp,
                            start=start,
                            stop=stop,
                            backbone=backbone,
                            embed_arr=embed_arr,
                            label_arr=label_arr,
                            aug_write_idx=aug_write_idx,
                            aug_rows=aug_rows,
                            image_rel=str(image_rel),
                            mplus_aug_variants=mplus_aug_variants,
                            dino_input_size=dino_input_size,
                            seg_target_size=seg_target_size,
                            attention_cfg=attention_cfg,
                            attn_arr=attn_arr,
                            u2net=u2net,
                            pooling_mode=pooling_mode,
                        )
                    del feat_np
                finally:
                    if rgb is not None:
                        del rgb, sal, feat
                    maybe_empty_cache(batch_num, every_n=empty_cache_every_n_batches)
                    maybe_gc_collect(batch_num, every_n=gc_collect_every_n_batches)

                now = time.perf_counter()
                tiles_done = write_idx + (aug_write_idx - n)
                if batch_num == 1 or batch_num == n_batches or (now - last_progress_t) >= 5.0:
                    pct = 100.0 * tiles_done / max(n_total, 1)
                    elapsed = now - t0
                    tiles_per_s = tiles_done / max(elapsed, 0.001)
                    eta_s = (n_total - tiles_done) / max(tiles_per_s, 0.001)
                    _progress_print(
                        f"[Gate embed]   {img_name} batch {batch_num}/{n_batches} [HDF5] | "
                        f"tiles {tiles_done:,}/{n_total:,} ({pct:.1f}%) | "
                        f"{tiles_per_s:.0f} tiles/s | ETA ~{_format_duration(eta_s)}"
                    )
                    last_progress_t = now
                    _write_build_progress(
                        progress_path,
                        {
                            "status": "running",
                            "source": "hdf5",
                            "images_total": n_images,
                            "images_done": img_idx - 1,
                            "tiles_total": n_total,
                            "tiles_done": tiles_done,
                            "pct": round(pct, 2),
                        },
                    )

            img_times.append(time.perf_counter() - t_img)
            if memmap_flush_every_n_images and img_idx % memmap_flush_every_n_images == 0:
                if embed_arr is not None:
                    embed_arr.flush()
                if attn_arr is not None:
                    attn_arr.flush()
            release_cuda_memory(gc_collect=True)
            continue

        _progress_print(
            f"[Gate embed] ({img_idx}/{n_images}) decode {img_name} "
            f"({n_img:,} tiles, acumulado {write_idx:,}/{n:,})"
        )

        full = paths_root.root / image_rel
        use_cpu = _use_cpu_decode_for_cache(full, cpu_decode_above_mb)
        gimg: GPUImage | None = None
        image_arr: np.ndarray | None = None

        try:
            if use_cpu:
                from ..phase_b_tiling.tile_cutter import open_image_rgb

                w, h = _jpeg_dimensions(full)
                est_mb = _jpeg_rgb_mb(w, h)
                _progress_print(
                    f"[Gate embed]   imagen grande ({est_mb:.0f} MB) -> decode CPU + infer GPU batch"
                )
                image_arr = open_image_rgb(full)
                t_decode = time.perf_counter() - t_img
                ram_mb = image_arr.nbytes / (1024 * 1024)
                img_batch = batch_size
                _progress_print(
                    f"[Gate embed]   decode OK {w}x{h} "
                    f"({ram_mb:.0f} MB RAM, {t_decode:.1f}s) -> GPU inferencia batch={img_batch}..."
                )
            else:
                gimg = decode_jpeg_gpu(full, device=device)
                t_decode = time.perf_counter() - t_img
                img_batch = _batch_size_for_image(
                    batch_size,
                    gimg,
                    dynamic=dynamic_batch,
                    vram_budget_mb=vram_budget_mb,
                    model_reserve_mb=MODEL_RESERVE_MB,
                )
                _progress_print(
                    f"[Gate embed]   decode OK {gimg.width}x{gimg.height} "
                    f"({gimg.vram_mb:.0f} MB VRAM, {t_decode:.1f}s) -> GPU inferencia batch={img_batch}..."
                )
        except Exception as e:
            raise RuntimeError(f"[Gate embed] fallo en {image_rel}: {e}") from e

        rows = grp["row"].to_numpy(dtype=np.int32)
        cols = grp["col"].to_numpy(dtype=np.int32)
        tile_sizes = grp["tile_size"].to_numpy(dtype=np.int32)
        uniform_batches = list(
            iter_uniform_tile_rowcol_batches(rows, cols, tile_sizes, batch_size=img_batch)
        )
        n_batches = max(1, len(uniform_batches))

        for batch_num, (batch_indices, rowcols) in enumerate(uniform_batches, start=1):
            bs_try = len(rowcols)
            chunk_indices = list(batch_indices)
            while True:
                tiles = views = d0 = sal = feat = None
                try:
                    sub_rowcols = rowcols[:bs_try]
                    sub_indices = chunk_indices[:bs_try]
                    if image_arr is not None:
                        tiles = _batch_tiles_cpu_to_gpu(image_arr, sub_rowcols, device)
                    else:
                        assert gimg is not None
                        tiles = batch_tiles_gpu(gimg, sub_rowcols)
                    label_slice = grp["stage1"].to_numpy()[sub_indices]
                    label_idx = torch.from_numpy(encode_gate_indices(label_slice)).to(device)
                    out = _forward_embed_batch(
                        backbone,
                        tiles,
                        dino_input_size=dino_input_size,
                        seg_target_size=seg_target_size,
                        attention_cfg=attention_cfg,
                        labels=label_idx,
                        u2net=u2net,
                        pooling_mode=pooling_mode,
                    )
                    if attention_cfg is None:
                        feat_np = out
                    else:
                        feat_np, attn_np = out
                    b = int(feat_np.shape[0])
                    if embed_arr is not None:
                        embed_arr[write_idx : write_idx + b] = feat_np
                    if attn_arr is not None and attention_cfg is not None:
                        attn_arr[write_idx : write_idx + b] = _flatten_attn_batch(attn_np)
                    write_idx += b
                    if do_aug and str(image_rel) in aug_train and tiles is not None:
                        aug_write_idx = _write_mplus_aug_embeddings(
                            tiles=tiles,
                            grp=grp,
                            start=sub_indices[0],
                            stop=sub_indices[0] + b,
                            backbone=backbone,
                            embed_arr=embed_arr,
                            label_arr=label_arr,
                            aug_write_idx=aug_write_idx,
                            aug_rows=aug_rows,
                            image_rel=str(image_rel),
                            mplus_aug_variants=mplus_aug_variants,
                            dino_input_size=dino_input_size,
                            seg_target_size=seg_target_size,
                            attention_cfg=attention_cfg,
                            attn_arr=attn_arr,
                            u2net=u2net,
                            pooling_mode=pooling_mode,
                        )
                    del feat_np
                    if bs_try < len(rowcols):
                        rowcols = rowcols[bs_try:]
                        chunk_indices = chunk_indices[bs_try:]
                        bs_try = len(rowcols)
                        continue
                    break
                except torch.cuda.OutOfMemoryError:
                    release_cuda_memory(gc_collect=False)
                    if bs_try <= 1:
                        raise
                    bs_try = max(1, bs_try // 2)
                    _progress_print(
                        f"[Gate embed]   OOM -> reintento batch {batch_num} con bs={bs_try}"
                    )
                finally:
                    if tiles is not None:
                        del tiles, views, d0, sal, feat
                    maybe_empty_cache(batch_num, every_n=empty_cache_every_n_batches)
                    maybe_gc_collect(batch_num, every_n=gc_collect_every_n_batches)

            now = time.perf_counter()
            tiles_done = write_idx + (aug_write_idx - n)
            if batch_num == 1 or batch_num == n_batches or (now - last_progress_t) >= 5.0:
                pct = 100.0 * tiles_done / max(n_total, 1)
                elapsed = now - t0
                tiles_per_s = tiles_done / max(elapsed, 0.001)
                eta_s = (n_total - tiles_done) / max(tiles_per_s, 0.001)
                mode = "CPU-decode+GPU" if image_arr is not None else "GPU"
                _progress_print(
                    f"[Gate embed]   {img_name} batch {batch_num}/{n_batches} [{mode}] | "
                    f"tiles {tiles_done:,}/{n_total:,} ({pct:.1f}%) | "
                    f"{tiles_per_s:.0f} tiles/s | ETA ~{_format_duration(eta_s)}"
                )
                _write_build_progress(
                    progress_path,
                    {
                        "status": "running",
                        "images_total": n_images,
                        "images_done": img_idx - 1,
                        "tiles_total": n_total,
                        "tiles_done": tiles_done,
                        "pct": round(pct, 2),
                        "current_image": image_rel,
                        "current_batch": batch_num,
                        "batches_in_image": n_batches,
                        "decode_mode": mode.lower(),
                        "tiles_per_second": round(tiles_per_s, 1),
                        "eta_seconds": round(eta_s, 1),
                        "eta": _format_duration(eta_s),
                    },
                )
                last_progress_t = now

        if gimg is not None:
            del gimg
        if image_arr is not None:
            del image_arr
        release_cuda_memory(gc_collect=True)
        if memmap_flush_every_n_images > 0 and img_idx % memmap_flush_every_n_images == 0:
            if embed_arr is not None:
                embed_arr.flush()
            if attn_arr is not None:
                attn_arr.flush()

        dt_img = time.perf_counter() - t_img
        img_times.append(dt_img)
        avg_img = sum(img_times) / len(img_times)
        eta_s = avg_img * (n_images - img_idx)
        tiles_done = write_idx + (aug_write_idx - n)
        pct = 100.0 * tiles_done / max(n_total, 1)
        msg = (
            f"[Gate embed] ({img_idx}/{n_images}) {pct:5.1f}% | "
            f"tiles {tiles_done:,}/{n_total:,} | {img_name} {dt_img:.1f}s | "
            f"media {avg_img:.1f}s/img | ETA ~{_format_duration(eta_s)}"
        )
        log.info(msg)
        _progress_print(msg)
        _write_build_progress(
            progress_path,
            {
                "status": "running",
                "images_total": n_images,
                "images_done": img_idx,
                "tiles_total": n_total,
                "tiles_done": tiles_done,
                "pct": round(pct, 2),
                "last_image": image_rel,
                "last_image_seconds": round(dt_img, 2),
                "avg_seconds_per_image": round(avg_img, 2),
                "eta_seconds": round(eta_s, 1),
                "eta": _format_duration(eta_s),
            },
        )

    if write_idx != n:
        log.warning(f"[Gate embed] base escrito {write_idx} tiles, esperado {n}")
    if aug_write_idx != n_total:
        log.warning(f"[Gate embed] aug escrito {aug_write_idx - n} tiles, esperado {n_total - n}")

    if label_arr is not None:
        label_arr.flush()
    if embed_arr is not None:
        embed_arr.flush()
    if attn_arr is not None:
        attn_arr.flush()
        del attn_arr
    if embed_arr is not None:
        del embed_arr
    if label_arr is not None:
        del label_arr

    elapsed_build = time.perf_counter() - t0
    if attn_only:
        assert existing_meta is not None and attention_meta is not None
        _patch_meta_with_attention(
            cpaths,
            existing_meta,
            attention_meta=attention_meta,
            fingerprint=fingerprint,
            attn_build_seconds=elapsed_build,
        )
    else:
        finalize_embed_cache_metadata(
            df,
            backbone_name=backbone_name,
            embed_dim=embed_dim,
            dino_input_size=dino_input_size,
            seg_target_size=seg_target_size,
            tiles_index_path=tiles_index_path,
            cache_paths=cpaths,
            build_seconds=elapsed_build,
            aug_rows=aug_rows,
            mplus_aug_variants=mplus_aug_variants,
            n_tiles_total=n_total,
            attention_meta=attention_meta,
            pooling_mode=pooling_mode,
        )
    elapsed = elapsed_build
    size_parts = [cpaths.embed, cpaths.label]
    if cpaths.attn.exists():
        size_parts.append(cpaths.attn)
    size_mb = sum(p.stat().st_size for p in size_parts) / (1024**2)
    if attn_only:
        done_msg = (
            f"[Gate embed] Atencion ViT lista: {n_total:,} tiles, "
            f"+{cpaths.attn.stat().st_size / (1024**2):.1f} MB attn, {elapsed / 60:.1f} min"
        )
    else:
        done_msg = (
            f"[Gate embed] Listo: {n_total:,} tiles ({n:,} base + {n_total - n:,} aug M+), "
            f"{size_mb:.1f} MB, {elapsed / 60:.1f} min"
        )
    log.info(done_msg)
    _progress_print(done_msg)
    return cpaths


class GateEmbedStore:
    """Lectura por lotes desde memmap de embeddings."""

    def __init__(self, paths: EmbedCachePaths):
        with open(paths.meta, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embed_dim = int(self.meta["embed_dim"])
        self.embed = np.memmap(
            paths.embed, dtype=np.float16, mode="r", shape=(self.meta["n_tiles"], self.embed_dim)
        )
        self.labels = np.memmap(paths.label, dtype=np.int8, mode="r", shape=(self.meta["n_tiles"],))
        self.lookup = pd.read_parquet(paths.lookup)
        attn_meta = self.meta.get("attention") or {}
        self.attn_flat_dim = int(attn_meta.get("flat_dim", 0))
        self.attn_shape = tuple(attn_meta.get("tensor_shape_per_tile") or ())
        self.attn = None
        if self.attn_flat_dim > 0 and paths.attn.exists():
            self.attn = np.memmap(
                paths.attn,
                dtype=np.float16,
                mode="r",
                shape=(self.meta["n_tiles"], self.attn_flat_dim),
            )
        if "aug_id" in self.lookup.columns:
            key_cols = ["image_path", "row", "col", "aug_id"]
            if "tile_size" in self.lookup.columns and self.lookup["tile_size"].notna().any():
                key_cols = ["image_path", "row", "col", "tile_size", "aug_id"]
            self._lookup_key = self.lookup.set_index(key_cols)["embed_idx"]
            self._lookup_key_cols = key_cols
        else:
            self._lookup_key = self.lookup.set_index(["image_path", "row", "col"])["embed_idx"]
            self._lookup_key_cols = ["image_path", "row", "col"]

    @property
    def n_tiles(self) -> int:
        return int(self.meta["n_tiles"])

    def mplus_aug_variant_names(self) -> tuple[str, ...]:
        raw = self.meta.get("mplus_aug_variants") or []
        return tuple(str(v) for v in raw)

    def expand_train_df_mplus_augs(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Añade filas M+ augmentadas (solo train) para planificar épocas con diversidad."""
        if "aug_id" not in self.lookup.columns:
            out = train_df.copy()
            if "aug_id" not in out.columns:
                out["aug_id"] = 0
            return out
        aug_lookup = self.lookup[self.lookup["aug_id"] > 0]
        if aug_lookup.empty:
            out = train_df.copy()
            out["aug_id"] = 0
            return out
        base = train_df.copy()
        base["aug_id"] = 0
        key_cols = ["image_path", "row", "col"]
        base_keys = {
            (str(r["image_path"]), int(r["row"]), int(r["col"]))
            for _, r in base[key_cols].iterrows()
        }
        aug_rows: list[dict] = []
        for _, lk in aug_lookup.iterrows():
            key = (str(lk["image_path"]), int(lk["row"]), int(lk["col"]))
            if key not in base_keys:
                continue
            src = base[
                (base["image_path"].astype(str) == key[0])
                & (base["row"].astype(int) == key[1])
                & (base["col"].astype(int) == key[2])
            ].iloc[0]
            row = src.to_dict()
            row["aug_id"] = int(lk["aug_id"])
            aug_rows.append(row)
        if not aug_rows:
            return base
        return pd.concat([base, pd.DataFrame(aug_rows)], ignore_index=True)

    def indices_for_sub(self, sub: pd.DataFrame) -> np.ndarray:
        out: list[int] = []
        cols = getattr(self, "_lookup_key_cols", ["image_path", "row", "col"])
        for i in range(len(sub)):
            parts: list = [str(sub.iloc[i]["image_path"]), int(sub.iloc[i]["row"]), int(sub.iloc[i]["col"])]
            if "tile_size" in cols:
                parts.append(int(sub.iloc[i].get("tile_size", 252)))
            if "aug_id" in cols:
                parts.append(int(sub.iloc[i]["aug_id"]) if "aug_id" in sub.columns else 0)
            out.append(int(self._lookup_key[tuple(parts)]))
        return np.array(out, dtype=np.int64)

    def read_batch(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.from_numpy(self.embed[indices].astype(np.float32)).to(
            device, non_blocking=True
        )
        lbl = torch.from_numpy(self.labels[indices].astype(np.int64)).to(
            device, non_blocking=True
        )
        return feat, lbl

    def has_attention(self) -> bool:
        return self.attn is not None and self.attn_flat_dim > 0

    def read_attn_batch(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.attn is None:
            return None
        flat = torch.from_numpy(self.attn[indices].astype(np.float32)).to(
            device, non_blocking=True
        )
        if not self.attn_shape:
            return flat
        return flat.view(len(indices), *self.attn_shape)


def open_gate_embed_store(paths: EmbedCachePaths) -> GateEmbedStore:
    """Abre memmap existente (no compila)."""
    return GateEmbedStore(paths)


def ensure_gate_embed_cache(
    tiles_df: pd.DataFrame,
    backbone: nn.Module,
    device: torch.device,
    *,
    backbone_name: str = "dinov2_vits14",
    embed_dim: int = DEFAULT_EMBED_DIM,
    tiles_index_path: Optional[Path] = None,
    force_rebuild: bool = False,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
    cache_attention: bool = False,
    attention_layers: str = "all",
    attention_head_reduce: str = "mean",
    cache_paths: Optional[EmbedCachePaths] = None,
    pooling_mode: str = "none",
) -> GateEmbedStore:
    paths = build_gate_embed_cache(
        tiles_df,
        backbone,
        device=device,
        backbone_name=backbone_name,
        embed_dim=embed_dim,
        tiles_index_path=tiles_index_path,
        force_rebuild=force_rebuild,
        mplus_aug_variants=mplus_aug_variants,
        mplus_aug_train_images=mplus_aug_train_images,
        cache_attention=cache_attention,
        attention_layers=attention_layers,
        attention_head_reduce=attention_head_reduce,
        cache_paths=cache_paths,
        pooling_mode=pooling_mode,
    )
    return GateEmbedStore(paths)


def iter_embed_gate_batches(
    plan: EpochPlan,
    store: GateEmbedStore,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = torch.device("cuda"),
    max_batches: Optional[int] = None,
) -> Iterator[GPUImageBatch]:
    """Itera batches desde memmap de embeddings (sin JPEG/U2Net/DINO)."""
    seen = 0
    batch_sources = plan.stratified_batches if plan.stratified_batches is not None else None

    if batch_sources is not None:
        for batch_idx, sub in enumerate(batch_sources):
            rows = sub["row"].to_numpy(dtype=np.int32)
            cols = sub["col"].to_numpy(dtype=np.int32)
            try:
                all_idx = store.indices_for_sub(sub)
            except KeyError as e:
                log.warning(f"[Gate embed] tile missing in stratified batch {batch_idx}: {e}")
                continue
            feat, labels = store.read_batch(all_idx, device)
            vit_attn = store.read_attn_batch(all_idx, device)
            image_rel = str(sub["image_path"].iloc[0]) if len(sub) else "__stratified__"
            domains = (
                sub["domain_bucket"].astype(str).tolist()
                if "domain_bucket" in sub.columns
                else None
            )
            edges = (
                sub["tile_edge"].astype(int).tolist()
                if "tile_edge" in sub.columns
                else (
                    sub["tile_size"].astype(int).tolist()
                    if "tile_size" in sub.columns
                    else None
                )
            )
            yield GPUImageBatch(
                rgb=feat,
                seg=feat,
                freq=feat,
                labels=labels,
                label_mode="gate",
                rows=torch.from_numpy(rows).to(device, non_blocking=True),
                cols=torch.from_numpy(cols).to(device, non_blocking=True),
                image_path=image_rel,
                features=feat,
                vit_attention=vit_attn,
                domain_buckets=domains,
                tile_edges=edges,
            )
            seen += 1
            if max_batches and seen >= max_batches:
                return
        return

    for image_rel, sub in plan.items:
        rows = sub["row"].to_numpy(dtype=np.int32)
        cols = sub["col"].to_numpy(dtype=np.int32)
        n = len(sub)
        try:
            all_idx = store.indices_for_sub(sub)
        except KeyError as e:
            log.warning(f"[Gate embed] tile missing in lookup {image_rel}: {e}")
            continue

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            idx = all_idx[start:stop]
            feat, labels = store.read_batch(idx, device)
            vit_attn = store.read_attn_batch(idx, device)
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
            yield GPUImageBatch(
                rgb=feat,
                seg=feat,
                freq=feat,
                labels=labels,
                label_mode="gate",
                rows=torch.from_numpy(rows[start:stop]).to(device, non_blocking=True),
                cols=torch.from_numpy(cols[start:stop]).to(device, non_blocking=True),
                image_path=image_rel,
                features=feat,
                vit_attention=vit_attn,
                domain_buckets=domains,
                tile_edges=edges,
            )
            seen += 1
            if max_batches and seen >= max_batches:
                return
