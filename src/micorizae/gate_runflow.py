"""Pipeline interactivo gate AM: cache embeddings + entrenamiento.

Invocado por `python run.py` sin argumentos. Parametros en `config.py`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .common.logging_utils import get_logger

log = get_logger("gate_runflow")

CacheAction = Literal["use", "build", "build_attn", "abort", "finalize"]


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _interactive(cfg: Any) -> bool:
    return bool(_cfg(cfg, "INTERACTIVE_PROMPTS", True)) and sys.stdin.isatty()


def _ask_yes_no(prompt: str, *, default: bool, cfg: Any) -> bool:
    if not _interactive(cfg):
        return default
    hint = "S/n" if default else "s/N"
    while True:
        try:
            raw = input(f"{prompt} [{hint}]: ").strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        if raw in {"s", "si", "y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Responde s (si) o n (no).", flush=True)


def resolve_cache_action(status, cfg: Any) -> CacheAction:
    """Decide usar, recompilar o abortar segun estado de cache y config."""
    if status.state == "finalize":
        print(f"\n{status.message}", flush=True)
        return "finalize"

    if status.state == "valid":
        if _interactive(cfg):
            print(f"\n{status.message}", flush=True)
            if _ask_yes_no("Usar cache existente", default=True, cfg=cfg):
                return "use"
            if _ask_yes_no("Sobreescribir y recompilar cache", default=False, cfg=cfg):
                return "build"
            return "abort"
        if _cfg(cfg, "AUTO_REBUILD_CACHE", False):
            return "build"
        if _cfg(cfg, "AUTO_USE_VALID_CACHE", True):
            return "use"
        return "abort"

    if status.state in {"missing", "partial", "stale", "obsolete"}:
        print(f"\n{status.message}", flush=True)
        if _interactive(cfg):
            if _ask_yes_no("Compilar cache de embeddings", default=True, cfg=cfg):
                return "build"
            return "abort"
        if _cfg(cfg, "AUTO_BUILD_CACHE_IF_MISSING", True):
            return "build"
        return "abort"

    if status.state == "attn_missing":
        print(f"\n{status.message}", flush=True)
        if _interactive(cfg):
            if _ask_yes_no("Compilar mapas de atencion ViT (sin recomputar embeddings)", default=True, cfg=cfg):
                return "build_attn"
            return "abort"
        if _cfg(cfg, "AUTO_BUILD_CACHE_IF_MISSING", True):
            return "build_attn"
        return "abort"

    return "abort"


def _mplus_aug_variants_from_cfg(cfg: Any) -> tuple[str, ...]:
    if not bool(_cfg(cfg, "GATE_CACHE_MPLUS_AUGMENT", True)):
        return ()
    from .phase_d_stage1.gate_tile_augment import parse_aug_variants

    return parse_aug_variants(str(_cfg(cfg, "GATE_CACHE_MPLUS_AUG_VARIANTS", "hflip,vflip")))


def _use_gate_v4(cfg: Any) -> bool:
    return bool(_cfg(cfg, "GATE_MULTIDENSITY_ENABLED", False)) or bool(
        _cfg(cfg, "GATE_AMFINDER_TRAIN_ENABLED", False)
    )


def _cache_basename_from_cfg(cfg: Any) -> str:
    return str(_cfg(cfg, "GATE_CACHE_BASENAME", "gate_am_embeds_v3"))


def _pipeline_steps(
    *,
    step1_status: str,
    step2_status: str,
    step3_status: str,
    step1_pct: int = 0,
    step2_pct: int = 0,
    step3_pct: int = 0,
    step1_detail: str = "",
    step2_detail: str = "",
    step3_detail: str = "",
) -> list[dict[str, Any]]:
    return [
        {
            "id": "hdf5_tiles",
            "label": "Paso 1: HDF5 tiles (luma+label)",
            "status": step1_status,
            "pct": step1_pct,
            "eta": None,
            "detail": step1_detail or "Pendiente",
        },
        {
            "id": "embed_v5",
            "label": "Paso 2: Embeddings v5 (DINO bg_only)",
            "status": step2_status,
            "pct": step2_pct,
            "eta": None,
            "detail": step2_detail or "Pendiente",
        },
        {
            "id": "train_slice_ms",
            "label": "Paso 3: Entrenamiento Slice-MS",
            "status": step3_status,
            "pct": step3_pct,
            "eta": None,
            "detail": step3_detail or "Pendiente",
        },
    ]


def resolve_gate_am_splits(
    cfg: Any,
    *,
    exclude_unreadable: bool = True,
):
    """Train/val/external + tiles para cache segun config v4 o legacy."""
    import pandas as pd

    from .gate_data_splits import cache_tiles_for_gate_v4, split_gate_am_v4
    from .phase_d_stage1 import split_by_image

    if _use_gate_v4(cfg):
        train_tiers = None
        if not bool(_cfg(cfg, "GATE_MULTIDENSITY_ENABLED", False)):
            train_tiers = ("base",)
        elif str(_cfg(cfg, "GATE_MULTIDENSITY_TIERS", "")).strip():
            extra = tuple(
                t.strip()
                for t in str(_cfg(cfg, "GATE_MULTIDENSITY_TIERS", "")).split(",")
                if t.strip()
            )
            train_tiers = ("base", *extra)
        train_df, val_df, external_df, info = split_gate_am_v4(
            exclude_unreadable=exclude_unreadable,
            train_density_tiers=train_tiers,
        )
        cache_tiles = cache_tiles_for_gate_v4(train_df, val_df)
        return train_df, val_df, external_df, info, cache_tiles

    train_df, val_df, info = split_by_image(
        lineages=["AM"],
        split_mode="fixed",
        train_splits=("train",),
        val_splits=("test",),
        exclude_unreadable=exclude_unreadable,
    )
    from .gate_domain_buckets import attach_domain_buckets

    train_df = attach_domain_buckets(train_df)
    val_df = attach_domain_buckets(val_df)
    external_df = pd.DataFrame()
    cache_tiles = pd.concat([train_df, val_df], ignore_index=True).drop_duplicates(
        subset=["image_path", "row", "col"]
    )
    return train_df, val_df, external_df, info, cache_tiles


def _cache_build_kwargs(cfg: Any) -> dict:
    from .phase_d_stage1.gate_vision import dino_input_size_from_config, seg_target_size_for

    dino_in = dino_input_size_from_config(cfg)
    empty_n = int(_cfg(cfg, "GATE_CACHE_EMPTY_CACHE_EVERY_N_BATCHES", 0))
    if empty_n == 0 and bool(_cfg(cfg, "GATE_CACHE_GPU_CLEANUP_EACH_BATCH", False)):
        empty_n = 1
    return {
        "dynamic_batch": bool(_cfg(cfg, "GATE_CACHE_DYNAMIC_BATCH", True)),
        "vram_budget_mb": float(_cfg(cfg, "GATE_CACHE_VRAM_BUDGET_MB", 7500.0)),
        "cpu_decode_above_mb": float(_cfg(cfg, "GATE_CACHE_CPU_DECODE_ABOVE_MB", 500.0)),
        "empty_cache_every_n_batches": empty_n,
        "gc_collect_every_n_batches": int(_cfg(cfg, "GATE_CACHE_GC_COLLECT_EVERY_N_BATCHES", 0)),
        "memmap_flush_every_n_images": int(_cfg(cfg, "GATE_CACHE_MEMMAP_FLUSH_EVERY_N_IMAGES", 5)),
        "dino_input_size": dino_in,
        "seg_target_size": seg_target_size_for(dino_in, cfg),
        "cache_attention": bool(_cfg(cfg, "GATE_CACHE_ATTENTION", False)),
        "attention_layers": str(_cfg(cfg, "GATE_CACHE_ATTENTION_LAYERS", "all")),
        "attention_head_reduce": str(_cfg(cfg, "GATE_CACHE_ATTENTION_HEAD_REDUCE", "mean")),
    }


def _open_gate_h5_store(cfg: Any) -> Optional[object]:
    """Abre HDF5 RGB+saliencia para entrenamiento E6 (evita decode JPEG)."""
    if not bool(_cfg(cfg, "GATE_H5_CACHE_ENABLED", True)):
        return None
    from .common.paths import get_paths
    from .phase_d_stage1.gate_tile_h5_cache import GateTileH5Store, _cache_paths as h5_cache_paths

    paths = get_paths()
    h5_path, _, lookup_path = h5_cache_paths(paths.root)
    if h5_path.is_file() and lookup_path.is_file():
        return GateTileH5Store(h5_path, lookup_path)
    return None


def _gate_h5_cache_ready(cfg: Any, all_tiles) -> bool:
    """True si HDF5 + meta + lookup existen y coinciden con el manifest actual."""
    if not bool(_cfg(cfg, "GATE_H5_CACHE_ENABLED", True)):
        return False
    from .common.paths import get_paths
    from .phase_d_stage1.gate_tile_h5_cache import _cache_paths as h5_cache_paths, cache_is_valid

    paths = get_paths()
    tiles_index_path = paths.manifests / "tiles_index.csv"
    h5_path, meta_path, lookup_path = h5_cache_paths(paths.root)
    from .phase_d_stage1.gate_tile_h5_cache import _manifest_fingerprint

    fingerprint = _manifest_fingerprint(tiles_index_path)
    return cache_is_valid(
        n_tiles=len(all_tiles),
        fingerprint=fingerprint,
        h5_path=h5_path,
        meta_path=meta_path,
        lookup_path=lookup_path,
    )


def execute_ensure_gate_h5_cache(cfg: Any, all_tiles) -> Optional[object]:
    """Paso 1 SSD: compila HDF5 RGB+saliencia si falta (una vez)."""
    if not bool(_cfg(cfg, "GATE_H5_CACHE_ENABLED", True)):
        return None
    if not bool(_cfg(cfg, "GATE_H5_BUILD_BEFORE_TRAIN", True)):
        return None
    import torch

    from .common.paths import get_paths
    from .phase_d_stage1.gate_tile_h5_cache import build_gate_tile_h5_cache, load_frozen_u2net_saliency

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")

    paths = get_paths()
    device = torch.device("cuda")
    skip_u2net = bool(_cfg(cfg, "GATE_H5_SKIP_U2NET", False))
    grayscale = bool(_cfg(cfg, "GATE_H5_GRAYSCALE", True))
    store_saliency = bool(_cfg(cfg, "GATE_H5_STORE_SALIENCY", False))
    if skip_u2net:
        store_saliency = False
    u2net = None
    if not skip_u2net and store_saliency:
        weights = paths.root / "models" / "weights" / "u2netp.pth"
        u2net = load_frozen_u2net_saliency(weights, device)
    batch = int(_cfg(cfg, "GATE_H5_BATCH_SIZE", _cfg(cfg, "GATE_CACHE_BATCH_SIZE", 32)))
    empty_n = int(_cfg(cfg, "GATE_CACHE_EMPTY_CACHE_EVERY_N_BATCHES", 0))
    gc_n = int(_cfg(cfg, "GATE_CACHE_GC_COLLECT_EVERY_N_BATCHES", 0))
    compression = str(_cfg(cfg, "GATE_H5_COMPRESSION", "lzf"))
    fmt = f"luma+lzf" if grayscale else "rgb"
    if store_saliency:
        fmt += "+sal"

    print(
        f"\n[Gate SSD] Paso 1/3: HDF5 tiles ({fmt}) -> cache/gate_am_tiles_v1.h5 "
        f"({len(all_tiles):,} tiles, batch={batch}, sin U2Net)",
        flush=True,
    )
    build_gate_tile_h5_cache(
        all_tiles,
        u2net,
        device=device,
        batch_size=batch,
        empty_cache_every_n_batches=empty_n,
        gc_collect_every_n_batches=gc_n,
        cpu_decode_above_mb=float(_cfg(cfg, "GATE_CACHE_CPU_DECODE_ABOVE_MB", 300.0)),
        chunk_tiles=int(_cfg(cfg, "GATE_H5_CHUNK_TILES", 64)),
        compression=compression,
        u2net_bg_only=bool(_cfg(cfg, "GATE_H5_U2NET_BG_ONLY", True)),
        skip_u2net=skip_u2net,
        store_saliency=store_saliency,
        grayscale=grayscale,
        streaming_decode=bool(_cfg(cfg, "GATE_H5_STREAMING_DECODE", True)),
        resume_enabled=bool(_cfg(cfg, "GATE_H5_RESUME", True)),
    )
    return None


def execute_build_gate_cache(
    *,
    backbone: str,
    batch_size: int,
    force_rebuild: bool = False,
    cfg: Any = None,
    dynamic_batch: bool = True,
    vram_budget_mb: float = 7500.0,
    cpu_decode_above_mb: float = 500.0,
    empty_cache_every_n_batches: int = 0,
    gc_collect_every_n_batches: int = 0,
    memmap_flush_every_n_images: int = 5,
    dino_input_size: int = 252,
    seg_target_size: int = 360,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
    cache_attention: bool = False,
    attention_layers: str = "all",
    attention_head_reduce: str = "mean",
) -> None:
    import warnings

    import pandas as pd
    import torch

    warnings.filterwarnings("ignore", message=".*xFormers.*")
    warnings.filterwarnings("ignore", message=".*not writable.*")

    from .phase_d_stage1 import build_branch_a
    from .phase_d_stage1.gate_embed_cache import build_gate_embed_cache, inspect_embed_cache_status
    from .common.paths import get_paths

    try:
        import config as user_config  # type: ignore

        cfg = cfg or user_config
    except Exception:
        cfg = cfg or type("Cfg", (), {})()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")

    paths = get_paths()
    device = torch.device("cuda")
    train_df, val_df, _external_df, info, all_tiles = resolve_gate_am_splits(cfg)
    log.info("[Gate cache] Compilacion DINO mean-pool")
    log.info(
        f"[Gate cache] {len(all_tiles):,} tiles "
        f"({info['n_train_images']}+{info['n_val_images']} imgs)"
    )

    classifier = build_branch_a(backbone_name=backbone, num_classes=3, freeze_backbone=True)
    attn_note = f", attn={attention_layers}" if cache_attention else ""
    log.info(
        f"[Gate cache] DINOv2 {backbone}; compilando embeddings "
        f"(batch={batch_size}, dynamic_batch={dynamic_batch}, mean-pool{attn_note})..."
    )
    from .phase_d_stage1.gate_vision import describe_dino_resolution

    log.info(f"[Gate cache] DINO vision: {describe_dino_resolution(dino_input_size)}")
    train_images = set(train_df["image_path"].astype(str))
    cache_bn = _cache_basename_from_cfg(cfg)
    pooling_mode = str(_cfg(cfg, "GATE_EMBED_POOLING_MODE", "none"))
    status = inspect_embed_cache_status(
        all_tiles,
        backbone_name=backbone,
        dino_input_size=dino_input_size,
        cache_basename=cache_bn,
        mplus_aug_variants=mplus_aug_variants,
        mplus_aug_train_images=train_images if mplus_aug_variants else None,
        cache_attention=cache_attention,
        attention_layers=attention_layers,
        attention_head_reduce=attention_head_reduce,
        pooling_mode=pooling_mode,
    )
    if status.state == "attn_missing" and cache_attention and not force_rebuild:
        log.info("[Gate cache] Embeddings OK; compilando solo mapas de atencion ViT...")
        execute_build_gate_attention_cache(
            backbone=backbone,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
            dynamic_batch=dynamic_batch,
            vram_budget_mb=vram_budget_mb,
            cpu_decode_above_mb=cpu_decode_above_mb,
            empty_cache_every_n_batches=empty_cache_every_n_batches,
            gc_collect_every_n_batches=gc_collect_every_n_batches,
            memmap_flush_every_n_images=memmap_flush_every_n_images,
            dino_input_size=dino_input_size,
            seg_target_size=seg_target_size,
            mplus_aug_variants=mplus_aug_variants,
            mplus_aug_train_images=train_images if mplus_aug_variants else None,
            attention_layers=attention_layers,
            attention_head_reduce=attention_head_reduce,
        )
        return

    from .phase_d_stage1.gate_embed_cache import cache_paths_with_basename

    h5_store = None
    if bool(_cfg(cfg, "GATE_EMBED_BUILD_FROM_H5", True)) and bool(
        _cfg(cfg, "GATE_H5_CACHE_ENABLED", True)
    ):
        from .phase_d_stage1.gate_tile_h5_cache import GateTileH5Store, _cache_paths as h5_cache_paths

        h5_path, _meta, lookup_path = h5_cache_paths(paths.root)
        if h5_path.is_file() and lookup_path.is_file():
            h5_store = GateTileH5Store(h5_path, lookup_path)
            print(
                "[Gate SSD] Paso 2/3: embeddings DINO desde HDF5 (sin decodificar JPEG)",
                flush=True,
            )
        else:
            log.warning("[Gate SSD] HDF5 no encontrado; embeddings desde JPEG")

    try:
        cpaths = build_gate_embed_cache(
            all_tiles,
            classifier.backbone,
            device=device,
            backbone_name=backbone,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
            cache_paths=cache_paths_with_basename(paths.root, cache_bn),
            dynamic_batch=dynamic_batch,
            vram_budget_mb=vram_budget_mb,
            cpu_decode_above_mb=cpu_decode_above_mb,
            empty_cache_every_n_batches=empty_cache_every_n_batches,
            gc_collect_every_n_batches=gc_collect_every_n_batches,
            memmap_flush_every_n_images=memmap_flush_every_n_images,
            dino_input_size=dino_input_size,
            seg_target_size=seg_target_size,
            mplus_aug_variants=mplus_aug_variants,
            mplus_aug_train_images=train_images if mplus_aug_variants else None,
            cache_attention=cache_attention,
            attention_layers=attention_layers,
            attention_head_reduce=attention_head_reduce,
            pooling_mode=pooling_mode,
            h5_store=h5_store,
        )
    finally:
        if h5_store is not None:
            h5_store.close()
    log.info(f"[bold green]Cache embeddings listo[/bold green] -> {cpaths.embed}")


def execute_build_gate_attention_cache(
    *,
    backbone: str,
    batch_size: int,
    force_rebuild: bool = False,
    dynamic_batch: bool = True,
    vram_budget_mb: float = 7500.0,
    cpu_decode_above_mb: float = 500.0,
    empty_cache_every_n_batches: int = 0,
    gc_collect_every_n_batches: int = 0,
    memmap_flush_every_n_images: int = 5,
    dino_input_size: int = 252,
    seg_target_size: int = 360,
    mplus_aug_variants: tuple[str, ...] = (),
    mplus_aug_train_images: Optional[set[str]] = None,
    attention_layers: str = "all",
    attention_head_reduce: str = "mean",
) -> None:
    """Compila solo mapas ViT CLS->patch sobre cache de embeddings existente."""
    import warnings

    import pandas as pd
    import torch

    warnings.filterwarnings("ignore", message=".*xFormers.*")
    warnings.filterwarnings("ignore", message=".*not writable.*")

    from .phase_d_stage1 import build_branch_a, split_by_image
    from .phase_d_stage1.gate_embed_cache import build_gate_attention_cache_only
    from .common.paths import get_paths

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")

    paths = get_paths()
    device = torch.device("cuda")
    train_df, val_df, info = split_by_image(
        lineages=["AM"], split_mode="fixed", train_splits=("train",), val_splits=("test",)
    )
    all_tiles = pd.concat([train_df, val_df], ignore_index=True).drop_duplicates(
        subset=["image_path", "row", "col"]
    )
    log.info("[Gate attn] Atencion ViT sobre cache DINO mean-pool")
    log.info(
        f"[Gate attn] {len(all_tiles):,} tiles "
        f"({info['n_train_images']}+{info['n_val_images']} imgs), capas={attention_layers}"
    )

    classifier = build_branch_a(backbone_name=backbone, num_classes=3, freeze_backbone=True)
    train_images = set(train_df["image_path"].astype(str))
    cpaths = build_gate_attention_cache_only(
        all_tiles,
        classifier.backbone,
        device=device,
        backbone_name=backbone,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
        dynamic_batch=dynamic_batch,
        vram_budget_mb=vram_budget_mb,
        cpu_decode_above_mb=cpu_decode_above_mb,
        empty_cache_every_n_batches=empty_cache_every_n_batches,
        gc_collect_every_n_batches=gc_collect_every_n_batches,
        memmap_flush_every_n_images=memmap_flush_every_n_images,
        dino_input_size=dino_input_size,
        seg_target_size=seg_target_size,
        mplus_aug_variants=mplus_aug_variants,
        mplus_aug_train_images=train_images,
        attention_layers=attention_layers,
        attention_head_reduce=attention_head_reduce,
    )
    log.info(f"[bold green]Cache atencion ViT listo[/bold green] -> {cpaths.attn}")


@dataclass(frozen=True)
class GateTrainParams:
    backbone: str
    epochs: int
    batch_size: int
    probe: bool
    full_dataset: bool
    max_bg_per_image: int
    max_train_batches: Optional[int]
    max_val_batches: Optional[int]
    max_vis_images: Optional[int]
    vis_all_test: bool
    render_maps: bool
    vis_downscale: int
    spatial_audit: bool
    spatial_audit_sample: int
    skip_pretrain_viz: bool
    report_from_cache: bool
    save_live_snapshots: bool
    dino_input_size: int
    seg_target_size: int
    mplus_aug_variants: tuple[str, ...] = ()
    pretrain_tile_samples: int = 12
    cache_attention: bool = False
    attention_layers: str = "all"
    attention_head_reduce: str = "mean"
    use_probe_attention: bool = True
    protocol: Any = None  # GateTrainProtocol
    gate4: Any = None  # Gate4SliceMSConfig
    formal_train: bool = True
    finetune_mode: str = "none"
    pooling_mode: str = "none"


def gate_train_params_from_config(cfg: Any) -> GateTrainParams:
    from dataclasses import replace

    from .phase_d_stage1.gate_training_protocol import protocol_from_config

    from .phase_d_stage1.gate_vision import dino_input_size_from_config, seg_target_size_for

    from .phase_d_stage1.gate4.config import gate4_config_from_module

    probe = bool(_cfg(cfg, "GATE_PROBE", True))
    full = bool(_cfg(cfg, "GATE_FULL_DATASET", True))
    dino_in = dino_input_size_from_config(cfg)
    protocol = protocol_from_config(cfg)
    if full:
        max_train = None
        max_val = None
        max_bg = protocol.max_bg_per_image
    else:
        max_train = int(_cfg(cfg, "GATE_FAST_MAX_TRAIN_BATCHES", 25))
        max_val = int(_cfg(cfg, "GATE_FAST_MAX_VAL_BATCHES", 12))
        max_bg = int(_cfg(cfg, "GATE_FAST_MAX_BG_PER_IMAGE", 15))
        protocol = replace(protocol, max_bg_per_image=max_bg)

    return GateTrainParams(
        backbone=str(_cfg(cfg, "GATE_BACKBONE", "dinov2_vits14")),
        epochs=protocol.max_epochs,
        batch_size=int(_cfg(cfg, "GATE_TRAIN_BATCH_SIZE", 64)),
        probe=probe,
        full_dataset=full,
        max_bg_per_image=max_bg,
        max_train_batches=max_train,
        max_val_batches=max_val,
        max_vis_images=_cfg(cfg, "GATE_MAX_VIS_IMAGES", None),
        vis_all_test=bool(_cfg(cfg, "GATE_VIS_ALL_TEST", True)),
        render_maps=bool(_cfg(cfg, "GATE_RENDER_MAPS", True)),
        vis_downscale=int(_cfg(cfg, "GATE_VIS_DOWNSCALE", 4)),
        spatial_audit=bool(_cfg(cfg, "GATE_SPATIAL_AUDIT", True)),
        spatial_audit_sample=int(_cfg(cfg, "GATE_SPATIAL_AUDIT_SAMPLE", 5)),
        skip_pretrain_viz=bool(_cfg(cfg, "GATE_SKIP_PRETRAIN_VIZ", False)),
        report_from_cache=bool(_cfg(cfg, "GATE_REPORT_FROM_CACHE", True)),
        save_live_snapshots=bool(_cfg(cfg, "GATE_SAVE_LIVE_SNAPSHOTS", True)),
        dino_input_size=dino_in,
        seg_target_size=seg_target_size_for(dino_in, cfg),
        mplus_aug_variants=_mplus_aug_variants_from_cfg(cfg),
        pretrain_tile_samples=int(_cfg(cfg, "GATE_PRETRAIN_TILE_SAMPLES", 12)),
        cache_attention=bool(_cfg(cfg, "GATE_CACHE_ATTENTION", False)),
        attention_layers=str(_cfg(cfg, "GATE_CACHE_ATTENTION_LAYERS", "all")),
        attention_head_reduce=str(_cfg(cfg, "GATE_CACHE_ATTENTION_HEAD_REDUCE", "mean")),
        use_probe_attention=bool(_cfg(cfg, "GATE_PROBE_USE_ATTENTION", True)),
        protocol=protocol,
        gate4=gate4_config_from_module(cfg),
        formal_train=bool(_cfg(cfg, "GATE_FORMAL_TRAIN", True)),
        finetune_mode=str(_cfg(cfg, "GATE_FINETUNE_MODE", "none")),
        pooling_mode=str(_cfg(cfg, "GATE_EMBED_POOLING_MODE", "none")),
    )


def _probe_in_dim(params: GateTrainParams, cache_paths: Any = None) -> int:
    """in_dim del Slice probe: embed DINO (+ mean-pool attn por capa si aplica)."""
    from .phase_d_stage1.gate_embed_cache import DEFAULT_EMBED_DIM
    from .phase_d_stage1.gate_probe_input import probe_in_dim_from_attention_meta

    if not params.use_probe_attention or not params.cache_attention:
        return DEFAULT_EMBED_DIM
    if cache_paths is not None and cache_paths.meta.exists():
        import json

        with open(cache_paths.meta, encoding="utf-8") as f:
            meta = json.load(f)
        return probe_in_dim_from_attention_meta(meta.get("attention"), DEFAULT_EMBED_DIM)
    from .phase_d_stage1.gate_dino_attention import parse_attention_layers
    from .phase_d_stage1.gate_embed_cache import _dino_num_blocks

    n_layers = len(
        parse_attention_layers(params.attention_layers, _dino_num_blocks(params.backbone))
    )
    return DEFAULT_EMBED_DIM + n_layers


def _write_gate_run_meta(ckpt_dir: Path, run_id: str) -> None:
    import json
    from datetime import datetime

    payload = {
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (ckpt_dir / "run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def history_dict_from_artifacts(ckpt_dir: Path) -> dict:
    """Reconstruye history de entrenamiento desde live_metrics + checkpoint."""
    import json

    from .phase_d_stage1.gate_tile_dino import CHECKPOINT_NAME

    live_path = ckpt_dir / "live_metrics.json"
    if not live_path.exists():
        raise FileNotFoundError(f"Falta {live_path}. No hay metricas de entrenamiento para recuperar.")

    live = json.loads(live_path.read_text(encoding="utf-8"))
    progress_path = ckpt_dir / "training_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    )

    history = {
        "epochs": live.get("epochs_done", []),
        "train_loss": live.get("train_loss", []),
        "val_loss": [],
        "val_auroc": live.get("val_macro_f1", []),
        "val_acc": live.get("val_acc", []),
        "val_f1": live.get("val_macro_f1", []),
        "elapsed_s": [],
        "best_epoch": live.get("best_epoch", progress.get("best_epoch", -1)),
        "best_val_auroc": live.get("best_score", progress.get("best_checkpoint_score", -1.0)),
        "epoch_details": live.get("epoch_details", progress.get("epoch_details", [])),
        "pretrain_baseline": live.get("pretrain_baseline"),
        "checkpoint_metric": live.get(
            "checkpoint_metric", progress.get("checkpoint_metric", "macro_f1")
        ),
        "calibration": None,
        "calibrated_val_metrics": None,
    }

    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    if ckpt_path.exists():
        import torch

        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        history["calibration"] = st.get("calibration")
        history["calibrated_val_metrics"] = st.get("calibrated_val_metrics")
    return history


def find_incomplete_gate_run(outputs_root: Path | None = None) -> Optional[Any]:
    """Ultima corrida gate_am_train sin reporte final publicado."""
    from .common.paths import get_paths
    from .common.run_outputs import RunOutputs

    base = outputs_root or get_paths().outputs
    if not base.is_dir():
        return None
    for run_dir in sorted(base.glob("*_gate_am_train"), reverse=True):
        if (run_dir / "training_report_test.md").exists():
            continue
        if (run_dir / "reports" / "gate_tile_dino_report.json").exists():
            continue
        return RunOutputs(run_id=run_dir.name, root=run_dir).ensure()
    return None


def _resolve_gate_run_id(ckpt_dir: Path, run_id: str | None) -> str:
    if run_id:
        return run_id
    meta_path = ckpt_dir / "run_meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("run_id"):
            return str(meta["run_id"])
    incomplete = find_incomplete_gate_run()
    if incomplete is not None:
        return incomplete.run_id
    raise ValueError(
        "No se encontro run_id. Pasa --run-id outputs/YYYYMMDD_HHMMSS_gate_am_train "
        "o asegurate de que exista run_meta.json en models/checkpoints/gate_am/."
    )


def _build_gate_train_config(params: GateTrainParams, *, mode_label: str, embed_store: Any) -> dict:
    return {
        "mode": mode_label,
        "epochs_max": params.epochs,
        "batch_size": params.batch_size,
        "max_bg_per_image": params.max_bg_per_image,
        "max_train_batches": params.max_train_batches,
        "max_val_batches": params.max_val_batches,
        "eval_every_s": None,
        "embed_cache": embed_store is not None,
        "h5_cache": False,
        "freeze_backbone": params.probe,
        "backbone": params.backbone,
        "pipeline": "dinov2_embed_cache+slice_ms",
        "dino_input_size": params.dino_input_size,
        "seg_target_size": params.seg_target_size,
        "balance_mode": params.protocol.balance_mode,
        "eval_balance_mode": params.protocol.eval_balance_mode,
        "checkpoint_eval": params.protocol.checkpoint_eval,
        "eval_stratified_samples_per_class": params.protocol.eval_stratified_samples_per_class,
        "pretrain_tile_samples": params.pretrain_tile_samples,
        "checkpoint_metric": params.protocol.checkpoint_metric,
        "loss_type": params.protocol.loss_type,
        "focal_gamma": params.protocol.focal_gamma,
        "calibrate_post_train": params.protocol.calibrate_post_train,
        "early_stop_patience": params.protocol.early_stop_patience,
        "min_epochs": params.protocol.min_epochs,
        "use_class_weights_train": params.protocol.use_class_weights_train,
        "eval_unweighted_loss": params.protocol.eval_unweighted_loss,
        "recall_thresh": params.protocol.recall_thresh,
        "spec_thresh": params.protocol.spec_thresh,
        "gate4": {
            "enabled": params.gate4.enabled if params.gate4 else False,
            "num_slices": params.gate4.num_slices if params.gate4 else None,
            "embed_dim": params.gate4.embed_dim if params.gate4 else None,
            "loss_weight": params.gate4.loss_weight if params.gate4 else None,
            "proto_subcenters": params.gate4.proto_subcenters_max if params.gate4 else None,
            "proto_subcenters_per_class": (
                list(params.gate4.proto_subcenters_per_class) if params.gate4 else None
            ),
            "domain_aware_subcenters": (
                params.gate4.domain_aware_subcenters if params.gate4 else False
            ),
            "train_domain_stratified": params.protocol.train_domain_stratified,
        },
    }


def _run_post_train_analysis(
    *,
    run: Any,
    params: GateTrainParams,
    ckpt_dir: Path,
    train_df: Any,
    val_df: Any,
    classifier: Any,
    embed_store: Any,
    device: Any,
) -> None:
    """Fase 2+6: audit, embeddings, viz post-train (automático al finalizar)."""
    from pathlib import Path

    import torch

    from .common.paths import get_paths
    from .phase_d_stage1.gate_embed_analysis import run_full_embed_analysis
    from .phase_d_stage1.gate_metric_inference import prototype_bank_from_gate4
    from .phase_d_stage1.gate_posttrain_viz import plot_tile_audit_best_ckpt
    from .phase_d_stage1.gate_probe_input import probe_in_dim_from_attention_meta
    from .phase_d_stage1.gate_run_layout import layout_for_run
    from .phase_d_stage1.gate_tile_dino import CHECKPOINT_NAME

    paths = get_paths()
    layout = layout_for_run(run)
    ckpt_path = ckpt_dir / CHECKPOINT_NAME

    try:
        import subprocess
        import sys

        audit_script = paths.root / "tools" / "gate_run_audit.py"
        if audit_script.exists():
            subprocess.run(
                [sys.executable, str(audit_script), "--run", str(run.root)],
                check=False,
                cwd=str(paths.root),
            )
    except Exception as e:
        log.warning(f"[Gate AM] audit skip: {e}")

    try:
        import config as user_config  # type: ignore

        cfg = user_config
    except Exception:
        cfg = None

    try:
        run_full_embed_analysis(
            run_dir=run.root,
            checkpoint=ckpt_path,
            cache_dir=paths.root / "cache",
            val_df=val_df,
            train_df=train_df,
            cfg=cfg,
        )
    except Exception as e:
        log.exception(f"[Gate AM] embed analysis fallo: {e}")

    try:
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        proto = None
        from .phase_d_stage1.gate_classes import GATE_CLASS_NAMES

        if "prototype_bank" in st and params.gate4 is not None:
            proto = prototype_bank_from_gate4(
                params.gate4, num_classes=len(GATE_CLASS_NAMES), device=device
            )
            proto.load_state_dict(st["prototype_bank"])
        has_subcenters = proto is not None and (
            getattr(proto, "num_subcenters", 1) > 1
            or getattr(proto, "domain_aware", False)
        )
        if has_subcenters:
            from .phase_d_stage1.gate_subcenter_audit import write_subcenter_audit

            try:
                summary = write_subcenter_audit(
                    run_dir=run.root,
                    model=classifier,
                    tiles_df=val_df,
                    embed_store=embed_store,
                    prototype_bank=proto,
                    device=device,
                    batch_size=params.batch_size,
                    split_name="holdout",
                )
                log.info(
                    f"[Gate AM] subcenter audit OK: K={summary.get('num_subcenters')} "
                    f"tiles={summary.get('n_tiles')}"
                )
            except Exception as e:
                log.exception(f"[Gate AM] subcenter audit fallo: {e}")
        in_dim = probe_in_dim_from_attention_meta(
            embed_store.meta.get("attention"), embed_store.embed_dim
        )
        plot_tile_audit_best_ckpt(
            val_df,
            model=classifier,
            embed_store=embed_store,
            prototype_bank=proto,
            use_attn=in_dim > embed_store.embed_dim,
            out_png=layout.post_training / "tile_audit_best_ckpt.png",
            device=device,
        )
    except Exception as e:
        log.exception(f"[Gate AM] post-train viz fallo: {e}")


def _finalize_gate_am_train(
    *,
    run: Any,
    params: GateTrainParams,
    history: dict,
    classifier: Any,
    embed_store: Any,
    train_df: Any,
    val_df: Any,
    info_split: dict,
    ckpt_dir: Path,
    device: Any,
    mode_label: str,
) -> Path:
    from .common.paths import get_paths
    from .phase_d_stage1.gate_tile_dino import GateTileDinoGPU
    from .phase_d_stage1.gate_train_report import finalize_gate_tile_dino_run

    paths = get_paths()
    gate = GateTileDinoGPU(classifier=classifier, device=device).to(device)

    test_images = sorted({paths.root / p for p in val_df["image_path"].unique()})
    train_config = _build_gate_train_config(params, mode_label=mode_label, embed_store=embed_store)
    return finalize_gate_tile_dino_run(
        run=run,
        history=history,
        info_split=info_split,
        gate=gate,
        train_df=train_df,
        val_df=val_df,
        val_image_paths=test_images,
        backbone=params.backbone,
        train_config=train_config,
        device=device,
        batch_size=params.batch_size,
        max_vis_images=params.max_vis_images,
        ckpt_dir=ckpt_dir,
        downscale=params.vis_downscale,
        protocol=params.protocol,
        embed_store=embed_store,
        report_from_cache=params.report_from_cache,
        vis_all_test=params.vis_all_test,
        render_maps=params.render_maps,
    )


def recover_gate_am_run_report(
    *,
    run_id: str | None = None,
    params: GateTrainParams | None = None,
) -> Path:
    """Regenera reportes/mapas para una corrida cuyo train termino sin finalize."""
    import pandas as pd
    import torch

    from .common.paths import get_paths
    from .common.run_outputs import RunOutputs
    from .phase_d_stage1.gate_classes import GATE_CLASS_NAMES
    from .phase_d_stage1.gate4.probe_model import build_gate_slice_probe
    from .phase_d_stage1 import build_branch_a, split_by_image
    from .phase_d_stage1.gate_embed_cache import inspect_embed_cache_status, open_gate_embed_store
    from .phase_d_stage1.gate_tile_dino import CHECKPOINT_NAME

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Recovery requiere GPU para eval/mapas.")

    paths = get_paths()
    ckpt_dir = paths.root / "models" / "checkpoints" / "gate_am"
    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Falta checkpoint {ckpt_path}")

    run_id = _resolve_gate_run_id(ckpt_dir, run_id)
    run = RunOutputs.open(run_id)
    log.info(f"[Gate AM] Recovery reportes -> run={run_id}")

    if params is None:
        try:
            import config as user_config  # type: ignore

            params = gate_train_params_from_config(user_config)
        except Exception:
            params = gate_train_params_from_config(type("Cfg", (), {})())

    history = history_dict_from_artifacts(ckpt_dir)
    device = torch.device("cuda")
    num_classes = len(GATE_CLASS_NAMES)
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    try:
        import config as user_config  # type: ignore

        cfg = user_config
    except Exception:
        cfg = None
    train_df, val_df, _external_df, info_split, cache_tiles = resolve_gate_am_splits(
        cfg,
        exclude_unreadable=not include_unknown,
    )
    train_images = set(train_df["image_path"].astype(str))
    status = inspect_embed_cache_status(
        cache_tiles,
        backbone_name=params.backbone,
        dino_input_size=params.dino_input_size,
        mplus_aug_variants=params.mplus_aug_variants,
        mplus_aug_train_images=train_images,
        cache_attention=params.cache_attention,
        attention_layers=params.attention_layers,
        attention_head_reduce=params.attention_head_reduce,
        cache_basename=_cache_basename_from_cfg(cfg) if cfg is not None else "gate_am_embeds_v4",
    )
    if status.state not in {"valid", "stale", "obsolete"}:
        raise RuntimeError(f"Cache embeddings no valida ({status.state}); no se puede evaluar.")
    if status.state in {"stale", "obsolete"}:
        log.warning(
            f"[Gate AM] Recovery: cache marcada {status.state}; "
            "reutilizando memmap existente para eval post-train."
        )
    embed_store = open_gate_embed_store(status.paths)

    use_slice_probe = params.probe and params.gate4 is not None and params.gate4.enabled
    if use_slice_probe:
        probe_in = _probe_in_dim(params, status.paths)

        classifier = build_gate_slice_probe(
            in_dim=probe_in,
            embed_dim=params.gate4.embed_dim,
            num_slices=params.gate4.num_slices,
            num_classes=num_classes,
        )
    else:
        classifier = build_branch_a(
            backbone_name=params.backbone, num_classes=num_classes, freeze_backbone=params.probe
        )

    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    classifier.load_state_dict(st["model_state_dict"])
    classifier.to(device)

    mode_label = (
        ("FULL" if params.full_dataset else "FAST")
        + ("_PROBE" if params.probe else "_FINETUNE")
    )

    report_md = _finalize_gate_am_train(
        run=run,
        params=params,
        history=history,
        classifier=classifier,
        embed_store=embed_store,
        train_df=train_df,
        val_df=val_df,
        info_split=info_split,
        ckpt_dir=ckpt_dir,
        device=device,
        mode_label=mode_label,
    )
    _run_post_train_analysis(
        run=run,
        params=params,
        ckpt_dir=ckpt_dir,
        train_df=train_df,
        val_df=val_df,
        classifier=classifier,
        embed_store=embed_store,
        device=device,
    )

    from .phase_d_stage1.gate_explainability import print_run_artifact_index

    explain_path = run.reports / "EXPLICABILIDAD.md"
    if explain_path.exists():
        print_run_artifact_index(run, explain_path)
    log.info(f"[Gate AM] Recovery OK -> {report_md}")
    return report_md


def run_gate_am_external_eval(
    *,
    run_id: str | None = None,
    max_images: int | None = None,
    force_rebuild: bool = False,
    skip_maps: bool = False,
    cfg: Any = None,
) -> dict[str, Any]:
    """Validacion externa AMFinder post-Gate (Stage1). Obligatoria antes de Stage2."""
    import torch

    from .common.paths import get_paths
    from .common.run_outputs import RunOutputs
    from .gate_external_eval import print_external_validation_banner, run_amfinder_external_eval
    from .phase_d_stage1.gate_tile_dino import CHECKPOINT_NAME

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible para validacion externa AMFinder.")

    try:
        import config as user_config  # type: ignore

        cfg = cfg or user_config
    except Exception:
        cfg = type("Cfg", (), {})()

    paths = get_paths()
    ckpt_dir = paths.root / "models" / "checkpoints" / "gate_am"
    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Falta checkpoint Gate: {ckpt_path}")

    run_id = _resolve_gate_run_id(ckpt_dir, run_id)
    run = RunOutputs.open(run_id)
    params = gate_train_params_from_config(cfg)
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    _train_df, _val_df, external_df, _info, _cache_tiles = resolve_gate_am_splits(
        cfg,
        exclude_unreadable=not include_unknown,
    )
    if external_df.empty:
        raise RuntimeError("Sin tiles amfinder_external en splits; no hay holdout externo.")

    max_im = int(max_images if max_images is not None else _cfg(cfg, "GATE_AMFINDER_EXTERNAL_MAX_IMAGES", 0))
    print_external_validation_banner(
        n_tiles=len(external_df),
        phase="inicio",
    )
    log.info(
        f"[Gate AM] Validacion externa AMFinder run={run_id} max_images={max_im} "
        f"force_rebuild={force_rebuild}"
    )
    return run_amfinder_external_eval(
        external_df=external_df,
        run_dir=run.root,
        ckpt_dir=ckpt_dir,
        cfg=cfg,
        batch_size=params.batch_size,
        skip_maps=skip_maps,
        max_images=max_im,
        force_rebuild=force_rebuild,
    )


def execute_train_gate_am(
    params: GateTrainParams,
    *,
    require_cache: bool = True,
    cfg: Any = None,
) -> None:
    import pandas as pd
    import torch

    from .common.paths import get_paths
    from .common.run_outputs import RunOutputs
    from .phase_d_stage1.gate_classes import GATE_CLASS_NAMES
    from .phase_d_stage1.gate4.probe_model import build_gate_slice_probe
    from .phase_d_stage1 import build_branch_a, count_parameters, split_by_image
    from .phase_d_stage1.gate_embed_cache import (
        inspect_embed_cache_status,
        open_gate_embed_store,
    )
    from .phase_d_stage1.gate_tile_dino import CHECKPOINT_NAME, train_gate_tile_dino_gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")
    if params.formal_train and params.protocol.loss_type != "slice_ms_only":
        raise RuntimeError(
            "Entrenamiento formal requiere GATE_LOSS='slice_ms_only' (Slice-MS pura, publicable)."
        )
    if not params.backbone.startswith("dinov2"):
        raise ValueError("Este pipeline requiere backbone DINOv2 (ej. dinov2_vits14).")

    try:
        import config as user_config  # type: ignore

        cfg = cfg or user_config
    except Exception:
        cfg = cfg or type("Cfg", (), {})()

    freeze_backbone = params.probe
    mode_label = (
        ("FULL" if params.full_dataset else "FAST")
        + ("_PROBE" if freeze_backbone else "_FINETUNE")
    )

    paths = get_paths()
    run = RunOutputs.create("gate_am_train")
    device = torch.device("cuda")
    num_classes = len(GATE_CLASS_NAMES)
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False

    log.info(
        f"[Gate AM] modo={mode_label} | "
        f"{'Slice-MS probe + cache embeddings' if freeze_backbone else 'DINO fine-tune on-the-fly'} | "
        f"classes={num_classes} gate4_ms={getattr(params.gate4, 'enabled', False)} | "
        f"epochs<={params.epochs} balance={params.protocol.balance_mode} "
        f"checkpoint={params.protocol.checkpoint_metric} | device={device} run={run.run_id}"
    )

    from .phase_d_stage1.gate_vision import describe_dino_resolution

    log.info(f"[Gate AM] DINO vision: {describe_dino_resolution(params.dino_input_size)}")
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    train_df, val_df, external_df, info_split, cache_tiles = resolve_gate_am_splits(
        cfg, exclude_unreadable=not include_unknown
    )
    log.info(
        f"[Gate AM] train_imgs={info_split['n_train_images']} test_imgs={info_split['n_val_images']} "
        f"train_tiles={info_split['n_train_tiles']} test_tiles={info_split['n_val_tiles']}"
    )
    log.info(f"[Gate AM] Salida incremental -> {run.root}")

    if params.spatial_audit:
        from .phase_d_stage1.gate_spatial_audit import run_gate_spatial_audit

        n_audit_tiles = len(pd.concat([train_df, val_df], ignore_index=True))
        log.info(
            f"[Gate AM] Auditoría espacial: {n_audit_tiles:,} tiles "
            f"(muestra visual={params.spatial_audit_sample} PNG)..."
        )
        audit_dir = run.reports / "spatial_audit"
        run_gate_spatial_audit(
            pd.concat([train_df, val_df], ignore_index=True),
            out_dir=audit_dir,
            sample_n=params.spatial_audit_sample,
            downscale=params.vis_downscale,
        )

    ckpt_dir = paths.root / "models" / "checkpoints" / "gate_am"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_gate_run_meta(ckpt_dir, run.run_id)

    use_slice_probe = (
        freeze_backbone
        and params.gate4 is not None
        and params.gate4.enabled
    )
    use_slice_finetune = (
        not freeze_backbone
        and params.gate4 is not None
        and params.gate4.enabled
        and params.finetune_mode not in {"", "none"}
    )
    pooling_mode = params.pooling_mode

    embed_store = None
    all_tiles = cache_tiles
    train_images = set(train_df["image_path"].astype(str))
    cache_bn = _cache_basename_from_cfg(cfg)
    cache_status = None
    if freeze_backbone:
        log.info("[Gate AM] Validando cache embeddings (memmap + metadatos)...")
        cache_status = inspect_embed_cache_status(
            all_tiles,
            backbone_name=params.backbone,
            dino_input_size=params.dino_input_size,
            cache_basename=cache_bn,
            mplus_aug_variants=params.mplus_aug_variants,
            mplus_aug_train_images=train_images,
            cache_attention=params.cache_attention,
            attention_layers=params.attention_layers,
            attention_head_reduce=params.attention_head_reduce,
            pooling_mode=pooling_mode,
        )

    if use_slice_probe:
        from .phase_d_stage1.gate_embed_cache import DEFAULT_EMBED_DIM

        probe_in = _probe_in_dim(
            params, cache_status.paths if cache_status is not None else None
        )
        classifier = build_gate_slice_probe(
            in_dim=probe_in,
            embed_dim=params.gate4.embed_dim,
            num_slices=params.gate4.num_slices,
            num_classes=num_classes,
        )
        attn_note = f"+attn({probe_in - DEFAULT_EMBED_DIM})" if probe_in > DEFAULT_EMBED_DIM else ""
        log.info(
            f"[Gate4 SliceMS] probe in={probe_in}{attn_note} embed={params.gate4.embed_dim} "
            f"slices={params.gate4.num_slices} params={count_parameters(classifier):,}"
        )
    elif use_slice_finetune:
        from .phase_d_stage1.gate4.slice_dino_model import build_gate_slice_dino
        from .phase_d_stage1.gate_dino_finetune import configure_dino_finetune
        from .phase_d_stage1.gate_embed_cache import DEFAULT_EMBED_DIM

        branch = build_branch_a(
            backbone_name=params.backbone,
            num_classes=num_classes,
            freeze_backbone=True,
        )
        ft_info = configure_dino_finetune(
            branch.backbone,
            mode=params.finetune_mode,
            last_n_blocks=int(_cfg(cfg, "GATE_FINETUNE_LAST_N_BLOCKS", 2)),
            lora_rank=int(_cfg(cfg, "GATE_LORA_RANK", 8)),
            lora_alpha=float(_cfg(cfg, "GATE_LORA_ALPHA", 16.0)),
            lora_blocks=str(_cfg(cfg, "GATE_LORA_BLOCKS", "9,10,11")),
            lora_target=str(_cfg(cfg, "GATE_LORA_TARGET", "qkv")),
        )
        classifier = build_gate_slice_dino(
            branch.backbone,
            in_dim=DEFAULT_EMBED_DIM,
            embed_dim=params.gate4.embed_dim,
            num_slices=params.gate4.num_slices,
            num_classes=num_classes,
            pooling_mode=pooling_mode,
        )
        log.info(
            f"[Gate4 Slice-DINO] E6/E6b mode={params.finetune_mode} pooling={pooling_mode} "
            f"blocks={ft_info.get('last_n_blocks')} lora={ft_info.get('lora_adapters')} "
            f"embed={params.gate4.embed_dim} slices={params.gate4.num_slices} "
            f"params={count_parameters(classifier):,}"
        )
    else:
        classifier = build_branch_a(
            backbone_name=params.backbone, num_classes=num_classes, freeze_backbone=freeze_backbone
        )
        log.info(f"[DINOv2] {params.backbone} params={count_parameters(classifier):,}")

    if freeze_backbone:
        status = cache_status
        assert status is not None
        if status.state != "valid":
            if require_cache:
                raise RuntimeError(
                    f"Cache embeddings no valida ({status.state}). "
                    "Compila primero con el paso de cache en run.py."
                )
            log.warning("[Gate AM] Cache invalida; recompilando embeddings (DINO)...")
            from .phase_d_stage1.gate_embed_cache import cache_paths_with_basename, ensure_gate_embed_cache

            if use_slice_probe:
                dino_backbone = build_branch_a(
                    backbone_name=params.backbone,
                    num_classes=num_classes,
                    freeze_backbone=True,
                ).backbone
            else:
                dino_backbone = classifier.backbone
            embed_store = ensure_gate_embed_cache(
                all_tiles,
                dino_backbone,
                device,
                backbone_name=params.backbone,
                mplus_aug_variants=params.mplus_aug_variants,
                mplus_aug_train_images=train_images,
                cache_attention=params.cache_attention,
                attention_layers=params.attention_layers,
                attention_head_reduce=params.attention_head_reduce,
                cache_paths=cache_paths_with_basename(paths.root, cache_bn),
                pooling_mode=pooling_mode,
            )
        else:
            log.info(
                f"[Gate AM] Cache embeddings OK -> {status.paths.embed} "
                f"(vectores DINO mean-pool precomputados)"
            )
            embed_store = open_gate_embed_store(status.paths)
        n_train_before = len(train_df)
        train_df = embed_store.expand_train_df_mplus_augs(train_df)
        if len(train_df) > n_train_before:
            log.info(
                f"[Gate AM] M+ aug cache: +{len(train_df) - n_train_before:,} tiles train "
                f"({len(train_df):,} total, variants={embed_store.mplus_aug_variant_names()})"
            )
        elif params.mplus_aug_variants:
            log.warning(
                "[Gate AM] M+ aug configurado pero sin filas aug en cache; "
                "recompila cache (python run.py build-gate-cache)."
            )
    else:
        log.info(
            f"[Gate AM] Fine-tune on-the-fly (Slice-DINO E6/E6b, pooling={pooling_mode})"
        )

    h5_store = None
    if not freeze_backbone:
        h5_store = _open_gate_h5_store(cfg)
        if h5_store is not None:
            log.info("[Gate AM] E6: entrenamiento desde HDF5 (RGB+saliencia, sin JPEG)")
        else:
            log.info("[Gate AM] E6: entrenamiento on-the-fly desde JPEG (HDF5 no disponible)")

    use_slice_ms = use_slice_probe or use_slice_finetune

    train_config = _build_gate_train_config(params, mode_label=mode_label, embed_store=embed_store)
    from .phase_d_stage1.gate_run_live import GateRunLivePublisher

    run_live = GateRunLivePublisher.create(
        run=run,
        train_df=train_df,
        val_df=val_df,
        info_split=info_split,
        train_config=train_config,
        protocol=params.protocol,
        ckpt_dir=ckpt_dir,
        embed_store=embed_store,
        gate4_config=params.gate4 if use_slice_ms else None,
        device=device,
        skip_pretrain_viz=params.skip_pretrain_viz,
    )
    from .phase_d_stage1.gate_run_live import print_train_console_banner, publish_pipeline_panel

    publish_pipeline_panel(
        pipeline_phase=1,
        phases=_pipeline_steps(
            step1_status="done",
            step2_status="done",
            step3_status="running",
            step1_pct=100,
            step2_pct=100,
            step3_detail="Entrenamiento iniciado",
        ),
    )
    print_train_console_banner(
        run_id=run.run_id,
        run_root=run.root,
        mode_label=mode_label,
        pooling_mode=pooling_mode,
        batch_size=params.batch_size,
        epochs=params.epochs,
        canvas_path=run_live.canvas_path,
    )
    stratified_sample_df = None
    if not params.skip_pretrain_viz and params.protocol.balance_mode == "g1_stratified":
        import pandas as pd

        from .phase_d_stage1.gate_tile_dino import _plan_train_epoch

        log.info(
            f"[Gate AM] Planificando sampler g1_stratified "
            f"({len(train_df):,} tiles train, muestra viz=4 batches)..."
        )
        sample_plan = _plan_train_epoch(
            train_df,
            max_bg_per_image=params.max_bg_per_image,
            balance_mode=params.protocol.balance_mode,
            mplus_oversample=params.protocol.mplus_oversample_factor,
            batch_size=params.batch_size,
            protocol=params.protocol,
            seed=0,
        )
        if sample_plan.stratified_batches:
            stratified_sample_df = pd.concat(sample_plan.stratified_batches[:4], ignore_index=True)
        log.info("[Gate AM] Sampler listo; generando artefactos pre-training...")
    elif params.skip_pretrain_viz:
        log.info("[Gate AM] Pre-training viz omitido (GATE_SKIP_PRETRAIN_VIZ); arranque rapido")
    else:
        log.info("[Gate AM] Generando artefactos pre-training...")
    run_live.publish_startup(
        stratified_sample_df=stratified_sample_df,
        skip_viz=params.skip_pretrain_viz,
    )

    log.info(
        f"[Gate AM] Entrando al loop de entrenamiento "
        f"(max {params.epochs} épocas, batch={params.batch_size}, "
        f"eval={params.protocol.eval_balance_mode}, "
        f"early_stop={params.protocol.early_stop_patience})..."
    )
    history = None
    train_error: Optional[str] = None
    try:
        history = train_gate_tile_dino_gpu(
            model=classifier,
            train_df=train_df,
            val_df=val_df,
            device=device,
            epochs=params.epochs,
            batch_size=params.batch_size,
            checkpoint_dir=ckpt_dir,
            max_train_batches=params.max_train_batches,
            max_val_batches=params.max_val_batches,
            max_bg_per_image=params.max_bg_per_image,
            embed_store=embed_store,
            h5_store=h5_store,
            freeze_backbone=freeze_backbone,
            protocol=params.protocol,
            save_live_snapshots=params.save_live_snapshots,
            live_snapshot_dir=ckpt_dir,
            dino_input_size=params.dino_input_size,
            seg_target_size=params.seg_target_size,
            gate4_config=params.gate4 if use_slice_ms else None,
            run_live=run_live,
            pooling_mode=pooling_mode,
            backbone_lr_factor=float(_cfg(cfg, "GATE_FINETUNE_BACKBONE_LR_FACTOR", 0.1)),
            force_cpu_decode=bool(_cfg(cfg, "GATE_TRAIN_FORCE_CPU_DECODE", False)),
            cpu_decode_above_mb=float(_cfg(cfg, "GATE_TRAIN_CPU_DECODE_ABOVE_MB", 300.0)),
        )
    except Exception as exc:
        train_error = str(exc)
        log.exception(f"[Gate AM] Entrenamiento interrumpido (run={run.run_id})")
        import json
        from datetime import datetime

        (run.root / "training_status.json").write_text(
            json.dumps(
                {
                    "status": "interrupted",
                    "run_id": run.run_id,
                    "error": train_error,
                    "recover_cmd": f"python run.py recover-gate-am-report --run-id {run.run_id}",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if (ckpt_dir / CHECKPOINT_NAME).exists() and embed_store is not None:
            log.info("[Gate AM] Intentando finalize parcial con mejor checkpoint...")
            try:
                history_dict = history_dict_from_artifacts(ckpt_dir)
                _finalize_gate_am_train(
                    run=run,
                    params=params,
                    history=history_dict,
                    classifier=classifier,
                    embed_store=embed_store,
                    train_df=train_df,
                    val_df=val_df,
                    info_split=info_split,
                    ckpt_dir=ckpt_dir,
                    device=device,
                    mode_label=mode_label,
                )
                _run_post_train_analysis(
                    run=run,
                    params=params,
                    ckpt_dir=ckpt_dir,
                    train_df=train_df,
                    val_df=val_df,
                    classifier=classifier,
                    embed_store=embed_store,
                    device=device,
                )
            except Exception:
                log.exception("[Gate AM] Finalize parcial fallo")
        raise
    finally:
        if h5_store is not None:
            h5_store.close()

    _write_gate_run_meta(ckpt_dir, run.run_id)

    try:
        report_md = _finalize_gate_am_train(
            run=run,
            params=params,
            history=history.to_dict(),
            classifier=classifier,
            embed_store=embed_store,
            train_df=train_df,
            val_df=val_df,
            info_split=info_split,
            ckpt_dir=ckpt_dir,
            device=device,
            mode_label=mode_label,
        )
        _run_post_train_analysis(
            run=run,
            params=params,
            ckpt_dir=ckpt_dir,
            train_df=train_df,
            val_df=val_df,
            classifier=classifier,
            embed_store=embed_store,
            device=device,
        )
        if bool(_cfg(cfg, "GATE_AMFINDER_EXTERNAL_EVAL", False)) and not external_df.empty:
            from .gate_external_eval import print_external_validation_banner, run_amfinder_external_eval

            print_external_validation_banner(n_tiles=len(external_df), phase="inicio")
            log.info(
                f"[Gate AM] Post-train: {len(external_df):,} tiles — "
                "validacion externa AMFinder (no Stage2; ver banner terminal)"
            )
            max_ext = int(_cfg(cfg, "GATE_AMFINDER_EXTERNAL_MAX_IMAGES", 0))
            run_amfinder_external_eval(
                external_df=external_df,
                run_dir=run.root,
                ckpt_dir=ckpt_dir,
                cfg=cfg,
                batch_size=params.batch_size,
                max_images=max_ext,
            )
    except Exception:
        log.exception(
            f"[Gate AM] Finalize fallo (artefactos parciales en {run.root}). "
            f"Reintenta reporte: python run.py recover-gate-am-report --run-id {run.run_id}"
        )
        raise
    from .phase_d_stage1.gate_explainability import print_run_artifact_index

    explain_path = run.reports / "EXPLICABILIDAD.md"
    if explain_path.exists():
        print_run_artifact_index(run, explain_path)
    log.info(f"[Gate AM] [bold green]OK[/bold green] -> {report_md}")


def run_gate_pipeline(cfg: Any) -> None:
    """Flujo principal: cache (preguntas) -> entrenamiento segun config.py."""
    import torch

    from .phase_d_stage1 import split_by_image
    from .phase_d_stage1.gate_embed_cache import inspect_embed_cache_status
    from .phase_d_stage1.gate_run_live import publish_pipeline_panel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible. Conecta GPU e intenta de nuevo.")

    backbone = str(_cfg(cfg, "GATE_BACKBONE", "dinov2_vits14"))
    cache_batch = int(_cfg(cfg, "GATE_CACHE_BATCH_SIZE", 32))
    pooling_mode = str(_cfg(cfg, "GATE_EMBED_POOLING_MODE", "none"))
    train_params = gate_train_params_from_config(cfg)

    from .phase_d_stage1.gate_vision import describe_dino_resolution

    print("=== MicorizaeVision — Gate AM (cache + entrenamiento) ===", flush=True)
    print(
        f"Config: backbone={backbone}, DINO={describe_dino_resolution(train_params.dino_input_size)}, "
        f"cache_batch={cache_batch}, pooling={pooling_mode}, "
        f"epochs<={train_params.epochs}, balance={train_params.protocol.balance_mode}, "
        f"checkpoint={train_params.protocol.checkpoint_metric}, "
        f"dynamic_batch={_cfg(cfg, 'GATE_CACHE_DYNAMIC_BATCH', True)}, "
        f"cpu_decode>{_cfg(cfg, 'GATE_CACHE_CPU_DECODE_ABOVE_MB', 500)}MB",
        flush=True,
    )
    print(
        "Live: metricas y canvas se actualizan automaticamente durante el train (sin scripts externos).",
        flush=True,
    )
    if train_params.probe:
        print(
            "Pipeline SSD: HDF5 (RGB+saliencia) -> embeddings DINO -> train desde cache.",
            flush=True,
        )
    else:
        print(
            "Pipeline E6: HDF5 (RGB+saliencia) -> LoRA fine-tune DINO on-the-fly (Slice-MS).",
            flush=True,
        )
    publish_pipeline_panel(
        pipeline_phase=0,
        phases=_pipeline_steps(
            step1_status="running",
            step2_status="pending",
            step3_status="pending",
            step1_detail="Conformando cache HDF5",
        ),
    )

    train_df, val_df, _ext, _info, all_tiles = resolve_gate_am_splits(cfg)
    h5_ready = _gate_h5_cache_ready(cfg, all_tiles)
    build_h5_before_train = bool(_cfg(cfg, "GATE_H5_BUILD_BEFORE_TRAIN", True))
    if bool(_cfg(cfg, "GATE_H5_CACHE_ENABLED", True)):
        if h5_ready:
            print(
                f"\n[Gate SSD] Paso 1/3: HDF5 tiles listo ({len(all_tiles):,} tiles, cache valido)",
                flush=True,
            )
        elif build_h5_before_train:
            execute_ensure_gate_h5_cache(cfg, all_tiles)
        elif not train_params.probe:
            from .common.paths import get_paths
            from .phase_d_stage1.gate_tile_h5_cache import cleanup_incomplete_h5_cache

            removed = cleanup_incomplete_h5_cache(get_paths().root)
            if removed:
                print(
                    f"\n[Gate SSD] Paso 1/3: HDF5 parcial eliminado ({len(removed)} archivos, ~liberado espacio SSD)",
                    flush=True,
                )
            print(
                "\n[Gate SSD] Paso 1/3: omitido (GATE_H5_BUILD_BEFORE_TRAIN=False). "
                "E6 LoRA entrena desde JPEG (decode CPU); reactiva el flag para compilar HDF5.",
                flush=True,
            )
        else:
            print(
                "\n[Gate SSD] Paso 1/3: HDF5 incompleto y build desactivado. "
                "Activa GATE_H5_BUILD_BEFORE_TRAIN o compila cache antes del probe.",
                flush=True,
            )
    skip_embed_cache = (not train_params.probe) and bool(_cfg(cfg, "GATE_E6_SKIP_EMBED_CACHE", True))
    if skip_embed_cache:
        print(
            "\n[Gate SSD] Paso 2/3: omitido (E6 LoRA usa HDF5; embed cache principal no requerida)",
            flush=True,
        )
        publish_pipeline_panel(
            pipeline_phase=0,
            phases=_pipeline_steps(
                step1_status="done",
                step2_status="done",
                step3_status="running",
                step1_pct=100,
                step2_pct=100,
                step2_detail="Omitido (E6 fine-tune)",
                step3_detail="Listo para entrenar",
            ),
        )
    else:
        publish_pipeline_panel(
            pipeline_phase=0,
            phases=_pipeline_steps(
                step1_status="done",
                step2_status="running",
                step3_status="pending",
                step1_pct=100,
                step2_detail="Compilando embeddings DINO",
            ),
        )

        train_images = set(train_df["image_path"].astype(str))
        mplus_aug = _mplus_aug_variants_from_cfg(cfg)
        cache_attn = bool(_cfg(cfg, "GATE_CACHE_ATTENTION", False))
        attn_layers = str(_cfg(cfg, "GATE_CACHE_ATTENTION_LAYERS", "all"))
        attn_reduce = str(_cfg(cfg, "GATE_CACHE_ATTENTION_HEAD_REDUCE", "mean"))
        cache_bn = _cache_basename_from_cfg(cfg)
        status = inspect_embed_cache_status(
            all_tiles,
            backbone_name=backbone,
            dino_input_size=train_params.dino_input_size,
            cache_basename=cache_bn,
            mplus_aug_variants=mplus_aug,
            mplus_aug_train_images=train_images,
            cache_attention=cache_attn,
            attention_layers=attn_layers,
            attention_head_reduce=attn_reduce,
            pooling_mode=pooling_mode,
        )
        action = resolve_cache_action(status, cfg)

        if action == "abort":
            raise SystemExit("Operacion cancelada.")
        if action == "finalize":
            from .phase_d_stage1.gate_embed_cache import finalize_embed_cache_metadata
            from .phase_d_stage1.gate_vision import dino_input_size_from_config, seg_target_size_for

            dino_in = dino_input_size_from_config(cfg)
            finalize_embed_cache_metadata(
                all_tiles,
                backbone_name=backbone,
                dino_input_size=dino_in,
                seg_target_size=seg_target_size_for(dino_in, cfg),
                n_tiles_total=status.n_tiles_expected,
                mplus_aug_variants=mplus_aug,
                mplus_aug_train_images=train_images,
            )
            publish_pipeline_panel(
                pipeline_phase=0,
                phases=_pipeline_steps(
                    step1_status="done",
                    step2_status="done",
                    step3_status="running",
                    step1_pct=100,
                    step2_pct=100,
                    step2_detail="Cache embeddings finalizada",
                    step3_detail="Listo para entrenar",
                ),
            )
        elif action == "build_attn":
            build_kw = _cache_build_kwargs(cfg)
            execute_build_gate_attention_cache(
                backbone=backbone,
                batch_size=cache_batch,
                force_rebuild=False,
                mplus_aug_variants=mplus_aug,
                mplus_aug_train_images=train_images,
                attention_layers=attn_layers,
                attention_head_reduce=attn_reduce,
                **{
                    k: v
                    for k, v in build_kw.items()
                    if k not in {"cache_attention", "attention_layers", "attention_head_reduce"}
                },
            )
            publish_pipeline_panel(
                pipeline_phase=0,
                phases=_pipeline_steps(
                    step1_status="done",
                    step2_status="done",
                    step3_status="running",
                    step1_pct=100,
                    step2_pct=100,
                    step2_detail="Embeddings + atencion listos",
                    step3_detail="Listo para entrenar",
                ),
            )
        elif action == "build":
            build_kw = _cache_build_kwargs(cfg)
            execute_build_gate_cache(
                backbone=backbone,
                batch_size=cache_batch,
                force_rebuild=status.state in {"valid", "partial", "stale", "obsolete"},
                cfg=cfg,
                mplus_aug_variants=mplus_aug,
                mplus_aug_train_images=train_images,
                **build_kw,
            )
            publish_pipeline_panel(
                pipeline_phase=0,
                phases=_pipeline_steps(
                    step1_status="done",
                    step2_status="done",
                    step3_status="running",
                    step1_pct=100,
                    step2_pct=100,
                    step2_detail="Cache embeddings lista",
                    step3_detail="Listo para entrenar",
                ),
            )
        else:
            print(f"Usando cache existente: {status.paths.embed}", flush=True)
            publish_pipeline_panel(
                pipeline_phase=0,
                phases=_pipeline_steps(
                    step1_status="done",
                    step2_status="done",
                    step3_status="running",
                    step1_pct=100,
                    step2_pct=100,
                    step2_detail="Usando cache existente",
                    step3_detail="Listo para entrenar",
                ),
            )

    step3_label = (
        "entrenamiento E6 LoRA (on-the-fly)"
        if not train_params.probe
        else "entrenamiento desde cache"
    )
    print(f"\n--- Paso 3/3: {step3_label} ---", flush=True)
    execute_train_gate_am(
        train_params,
        require_cache=train_params.probe,
        cfg=cfg,
    )
