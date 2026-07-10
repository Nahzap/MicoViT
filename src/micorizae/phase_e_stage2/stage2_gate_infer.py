"""Inferencia Gate secuencial para Stage2-Pixel — Modelo 1 sin atajos a gold."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..common.logging_utils import get_logger
from ..phase_d_stage1.gate_tile_dino import GateProbeBundle, infer_image_gate_probe_gpu, load_gate_probe_bundle

log = get_logger("phase_e.stage2_gate_infer")

_GATE_SCOPE_TILES: Optional[pd.DataFrame] = None


def load_stage2_gate_bundle(
    *,
    device,
    gate_run_id: str = "",
    cfg: Optional[object] = None,
) -> GateProbeBundle:
    """Carga el bundle Gate AM asociado al ``gate_run_id`` de Stage2."""
    return load_gate_probe_bundle(device=device, gate_run_id=gate_run_id, cfg=cfg)


def gate_infer_tile_scope(cfg: Optional[object] = None) -> pd.DataFrame:
    """Malla de tiles usada por Gate (misma que ``build-gate-cache``)."""
    global _GATE_SCOPE_TILES
    if _GATE_SCOPE_TILES is not None:
        return _GATE_SCOPE_TILES

    import config as user_config  # type: ignore

    from ..gate_runflow import gate_train_params_from_config, resolve_gate_am_splits

    cfg = cfg or user_config
    params = gate_train_params_from_config(cfg)
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    _train_df, _val_df, _ext, _info, cache_tiles = resolve_gate_am_splits(
        cfg, exclude_unreadable=not include_unknown
    )
    _GATE_SCOPE_TILES = cache_tiles.copy()
    return _GATE_SCOPE_TILES


def _required_gate_infer_tiles(tiles_df: pd.DataFrame, *, cfg: Optional[object] = None) -> pd.DataFrame:
    """Tiles Gate (scope cache) para las imágenes solicitadas."""
    req_images = set(tiles_df["image_path"].astype(str).unique())
    scope = gate_infer_tile_scope(cfg)
    required = scope[scope["image_path"].astype(str).isin(req_images)].copy()
    return required.drop_duplicates(subset=["image_path", "row", "col"]).reset_index(drop=True)


def infer_image_gate_stage2(
    image_path: str | Path,
    bundle: GateProbeBundle,
    *,
    batch_size: int = 64,
    strict: bool = True,
    cfg: Optional[object] = None,
    tile_scope_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Modelo 1 sobre la malla Gate de la imagen (manifest ∩ cache scope).

    ``strict=True`` (default Stage2): prohibido rellenar con gold ``stage1`` del
    manifest si falta embed cache — el pipeline debe usar predicciones del Gate.
    """
    scope = tile_scope_df if tile_scope_df is not None else gate_infer_tile_scope(cfg)
    return infer_image_gate_probe_gpu(
        image_path,
        bundle,
        batch_size=batch_size,
        strict=strict,
        tile_scope_df=scope,
    )


def assert_gate_embed_cache_ready(
    tiles_df: pd.DataFrame,
    *,
    cfg: Optional[object] = None,
) -> None:
    """Verifica cache Gate válido y cobertura lookup para inferencia secuencial Stage2."""
    import config as user_config  # type: ignore

    from ..gate_runflow import _cache_basename_from_cfg, gate_train_params_from_config, resolve_gate_am_splits
    from ..common.paths import get_paths
    from ..phase_d_stage1.gate_embed_cache import _cache_paths, inspect_embed_cache_status

    cfg = cfg or user_config
    params = gate_train_params_from_config(cfg)
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    train_df, _val_df, _ext, _info, cache_tiles = resolve_gate_am_splits(
        cfg, exclude_unreadable=not include_unknown
    )
    train_images = set(train_df["image_path"].astype(str))
    cache_bn = _cache_basename_from_cfg(cfg)

    status = inspect_embed_cache_status(
        cache_tiles,
        backbone_name=params.backbone,
        dino_input_size=params.dino_input_size,
        cache_basename=cache_bn,
        mplus_aug_variants=params.mplus_aug_variants,
        mplus_aug_train_images=train_images,
        cache_attention=params.cache_attention,
        attention_layers=params.attention_layers,
        attention_head_reduce=params.attention_head_reduce,
        pooling_mode=params.pooling_mode,
    )
    if status.state != "valid":
        raise RuntimeError(
            "Gate embed cache no válido para inferencia secuencial Stage2 "
            f"(estado={status.state!r}, tiles_meta={status.n_tiles_meta}, "
            f"esperados={status.n_tiles_expected}). "
            "Paso obligatorio Fase 1: python run.py build-gate-cache\n"
            f"Detalle: {status.message}"
        )

    required = _required_gate_infer_tiles(tiles_df, cfg=cfg)
    cpaths = _cache_paths(get_paths().root, cache_bn)
    if not cpaths.lookup.is_file():
        raise RuntimeError(
            f"Gate embed lookup ausente: {cpaths.lookup}. "
            "Ejecuta: python run.py build-gate-cache"
        )
    lookup = pd.read_parquet(cpaths.lookup)
    if "aug_id" in lookup.columns:
        lookup = lookup[lookup["aug_id"].fillna(0).astype(int) == 0]
    lookup_keys = {
        (str(r.image_path), int(r.row), int(r.col))
        for r in lookup[["image_path", "row", "col"]].itertuples(index=False)
    }
    missing = 0
    for r in required.itertuples(index=False):
        key = (str(r.image_path), int(r.row), int(r.col))
        if key not in lookup_keys:
            missing += 1
    if missing:
        raise RuntimeError(
            f"Gate embed cache incompleto para Stage2: faltan {missing}/{len(required)} tiles "
            f"en lookup ({required['image_path'].nunique()} imágenes). "
            "Ejecuta: python run.py build-gate-cache"
        )

    log.info(
        f"[Stage2-Gate] embed cache OK ({status.n_tiles_meta:,} tiles globales, "
        f"{len(required):,} tiles requeridos cubiertos, {status.embed_mb:.1f} MB)"
    )
