"""Sampler estratificado para entrenamiento gate AM (Bg / M- / M+).

Problema del plan legacy (por imagen):
    - Batches mono-clase frecuentes -> Slice-MS y CE no contrastan M+/M-/Bg.
    - Balance por imagen no garantiza proporciones globales ni mezcla.

Este modulo garantiza por epoca de TRAIN:
    1. Mismo numero de muestras por clase (samples_per_class).
    2. Cada mini-batch contiene las tres clases gate con cuota fija (min_per_batch).
    3. Orden global barajado (no agrupado por imagen).

Eval/holdout NO usa este sampler (distribucion natural).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..common.logging_utils import get_logger
from .gpu_pipeline import EpochPlan

log = get_logger("phase_d.gate_sampler")

GATE_STRATIFIED_CLASSES: tuple[str, ...] = ("Background", "Mminus", "Mplus")


@dataclass(frozen=True)
class GateEpochSamplerConfig:
    """Configuracion del sampler estratificado G1.

    class_weights permite cuotas por-batch no uniformes (p.ej. más M-/M+ que
    Background) para maximizar pares duros M-/M+ disponibles a la minería MS.
    """

    samples_per_class: int = 3840
    min_per_class_per_batch: int = 16
    classes: tuple[str, ...] = GATE_STRATIFIED_CLASSES
    class_weights: tuple[tuple[str, float], ...] = ()
    domain_stratified: bool = False
    root_only_batch_ratio: float = 0.0
    hard_image_substrings: tuple[str, ...] = ()
    hard_image_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class debe ser > 0")
        if self.min_per_class_per_batch <= 0:
            raise ValueError("min_per_class_per_batch debe ser > 0")
        n_cls = len(self.classes)
        if n_cls == 0:
            raise ValueError("classes no puede estar vacio")
        batch_need = sum(self.quota_per_class().values())
        if batch_need > 512:
            raise ValueError(f"cuota total por batch demasiado grande ({batch_need})")

    def quota_per_class(self) -> dict[str, int]:
        """Tiles por clase y batch (>=1), aplicando class_weights sobre el mínimo."""
        weights = dict(self.class_weights)
        quota: dict[str, int] = {}
        for cls in self.classes:
            w = float(weights.get(cls, 1.0))
            quota[cls] = max(1, int(round(self.min_per_class_per_batch * w)))
        return quota


def _class_weights_from_protocol(protocol: object) -> tuple[tuple[str, float], ...]:
    """Lee pesos de cuota por clase (str -> tuplas), p.ej. {'Mminus':1.5,'Mplus':1.5}."""
    raw = getattr(protocol, "stratified_class_weights", None)
    if not raw:
        return ()
    return tuple((str(k), float(v)) for k, v in dict(raw).items())


ROOT_ONLY_CLASSES: tuple[str, ...] = ("Mminus", "Mplus")


def _boost_hard_tiles(
    raw: pd.DataFrame,
    cfg: GateEpochSamplerConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if cfg.hard_image_weight <= 1.0 or not cfg.hard_image_substrings:
        return raw
    if "image_path" not in raw.columns or raw.empty:
        return raw
    paths = raw["image_path"].astype(str)
    hard = raw[paths.apply(lambda p: any(s in p for s in cfg.hard_image_substrings))]
    if hard.empty:
        return raw
    extra_n = int(len(raw) * (cfg.hard_image_weight - 1.0))
    if extra_n <= 0:
        return raw
    extra = _sample_class_pool(hard, extra_n, rng)
    return pd.concat([raw, extra], ignore_index=False)


def sampler_config_from_protocol(protocol: object, batch_size: int) -> GateEpochSamplerConfig:
    """Construye config desde GateTrainProtocol + batch_size."""
    n_cls = len(GATE_STRATIFIED_CLASSES)
    min_pb = int(getattr(protocol, "stratified_min_per_batch", 0) or 0)
    if min_pb <= 0:
        min_pb = max(1, batch_size // n_cls)
    return GateEpochSamplerConfig(
        samples_per_class=int(getattr(protocol, "stratified_samples_per_class", 3840)),
        min_per_class_per_batch=min_pb,
        classes=GATE_STRATIFIED_CLASSES,
        class_weights=_class_weights_from_protocol(protocol),
        domain_stratified=bool(getattr(protocol, "train_domain_stratified", False)),
        root_only_batch_ratio=float(
            getattr(protocol, "stratified_root_only_batch_ratio", 0.0) or 0.0
        ),
        hard_image_substrings=tuple(
            getattr(protocol, "hard_image_substrings", ()) or ()
        ),
        hard_image_weight=float(getattr(protocol, "hard_image_weight", 1.0) or 1.0),
    )


def eval_sampler_config_from_protocol(protocol: object, batch_size: int) -> GateEpochSamplerConfig:
    """Config estratificada para eval/val (menos tiles/clase que train).

    Eval mantiene cuota uniforme (sin class_weights) para no sesgar métricas.
    """
    n_cls = len(GATE_STRATIFIED_CLASSES)
    min_pb = int(getattr(protocol, "stratified_min_per_batch", 0) or 0)
    if min_pb <= 0:
        min_pb = max(1, batch_size // n_cls)
    return GateEpochSamplerConfig(
        samples_per_class=int(getattr(protocol, "eval_stratified_samples_per_class", 384)),
        min_per_class_per_batch=min_pb,
        classes=GATE_STRATIFIED_CLASSES,
    )


def _sample_class_pool(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(pool) == 0:
        return pool.iloc[0:0]
    if len(pool) >= n:
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        return pool.loc[idx]
    idx = rng.choice(pool.index.to_numpy(), size=n, replace=True)
    return pool.loc[idx]


def _build_shuffled_pools(
    tiles_df: pd.DataFrame,
    cfg: GateEpochSamplerConfig,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    pools: dict[str, pd.DataFrame] = {}
    for cls in cfg.classes:
        raw = tiles_df[tiles_df["stage1"].astype(str) == cls]
        raw = _boost_hard_tiles(raw, cfg, rng)
        if cfg.domain_stratified and "domain_bucket" in raw.columns and len(raw):
            domains = [
                d
                for d in sorted(raw["domain_bucket"].astype(str).unique().tolist())
                if len(raw[raw["domain_bucket"].astype(str) == d]) > 0
            ]
            n_dom = max(len(domains), 1)
            per_dom = max(1, cfg.samples_per_class // n_dom)
            parts: list[pd.DataFrame] = []
            for dom in domains:
                sub = raw[raw["domain_bucket"].astype(str) == dom]
                if sub.empty:
                    continue
                parts.append(_sample_class_pool(sub, per_dom, rng))
            combined = pd.concat(parts, ignore_index=False) if parts else raw.iloc[0:0]
            if len(combined) < cfg.samples_per_class:
                extra = _sample_class_pool(raw, cfg.samples_per_class - len(combined), rng)
                combined = pd.concat([combined, extra], ignore_index=False)
            if len(combined) > cfg.samples_per_class:
                idx = rng.choice(combined.index.to_numpy(), size=cfg.samples_per_class, replace=False)
                combined = combined.loc[idx]
            sampled = combined
        else:
            sampled = _sample_class_pool(raw, cfg.samples_per_class, rng)
        if len(sampled) == 0:
            log.warning(f"[Gate sampler] clase {cls!r} sin tiles en train")
        pools[cls] = sampled.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    return pools


def _take_from_pool(
    pool: pd.DataFrame,
    cursor: int,
    n: int,
    rng: np.random.Generator,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Toma n filas avanzando cursor; re-baraja source si hace falta."""
    if len(pool) == 0:
        return pool.iloc[0:0], cursor
    out_parts: list[pd.DataFrame] = []
    remaining = n
    pos = cursor
    working = pool
    while remaining > 0:
        avail = len(working) - pos
        if avail <= 0:
            working = source.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            pos = 0
            avail = len(working)
        take = min(remaining, avail)
        out_parts.append(working.iloc[pos : pos + take])
        pos += take
        remaining -= take
    return pd.concat(out_parts, ignore_index=False), pos


def build_stratified_batches(
    tiles_df: pd.DataFrame,
    *,
    cfg: GateEpochSamplerConfig,
    seed: int = 0,
) -> list[pd.DataFrame]:
    """Lista de DataFrames, uno por mini-batch, cada uno con las 3 clases."""
    rng = np.random.default_rng(seed)
    pools = _build_shuffled_pools(tiles_df, cfg, rng)
    sources = {cls: pools[cls] for cls in cfg.classes}
    cursors = {cls: 0 for cls in cfg.classes}

    n_cls = len(cfg.classes)
    quota = cfg.quota_per_class()
    batch_size = sum(quota.values())
    n_batches = (cfg.samples_per_class * n_cls) // batch_size
    if n_batches <= 0:
        raise ValueError(
            f"samples_per_class={cfg.samples_per_class} y cuota batch={batch_size} "
            f"producen 0 batches"
        )

    ratio = max(0.0, min(1.0, float(cfg.root_only_batch_ratio)))
    n_root = int(round(n_batches * ratio)) if ratio > 0.0 else 0
    n_mixed = n_batches - n_root

    mixed_batches: list[pd.DataFrame] = []
    for _ in range(n_mixed):
        parts: list[pd.DataFrame] = []
        for cls in cfg.classes:
            chunk, cursors[cls] = _take_from_pool(
                pools[cls], cursors[cls], quota[cls], rng, sources[cls]
            )
            parts.append(chunk)
        batch = pd.concat(parts, ignore_index=False)
        batch = batch.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        mixed_batches.append(batch)

    root_batches: list[pd.DataFrame] = []
    root_quota = cfg.min_per_class_per_batch
    for _ in range(n_root):
        parts: list[pd.DataFrame] = []
        for cls in ROOT_ONLY_CLASSES:
            chunk, cursors[cls] = _take_from_pool(
                pools[cls], cursors[cls], root_quota, rng, sources[cls]
            )
            parts.append(chunk)
        batch = pd.concat(parts, ignore_index=False)
        batch = batch.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        root_batches.append(batch)

    batches = mixed_batches + root_batches
    rng.shuffle(batches)
    return batches


def plan_epoch_stratified(
    tiles_df: pd.DataFrame,
    *,
    cfg: GateEpochSamplerConfig,
    seed: int = 0,
) -> EpochPlan:
    """Plan de epoca con batches pre-mezclados (stratified)."""
    batches = build_stratified_batches(tiles_df, cfg=cfg, seed=seed)
    return EpochPlan(items=[], stratified_batches=batches)


def stratified_epoch_class_counts(plan: EpochPlan) -> dict[str, int]:
    if not plan.stratified_batches:
        return {}
    df = pd.concat(plan.stratified_batches, ignore_index=True)
    counts = {c: 0 for c in GATE_STRATIFIED_CLASSES}
    for name, n in df["stage1"].astype(str).value_counts().items():
        counts[str(name)] = int(n)
    return counts


def audit_stratified_batches(
    plan: EpochPlan,
    *,
    classes: tuple[str, ...] = GATE_STRATIFIED_CLASSES,
) -> dict[str, float | int]:
    """Estadisticas de mezcla por batch (min/max de presencia por clase)."""
    if not plan.stratified_batches:
        return {"n_batches": 0}
    per_class_counts: dict[str, list[int]] = {c: [] for c in classes}
    for batch in plan.stratified_batches:
        vc = batch["stage1"].astype(str).value_counts()
        for cls in classes:
            per_class_counts[cls].append(int(vc.get(cls, 0)))
    return {
        "n_batches": len(plan.stratified_batches),
        "tiles_per_batch_min": min(len(b) for b in plan.stratified_batches),
        "tiles_per_batch_max": max(len(b) for b in plan.stratified_batches),
        **{
            f"{cls}_per_batch_min": min(per_class_counts[cls]) if per_class_counts[cls] else 0
            for cls in classes
        },
        **{
            f"{cls}_per_batch_max": max(per_class_counts[cls]) if per_class_counts[cls] else 0
            for cls in classes
        },
    }
