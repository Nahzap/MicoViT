"""Configuracion tipada Gate4 + Slice MS (lee config.py del proyecto)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..gate_classes import GATE_CLASS_TO_IDX


@dataclass(frozen=True)
class Gate4SliceMSConfig:
    enabled: bool = True
    embed_dim: int = 128
    num_slices: int = 4
    alpha: float = 2.0
    beta: float = 50.0
    base: float = 0.5
    loss_weight: float = 0.5
    warmup_epochs: int = 3
    include_unknown_in_split: bool = False  # True requiere cache recompilado con Unknown
    # Minería de pares + margen confundible (Slice-MS pura, FASE 5.1/5.3).
    hard_mining: bool = False
    mining_margin: float = 0.1
    confusable_pairs: tuple[tuple[int, int], ...] = ()
    confusable_neg_weight: float = 1.0
    confusable_guard_neg_weight: float = 1.0
    confusable_base: float | None = None
    confusable_band_low: float | None = None
    confusable_band_high: float | None = None
    confusable_directed_weights: tuple[tuple[int, int, float], ...] = ()
    band_start_epoch: int = 1
    # Sub-prototipos por clase (inferencia métrica; no cambia la pérdida).
    proto_subcenters: int = 1
    proto_subcenters_per_class: tuple[int, ...] = ()
    domain_aware_subcenters: bool = False

    @property
    def proto_subcenters_max(self) -> int:
        if self.proto_subcenters_per_class:
            return max(self.proto_subcenters_per_class)
        return max(1, self.proto_subcenters)

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_slices != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) debe ser divisible por num_slices ({self.num_slices})"
            )
        k = self.proto_subcenters_max
        if self.proto_subcenters_per_class:
            if any(k < 1 for k in self.proto_subcenters_per_class):
                raise ValueError(f"proto_subcenters_per_class debe ser >= 1: {self.proto_subcenters_per_class}")
        elif self.proto_subcenters < 1:
            raise ValueError(f"proto_subcenters debe ser >= 1, recibido {self.proto_subcenters}")
        _ = k  # validado


def _parse_confusable_directed_weights(spec: str) -> tuple[tuple[int, int, float], ...]:
    """'Mminus:Mplus:1.5,Mplus:Mminus:1.1' -> ((1,2,1.5),(2,1,1.1))."""
    spec = (spec or "").strip()
    if not spec:
        return ()
    out: list[tuple[int, int, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"peso dirigido inválido (esperado ClsA:ClsB:peso): {chunk!r}")
        a, b, w = parts[0].strip(), parts[1].strip(), float(parts[2].strip())
        if a not in GATE_CLASS_TO_IDX or b not in GATE_CLASS_TO_IDX:
            raise ValueError(f"clase inválida en peso dirigido: {chunk!r}")
        out.append((GATE_CLASS_TO_IDX[a], GATE_CLASS_TO_IDX[b], w))
    return tuple(out)


def _parse_confusable_pairs(spec: str) -> tuple[tuple[int, int], ...]:
    """'Mminus:Mplus,Background:Mminus' -> ((1,2),(0,1)) usando índices gate."""
    spec = (spec or "").strip()
    if not spec:
        return ()
    pairs: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, _, b = chunk.partition(":")
        a, b = a.strip(), b.strip()
        if a not in GATE_CLASS_TO_IDX or b not in GATE_CLASS_TO_IDX:
            raise ValueError(f"par confundible inválido: {chunk!r} (clases: {list(GATE_CLASS_TO_IDX)})")
        pairs.append((GATE_CLASS_TO_IDX[a], GATE_CLASS_TO_IDX[b]))
    return tuple(pairs)


def _parse_subcenters_by_class(spec: str, *, num_classes: int, fallback: int) -> tuple[int, ...]:
    """'Background:1,Mminus:3,Mplus:3' -> tupla alineada con GATE_CLASS_NAMES."""
    from ..gate_classes import GATE_CLASS_NAMES

    spec = (spec or "").strip()
    if not spec:
        return (max(1, fallback),) * num_classes
    overrides: dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, val = chunk.partition(":")
        overrides[name.strip()] = max(1, int(val.strip()))
    out: list[int] = []
    for cls in GATE_CLASS_NAMES[:num_classes]:
        out.append(overrides.get(cls, max(1, fallback)))
    while len(out) < num_classes:
        out.append(max(1, fallback))
    return tuple(out)


def gate4_config_from_module(cfg: Any) -> Gate4SliceMSConfig:
    from ..gate_classes import GATE_CLASS_NAMES

    conf_base_raw = getattr(cfg, "GATE_MS_CONFUSABLE_BASE", None)
    fallback_k = int(getattr(cfg, "GATE_PROTO_SUBCENTERS", 1))
    k_by_class = _parse_subcenters_by_class(
        str(getattr(cfg, "GATE_PROTO_SUBCENTERS_BY_CLASS", "")),
        num_classes=len(GATE_CLASS_NAMES),
        fallback=fallback_k,
    )
    return Gate4SliceMSConfig(
        enabled=bool(getattr(cfg, "GATE4_SLICE_MS_ENABLED", True)),
        embed_dim=int(getattr(cfg, "GATE_MS_EMBED_DIM", 128)),
        num_slices=int(getattr(cfg, "GATE_MS_NUM_SLICES", 4)),
        alpha=float(getattr(cfg, "GATE_MS_ALPHA", 2.0)),
        beta=float(getattr(cfg, "GATE_MS_BETA", 50.0)),
        base=float(getattr(cfg, "GATE_MS_BASE", 0.5)),
        loss_weight=float(getattr(cfg, "GATE_MS_LOSS_WEIGHT", 0.5)),
        warmup_epochs=int(getattr(cfg, "GATE_MS_WARMUP_EPOCHS", 3)),
        include_unknown_in_split=bool(getattr(cfg, "GATE4_INCLUDE_UNKNOWN", False)),
        hard_mining=bool(getattr(cfg, "GATE_MS_HARD_MINING", False)),
        mining_margin=float(getattr(cfg, "GATE_MS_MINING_MARGIN", 0.1)),
        confusable_pairs=_parse_confusable_pairs(str(getattr(cfg, "GATE_MS_CONFUSABLE_PAIRS", ""))),
        confusable_neg_weight=float(getattr(cfg, "GATE_MS_CONFUSABLE_NEG_WEIGHT", 1.0)),
        confusable_guard_neg_weight=float(getattr(cfg, "GATE_MS_CONFUSABLE_GUARD_NEG_WEIGHT", 1.0)),
        confusable_base=(None if conf_base_raw is None else float(conf_base_raw)),
        confusable_band_low=(
            None
            if getattr(cfg, "GATE_MS_CONFUSABLE_BAND_LOW", None) is None
            else float(getattr(cfg, "GATE_MS_CONFUSABLE_BAND_LOW"))
        ),
        confusable_band_high=(
            None
            if getattr(cfg, "GATE_MS_CONFUSABLE_BAND_HIGH", None) is None
            else float(getattr(cfg, "GATE_MS_CONFUSABLE_BAND_HIGH"))
        ),
        confusable_directed_weights=_parse_confusable_directed_weights(
            str(getattr(cfg, "GATE_MS_CONFUSABLE_DIRECTED_WEIGHTS", ""))
        ),
        band_start_epoch=int(getattr(cfg, "GATE_MS_BAND_START_EPOCH", 1)),
        proto_subcenters=fallback_k,
        proto_subcenters_per_class=k_by_class,
        domain_aware_subcenters=bool(getattr(cfg, "GATE_DOMAIN_AWARE_SUBCENTERS", False)),
    )
