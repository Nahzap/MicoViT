"""Clases canónicas del gate Stage1 (Evangelisti CNN1 / AMFinder)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Orden fijo de índices softmax (0..3). Indices 0..2 compatibles con cache v3.
GATE_CLASS_NAMES: tuple[str, ...] = ("Background", "Mminus", "Mplus", "Unknown")

GATE_CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(GATE_CLASS_NAMES)}

GATE_IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(GATE_CLASS_NAMES)}

# stage1 del manifest -> clase gate4
STAGE1_TO_GATE: dict[str, str] = {
    "Background": "Background",
    "Mminus": "Mminus",
    "Mplus": "Mplus",
    "Unreadable": "Unknown",
}

# Alias visual L2 (composer)
GATE4_VISUAL_ALIASES: dict[str, str] = {
    "Unknown": "Unreadable",
    "Unreadable": "Unreadable",
}


def is_valid_stage1(stage1: object) -> bool:
    """True si stage1 es mapeable a clase gate (no vacío / NaN)."""
    if stage1 is None:
        return False
    try:
        if pd.isna(stage1):
            return False
    except (TypeError, ValueError):
        pass
    key = str(stage1).strip()
    return bool(key) and key in STAGE1_TO_GATE


def stage1_to_gate_label(stage1: str) -> str:
    key = str(stage1).strip()
    if key not in STAGE1_TO_GATE:
        raise ValueError(f"stage1 no mapeable al gate: {stage1!r}")
    return STAGE1_TO_GATE[key]


def drop_invalid_stage1_rows(df: pd.DataFrame, *, log_prefix: str = "") -> pd.DataFrame:
    """Elimina filas con stage1 vacío o no mapeable; log si se descartan."""
    if df.empty or "stage1" not in df.columns:
        return df
    mask = df["stage1"].map(is_valid_stage1)
    n_bad = int((~mask).sum())
    if n_bad > 0:
        from ..common.logging_utils import get_logger

        get_logger("phase_d.gate_classes").warning(
            f"{log_prefix}Descartando {n_bad} tiles con stage1 invalido/vacio "
            f"(de {len(df)} total)"
        )
    return df.loc[mask].reset_index(drop=True)


def encode_gate_indices(stage1_values: np.ndarray | pd.Series) -> np.ndarray:
    """Convierte etiquetas stage1 a índices int64 para CrossEntropy."""
    arr = np.asarray(stage1_values, dtype=object)
    out = np.empty(len(arr), dtype=np.int64)
    for i, s in enumerate(arr):
        if not is_valid_stage1(s):
            raise ValueError(f"stage1 no mapeable al gate: {s!r}")
        out[i] = GATE_CLASS_TO_IDX[stage1_to_gate_label(str(s))]
    return out


def encode_gate_indices_filtered(
    stage1_values: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices validos + mascara booleana (para eval con gold parcialmente vacio)."""
    arr = np.asarray(stage1_values, dtype=object)
    mask = np.array([is_valid_stage1(s) for s in arr], dtype=bool)
    if not mask.any():
        return np.empty(0, dtype=np.int64), mask
    valid = arr[mask]
    encoded = np.empty(int(mask.sum()), dtype=np.int64)
    for i, s in enumerate(valid):
        encoded[i] = GATE_CLASS_TO_IDX[stage1_to_gate_label(str(s))]
    return encoded, mask


def decode_gate_indices(indices: np.ndarray) -> list[str]:
    return [GATE_IDX_TO_CLASS[int(i)] for i in indices]


def gate_label_for_visual(stage1_or_gate: str) -> str:
    """Nombre para capas visuales (Unknown -> Unreadable en leyenda)."""
    g = stage1_to_gate_label(stage1_or_gate) if stage1_or_gate in STAGE1_TO_GATE else str(stage1_or_gate)
    return GATE4_VISUAL_ALIASES.get(g, g)
