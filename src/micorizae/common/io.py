"""I/O resiliente: prefiere parquet si hay engine, cae con elegancia a CSV.

Toda fase que escriba/lea manifests debe usar `write_table` y `read_table`
para mantener compatibilidad cuando el entorno no tiene pyarrow/fastparquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .logging_utils import get_logger

log = get_logger("io")


def has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401

        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, base_path: Path, warn_csv: bool = True) -> Path:
    """Escribe parquet si hay engine; si no, CSV. Devuelve la ruta final."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if has_parquet_engine():
        path = base_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    path = base_path.with_suffix(".csv")
    df.to_csv(path, index=False)
    if warn_csv:
        log.warning(
            f"[yellow]pyarrow/fastparquet no disponibles -> {path.name} escrito en CSV[/yellow]"
        )
    return path


def read_table(base_path: Path, candidates: Iterable[str] = (".parquet", ".csv")) -> pd.DataFrame:
    """Lee parquet o csv, en ese orden de preferencia."""
    base_path = Path(base_path)
    if base_path.suffix in {".parquet", ".csv"} and base_path.exists():
        return _read(base_path)
    for ext in candidates:
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            return _read(candidate)
    raise FileNotFoundError(f"No se encontró {base_path} con ninguna extensión {tuple(candidates)}")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")
