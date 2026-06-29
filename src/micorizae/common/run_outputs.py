"""Directorios de salida con timestamp para cada corrida.

Cada ejecución que produce artefactos crea:

    outputs/<YYYYMMDD_HHMMSS>_<name>/
        maps/
        tables/
        reports/

Ver MASTER_PLAN §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .paths import get_paths


def timestamp_str(when: Optional[datetime] = None) -> str:
    """Prefijo `YYYYMMDD_HHMMSS` en hora local."""
    dt = when or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunOutputs:
    """Raíz de una corrida reproducible bajo `outputs/`."""

    run_id: str
    root: Path

    @property
    def maps(self) -> Path:
        return self.root / "maps"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "RunOutputs":
        for p in (self.root, self.maps, self.tables, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def open(
        cls,
        run_id: str,
        *,
        outputs_root: Optional[Path] = None,
    ) -> "RunOutputs":
        """Abre una corrida existente bajo `outputs/<run_id>/`."""
        paths = get_paths()
        base = outputs_root or paths.outputs
        root = base / run_id
        if not root.is_dir():
            raise FileNotFoundError(f"Run no encontrado: {root}")
        return cls(run_id=run_id, root=root).ensure()

    @classmethod
    def create(
        cls,
        name: str,
        *,
        outputs_root: Optional[Path] = None,
        when: Optional[datetime] = None,
        suffix: Optional[str] = None,
    ) -> "RunOutputs":
        """Crea `outputs/<ts>_<name>[__suffix]/` con subcarpetas estándar."""
        paths = get_paths()
        base = outputs_root or paths.outputs
        run_id = f"{timestamp_str(when)}_{name}"
        if suffix:
            run_id = f"{run_id}__{suffix}"
        run = cls(run_id=run_id, root=base / run_id)
        return run.ensure()
