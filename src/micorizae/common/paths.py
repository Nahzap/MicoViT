"""Paths centralizados del proyecto.

Cualquier módulo del paquete debe pedir rutas a `get_paths()` en vez de hard-codear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _find_project_root(start: Path | None = None) -> Path:
    """Sube hasta encontrar el marcador `pyproject` o `requirements.txt`.

    Si no encuentra nada, devuelve dos niveles arriba de este archivo
    (`src/micorizae/common/paths.py` → repo root).
    """
    here = (start or Path(__file__).resolve()).parent
    for candidate in [here, *here.parents]:
        if (candidate / "requirements.txt").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    docs: Path
    configs: Path
    manifests: Path
    outputs: Path
    src: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data=root / "Data",
            docs=root / "Docs",
            configs=root / "configs",
            manifests=root / "manifests",
            outputs=root / "outputs",
            src=root / "src",
        )

    def ensure(self) -> None:
        for p in (self.manifests, self.outputs):
            p.mkdir(parents=True, exist_ok=True)


def get_paths() -> ProjectPaths:
    paths = ProjectPaths.from_root(_find_project_root())
    paths.ensure()
    return paths
