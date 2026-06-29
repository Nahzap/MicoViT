"""Mapeo de subclases Stage2 por linaje (desde schema_map.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..common.paths import get_paths
from ..common.schema import load_schema_map


@dataclass(frozen=True)
class Stage2ClassMap:
    lineage: str
    classes: tuple[str, ...]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def encode(self, labels: pd.Series) -> pd.Series:
        unknown = set(labels.dropna().unique()) - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"Etiquetas Stage2 desconocidas para {self.lineage}: {unknown}")
        return labels.map(self.class_to_idx).astype("Int64")

    def decode(self, indices) -> list[str]:
        return [self.idx_to_class[int(i)] for i in indices]


def load_stage2_class_map(
    lineage: str,
    schema_path: Optional[Path] = None,
    *,
    only_present_in: Optional[pd.DataFrame] = None,
) -> Stage2ClassMap:
    """Clases Stage2 válidas para tiles M+ de un linaje."""
    paths = get_paths()
    schema = load_schema_map(schema_path or (paths.configs / "schema_map.yaml"))
    raw = list(schema.stage2_classes.get(lineage, []))
    if only_present_in is not None:
        present = set(
            only_present_in.loc[
                only_present_in["stage1"] == "Mplus", "stage2"
            ].dropna().unique()
        )
        classes = [c for c in raw if c in present]
    else:
        classes = [c for c in raw if c != "DSE"]
    if not classes:
        raise ValueError(f"Sin clases Stage2 para linaje {lineage}")
    c2i = {c: i for i, c in enumerate(classes)}
    return Stage2ClassMap(
        lineage=lineage,
        classes=tuple(classes),
        class_to_idx=c2i,
        idx_to_class={i: c for c, i in c2i.items()},
    )
