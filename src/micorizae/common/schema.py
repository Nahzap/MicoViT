"""Carga y aplicación del `schema_map.yaml`.

Convierte un CSV one-hot del Teacher (AMFinder/MycorrhizaFinder) en etiquetas
canónicas Stage1/Stage2 según la política definida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml


STAGE1_CANONICAL = ("Mplus", "Mminus", "Background", "Unreadable")


@dataclass
class ColumnMap:
    name: str
    stage1: str
    stage2: Optional[str] = None
    lineage: Optional[str] = None
    aux_class: Optional[str] = None


@dataclass
class SchemaMap:
    version: int
    stage1_classes: list[str]
    stage2_classes: dict[str, list[str]]
    columns: dict[str, ColumnMap]
    conflict_policy: str = "priority"
    priority_order: list[str] = field(default_factory=list)
    lineage_detectors: list[dict] = field(default_factory=list)

    def known_columns(self) -> set[str]:
        return set(self.columns.keys())

    def detect_lineage(self, path: Optional[str], header: Iterable[str]) -> Optional[str]:
        header_set = set(header)
        for det in self.lineage_detectors:
            needed = set(det.get("header_must_include", []) or [])
            substrs = det.get("path_substrings", []) or []
            path_lc = (path or "").lower()
            ok_header = needed.issubset(header_set) if needed else False
            ok_path = any(s.lower() in path_lc for s in substrs) if substrs else False
            if ok_header or ok_path:
                return det["name"]
        return None

    def reduce_row(self, row: dict[str, int]) -> tuple[str, Optional[str], Optional[str]]:
        """Reduce un one-hot a (stage1, stage2, aux_class) según la política.

        `row` debe contener pares {columna_csv: 0/1}. Las columnas desconocidas
        se ignoran.
        """
        positives = [name for name, val in row.items() if val and name in self.columns]
        if not positives:
            return ("Background", None, None)

        if self.conflict_policy == "priority":
            ordered = [c for c in self.priority_order if c in positives]
            chosen = ordered[0] if ordered else positives[0]
        else:
            chosen = positives[0]

        col = self.columns[chosen]
        return (col.stage1, col.stage2, col.aux_class)


def load_schema_map(path: Path) -> SchemaMap:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    columns_raw = raw.get("columns", {}) or {}
    columns: dict[str, ColumnMap] = {}
    for name, spec in columns_raw.items():
        columns[name] = ColumnMap(
            name=name,
            stage1=spec["stage1"],
            stage2=spec.get("stage2"),
            lineage=spec.get("lineage"),
            aux_class=spec.get("aux_class"),
        )

    return SchemaMap(
        version=int(raw.get("schema_version", 1)),
        stage1_classes=list(raw.get("stage1_classes", list(STAGE1_CANONICAL))),
        stage2_classes={k: list(v) for k, v in (raw.get("stage2_classes") or {}).items()},
        columns=columns,
        conflict_policy=str(raw.get("conflict_policy", "priority")),
        priority_order=list(raw.get("priority_order", []) or []),
        lineage_detectors=list(raw.get("lineage_detectors", []) or []),
    )
