"""Fase A — Ingesta y normalización de datos.

Entrega E1 del master plan:
    - manifest_images.parquet
    - manifest_labels.parquet
    - schema_map.yaml (versión + verificación)
    - ingest_report.md
"""

from .build_manifests import build_manifests, IngestResult

__all__ = ["build_manifests", "IngestResult"]
