"""Construcción de manifests para la Fase A (E1).

Lee `configs/datasets.yaml` y `configs/schema_map.yaml`, recorre `Data/`,
empareja cada imagen con su CSV `*_cnn_1_annotations.csv` y produce:

    manifests/manifest_images.parquet
    manifests/manifest_labels.parquet
    manifests/ingest_report.md

Validaciones implementadas:
    - imagen existe y es legible (sin abrir el bitmap completo: chequeo de header).
    - columnas del CSV están en el schema.
    - unicidad (file, row, col).
    - consistencia one-hot por fila (al menos un 1, política de conflicto aplicada).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml
from tqdm import tqdm

from ..common.io import write_table
from ..common.logging_utils import get_logger
from ..common.paths import ProjectPaths, get_paths
from ..common.schema import SchemaMap, load_schema_map

log = get_logger("phase_a_ingest")

_ANNOTATION_TS_RE = re.compile(
    r"^(?P<stem>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+)_cnn_1_annotations\.csv$"
)


@dataclass
class IngestResult:
    images: pd.DataFrame
    labels: pd.DataFrame
    report: dict
    images_path: Path
    labels_path: Path
    report_path: Path


def _load_datasets_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _match_annotation(image_path: Path) -> Optional[Path]:
    """Dada una imagen, encuentra el CSV de anotaciones más reciente que comparte el stem."""
    stem = image_path.stem
    parent = image_path.parent
    candidates = sorted(parent.glob(f"{stem}_*_cnn_1_annotations.csv"))
    if not candidates:
        flat = parent / f"{stem}.csv"
        if flat.exists():
            return flat
        return None
    return candidates[-1]


def _image_dimensions(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Lee solo el header de la imagen para obtener (width, height).

    Las imágenes microscópicas pueden ser panorámicas muy grandes (>1 GP);
    desactivamos el guard de "decompression bomb" porque son legítimas.
    """
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # microscopía: imágenes legítimamente enormes
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception as e:  # pragma: no cover - defensive
        log.warning(f"[yellow]No se pudo leer header de {path.name}: {e}[/yellow]")
        return (None, None)


def _scan_subset(
    subset: dict,
    project_paths: ProjectPaths,
    schema: SchemaMap,
) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    """Devuelve (filas_images, dataframes_labels_por_imagen, issues)."""
    images_rows: list[dict] = []
    label_frames: list[pd.DataFrame] = []
    issues: list[dict] = []

    data_root = project_paths.data
    image_glob = subset["glob"]
    matches = sorted(data_root.glob(image_glob))

    log.info(
        f"subset [bold]{subset['name']}[/bold] (linaje {subset['lineage']}, split {subset['split']}): "
        f"{len(matches)} imágenes detectadas"
    )

    for image_path in tqdm(matches, desc=subset["name"], unit="img"):
        ann_path = _match_annotation(image_path)
        rel_image = image_path.relative_to(project_paths.root).as_posix()
        rel_ann = ann_path.relative_to(project_paths.root).as_posix() if ann_path else None

        width, height = _image_dimensions(image_path)
        row = {
            "subset": subset["name"],
            "lineage": subset["lineage"],
            "split": subset["split"],
            "image_path": rel_image,
            "annotation_path": rel_ann,
            "image_stem": image_path.stem,
            "width": width,
            "height": height,
        }

        if ann_path is None:
            row["status"] = "missing_annotation"
            images_rows.append(row)
            issues.append({"subset": subset["name"], "image": rel_image, "type": "missing_annotation"})
            continue

        try:
            df = pd.read_csv(ann_path)
        except Exception as e:
            row["status"] = "csv_read_error"
            issues.append({"subset": subset["name"], "image": rel_image, "type": "csv_read_error", "detail": str(e)})
            images_rows.append(row)
            continue

        if "row" not in df.columns or "col" not in df.columns:
            row["status"] = "csv_schema_error"
            issues.append(
                {"subset": subset["name"], "image": rel_image, "type": "csv_schema_error", "detail": "no row/col"}
            )
            images_rows.append(row)
            continue

        known = schema.known_columns()
        unknown_cols = [c for c in df.columns if c not in {"row", "col"} | known and not c.startswith("Question")]
        if unknown_cols:
            issues.append(
                {
                    "subset": subset["name"],
                    "image": rel_image,
                    "type": "unknown_columns",
                    "detail": ",".join(unknown_cols),
                }
            )

        n_tiles = len(df)
        unique = df[["row", "col"]].drop_duplicates()
        if len(unique) != n_tiles:
            issues.append(
                {
                    "subset": subset["name"],
                    "image": rel_image,
                    "type": "duplicate_tiles",
                    "detail": f"{n_tiles - len(unique)} duplicados",
                }
            )

        label_class_cols = [c for c in df.columns if c in schema.columns]
        binary = df[label_class_cols].fillna(0).astype(int)

        stage1_vals: list[str] = []
        stage2_vals: list[Optional[str]] = []
        aux_vals: list[Optional[str]] = []
        for _, r in binary.iterrows():
            s1, s2, aux = schema.reduce_row(r.to_dict())
            stage1_vals.append(s1)
            stage2_vals.append(s2)
            aux_vals.append(aux)

        labels_df = pd.DataFrame(
            {
                "image_path": rel_image,
                "row": df["row"].astype(int).values,
                "col": df["col"].astype(int).values,
                "stage1": stage1_vals,
                "stage2": stage2_vals,
                "aux_class": aux_vals,
                "lineage": subset["lineage"],
                "subset": subset["name"],
                "split": subset["split"],
            }
        )

        row["status"] = "ok"
        row["n_tiles"] = n_tiles
        row["n_mplus"] = int((labels_df["stage1"] == "Mplus").sum())
        row["n_mminus"] = int((labels_df["stage1"] == "Mminus").sum())
        row["n_background"] = int((labels_df["stage1"] == "Background").sum())
        row["n_unreadable"] = int((labels_df["stage1"] == "Unreadable").sum())
        images_rows.append(row)
        label_frames.append(labels_df)

    return images_rows, label_frames, issues


def _write_report(report: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Fase A — Reporte de ingesta\n")
    lines.append(f"- Generado: `{pd.Timestamp.utcnow().isoformat()}Z`")
    lines.append(f"- Esquema versión: `{report['schema_version']}`")
    lines.append(f"- Imágenes detectadas: **{report['n_images']}** (válidas: **{report['n_images_ok']}**)")
    lines.append(f"- Tiles totales (con anotación): **{report['n_tiles']}**\n")

    lines.append("## Distribución Stage1 por linaje\n")
    lines.append("| Linaje | M+ | M- | Background | Unreadable | Total |")
    lines.append("|--------|----|----|------------|------------|-------|")
    for lin, d in report["by_lineage"].items():
        lines.append(
            f"| {lin} | {d['Mplus']} | {d['Mminus']} | {d['Background']} | {d['Unreadable']} | {d['total']} |"
        )

    lines.append("\n## Distribución Stage2 por linaje\n")
    for lin, classes in report["stage2_by_lineage"].items():
        if not classes:
            continue
        lines.append(f"\n### {lin}\n")
        lines.append("| Subclase | n_tiles |")
        lines.append("|----------|---------|")
        for cls, n in classes.items():
            lines.append(f"| {cls} | {n} |")

    if report.get("issues"):
        lines.append("\n## Issues detectados\n")
        lines.append("| Subset | Imagen | Tipo | Detalle |")
        lines.append("|--------|--------|------|---------|")
        for it in report["issues"][:200]:
            lines.append(
                f"| {it['subset']} | {Path(it['image']).name} | {it['type']} | {it.get('detail','')} |"
            )
        if len(report["issues"]) > 200:
            lines.append(f"\n_({len(report['issues']) - 200} issues adicionales no listadas)_")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifests(
    data_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    datasets_config_path: Optional[Path] = None,
    schema_map_path: Optional[Path] = None,
) -> IngestResult:
    paths = get_paths()
    out_dir = out_dir or paths.manifests
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_config_path = datasets_config_path or paths.configs / "datasets.yaml"
    schema_map_path = schema_map_path or paths.configs / "schema_map.yaml"

    cfg = _load_datasets_config(datasets_config_path)
    schema = load_schema_map(schema_map_path)

    if data_root is not None:
        # Reasigna el data root del ProjectPaths para esta corrida.
        paths = ProjectPaths(
            root=paths.root,
            data=Path(data_root).resolve(),
            docs=paths.docs,
            configs=paths.configs,
            manifests=paths.manifests,
            outputs=paths.outputs,
            src=paths.src,
        )

    all_images: list[dict] = []
    all_labels: list[pd.DataFrame] = []
    all_issues: list[dict] = []

    for subset in cfg.get("subsets", []):
        imgs, labels, issues = _scan_subset(subset, paths, schema)
        all_images.extend(imgs)
        all_labels.extend(labels)
        all_issues.extend(issues)

    df_images = pd.DataFrame(all_images)
    df_labels = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame(
        columns=["image_path", "row", "col", "stage1", "stage2", "aux_class", "lineage", "subset", "split"]
    )

    report: dict = {
        "schema_version": schema.version,
        "n_images": int(len(df_images)),
        "n_images_ok": int((df_images.get("status") == "ok").sum()) if "status" in df_images else 0,
        "n_tiles": int(len(df_labels)),
        "by_lineage": {},
        "stage2_by_lineage": {},
        "issues": all_issues,
    }

    if not df_labels.empty:
        for lin, sub in df_labels.groupby("lineage"):
            counts = sub["stage1"].value_counts().to_dict()
            report["by_lineage"][lin] = {
                "Mplus": int(counts.get("Mplus", 0)),
                "Mminus": int(counts.get("Mminus", 0)),
                "Background": int(counts.get("Background", 0)),
                "Unreadable": int(counts.get("Unreadable", 0)),
                "total": int(len(sub)),
            }
            stage2_counts = (
                sub.loc[sub["stage1"] == "Mplus", "stage2"].dropna().value_counts().to_dict()
            )
            report["stage2_by_lineage"][lin] = {k: int(v) for k, v in stage2_counts.items()}

    images_path = write_table(df_images, out_dir / "manifest_images")
    labels_path = write_table(df_labels, out_dir / "manifest_labels")
    report_path = out_dir / "ingest_report.md"
    _write_report(report, report_path)

    log.info(f"[green]Manifests escritos en:[/green] {out_dir}")
    log.info(f"  images  -> {images_path.name} ({len(df_images)} filas)")
    log.info(f"  labels  -> {labels_path.name} ({len(df_labels)} filas)")
    log.info(f"  report  -> {report_path.name}")

    return IngestResult(
        images=df_images,
        labels=df_labels,
        report=report,
        images_path=images_path,
        labels_path=labels_path,
        report_path=report_path,
    )
