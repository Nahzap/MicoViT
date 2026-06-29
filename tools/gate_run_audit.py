#!/usr/bin/env python3
"""Auditoría de run Gate AM — genera RUN_AUDIT.md desde outputs/<run_id>/."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def audit_run(run_dir: Path) -> str:
    run_dir = run_dir.resolve()
    status = _read_json(run_dir / "training_status.json") or {}
    config = _read_json(run_dir / "config" / "config.json") or {}
    metrics_csv = run_dir / "tables" / "epoch_metrics.csv"
    progress = run_dir / "training_progress.json"
    best_ep = status.get("best_epoch")
    best_score = status.get("best_score")

    stale_note = ""
    if status.get("status") == "running" and progress.exists():
        import os
        import time

        age_s = time.time() - os.path.getmtime(progress)
        if age_s > 300:
            stale_note = f"\n\n⚠️ **Run stale:** sin cambios en progress >5 min ({age_s/60:.1f} min)\n"

    lines = [
        f"# RUN_AUDIT — {run_dir.name}",
        "",
        f"**Generado:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Run dir:** `{run_dir}`",
        "",
        "## Estado",
        "",
        f"| Campo | Valor |",
        f"|-------|-------|",
        f"| status | {status.get('status', 'unknown')} |",
        f"| run_id | {status.get('run_id', run_dir.name)} |",
        f"| best_epoch | {best_ep} |",
        f"| best_score ({status.get('checkpoint_metric', 'min_class_recall')}) | {best_score} |",
        f"| completed_at | {status.get('completed_at', '—')} |",
        "",
        "## Config clave",
        "",
        f"| Parámetro | Valor |",
        f"|-----------|-------|",
    ]
    for key in (
        "loss_type",
        "balance_mode",
        "checkpoint_metric",
        "checkpoint_eval",
        "gate4",
        "embed_cache",
        "run_id",
    ):
        if key in config:
            lines.append(f"| {key} | `{config[key]}` |")

    if metrics_csv.exists():
        import pandas as pd

        df = pd.read_csv(metrics_csv)
        lines.extend(["", "## Métricas", "", f"- Épocas registradas: **{len(df)}**"])
        if best_ep is not None and "epoch" in df.columns:
            row = df[df["epoch"] == int(best_ep)]
            if not row.empty:
                r = row.iloc[0]
                lines.append(f"- Best ep {best_ep}: min_class_recall={r.get('min_class_recall', '—')}")
                lines.append(f"- G1 pass: {r.get('evangelisti_g1_pass', '—')}")

    ckpt = ROOT / "models" / "checkpoints" / "gate_am" / "gate_tile_dino_best.pt"
    lines.extend(["", "## Checkpoint", "", f"- `{ckpt}` — exists={ckpt.exists()}"])

    pre = run_dir / "images" / "pre_training"
    ann = pre / "stratified_tile_samples_annotated.png"
    lines.extend(
        [
            "",
            "## Artefactos pre-training",
            "",
            f"- Viz anotada stratified: {'✅' if ann.exists() else '❌'} `{ann.name}`",
            f"- tile_manifest_stratified.csv: {'✅' if (pre / 'tile_manifest_stratified.csv').exists() else '❌'}",
            f"- INDEX.md: {'✅' if (pre / 'INDEX.md').exists() else '❌'}",
        ]
    )

    analysis = run_dir / "analysis" / "EMBED_REPORT.md"
    lines.extend(["", "## Análisis embeddings", "", f"- EMBED_REPORT: {'✅' if analysis.exists() else '❌ pendiente'}"])

    return "\n".join(lines) + stale_note + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría run Gate AM")
    parser.add_argument("--run", type=Path, required=True, help="Carpeta outputs/<run_id>")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Salida MD (default: <run>/reports/RUN_AUDIT.md)",
    )
    args = parser.parse_args()
    out = args.out or (args.run / "reports" / "RUN_AUDIT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    md = audit_run(args.run)
    out.write_text(md, encoding="utf-8")
    print(f"[gate_run_audit] -> {out}")


if __name__ == "__main__":
    main()
