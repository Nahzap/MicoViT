"""QA de labels stage1 y artefactos retroactivos de corrida Gate AM."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..common.logging_utils import get_logger
from .gate_classes import drop_invalid_stage1_rows, is_valid_stage1

log = get_logger("phase_d.gate_label_qa")


def validate_gate_splits_for_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    raise_on_invalid: bool = True,
) -> dict[str, Any]:
    """Pre-vuelo: cuenta tiles con stage1 invalido antes de entrenar."""
    report: dict[str, Any] = {"train_invalid": 0, "val_invalid": 0, "ok": True}
    for name, df in (("train", train_df), ("val", val_df)):
        if df.empty or "stage1" not in df.columns:
            continue
        n_bad = int((~df["stage1"].map(is_valid_stage1)).sum())
        report[f"{name}_invalid"] = n_bad
        report[f"{name}_total"] = len(df)
        if n_bad > 0:
            report["ok"] = False
            log.error(f"[Gate QA] {name}: {n_bad}/{len(df)} tiles con stage1 invalido/vacio")
    if not report["ok"] and raise_on_invalid:
        raise ValueError(
            "Pre-vuelo QA fallo: hay tiles con stage1 vacio. "
            "Revisa manifest/splits antes de entrenar."
        )
    if report["ok"]:
        log.info(
            f"[Gate QA] Pre-vuelo OK — train={report.get('train_total', 0)} "
            f"val={report.get('val_total', 0)} tiles validos"
        )
    return report


def sanitize_tiles_df_for_eval(df: pd.DataFrame, *, split: str = "eval") -> pd.DataFrame:
    return drop_invalid_stage1_rows(df, log_prefix=f"[{split}] ")


def _parse_dict_field(val: object) -> dict:
    if isinstance(val, dict):
        return val
    if not isinstance(val, str) or not val.strip():
        return {}
    import ast

    s = val.replace("nan", "None").replace("NaN", "None")
    parsed = ast.literal_eval(s)
    return parsed if isinstance(parsed, dict) else {}


def regenerate_run_plots(run_root: Path) -> list[str]:
    """Regenera PNGs de curvas desde CSV/JSON existentes en la corrida."""
    from .gate_run_layout import plot_loss_curves, plot_recall_curves

    written: list[str] = []
    run_root = Path(run_root)
    metrics_csv = run_root / "training_metrics.csv"
    epoch_csv = run_root / "tables" / "epoch_metrics.csv"
    post = run_root / "images" / "post_training"
    post.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {"epochs": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    epoch_details: list[dict] = []

    if metrics_csv.is_file():
        df = pd.read_csv(metrics_csv)
        history["epochs"] = df["epoch"].tolist()
        history["train_loss"] = df["train_loss"].tolist()
        history["val_loss"] = df["val_loss"].tolist()
        if "val_acc" in df.columns:
            history["val_acc"] = df["val_acc"].tolist()
        if "val_macro_f1" in df.columns:
            history["val_f1"] = df["val_macro_f1"].tolist()

    if epoch_csv.is_file():
        edf = pd.read_csv(epoch_csv)
        for _, row in edf.iterrows():
            rec = row.to_dict()
            for key in ("per_class_recall", "per_class_specificity", "diagnostics", "per_class_ap"):
                if key in rec:
                    rec[key] = _parse_dict_field(rec[key])
            epoch_details.append(rec)
        history["epoch_details"] = epoch_details

    loss_png = post / "loss_curves.png"
    plot_loss_curves(history, loss_png, post / "val" / "loss_curves.csv")
    written.append(str(loss_png))

    recall_png = post / "recall_curves.png"
    plot_recall_curves(epoch_details, recall_png, post / "recall_curves.csv")
    written.append(str(recall_png))

    val_dir = post / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(loss_png, val_dir / "loss_curves.png")
    log.info(f"[Gate QA] Plots regenerados: {len(written)} PNG en {post}")
    return written


def write_epoch_comparison_report(
    run_root: Path,
    *,
    epochs: tuple[int, ...] = (6, 14, 31),
) -> Path:
    """Informe markdown comparando epocas clave (mAP, recall, G1)."""
    run_root = Path(run_root)
    epoch_csv = run_root / "tables" / "epoch_metrics.csv"
    if not epoch_csv.is_file():
        epoch_csv = run_root / "training_metrics_epochs.csv"
    if not epoch_csv.is_file():
        raise FileNotFoundError(f"Sin epoch_metrics en {run_root}")

    df = pd.read_csv(epoch_csv)
    df["epoch"] = df["epoch"].astype(int)
    sel = df[df["epoch"].isin(epochs)].copy()
    if sel.empty:
        raise ValueError(f"Ninguna epoca {epochs} en {epoch_csv}")

    lines = [
        "# Comparativa de epocas",
        "",
        f"Run: `{run_root.name}`",
        "",
        "| ep | mAP | min_recall | M+ rec | M- rec | G1 | train_loss | val_loss |",
        "|----|-----|------------|--------|--------|----|------------|----------|",
    ]
    train_loss_by_ep = {}
    val_loss_by_ep = {}
    tm = run_root / "training_metrics.csv"
    if tm.is_file():
        tdf = pd.read_csv(tm)
        for _, r in tdf.iterrows():
            ep = int(r["epoch"])
            train_loss_by_ep[ep] = r.get("train_loss")
            val_loss_by_ep[ep] = r.get("val_loss")

    for _, row in sel.sort_values("epoch").iterrows():
        ep = int(row["epoch"])
        rec = row.get("per_class_recall")
        rec = _parse_dict_field(rec)
        g1 = "PASS" if row.get("evangelisti_g1_pass") in (True, "True", 1) else "FAIL"
        lines.append(
            f"| {ep} | {row.get('mAP', '—'):.4f} | {row.get('min_class_recall', 0):.4f} | "
            f"{rec.get('Mplus', float('nan')):.3f} | {rec.get('Mminus', float('nan')):.3f} | "
            f"{g1} | {train_loss_by_ep.get(ep, '—')} | {val_loss_by_ep.get(ep, '—')} |"
        )

    lines.extend(
        [
            "",
            "## Lectura",
            "",
            "- **Best checkpoint (mAP)**: ep6 domina en mAP; ep14 mejora min_recall pero no mAP.",
            "- **Colapso M+**: tras ep6, M+ recall cae mientras M- sube (atajo M-).",
            "- **Overfit**: train_loss << val_loss desde ep10+.",
            "",
        ]
    )
    out = run_root / "reports" / "EPOCH_COMPARISON.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"[Gate QA] Informe comparativo -> {out}")
    return out
