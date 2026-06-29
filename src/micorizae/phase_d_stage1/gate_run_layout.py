"""Layout ordenado de artefactos de corrida gate AM (pre/post entrenamiento).

Estructura bajo ``outputs/<run_id>/``::

    checkpoints/
    config/config.json
    images/
        pre_training/          # baseline ep0, estadísticas dataset
        post_training/
            loss_curves.png|.csv
            val/                 # métricas por época (holdout en validación)
            test/                # evaluación final checkpoint + mapas
                fullimage/
    logs/
    training_metrics.csv
    training_protocol.md
    training_report_val.md
    training_report_test.md
    evaluation_metrics_summary_val.json
    evaluation_metrics_summary_test.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..common.io import write_table
from ..common.logging_utils import get_logger
from ..common.run_outputs import RunOutputs
from .gate_classes import GATE_CLASS_NAMES

log = get_logger("phase_d.gate_layout")


@dataclass(frozen=True)
class GateRunLayout:
    """Rutas canónicas de una corrida gate AM."""

    run_root: Path

    @property
    def checkpoints(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def config_dir(self) -> Path:
        return self.run_root / "config"

    @property
    def images(self) -> Path:
        return self.run_root / "images"

    @property
    def pre_training(self) -> Path:
        return self.images / "pre_training"

    @property
    def post_training(self) -> Path:
        return self.images / "post_training"

    @property
    def post_val(self) -> Path:
        return self.post_training / "val"

    @property
    def post_test(self) -> Path:
        return self.post_training / "test"

    @property
    def post_test_fullimage(self) -> Path:
        return self.post_test / "fullimage"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    def ensure(self) -> GateRunLayout:
        for p in (
            self.checkpoints,
            self.config_dir,
            self.pre_training,
            self.post_training,
            self.post_val,
            self.post_test,
            self.post_test_fullimage,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def layout_for_run(run: RunOutputs) -> GateRunLayout:
    return GateRunLayout(run.root).ensure()


def _copy_file(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg, legend_if_labeled

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    order = list(GATE_CLASS_NAMES)
    stage1_map = {"Unknown": "Unreadable"}

    def _counts(df: pd.DataFrame) -> dict[str, int]:
        if df.empty or "stage1" not in df.columns:
            return {c: 0 for c in order}
        vc = df["stage1"].value_counts()
        out: dict[str, int] = {}
        for c in order:
            key = stage1_map.get(c, c)
            out[c] = int(vc.get(key, vc.get(c, 0)))
        return out

    train_c = _counts(train_df)
    val_c = _counts(val_df)
    rows = []
    for c in order:
        rows.append({"class": c, "split": "train", "count": train_c[c]})
        rows.append({"class": c, "split": "val", "count": val_c[c]})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    x = range(len(order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], [train_c[c] for c in order], w, label="train", color="#4C78A8")
    ax.bar([i + w / 2 for i in x], [val_c[c] for c in order], w, label="val/test holdout", color="#F58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Tiles")
    ax.set_title("Distribución de clases gate — train vs holdout")
    legend_if_labeled(ax)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_train_val_balance(info_split: dict, out_png: Path) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    labels = ["train tiles", "val tiles", "train imgs", "val imgs"]
    vals = [
        info_split.get("n_train_tiles", 0),
        info_split.get("n_val_tiles", 0),
        info_split.get("n_train_images", 0),
        info_split.get("n_val_images", 0),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#72B7B2", "#E45756"])
    ax.set_title("Balance train / holdout")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_loss_curves(history: dict, out_png: Path, out_csv: Path) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg, legend_if_labeled

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    epochs = history.get("epochs", [])
    epoch_details = history.get("epoch_details") or []
    min_recalls: list = []
    macro_f1s: list = []
    if epoch_details:
        for ed in epoch_details:
            min_recalls.append(ed.get("min_class_recall"))
            macro_f1s.append(ed.get("checkpoint_score") or ed.get("macro_f1"))

    n = len(epochs)
    train_loss = list(history.get("train_loss", []))
    val_loss = list(history.get("val_loss", []))
    val_acc = list(history.get("val_acc", []))
    val_f1 = list(history.get("val_f1", []))

    def _pad(xs: list, fill=None):
        xs = list(xs)
        if len(xs) < n:
            xs.extend([fill] * (n - len(xs)))
        return xs[:n]

    train_loss = _pad(train_loss)
    val_loss = _pad(val_loss)
    val_acc = _pad(val_acc)
    val_f1 = _pad(val_f1)
    min_recalls = _pad(min_recalls)
    macro_f1s = _pad(macro_f1s)

    rows = []
    for i, ep in enumerate(epochs):
        rows.append(
            {
                "epoch": ep,
                "train_loss": train_loss[i],
                "val_loss": val_loss[i],
                "val_acc": val_acc[i],
                "val_macro_f1": val_f1[i],
                "min_class_recall": min_recalls[i],
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if epochs:
        axes[0].plot(epochs, train_loss, marker="o", label="Slice-MS train", color="#4C78A8")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MS loss (~1.2–1.4 normal)")
        axes[0].set_title("Loss entrenamiento (Slice-MS)")
        legend_if_labeled(axes[0])
        axes[0].grid(True, alpha=0.3)

        if any(x is not None for x in min_recalls):
            axes[1].plot(epochs, min_recalls, marker="^", label="min_class_recall (ckpt)", color="#E45756")
            if any(x is not None for x in macro_f1s):
                axes[1].plot(epochs, macro_f1s, marker="d", label="checkpoint score", color="#72B7B2", alpha=0.8)
        else:
            axes[1].plot(epochs, val_f1, marker="^", label="macro F1", color="#E45756")
            axes[1].plot(epochs, val_acc, marker="d", label="accuracy", color="#72B7B2")
        axes[1].axhline(0.90, ls="--", color="gray", alpha=0.4, label="G1 0.90")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Recall / score")
        axes[1].set_title("Métrica checkpoint (eval stratified)")
        legend_if_labeled(axes[1], fontsize=8)
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_recall_curves(epoch_details: list[dict], out_png: Path, out_csv: Path) -> None:
    """Curvas Sens M−, M+, min_class_recall por época (Fase 6.3)."""
    from .gate_pretrain_viz import ensure_matplotlib_agg, legend_if_labeled

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    if not epoch_details:
        return
    rows = []
    for ed in epoch_details:
        rec = ed.get("per_class_recall") or {}
        if isinstance(rec, str):
            import ast

            rec = ast.literal_eval(rec)
        rows.append(
            {
                "epoch": ed.get("epoch"),
                "min_class_recall": ed.get("min_class_recall"),
                "recall_Mminus": rec.get("Mminus"),
                "recall_Mplus": rec.get("Mplus"),
                "recall_Background": rec.get("Background"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    if not df.empty:
        ax.plot(df["epoch"], df["recall_Mminus"], marker="o", label="Sens M−", color="#F58518")
        ax.plot(df["epoch"], df["recall_Mplus"], marker="s", label="Sens M+", color="#E45756")
        ax.plot(df["epoch"], df["min_class_recall"], marker="^", label="min_class_recall", color="#4C78A8")
        ax.axhline(0.90, ls="--", color="gray", alpha=0.5, label="G1 umbral 0.90")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.set_title("Recall por clase (checkpoint eval)")
        legend_if_labeled(ax, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_training_protocol_md(
    layout: GateRunLayout,
    *,
    train_config: dict[str, Any],
    protocol: Any,
    info_split: dict,
) -> Path:
    lines = [
        "# Training Protocol",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Split Strategy",
        "",
        f"- Train tiles: {info_split.get('n_train_tiles', '?')} ({info_split.get('n_train_images', '?')} imágenes)",
        f"- Val/test holdout tiles: {info_split.get('n_val_tiles', '?')} ({info_split.get('n_val_images', '?')} imágenes)",
        f"- Split mode: {info_split.get('split_mode', 'fixed')}",
        "",
        "## Training Hyperparameters",
        "",
        f"- Epochs (max): {train_config.get('epochs_max', '?')}",
        f"- Batch size: {train_config.get('batch_size', '?')}",
        f"- Balance mode: {train_config.get('balance_mode', '?')}",
        f"- Loss: {train_config.get('loss_type', '?')}",
        f"- Checkpoint metric: {train_config.get('checkpoint_metric', '?')}",
        f"- Early stop patience: {train_config.get('early_stop_patience', '?')}",
        f"- Probe / freeze backbone: {train_config.get('freeze_backbone', '?')}",
        f"- Embed cache: {train_config.get('embed_cache', '?')}",
        f"- DINO input size: {train_config.get('dino_input_size', '?')}",
        "",
        "## Gate4 Slice MS (si activo)",
        "",
    ]
    g4 = train_config.get("gate4") or {}
    if g4:
        lines += [
            f"- Enabled: {g4.get('enabled', False)}",
            f"- Slices: {g4.get('num_slices', '?')}",
            f"- Embed dim: {g4.get('embed_dim', '?')}",
            f"- MS weight: {g4.get('loss_weight', '?')}",
            "",
        ]
    else:
        lines += ["- No configurado en esta corrida.", ""]
    lines += [
        "## Reproducibility",
        "",
        f"- Run id: `{layout.run_root.name}`",
        f"- Config snapshot: `config/config.json`",
        "",
    ]
    path = layout.run_root / "training_protocol.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _metrics_block_md(title: str, metrics: dict[str, Any]) -> list[str]:
    if not metrics:
        return [f"## {title}", "", "_No disponible._", ""]
    lines = [f"## {title}", "", "| Metric | Value |", "|--------|-------|"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def write_training_report_md(
    layout: GateRunLayout,
    *,
    split: str,
    metrics: dict[str, Any],
    history: dict,
    protocol: Any,
) -> Path:
    """Informe markdown por split (`val` = mejor época; `test` = eval final)."""
    best_ep = history.get("best_epoch", "?")
    ckpt_metric = history.get("checkpoint_metric", protocol.checkpoint_metric)
    best_score = history.get("best_val_auroc", 0.0)
    elapsed = sum(history.get("elapsed_s") or [])
    lines = [
        f"# Training Results Report — {split}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total training time:** {elapsed:.1f} s",
        f"**Epochs run:** {len(history.get('epochs', []))}",
        "",
        "## Best Model Performance",
        "",
        f"Checkpoint seleccionado por **`{ckpt_metric}`**.",
        "",
        f"| Best epoch | {best_ep} |",
        f"| Best {ckpt_metric} | {best_score:.4f} |",
        "",
    ]
    lines += _metrics_block_md("Holdout metrics", metrics)
    if metrics.get("per_class_recall"):
        lines += ["## Per-class recall (sensibilidad)", ""]
        lines += ["| Class | Recall |", "|-------|--------|"]
        for cls in GATE_CLASS_NAMES:
            v = metrics["per_class_recall"].get(cls)
            cell = f"{float(v):.4f}" if v is not None and v == v else "—"
            lines.append(f"| {cls} | {cell} |")
        lines.append("")
    if metrics.get("per_class_specificity"):
        lines += ["## Per-class specificity", ""]
        lines += ["| Class | Specificity |", "|-------|-------------|"]
        for cls in GATE_CLASS_NAMES:
            v = metrics["per_class_specificity"].get(cls)
            cell = f"{float(v):.4f}" if v is not None and v == v else "—"
            lines.append(f"| {cls} | {cell} |")
        lines.append("")
    if split == "test":
        lines += [
            "## Visual artifacts",
            "",
            "- `images/post_training/test/fullimage/` — mapas por imagen (L0, gold, pred, errores)",
            "- `images/post_training/test/confusion_matrix.png`",
            "- `images/post_training/test/recall_by_class.png`",
            "",
        ]
    else:
        lines += [
            "## Visual artifacts",
            "",
            "- `images/post_training/loss_curves.png`",
            "- `images/post_training/val/` — curvas y métricas de validación por época",
            "",
        ]
    path = layout.run_root / f"training_report_{split}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_evaluation_summary(
    *,
    prefix: str,
    run_dir: Path,
    metrics: dict[str, Any],
    protocol: Any,
) -> dict[str, Any]:
    th = metrics or {}
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "prefix": prefix,
        "gate_protocol": {
            "checkpoint_metric": getattr(protocol, "checkpoint_metric", "macro_f1"),
            "recall_thresh": getattr(protocol, "recall_thresh", {}),
            "spec_thresh": getattr(protocol, "spec_thresh", {}),
        },
        "classification": {
            "accuracy": th.get("acc"),
            "balanced_accuracy": th.get("balanced_accuracy"),
            "macro_f1": th.get("macro_f1"),
            "min_class_recall": th.get("min_class_recall"),
            "min_class_specificity": th.get("min_class_specificity"),
            "evangelisti_g1_pass": th.get("evangelisti_g1_pass"),
            "evangelisti_g1_score": th.get("evangelisti_g1_score"),
            "n_tiles": th.get("n_tiles"),
            "n_correct": th.get("n_correct"),
        },
        "per_class": {
            "recall": th.get("per_class_recall", {}),
            "specificity": th.get("per_class_specificity", {}),
        },
    }


def _best_val_metrics_from_history(history: dict) -> dict[str, Any]:
    """Métricas del holdout en la mejor época (validación durante train)."""
    best_ep = history.get("best_epoch")
    details = history.get("epoch_details") or []
    for row in details:
        if row.get("epoch") == best_ep:
            return {
                "epoch": best_ep,
                "balanced_accuracy": row.get("balanced_accuracy"),
                "min_class_recall": row.get("min_class_recall"),
                "min_class_specificity": row.get("min_class_specificity"),
                "evangelisti_g1_pass": row.get("evangelisti_g1_pass"),
                "per_class_recall": row.get("per_class_recall"),
                "per_class_specificity": row.get("per_class_specificity"),
                "macro_f1": history.get("val_f1", [None])[history.get("epochs", []).index(best_ep)]
                if best_ep in history.get("epochs", [])
                else None,
            }
    calibrated = history.get("calibrated_val_metrics")
    if calibrated:
        return dict(calibrated)
    pb = history.get("pretrain_baseline") or {}
    return {
        "note": "fallback_pretrain_or_empty",
        "macro_f1": pb.get("macro_f1"),
        "balanced_accuracy": pb.get("balanced_accuracy"),
        "per_class_recall": pb.get("per_class_recall"),
        "per_class_specificity": pb.get("per_class_specificity"),
    }


def publish_gate_run_layout(
    *,
    run: RunOutputs,
    layout: Optional[GateRunLayout] = None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    info_split: dict,
    history: dict,
    metrics_summary: dict[str, Any],
    train_config: dict[str, Any],
    protocol: Any,
    ckpt_dir: Optional[Path] = None,
    map_manifest: Optional[list[dict]] = None,
    legacy_maps_dir: Optional[Path] = None,
    legacy_test_images_dir: Optional[Path] = None,
) -> GateRunLayout:
    """Publica artefactos en layout pre/post train. Conserva rutas legacy en paralelo."""
    layout = layout or layout_for_run(run)
    map_manifest = map_manifest or []

    # --- config ---
    cfg_payload = dict(train_config)
    cfg_payload["gate4"] = train_config.get("gate4") or {}
    cfg_payload["class_names"] = list(GATE_CLASS_NAMES)
    _write_json(layout.config_dir / "config.json", cfg_payload)

    # --- pre_training ---
    plot_class_distribution(
        train_df,
        val_df,
        layout.pre_training / "class_distribution.png",
        layout.pre_training / "class_distribution.csv",
    )
    plot_train_val_balance(info_split, layout.pre_training / "train_val_balance.png")
    _write_json(
        layout.pre_training / "dataset_stats.json",
        {
            "split_info": info_split,
            "train_stage1": train_df["stage1"].value_counts().to_dict() if "stage1" in train_df.columns else {},
            "val_stage1": val_df["stage1"].value_counts().to_dict() if "stage1" in val_df.columns else {},
        },
    )
    pb = history.get("pretrain_baseline") or {}
    if pb:
        _write_json(layout.pre_training / "baseline_metrics.json", pb)

    # --- post_training global curves ---
    plot_loss_curves(
        history,
        layout.post_training / "loss_curves.png",
        layout.post_training / "loss_curves.csv",
    )
    epoch_csv = run.tables / "epoch_metrics.csv"
    if epoch_csv.exists():
        shutil.copy2(epoch_csv, layout.run_root / "training_metrics.csv")
    else:
        plot_loss_curves(history, layout.post_training / "loss_curves.png", layout.run_root / "training_metrics.csv")

    # --- post_training / val ---
    _copy_file(layout.post_training / "loss_curves.png", layout.post_val / "loss_curves.png")
    _copy_file(layout.post_training / "loss_curves.csv", layout.post_val / "loss_curves.csv")
    if legacy_maps_dir:
        for name in ("train_curves_all_branches.png",):
            _copy_file(legacy_maps_dir / name, layout.post_val / name)

    val_metrics = _best_val_metrics_from_history(history)
    _write_json(layout.post_val / "validation_summary.json", val_metrics)
    val_summary = build_evaluation_summary(prefix="val", run_dir=run.root, metrics=val_metrics, protocol=protocol)
    _write_json(layout.run_root / "evaluation_metrics_summary_val.json", val_summary)
    write_training_report_md(layout, split="val", metrics=val_metrics, history=history, protocol=protocol)

    # --- post_training / test ---
    th = metrics_summary.get("test_holdout") or {}
    if th:
        from .gate_train_report import (
            plot_class_recall_bars,
            plot_confusion_matrix,
        )
        from .gate_explainability import plot_class_specificity_bars, plot_g1_combined_bars
        from .gate_classes import encode_gate_indices

        test_pred_path = run.tables / "test_predictions_all_tiles.parquet"
        if not test_pred_path.exists():
            test_pred_path = run.tables / "test_predictions_all_tiles.csv"
        if test_pred_path.exists():
            test_pred = (
                pd.read_parquet(test_pred_path)
                if test_pred_path.suffix == ".parquet"
                else pd.read_csv(test_pred_path)
            )
            if not test_pred.empty:
                y_true = encode_gate_indices(test_pred["stage1_gold"].to_numpy())
                y_pred = test_pred["gate_pred_idx"].to_numpy()
                plot_confusion_matrix(y_true, y_pred, layout.post_test / "confusion_matrix.png")
                plot_class_recall_bars(th.get("per_class_recall", {}), layout.post_test / "recall_by_class.png")
                plot_class_specificity_bars(
                    th.get("per_class_specificity", {}), layout.post_test / "specificity_by_class.png"
                )
                plot_g1_combined_bars(
                    th.get("per_class_recall", {}),
                    th.get("per_class_specificity", {}),
                    protocol.recall_thresh,
                    protocol.spec_thresh,
                    layout.post_test / "g1_sens_spec_bars.png",
                )

    for src_name, dst_name in (
        ("confusion_matrix_test.png", "confusion_matrix.png"),
        ("recall_by_class_test.png", "recall_by_class.png"),
        ("specificity_by_class_test.png", "specificity_by_class.png"),
        ("g1_sens_spec_bars.png", "g1_sens_spec_bars.png"),
        ("train_curves_all_branches.png", "train_curves.png"),
    ):
        if legacy_maps_dir:
            _copy_file(legacy_maps_dir / src_name, layout.post_test / dst_name)

    # Mapas por imagen → fullimage/
    src_img_dir = legacy_test_images_dir or (run.maps / "test_images")
    if src_img_dir.exists():
        for png in sorted(src_img_dir.glob("*.png")):
            _copy_file(png, layout.post_test_fullimage / png.name)

    test_summary = build_evaluation_summary(prefix="test", run_dir=run.root, metrics=th, protocol=protocol)
    _write_json(layout.post_test / "evaluation_metrics_summary.json", test_summary)
    _write_json(layout.run_root / "evaluation_metrics_summary_test.json", test_summary)
    write_training_report_md(layout, split="test", metrics=th, history=history, protocol=protocol)

    # --- logs ---
    for name in ("training_progress.json", "live_metrics.json"):
        _copy_file(run.reports / name, layout.logs / name)
        if ckpt_dir:
            _copy_file(ckpt_dir / name, layout.logs / name)

    write_training_protocol_md(layout, train_config=train_config, protocol=protocol, info_split=info_split)

    # Índice de mapas en test/fullimage
    index_lines = [
        "# Mapas por imagen — holdout test",
        "",
        f"Total imágenes: {len(map_manifest)}",
        "",
    ]
    for entry in map_manifest:
        rel = entry.get("image", "")
        maps = entry.get("maps", {})
        index_lines.append(f"## `{Path(rel).name}`")
        for key, fname in maps.items():
            index_lines.append(f"- **{key}:** `fullimage/{fname}`")
        index_lines.append("")
    (layout.post_test_fullimage / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    log.info(f"[Gate layout] Artefactos pre/post -> {layout.images}")
    return layout
