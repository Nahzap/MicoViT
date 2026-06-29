"""Artefactos de entrenamiento gate AM: gráficos, tablas y mapas gold vs pred."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..common.io import write_table
from ..common.logging_utils import get_logger
from ..common.run_outputs import RunOutputs
from ..layers import LayerContext, compose, downscale_context, render_layer, save_png
from .gate_classes import GATE_CLASS_NAMES, encode_gate_indices, gate_label_for_visual, is_valid_stage1, stage1_to_gate_label

log = get_logger("phase_d.gate_report")


def _stage1_to_gate_name(stage1: str) -> str:
    try:
        return gate_label_for_visual(stage1_to_gate_label(str(stage1)))
    except ValueError:
        return str(stage1)


def plot_training_curves(history_all: dict[str, dict], out_dir: Path) -> list[Path]:
    """Gráficos loss/acc/F1 por rama y panel combinado."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("train_loss", "Train loss"),
        ("val_acc", "Val accuracy"),
        ("val_f1", "Val macro F1"),
    ]
    for branch, hist in history_all.items():
        epochs = hist.get("epochs", [])
        for ax, (key, title) in zip(axes, metrics):
            ys = hist.get(key, [])
            if epochs and ys:
                ax.plot(epochs, ys, marker="o", label=f"Branch {branch}")
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    combined = out_dir / "train_curves_all_branches.png"
    fig.savefig(combined, dpi=150)
    plt.close(fig)
    saved.append(combined)

    for branch, hist in history_all.items():
        epochs = hist.get("epochs", [])
        if not epochs:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, marker in (("train_loss", "o"), ("val_loss", "s"), ("val_acc", "^")):
            ys = hist.get(key, [])
            if ys and len(ys) == len(epochs):
                ax.plot(epochs, ys, label=key, marker=marker)
        ax.set_xlabel("Epoch")
        ax.set_title(f"Branch {branch}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"train_curves_branch_{branch}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GATE_CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GATE_CLASS_NAMES,
        yticklabels=GATE_CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Gold")
    ax.set_title("Matriz de confusión — val (ensemble)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_class_recall_bars(per_class_recall: dict[str, float], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    names = [n for n in per_class_recall if not np.isnan(per_class_recall.get(n, float("nan")))]
    vals = [float(per_class_recall[n]) for n in names]
    if not names:
        names = list(per_class_recall.keys())
        vals = [0.0] * len(names)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#666666", "#D2B48C", "#00C8FF", "#A050C8"][: len(names)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (sensibilidad)")
    ax.set_title("Recall por clase — val ensemble")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _hstack_images(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + gap + right.shape[1]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] + gap :] = right
    return out


def _hstack_many(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("images vacio")
    out = images[0]
    for img in images[1:]:
        out = _hstack_images(out, img, gap=gap)
    return out


def render_gate_image_maps(
    *,
    image_path: Path,
    tiles_gold: pd.DataFrame,
    tiles_pred: pd.DataFrame,
    maps_dir: Path,
    device,
    tile_size: int = 252,
    downscale: int = 4,
    image_stem: Optional[str] = None,
    decode_cpu: bool = True,
) -> dict[str, Path]:
    """Mapas L0+L1 grilla, gold L2, pred L2, diff y panel comparativo.

    decode_cpu=True: PIL en RAM (rapido, sin VRAM). Predicciones ya vienen del cache.
    """
    stem = image_stem or image_path.stem
    maps_dir.mkdir(parents=True, exist_ok=True)

    if decode_cpu:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        img_np = np.array(Image.open(image_path).convert("RGB"))
    else:
        import torch
        from ..phase_b_tiling.gpu_io import decode_jpeg_gpu

        gimg = decode_jpeg_gpu(image_path, device=torch.device(device))
        img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        del gimg
        import torch as _torch

        _torch.cuda.empty_cache()

    gold = tiles_gold.copy()
    gold["stage1"] = gold["stage1"].map(_stage1_to_gate_name)
    pred = tiles_pred.copy()
    pred["stage1"] = pred["stage1_pred"].astype(str)

    ctx_base = LayerContext(image=img_np, tile_size=tile_size)
    ctx_gold = LayerContext(image=img_np, tile_size=tile_size, tiles=gold)
    ctx_pred = LayerContext(image=img_np, tile_size=tile_size, tiles=pred)
    if "consensus" not in pred.columns:
        l0_l6_pred = None

    ctx_base = downscale_context(ctx_base, downscale)
    ctx_gold = downscale_context(ctx_gold, downscale)
    ctx_pred = downscale_context(ctx_pred, downscale)

    l0_only = compose(["L0"], ctx_base, alphas=[1.0])
    l0_l1 = compose(["L0", "L1"], ctx_base, alphas=[1.0, 0.85])
    l0_l2_gold = compose(["L0", "L2"], ctx_gold, alphas=[1.0, 0.55])
    l0_l2_pred = compose(["L0", "L2"], ctx_pred, alphas=[1.0, 0.55])
    l0_l6_pred = compose(["L0", "L6"], ctx_pred, alphas=[1.0, 0.5]) if "consensus" in pred.columns else None

    pred_cols = ["row", "col", "stage1_pred", "p_mplus", "p_mminus", "p_bg"]
    if "consensus" in pred.columns:
        pred_cols.append("consensus")
    merged = gold[["row", "col", "stage1"]].merge(
        pred[pred_cols],
        on=["row", "col"],
        how="inner",
    )
    merged["correct"] = merged["stage1"].astype(str) == merged["stage1_pred"].astype(str)
    wrong = merged.loc[~merged["correct"], ["row", "col"]].assign(stage1="Unreadable")
    ctx_wrong = LayerContext(image=img_np, tile_size=tile_size, tiles=wrong)
    ctx_wrong = downscale_context(ctx_wrong, downscale)
    l0_errors = compose(["L0", "L2"], ctx_wrong, alphas=[1.0, 0.65])

    compare = _hstack_images(l0_l2_gold, l0_l2_pred)
    audit_panel = _hstack_many([l0_only, l0_l2_gold, l0_l2_pred])

    paths = {
        "L0_L1_grid": save_png(l0_l1, maps_dir / f"{stem}__L0_L1_grid.png"),
        "L0_L2_gold": save_png(l0_l2_gold, maps_dir / f"{stem}__L0_L2_gold.png"),
        "L0_L2_pred": save_png(l0_l2_pred, maps_dir / f"{stem}__L0_L2_pred.png"),
        "L0_L2_errors": save_png(l0_errors, maps_dir / f"{stem}__L0_L2_errors.png"),
        "gold_vs_pred": save_png(compare, maps_dir / f"{stem}__gold_vs_pred.png"),
        "L0_gold_pred_audit": save_png(audit_panel, maps_dir / f"{stem}__L0_gold_pred_audit.png"),
    }
    if l0_l6_pred is not None:
        paths["L0_L6_pred"] = save_png(l0_l6_pred, maps_dir / f"{stem}__L0_L6_pred.png")
    return paths


def _pred_metrics_arrays(pred_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Filtra gold invalido/vacio antes de metricas sklearn."""
    valid = pred_df[pred_df["stage1_gold"].map(is_valid_stage1)].reset_index(drop=True)
    if valid.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), valid
    y_true = encode_gate_indices(valid["stage1_gold"].to_numpy())
    y_pred = valid["gate_pred_idx"].to_numpy()
    return y_true, y_pred, valid


def _classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int],
    target_names: list[str],
) -> dict:
    """classification_report sin warning cuando eval estratificada omite Unknown."""
    import warnings
    from sklearn.metrics import classification_report

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )


def evaluate_gate_ensemble_on_df(
    ensemble,
    tiles_df: pd.DataFrame,
    *,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Inferencia ensemble por imagen; conserva etiqueta gold en `stage1_gold`."""
    from .gate_multiclass import infer_image_gate_gpu

    paths_root = tiles_df["image_path"].unique()
    parts = []
    for rel in paths_root:
        sub = tiles_df[tiles_df["image_path"] == rel]
        full = Path(rel)
        if not full.is_absolute():
            from ..common.paths import get_paths

            full = get_paths().root / rel
        pred = infer_image_gate_gpu(full, ensemble, batch_size=batch_size)
        if pred.empty:
            continue
        gold_map = sub.set_index(["row", "col"])["stage1"]
        pred["stage1_gold"] = [
            str(gold_map.get((int(r), int(c)), "")) for r, c in zip(pred["row"], pred["col"])
        ]
        pred["correct"] = pred["stage1_gold"].astype(str) == pred["stage1_pred"].astype(str)
        parts.append(pred)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def save_epoch_metrics_table(history_all: dict[str, dict], out_path: Path) -> Path:
    rows = []
    for branch, hist in history_all.items():
        for i, ep in enumerate(hist.get("epochs", [])):
            rows.append(
                {
                    "branch": branch,
                    "epoch": ep,
                    "train_loss": hist["train_loss"][i] if i < len(hist.get("train_loss", [])) else None,
                    "val_loss": hist["val_loss"][i] if i < len(hist.get("val_loss", [])) else None,
                    "val_acc": hist["val_acc"][i] if i < len(hist.get("val_acc", [])) else None,
                    "val_f1": hist["val_f1"][i] if i < len(hist.get("val_f1", [])) else None,
                    "elapsed_s": hist["elapsed_s"][i] if i < len(hist.get("elapsed_s", [])) else None,
                }
            )
    df = pd.DataFrame(rows)
    return Path(write_table(df, out_path))


def copy_checkpoints_to_run(ckpt_dir: Path, run: RunOutputs, pattern: str = "gate_*_best.pt") -> list[str]:
    dest = run.root / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(ckpt_dir.glob(pattern)):
        dst = dest / src.name
        shutil.copy2(src, dst)
        copied.append(dst.name)
    return copied


def finalize_gate_am_train_run(
    *,
    run: RunOutputs,
    history_all: dict[str, dict],
    info_split: dict,
    ensemble,
    val_df: pd.DataFrame,
    val_image_paths: list[Path],
    backbone: str,
    train_config: dict[str, Any],
    device,
    batch_size: int = 16,
    max_vis_images: int = 5,
    ckpt_dir: Path,
    downscale: int = 4,
) -> Path:
    """Genera gráficos, CSV, mapas gold/pred y reporte markdown en `run.root`."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    maps_per_image = run.maps / "val_images"
    maps_per_image.mkdir(parents=True, exist_ok=True)

    plot_training_curves(history_all, run.maps)
    save_epoch_metrics_table(history_all, run.tables / "epoch_metrics")

    val_pred = evaluate_gate_ensemble_on_df(ensemble, val_df, batch_size=batch_size)
    write_table(val_pred, run.tables / "val_predictions_all_tiles")

    metrics_summary: dict[str, Any] = {"split_info": info_split, "train_config": train_config}
    if not val_pred.empty:
        y_true, y_pred, val_pred = _pred_metrics_arrays(val_pred)
        if len(y_true) == 0:
            log.warning("[Gate tile] val_pred sin gold valido; metricas omitidas")
        else:
            metrics_summary["val_ensemble"] = {
                "acc": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(macro_f1_score(y_true, y_pred)),
                "macro_f1_all_classes": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=list(range(len(GATE_CLASS_NAMES))),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "n_tiles": int(len(val_pred)),
                "n_correct": int(val_pred["correct"].sum()),
                "classification_report": _classification_report_dict(
                    y_true,
                    y_pred,
                    labels=list(range(len(GATE_CLASS_NAMES))),
                    target_names=list(GATE_CLASS_NAMES),
                ),
            }
            per_class = {}
            for idx, name in enumerate(GATE_CLASS_NAMES):
                mask = y_true == idx
                if mask.any():
                    per_class[name] = float((y_pred[mask] == idx).mean())
            metrics_summary["val_ensemble"]["per_class_recall"] = per_class

            plot_confusion_matrix(y_true, y_pred, run.maps / "confusion_matrix_val.png")
            plot_class_recall_bars(per_class, run.maps / "recall_by_class_val.png")

            per_image_rows = []
            for rel in val_pred["image_path"].unique():
                sub = val_pred[val_pred["image_path"] == rel]
                acc = float(sub["correct"].mean())
                per_image_rows.append(
                    {
                        "image_path": rel,
                        "n_tiles": len(sub),
                        "acc": acc,
                        "n_mplus_pred": int((sub["stage1_pred"] == "Mplus").sum()),
                        "n_mminus_pred": int((sub["stage1_pred"] == "Mminus").sum()),
                        "n_bg_pred": int((sub["stage1_pred"] == "Background").sum()),
                    }
                )
            write_table(pd.DataFrame(per_image_rows), run.tables / "val_per_image_summary")

    vis_paths: list[Path] = list(val_image_paths)[: max(1, max_vis_images)]
    map_manifest = []
    for img_path in vis_paths:
        from ..common.paths import get_paths

        rel = img_path.resolve().relative_to(get_paths().root).as_posix()
        gold_sub = val_df[val_df["image_path"] == rel]
        pred_sub = val_pred[val_pred["image_path"] == rel] if not val_pred.empty else pd.DataFrame()
        if gold_sub.empty or pred_sub.empty:
            continue
        ts = int(gold_sub["tile_size"].iloc[0]) if "tile_size" in gold_sub.columns else 252
        stem = img_path.stem
        write_table(pred_sub, run.tables / f"val_images/{stem}__gate_probs")
        rendered = render_gate_image_maps(
            image_path=img_path,
            tiles_gold=gold_sub,
            tiles_pred=pred_sub,
            maps_dir=maps_per_image,
            device=device,
            tile_size=ts,
            downscale=downscale,
            image_stem=stem,
        )
        map_manifest.append({"image": rel, "maps": {k: str(v.name) for k, v in rendered.items()}})

    copied_ckpts = copy_checkpoints_to_run(ckpt_dir, run)
    metrics_summary["checkpoints_in_run"] = copied_ckpts
    metrics_summary["history"] = history_all
    metrics_summary["backbone_a"] = backbone
    metrics_summary["map_manifest"] = map_manifest

    json_path = run.reports / "gate_am_train_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    md_lines = [
        "# Entrenamiento gate AM",
        "",
        f"- **run_id:** `{run.run_id}`",
        f"- **backbone A:** {backbone}",
        f"- **train images:** {info_split.get('n_train_images')} | **val/test images:** {info_split.get('n_val_images')}",
        "",
        "## Métricas val (ensemble A+B+C)",
    ]
    if "val_ensemble" in metrics_summary:
        ve = metrics_summary["val_ensemble"]
        md_lines += [
            f"- Accuracy: **{ve['acc']:.4f}**",
            f"- Macro F1: **{ve['macro_f1']:.4f}**",
            f"- Tiles correctos: **{ve['n_correct']} / {ve['n_tiles']}**",
            "",
            "### Recall por clase",
        ]
        for cls, rec in ve.get("per_class_recall", {}).items():
            md_lines.append(f"- {cls}: {rec:.4f}")
    md_lines += [
        "",
        "## Gráficos",
        "- `maps/train_curves_all_branches.png`",
        "- `maps/confusion_matrix_val.png`",
        "- `maps/recall_by_class_val.png`",
        "",
        "## Mapas val (gold vs pred)",
        "Por imagen en `maps/val_images/`:",
        "- `*__L0_L1_grid.png` — imagen + grilla de tiles",
        "- `*__L0_L2_gold.png` — anotaciones manuales (CSV)",
        "- `*__L0_L2_pred.png` — predicción del modelo",
        "- `*__gold_vs_pred.png` — comparación lado a lado",
        "- `*__L0_L2_errors.png` — tiles mal clasificados (morado)",
        "",
        "## Tablas",
        "- `tables/epoch_metrics.csv`",
        "- `tables/val_predictions_all_tiles.csv`",
        "- `tables/val_per_image_summary.csv`",
        "",
        "## Checkpoints (copia en esta corrida)",
        *[f"- `checkpoints/{c}`" for c in copied_ckpts],
    ]
    md_path = run.reports / "run_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    log.info(f"[Gate AM] Reporte completo -> {run.root}")
    return md_path


def finalize_gate_tile_dino_run(
    *,
    run: RunOutputs,
    history: dict,
    info_split: dict,
    gate,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_image_paths: list[Path],
    backbone: str,
    train_config: dict[str, Any],
    device,
    batch_size: int = 16,
    max_vis_images: Optional[int] = None,
    ckpt_dir: Path,
    downscale: int = 4,
    protocol: Optional[Any] = None,
    embed_store: Optional[Any] = None,
    report_from_cache: bool = True,
    vis_all_test: bool = True,
    render_maps: bool = True,
) -> Path:
    """Reporte post-entrenamiento. Por defecto evalua desde cache (rapido, sin JPEG masivo)."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from tqdm.auto import tqdm

    from .gate_explainability import (
        copy_checkpoint_live_artifacts,
        plot_class_specificity_bars,
        plot_g1_combined_bars,
        resolve_vis_image_paths,
        write_explainability_md,
        write_map_gallery_md,
    )

    from .gate_tile_dino import evaluate_gate_probe_on_df, evaluate_gate_tile_dino_on_df
    from .gate_training_protocol import GateTrainProtocol, compute_gate_metrics, format_g1_status, g1_metric_display, macro_f1_score

    protocol = protocol or GateTrainProtocol()

    calibration: Optional[dict] = history.get("calibration") if isinstance(history, dict) else None
    ckpt_path = ckpt_dir / "gate_tile_dino_best.pt"
    if calibration is None and ckpt_path.exists():
        import torch

        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        calibration = st.get("calibration")

    maps_per_image = run.maps / "test_images"
    maps_per_image.mkdir(parents=True, exist_ok=True)

    history_all = {"DINOv2": history}
    plot_training_curves(history_all, run.maps)
    save_epoch_metrics_table(history_all, run.tables / "epoch_metrics")

    metrics_summary: dict[str, Any] = {
        "split_info": info_split,
        "train_config": train_config,
        "pipeline": "dinov2_embed_cache+slice_ms",
    }
    calibration_extra: dict[str, Any] = {}

    if embed_store is not None and report_from_cache:
        log.info("[Gate tile] Eval final desde cache embeddings (~segundos)")
        proto = None
        slice_ms = protocol.loss_type == "slice_ms_only"
        if ckpt_path.exists() and slice_ms:
            import torch

            from .gate_metric_inference import ClassPrototypeBank

            st_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "prototype_bank" in st_ckpt and train_config.get("gate4", {}).get("enabled"):
                from .gate_metric_inference import prototype_bank_from_gate4

                g4_cfg = train_config.get("gate4", {})
                from types import SimpleNamespace

                g4_shim = SimpleNamespace(
                    embed_dim=int(g4_cfg.get("embed_dim", 128)),
                    proto_subcenters_max=int(g4_cfg.get("proto_subcenters") or 1),
                    proto_subcenters_per_class=tuple(g4_cfg.get("proto_subcenters_per_class") or ()),
                    domain_aware_subcenters=bool(g4_cfg.get("domain_aware_subcenters", False)),
                )
                proto = prototype_bank_from_gate4(
                    g4_shim, num_classes=len(GATE_CLASS_NAMES), device=device
                )
                proto.load_state_dict(st_ckpt["prototype_bank"])
        if calibration:
            log.info(
                f"[Gate tile] Aplicando calibracion T={calibration.get('temperature', 1.0):.3f} "
                f"bias={calibration.get('class_bias', [0, 0, 0])}"
            )
        test_pred = evaluate_gate_probe_on_df(
            gate.classifier,
            val_df,
            embed_store,
            device,
            batch_size=batch_size,
            calibration=calibration,
            prototype_bank=proto,
            slice_ms_only=slice_ms,
        )
        if calibration:
            from .gate_tile_dino import collect_probe_logits

            raw_logits, raw_labels, _, _, _ = collect_probe_logits(
                gate.classifier, val_df, embed_store, device, batch_size=batch_size
            )
            raw_g1 = compute_gate_metrics(raw_logits, raw_labels, protocol=protocol)
            calibration_extra = {
                "calibration": calibration,
                "test_holdout_raw": {
                    "macro_f1": raw_g1["macro_f1"],
                    "min_class_recall": raw_g1["min_class_recall"],
                    "evangelisti_g1_pass": raw_g1["evangelisti_g1_pass"],
                    "g1_status": format_g1_status(raw_g1, protocol),
                },
                "test_holdout_calibrated_val_split": history.get("calibrated_val_metrics"),
            }
    else:
        log.warning(
            "[Gate tile] Eval final on-the-fly (lento, mucha VRAM). "
            "Activa GATE_REPORT_FROM_CACHE=True con cache valida."
        )
        test_pred = evaluate_gate_tile_dino_on_df(gate, val_df, batch_size=batch_size)
    write_table(test_pred, run.tables / "test_predictions_all_tiles")
    metrics_summary.update(calibration_extra)
    if not test_pred.empty:
        test_pred_full = test_pred
        y_true, y_pred, test_pred_metrics = _pred_metrics_arrays(test_pred)
        if len(y_true) == 0:
            log.warning("[Gate tile] test_pred sin gold valido; metricas omitidas")
        else:
            n_cls = len(GATE_CLASS_NAMES)
            logits_onehot = np.zeros((len(y_pred), n_cls), dtype=np.float32)
            logits_onehot[np.arange(len(y_pred)), y_pred] = 1.0
            g1_metrics = compute_gate_metrics(logits_onehot, y_true, protocol=protocol)

            metrics_summary["test_holdout"] = {
                "acc": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(macro_f1_score(y_true, y_pred)),
                "balanced_accuracy": g1_metrics["balanced_accuracy"],
                "min_class_recall": g1_metrics["min_class_recall"],
                "min_class_specificity": g1_metrics["min_class_specificity"],
                "evangelisti_g1_pass": g1_metrics["evangelisti_g1_pass"],
                "evangelisti_g1_score": g1_metrics["evangelisti_g1_score"],
                "n_tiles": int(len(test_pred_metrics)),
                "n_correct": int(test_pred_metrics["correct"].sum()),
                "per_class_recall": g1_metrics["per_class_recall"],
                "per_class_specificity": g1_metrics["per_class_specificity"],
                "g1_status": format_g1_status(g1_metrics, protocol),
                "classification_report": _classification_report_dict(
                    y_true,
                    y_pred,
                    labels=list(range(len(GATE_CLASS_NAMES))),
                    target_names=list(GATE_CLASS_NAMES),
                ),
            }
            per_class = g1_metrics["per_class_recall"]

            plot_confusion_matrix(y_true, y_pred, run.maps / "confusion_matrix_test.png")
            plot_class_recall_bars(per_class, run.maps / "recall_by_class_test.png")
            per_spec = g1_metrics["per_class_specificity"]
            plot_class_specificity_bars(per_spec, run.maps / "specificity_by_class_test.png")
            plot_g1_combined_bars(
                g1_metrics["per_class_recall"],
                per_spec,
                protocol.recall_thresh,
                protocol.spec_thresh,
                run.maps / "g1_sens_spec_bars.png",
            )

            per_image_rows = []
            for rel in test_pred_full["image_path"].unique():
                sub = test_pred_full[test_pred_full["image_path"] == rel]
                per_image_rows.append(
                    {
                        "image_path": rel,
                        "n_tiles": len(sub),
                        "acc": float(sub["correct"].mean()),
                        "n_mplus_pred": int((sub["stage1_pred"] == "Mplus").sum()),
                        "n_mminus_pred": int((sub["stage1_pred"] == "Mminus").sum()),
                        "n_bg_pred": int((sub["stage1_pred"] == "Background").sum()),
                    }
                )
            write_table(pd.DataFrame(per_image_rows), run.tables / "test_per_image_summary")

    live_artifacts = copy_checkpoint_live_artifacts(ckpt_dir, run)

    vis_paths = resolve_vis_image_paths(
        list(val_image_paths),
        render_maps=render_maps,
        vis_all_test=vis_all_test,
        max_vis_images=max_vis_images,
    )
    map_manifest = []
    if vis_paths:
        log.info(
            f"[Gate tile] Generando mapas test: {len(vis_paths)} imagenes "
            f"(pred cache + decode CPU, ~segundos/imagen)"
        )
        from ..common.paths import get_paths

        paths_root = get_paths()
        for img_path in tqdm(vis_paths, desc="mapas test", unit="img"):
            rel = img_path.resolve().relative_to(paths_root.root).as_posix()
            gold_sub = val_df[val_df["image_path"] == rel]
            pred_sub = test_pred[test_pred["image_path"] == rel] if not test_pred.empty else pd.DataFrame()
            if gold_sub.empty or pred_sub.empty:
                continue
            ts = int(gold_sub["tile_size"].iloc[0]) if "tile_size" in gold_sub.columns else 252
            stem = img_path.stem
            write_table(pred_sub, run.tables / f"test_images/{stem}__gate_probs")
            try:
                rendered = render_gate_image_maps(
                    image_path=img_path,
                    tiles_gold=gold_sub,
                    tiles_pred=pred_sub,
                    maps_dir=maps_per_image,
                    device=device,
                    tile_size=ts,
                    downscale=downscale,
                    image_stem=stem,
                    decode_cpu=True,
                )
                map_manifest.append({"image": rel, "maps": {k: str(v.name) for k, v in rendered.items()}})
            except OSError as e:
                log.warning(f"[Gate tile] Mapa omitido {stem}: {e}")
            except Exception as e:
                log.warning(f"[Gate tile] Mapa fallo {stem}: {e}")
        write_map_gallery_md(map_manifest, run.reports / "map_gallery.md", run)
    else:
        log.info("[Gate tile] Mapas visuales omitidos (GATE_RENDER_MAPS=False)")

    copied_ckpts = copy_checkpoints_to_run(ckpt_dir, run, pattern="gate_tile_dino_best.pt")
    metrics_summary["checkpoints_in_run"] = copied_ckpts
    metrics_summary["history"] = history_all
    metrics_summary["backbone"] = backbone
    metrics_summary["map_manifest"] = map_manifest

    json_path = run.reports / "gate_tile_dino_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    md_lines = [
        "# Gate AM — DINOv2 mean-pool + Slice-MS",
        "",
        f"- **run_id:** `{run.run_id}`",
        f"- **backbone:** {backbone}",
        f"- **tiles:** 252 px (manifest `tiles_index.csv`)",
        f"- **train:** {info_split.get('n_train_images')} imágenes | **test holdout:** {info_split.get('n_val_images')} imágenes",
        "",
        "## Pipeline",
        "1. DINOv2 mean-pool sobre cache de embeddings",
        "2. Slice-MS probe → Background / M− / M+",
        "3. Métricas en split test fijo",
        "",
    ]
    if history.get("pretrain_baseline"):
        pb = history["pretrain_baseline"]
        md_lines += [
            "## Baseline pre-entrenamiento (ep 0, cabeza sin entrenar)",
            f"- Macro F1: **{pb.get('macro_f1', 0):.4f}**",
            f"- Balanced acc: **{pb.get('balanced_accuracy', 0):.4f}**",
            "",
        ]
    md_lines += [
        "## Métricas test (10 imágenes holdout)",
    ]
    if "test_holdout" in metrics_summary:
        th = metrics_summary["test_holdout"]
        g1_flag = "PASS" if th.get("evangelisti_g1_pass") else "FAIL"
        md_lines += [
            f"- Accuracy (referencia, dataset desbalanceado): **{th['acc']:.4f}**",
            f"- Macro F1 (checkpoint): **{th['macro_f1']:.4f}**",
            f"- Balanced accuracy: **{th.get('balanced_accuracy', 0):.4f}**",
            f"- Evangelisti G1: **{g1_flag}** (score={th.get('evangelisti_g1_score', 0):.4f})",
            f"- Tiles correctos: **{th['n_correct']} / {th['n_tiles']}**",
            "",
            "### Sensibilidad (recall) y especificidad G1",
        ]
        for cls in GATE_CLASS_NAMES:
            rec = th.get("per_class_recall", {}).get(cls, float("nan"))
            spec = th.get("per_class_specificity", {}).get(cls, float("nan"))
            rt = protocol.recall_thresh.get(cls, 0.0)
            st = protocol.spec_thresh.get(cls, 0.0)
            rec_txt, _ = g1_metric_display(rec, rt, decimals=4)
            spec_txt, _ = g1_metric_display(spec, st, decimals=4)
            md_lines.append(
                f"- {cls}: Sens={rec_txt} (umbral {rt:.2f}) | Spec={spec_txt} (umbral {st:.2f})"
            )
        md_lines.append("")
        md_lines.append(f"Estado G1: `{th.get('g1_status', '')}`")
    md_lines += [
        "",
        "## Gráficos",
        "- `maps/train_curves_all_branches.png`",
        "- `maps/training_live_curves.png`",
        "- `maps/confusion_matrix_test.png`",
        "- `maps/recall_by_class_test.png`",
        "- `maps/specificity_by_class_test.png`",
        "- `maps/g1_sens_spec_bars.png`",
        "",
        "## Explicabilidad",
        "- **`reports/EXPLICABILIDAD.md`** — guía interpretación + índice de artefactos",
        "- `reports/map_gallery.md` — enlaces a mapas por imagen test",
        "",
        "## Layout pre/post entrenamiento",
        "- `images/pre_training/` — baseline ep0, distribución clases, stats dataset",
        "- `images/post_training/val/` — curvas y validación por época",
        "- `images/post_training/test/fullimage/` — mapas por imagen holdout",
        "- `training_protocol.md`, `training_report_test.md`, `training_report_val.md`",
        "- `evaluation_metrics_summary_test.json`, `evaluation_metrics_summary_val.json`",
        "",
        "## Mapas test (gold vs pred)",
        f"En `maps/test_images/` ({len(map_manifest)} imágenes): grilla, gold, pred, errores, comparación.",
        "",
        "## Tablas",
        "- `tables/epoch_metrics.csv`",
        "- `tables/test_predictions_all_tiles.csv`",
        "- `tables/test_per_image_summary.csv`",
        "",
        "## Checkpoint",
        *[f"- `checkpoints/{c}`" for c in copied_ckpts],
    ]
    md_path = run.reports / "run_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    write_explainability_md(
        run=run,
        metrics_summary=metrics_summary,
        history=history,
        train_config=train_config,
        map_manifest=map_manifest,
        live_artifacts=live_artifacts,
        protocol=protocol,
    )

    validation_md = _write_am_gate_validation_doc(
        run=run,
        metrics_summary=metrics_summary,
        protocol=protocol,
        history=history,
    )
    if validation_md:
        metrics_summary["am_gate_validation_doc"] = str(validation_md)

    from .gate_run_layout import publish_gate_run_layout

    layout = publish_gate_run_layout(
        run=run,
        train_df=train_df,
        val_df=val_df,
        info_split=info_split,
        history=history,
        metrics_summary=metrics_summary,
        train_config=train_config,
        protocol=protocol,
        ckpt_dir=ckpt_dir,
        map_manifest=map_manifest,
        legacy_maps_dir=run.maps,
        legacy_test_images_dir=maps_per_image,
    )
    metrics_summary["run_layout"] = str(layout.images)

    log.info(f"[Gate tile] Reporte -> {run.root}")
    return md_path


def _write_am_gate_validation_doc(
    *,
    run: RunOutputs,
    metrics_summary: dict[str, Any],
    protocol: Any,
    history: dict,
) -> Optional[Path]:
    """Genera Docs/AM_GATE_VALIDATION.md con criterios Evangelisti G1."""
    from ..common.paths import get_paths
    from .gate_training_protocol import g1_metric_display

    th = metrics_summary.get("test_holdout")
    if not th:
        return None

    paths = get_paths()
    doc_path = paths.root / "Docs" / "AM_GATE_VALIDATION.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    best_ep = history.get("best_epoch", history.get("epochs", [-1])[-1] if history.get("epochs") else -1)
    ckpt_metric = history.get("checkpoint_metric", protocol.checkpoint_metric)
    best_score = history.get("best_val_auroc", 0.0)

    lines = [
        "# Validacion gate AM — protocolo Evangelisti G1",
        "",
        f"Ultima corrida: `{run.run_id}`",
        "",
        "## Criterios G1 (holdout am_test, 10 imagenes)",
        "",
        "| Clase | Sensibilidad (recall) | Especificidad |",
        "|-------|----------------------|---------------|",
    ]
    for cls in GATE_CLASS_NAMES:
        rt = protocol.recall_thresh.get(cls, 0.0)
        st = protocol.spec_thresh.get(cls, 0.0)
        lines.append(f"| {cls} | >= {rt:.0%} | >= {st:.0%} |")

    g1_flag = "PASS" if th.get("evangelisti_g1_pass") else "FAIL"
    lines += [
        "",
        "## Resultado holdout",
        "",
        f"- **Evangelisti G1:** {g1_flag}",
        f"- Macro F1: {th.get('macro_f1', 0):.4f}",
        f"- Balanced accuracy: {th.get('balanced_accuracy', 0):.4f}",
        f"- Accuracy (solo referencia; test ~83% Background): {th.get('acc', 0):.4f}",
        f"- Mejor checkpoint: ep {best_ep} por `{ckpt_metric}`={best_score:.4f}",
        "",
        "### Por clase",
        "",
        "| Clase | Sens | Spec | Cumple Sens | Cumple Spec |",
        "|-------|------|------|-------------|-------------|",
    ]
    for cls in GATE_CLASS_NAMES:
        rec = th.get("per_class_recall", {}).get(cls, float("nan"))
        spec = th.get("per_class_specificity", {}).get(cls, float("nan"))
        rt = protocol.recall_thresh.get(cls, 0.0)
        st = protocol.spec_thresh.get(cls, 0.0)
        rec_txt, ok_r = g1_metric_display(rec, rt)
        spec_txt, ok_s = g1_metric_display(spec, st)
        ok_r = "—" if ok_r == "N/A" else ("si" if ok_r == "OK" else "no")
        ok_s = "—" if ok_s == "N/A" else ("si" if ok_s == "OK" else "no")
        lines.append(f"| {cls} | {rec_txt} | {spec_txt} | {ok_r} | {ok_s} |")

    lines += [
        "",
        "## Protocolo de entrenamiento",
        "",
        f"- Balance: `{protocol.balance_mode}`",
        f"- Checkpoint: `{ckpt_metric}`",
        f"- Early stopping: paciencia {protocol.early_stop_patience} (min {protocol.min_epochs} epocas)",
        f"- Loss train ponderada: {protocol.use_class_weights_train}",
        f"- Loss eval sin pesos: {protocol.eval_unweighted_loss}",
        "",
        f"Reporte completo: `{run.root}`",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"[Gate tile] AM_GATE_VALIDATION -> {doc_path}")
    return doc_path
