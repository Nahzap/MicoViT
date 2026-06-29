"""Evaluacion externa AMFinder bloqueada (post-entrenamiento gate AM v4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .common.io import write_table
from .common.logging_utils import get_logger
from .common.paths import get_paths
from .phase_d_stage1 import build_branch_a
from .phase_d_stage1.gate4.config import gate4_config_from_module
from .phase_d_stage1.gate_checkpoint_snapshot import _load_classifier_from_checkpoint
from .phase_d_stage1.gate_classes import GATE_CLASS_NAMES, encode_gate_indices
from .phase_d_stage1.gate_embed_cache import (
    GateEmbedStore,
    build_gate_embed_cache,
    cache_paths_with_basename,
)
from .phase_d_stage1.gate_tile_dino import (
    CHECKPOINT_NAME,
    collect_probe_logits,
    evaluate_gate_probe_on_df,
)
from .phase_d_stage1.gate_train_report import (
    plot_class_recall_bars,
    plot_confusion_matrix,
    render_gate_image_maps,
)
from .phase_d_stage1.gate_training_protocol import (
    GateTrainProtocol,
    compute_gate_metrics,
    macro_f1_score,
)

log = get_logger("gate_external_eval")

EXTERNAL_CACHE_BASENAME = "gate_amfinder_external_v1"


def _ensure_external_cache(
    external_df: pd.DataFrame,
    tiles_index_path: Path,
    *,
    cfg: Any,
    force_rebuild: bool = False,
) -> GateEmbedStore:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible para cache externa AMFinder.")

    paths = get_paths()
    cpaths = cache_paths_with_basename(paths.root, EXTERNAL_CACHE_BASENAME)
    if not force_rebuild and cpaths.meta.exists():
        try:
            store = GateEmbedStore(cpaths)
            store.indices_for_sub(external_df)
            return store
        except KeyError:
            log.info("[External] Cache incompleta; recompilando...")

    device = torch.device("cuda")
    backbone = build_branch_a(
        backbone_name=str(getattr(cfg, "GATE_BACKBONE", "dinov2_vits14")),
        num_classes=3,
        freeze_backbone=True,
    ).backbone
    build_gate_embed_cache(
        external_df,
        backbone,
        device=device,
        backbone_name=str(getattr(cfg, "GATE_BACKBONE", "dinov2_vits14")),
        dino_input_size=int(getattr(cfg, "GATE_DINO_INPUT_SIZE", 280)),
        seg_target_size=int(getattr(cfg, "GATE_SEG_TARGET_SIZE", 400)),
        batch_size=int(getattr(cfg, "GATE_CACHE_BATCH_SIZE", 20)),
        tiles_index_path=tiles_index_path,
        cache_paths=cpaths,
        force_rebuild=force_rebuild,
        dynamic_batch=bool(getattr(cfg, "GATE_CACHE_DYNAMIC_BATCH", True)),
        vram_budget_mb=float(getattr(cfg, "GATE_CACHE_VRAM_BUDGET_MB", 7500.0)),
        cpu_decode_above_mb=float(getattr(cfg, "GATE_CACHE_CPU_DECODE_ABOVE_MB", 500.0)),
        empty_cache_every_n_batches=int(getattr(cfg, "GATE_CACHE_EMPTY_CACHE_EVERY_N_BATCHES", 0)),
        gc_collect_every_n_batches=int(getattr(cfg, "GATE_CACHE_GC_COLLECT_EVERY_N_BATCHES", 0)),
        memmap_flush_every_n_images=int(getattr(cfg, "GATE_CACHE_MEMMAP_FLUSH_EVERY_N_IMAGES", 5)),
        mplus_aug_variants=(),
        cache_attention=bool(getattr(cfg, "GATE_CACHE_ATTENTION", False)),
        attention_layers=str(getattr(cfg, "GATE_CACHE_ATTENTION_LAYERS", "all")),
        attention_head_reduce=str(getattr(cfg, "GATE_CACHE_ATTENTION_HEAD_REDUCE", "mean")),
    )
    return GateEmbedStore(cpaths)


def run_amfinder_external_eval(
    *,
    external_df: pd.DataFrame,
    run_dir: Path,
    ckpt_dir: Path,
    cfg: Any,
    batch_size: int = 48,
    skip_maps: bool = False,
    downscale: int = 4,
) -> dict[str, Any]:
    """Evalua holdout amfinder_external y escribe artefactos bajo run_dir/external/."""
    import torch
    from sklearn.metrics import classification_report

    if external_df.empty:
        log.warning("[External] Sin tiles external_test; omitiendo eval.")
        return {}

    paths = get_paths()
    ext_root = run_dir / "external"
    maps_dir = ext_root / "maps"
    tables_dir = ext_root / "tables"
    metrics_dir = ext_root / "metrics"
    reports_dir = ext_root / "reports"
    for d in (maps_dir, tables_dir, metrics_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    ext_df = external_df.copy()
    if "tile_edge" not in ext_df.columns:
        ext_df["tile_edge"] = ext_df["tile_size"].astype(int)
    from .gate_domain_buckets import attach_domain_buckets

    ext_df = attach_domain_buckets(ext_df)

    tiles_index_path = write_table(ext_df, paths.manifests / "amfinder_external_tiles_index")
    embed_store = _ensure_external_cache(ext_df, tiles_index_path, cfg=cfg)

    device = torch.device("cuda")
    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    gate4 = gate4_config_from_module(cfg)
    protocol = GateTrainProtocol(
        loss_type=str(getattr(cfg, "GATE_LOSS", "slice_ms_only")),
        checkpoint_metric=str(getattr(cfg, "GATE_CHECKPOINT_METRIC", "min_class_recall")),
    )
    model, st, proto = _load_classifier_from_checkpoint(
        ckpt_path,
        embed_store=embed_store,
        device=device,
        gate4_config=gate4,
    )
    calibration = st.get("calibration")
    slice_ms = protocol.loss_type == "slice_ms_only"

    log.info(f"[External] Inferencia {len(ext_df):,} tiles AMFinder holdout...")
    pred = evaluate_gate_probe_on_df(
        model,
        ext_df,
        embed_store,
        device,
        batch_size=batch_size,
        calibration=calibration,
        prototype_bank=proto,
        slice_ms_only=slice_ms,
    )
    write_table(pred, tables_dir / "predictions_external_all_tiles")

    logits, labels_arr, _, _, _ = collect_probe_logits(
        model,
        ext_df,
        embed_store,
        device,
        batch_size=batch_size,
        prototype_bank=proto,
        slice_ms_only=slice_ms,
    )
    if calibration:
        from .phase_d_stage1.gate_calibrate import GateCalibration, apply_gate_calibration

        logits = apply_gate_calibration(logits, GateCalibration.from_dict(calibration))

    y_true = labels_arr
    y_pred = pred["gate_pred_idx"].to_numpy()
    g1 = compute_gate_metrics(logits, y_true, protocol=protocol)

    metrics_summary: dict[str, Any] = {
        "dataset": "amfinder_external_test",
        "checkpoint": str(ckpt_path),
        "n_images": int(ext_df["image_path"].nunique()),
        "n_tiles": len(pred),
        "acc": float(g1["acc"]),
        "macro_f1": float(g1["macro_f1"]),
        "min_class_recall": float(g1["min_class_recall"]),
        "per_class_recall": g1.get("per_class_recall", {}),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(GATE_CLASS_NAMES))),
            target_names=list(GATE_CLASS_NAMES),
            zero_division=0,
        ),
    }

    plot_confusion_matrix(y_true, y_pred, metrics_dir / "confusion_matrix_external.png")
    if g1.get("per_class_recall"):
        plot_class_recall_bars(
            g1["per_class_recall"], metrics_dir / "class_recall_bars_external.png"
        )

    by_edge: list[dict] = []
    pred_edge = pred.merge(
        ext_df[["image_path", "row", "col", "tile_edge"]].drop_duplicates(),
        on=["image_path", "row", "col"],
        how="left",
    )
    for edge, sub in pred_edge.groupby("tile_edge", sort=False):
        if sub.empty:
            continue
        yt = encode_gate_indices(sub["stage1_gold"].to_numpy())
        yp = sub["gate_pred_idx"].to_numpy()
        rec = {c: float((yp[yt == i] == i).mean()) if (yt == i).any() else 0.0 for i, c in enumerate(GATE_CLASS_NAMES)}
        by_edge.append(
            {
                "tile_edge": int(edge),
                "n_tiles": len(sub),
                "acc": float(sub["correct"].mean()),
                "macro_f1": float(macro_f1_score(yt, yp)),
                "per_class_recall": rec,
            }
        )
    metrics_summary["by_tile_edge"] = by_edge

    summary_path = ext_root / "evaluation_metrics_summary_external.json"
    summary_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Validacion externa AMFinder (holdout bloqueado)",
        "",
        f"- Tiles: **{len(pred):,}** ({metrics_summary['n_images']} imagenes)",
        f"- Accuracy: **{metrics_summary['acc']:.3f}**",
        f"- Macro F1: **{metrics_summary['macro_f1']:.3f}**",
        f"- Recall M+: **{metrics_summary['per_class_recall'].get('Mplus', 0):.3f}**",
        "",
        "## Por tile_edge",
        "",
        "| tile_edge | n_tiles | acc | macro_f1 | recall M+ |",
        "|-----------|---------|-----|----------|-----------|",
    ]
    for row in by_edge:
        rm = row["per_class_recall"].get("Mplus", 0)
        report_lines.append(
            f"| {row['tile_edge']} | {row['n_tiles']} | {row['acc']:.3f} | "
            f"{row['macro_f1']:.3f} | {rm:.3f} |"
        )
    (reports_dir / "EXTERNAL_VALIDATION_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    if not skip_maps:
        import torch

        for rel in sorted(ext_df["image_path"].unique())[:10]:
            sub = ext_df[ext_df["image_path"] == rel]
            img_path = paths.root / "Data" / rel
            if not img_path.exists():
                img_path = paths.root / rel
            if not img_path.exists():
                continue
            pred_sub = pred[pred["image_path"] == rel]
            ts = int(sub["tile_size"].iloc[0]) if "tile_size" in sub.columns else 252
            render_gate_image_maps(
                image_path=img_path,
                tiles_gold=sub,
                tiles_pred=pred_sub,
                maps_dir=maps_dir,
                device=torch.device("cpu"),
                tile_size=ts,
                downscale=downscale,
                image_stem=Path(rel).stem,
            )

    log.info(
        f"[External] OK macro_f1={metrics_summary['macro_f1']:.3f} "
        f"recall_M+={metrics_summary['per_class_recall'].get('Mplus', 0):.3f}"
    )
    return metrics_summary
