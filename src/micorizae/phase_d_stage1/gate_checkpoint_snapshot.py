"""Evaluación incremental del mejor checkpoint durante el entrenamiento.

Escribe matriz de confusión, CSV y barras en ``images/post_training/test/``
sin esperar al finalize final (evita carpetas vacías mientras corre el loop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..common.io import write_table
from ..common.logging_utils import get_logger
from .gate_classes import GATE_CLASS_NAMES, encode_gate_indices
from .gate_training_protocol import compute_gate_metrics, format_g1_status, macro_f1_score

log = get_logger("phase_d.gate_snapshot")


def _load_classifier_from_checkpoint(
    checkpoint: Path,
    *,
    embed_store: Any,
    device: Any,
    gate4_config: Any = None,
) -> Any:
    import torch

    from .gate4.probe_model import build_gate_slice_probe
    from .gate_probe_input import probe_in_dim_from_attention_meta

    st = torch.load(checkpoint, map_location=device, weights_only=False)
    in_dim = probe_in_dim_from_attention_meta(
        embed_store.meta.get("attention"), embed_store.embed_dim
    )
    proto = None
    if gate4_config is not None and getattr(gate4_config, "enabled", False):
        model = build_gate_slice_probe(
            in_dim=in_dim,
            embed_dim=gate4_config.embed_dim,
            num_slices=gate4_config.num_slices,
            num_classes=len(GATE_CLASS_NAMES),
        )
        if "prototype_bank" in st:
            from .gate_metric_inference import prototype_bank_from_gate4

            proto = prototype_bank_from_gate4(
                gate4_config, num_classes=len(GATE_CLASS_NAMES), device=device
            )
            proto.load_state_dict(st["prototype_bank"])
    else:
        from . import build_branch_a

        model = build_branch_a(
            backbone_name=st.get("backbone", "dinov2_vits14"),
            num_classes=len(GATE_CLASS_NAMES),
            freeze_backbone=True,
        )
    model.load_state_dict(st["model_state_dict"])
    return model.to(device).eval(), st, proto


def publish_eval_snapshot(
    *,
    run_root: Path,
    val_df: pd.DataFrame,
    embed_store: Any,
    checkpoint: Path,
    device: Any,
    protocol: Any,
    gate4_config: Any = None,
    best_epoch: int = 0,
    best_score: float = 0.0,
) -> Optional[Path]:
    """Eval holdout desde cache → PNG/CSV en post_training/test/."""
    from sklearn.metrics import accuracy_score, classification_report

    from .gate_run_layout import GateRunLayout
    from .gate_tile_dino import evaluate_gate_probe_on_df
    from .gate_train_report import (
        plot_class_recall_bars,
        plot_confusion_matrix,
    )
    from .gate_explainability import plot_class_specificity_bars, plot_g1_combined_bars

    checkpoint = Path(checkpoint)
    if not checkpoint.exists() or embed_store is None or val_df.empty:
        return None

    out_test = GateRunLayout(run_root).post_test
    out_test.mkdir(parents=True, exist_ok=True)

    try:
        model, st, proto = _load_classifier_from_checkpoint(
            checkpoint, embed_store=embed_store, device=device, gate4_config=gate4_config
        )
        calibration = st.get("calibration")
        slice_ms = getattr(protocol, "loss_type", "") == "slice_ms_only"
        test_pred = evaluate_gate_probe_on_df(
            model,
            val_df,
            embed_store,
            device,
            batch_size=48,
            calibration=calibration,
            prototype_bank=proto,
            slice_ms_only=slice_ms,
        )
    except Exception as e:
        log.warning(f"[Gate snapshot] eval skip: {e}")
        return None

    if test_pred.empty:
        return None

    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_table(test_pred, tables_dir / "test_predictions_best_ckpt")

    y_true = encode_gate_indices(test_pred["stage1_gold"].to_numpy())
    y_pred = test_pred["gate_pred_idx"].to_numpy()
    n_cls = len(GATE_CLASS_NAMES)
    logits_onehot = np.zeros((len(y_pred), n_cls), dtype=np.float32)
    logits_onehot[np.arange(len(y_pred)), y_pred] = 1.0
    g1 = compute_gate_metrics(logits_onehot, y_true, protocol=protocol)

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1_score(y_true, y_pred)),
        "min_class_recall": g1["min_class_recall"],
        "evangelisti_g1_pass": g1["evangelisti_g1_pass"],
        "g1_status": format_g1_status(g1, protocol),
        "per_class_recall": g1["per_class_recall"],
        "per_class_specificity": g1["per_class_specificity"],
        "n_tiles": int(len(test_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(n_cls)),
            target_names=list(GATE_CLASS_NAMES),
            zero_division=0,
            output_dict=True,
        ),
    }
    summary_path = out_test / "evaluation_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_root / "evaluation_metrics_summary_test.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    plot_confusion_matrix(y_true, y_pred, out_test / "confusion_matrix.png")
    plot_class_recall_bars(g1["per_class_recall"], out_test / "recall_by_class.png")
    plot_class_specificity_bars(g1["per_class_specificity"], out_test / "specificity_by_class.png")
    plot_g1_combined_bars(
        g1["per_class_recall"],
        g1["per_class_specificity"],
        protocol.recall_thresh,
        protocol.spec_thresh,
        out_test / "g1_sens_spec_bars.png",
    )

    cm_rows = []
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    for i, gold in enumerate(GATE_CLASS_NAMES):
        for j, pred in enumerate(GATE_CLASS_NAMES):
            cm_rows.append({"gold": gold, "pred": pred, "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv(out_test / "confusion_matrix.csv", index=False)

    per_class_rows = []
    for cls in GATE_CLASS_NAMES:
        per_class_rows.append(
            {
                "class": cls,
                "recall": g1["per_class_recall"].get(cls),
                "specificity": g1["per_class_specificity"].get(cls),
                "recall_thresh": protocol.recall_thresh.get(cls),
                "spec_thresh": protocol.spec_thresh.get(cls),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(out_test / "per_class_g1.csv", index=False)

    per_image = []
    for rel in test_pred["image_path"].unique():
        sub = test_pred[test_pred["image_path"] == rel]
        per_image.append(
            {
                "image_path": rel,
                "n_tiles": len(sub),
                "acc": float(sub["correct"].mean()),
                "n_mplus_pred": int((sub["stage1_pred"] == "Mplus").sum()),
                "n_mminus_pred": int((sub["stage1_pred"] == "Mminus").sum()),
                "n_bg_pred": int((sub["stage1_pred"] == "Background").sum()),
            }
        )
    pd.DataFrame(per_image).to_csv(out_test / "per_image_summary.csv", index=False)

    log.info(
        f"[Gate snapshot] test holdout ep{best_epoch}: "
        f"min_recall={g1['min_class_recall']:.4f} macro_f1={summary['macro_f1']:.4f} "
        f"-> {out_test}"
    )
    return summary_path


def write_run_status_md(
    run_root: Path,
    *,
    status: str,
    epoch_current: int,
    epochs_total: int,
    best_epoch: int,
    best_score: float,
    protocol: Any,
    last_holdout: Optional[dict] = None,
    post_training_ready: bool = False,
) -> Path:
    """Estado legible: por qué las métricas parecen planas y qué falta."""
    lines = [
        "# Estado del entrenamiento Gate AM",
        "",
        f"- **Estado:** `{status}`",
        f"- **Época:** {epoch_current} / {epochs_total}",
        f"- **Mejor checkpoint:** época {best_epoch}, `{protocol.checkpoint_metric}`={best_score:.4f}",
        "",
        "## Por qué la loss parece «pegada»",
        "",
        "Con **Slice-MS pura** la loss de entrenamiento es Multi-Similarity (~1.2–1.4);",
        "no es CE y **no debe bajar a cero**. Lo que importa es `min_class_recall` y las",
        "curvas en `images/post_training/recall_curves.png`.",
        "",
        "## Evangelisti G1 (último holdout)",
        "",
    ]
    if last_holdout:
        lines.append(f"- G1: **{'PASS' if last_holdout.get('evangelisti_g1_pass') else 'FAIL'}**")
        mcr = last_holdout.get("min_class_recall")
        if mcr is not None:
            lines.append(f"- min_class_recall: **{float(mcr):.4f}**")
        for cls in ("Background", "Mminus", "Mplus"):
            rec = (last_holdout.get("per_class_recall") or {}).get(cls, float("nan"))
            if rec != rec:
                continue
            rt = protocol.recall_thresh.get(cls, 0.9)
            ok = "OK" if rec >= rt else "FAIL"
            lines.append(f"- {cls} recall: {rec:.3f} (umbral {rt:.2f}) → {ok}")
    lines += [
        "",
        "## Artefactos post-training",
        "",
    ]
    if post_training_ready:
        lines += [
            "- Listo: matriz confusión, embeddings UMAP, análisis en `analysis/`",
        ]
    else:
        lines += [
            "- **En progreso:** eval incremental en `images/post_training/test/` (mejor ckpt).",
            "- **Al terminar:** finalize completo + `analysis/` (UMAP, separabilidad, bottleneck).",
            "- Si el proceso se interrumpe: `python run.py recover-gate-am-report --run-id <run_id>`",
        ]
    path = Path(run_root) / "RUN_STATUS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
