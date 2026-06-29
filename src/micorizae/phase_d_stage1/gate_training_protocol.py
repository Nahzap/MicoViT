"""Protocolo de entrenamiento gate AM (Evangelisti G1 + practicas IEEE/ML).

Referencias internas:
- Docs/00_INDICE_DOCUMENTACION.md (plan maestro AM, G1 Sens/Spec)
- Docs/20260618_120000_PLAN_CACHE_EMBEDDINGS_GATE_AM.md (E5)
- Stage2 train_gpu.py: checkpoint por macro F1 (no accuracy bruta)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.special import softmax as _softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from .gate_classes import GATE_CLASS_NAMES, GATE_CLASS_TO_IDX, GATE_IDX_TO_CLASS

# Evangelisti et al. (2021) — umbrales G1 en am_test (10 imagenes holdout).
DEFAULT_RECALL_THRESH: dict[str, float] = {
    "Mplus": 0.90,
    "Mminus": 0.90,
    "Background": 0.85,
    "Unknown": 0.80,
}
DEFAULT_SPEC_THRESH: dict[str, float] = {
    "Mplus": 0.90,
    "Mminus": 0.90,
    "Background": 0.90,
    "Unknown": 0.85,
}

CHECKPOINT_METRICS = (
    "macro_f1",
    "min_class_recall",
    "balanced_accuracy",
    "evangelisti_g1",
    "mAP",
    "composite_recall_map",
)


@dataclass(frozen=True)
class GateTrainProtocol:
    """Hiperparametros y criterios de evaluacion del gate (config.py)."""

    balance_mode: str = "root_focused"  # cap_bg | evangelisti_1to1 | balanced_3class | root_focused
    max_bg_per_image: int = 25
    checkpoint_metric: str = "min_class_recall"
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    max_epochs: int = 40
    min_epochs: int = 8
    probe_lr: float = 5e-4
    finetune_lr: float = 1e-4
    probe_lr_warmup_epochs: int = 5
    probe_lr_min_factor: float = 0.1
    mplus_oversample_factor: float = 1.0
    mplus_focal_boost: float = 1.0
    stratified_samples_per_class: int = 3840
    stratified_min_per_batch: int = 0  # 0 = auto batch_size // 3
    # Pesos de cuota por clase en el batch estratificado (más M-/M+ => más pares duros).
    stratified_class_weights: dict[str, float] = field(default_factory=dict)
    stratified_root_only_batch_ratio: float = 0.0
    hard_image_substrings: tuple[str, ...] = ()
    hard_image_weight: float = 1.0
    eval_balance_mode: str = "dual"  # natural | g1_stratified | dual
    eval_stratified_samples_per_class: int = 384
    checkpoint_eval: str = "stratified"  # stratified | natural
    # 0 = holdout natural solo al finalizar (no 613 batches/ep durante train)
    holdout_eval_every_n_epochs: int = 0
    metric_inference: str = "prototype"  # prototype | knn (knn solo analisis)
    use_class_weights_train: bool = False
    eval_unweighted_loss: bool = True
    loss_type: str = "slice_ms_only"  # slice_ms_only | ce | focal_ce (legacy)
    focal_gamma: float = 2.0
    calibrate_post_train: bool = True
    recall_thresh: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RECALL_THRESH))
    spec_thresh: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPEC_THRESH))
    train_domain_stratified: bool = False
    checkpoint_composite_bg_weight: float = 0.0
    checkpoint_tile_edge_weight: float = 0.0
    checkpoint_tile_edge_targets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.checkpoint_metric not in CHECKPOINT_METRICS:
            raise ValueError(
                f"checkpoint_metric debe ser uno de {CHECKPOINT_METRICS}, "
                f"recibido {self.checkpoint_metric!r}"
            )
        if self.balance_mode not in {
            "cap_bg",
            "evangelisti_1to1",
            "balanced_3class",
            "balanced_4class",
            "root_focused",
            "g1_stratified",
        }:
            raise ValueError(
                f"balance_mode debe ser cap_bg, evangelisti_1to1, balanced_3class, "
                f"balanced_4class, root_focused o g1_stratified, recibido {self.balance_mode!r}"
            )
        if self.loss_type not in {"ce", "focal_ce", "slice_ms_only"}:
            raise ValueError(
                f"loss_type debe ser slice_ms_only, ce o focal_ce, recibido {self.loss_type!r}"
            )
        if self.eval_balance_mode not in {"natural", "g1_stratified", "dual"}:
            raise ValueError(
                f"eval_balance_mode debe ser natural, g1_stratified o dual, "
                f"recibido {self.eval_balance_mode!r}"
            )
        if self.checkpoint_eval not in {"stratified", "natural"}:
            raise ValueError(
                f"checkpoint_eval debe ser stratified o natural, recibido {self.checkpoint_eval!r}"
            )
        if self.metric_inference not in {"prototype", "knn"}:
            raise ValueError(
                f"metric_inference debe ser prototype o knn, recibido {self.metric_inference!r}"
            )


def _parse_hard_image_substrings(spec: str) -> tuple[str, ...]:
    spec = (spec or "").strip()
    if not spec:
        return ()
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def _parse_class_weights(spec: Any) -> dict[str, float]:
    """'Background:1.0,Mminus:1.5,Mplus:1.5' o dict -> {clase: peso}."""
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    spec = str(spec or "").strip()
    if not spec:
        return {}
    out: dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, val = chunk.partition(":")
        name = name.strip()
        if name not in GATE_CLASS_TO_IDX:
            raise ValueError(f"clase desconocida en GATE_STRATIFIED_CLASS_WEIGHTS: {name!r}")
        out[name] = float(val)
    return out


def _parse_tile_edge_targets(spec: str) -> tuple[int, ...]:
    spec = (spec or "").strip()
    if not spec:
        return ()
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def protocol_from_config(cfg: Any) -> GateTrainProtocol:
    """Construye protocolo desde config.py."""
    recall = {
        "Mplus": float(getattr(cfg, "GATE_RECALL_THRESH_MPLUS", 0.90)),
        "Mminus": float(getattr(cfg, "GATE_RECALL_THRESH_MMINUS", 0.90)),
        "Background": float(getattr(cfg, "GATE_RECALL_THRESH_BACKGROUND", 0.85)),
        "Unknown": float(getattr(cfg, "GATE_RECALL_THRESH_UNKNOWN", 0.80)),
    }
    spec = {
        "Mplus": float(getattr(cfg, "GATE_SPEC_THRESH_MPLUS", 0.90)),
        "Mminus": float(getattr(cfg, "GATE_SPEC_THRESH_MMINUS", 0.90)),
        "Background": float(getattr(cfg, "GATE_SPEC_THRESH_BACKGROUND", 0.90)),
        "Unknown": float(getattr(cfg, "GATE_SPEC_THRESH_UNKNOWN", 0.85)),
    }
    balance = str(getattr(cfg, "GATE_BALANCE_MODE", "root_focused"))
    use_w = bool(getattr(cfg, "GATE_USE_CLASS_WEIGHTS_TRAIN", False))
    if balance in (
        "evangelisti_1to1",
        "balanced_3class",
        "balanced_4class",
        "root_focused",
        "g1_stratified",
    ) and use_w:
        use_w = False
    formal = bool(getattr(cfg, "GATE_FORMAL_TRAIN", True))
    loss = str(getattr(cfg, "GATE_LOSS", "slice_ms_only"))
    if formal and loss != "slice_ms_only":
        raise ValueError(
            "GATE_FORMAL_TRAIN=True exige GATE_LOSS='slice_ms_only' "
            "(pipeline publicable; CE/focal no permitidos)."
        )
    calibrate = bool(getattr(cfg, "GATE_CALIBRATE_POST_TRAIN", False))
    if loss == "slice_ms_only":
        calibrate = False
    return GateTrainProtocol(
        balance_mode=balance,
        max_bg_per_image=int(getattr(cfg, "GATE_MAX_BG_PER_IMAGE", 25)),
        checkpoint_metric=str(getattr(cfg, "GATE_CHECKPOINT_METRIC", "min_class_recall")),
        early_stop_patience=int(getattr(cfg, "GATE_EARLY_STOP_PATIENCE", 5)),
        early_stop_min_delta=float(getattr(cfg, "GATE_EARLY_STOP_MIN_DELTA", 0.0)),
        max_epochs=int(getattr(cfg, "GATE_EPOCHS", 40)),
        min_epochs=int(getattr(cfg, "GATE_MIN_EPOCHS", 8)),
        probe_lr=float(getattr(cfg, "GATE_PROBE_LR", 5e-4)),
        finetune_lr=float(getattr(cfg, "GATE_FINETUNE_LR", 1e-4)),
        probe_lr_warmup_epochs=int(getattr(cfg, "GATE_PROBE_LR_WARMUP_EPOCHS", 5)),
        probe_lr_min_factor=float(getattr(cfg, "GATE_PROBE_LR_MIN_FACTOR", 0.1)),
        mplus_oversample_factor=float(getattr(cfg, "GATE_MPLUS_OVERSAMPLE_FACTOR", 1.0)),
        mplus_focal_boost=float(getattr(cfg, "GATE_MPLUS_FOCAL_BOOST", 1.0)),
        stratified_samples_per_class=int(getattr(cfg, "GATE_STRATIFIED_SAMPLES_PER_CLASS", 3840)),
        stratified_min_per_batch=int(getattr(cfg, "GATE_STRATIFIED_MIN_PER_BATCH", 0)),
        stratified_class_weights=_parse_class_weights(
            getattr(cfg, "GATE_STRATIFIED_CLASS_WEIGHTS", "")
        ),
        stratified_root_only_batch_ratio=float(
            getattr(cfg, "GATE_STRATIFIED_ROOT_ONLY_BATCH_RATIO", 0.0)
        ),
        hard_image_substrings=_parse_hard_image_substrings(
            str(getattr(cfg, "GATE_HARD_IMAGES", ""))
        ),
        hard_image_weight=float(getattr(cfg, "GATE_HARD_IMAGE_WEIGHT", 1.0)),
        eval_balance_mode=str(getattr(cfg, "GATE_EVAL_BALANCE_MODE", "dual")),
        eval_stratified_samples_per_class=int(
            getattr(cfg, "GATE_EVAL_STRATIFIED_SAMPLES_PER_CLASS", 384)
        ),
        checkpoint_eval=str(getattr(cfg, "GATE_CHECKPOINT_EVAL", "stratified")),
        holdout_eval_every_n_epochs=int(getattr(cfg, "GATE_HOLDOUT_EVAL_EVERY_N_EPOCHS", 0)),
        metric_inference=str(getattr(cfg, "GATE_METRIC_INFERENCE", "prototype")),
        use_class_weights_train=use_w,
        eval_unweighted_loss=bool(getattr(cfg, "GATE_EVAL_UNWEIGHTED_LOSS", True)),
        loss_type=str(getattr(cfg, "GATE_LOSS", "slice_ms_only")),
        focal_gamma=float(getattr(cfg, "GATE_FOCAL_GAMMA", 2.0)),
        calibrate_post_train=calibrate,
        recall_thresh=recall,
        spec_thresh=spec,
        train_domain_stratified=bool(getattr(cfg, "GATE_TRAIN_DOMAIN_STRATIFIED", False)),
        checkpoint_composite_bg_weight=float(
            getattr(cfg, "GATE_CHECKPOINT_COMPOSITE_BG_WEIGHT", 0.0)
        ),
        checkpoint_tile_edge_weight=float(
            getattr(cfg, "GATE_CHECKPOINT_TILE_EDGE_WEIGHT", 0.0)
        ),
        checkpoint_tile_edge_targets=_parse_tile_edge_targets(
            str(getattr(cfg, "GATE_CHECKPOINT_TILE_EDGE_TARGETS", ""))
        ),
    )


def active_class_indices(labels: np.ndarray, n_classes: int | None = None) -> list[int]:
    """Indices de clase presentes en labels (support > 0)."""
    n_classes = n_classes or len(GATE_CLASS_NAMES)
    return [idx for idx in range(n_classes) if np.any(labels == idx)]


def operational_class_indices(labels: np.ndarray) -> list[int]:
    """Clases con soporte en eval, excluyendo Unknown si no hay muestras Unreadable."""
    active = active_class_indices(labels)
    unk = GATE_CLASS_TO_IDX["Unknown"]
    if (labels == unk).sum() == 0 and unk in active:
        active = [i for i in active if i != unk]
    return active


def macro_f1_score(
    labels: np.ndarray,
    pred: np.ndarray,
    *,
    n_classes: int | None = None,
    labels_subset: list[int] | None = None,
) -> float:
    """Macro F1; por defecto solo clases con soporte (Unknown omitido si vacio)."""
    n_classes = n_classes or len(GATE_CLASS_NAMES)
    active = labels_subset if labels_subset is not None else operational_class_indices(labels)
    if not active:
        return 0.0
    return float(
        f1_score(labels, pred, labels=active, average="macro", zero_division=0)
    )


def compute_gate_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    protocol: Optional[GateTrainProtocol] = None,
) -> dict:
    """Metricas tile-level: acc, macro F1, recall/sensitivity, specificity (G1)."""
    protocol = protocol or GateTrainProtocol()
    n_classes = len(GATE_CLASS_NAMES)
    class_labels = list(range(n_classes))
    if len(labels) == 0:
        empty = {name: float("nan") for name in GATE_CLASS_NAMES}
        return {
            "acc": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "min_class_recall": 0.0,
            "min_class_specificity": 0.0,
            "per_class_recall": empty,
            "per_class_specificity": empty,
            "evangelisti_g1_pass": False,
            "evangelisti_g1_score": 0.0,
            "confusion_matrix": np.zeros((n_classes, n_classes), dtype=int).tolist(),
            "diagnostics": {},
        }

    pred = logits.argmax(axis=1)
    acc = float(accuracy_score(labels, pred))
    active_eval = operational_class_indices(labels)
    macro_f1 = macro_f1_score(labels, pred, labels_subset=active_eval)
    macro_f1_all = float(
        f1_score(labels, pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    )
    bal_acc = float(balanced_accuracy_score(labels, pred))

    cm = confusion_matrix(labels, pred, labels=class_labels)
    per_class_recall: dict[str, float] = {}
    per_class_specificity: dict[str, float] = {}
    for idx, name in GATE_IDX_TO_CLASS.items():
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        support = tp + fn
        per_class_recall[name] = tp / support if support > 0 else float("nan")
        denom_spec = tn + fp
        per_class_specificity[name] = tn / denom_spec if denom_spec > 0 else float("nan")

    finite_recalls = [
        v
        for name, v in per_class_recall.items()
        if not np.isnan(v) and name != "Unknown"
    ]
    if not finite_recalls:
        finite_recalls = [v for v in per_class_recall.values() if not np.isnan(v)]
    finite_specs = [v for v in per_class_specificity.values() if not np.isnan(v)]
    min_recall = float(min(finite_recalls)) if finite_recalls else 0.0
    min_spec = float(min(finite_specs)) if finite_specs else 0.0

    g1_pass = all(
        per_class_recall.get(name, 0.0) >= protocol.recall_thresh[name]
        and per_class_specificity.get(name, 0.0) >= protocol.spec_thresh[name]
        for name in GATE_CLASS_NAMES
        if name != "Unknown"
        and not np.isnan(per_class_recall.get(name, float("nan")))
        and not np.isnan(per_class_specificity.get(name, float("nan")))
    )
    g1_score = _evangelisti_g1_score(per_class_recall, per_class_specificity, protocol)

    diagnostics = compute_confusion_diagnostics(labels, pred, cm=cm)

    probs = _softmax(logits, axis=1)
    y_onehot = np.zeros((len(labels), n_classes), dtype=np.int32)
    y_onehot[np.arange(len(labels)), labels] = 1
    per_class_ap: dict[str, float] = {}
    for idx, name in GATE_IDX_TO_CLASS.items():
        if y_onehot[:, idx].sum() == 0:
            continue
        per_class_ap[name] = float(average_precision_score(y_onehot[:, idx], probs[:, idx]))
    mAP_val = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_f1_all_classes": macro_f1_all,
        "n_classes_eval": len(active_eval),
        "balanced_accuracy": bal_acc,
        "min_class_recall": min_recall,
        "min_class_specificity": min_spec,
        "per_class_recall": per_class_recall,
        "per_class_specificity": per_class_specificity,
        "mAP": mAP_val,
        "per_class_ap": per_class_ap,
        "evangelisti_g1_pass": g1_pass,
        "evangelisti_g1_score": g1_score,
        "confusion_matrix": cm.tolist(),
        "diagnostics": diagnostics,
    }


def compute_confusion_diagnostics(
    labels: np.ndarray,
    pred: np.ndarray,
    *,
    cm: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Conteos y precisiones derivados de la matriz de confusión (fronteras críticas)."""
    n_classes = len(GATE_CLASS_NAMES)
    if cm is None:
        cm = confusion_matrix(labels, pred, labels=list(range(n_classes)))
    cm = np.asarray(cm, dtype=np.int64)
    idx_bg = GATE_CLASS_TO_IDX["Background"]
    idx_mm = GATE_CLASS_TO_IDX["Mminus"]
    idx_mp = GATE_CLASS_TO_IDX["Mplus"]

    def _prec(cls_idx: int) -> float:
        col = int(cm[:, cls_idx].sum())
        return float(cm[cls_idx, cls_idx] / col) if col > 0 else float("nan")

    def _count(gold: int, pred_c: int) -> int:
        return int(cm[gold, pred_c])

    bg_total = int(cm[idx_bg, :].sum())
    bg_to_mplus = _count(idx_bg, idx_mp)
    bg_to_mminus = _count(idx_bg, idx_mm)
    mminus_to_mplus = _count(idx_mm, idx_mp)
    pred_mplus = int(cm[:, idx_mp].sum())

    return {
        "mplus_precision": _prec(idx_mp),
        "mminus_precision": _prec(idx_mm),
        "background_precision": _prec(idx_bg),
        "bg_to_mplus": bg_to_mplus,
        "bg_to_mminus": bg_to_mminus,
        "mminus_to_mplus": mminus_to_mplus,
        "mplus_to_mminus": _count(idx_mp, idx_mm),
        "bg_to_mplus_rate": (bg_to_mplus / bg_total) if bg_total > 0 else float("nan"),
        "bg_to_mminus_rate": (bg_to_mminus / bg_total) if bg_total > 0 else float("nan"),
        "pred_mplus_total": pred_mplus,
    }


def compute_per_image_bg_diagnostics(
    labels: np.ndarray,
    pred: np.ndarray,
    image_paths: np.ndarray | list[str],
) -> dict[str, Any]:
    """Recall BG por imagen: percentil 10 y peor imagen (detecta colapsos ABS710)."""
    idx_bg = GATE_CLASS_TO_IDX["Background"]
    paths = np.asarray(image_paths, dtype=object)
    if len(labels) == 0 or len(paths) != len(labels):
        return {}

    recalls: list[float] = []
    worst_name = ""
    worst_recall = 1.0
    for img in np.unique(paths):
        mask = paths == img
        gold_bg = mask & (labels == idx_bg)
        n_bg = int(gold_bg.sum())
        if n_bg == 0:
            continue
        r = float((pred[gold_bg] == idx_bg).sum() / n_bg)
        recalls.append(r)
        if r < worst_recall:
            worst_recall = r
            worst_name = str(img).split("/")[-1]

    if not recalls:
        return {}

    arr = np.asarray(recalls, dtype=np.float64)
    return {
        "bg_recall_p10": float(np.percentile(arr, 10)),
        "bg_recall_min": float(arr.min()),
        "bg_recall_median": float(np.median(arr)),
        "worst_bg_image": worst_name,
        "worst_bg_recall": float(worst_recall),
        "n_images_with_bg": len(recalls),
    }


def _evangelisti_g1_score(
    recall: dict[str, float],
    specificity: dict[str, float],
    protocol: GateTrainProtocol,
) -> float:
    """0..1 por clase; 1.0 = todas las clases cumplen Sens y Spec G1."""
    parts: list[float] = []
    for name in GATE_CLASS_NAMES:
        r_val = recall.get(name, float("nan"))
        s_val = specificity.get(name, float("nan"))
        if np.isnan(r_val) or np.isnan(s_val):
            continue
        r = r_val / max(protocol.recall_thresh[name], 1e-9)
        s = s_val / max(protocol.spec_thresh[name], 1e-9)
        parts.append(min(1.0, r) * 0.5 + min(1.0, s) * 0.5)
    return float(min(parts)) if parts else 0.0


def compute_by_tile_edge_diagnostics(
    labels: np.ndarray,
    pred: np.ndarray,
    tile_edges: np.ndarray,
) -> dict[str, dict]:
    """Recall minimo por tile_edge (Tier B checkpoint)."""
    out: dict[str, dict] = {}
    if len(labels) == 0 or len(tile_edges) != len(labels):
        return out
    for edge in np.unique(tile_edges):
        mask = tile_edges == edge
        if not mask.any():
            continue
        y = labels[mask]
        p = pred[mask]
        recalls: list[float] = []
        for idx in operational_class_indices(y):
            name = GATE_IDX_TO_CLASS[idx]
            cls_mask = y == idx
            if cls_mask.any():
                recalls.append(float((p[cls_mask] == idx).mean()))
        out[str(int(edge))] = {
            "n_tiles": int(mask.sum()),
            "min_recall": float(min(recalls)) if recalls else 0.0,
        }
    return out


def checkpoint_score(
    metrics: dict,
    metric_name: str,
    *,
    composite_bg_weight: float = 0.0,
    tile_edge_weight: float = 0.0,
    tile_edge_targets: tuple[int, ...] = (),
) -> float:
    """Score unico para seleccionar checkpoint (mayor es mejor)."""
    if metric_name == "macro_f1":
        base = float(metrics["macro_f1"])
    elif metric_name == "min_class_recall":
        base = float(metrics["min_class_recall"])
        if composite_bg_weight > 0.0:
            per_img = metrics.get("per_image_bg") or {}
            bg_p10 = float(per_img.get("bg_recall_p10", base))
            if np.isnan(bg_p10):
                bg_p10 = base
            base = (1.0 - composite_bg_weight) * base + composite_bg_weight * bg_p10
    elif metric_name == "balanced_accuracy":
        base = float(metrics["balanced_accuracy"])
    elif metric_name == "evangelisti_g1":
        base = float(metrics["evangelisti_g1_score"])
    elif metric_name == "mAP":
        base = float(metrics.get("mAP", 0.0))
    elif metric_name == "composite_recall_map":
        m_ap = float(metrics.get("mAP", 0.0))
        m_rec = float(metrics.get("min_class_recall", 0.0))
        base = 0.5 * m_ap + 0.5 * m_rec
    else:
        raise ValueError(f"metric_name desconocido: {metric_name}")

    if tile_edge_weight > 0.0 and tile_edge_targets:
        by_edge = metrics.get("by_tile_edge") or {}
        edge_recalls: list[float] = []
        for edge in tile_edge_targets:
            key = str(edge)
            row = by_edge.get(key) or by_edge.get(edge)
            if row is None:
                continue
            edge_recalls.append(float(row.get("min_recall", row.get("min_class_recall", base))))
        if edge_recalls:
            edge_min = float(min(edge_recalls))
            base = (1.0 - tile_edge_weight) * base + tile_edge_weight * edge_min
    return base


def g1_metric_display(value: float, threshold: float, *, decimals: int = 3) -> tuple[str, str]:
    """Formatea valor vs umbral G1. Retorna (texto, OK|FAIL|N/A)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A", "N/A"
    status = "OK" if value >= threshold else "FAIL"
    return f"{value:.{decimals}f}", status


def format_g1_status(metrics: dict, protocol: Optional[GateTrainProtocol] = None) -> str:
    """Linea legible Sens/Spec vs umbrales G1."""
    protocol = protocol or GateTrainProtocol()
    parts = []
    n_eval = 0
    for name in GATE_CLASS_NAMES:
        r = metrics["per_class_recall"].get(name, float("nan"))
        s = metrics["per_class_specificity"].get(name, float("nan"))
        rt = protocol.recall_thresh[name]
        st = protocol.spec_thresh[name]
        r_txt, r_ok = g1_metric_display(r, rt)
        s_txt, s_ok = g1_metric_display(s, st)
        if r_ok != "N/A":
            n_eval += 1
        parts.append(f"{name}:Sens={r_txt}/{rt:.2f}({r_ok}) Spec={s_txt}/{st:.2f}({s_ok})")
    flag = "G1_PASS" if metrics.get("evangelisti_g1_pass") else "G1_FAIL"
    suffix = ""
    if n_eval < len(GATE_CLASS_NAMES):
        suffix = f" [{n_eval}/{len(GATE_CLASS_NAMES)} clases eval]"
    return f"{flag}{suffix} | " + " | ".join(parts)
