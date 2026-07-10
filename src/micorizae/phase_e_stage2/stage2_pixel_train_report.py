"""Artefactos de entrenamiento Stage2-Pixel (ViT morfológico IH/A/V/H)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_NAMES

from ..common.logging_utils import get_logger

_log = get_logger("phase_e.pixel_train_report")


def _safe_console_text(msg: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return msg.encode(enc, errors="replace").decode(enc, errors="replace")


def phase_log(msg: str) -> None:
    """Log + print con flush — visible en terminal PowerShell."""
    safe = _safe_console_text(msg)
    _log.info(safe)
    print(f"[Stage2-Pixel] {safe}", flush=True)


def terminal_alarm(
    code: str,
    message: str,
    *,
    level: str = "error",
    beeps: int = 3,
) -> None:
    """Alarma ruidosa en terminal: banner + beeps (Windows) + stderr.

    Pensada para excepciones, kill/señales y watchdog. No depende del logger
    Rich (que a veces no flashea si el proceso ya está muriendo).
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = "=" * 72
    lines = [
        "",
        bar,
        f"  !! Stage2-Pixel ALARMA [{level.upper()}]  code={code}  @ {ts}",
        f"  {message}",
        bar,
        "",
    ]
    text = "\n".join(lines)
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        print(text, flush=True)
    except Exception:
        pass
    _log.error(f"ALARMA {code}: {message}")
    # Beep audibles (Windows: winsound; fallback ASCII BEL)
    try:
        if os.name == "nt":
            import winsound

            for _ in range(max(1, beeps)):
                winsound.Beep(880, 350)
        else:
            for _ in range(max(1, beeps)):
                sys.stdout.write("\a")
                sys.stdout.flush()
    except Exception:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


def write_train_warning(
    run_dir: Path | None,
    code: str,
    message: str,
    *,
    level: str = "warning",
    alarm: bool = False,
    **extra: Any,
) -> None:
    """Append warning/event to pretrain/train_warnings.jsonl + terminal."""
    line = f"[Stage2-Pixel {level.upper()}] {code}: {message}"
    print(line, flush=True)
    _log.warning(line)
    if alarm or level == "error":
        terminal_alarm(code, message, level=level)
    if run_dir is None:
        return
    from .stage2_pixel_run_layout import atomic_write_text

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "code": code,
        "message": message,
        "pid": os.getpid(),
        **extra,
    }
    path = Path(run_dir) / "train_warnings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def write_train_heartbeat(run_dir: Path | None, **payload: Any) -> None:
    """Último latido del entrenamiento — detectar cuelgues/muertes."""
    if run_dir is None:
        return
    from .stage2_pixel_run_layout import atomic_write_text

    body = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        **payload,
    }
    atomic_write_text(Path(run_dir) / "train_heartbeat.json", json.dumps(body, indent=2, default=str))


def confusion_from_pred_labels(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = pred.reshape(-1).astype(np.int64)
    labels = labels.reshape(-1).astype(np.int64)
    mask = (labels >= 0) & (labels < NUM_PIXEL_CLASSES)
    pred, labels = pred[mask], labels[mask]
    cm = np.zeros((NUM_PIXEL_CLASSES, NUM_PIXEL_CLASSES), dtype=np.int64)
    if len(labels):
        np.add.at(cm, (labels, pred), 1)
    return cm


def per_class_iou(cm: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for c, name in enumerate(PIXEL_CLASS_NAMES):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        union = tp + fn + fp
        out[name] = float(tp / union) if union > 0 else float("nan")
    return out


def macro_iou(per_class: dict[str, float]) -> float:
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def morph_iou_summary(per_class: dict[str, float]) -> str:
    parts = []
    for name in ("IH", "V", "A", "H"):
        v = per_class.get(name, float("nan"))
        parts.append(f"{name}={v:.3f}" if not np.isnan(v) else f"{name}=—")
    return " ".join(parts)


def gpu_mem_gb() -> float:
    """Pico de memoria GPU asignada en la corrida actual."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        pass
    return 0.0


def current_gpu_mem_gb() -> tuple[float, float]:
    """(allocated_GB, reserved_GB) en el instante actual."""
    try:
        import torch

        if torch.cuda.is_available():
            return (
                float(torch.cuda.memory_allocated() / (1024**3)),
                float(torch.cuda.memory_reserved() / (1024**3)),
            )
    except Exception:
        pass
    return 0.0, 0.0


def log_epoch_gpu_stats(epoch: int, epochs_total: int) -> None:
    alloc, reserved = current_gpu_mem_gb()
    peak = gpu_mem_gb()
    phase_log(
        f"GPU ep {epoch}/{epochs_total} — alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_live_metrics(
    run_dir: Path,
    *,
    run_id: str,
    epoch: int,
    epochs_total: int,
    history: dict[str, Any],
    last_val: dict[str, Any],
    best_epoch: int,
    best_val_miou: float,
    improved: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "phase": "stage2_pixel_vit",
        "classes": list(PIXEL_CLASS_NAMES),
        "checkpoint_metric": "val_miou",
        "epochs_total": epochs_total,
        "epochs_done": history.get("epochs", []),
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_miou": best_val_miou,
        "improved_last": improved,
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "val_miou": history.get("val_miou", []),
        "val_acc": history.get("val_acc", []),
        "elapsed_s": history.get("elapsed_s", []),
        "last_val": last_val,
        "last_per_class_iou": last_val.get("per_class_iou", {}),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    _atomic_write_text(run_dir / "live_metrics.json", payload_text)


def write_training_metrics_csv(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv
    import io

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "training_metrics.csv"
    keys = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_miou",
        "val_acc",
        "val_iou_BG",
        "val_iou_IH",
        "val_iou_V",
        "val_iou_A",
        "val_iou_H",
        "elapsed_s",
        "best",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    _atomic_write_text(path, buf.getvalue())


def format_epoch_line(
    *,
    epoch: int,
    epochs_total: int,
    improved: bool,
    train_loss: float,
    val: dict[str, Any],
    elapsed_s: float,
    best_miou: float,
    best_epoch: int,
) -> str:
    flag = "*" if improved else " "
    iou_morph = morph_iou_summary(val.get("per_class_iou", {}))
    return (
        f"[Stage2-Pixel ViT] ep {epoch}/{epochs_total} {flag} "
        f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
        f"val_mIoU={val['miou']:.4f} val_acc={val['acc']:.4f} | {iou_morph} | "
        f"best={best_miou:.4f}@ep{best_epoch} ({elapsed_s:.0f}s)"
    )


def format_progress_line(
    *,
    epoch: int,
    epochs_total: int,
    pct: float,
    batch_idx: int,
    n_batches: int,
    tiles_done: int,
    n_tiles: int,
    loss: float,
    train_miou: float,
    train_acc: float,
    tiles_per_s: float,
    lr_head: float,
    lr_backbone: Optional[float],
    eta_s: float,
) -> str:
    lr_bb = f" lr_bb={lr_backbone:.1e}" if lr_backbone is not None else ""
    return (
        f"[Stage2-Pixel ViT] ep {epoch}/{epochs_total} {pct:5.1f}% "
        f"batch {batch_idx}/{n_batches} tiles {tiles_done}/{n_tiles} "
        f"loss={loss:.4f} mIoU_tr={train_miou:.4f} acc_tr={train_acc:.4f} "
        f"{tiles_per_s:.1f} tiles/s lr={lr_head:.1e}{lr_bb} "
        f"gpu={gpu_mem_gb():.1f}GB eta={eta_s / 60:.1f}m"
    )
