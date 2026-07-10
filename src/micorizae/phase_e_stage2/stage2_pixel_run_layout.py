"""Layout Stage2-Pixel: pretrain (setup + métricas en curso) y posttrain (resultados).

Artefactos para reconstrucción tras crash (orden de preferencia):
  1. ``pretrain/training_state.json`` — historial completo + rutas checkpoint
  2. ``pretrain/live_metrics.json`` + ``pretrain/training_metrics.csv``
  3. ``pretrain/checkpoints/epoch_NNN.pt`` — pesos por época
  4. ``posttrain/checkpoint/stage2_pixel_vit_best.pt`` o global
     ``models/checkpoints/stage2_am/stage2_pixel_vit_best.pt``

Recovery: ``recover_stage2_pixel_run(run_root)`` materializa ``posttrain/`` sin re-entrenar.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..common.logging_utils import get_logger
from ..common.run_outputs import RunOutputs
from .pixel_class_map import PIXEL_CLASS_NAMES

log = get_logger("phase_e.pixel_layout")

_EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)\.pt$")


@dataclass(frozen=True)
class Stage2PixelRunLayout:
    """Rutas bajo ``outputs/<run_id>/``."""

    run_root: Path

    @property
    def pretrain(self) -> Path:
        return self.run_root / "pretrain"

    @property
    def posttrain(self) -> Path:
        return self.run_root / "posttrain"

    @property
    def config_dir(self) -> Path:
        return self.run_root / "config"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    @property
    def pretrain_checkpoints(self) -> Path:
        return self.pretrain / "checkpoints"

    @property
    def posttrain_checkpoint_dir(self) -> Path:
        return self.posttrain / "checkpoint"

    def ensure(self) -> "Stage2PixelRunLayout":
        for p in (self.pretrain, self.posttrain, self.config_dir, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        self.pretrain_checkpoints.mkdir(parents=True, exist_ok=True)
        self.posttrain_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self


def layout_for_run(run: RunOutputs) -> Stage2PixelRunLayout:
    return Stage2PixelRunLayout(run.root).ensure()


def layout_from_run_root(run_root: Path) -> Stage2PixelRunLayout:
    return Stage2PixelRunLayout(run_root.resolve()).ensure()


def layout_from_run_id(run_id: str, *, outputs_root: Optional[Path] = None) -> Stage2PixelRunLayout:
    run = RunOutputs.open(run_id, outputs_root=outputs_root)
    return layout_from_run_root(run.root)


def resolve_run_layout(run_root: Path | str) -> Stage2PixelRunLayout:
    """Resuelve layout priorizando ``outputs/<run_id>/``."""
    from ..common.paths import get_paths

    token = str(run_root)
    run_path = Path(run_root)
    outputs_candidate = get_paths().outputs / token
    if outputs_candidate.is_dir():
        return layout_from_run_root(outputs_candidate)
    if run_path.is_dir() and (run_path / "pretrain").is_dir():
        return layout_from_run_root(run_path)
    return layout_from_run_id(token)


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def atomic_copy(src: Path, dst: Path) -> Optional[Path]:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return dst


def epoch_checkpoint_path(layout: Stage2PixelRunLayout, epoch: int, *, preval: bool = False) -> Path:
    tag = f"epoch_{epoch:03d}"
    if preval:
        tag += "_preval"
    return layout.pretrain_checkpoints / f"{tag}.pt"


def find_latest_epoch_checkpoint(layout: Stage2PixelRunLayout) -> Optional[tuple[int, Path]]:
    ckpt_dir = layout.pretrain_checkpoints
    if not ckpt_dir.is_dir():
        return None
    best_ep, best_path = -1, None
    for p in ckpt_dir.glob("epoch_*.pt"):
        if "_preval" in p.name:
            continue
        m = _EPOCH_CKPT_RE.match(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep, best_path = ep, p
    if best_path is None:
        return None
    return best_ep, best_path


def resolve_best_checkpoint(
    layout: Stage2PixelRunLayout,
    *,
    global_ckpt_dir: Optional[Path] = None,
) -> Optional[Path]:
    candidates = [
        layout.posttrain_checkpoint_dir / "stage2_pixel_vit_best.pt",
        layout.pretrain_checkpoints / "stage2_pixel_vit_best.pt",
    ]
    if global_ckpt_dir is not None:
        candidates.append(global_ckpt_dir / "stage2_pixel_vit_best.pt")
    for p in candidates:
        if p.is_file():
            return p
    latest = find_latest_epoch_checkpoint(layout)
    return latest[1] if latest else None


def is_posttrain_complete(layout: Stage2PixelRunLayout) -> bool:
    return (layout.posttrain / "training_history.json").is_file()


def write_pretrain_setup(
    layout: Stage2PixelRunLayout,
    *,
    split_info: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    h5_meta: dict,
    train_config: dict,
) -> None:
    """Artefactos previos / durante entrenamiento (dataset, H5, config)."""
    profile = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_info": split_info,
        "n_train_tiles": int(len(train_df)),
        "n_val_tiles": int(len(val_df)),
        "n_test_tiles": int(len(test_df)),
        "n_train_images": int(train_df["image_path"].nunique()) if len(train_df) else 0,
        "n_val_images": int(val_df["image_path"].nunique()) if len(val_df) else 0,
        "n_test_images": int(test_df["image_path"].nunique()) if len(test_df) else 0,
    }
    _write_json(layout.pretrain / "dataset_profile.json", profile)
    _write_json(layout.pretrain / "h5_cache.json", h5_meta)
    _write_json(layout.config_dir / "train_config.json", train_config)
    log.info(f"[Stage2-Pixel] pretrain/ -> {layout.pretrain}")


def write_baseline_ep0(layout: Stage2PixelRunLayout, baseline: dict) -> None:
    _write_json(layout.pretrain / "baseline_ep0.json", baseline)


def write_training_state(
    layout: Stage2PixelRunLayout,
    *,
    epoch: int,
    epochs_total: int,
    history: dict,
    best_epoch: int,
    best_val_miou: float,
    last_checkpoint: Optional[Path] = None,
    best_checkpoint: Optional[Path] = None,
    global_checkpoint: Optional[Path] = None,
    gpu_mem_gb: Optional[float] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Estado incremental — permite resume y recovery sin re-entrenar."""
    payload = {
        "run_id": run_id,
        "epoch": epoch,
        "epochs_total": epochs_total,
        "best_epoch": best_epoch,
        "best_val_miou": best_val_miou,
        "last_epoch_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "global_checkpoint": str(global_checkpoint) if global_checkpoint else None,
        "gpu_mem_gb_last": gpu_mem_gb,
        "history": history,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "recovery_note": (
            "Reconstruir posttrain/: recover_stage2_pixel_run(run_root) o "
            "python run.py recover-stage2-pixel-run --run-id <run_id>"
        ),
    }
    return _write_json(layout.pretrain / "training_state.json", payload)


def save_epoch_checkpoint(
    path: Path,
    *,
    model,
    epoch: int,
    val_miou: Optional[float] = None,
    val_acc: Optional[float] = None,
    val_loss: Optional[float] = None,
    per_class_iou: Optional[dict] = None,
    input_mode: str = "vit",
    input_size: int = 224,
    is_best: bool = False,
    optimizer=None,
    scheduler=None,
) -> Path:
    """Guarda checkpoint atómico (temp + rename)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_miou": val_miou,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "per_class_iou": per_class_iou,
        "input_mode": input_mode,
        "input_size": input_size,
        "classes": list(PIXEL_CLASS_NAMES),
        "is_best": is_best,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def publish_best_checkpoint(
    layout: Stage2PixelRunLayout,
    src: Path,
    *,
    global_ckpt_dir: Path,
    checkpoint_name: str = "stage2_pixel_vit_best",
) -> tuple[Path, Path]:
    """Copia best a posttrain/ y checkpoint global."""
    dst_post = layout.posttrain_checkpoint_dir / f"{checkpoint_name}.pt"
    dst_global = global_ckpt_dir / f"{checkpoint_name}.pt"
    global_ckpt_dir.mkdir(parents=True, exist_ok=True)
    atomic_copy(src, dst_post)
    atomic_copy(src, dst_global)
    return dst_post, dst_global


def history_from_pretrain(layout: Stage2PixelRunLayout) -> dict:
    """Reconstruye historial desde artefactos pretrain/."""
    state_path = layout.pretrain / "training_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        hist = state.get("history")
        if isinstance(hist, dict) and hist.get("epochs"):
            return hist

    live_path = layout.pretrain / "live_metrics.json"
    if live_path.is_file():
        live = json.loads(live_path.read_text(encoding="utf-8"))
        hist = {
            "epochs": live.get("epochs_done", []),
            "train_loss": live.get("train_loss", []),
            "val_loss": live.get("val_loss", []),
            "val_miou": live.get("val_miou", []),
            "val_acc": live.get("val_acc", []),
            "elapsed_s": live.get("elapsed_s", []),
            "best_epoch": live.get("best_epoch", -1),
            "best_val_miou": live.get("best_val_miou", -1.0),
            "val_per_class_iou": [],
        }
        csv_path = layout.pretrain / "training_metrics.csv"
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            per_epoch = []
            for _, row in df.iterrows():
                entry = {c: float(row.get(f"val_iou_{c}", float("nan"))) for c in PIXEL_CLASS_NAMES}
                per_epoch.append(entry)
            hist["val_per_class_iou"] = per_epoch
        last_val = live.get("last_val") or {}
        if last_val.get("confusion"):
            hist["last_confusion"] = last_val["confusion"]
        return hist

    raise FileNotFoundError(
        f"No hay training_state.json ni live_metrics.json en {layout.pretrain}"
    )


def load_training_state(layout: Stage2PixelRunLayout) -> dict:
    path = layout.pretrain / "training_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_training_profile(layout: Stage2PixelRunLayout, history: dict) -> None:
    """Estadísticas de perfil al cierre del entrenamiento."""
    elapsed = history.get("elapsed_s") or []
    profile = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_epochs_completed": len(history.get("epochs", [])),
        "mean_epoch_s": float(sum(elapsed) / len(elapsed)) if elapsed else None,
        "total_train_s": float(sum(elapsed)) if elapsed else None,
        "best_epoch": history.get("best_epoch"),
        "best_val_miou": history.get("best_val_miou"),
        "val_miou_curve": history.get("val_miou", []),
        "train_loss_curve": history.get("train_loss", []),
    }
    _write_json(layout.pretrain / "training_profile.json", profile)


def write_per_class_iou_csv(layout: Stage2PixelRunLayout, history: dict, *, dest_dir: Optional[Path] = None) -> Path:
    dest = (dest_dir or layout.posttrain) / "per_class_iou_by_epoch.csv"
    rows = []
    epochs = history.get("epochs", [])
    per_class = history.get("val_per_class_iou") or []
    for i, ep in enumerate(epochs):
        pc = per_class[i] if i < len(per_class) else {}
        row: dict[str, Any] = {"epoch": ep}
        for name in PIXEL_CLASS_NAMES:
            row[f"iou_{name}"] = pc.get(name, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)
    return dest


def _ensure_matplotlib_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def plot_val_miou_curve(history: dict, out_png: Path) -> Optional[Path]:
    epochs = history.get("epochs", [])
    miou = history.get("val_miou", [])
    if not epochs or not miou:
        return None
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, miou, marker="o", color="#4C78A8", label="val mIoU")
    be = history.get("best_epoch")
    if be and be in epochs:
        idx = epochs.index(be)
        ax.scatter([be], [miou[idx]], color="#E45756", zorder=5, label=f"best ep{be}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val mIoU")
    ax.set_title("Stage2-Pixel ViT — val mIoU")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_confusion_matrix(
    confusion: list,
    out_png: Path,
    *,
    title: str = "Confusion matrix (val, best epoch)",
) -> Optional[Path]:
    if not confusion:
        return None
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(confusion, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(PIXEL_CLASS_NAMES)))
    ax.set_yticks(range(len(PIXEL_CLASS_NAMES)))
    ax.set_xticklabels(PIXEL_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(PIXEL_CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def write_training_plots(layout: Stage2PixelRunLayout, history: dict, *, dest_dir: Optional[Path] = None) -> dict[str, str]:
    """Genera PNGs de curvas y matriz de confusión en posttrain/plots/."""
    dest = dest_dir or (layout.posttrain / "plots")
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    miou_png = dest / "val_miou_curve.png"
    if plot_val_miou_curve(history, miou_png):
        written["val_miou_curve"] = str(miou_png)

    confusion = history.get("last_confusion")
    if confusion is None:
        live_path = layout.pretrain / "live_metrics.json"
        if live_path.is_file():
            last_val = json.loads(live_path.read_text(encoding="utf-8")).get("last_val") or {}
            confusion = last_val.get("confusion")
    cm_png = dest / "confusion_matrix_best.png"
    if confusion and plot_confusion_matrix(confusion, cm_png):
        written["confusion_matrix"] = str(cm_png)

    return written


def build_run_meta(
    layout: Stage2PixelRunLayout,
    *,
    history: dict,
    gate_run_id: str,
    vit_name: str,
    epochs: int,
    input_size: int,
    split_info: dict,
    n_test_tiles: int,
    checkpoint_path: Path,
    h5_meta: dict,
    interrupted: bool = False,
) -> dict:
    return {
        "run_id": layout.run_root.name,
        "gate_run_id": gate_run_id,
        "model": "dinov2_vit_pixel",
        "backbone": vit_name,
        "epochs": epochs,
        "epochs_completed": len(history.get("epochs", [])),
        "input_size": input_size,
        "split_info": split_info,
        "n_test_tiles": n_test_tiles,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "plan_doc": "Docs/20260629_134156_PLAN_PIXEL_MORFO_COLONIAS_AM_IH_A_V_H.md",
        "pixel_morph": True,
        "h5_cache": h5_meta,
        "interrupted": interrupted,
        "recovered_at": datetime.now().isoformat(timespec="seconds") if interrupted else None,
    }


def finalize_posttrain(
    layout: Stage2PixelRunLayout,
    *,
    history: dict,
    meta: dict,
    checkpoint_src: Path,
    write_plots: bool = True,
) -> Path:
    """Resultados finales: checkpoint, meta, informe, plots."""
    post = layout.posttrain
    post.mkdir(parents=True, exist_ok=True)

    hist = history if isinstance(history, dict) else history.to_dict()
    _write_json(post / "training_history.json", hist)
    _write_json(post / "STAGE2_PIXEL_RUN_META.json", meta)
    _write_json(post / "stage2_pixel_train_report.json", meta)

    best = {
        "best_epoch": hist.get("best_epoch"),
        "best_val_miou": hist.get("best_val_miou"),
        "val_miou": hist.get("val_miou", []),
        "val_acc": hist.get("val_acc", []),
    }
    if hist.get("val_per_class_iou"):
        be = int(hist.get("best_epoch", 1)) - 1
        if 0 <= be < len(hist["val_per_class_iou"]):
            best["per_class_iou_at_best"] = hist["val_per_class_iou"][be]
    _write_json(post / "best_metrics.json", best)

    ckpt_dst = post / "checkpoint" / checkpoint_src.name
    ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_src.is_file():
        atomic_copy(checkpoint_src, ckpt_dst)

    for src_name in ("training_metrics.csv", "live_metrics.json", "training_profile.json"):
        src = layout.pretrain / src_name
        if src.is_file():
            atomic_copy(src, post / src_name)

    write_per_class_iou_csv(layout, hist, dest_dir=post)
    if write_plots:
        plots = write_training_plots(layout, hist, dest_dir=post / "plots")
        if plots:
            meta = dict(meta)
            meta["plots"] = plots
            _write_json(post / "STAGE2_PIXEL_RUN_META.json", meta)
            _write_json(post / "stage2_pixel_train_report.json", meta)

    _write_json(layout.run_root / "STAGE2_PIXEL_RUN_META.json", meta)
    log.info(f"[Stage2-Pixel] posttrain/ -> {post}")
    return post


def recover_stage2_pixel_run(
    run_root: Path | str,
    *,
    global_ckpt_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Materializa posttrain/ desde pretrain/ + checkpoints (sin re-entrenar).

    Usa ``training_state.json``, ``live_metrics.json``, ``training_metrics.csv``
    y el mejor checkpoint disponible.
    """
    run_path = Path(run_root)
    layout = resolve_run_layout(run_path if run_path.is_dir() else str(run_root))
    if is_posttrain_complete(layout) and not force:
        log.info(f"[Stage2-Pixel] posttrain/ ya completo -> {layout.posttrain}")
        return layout.posttrain

    history = history_from_pretrain(layout)
    state: dict = {}
    try:
        state = load_training_state(layout)
    except FileNotFoundError:
        pass

    train_cfg_path = layout.config_dir / "train_config.json"
    train_cfg = {}
    if train_cfg_path.is_file():
        train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))

    if global_ckpt_dir is None:
        from ..common.paths import get_paths

        global_ckpt_dir = get_paths().root / "models" / "checkpoints" / "stage2_am"

    ckpt = resolve_best_checkpoint(layout, global_ckpt_dir=global_ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint encontrado para recovery en {layout.run_root}"
        )

    n_test = 0
    profile_path = layout.pretrain / "dataset_profile.json"
    if profile_path.is_file():
        n_test = json.loads(profile_path.read_text(encoding="utf-8")).get("n_test_tiles", 0)

    split_info = train_cfg.get("split_info") or {}
    if not split_info and profile_path.is_file():
        split_info = json.loads(profile_path.read_text(encoding="utf-8")).get("split_info", {})

    h5_meta_path = layout.pretrain / "h5_cache.json"
    h5_meta = {}
    if h5_meta_path.is_file():
        h5_meta = json.loads(h5_meta_path.read_text(encoding="utf-8"))

    interrupted = len(history.get("epochs", [])) < int(train_cfg.get("epochs", 40))
    meta = build_run_meta(
        layout,
        history=history,
        gate_run_id=str(train_cfg.get("gate_run_id", "")),
        vit_name=str(train_cfg.get("vit_name", "dinov2_vits14")),
        epochs=int(train_cfg.get("epochs", 40)),
        input_size=int(train_cfg.get("input_size", 224)),
        split_info=split_info,
        n_test_tiles=int(n_test),
        checkpoint_path=ckpt,
        h5_meta=h5_meta,
        interrupted=interrupted,
    )
    if state:
        meta["training_state"] = {
            k: state.get(k)
            for k in ("epoch", "epochs_total", "gpu_mem_gb_last", "updated_at")
        }

    write_training_profile(layout, history)
    finalize_posttrain(
        layout,
        history=history,
        meta=meta,
        checkpoint_src=ckpt,
        write_plots=True,
    )
    log.info(
        f"[Stage2-Pixel] Recovery OK — {len(history.get('epochs', []))} épocas, "
        f"best_mIoU={history.get('best_val_miou')} ep={history.get('best_epoch')}"
    )
    return layout.posttrain
