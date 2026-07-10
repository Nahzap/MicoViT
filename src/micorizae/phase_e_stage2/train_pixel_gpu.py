"""Entrenamiento GPU Stage2-Pixel — ViT DINOv2 segmentación morfológica IH/A/V/H."""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..common.logging_utils import get_logger
from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX, PIXEL_IGNORE_INDEX
from .pixel_data import iter_pixel_batches
from .pixel_morph import PixelMorphParams
from .stage2_pixel_run_layout import (
    Stage2PixelRunLayout,
    atomic_write_text,
    epoch_checkpoint_path,
    find_latest_epoch_checkpoint,
    publish_best_checkpoint,
    save_epoch_checkpoint,
    write_training_state,
)
from .stage2_pixel_train_report import (
    confusion_from_pred_labels,
    format_epoch_line,
    format_progress_line,
    gpu_mem_gb,
    log_epoch_gpu_stats,
    macro_iou,
    per_class_iou,
    phase_log,
    write_live_metrics,
    write_training_metrics_csv,
    write_train_warning,
    write_train_heartbeat,
)
from .stage2_pixel_posttrain_report import render_sample_tile_panels

log = get_logger("phase_e.train_pixel")


@dataclass
class PixelTrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_miou: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_per_class_iou: list[dict[str, float]] = field(default_factory=list)
    elapsed_s: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_miou: float = -1.0
    last_confusion: Optional[list] = None

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dict__}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PixelTrainHistory":
        h = cls()
        for k in (
            "epochs",
            "train_loss",
            "val_loss",
            "val_miou",
            "val_acc",
            "val_per_class_iou",
            "elapsed_s",
        ):
            setattr(h, k, list(d.get(k, [])))
        h.best_epoch = int(d.get("best_epoch", -1))
        h.best_val_miou = float(d.get("best_val_miou", -1.0))
        h.last_confusion = d.get("last_confusion")
        return h


def _flush_logs() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def _cleanup_gpu(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _pixel_ce_loss(logits: torch.Tensor, labels: torch.Tensor, class_w: Optional[torch.Tensor] = None) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(logits, labels, weight=class_w, ignore_index=PIXEL_IGNORE_INDEX)


def _focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_w: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss con alpha (class_w) separado del factor focal.

    El factor (1-p_t)^gamma se calcula sobre la CE SIN ponderar para que p_t sea
    la probabilidad real de la clase verdadera; el peso de clase se aplica aparte
    como alpha_t. Mezclarlos (p_t=exp(-ce_ponderada)) rompe la modulación focal
    en clases de peso alto y dispara gradientes/val_loss en épocas tempranas.
    """
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    valid = labels != PIXEL_IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    safe_labels = labels.clone()
    safe_labels[~valid] = 0
    ce = F.cross_entropy(logits, safe_labels, reduction="none")
    ce = ce * valid  # anula píxeles ignore
    p_t = torch.exp(-ce)
    focal_factor = (1.0 - p_t) ** gamma
    if class_w is not None:
        alpha_t = class_w[safe_labels] * valid
        loss = alpha_t * focal_factor * ce
        return loss.sum() / alpha_t.sum().clamp_min(1.0)
    return (focal_factor * ce).sum() / valid.sum().clamp_min(1)


def _pixel_slice_ms_loss(
    feat: torch.Tensor,
    labels: torch.Tensor,
    ms_module,
    *,
    k_per_class: int = 256,
    epoch: int = 1,
) -> torch.Tensor:
    """Slice-MS metric loss sobre embeddings de píxel del decoder.

    Muestrea K píxeles por clase presente (estratificado sobre el batch), toma
    sus embeddings del mapa de features penúltimo y aplica Slice Multi-Similarity
    (Wang et al. 2019). Estructura el espacio de features por morfología sin
    depender de asignaciones absolutas por píxel — robusto al ruido de etiqueta.
    """
    b, c, fh, fw = feat.shape
    lab = labels
    if lab.shape[-2:] != (fh, fw):
        lab = F.interpolate(lab.unsqueeze(1).float(), size=(fh, fw), mode="nearest").squeeze(1).long()
    emb = feat.permute(0, 2, 3, 1).reshape(-1, c)
    lab_flat = lab.reshape(-1)
    picks_e, picks_y = [], []
    for cls in torch.unique(lab_flat):
        if int(cls) == PIXEL_IGNORE_INDEX:
            continue
        idx = (lab_flat == cls).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if idx.numel() > k_per_class:
            sel = torch.randint(0, idx.numel(), (k_per_class,), device=idx.device)
            idx = idx[sel]
        picks_e.append(emb[idx])
        picks_y.append(torch.full((idx.numel(),), int(cls), device=emb.device, dtype=torch.long))
    if len(picks_y) < 2:
        return feat.sum() * 0.0
    e = torch.cat(picks_e, dim=0)
    y = torch.cat(picks_y, dim=0)
    return ms_module(e, y, epoch=epoch)


def _aux_entropy_loss(
    logits: torch.Tensor,
    stage2_gold: list[str],
) -> torch.Tensor:
    if not stage2_gold or len(stage2_gold) != logits.shape[0]:
        return torch.tensor(0.0, device=logits.device)
    probs = F.softmax(logits, dim=1).mean(dim=[-2, -1])
    eps = 1e-6
    max_h = float(np.log(probs.shape[1]))
    entropy = - (probs * torch.log(probs + eps)).sum(dim=1) / max_h
    
    loss_aux = torch.tensor(0.0, device=logits.device)
    n_valid = 0
    for i, g in enumerate(stage2_gold):
        if g == "AMColonised":
            loss_aux = loss_aux + (entropy[i] ** 2)
            n_valid += 1
        elif g == "Hybrid":
            loss_aux = loss_aux + ((1.0 - entropy[i]) ** 2)
            n_valid += 1
    if n_valid > 0:
        loss_aux = loss_aux / n_valid
    return loss_aux


def _compute_class_weights(
    h5_store: Optional[Any],
    train_df: Any,
    device: torch.device,
    mode: str = "inv_freq",
) -> torch.Tensor:
    if mode in ("uniform", "", None):
        return torch.ones(NUM_PIXEL_CLASSES, dtype=torch.float32, device=device)
    
    counts = np.zeros(NUM_PIXEL_CLASSES, dtype=np.int64)
    if h5_store is not None and hasattr(h5_store, "indices_for_sub"):
        idx = np.unique(h5_store.indices_for_sub(train_df))
        lbls = None
        ram = getattr(h5_store, "_ram_cache", None)
        if ram is not None and getattr(ram, "label", None) is not None:
            lbls = ram.label[idx]
        elif hasattr(h5_store, "_hf") and h5_store._hf is not None and "label" in h5_store._hf:
            lbls = h5_store._hf["label"][np.sort(idx)]
        if lbls is not None and lbls.size > 0:
            binc = np.bincount(lbls.reshape(-1).astype(np.int64), minlength=NUM_PIXEL_CLASSES)
            counts = binc[:NUM_PIXEL_CLASSES].astype(np.int64)
    if np.sum(counts) == 0:
        counts = np.array([14394253, 1657484, 4767, 1028617, 8002879], dtype=np.int64)

    total = max(float(np.sum(counts)), 1.0)
    freq = counts / total
    w = np.zeros(NUM_PIXEL_CLASSES, dtype=np.float32)
    valid_mask = counts > 0
    if np.any(valid_mask):
        if mode == "inv_sqrt":
            w[valid_mask] = (1.0 / freq[valid_mask]) ** 0.5
        else:
            w[valid_mask] = (1.0 / freq[valid_mask]) ** 0.55

        h_idx = PIXEL_CLASS_TO_IDX["H"]
        ref_idx = h_idx if counts[h_idx] > 0 else int(np.argmax(counts))
        w[valid_mask] = w[valid_mask] / max(w[ref_idx], 1e-6)
        w[valid_mask] = np.clip(w[valid_mask], 0.3, 15.0)
    
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    log.info(f"[Class Weights ({mode})] " + " ".join(f"{PIXEL_CLASS_NAMES[c]}={w[c]:.2f}" for c in range(NUM_PIXEL_CLASSES)))
    return w_t


def _lr_for_group(opt: torch.optim.Optimizer, idx: int) -> float:
    if idx < len(opt.param_groups):
        return float(opt.param_groups[idx]["lr"])
    return float(opt.param_groups[0]["lr"])


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


@torch.no_grad()
def evaluate_pixel_model(
    model: nn.Module,
    tiles_df,
    *,
    device: torch.device,
    batch_size: int,
    input_size: int = 224,
    input_mode: str = "vit",
    morph_params: PixelMorphParams,
    max_batches: Optional[int] = None,
    use_amp: bool = True,
    desc: str = "val_pixel",
    phase: str = "val",
    h5_store: Optional[object] = None,
    class_w: Optional[torch.Tensor] = None,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    cm = np.zeros((NUM_PIXEL_CLASSES, NUM_PIXEL_CLASSES), dtype=np.int64)
    correct_px, total_px = 0, 0
    batch_idx = 0

    it = iter_pixel_batches(
        tiles_df,
        batch_size=batch_size,
        device=device,
        input_size=input_size,
        input_mode=input_mode,
        morph_params=morph_params,
        shuffle=False,
        max_batches=max_batches,
        phase=phase,
        log_every_batches=0,
        h5_store=h5_store,
    )
    import sys
    from tqdm import tqdm

    n_val_tiles = len(tiles_df)
    n_est = max(1, (n_val_tiles + batch_size - 1) // batch_size)
    if max_batches is not None:
        n_est = min(n_est, max_batches)

    is_tty = sys.stdout.isatty()
    it_wrap = tqdm(it, total=n_est, desc=f"[{desc}]", leave=False, dynamic_ncols=True, file=sys.stdout) if is_tty else it

    for batch in it_wrap:
        batch_idx += 1
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(batch.model_input)
            if loss_type == "focal":
                loss = _focal_loss(logits, batch.labels, class_w, gamma=focal_gamma)
            else:
                loss = _pixel_ce_loss(logits, batch.labels, class_w)
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch.labels.size(0)
        n += batch.labels.size(0)
        pred_np = pred.cpu().numpy()
        lab_np = batch.labels.cpu().numpy()
        cm += confusion_from_pred_labels(pred_np, lab_np)
        valid_px = batch.labels != PIXEL_IGNORE_INDEX
        correct_px += int(((pred == batch.labels) & valid_px).sum().item())
        total_px += int(valid_px.sum().item())
        del logits, pred, batch
        if is_tty:
            if batch_idx % 2 == 0 or batch_idx == n_est:
                it_wrap.set_postfix(loss=f"{total_loss / max(n, 1):.3f}", gpu=f"{gpu_mem_gb():.1f}GB")
        elif batch_idx % 30 == 0 or batch_idx == n_est:
            pct = 100.0 * batch_idx / max(n_est, 1)
            print(f"  [{desc}] {pct:5.1f}% ({batch_idx}/{n_est}) | loss={total_loss / max(n, 1):.4f} | gpu={gpu_mem_gb():.1f}GB", flush=True)

    pc_iou = per_class_iou(cm)
    return {
        "loss": total_loss / max(n, 1),
        "miou": macro_iou(pc_iou),
        "acc": float(correct_px / max(total_px, 1)),
        "per_class_iou": pc_iou,
        "confusion": cm.tolist(),
    }


def _publish_epoch(
    *,
    run_dir: Optional[Path],
    run_id: Optional[str],
    history: PixelTrainHistory,
    epoch: int,
    epochs: int,
    val: dict,
    improved: bool,
    csv_rows: list[dict[str, Any]],
) -> None:
    if run_dir is None:
        return
    pc = val.get("per_class_iou", {})
    csv_rows.append(
        {
            "epoch": epoch,
            "train_loss": history.train_loss[-1],
            "val_loss": val["loss"],
            "val_miou": val["miou"],
            "val_acc": val["acc"],
            "val_iou_BG": pc.get("BG", float("nan")),
            "val_iou_IH": pc.get("IH", float("nan")),
            "val_iou_V": pc.get("V", float("nan")),
            "val_iou_A": pc.get("A", float("nan")),
            "val_iou_H": pc.get("H", float("nan")),
            "elapsed_s": history.elapsed_s[-1],
            "best": int(improved),
        }
    )
    write_training_metrics_csv(run_dir, csv_rows)
    if run_id:
        write_live_metrics(
            run_dir,
            run_id=run_id,
            epoch=epoch,
            epochs_total=epochs,
            history=history.to_dict(),
            last_val=val,
            best_epoch=history.best_epoch,
            best_val_miou=history.best_val_miou,
            improved=improved,
        )


def train_pixel_morph_gpu(
    model: nn.Module,
    train_df,
    val_df,
    *,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 8,
    input_size: int = 224,
    input_mode: str = "vit",
    lr: float = 1e-4,
    checkpoint_dir: Path,
    checkpoint_name: str = "stage2_pixel_vit_best",
    morph_params: Optional[PixelMorphParams] = None,
    use_amp: bool = True,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    backbone_lr_factor: float = 0.1,
    metrics_run_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    n_params: Optional[int] = None,
    h5_store: Optional[object] = None,
    layout: Optional[Stage2PixelRunLayout] = None,
    start_epoch: int = 1,
    initial_history: Optional[PixelTrainHistory] = None,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    class_weight_mode: str = "inv_freq",
    aux_entropy_loss: bool = True,
    aux_entropy_weight: float = 0.1,
    prior_loss_enabled: bool = False,
    prior_loss_weights: Optional[Any] = None,
    morph_params_for_prior: Optional[PixelMorphParams] = None,
    prior_loss_workers: int = 4,
    ms_loss_enabled: bool = False,
    ms_loss_weight: float = 0.5,
    ms_slices: int = 4,
    ms_k_per_class: int = 256,
    ms_alpha: float = 2.0,
    ms_beta: float = 50.0,
    ms_base: float = 0.5,
) -> PixelTrainHistory:
    """Entrenamiento formal Stage2-Pixel (único entrenamiento de Fase E)."""
    morph_params = morph_params or PixelMorphParams()
    morph_params_for_prior = morph_params_for_prior or morph_params
    ms_module = None
    if ms_loss_enabled:
        from ..phase_d_stage1.gate4.slice_ms_loss import SliceMultiSimilarityLoss

        feat_dim = int(getattr(model, "feat_dim_out", 32))
        n_slices = ms_slices
        while n_slices > 1 and feat_dim % n_slices != 0:
            n_slices -= 1
        ms_module = SliceMultiSimilarityLoss(
            num_slices=max(1, n_slices),
            alpha=ms_alpha,
            beta=ms_beta,
            base=ms_base,
            hard_mining=True,
            mining_margin=0.1,
        )
    if h5_store is not None and hasattr(h5_store, "ensure_ram_cache"):
        try:
            from ..cli import _cfg

            h5_store.ensure_ram_cache(enabled=bool(_cfg("STAGE2_PIXEL_H5_RAM_CACHE", True)))
        except Exception:
            pass
    if prior_loss_enabled:
        from .pixel_prior_loss import PriorLossWeights, combine_prior_losses

        plw = prior_loss_weights or PriorLossWeights()
    else:
        plw = None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = model.to(device)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups: list[dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_factor})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history = initial_history or PixelTrainHistory()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{checkpoint_name}.pt"
    csv_rows: list[dict[str, Any]] = []

    if metrics_run_dir and (metrics_run_dir / "training_metrics.csv").is_file():
        import csv

        with open(metrics_run_dir / "training_metrics.csv", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

    if metrics_run_dir:
        metrics_run_dir.mkdir(parents=True, exist_ok=True)

    if start_epoch > 1:
        restored = False
        if layout is not None:
            latest = find_latest_epoch_checkpoint(layout)
            if latest is not None:
                try:
                    ck = torch.load(latest[1], map_location=device, weights_only=False)
                    if "optimizer_state_dict" in ck:
                        opt.load_state_dict(ck["optimizer_state_dict"])
                    if "scheduler_state_dict" in ck:
                        sched.load_state_dict(ck["scheduler_state_dict"])
                        restored = "optimizer_state_dict" in ck
                except Exception as exc:
                    phase_log(f"RESUME estado opt/sched no restaurado ({exc}); usando fast-forward LR")
        if not restored:
            # Fallback (checkpoints antiguos sin estado opt/sched): avanza solo el LR.
            for _ in range(start_epoch - 1):
                sched.step()

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    n_val = len(val_df)

    class_w = _compute_class_weights(h5_store, train_df, device, mode=class_weight_mode)

    if h5_store is not None:
        try:
            from ..cli import _cfg
            from .pixel_data import expand_train_df_v_flip_oversample

            if bool(_cfg("STAGE2_PIXEL_FLIP_OVERSAMPLE_V", True)):
                train_df = expand_train_df_v_flip_oversample(
                    train_df,
                    h5_store,
                    min_v_px=int(_cfg("STAGE2_PIXEL_V_MIN_PX", 30)),
                    variants=int(_cfg("STAGE2_PIXEL_RARE_FLIP_VARIANTS", 4)),
                )
        except Exception as exc:
            phase_log(f"oversample V omitido: {exc}")
            write_train_warning(
                metrics_run_dir,
                "OVERSAMPLE_SKIP",
                str(exc),
                level="warning",
            )

    n_train = len(train_df)
    n_batches_est = max(1, (n_train + batch_size - 1) // batch_size)
    if max_train_batches is not None:
        n_batches_est = min(n_batches_est, max_train_batches)

    phase_log(
        f"INICIO Stage2-Pixel ViT | backbone={getattr(model, 'backbone_name', 'dinov2_vits14')} "
        f"classes={list(PIXEL_CLASS_NAMES)} train={n_train} val={n_val} epochs={epochs} "
        f"batch={batch_size} input={input_size}px amp={use_amp} gpu={gpu_name}"
    )
    if start_epoch > 1:
        phase_log(f"RESUME desde ep {start_epoch}/{epochs} best_mIoU={history.best_val_miou:.4f}@ep{history.best_epoch}")
    if n_params is not None:
        phase_log(f"params={n_params:,} lr_head={lr:.1e} lr_backbone={lr * backbone_lr_factor:.1e}")
    if run_id:
        phase_log(f"run={run_id}")
    if metrics_run_dir:
        phase_log(f"metricas -> {metrics_run_dir / 'training_metrics.csv'}")
    if layout is not None:
        phase_log(f"checkpoints/epoch -> {layout.pretrain_checkpoints}")
    phase_log(
        "NO conforma HDF5 ni cache. Lee tiles M+ de manifests/tiles_index.csv, "
        "genera pseudo-mascaras weak on-the-fly (CPU), entrena decoder ViT (GPU)."
        if h5_store is None else
        "Usando HDF5 Stage2-Pixel (rgb+label precomputados). Entrena decoder ViT (GPU)."
    )

    write_train_heartbeat(
        metrics_run_dir,
        phase="init",
        run_id=run_id or "",
        epoch=0,
        epochs_total=epochs,
        n_train_tiles=n_train,
        n_batches_est=n_batches_est,
        best_miou=history.best_val_miou,
        best_epoch=history.best_epoch,
    )

    if start_epoch == 1:
        phase_log(f"FASE baseline val ep0/{epochs} — evaluacion antes de entrenar ({n_val} tiles val)...")
        baseline = evaluate_pixel_model(
            model,
            val_df,
            device=device,
            batch_size=batch_size,
            input_size=input_size,
            input_mode=input_mode,
            morph_params=morph_params,
            max_batches=max_val_batches,
            use_amp=use_amp,
            phase="val_ep0",
            h5_store=h5_store,
            class_w=class_w,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )
        phase_log(
            f"FASE baseline DONE ep0/{epochs} val_loss={baseline['loss']:.4f} "
            f"val_mIoU={baseline['miou']:.4f} val_acc={baseline['acc']:.4f} | "
            f"{_morph_iou_short(baseline['per_class_iou'])}"
        )
        if metrics_run_dir:
            atomic_write_text(
                Path(metrics_run_dir) / "baseline_ep0.json",
                json.dumps(baseline, indent=2, default=str),
            )
        if layout is not None and h5_store is not None:
            try:
                render_sample_tile_panels(
                    model, val_df, h5_store, device=device, out_dir=layout.pretrain / "previews" / "baseline", n_tiles=12
                )
                phase_log("FASE previews baseline -> pretrain/previews/baseline/")
            except Exception as exc:
                write_train_warning(metrics_run_dir, "PREVIEW_FAIL", f"baseline: {exc}", level="warning")
        _cleanup_gpu(device)

    log_interval = max(1, n_batches_est // 20)
    last_epoch_ckpt: Optional[Path] = None
    best_post_path: Optional[Path] = None
    prior_from_h5 = bool(h5_store is not None and getattr(h5_store, "has_priors", False))

    for ep in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        phase_log(f"FASE entrenamiento ep {ep}/{epochs} — inicio ({n_train} tiles train, backward ViT GPU)")
        write_train_heartbeat(
            metrics_run_dir,
            phase="train",
            epoch=ep,
            epochs_total=epochs,
            batch=0,
            n_batches_est=n_batches_est,
            best_miou=history.best_val_miou,
            best_epoch=history.best_epoch,
        )
        model.train()
        run_loss, run_main_loss, n_tiles = 0.0, 0.0, 0
        batch_idx = 0
        train_cm = np.zeros((NUM_PIXEL_CLASSES, NUM_PIXEL_CLASSES), dtype=np.int64)
        train_correct, train_total = 0, 0

        it = iter_pixel_batches(
            train_df,
            batch_size=batch_size,
            device=device,
            input_size=input_size,
            input_mode=input_mode,
            morph_params=morph_params,
            shuffle=True,
            max_batches=max_train_batches,
            phase=f"train_ep{ep}",
            log_every_batches=0,
            h5_store=h5_store,
            load_priors=prior_from_h5 and prior_loss_enabled and plw is not None,
        )

        is_tty = sys.stdout.isatty()
        it_wrap = tqdm(it, total=n_batches_est, desc=f"Ep {ep}/{epochs} [Train]", leave=False, dynamic_ncols=True, file=sys.stdout) if is_tty else it

        prior_prefetch = None
        if prior_loss_enabled and plw is not None and prior_from_h5:
            phase_log("FASE train — priors MEViT desde HDF5 (sin Frangi CPU en runtime)")
        elif prior_loss_enabled and plw is not None:
            from .pixel_prior_maps import PriorPrefetcher

            prior_prefetch = PriorPrefetcher(
                morph_params_for_prior, device, max_workers=prior_loss_workers
            )

        batch_iter = iter(it_wrap)
        current_batch = next(batch_iter, None)
        if prior_prefetch is not None and current_batch is not None:
            prior_prefetch.prime(current_batch.model_input)

        while current_batch is not None:
            batch_idx += 1
            next_batch = next(batch_iter, None)
            opt.zero_grad(set_to_none=True)
            try:
                prior_e = None
                ves_mask = None
                if prior_loss_enabled and plw is not None:
                    if current_batch.prior_evidence is not None and current_batch.prior_vesicle is not None:
                        prior_e = current_batch.prior_evidence
                        ves_mask = current_batch.prior_vesicle
                    elif prior_prefetch is not None:
                        if batch_idx == 1:
                            phase_log(
                                f"FASE train ep {ep} batch 1/{n_batches_est} — priors CPU listos, forward GPU..."
                            )
                        prior_e, ves_mask = prior_prefetch.get()
                        if next_batch is not None:
                            prior_prefetch.prime(next_batch.model_input)
                with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                    if ms_module is not None:
                        logits, dec_feat = model(current_batch.model_input, return_feat=True)
                    else:
                        logits = model(current_batch.model_input)
                        dec_feat = None
                    if loss_type == "focal":
                        main_loss = _focal_loss(logits, current_batch.labels, class_w, gamma=focal_gamma)
                    else:
                        main_loss = _pixel_ce_loss(logits, current_batch.labels, class_w)
                    if aux_entropy_loss and hasattr(current_batch, "stage2_gold") and current_batch.stage2_gold:
                        aux_loss = _aux_entropy_loss(logits, current_batch.stage2_gold)
                        loss = main_loss + aux_entropy_weight * aux_loss
                    else:
                        loss = main_loss
                    if prior_e is not None and ves_mask is not None:
                        prior_bd = combine_prior_losses(logits, prior_e, ves_mask, weights=plw)
                        loss = loss + prior_bd.total
                    if ms_module is not None and dec_feat is not None:
                        ms_loss = _pixel_slice_ms_loss(
                            dec_feat.float(),
                            current_batch.labels,
                            ms_module,
                            k_per_class=ms_k_per_class,
                            epoch=ep,
                        )
                        loss = loss + ms_loss_weight * ms_loss
                loss.backward()
                opt.step()
            except torch.cuda.OutOfMemoryError:
                write_train_warning(
                    metrics_run_dir,
                    "CUDA_OOM",
                    f"ep {ep} batch {batch_idx}",
                    level="error",
                )
                phase_log(
                    f"CUDA OOM ep {ep} batch {batch_idx} — liberando cache y guardando checkpoint preval"
                )
                _cleanup_gpu(device)
                if layout is not None:
                    preval = epoch_checkpoint_path(layout, ep, preval=True)
                    save_epoch_checkpoint(
                        preval,
                        model=model,
                        epoch=ep,
                        input_mode=input_mode,
                        input_size=input_size,
                    )
                    phase_log(f"checkpoint preval OOM -> {preval}")
                raise

            pred = logits.argmax(dim=1).detach()
            bs = current_batch.labels.size(0)
            run_loss += float(loss.item()) * bs
            run_main_loss += float(main_loss.item()) * bs
            n_tiles += bs
            _valid = current_batch.labels != PIXEL_IGNORE_INDEX
            train_correct += int(((pred == current_batch.labels) & _valid).sum().item())
            train_total += int(_valid.sum().item())
            if batch_idx % 5 == 0 or batch_idx == n_batches_est:
                train_cm += confusion_from_pred_labels(pred.cpu().numpy(), current_batch.labels.cpu().numpy())
            del logits, pred, loss, main_loss, prior_e, ves_mask
            if dec_feat is not None:
                del dec_feat
            current_batch = next_batch

            avg_loss = run_loss / max(n_tiles, 1)
            tr_pc = per_class_iou(train_cm)
            tr_miou = macro_iou(tr_pc)
            tr_acc = train_correct / max(train_total, 1)

            if is_tty:
                it_wrap.update(1)
                if batch_idx % 2 == 0 or batch_idx == n_batches_est:
                    it_wrap.set_postfix(
                        loss=f"{avg_loss:.3f}",
                        mIoU=f"{tr_miou:.3f}",
                        acc=f"{tr_acc*100:.1f}%",
                        IH=f"{tr_pc.get('IH', 0):.2f}",
                        V=f"{tr_pc.get('V', 0):.2f}",
                        A=f"{tr_pc.get('A', 0):.2f}",
                    )
            elif batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == n_batches_est:
                elapsed = time.perf_counter() - t0
                tiles_per_s = n_tiles / max(elapsed, 1e-6)
                pct = 100.0 * batch_idx / max(n_batches_est, 1)
                print(
                    f"  [Train Ep {ep:2d}/{epochs}] {pct:5.1f}% ({batch_idx:3d}/{n_batches_est}) | "
                    f"loss={avg_loss:.3f} mIoU={tr_miou:.3f} acc={tr_acc*100:.1f}% | "
                    f"IH={tr_pc.get('IH',0):.2f} V={tr_pc.get('V',0):.2f} A={tr_pc.get('A',0):.2f} | {tiles_per_s:.1f} tiles/s",
                    flush=True,
                )
                write_train_heartbeat(
                    metrics_run_dir,
                    phase="train",
                    epoch=ep,
                    epochs_total=epochs,
                    batch=batch_idx,
                    n_batches_est=n_batches_est,
                    pct=round(pct, 1),
                    train_loss=round(avg_loss, 4),
                    train_miou=round(tr_miou, 4),
                    best_miou=history.best_val_miou,
                    best_epoch=history.best_epoch,
                    )

        if prior_prefetch is not None:
            prior_prefetch.shutdown()

        phase_log(f"FASE entrenamiento ep {ep}/{epochs} — fin train, guardando checkpoint preval...")
        if layout is not None:
            preval_path = epoch_checkpoint_path(layout, ep, preval=True)
            save_epoch_checkpoint(
                preval_path,
                model=model,
                epoch=ep,
                input_mode=input_mode,
                input_size=input_size,
            )

        sched.step()

        val = evaluate_pixel_model(
            model,
            val_df,
            device=device,
            batch_size=batch_size,
            input_size=input_size,
            input_mode=input_mode,
            morph_params=morph_params,
            max_batches=max_val_batches,
            use_amp=use_amp,
            phase=f"val_ep{ep}",
            h5_store=h5_store,
            class_w=class_w,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )
        elapsed = time.perf_counter() - t0
        tr_loss = run_main_loss / max(n_tiles, 1)

        history.epochs.append(ep)
        history.train_loss.append(tr_loss)
        history.val_loss.append(val["loss"])
        history.val_miou.append(val["miou"])
        history.val_acc.append(val["acc"])
        history.val_per_class_iou.append(val["per_class_iou"])
        history.elapsed_s.append(elapsed)
        history.last_confusion = val.get("confusion")

        improved = val["miou"] > history.best_val_miou
        epoch_ckpt_path: Optional[Path] = None
        if layout is not None:
            epoch_ckpt_path = epoch_checkpoint_path(layout, ep)
            save_epoch_checkpoint(
                epoch_ckpt_path,
                model=model,
                epoch=ep,
                val_miou=val["miou"],
                val_acc=val["acc"],
                val_loss=val["loss"],
                per_class_iou=val["per_class_iou"],
                input_mode=input_mode,
                input_size=input_size,
                is_best=improved,
                optimizer=opt,
                scheduler=sched,
            )
            last_epoch_ckpt = epoch_ckpt_path

        write_train_heartbeat(
            metrics_run_dir,
            phase="val_done",
            epoch=ep,
            epochs_total=epochs,
            val_miou=val["miou"],
            val_acc=val["acc"],
            improved=improved,
            best_miou=history.best_val_miou,
            best_epoch=history.best_epoch,
        )

        if improved:
            history.best_val_miou = val["miou"]
            history.best_epoch = ep
            save_epoch_checkpoint(
                best_path,
                model=model,
                epoch=ep,
                val_miou=val["miou"],
                val_acc=val["acc"],
                val_loss=val["loss"],
                per_class_iou=val["per_class_iou"],
                input_mode=input_mode,
                input_size=input_size,
                is_best=True,
            )
            if layout is not None:
                best_post_path, _ = publish_best_checkpoint(
                    layout,
                    best_path,
                    global_ckpt_dir=ckpt_dir,
                    checkpoint_name=checkpoint_name,
                )

        if layout is not None and h5_store is not None:
            preview_dir = layout.pretrain / "previews" / f"epoch_{ep:03d}"
            try:
                render_sample_tile_panels(
                    model,
                    val_df,
                    h5_store,
                    device=device,
                    out_dir=preview_dir,
                    n_tiles=12,
                )
                render_sample_tile_panels(
                    model,
                    val_df,
                    h5_store,
                    device=device,
                    out_dir=layout.pretrain / "previews" / "latest",
                    n_tiles=12,
                    seed=42,
                )
                phase_log(f"FASE previews ep {ep} -> {preview_dir.name} + latest/")
            except Exception as exc:
                write_train_warning(metrics_run_dir, "PREVIEW_FAIL", f"ep {ep}: {exc}", level="warning")
                phase_log(f"FASE previews ep {ep} FALLÓ: {exc}")

        phase_log(
            format_epoch_line(
                epoch=ep,
                epochs_total=epochs,
                improved=improved,
                train_loss=tr_loss,
                val=val,
                elapsed_s=elapsed,
                best_miou=history.best_val_miou,
                best_epoch=history.best_epoch,
            )
        )
        if improved:
            phase_log(f"checkpoint guardado -> {best_path}")
            if best_post_path:
                phase_log(f"checkpoint posttrain/global -> {best_post_path}")
        if epoch_ckpt_path:
            phase_log(f"checkpoint epoch -> {epoch_ckpt_path}")

        _publish_epoch(
            run_dir=metrics_run_dir,
            run_id=run_id,
            history=history,
            epoch=ep,
            epochs=epochs,
            val=val,
            improved=improved,
            csv_rows=csv_rows,
        )

        if layout is not None:
            write_training_state(
                layout,
                epoch=ep,
                epochs_total=epochs,
                history=history.to_dict(),
                best_epoch=history.best_epoch,
                best_val_miou=history.best_val_miou,
                last_checkpoint=last_epoch_ckpt,
                best_checkpoint=best_post_path or (best_path if improved else None),
                global_checkpoint=best_path if improved else None,
                gpu_mem_gb=gpu_mem_gb(),
                run_id=run_id,
            )

        log_epoch_gpu_stats(ep, epochs)
        _flush_logs()
        _cleanup_gpu(device)

    if metrics_run_dir:
        atomic_write_text(
            metrics_run_dir / "STAGE2_PIXEL_TRAIN_LIVE.json",
            json.dumps(history.to_dict(), indent=2, default=str),
        )

    return history


def _morph_iou_short(per_class: dict[str, float]) -> str:
    from .stage2_pixel_train_report import morph_iou_summary

    return morph_iou_summary(per_class)
