"""Gate AM por tile: DINOv2 (Bg / M− / M+) → val/test → resultados.

Pipeline formal:
    1. Cache DINOv2 mean-pool por tile (252×252).
    2. Slice-MS probe entrena geometría en espacio de embeds.
    3. Validación holdout + checkpoint estratificado.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..common.logging_utils import get_logger
from .gate_classes import GATE_CLASS_NAMES, GATE_CLASS_TO_IDX, decode_gate_indices, is_valid_stage1, stage1_to_gate_label
from .gate_multiclass import _class_weights_from_df
from .gate_training_protocol import (
    GateTrainProtocol,
    checkpoint_score,
    compute_gate_metrics,
    compute_per_image_bg_diagnostics,
    compute_by_tile_edge_diagnostics,
    format_g1_status,
)
from .gpu_pipeline import EpochPlan, iter_image_batches, plan_epoch_gate, plan_epoch_class_counts
from .models import build_branch_a
from .train_gpu import GPUTrainHistory

log = get_logger("phase_d.gate_tile_dino")

CHECKPOINT_NAME = "gate_tile_dino_best.pt"
CHECKPOINT_MAP_NAME = "gate_tile_dino_best_map.pt"
PROGRESS_NAME = "training_progress.json"


def _write_training_progress(
    path: Path, payload: dict, *, run_live: Optional[object] = None
) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if run_live is not None:
        run_live.publish_progress(payload)


def _batch_total(plan: EpochPlan, batch_size: int, cap: Optional[int]) -> int:
    n = plan.n_batches(batch_size)
    if cap is not None:
        return min(n, cap)
    return n


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


@dataclass
class GateTileDinoGPU:
    """Clasificador / probe gate AM (inferencia desde cache o DINO on-the-fly)."""

    classifier: nn.Module
    device: torch.device = torch.device("cuda")

    def to(self, device: torch.device) -> "GateTileDinoGPU":
        self.device = device
        self.classifier.to(device).eval()
        return self


def _batch_iterator(
    plan: EpochPlan,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: Optional[int],
    h5_store: Optional[object],
    embed_store: Optional[object] = None,
    dino_input_size: int = 252,
    seg_target_size: int = 360,
    u2net_saliency: Optional[object] = None,
    force_cpu_decode: bool = False,
    cpu_decode_above_mb: float = 300.0,
):
    if embed_store is not None:
        from .gate_embed_cache import iter_embed_gate_batches

        return iter_embed_gate_batches(
            plan, embed_store, batch_size=batch_size, device=device, max_batches=max_batches
        )
    if plan.stratified_batches is not None:
        if h5_store is not None:
            from .gate_tile_h5_cache import iter_h5_stratified_gate_batches

            return iter_h5_stratified_gate_batches(
                plan,
                h5_store,
                device=device,
                max_batches=max_batches,
                u2net_saliency=u2net_saliency,
                saliency_size=dino_input_size,
            )
        from .gpu_pipeline import iter_stratified_gate_image_batches

        return iter_stratified_gate_image_batches(
            plan,
            batch_size=batch_size,
            device=device,
            label_mode="gate",
            max_batches=max_batches,
            target_size=dino_input_size,
            seg_target_size=seg_target_size,
            u2net_saliency=u2net_saliency,
            saliency_size=dino_input_size,
            force_cpu_decode=force_cpu_decode,
            cpu_decode_above_mb=cpu_decode_above_mb,
        )
    if h5_store is not None:
        from .gate_tile_h5_cache import iter_h5_gate_batches

        return iter_h5_gate_batches(
            plan, h5_store, batch_size=batch_size, device=device, max_batches=max_batches
        )
    return iter_image_batches(
        plan,
        batch_size=batch_size,
        device=device,
        label_mode="gate",
        max_batches=max_batches,
        target_size=dino_input_size,
        seg_target_size=seg_target_size,
        u2net_saliency=u2net_saliency,
        saliency_size=dino_input_size,
        force_cpu_decode=force_cpu_decode,
        cpu_decode_above_mb=cpu_decode_above_mb,
    )


def _is_slice_dino_model(model: nn.Module) -> bool:
    return hasattr(model, "encode_features") and hasattr(model, "encoder")


def _forward_classifier(
    model: nn.Module,
    batch,
    *,
    return_embed: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if batch.features is not None:
        from .gate_probe_input import probe_features_from_batch

        feats = probe_features_from_batch(batch)
        if return_embed:
            try:
                return model.forward_from_features(feats, return_embed=True)
            except TypeError:
                logits = model.forward_from_features(feats)
                return logits, None
        return model.forward_from_features(feats)
    if _is_slice_dino_model(model):
        labels = batch.labels.long()
        out = model(
            batch.rgb,
            mask=batch.saliency,
            labels=labels,
            return_embed=return_embed,
        )
        return out
    mask = batch.saliency
    logits = model(batch.rgb, mask=mask)
    if return_embed:
        return logits, None
    return logits


def evaluate_gate_tile_dino_gpu(
    model: nn.Module,
    val_plan: EpochPlan,
    device: torch.device,
    *,
    batch_size: int = 16,
    class_weights: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    desc: str = "val",
    h5_store: Optional[object] = None,
    embed_store: Optional[object] = None,
    protocol: Optional[GateTrainProtocol] = None,
    dino_input_size: int = 252,
    seg_target_size: int = 360,
    u2net_saliency: Optional[object] = None,
    prototype_bank: Optional[object] = None,
    slice_ms_only: bool = False,
    gate4_ms_only: Optional[nn.Module] = None,
    force_cpu_decode: bool = False,
    cpu_decode_above_mb: float = 300.0,
) -> dict:
    model.eval()
    all_logits, all_labels, all_paths = [], [], []
    all_edges: list[int] = []
    total_loss = 0.0
    n = 0
    total = _batch_total(val_plan, batch_size, max_batches)
    tiles_est = min(val_plan.n_tiles, (total * batch_size) if total else val_plan.n_tiles)
    log.info(
        f"[Gate tile] {desc}: {total} batches, ~{tiles_est:,} tiles "
        f"(batch_size={batch_size})"
    )
    if slice_ms_only and prototype_bank is not None:
        if prototype_bank.is_ready():
            log.info(
                f"[Gate tile] {desc}: inferencia por prototipos "
                f"(clases={prototype_bank.ready_class_indices()})"
            )
        else:
            log.warning(
                f"[Gate tile] {desc}: prototipos no listos — metricas omitidas "
                f"(slice_ms_only no usa gate_head)"
            )
            return {
                "acc": 0.0,
                "macro_f1": 0.0,
                "balanced_accuracy": 0.0,
                "min_class_recall": 0.0,
                "per_class_recall": {},
                "per_class_specificity": {},
                "loss": 0.0,
            }
    pbar = tqdm(
        _batch_iterator(
            val_plan,
            batch_size=batch_size,
            device=device,
            max_batches=max_batches,
            h5_store=h5_store,
            embed_store=embed_store,
            dino_input_size=dino_input_size,
            seg_target_size=seg_target_size,
            u2net_saliency=u2net_saliency,
            force_cpu_decode=force_cpu_decode,
            cpu_decode_above_mb=cpu_decode_above_mb,
        ),
        total=total,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    with torch.no_grad():
        for batch in pbar:
            y = batch.labels.long()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if slice_ms_only and prototype_bank is not None:
                    _, embed = _forward_classifier(model, batch, return_embed=True)
                    if embed is None:
                        logits = _forward_classifier(model, batch)
                        loss = F.cross_entropy(logits, y, weight=class_weights)
                    elif prototype_bank.is_ready():
                        from ..gate_domain_buckets import domain_buckets_from_batch

                        logits = prototype_bank.logits(
                            embed, domain_buckets=domain_buckets_from_batch(batch)
                        )
                        if gate4_ms_only is not None:
                            loss = gate4_ms_only(embed, y)["slice_ms"]
                        else:
                            loss = F.cross_entropy(logits, y, weight=class_weights)
                    else:
                        logits = _forward_classifier(model, batch)
                        loss = F.cross_entropy(logits, y, weight=class_weights)
                else:
                    logits = _forward_classifier(model, batch)
                    loss = F.cross_entropy(logits, y, weight=class_weights)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(y.cpu().numpy())
            n_batch = batch.features.size(0) if batch.features is not None else batch.rgb.size(0)
            all_paths.extend([batch.image_path] * n_batch)
            if batch.tile_edges:
                all_edges.extend(batch.tile_edges)
            total_loss += float(loss.item()) * (
                batch.features.size(0) if batch.features is not None else batch.rgb.size(0)
            )
            n += batch.features.size(0) if batch.features is not None else batch.rgb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", tiles=n)
    pbar.close()
    logits = np.concatenate(all_logits) if all_logits else np.zeros((0, len(GATE_CLASS_NAMES)))
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)
    metrics = (
        compute_gate_metrics(logits, labels, protocol=protocol)
        if len(labels)
        else {"acc": 0, "macro_f1": 0, "per_class_recall": {}}
    )
    if len(labels) and len(all_paths) == len(labels):
        pred = logits.argmax(axis=1)
        metrics["per_image_bg"] = compute_per_image_bg_diagnostics(
            labels, pred, np.asarray(all_paths, dtype=object)
        )
        if all_edges and len(all_edges) == len(labels):
            metrics["by_tile_edge"] = compute_by_tile_edge_diagnostics(
                labels, pred, np.asarray(all_edges, dtype=np.int32)
            )
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def _probe_lr_for_epoch(protocol: GateTrainProtocol, epoch: int, base_lr: float) -> float:
    """Linear warmup + cosine decay (estabiliza probe Slice-MS)."""
    warmup = max(int(protocol.probe_lr_warmup_epochs), 0)
    min_factor = float(protocol.probe_lr_min_factor)
    if warmup > 0 and epoch <= warmup:
        return base_lr * (epoch / warmup)
    span = max(protocol.max_epochs - warmup, 1)
    progress = min(max((epoch - warmup) / span, 0.0), 1.0)
    min_lr = base_lr * min_factor
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos


def _focal_class_weights(
    train_df: pd.DataFrame,
    device: torch.device,
    protocol: GateTrainProtocol,
) -> Optional[torch.Tensor]:
    if protocol.use_class_weights_train:
        weights = _class_weights_from_df(train_df, device).clone()
    elif protocol.mplus_focal_boost != 1.0:
        weights = torch.ones(len(GATE_CLASS_NAMES), dtype=torch.float32, device=device)
    else:
        return None
    if protocol.mplus_focal_boost != 1.0:
        idx = GATE_CLASS_TO_IDX["Mplus"]
        weights[idx] = weights[idx] * float(protocol.mplus_focal_boost)
    return weights


def _plan_train_epoch(
    train_df: pd.DataFrame,
    *,
    max_bg_per_image: int,
    balance_mode: str,
    mplus_oversample: float,
    batch_size: int,
    protocol: GateTrainProtocol,
    seed: int,
) -> EpochPlan:
    if balance_mode == "g1_stratified":
        from .gate_epoch_sampler import plan_epoch_stratified, sampler_config_from_protocol

        cfg = sampler_config_from_protocol(protocol, batch_size)
        return plan_epoch_stratified(train_df, cfg=cfg, seed=seed)
    return plan_epoch_gate(
        train_df,
        max_bg_per_image=max_bg_per_image,
        balance_mode=balance_mode,
        mplus_oversample_factor=mplus_oversample,
        seed=seed,
    )


def _plan_val_epoch(
    val_df: pd.DataFrame,
    *,
    eval_balance_mode: str,
    batch_size: int,
    protocol: GateTrainProtocol,
    seed: int,
) -> EpochPlan:
    """Plan de eval: holdout natural o estratificado (alineado con train)."""
    if eval_balance_mode == "g1_stratified":
        from .gate_epoch_sampler import eval_sampler_config_from_protocol, plan_epoch_stratified

        cfg = eval_sampler_config_from_protocol(protocol, batch_size)
        return plan_epoch_stratified(val_df, cfg=cfg, seed=seed)
    return plan_epoch_gate(val_df, max_bg_per_image=None, shuffle_images=False, seed=seed)


def _eval_plans_for_epoch(
    val_df: pd.DataFrame,
    *,
    batch_size: int,
    protocol: GateTrainProtocol,
    seed: int,
) -> tuple[EpochPlan, Optional[EpochPlan], EpochPlan]:
    """Retorna (holdout_natural, val_stratified|None, checkpoint_plan)."""
    holdout = _plan_val_epoch(
        val_df, eval_balance_mode="natural", batch_size=batch_size, protocol=protocol, seed=0
    )
    mode = protocol.eval_balance_mode
    if mode == "dual":
        stratified = _plan_val_epoch(
            val_df,
            eval_balance_mode="g1_stratified",
            batch_size=batch_size,
            protocol=protocol,
            seed=seed,
        )
        ckpt = stratified if protocol.checkpoint_eval == "stratified" else holdout
        return holdout, stratified, ckpt
    if mode == "g1_stratified":
        stratified = _plan_val_epoch(
            val_df,
            eval_balance_mode="g1_stratified",
            batch_size=batch_size,
            protocol=protocol,
            seed=seed,
        )
        return holdout, stratified, stratified
    return holdout, None, holdout


def _should_run_holdout_eval(
    protocol: GateTrainProtocol,
    *,
    epoch: int,
    epochs_total: int,
    force_final: bool = False,
) -> bool:
    """Holdout natural (~613 batches) solo cuando aporta; evita eval masiva cada epoca."""
    if protocol.eval_balance_mode == "g1_stratified":
        return False
    if protocol.eval_balance_mode == "natural":
        return True
    if force_final or epoch >= epochs_total:
        return True
    every = int(protocol.holdout_eval_every_n_epochs)
    if every > 0 and epoch % every == 0:
        return True
    return False


def _metrics_summary_line(prefix: str, val: dict, protocol: GateTrainProtocol) -> str:
    mplus = val.get("per_class_recall", {}).get("Mplus", float("nan"))
    diag = val.get("diagnostics") or {}
    mplus_prec = diag.get("mplus_precision", float("nan"))
    bg2mp = int(diag.get("bg_to_mplus", 0))
    m2mp = int(diag.get("mminus_to_mplus", 0))
    per_img = val.get("per_image_bg") or {}
    bg_p10 = per_img.get("bg_recall_p10", float("nan"))
    worst = per_img.get("worst_bg_image", "")
    worst_r = per_img.get("worst_bg_recall", float("nan"))
    line = (
        f"{prefix}: acc={val['acc']:.4f} macro_f1={val['macro_f1']:.4f} "
        f"min_recall={val.get('min_class_recall', 0):.4f} M+={mplus:.3f} "
        f"M+prec={mplus_prec:.3f} M2M+={m2mp} BG2M+={bg2mp}"
    )
    if worst and not (isinstance(worst_r, float) and np.isnan(worst_r)):
        line += f" BG_p10={bg_p10:.3f} worst={worst}({worst_r:.2f})"
    return line


def train_gate_tile_dino_gpu(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    *,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    backbone_lr_factor: float = 0.1,
    checkpoint_dir: Optional[Path] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    max_bg_per_image: int = 50,
    grad_clip: float = 5.0,
    use_amp: bool = True,
    h5_store: Optional[object] = None,
    embed_store: Optional[object] = None,
    freeze_backbone: bool = False,
    probe_lr: float = 1e-3,
    protocol: Optional[GateTrainProtocol] = None,
    save_live_snapshots: bool = True,
    live_snapshot_dir: Optional[Path] = None,
    dino_input_size: int = 252,
    seg_target_size: int = 360,
    gate4_config: Optional[object] = None,
    run_live: Optional[object] = None,
    u2net_saliency: Optional[object] = None,
    pooling_mode: str = "none",
    force_cpu_decode: bool = False,
    cpu_decode_above_mb: float = 300.0,
) -> GPUTrainHistory:
    protocol = protocol or GateTrainProtocol(max_epochs=epochs, max_bg_per_image=max_bg_per_image)
    epochs = protocol.max_epochs
    max_bg_per_image = protocol.max_bg_per_image
    balance_mode = protocol.balance_mode
    ckpt_metric = protocol.checkpoint_metric
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir and save_live_snapshots and live_snapshot_dir is None:
        live_snapshot_dir = checkpoint_dir

    model.to(device)

    if freeze_backbone:
        # Solo se congela el backbone DINO pesado. El slice-encoder + cabeza del
        # probe (gate4) DEBEN entrenarse: son el espacio Slice-MS que da la
        # separacion M+/M-. Congelar por "head not in name" dejaba el encoder en
        # init aleatorio (solo ~516 params entrenables) y mataba el gradiente MS.
        frozen = 0
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad = False
                frozen += 1
        if frozen == 0:
            log.info("[Gate tile] probe sin backbone congelable: encoder+head entrenables")

    class_weights = _focal_class_weights(train_df, device, protocol)
    eval_weights = None if protocol.eval_unweighted_loss else class_weights

    from .losses import FocalLossCE

    train_loss_fn: Optional[nn.Module]
    slice_ms_only = protocol.loss_type == "slice_ms_only"
    if slice_ms_only:
        train_loss_fn = None
    elif protocol.loss_type == "focal_ce":
        train_loss_fn = FocalLossCE(gamma=protocol.focal_gamma, weight=class_weights)
    else:
        train_loss_fn = None

    gate4_combined: Optional[nn.Module] = None
    gate4_ms_only: Optional[nn.Module] = None
    prototype_bank: Optional[object] = None
    slice_dino_finetune = _is_slice_dino_model(model) and not freeze_backbone
    use_gate4_ms = (
        gate4_config is not None
        and getattr(gate4_config, "enabled", False)
        and (embed_store is not None or slice_dino_finetune)
    )
    if use_gate4_ms:
        if slice_ms_only:
            from .gate4.training import Gate4SliceMSLossOnly
            from .gate_metric_inference import prototype_bank_from_gate4

            gate4_ms_only = Gate4SliceMSLossOnly(gate4_config)
            prototype_bank = prototype_bank_from_gate4(
                gate4_config, num_classes=len(GATE_CLASS_NAMES), device=device
            )
            k_by_class = getattr(gate4_config, "proto_subcenters_per_class", ()) or ()
            for name, p in model.named_parameters():
                if "gate_head" in name:
                    p.requires_grad = False
            _conf = getattr(gate4_config, "confusable_pairs", ())
            log.info(
                f"[Gate tile] Slice-MS PURO: slices={gate4_config.num_slices} "
                f"hard_mining={getattr(gate4_config, 'hard_mining', False)} "
                f"confusable_pairs={_conf} neg_w={getattr(gate4_config, 'confusable_neg_weight', 1.0)} "
                f"guard_w={getattr(gate4_config, 'confusable_guard_neg_weight', 1.0)} "
                f"conf_base={getattr(gate4_config, 'confusable_base', None)} "
                f"band=[{getattr(gate4_config, 'confusable_band_low', None)},"
                f"{getattr(gate4_config, 'confusable_band_high', None)}] "
                f"subcenters={k_by_class or getattr(gate4_config, 'proto_subcenters', 1)} "
                f"domain_aware={getattr(gate4_config, 'domain_aware_subcenters', False)} "
                f"inferencia={protocol.metric_inference} (sin CE/focal)"
            )
            if protocol.metric_inference == "knn":
                log.warning(
                    "[Gate tile] GATE_METRIC_INFERENCE=knn: entrenamiento usa prototipos EMA; "
                    "kNN disponible en gate_embed_analysis post-train."
                )
        else:
            from .gate4.training import Gate4CombinedLoss

            gate4_combined = Gate4CombinedLoss(
                gate4_config,
                gate_loss_fn=train_loss_fn,
                class_weights=class_weights if train_loss_fn is None else None,
            )
            log.info(
                f"[Gate tile] Gate4 Slice MS (legacy CE+MS): slices={gate4_config.num_slices} "
                f"weight={gate4_config.loss_weight} warmup={gate4_config.warmup_epochs}"
            )

    u2net_model = None
    if u2net_saliency is not None:
        log.info(
            "[Gate tile] U2Net recibido pero DESACTIVADO en train loop "
            "(regla operativa: sin inferencia U2Net en runtime, mean-pool puro)"
        )

    if slice_dino_finetune:
        from .gate_dino_finetune import collect_trainable_param_groups

        param_groups = collect_trainable_param_groups(
            model,
            head_lr=protocol.finetune_lr,
            backbone_lr=protocol.finetune_lr,
            backbone_lr_factor=backbone_lr_factor,
        )
    else:
        backbone_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "head" in name or "encoder" in name:
                head_params.append(p)
            else:
                backbone_params.append(p)
        param_groups = []
        head_lr = protocol.probe_lr if freeze_backbone else protocol.finetune_lr
        finetune_lr = protocol.finetune_lr if not freeze_backbone else lr
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": finetune_lr * backbone_lr_factor})
    head_lr = protocol.probe_lr if freeze_backbone else protocol.finetune_lr
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "probe" if freeze_backbone else "finetune"
    log.info(
        f"[Gate tile] modo={mode} balance={balance_mode} loss={protocol.loss_type} "
        f"checkpoint={ckpt_metric} early_stop={protocol.early_stop_patience} "
        f"params_entrenables={trainable:,}"
    )
    from .gate_vision import describe_dino_resolution

    log.info(f"[Gate tile] DINO vision: {describe_dino_resolution(dino_input_size)}")

    history = GPUTrainHistory()
    best_state: Optional[dict] = None
    epoch_times: list[float] = []
    epoch_details: list[dict] = []
    patience_counter = 0
    best_mplus_recall = 0.0
    best_map_score = 0.0
    min_delta = float(protocol.early_stop_min_delta)
    progress_path = (checkpoint_dir / PROGRESS_NAME) if checkpoint_dir else None

    mplus_oversample = protocol.mplus_oversample_factor
    log.info("[Gate tile] Planificando época de referencia (sampler + eval dual)...")
    ref_train = _plan_train_epoch(
        train_df,
        max_bg_per_image=max_bg_per_image,
        balance_mode=balance_mode,
        mplus_oversample=mplus_oversample,
        batch_size=batch_size,
        protocol=protocol,
        seed=1,
    )
    ref_val = plan_epoch_gate(val_df, max_bg_per_image=None, shuffle_images=False, seed=0)
    ref_val_strat = None
    ref_ckpt = ref_val
    if protocol.eval_balance_mode in ("dual", "g1_stratified"):
        from .gate_epoch_sampler import eval_sampler_config_from_protocol, plan_epoch_stratified

        scfg_eval = eval_sampler_config_from_protocol(protocol, batch_size)
        ref_val_strat = plan_epoch_stratified(val_df, cfg=scfg_eval, seed=0)
        _, _, ref_ckpt = _eval_plans_for_epoch(
            val_df, batch_size=batch_size, protocol=protocol, seed=0
        )
    train_batches_ref = _batch_total(ref_train, batch_size, max_train_batches)
    val_batches_ref = _batch_total(ref_ckpt, batch_size, max_val_batches)
    val_strat_batches_ref = (
        _batch_total(ref_val_strat, batch_size, max_val_batches) if ref_val_strat else 0
    )
    log.info(
        f"[Gate tile] Por epoca: ~{ref_train.n_tiles:,} tiles train "
        f"({train_batches_ref:,} batches)"
        + (
            f" + {ref_val_strat.n_tiles:,} val_strat ({val_strat_batches_ref:,} batches)"
            if ref_val_strat is not None
            else ""
        )
        + (
            f" + {ref_ckpt.n_tiles:,} ckpt ({val_batches_ref:,} batches)"
            if protocol.eval_balance_mode == "g1_stratified"
            else ""
        )
        + (
            f" + {ref_val.n_tiles:,} holdout ({_batch_total(ref_val, batch_size, max_val_batches):,} batches)"
            if protocol.eval_balance_mode in ("natural", "dual")
            else ""
        )
    )
    train_counts = plan_epoch_class_counts(ref_train)
    log.info(
        f"[Gate tile] Distribucion train/epoca: "
        f"Bg={train_counts.get('Background', 0):,} "
        f"M-={train_counts.get('Mminus', 0):,} "
        f"M+={train_counts.get('Mplus', 0):,} "
        f"Unr={train_counts.get('Unreadable', 0):,}"
    )
    if class_weights is not None:
        log.info(f"[Gate tile] class_weights={class_weights.detach().cpu().tolist()}")
    if mplus_oversample != 1.0:
        log.info(f"[Gate tile] mplus_oversample_factor={mplus_oversample}")
    if balance_mode == "g1_stratified":
        from .gate_epoch_sampler import audit_stratified_batches, sampler_config_from_protocol

        scfg = sampler_config_from_protocol(protocol, batch_size)
        audit = audit_stratified_batches(ref_train)
        dom_note = " domain_stratified=ON" if scfg.domain_stratified else ""
        log.info(
            f"[Gate tile] g1_stratified: {scfg.samples_per_class:,}/clase/epoca "
            f"min_per_batch={scfg.min_per_class_per_batch} "
            f"batches={audit.get('n_batches', 0)} "
            f"M+_per_batch={audit.get('Mplus_per_batch_min')}-{audit.get('Mplus_per_batch_max')}"
            f"{dom_note}"
        )
    if ref_val_strat is not None:
        from .gate_epoch_sampler import audit_stratified_batches, eval_sampler_config_from_protocol

        scfg_eval = eval_sampler_config_from_protocol(protocol, batch_size)
        audit_v = audit_stratified_batches(ref_val_strat)
        log.info(
            f"[Gate tile] eval {protocol.eval_balance_mode}: val_strat "
            f"{scfg_eval.samples_per_class:,}/clase "
            f"batches={audit_v.get('n_batches', 0)} | "
            f"checkpoint_eval={protocol.checkpoint_eval}"
        )
    log.info(
        f"[Gate tile] LR probe: base={head_lr:g} warmup={protocol.probe_lr_warmup_epochs} "
        f"min_factor={protocol.probe_lr_min_factor}"
    )
    log.info(
        f"[Gate tile] Protocolo: max_epochs={epochs} min_epochs={protocol.min_epochs} "
        f"class_weights_train={protocol.use_class_weights_train} eval_loss_unweighted={protocol.eval_unweighted_loss}"
    )
    if progress_path:
        _write_training_progress(
            progress_path,
            {
                "status": "running",
                "epochs_total": epochs,
                "epoch_current": 0,
                "batches_train_per_epoch": train_batches_ref,
                "batches_test_per_epoch": val_batches_ref,
                "tiles_train_per_epoch": ref_train.n_tiles,
                "tiles_test_per_epoch": ref_ckpt.n_tiles,
            },
            run_live=run_live,
        )

    if h5_store is not None:
        h5_mode = (
            "g1_stratified"
            if balance_mode == "g1_stratified"
            else "por-imagen"
        )
        log.info(f"[Gate tile] Entrenamiento desde HDF5 ({h5_mode}, luma+label)")
    elif embed_store is not None:
        log.info("[Gate tile] Entrenamiento desde cache embeddings (DINO precomputado)")
        if hasattr(model, "backbone"):
            model.backbone.cpu()
    elif slice_dino_finetune:
        log.info(
            f"[Gate tile] Entrenamiento Slice-DINO on-the-fly "
            f"(E6/E6b finetune, pooling={pooling_mode})"
        )

    # Baseline pre-entrenamiento: solo probe Slice-MS (sin prototipos EMA aun).
    pretrain_metrics: Optional[dict] = None
    if embed_store is not None and not slice_ms_only:
        baseline_batches = _batch_total(ref_val, batch_size, max_val_batches)
        log.info(
            f"[Gate tile] === BASELINE pre-entrenamiento (probe sin entrenar) === "
            f"holdout: {baseline_batches} batches, ~{ref_val.n_tiles:,} tiles"
        )
        model.eval()
        pretrain_metrics = evaluate_gate_tile_dino_gpu(
            model,
            ref_val,
            device,
            batch_size=batch_size,
            class_weights=eval_weights,
            max_batches=max_val_batches,
            desc="baseline ep0",
            embed_store=embed_store,
            protocol=protocol,
            dino_input_size=dino_input_size,
            seg_target_size=seg_target_size,
            prototype_bank=prototype_bank,
            slice_ms_only=slice_ms_only,
            gate4_ms_only=gate4_ms_only,
        )
        score0 = checkpoint_score(
            pretrain_metrics,
            ckpt_metric,
            composite_bg_weight=protocol.checkpoint_composite_bg_weight,
            tile_edge_weight=protocol.checkpoint_tile_edge_weight,
            tile_edge_targets=protocol.checkpoint_tile_edge_targets,
        )
        log.info(
            f"[Gate tile] BASELINE ep0: macro_f1={pretrain_metrics['macro_f1']:.4f} "
            f"bal_acc={pretrain_metrics.get('balanced_accuracy', 0):.4f} "
            f"{ckpt_metric}={score0:.4f} | {format_g1_status(pretrain_metrics, protocol)}"
        )
        if live_snapshot_dir:
            _write_live_training_snapshot(
                live_snapshot_dir,
                history,
                epoch_details=[],
                pretrain=pretrain_metrics,
                ckpt_metric=ckpt_metric,
            )
        if run_live is not None:
            run_live.publish_baseline(pretrain_metrics)
    elif embed_store is not None and slice_ms_only:
        log.info(
            "[Gate tile] BASELINE ep0 omitido (slice_ms_only): prototipos vacios; "
            "la inferencia metrica arranca en ep1 tras EMA de subcentros."
        )

    for ep in range(1, epochs + 1):
        ep_lr = _probe_lr_for_epoch(protocol, ep, head_lr)
        for pg in param_groups:
            pg["lr"] = ep_lr

        train_plan = _plan_train_epoch(
            train_df,
            max_bg_per_image=max_bg_per_image,
            balance_mode=balance_mode,
            mplus_oversample=mplus_oversample,
            batch_size=batch_size,
            protocol=protocol,
            seed=ep,
        )
        holdout_plan, val_strat_plan, ckpt_plan = _eval_plans_for_epoch(
            val_df, batch_size=batch_size, protocol=protocol, seed=ep
        )
        train_batches_ep = _batch_total(train_plan, batch_size, max_train_batches)
        val_batches_ep = _batch_total(ckpt_plan, batch_size, max_val_batches)

        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        batch_idx = 0
        if progress_path:
            _write_training_progress(
                progress_path,
                {
                    "status": "running",
                    "epochs_total": epochs,
                    "epoch_current": ep,
                    "batch_current": 0,
                    "batches_train_this_epoch": train_batches_ep,
                    "batches_test_this_epoch": val_batches_ep,
                    "detail": f"Ep {ep}/{epochs} iniciando batches ({train_batches_ep} train)",
                },
                run_live=run_live,
            )
        pbar = tqdm(
            _batch_iterator(
                train_plan,
                batch_size=batch_size,
                device=device,
                max_batches=max_train_batches,
                h5_store=h5_store,
                embed_store=embed_store,
                dino_input_size=dino_input_size,
                seg_target_size=seg_target_size,
                u2net_saliency=u2net_model,
                force_cpu_decode=force_cpu_decode,
                cpu_decode_above_mb=cpu_decode_above_mb,
            ),
            total=train_batches_ep,
            desc=f"train ep{ep}/{epochs}",
            dynamic_ncols=True,
            unit="batch",
        )
        for batch in pbar:
            batch_idx += 1
            y = batch.labels.long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if gate4_ms_only is not None:
                    _, embed = _forward_classifier(model, batch, return_embed=True)
                    if embed is None:
                        raise RuntimeError("Slice-MS puro requiere embeddings del probe (cache)")
                    out = gate4_ms_only(embed, y, epoch=ep)
                    loss = out["total"]
                    from ..gate_domain_buckets import domain_buckets_from_batch

                    prototype_bank.update(embed, y, domain_buckets=domain_buckets_from_batch(batch))
                elif gate4_combined is not None:
                    logits, embed = _forward_classifier(model, batch, return_embed=True)
                    if embed is None:
                        logits = _forward_classifier(model, batch)
                        if train_loss_fn is not None:
                            loss = train_loss_fn(logits, y)
                        else:
                            loss = F.cross_entropy(logits, y, weight=class_weights)
                    else:
                        out = gate4_combined(logits, embed, y, epoch=ep)
                        loss = out["total"]
                else:
                    logits = _forward_classifier(model, batch)
                    if train_loss_fn is not None:
                        loss = train_loss_fn(logits, y)
                    else:
                        loss = F.cross_entropy(logits, y, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_n = batch.features.size(0) if batch.features is not None else batch.rgb.size(0)
            running += float(loss.item()) * batch_n
            n += batch_n
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                avg=f"{running / max(n, 1):.3f}",
                tiles=n,
            )
            if progress_path and (batch_idx == 1 or batch_idx % 20 == 0):
                elapsed_ep = time.time() - t0
                eta_ep = (elapsed_ep / batch_idx) * max(train_batches_ep - batch_idx, 0)
                eta_all = 0.0
                if epoch_times:
                    avg_ep = sum(epoch_times) / len(epoch_times)
                    eta_all = avg_ep * (epochs - ep + 1) - elapsed_ep + eta_ep
                elif batch_idx >= 5:
                    eta_all = (elapsed_ep / batch_idx) * train_batches_ep * epochs
                _write_training_progress(
                    progress_path,
                    {
                        "status": "running",
                        "epochs_total": epochs,
                        "epoch_current": ep,
                        "batch_current": batch_idx,
                        "batches_train_this_epoch": train_batches_ep,
                        "batches_test_this_epoch": val_batches_ep,
                        "elapsed_epoch_s": round(elapsed_ep, 1),
                        "eta_epoch_s": round(eta_ep, 1),
                        "eta_total_s": round(eta_all, 1) if eta_all else None,
                        "eta_epoch": _format_duration(eta_ep),
                        "eta_total": _format_duration(eta_all) if eta_all else None,
                        "train_loss_avg": round(running / max(n, 1), 4),
                        "tiles_train_per_epoch": train_plan.n_tiles,
                    },
                    run_live=run_live,
                )
        pbar.close()

        train_loss = running / max(n, 1)
        eval_parts = []
        run_holdout = _should_run_holdout_eval(protocol, epoch=ep, epochs_total=epochs)
        if run_holdout:
            eval_parts.append(f"holdout ({_batch_total(holdout_plan, batch_size, max_val_batches)} batches)")
        if val_strat_plan is not None:
            eval_parts.append(
                f"val_strat ({_batch_total(val_strat_plan, batch_size, max_val_batches)} batches)"
            )
        if not run_holdout and val_strat_plan is None:
            eval_parts.append(
                f"checkpoint ({_batch_total(ckpt_plan, batch_size, max_val_batches)} batches)"
            )
        log.info(
            f"[Gate tile] ep {ep}/{epochs}: train loss={train_loss:.4f}; "
            f"eval: {', '.join(eval_parts) or 'checkpoint'}"
        )

        def _run_eval(plan: EpochPlan, desc: str) -> dict:
            return evaluate_gate_tile_dino_gpu(
                model,
                plan,
                device,
                batch_size=batch_size,
                class_weights=eval_weights,
                max_batches=max_val_batches,
                desc=desc,
                h5_store=h5_store,
                embed_store=embed_store,
                protocol=protocol,
                dino_input_size=dino_input_size,
                seg_target_size=seg_target_size,
                u2net_saliency=u2net_model,
                prototype_bank=prototype_bank,
                slice_ms_only=slice_ms_only,
                gate4_ms_only=gate4_ms_only,
                force_cpu_decode=force_cpu_decode,
                cpu_decode_above_mb=cpu_decode_above_mb,
            )

        val_holdout: Optional[dict] = None
        val_strat: Optional[dict] = None
        if run_holdout:
            if run_live is not None:
                run_live.publish_eval_phase("holdout", ep, epochs)
            val_holdout = _run_eval(holdout_plan, f"holdout ep{ep}/{epochs}")
        if val_strat_plan is not None:
            if run_live is not None:
                run_live.publish_eval_phase("stratified", ep, epochs)
            val_strat = _run_eval(val_strat_plan, f"val_strat ep{ep}/{epochs}")

        if protocol.checkpoint_eval == "stratified":
            val_ckpt = val_strat if val_strat is not None else _run_eval(
                ckpt_plan, f"ckpt ep{ep}/{epochs}"
            )
        elif val_holdout is not None:
            val_ckpt = val_holdout
        elif val_strat is not None:
            val_ckpt = val_strat
        else:
            val_ckpt = _run_eval(ckpt_plan, f"ckpt ep{ep}/{epochs}")

        val = val_holdout if val_holdout is not None else val_ckpt

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        avg_ep = sum(epoch_times) / len(epoch_times)
        eta_remaining = avg_ep * (epochs - ep)

        history.epochs.append(ep)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_ckpt["loss"])
        history.val_auroc.append(val_ckpt["macro_f1"])
        history.val_acc.append(val_ckpt["acc"])
        history.val_f1.append(val_ckpt["macro_f1"])
        history.elapsed_s.append(elapsed)

        score = checkpoint_score(
            val_ckpt,
            ckpt_metric,
            composite_bg_weight=protocol.checkpoint_composite_bg_weight,
            tile_edge_weight=protocol.checkpoint_tile_edge_weight,
            tile_edge_targets=protocol.checkpoint_tile_edge_targets,
        )
        pcr_ckpt = val_ckpt.get("per_class_recall") or {}
        mplus_r = float(pcr_ckpt.get("Mplus") or 0.0)
        mminus_r = float(pcr_ckpt.get("Mminus") or 0.0)
        map_score = float(val_ckpt.get("mAP") or 0.0)

        if mplus_r < 0.65 and mminus_r > 0.88 and ep >= protocol.min_epochs:
            log.warning(
                f"[Gate tile] Colapso M+/M- ep{ep}: M+={mplus_r:.3f} M-={mminus_r:.3f} "
                f"(posible atajo hacia M-)"
            )

        def _build_ckpt_payload(state_dict: dict, epoch: int, score_val: float) -> dict:
            payload: dict = {
                "model_state_dict": state_dict,
                "epoch": epoch,
                "acc": val_ckpt["acc"],
                "macro_f1": val_ckpt["macro_f1"],
                "balanced_accuracy": val_ckpt.get("balanced_accuracy"),
                "min_class_recall": val_ckpt.get("min_class_recall"),
                "mAP": val_ckpt.get("mAP"),
                "per_class_ap": val_ckpt.get("per_class_ap"),
                "evangelisti_g1_pass": val_ckpt.get("evangelisti_g1_pass"),
                "evangelisti_g1_score": val_ckpt.get("evangelisti_g1_score"),
                "checkpoint_metric": ckpt_metric,
                "checkpoint_score": score_val,
                "checkpoint_eval": protocol.checkpoint_eval,
                "eval_balance_mode": protocol.eval_balance_mode,
                "per_class_recall": val_ckpt.get("per_class_recall"),
                "per_class_specificity": val_ckpt.get("per_class_specificity"),
                "num_classes": len(GATE_CLASS_NAMES),
                "gate_mode": "tile_dino",
                "pipeline": "dinov2_embed_cache+slice_ms",
                "train_mode": mode,
                "balance_mode": balance_mode,
                "freeze_backbone": freeze_backbone,
                "loss_type": protocol.loss_type,
            }
            if prototype_bank is not None:
                payload["prototype_bank"] = prototype_bank.state_dict()
                payload["metric_inference"] = protocol.metric_inference
                payload["proto_subcenters"] = int(getattr(prototype_bank, "num_subcenters", 1))
                payload["proto_subcenters_per_class"] = list(
                    getattr(prototype_bank, "num_subcenters_per_class", ())
                )
                payload["domain_aware_subcenters"] = bool(getattr(prototype_bank, "domain_aware", False))
            return payload

        improved_primary = score > history.best_val_auroc + min_delta
        improved_mplus = mplus_r > best_mplus_recall + min_delta
        improved = improved_primary
        if improved_primary:
            history.best_val_auroc = score
            history.best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(
                    _build_ckpt_payload(best_state, ep, score),
                    checkpoint_dir / CHECKPOINT_NAME,
                )
        if map_score > best_map_score + min_delta:
            best_map_score = map_score
            if checkpoint_dir:
                map_state = (
                    best_state
                    if best_state is not None
                    else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                )
                map_payload = _build_ckpt_payload(map_state, ep, map_score)
                map_payload["checkpoint_metric"] = "mAP"
                map_payload["checkpoint_score"] = map_score
                torch.save(map_payload, checkpoint_dir / CHECKPOINT_MAP_NAME)
        if improved_mplus:
            best_mplus_recall = mplus_r
        if improved_primary or improved_mplus:
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_row: dict = {
                "epoch": ep,
                "checkpoint_score": score,
                "mAP": val_ckpt.get("mAP"),
                "per_class_ap": val_ckpt.get("per_class_ap"),
                "balanced_accuracy": val_ckpt.get("balanced_accuracy"),
                "min_class_recall": val_ckpt.get("min_class_recall"),
                "min_class_specificity": val_ckpt.get("min_class_specificity"),
                "evangelisti_g1_pass": val_ckpt.get("evangelisti_g1_pass"),
                "per_class_recall": val_ckpt.get("per_class_recall"),
                "per_class_specificity": val_ckpt.get("per_class_specificity"),
                "diagnostics": val_ckpt.get("diagnostics"),
                "per_image_bg": val_ckpt.get("per_image_bg"),
                "checkpoint_eval": protocol.checkpoint_eval,
            }
        if val_holdout is not None:
            epoch_row["holdout_natural"] = {
                "acc": val_holdout["acc"],
                "macro_f1": val_holdout["macro_f1"],
                "min_class_recall": val_holdout.get("min_class_recall"),
                "per_class_recall": val_holdout.get("per_class_recall"),
                "diagnostics": val_holdout.get("diagnostics"),
                "per_image_bg": val_holdout.get("per_image_bg"),
            }
        if val_strat is not None:
            epoch_row["val_stratified"] = {
                "acc": val_strat["acc"],
                "macro_f1": val_strat["macro_f1"],
                "min_class_recall": val_strat.get("min_class_recall"),
                "per_class_recall": val_strat.get("per_class_recall"),
            }
        epoch_details.append(epoch_row)

        flag = "*" if improved else " "
        holdout_line = _metrics_summary_line("holdout" if val_holdout else "eval", val, protocol)
        strat_line = (
            _metrics_summary_line("val_strat", val_strat, protocol) if val_strat is not None else ""
        )
        log.info(
            f"[gate_tile_dino] ep {ep}/{epochs} {flag} "
            f"train_loss={train_loss:.4f} {holdout_line} "
            f"{ckpt_metric}={score:.4f} (best={history.best_val_auroc:.4f} ep{history.best_epoch}) "
            f"G1={'PASS' if val_ckpt.get('evangelisti_g1_pass') else 'FAIL'} "
            f"| {_format_duration(elapsed)} ETA ~{_format_duration(eta_remaining)}"
        )
        if strat_line:
            log.info(f"[gate_tile_dino] ep {ep}/{epochs}   {strat_line}")
        def _g(d: dict, k: str) -> str:
            v = d.get(k) if isinstance(d, dict) else None
            try:
                return f"{float(v):.3f}" if v is not None and float(v) == float(v) else "—"
            except (TypeError, ValueError):
                return "—"

        pcr = val_ckpt.get("per_class_recall") or {}
        pca = val_ckpt.get("per_class_ap") or {}
        improved_mark = "<<BEST" if improved else ""
        print(
            f"\n{'=' * 56}\n"
            f"  EPOCH {ep}/{epochs} DONE ({_format_duration(elapsed)})\n"
            f"  loss_train={train_loss:.4f}  val_loss={_g(val_ckpt, 'loss')}  "
            f"acc={_g(val_ckpt, 'acc')}  macro_f1={_g(val_ckpt, 'macro_f1')}\n"
            f"  {ckpt_metric}={score:.4f} (best={history.best_val_auroc:.4f} "
            f"@ep{history.best_epoch}) {improved_mark}  "
            f"min_recall={_g(val_ckpt, 'min_class_recall')}  "
            f"G1={'PASS' if val_ckpt.get('evangelisti_g1_pass') else 'FAIL'}\n"
            f"  recall  Bg={_g(pcr, 'Background')} M-={_g(pcr, 'Mminus')} "
            f"M+={_g(pcr, 'Mplus')} Unr={_g(pcr, 'Unknown')}\n"
            f"  AP      Bg={_g(pca, 'Background')} M-={_g(pca, 'Mminus')} "
            f"M+={_g(pca, 'Mplus')} Unr={_g(pca, 'Unknown')}"
            + ("\n  checkpoint guardado (best)" if improved else "")
            + f"\n{'=' * 56}",
            flush=True,
        )
        if progress_path:
            _write_training_progress(
                progress_path,
                {
                    "status": "running" if ep < epochs else "epoch_done",
                    "epochs_total": epochs,
                    "epoch_current": ep,
                    "batches_train_per_epoch": train_batches_ep,
                    "batches_test_per_epoch": val_batches_ep,
                    "epoch_duration_s": round(elapsed, 1),
                    "avg_epoch_duration_s": round(avg_ep, 1),
                    "eta_remaining_s": round(eta_remaining, 1),
                    "eta_remaining": _format_duration(eta_remaining),
                    "checkpoint_metric": ckpt_metric,
                    "best_epoch": history.best_epoch,
                    "best_checkpoint_score": round(history.best_val_auroc, 4),
                    "last_test_acc": round(val["acc"], 4),
                    "last_macro_f1": round(val["macro_f1"], 4),
                    "last_balanced_accuracy": round(val.get("balanced_accuracy", 0), 4),
                    "last_evangelisti_g1_pass": bool(val.get("evangelisti_g1_pass")),
                    "epoch_details": epoch_details,
                },
                run_live=run_live,
            )

        if live_snapshot_dir and save_live_snapshots:
            _write_live_training_snapshot(
                live_snapshot_dir,
                history,
                epoch_details=epoch_details,
                pretrain=pretrain_metrics,
                ckpt_metric=ckpt_metric,
            )

        if run_live is not None:
            run_live.publish_epoch(
                epoch=ep,
                epochs_total=epochs,
                history=history,
                epoch_details=epoch_details,
                pretrain=pretrain_metrics,
                ckpt_metric=ckpt_metric,
                improved=improved,
                last_val=val,
                last_val_ckpt=val_ckpt,
                last_val_strat=val_strat,
            )

        if (
            ep >= protocol.min_epochs
            and protocol.early_stop_patience > 0
            and patience_counter >= protocol.early_stop_patience
        ):
            log.info(
                f"[Gate tile] Early stopping en ep {ep} "
                f"(sin mejora en {ckpt_metric} ni M+ recall durante "
                f"{protocol.early_stop_patience} epocas (min_delta={min_delta}); "
                f"best=ep{history.best_epoch} score={history.best_val_auroc:.4f} "
                f"M+_best={best_mplus_recall:.4f})"
            )
            if _should_run_holdout_eval(protocol, epoch=ep, epochs_total=epochs, force_final=True):
                if val_holdout is None:
                    log.info("[Gate tile] Eval holdout final pre-finalize...")
                    val_holdout = _run_eval(holdout_plan, f"holdout_final ep{ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    calibration_dict: Optional[dict] = None
    calibrated_val: Optional[dict] = None
    if protocol.calibrate_post_train and not slice_ms_only and embed_store is not None and len(val_df) > 0:
        try:
            from .gate_calibrate import fit_gate_calibration, metrics_with_calibration

            log.info("[Gate tile] Calibracion post-train (temperature + sesgos por clase)...")
            val_logits, val_labels, _, _, _ = collect_probe_logits(
                model, val_df, embed_store, device, batch_size=batch_size
            )
            calib = fit_gate_calibration(val_logits, val_labels, protocol)
            calibrated_val = metrics_with_calibration(val_logits, val_labels, calib, protocol)
            calibration_dict = calib.to_dict()
            log.info(
                f"[Gate tile] Calibrado: T={calib.temperature:.3f} bias={calib.class_bias} | "
                f"macro_f1={calibrated_val['macro_f1']:.4f} min_recall={calibrated_val['min_class_recall']:.4f} | "
                f"{format_g1_status(calibrated_val, protocol)}"
            )
            if checkpoint_dir and (checkpoint_dir / CHECKPOINT_NAME).exists():
                st = torch.load(checkpoint_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=False)
                st["calibration"] = calibration_dict
                st["calibrated_val_metrics"] = calibrated_val
                torch.save(st, checkpoint_dir / CHECKPOINT_NAME)
        except Exception:
            log.exception("[Gate tile] Calibracion post-train fallo; se continua sin calibrar")

    history.__dict__.update(
        {
            "checkpoint_metric": ckpt_metric,
            "epoch_details": epoch_details,
            "balance_mode": balance_mode,
            "loss_type": protocol.loss_type,
            "pretrain_baseline": pretrain_metrics,
            "calibration": calibration_dict,
            "calibrated_val_metrics": calibrated_val,
        }
    )
    if progress_path:
        _write_training_progress(
            progress_path,
            {
                "status": "completed",
                "epochs_total": epochs,
                "checkpoint_metric": ckpt_metric,
                "best_epoch": history.best_epoch,
                "best_checkpoint_score": round(history.best_val_auroc, 4),
                "epoch_details": epoch_details,
            },
            run_live=run_live,
        )
    if run_live is not None:
        run_live.publish_completed(history=history, ckpt_metric=ckpt_metric)
    return history


def _write_live_training_snapshot(
    out_dir: Path,
    history: GPUTrainHistory,
    *,
    epoch_details: list[dict],
    pretrain: Optional[dict],
    ckpt_metric: str,
) -> None:
    """Metricas y curvas incrementales (sin esperar al reporte final)."""
    import json
    import math
    from pathlib import Path

    from .gate_run_live import _sanitize_json_nan

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_metric": ckpt_metric,
        "best_epoch": history.best_epoch,
        "best_score": history.best_val_auroc,
        "epochs_done": history.epochs,
        "train_loss": history.train_loss,
        "val_macro_f1": history.val_f1,
        "val_acc": history.val_acc,
        "pretrain_baseline": pretrain,
        "epoch_details": epoch_details,
    }
    clean = _sanitize_json_nan(payload)
    with open(out_dir / "live_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)
    # Canvas: lo refleja el watcher externo (sync_gate_canvas.py), no el train loop.

    if len(history.epochs) >= 1:
        try:
            import matplotlib.pyplot as plt

            from .gate_pretrain_viz import legend_if_labeled

            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            if pretrain is not None:
                ep_x = [0] + history.epochs
                f1_y = [pretrain.get("macro_f1", 0)] + history.val_f1
                acc_y = [pretrain.get("acc", 0)] + history.val_acc
                axes[0].plot([0] + history.epochs, [0] + history.train_loss, marker="o", label="train")
                axes[1].plot(ep_x, acc_y, marker="s", color="C1", label="test acc")
                axes[2].plot(ep_x, f1_y, marker="^", color="C2", label="macro F1")
                axes[2].axhline(pretrain.get("macro_f1", 0), ls="--", alpha=0.5, label="baseline ep0")
            else:
                axes[0].plot(history.epochs, history.train_loss, marker="o", label="train")
                axes[1].plot(history.epochs, history.val_acc, marker="s", label="test acc")
                axes[2].plot(history.epochs, history.val_f1, marker="^", label="macro F1")
            for ax, title in zip(axes, ["Train loss", "Test acc", "Macro F1"]):
                ax.set_xlabel("Epoch")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                legend_if_labeled(ax, fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / "live_curves.png", dpi=120)
            plt.close(fig)
        except Exception as e:
            log.debug(f"[Gate tile] live_curves skip: {e}")


@torch.no_grad()
def infer_image_gate_tile_dino_gpu(
    image_path: str | Path,
    gate: GateTileDinoGPU,
    tiles_index_path: Optional[Path] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    from ..common.io import read_table
    from ..common.paths import get_paths

    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = df[df["image_path"] == rel].copy().reset_index(drop=True)
    if sub.empty:
        log.warning(f"No hay tiles para {rel} en el manifest")
        return sub

    plan = plan_epoch_gate(sub, max_bg_per_image=None, shuffle_images=False, shuffle_tiles_within_image=False)
    dev = gate.device
    probs_all, rows_all, cols_all = [], [], []

    iterator = tqdm(
        iter_image_batches(
            plan,
            batch_size=batch_size,
            device=dev,
            label_mode="gate",
            u2net_saliency=None,
        ),
        total=(len(sub) + batch_size - 1) // batch_size,
        desc=f"gate-tile {Path(rel).name}",
        dynamic_ncols=True,
    )
    for batch in iterator:
        logits = _forward_classifier(gate.classifier, batch)
        probs_all.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        rows_all.extend(batch.rows.cpu().tolist())
        cols_all.extend(batch.cols.cpu().tolist())

    p = np.concatenate(probs_all) if probs_all else np.zeros((0, len(GATE_CLASS_NAMES)))
    pred_idx = p.argmax(axis=1)

    order_df = pd.DataFrame(
        {
            "row": rows_all,
            "col": cols_all,
            "p_bg": p[:, 0],
            "p_mminus": p[:, 1],
            "p_mplus": p[:, 2],
            "p_fused_max": p.max(axis=1),
            "gate_pred_idx": pred_idx,
            "stage1_pred": decode_gate_indices(pred_idx),
        }
    )
    merged = sub.merge(order_df, on=["row", "col"], how="left")
    merged["stage1"] = merged["stage1_pred"]
    merged["is_mplus"] = (merged["gate_pred_idx"] == 2).astype(int)
    if "saliency_mean" not in merged.columns:
        merged["saliency_mean"] = np.nan
    return merged


def collect_probe_logits(
    model: nn.Module,
    tiles_df: pd.DataFrame,
    embed_store: object,
    device: torch.device,
    *,
    batch_size: int = 64,
    prototype_bank: Optional[object] = None,
    slice_ms_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str]]:
    """Logits, labels y metadatos tile desde cache (orden alineado con batches)."""
    model.eval()
    if slice_ms_only and prototype_bank is not None and not prototype_bank.is_ready():
        return (
            np.zeros((0, len(GATE_CLASS_NAMES))),
            np.zeros(0, dtype=np.int64),
            [],
            [],
            [],
        )
    plan = plan_epoch_gate(tiles_df, max_bg_per_image=None, shuffle_images=False, seed=0)
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    rows_all: list[int] = []
    cols_all: list[int] = []
    paths_all: list[str] = []
    with torch.no_grad():
        for batch in _batch_iterator(
            plan,
            batch_size=batch_size,
            device=device,
            max_batches=None,
            h5_store=None,
            embed_store=embed_store,
        ):
            y = batch.labels.long()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if slice_ms_only and prototype_bank is not None:
                    _, embed = _forward_classifier(model, batch, return_embed=True)
                    if embed is not None and prototype_bank.is_ready():
                        from ..gate_domain_buckets import domain_buckets_from_batch

                        logits = prototype_bank.logits(
                            embed, domain_buckets=domain_buckets_from_batch(batch)
                        )
                    else:
                        logits = _forward_classifier(model, batch)
                else:
                    logits = _forward_classifier(model, batch)
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(y.cpu().numpy())
            rows_all.extend(batch.rows.cpu().tolist())
            cols_all.extend(batch.cols.cpu().tolist())
            paths_all.extend([batch.image_path] * batch.labels.size(0))
    if not logits_all:
        return (
            np.zeros((0, len(GATE_CLASS_NAMES))),
            np.zeros(0, dtype=np.int64),
            [],
            [],
            [],
        )
    return (
        np.concatenate(logits_all),
        np.concatenate(labels_all).astype(np.int64),
        rows_all,
        cols_all,
        paths_all,
    )


def evaluate_gate_probe_on_df(
    model: nn.Module,
    tiles_df: pd.DataFrame,
    embed_store: object,
    device: torch.device,
    *,
    batch_size: int = 64,
    calibration: Optional[dict] = None,
    prototype_bank: Optional[object] = None,
    slice_ms_only: bool = False,
) -> pd.DataFrame:
    """Inferencia rapida desde cache de embeddings (sin JPEG)."""
    from .gate_calibrate import GateCalibration, apply_gate_calibration
    from .gate_classes import decode_gate_indices

    model.eval()
    logits, _, rows_all, cols_all, paths_all = collect_probe_logits(
        model,
        tiles_df,
        embed_store,
        device,
        batch_size=batch_size,
        prototype_bank=prototype_bank,
        slice_ms_only=slice_ms_only,
    )
    if calibration:
        logits = apply_gate_calibration(logits, GateCalibration.from_dict(calibration))

    if len(logits) == 0:
        return pd.DataFrame()

    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    pred_idx = p.argmax(axis=1)
    pred_df = pd.DataFrame(
        {
            "image_path": paths_all,
            "row": rows_all,
            "col": cols_all,
            "p_bg": p[:, 0],
            "p_mminus": p[:, 1],
            "p_mplus": p[:, 2],
            "gate_pred_idx": pred_idx,
            "stage1_pred": decode_gate_indices(pred_idx),
        }
    )
    gold = tiles_df[["image_path", "row", "col", "stage1"]].copy()
    merged = gold.merge(pred_df, on=["image_path", "row", "col"], how="inner")
    merged["stage1_gold"] = merged["stage1"].astype(str)
    merged["correct"] = merged["stage1_gold"] == merged["stage1_pred"].astype(str)
    return merged


def evaluate_gate_tile_dino_on_df(
    gate: GateTileDinoGPU,
    tiles_df: pd.DataFrame,
    *,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Inferencia por imagen; conserva gold en `stage1_gold`."""
    parts = []
    for rel in tiles_df["image_path"].unique():
        sub = tiles_df[tiles_df["image_path"] == rel]
        from ..common.paths import get_paths

        full = get_paths().root / rel
        pred = infer_image_gate_tile_dino_gpu(full, gate, batch_size=batch_size)
        if pred.empty:
            continue
        gold_map = sub.set_index(["row", "col"])["stage1"]
        pred["stage1_gold"] = [
            str(gold_map.get((int(r), int(c)), "")) for r, c in zip(pred["row"], pred["col"])
        ]
        if "stage1_pred" not in pred.columns and "stage1" in pred.columns:
            pred["stage1_pred"] = pred["stage1"].astype(str)
        pred["correct"] = pred.apply(
            lambda r: (
                is_valid_stage1(r["stage1_gold"])
                and stage1_to_gate_label(str(r["stage1_gold"]).strip()) == str(r["stage1_pred"])
            ),
            axis=1,
        )
        parts.append(pred)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


@dataclass
class GateProbeBundle:
    """Gate Slice-MS probe + cache embeddings + prototipos (modo vigente)."""

    classifier: nn.Module
    embed_store: object
    prototype_bank: Optional[object]
    slice_ms_only: bool
    device: torch.device
    checkpoint_meta: dict
    gate_run_id: str = ""


def _is_probe_state_dict(state: dict) -> bool:
    keys = list(state.keys())
    return any(k.startswith("encoder.") for k in keys) and any(k.startswith("gate_head") for k in keys)


def load_gate_probe_bundle(
    *,
    device: torch.device | None = None,
    ckpt_dir: Path | None = None,
    cfg: Optional[object] = None,
    gate_run_id: str = "",
) -> GateProbeBundle:
    """Carga probe Slice-MS + memmap embed + prototipos desde checkpoint gate AM."""
    import config as user_config  # type: ignore

    from ..common.paths import get_paths
    from ..gate_runflow import (
        _cache_basename_from_cfg,
        _open_gate_h5_store,
        _probe_in_dim,
        gate_train_params_from_config,
        resolve_gate_am_splits,
    )
    from .gate4.probe_model import build_gate_slice_probe
    from .gate_embed_cache import inspect_embed_cache_status, open_gate_embed_store
    from .gate_metric_inference import prototype_bank_from_gate4

    cfg = cfg or user_config
    paths = get_paths()
    ckpt_dir = ckpt_dir or (paths.root / "models" / "checkpoints" / "gate_am")
    device = device or torch.device("cuda")
    ckpt = ckpt_dir / CHECKPOINT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Falta checkpoint: {ckpt}. Entrena con: python run.py train-gate-am"
        )

    params = gate_train_params_from_config(cfg)
    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    train_df, _val_df, _ext, _info, cache_tiles = resolve_gate_am_splits(
        cfg, exclude_unreadable=not include_unknown
    )
    train_images = set(train_df["image_path"].astype(str))

    status = inspect_embed_cache_status(
        cache_tiles,
        backbone_name=params.backbone,
        dino_input_size=params.dino_input_size,
        mplus_aug_variants=params.mplus_aug_variants,
        mplus_aug_train_images=train_images,
        cache_attention=params.cache_attention,
        attention_layers=params.attention_layers,
        attention_head_reduce=params.attention_head_reduce,
        cache_basename=_cache_basename_from_cfg(cfg),
    )
    if status.state not in {"valid", "stale", "obsolete"}:
        raise RuntimeError(f"Cache embeddings no usable ({status.state})")
    embed_store = open_gate_embed_store(status.paths)

    probe_in = _probe_in_dim(params, status.paths)
    g4 = params.gate4
    if g4 is None or not g4.enabled:
        raise RuntimeError("Gate4 deshabilitado; no se puede cargar probe Slice-MS.")
    classifier = build_gate_slice_probe(
        in_dim=probe_in,
        embed_dim=g4.embed_dim,
        num_slices=g4.num_slices,
        num_classes=len(GATE_CLASS_NAMES),
    )

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not _is_probe_state_dict(st.get("model_state_dict", {})):
        raise RuntimeError(
            "Checkpoint no es probe Slice-MS (faltan claves encoder.*). "
            "Reentrena gate o usa checkpoint gate_tile_dino_best.pt actual."
        )
    classifier.load_state_dict(st["model_state_dict"])
    classifier.to(device).eval()

    slice_ms_only = str(st.get("loss_type", "slice_ms_only")) == "slice_ms_only"
    proto = None
    if slice_ms_only and "prototype_bank" in st:
        proto = prototype_bank_from_gate4(g4, num_classes=len(GATE_CLASS_NAMES), device=device)
        proto.load_state_dict(st["prototype_bank"])

    run_id = gate_run_id or str(getattr(cfg, "STAGE2_GATE_RUN_ID", "") or "")
    log.info(
        f"[Gate probe] cargado acc={st.get('acc', 0):.4f} "
        f"macro_f1={st.get('macro_f1', 0):.4f} min_recall={st.get('min_class_recall', 0):.4f} "
        f"run_ref={run_id or 'n/a'}"
    )
    return GateProbeBundle(
        classifier=classifier,
        embed_store=embed_store,
        prototype_bank=proto,
        slice_ms_only=slice_ms_only,
        device=device,
        checkpoint_meta=st,
        gate_run_id=run_id,
    )


@torch.no_grad()
def infer_image_gate_probe_gpu(
    image_path: str | Path,
    bundle: GateProbeBundle,
    tiles_index_path: Optional[Path] = None,
    *,
    batch_size: int = 64,
    strict: bool = False,
    tile_scope_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Inferencia gate desde cache embed (probe + prototipos) para una imagen.

    ``strict=True``: si faltan tiles en embed cache, lanza error en lugar de
    rellenar con gold ``stage1`` del manifest (incorrecto para pipeline secuencial).

    ``tile_scope_df``: restringe a la malla Gate (p. ej. ``cache_tiles`` de
    ``resolve_gate_am_splits``); evita mezclar densidades del manifest.
    """
    import numpy as np
    from ..common.io import read_table
    from ..common.paths import get_paths

    paths = get_paths()
    df = read_table(tiles_index_path or (paths.manifests / "tiles_index"))
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = df[df["image_path"] == rel].copy().reset_index(drop=True)
    if tile_scope_df is not None and not sub.empty:
        scope = tile_scope_df[tile_scope_df["image_path"].astype(str) == rel].copy()
        key_cols = ["row", "col"]
        if "tile_size" in scope.columns and scope["tile_size"].notna().any():
            key_cols.append("tile_size")
        sub = sub.merge(scope[key_cols].drop_duplicates(), on=key_cols, how="inner").reset_index(drop=True)
    if sub.empty:
        log.warning(f"No hay tiles para {rel} en el manifest")
        return sub

    gold = sub.copy()
    pred_cache = evaluate_gate_probe_on_df(
        bundle.classifier,
        sub,
        bundle.embed_store,
        bundle.device,
        batch_size=batch_size,
        prototype_bank=bundle.prototype_bank,
        slice_ms_only=bundle.slice_ms_only,
    )
    pred_cols = ["row", "col", "gate_pred_idx", "stage1_pred", "p_bg", "p_mminus", "p_mplus"]
    pred_key = ["row", "col"]
    if "tile_size" in sub.columns and sub["tile_size"].notna().any():
        pred_key = ["row", "col", "tile_size"]
        if "tile_size" not in pred_cache.columns:
            pred_key = ["row", "col"]
    merge_cols = pred_key + pred_cols[len(pred_key):]
    if pred_cache.empty:
        if strict:
            raise RuntimeError(
                f"[Gate probe] sin embeddings para {rel} ({len(sub)} tiles). "
                "Ejecuta: python run.py build-gate-cache antes de Stage2."
            )
        merged = gold.copy()
        for c in merge_cols[len(pred_key) :]:
            merged[c] = np.nan
    else:
        merged = gold.merge(pred_cache[merge_cols], on=pred_key, how="left")

    if merged["stage1_pred"].isna().any():
        from .gate_classes import decode_gate_indices

        n_miss = int(merged["stage1_pred"].isna().sum())
        if strict:
            raise RuntimeError(
                f"[Gate probe] cache incompleto para {rel}: "
                f"{len(pred_cache)}/{len(sub)} tiles con embed; faltan {n_miss}. "
                "Ejecuta: python run.py build-gate-cache "
                "(no usar gold stage1 como sustituto del Modelo 1)."
            )
        log.warning(
            f"[Gate probe] cache parcial ({len(pred_cache)}/{len(sub)} tiles); "
            f"fallback gold stage1 en {n_miss} tiles."
        )
        merged["stage1_pred"] = merged["stage1_pred"].astype("object")
        for idx, row in merged[merged["stage1_pred"].isna()].iterrows():
            s1 = str(row["stage1"]).strip()
            if s1 == "Mplus":
                gidx = 2
            elif s1 == "Mminus":
                gidx = 1
            elif s1 == "Background":
                gidx = 0
            else:
                gidx = 3
            merged.at[idx, "gate_pred_idx"] = gidx
            merged.at[idx, "stage1_pred"] = decode_gate_indices(np.array([gidx]))[0]
            merged.at[idx, "p_bg"] = 1.0 if gidx == 0 else 0.0
            merged.at[idx, "p_mminus"] = 1.0 if gidx == 1 else 0.0
            merged.at[idx, "p_mplus"] = 1.0 if gidx == 2 else 0.0

    merged["is_mplus"] = (merged["gate_pred_idx"] == 2).astype(int)
    merged["p_fused_max"] = merged[["p_bg", "p_mminus", "p_mplus"]].max(axis=1)
    merged["stage1"] = merged["stage1_pred"]
    return merged


def load_gate_tile_dino(
    *,
    backbone: str = "dinov2_vits14",
    num_classes: int = 3,
    device: torch.device | None = None,
    ckpt_dir: Path | None = None,
    weights_dir: Path | None = None,
) -> GateTileDinoGPU:
    from ..common.paths import get_paths

    paths = get_paths()
    ckpt_dir = ckpt_dir or (paths.root / "models" / "checkpoints" / "gate_am")
    device = device or torch.device("cuda")

    ckpt = ckpt_dir / CHECKPOINT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Falta checkpoint: {ckpt}. Entrena con: python run.py train-gate-am"
        )
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = st.get("model_state_dict", {})
    if _is_probe_state_dict(state):
        bundle = load_gate_probe_bundle(device=device, ckpt_dir=ckpt_dir)
        return GateTileDinoGPU(classifier=bundle.classifier, device=device).to(device)

    classifier = build_branch_a(backbone_name=backbone, num_classes=num_classes)
    classifier.load_state_dict(state)
    log.info(f"[Gate tile] checkpoint acc={st.get('acc', 0):.4f} macro_f1={st.get('macro_f1', 0):.4f}")
    return GateTileDinoGPU(classifier=classifier, device=device).to(device)
