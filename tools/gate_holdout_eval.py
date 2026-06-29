#!/usr/bin/env python3
"""Evaluacion holdout natural (29k tiles) para un checkpoint Gate AM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_run_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate4_shim(run_cfg: dict) -> SimpleNamespace:
    g4 = run_cfg.get("gate4") or {}
    return SimpleNamespace(
        enabled=bool(g4.get("enabled", True)),
        embed_dim=int(g4.get("embed_dim", 128)),
        num_slices=int(g4.get("num_slices", 4)),
        proto_subcenters_max=int(g4.get("proto_subcenters") or 1),
        proto_subcenters_per_class=tuple(g4.get("proto_subcenters_per_class") or ()),
        domain_aware_subcenters=bool(g4.get("domain_aware_subcenters", False)),
    )


def eval_holdout(
    *,
    ckpt_path: Path,
    run_cfg_path: Path,
    label: str,
    batch_size: int = 48,
) -> dict:
    import torch

    import config as user_config  # type: ignore

    from micorizae.gate_runflow import (
        _build_gate_classifier_for_recovery,
        _cache_basename_from_cfg,
        _open_gate_h5_store,
        gate_train_params_from_config,
        resolve_gate_am_splits,
    )
    from micorizae.phase_d_stage1.gate_embed_cache import inspect_embed_cache_status, open_gate_embed_store
    from micorizae.phase_d_stage1.gate_metric_inference import prototype_bank_from_gate4
    from micorizae.phase_d_stage1.gate_tile_dino import (
        _plan_val_epoch,
        evaluate_gate_tile_dino_gpu,
    )
    from micorizae.phase_d_stage1.gate_training_protocol import format_g1_status

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requerida para eval holdout.")

    ckpt_path = ckpt_path.resolve()
    run_cfg = _load_run_cfg(run_cfg_path)
    params = gate_train_params_from_config(user_config)
    device = torch.device("cuda")

    include_unknown = bool(params.gate4.include_unknown_in_split) if params.gate4 else False
    _train_df, val_df, _ext, _info, cache_tiles = resolve_gate_am_splits(
        user_config,
        exclude_unreadable=not include_unknown,
    )
    train_images = set(_train_df["image_path"].astype(str))

    freeze_backbone = bool(run_cfg.get("freeze_backbone", params.probe))
    embed_store = None
    h5_store = None
    cache_paths = None

    if freeze_backbone or bool(run_cfg.get("embed_cache", False)):
        status = inspect_embed_cache_status(
            cache_tiles,
            backbone_name=params.backbone,
            dino_input_size=params.dino_input_size,
            mplus_aug_variants=params.mplus_aug_variants,
            mplus_aug_train_images=train_images,
            cache_attention=params.cache_attention,
            attention_layers=params.attention_layers,
            attention_head_reduce=params.attention_head_reduce,
            cache_basename=_cache_basename_from_cfg(user_config),
        )
        if status.state not in {"valid", "stale", "obsolete"}:
            raise RuntimeError(f"Cache embeddings no usable ({status.state})")
        cache_paths = status.paths
        embed_store = open_gate_embed_store(status.paths)
    else:
        h5_store = _open_gate_h5_store(user_config)
        if h5_store is None:
            print("[warn] HDF5 no disponible; eval on-the-fly JPEG (lento).", flush=True)

    classifier = _build_gate_classifier_for_recovery(
        params=params,
        run_cfg=run_cfg,
        cfg=user_config,
        cache_paths=cache_paths,
        num_classes=4,
    )
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    classifier.load_state_dict(st["model_state_dict"])
    classifier.to(device).eval()

    proto = None
    g4_shim = _gate4_shim(run_cfg)
    slice_ms = str(run_cfg.get("loss_type", "slice_ms_only")) == "slice_ms_only"
    if slice_ms and "prototype_bank" in st and g4_shim.enabled:
        proto = prototype_bank_from_gate4(g4_shim, num_classes=4, device=device)
        proto.load_state_dict(st["prototype_bank"])

    holdout_plan = _plan_val_epoch(
        val_df,
        eval_balance_mode="natural",
        batch_size=batch_size,
        protocol=params.protocol,
        seed=0,
    )

    print(
        f"\n=== {label} ===\n"
        f"ckpt: {ckpt_path}\n"
        f"mode: {'PROBE+cache' if embed_store else 'FINETUNE+h5/jpeg'}\n"
        f"tiles: {holdout_plan.n_tiles:,}\n",
        flush=True,
    )

    metrics = evaluate_gate_tile_dino_gpu(
        classifier,
        holdout_plan,
        device,
        batch_size=batch_size,
        embed_store=embed_store,
        h5_store=h5_store,
        protocol=params.protocol,
        dino_input_size=params.dino_input_size,
        seg_target_size=params.seg_target_size,
        prototype_bank=proto,
        slice_ms_only=slice_ms,
        force_cpu_decode=bool(getattr(user_config, "GATE_TRAIN_FORCE_CPU_DECODE", True)),
        cpu_decode_above_mb=float(getattr(user_config, "GATE_TRAIN_CPU_DECODE_ABOVE_MB", 300.0)),
        desc=f"holdout {label}",
    )

    if embed_store is not None:
        embed_store.close()
    if h5_store is not None:
        h5_store.close()

    out = {
        "label": label,
        "ckpt": str(ckpt_path),
        "acc": metrics.get("acc"),
        "macro_f1": metrics.get("macro_f1"),
        "min_class_recall": metrics.get("min_class_recall"),
        "per_class_recall": metrics.get("per_class_recall"),
        "g1_status": format_g1_status(metrics, params.protocol),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval holdout natural Gate AM")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--label", type=str, default="checkpoint")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = eval_holdout(
        ckpt_path=args.ckpt,
        run_cfg_path=args.run_config,
        label=args.label,
        batch_size=args.batch_size,
    )

    lines = [
        f"label: {result['label']}",
        f"acc: {result['acc']:.4f}",
        f"macro_f1: {result['macro_f1']:.4f}",
        f"min_class_recall: {result['min_class_recall']:.4f}",
    ]
    for name, val in (result.get("per_class_recall") or {}).items():
        if val == val:
            lines.append(f"  recall_{name}: {val:.4f}")
    lines.append(f"g1: {result['g1_status']}")

    text = "\n".join(lines)
    print(text, flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
