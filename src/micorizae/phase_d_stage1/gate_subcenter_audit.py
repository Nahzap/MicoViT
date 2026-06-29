"""Auditoria de subcentros para inferencia metrica Gate AM."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..gate_domain_buckets import merge_tile_metadata, summarize_domain_metrics
from .gate_classes import GATE_CLASS_NAMES
from .gate_tile_dino import _batch_iterator, _forward_classifier
from .gpu_pipeline import plan_epoch_gate


def _class_name(idx: int) -> str:
    if 0 <= int(idx) < len(GATE_CLASS_NAMES):
        return str(GATE_CLASS_NAMES[int(idx)])
    return f"class_{idx}"


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = float((F.normalize(a.float(), dim=0) @ F.normalize(b.float(), dim=0)).clamp(-1, 1).item())
    return float(math.degrees(math.acos(sim)))


def _prototype_geometry(prototype_bank: Any) -> dict:
    proto = F.normalize(prototype_bank.prototypes.detach().float().cpu(), dim=-1)
    init = prototype_bank.initialized.detach().cpu().bool()
    rows: list[dict] = []
    intra_angles: list[float] = []
    inter_angles: list[float] = []
    mminus_mplus_angles: list[float] = []

    k_fn = getattr(prototype_bank, "k_for_class", lambda c: proto.size(1))
    domains = getattr(prototype_bank, "subcenter_domains", None)

    for c1 in range(proto.size(0)):
        k1_lim = int(k_fn(c1))
        for k1 in range(k1_lim):
            if not bool(init[c1, k1]):
                continue
            for c2 in range(c1, proto.size(0)):
                k2_lim = int(k_fn(c2))
                k_start = k1 + 1 if c1 == c2 else 0
                for k2 in range(k_start, k2_lim):
                    if not bool(init[c2, k2]):
                        continue
                    deg = _angle_deg(proto[c1, k1], proto[c2, k2])
                    row = {
                        "class_a": _class_name(c1),
                        "subcenter_a": int(k1),
                        "domain_a": domains[c1][k1] if domains else None,
                        "class_b": _class_name(c2),
                        "subcenter_b": int(k2),
                        "domain_b": domains[c2][k2] if domains else None,
                        "angle_deg": deg,
                    }
                    rows.append(row)
                    if c1 == c2:
                        intra_angles.append(deg)
                    else:
                        inter_angles.append(deg)
                    if {c1, c2} == {1, 2}:
                        mminus_mplus_angles.append(deg)

    def _summary(vals: list[float]) -> dict:
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size == 0:
            return {"n": 0, "mean_deg": None, "min_deg": None, "max_deg": None}
        return {
            "n": int(arr.size),
            "mean_deg": float(arr.mean()),
            "min_deg": float(arr.min()),
            "max_deg": float(arr.max()),
        }

    k_per = getattr(prototype_bank, "num_subcenters_per_class", (prototype_bank.num_subcenters,))
    return {
        "num_subcenters": int(prototype_bank.num_subcenters),
        "num_subcenters_per_class": list(k_per),
        "domain_aware": bool(getattr(prototype_bank, "domain_aware", False)),
        "subcenter_domains": domains,
        "initialized": {
            _class_name(c): [int(k) for k in torch.where(init[c])[0].tolist()]
            for c in range(init.size(0))
        },
        "pairwise_angles": rows,
        "summary": {
            "intra_class": _summary(intra_angles),
            "inter_class": _summary(inter_angles),
            "mminus_mplus": _summary(mminus_mplus_angles),
        },
    }


@torch.no_grad()
def collect_subcenter_assignments(
    model: torch.nn.Module,
    tiles_df: pd.DataFrame,
    embed_store: Any,
    prototype_bank: Any,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Inferencia desde cache con subcentro ganador y margen por tile."""
    from ..gate_domain_buckets import domain_buckets_from_batch

    model.eval()
    plan = plan_epoch_gate(tiles_df, max_bg_per_image=None, shuffle_images=False, seed=0)
    rows: list[pd.DataFrame] = []
    required = [0, 1, 2]

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
            _, embed = _forward_classifier(model, batch, return_embed=True)
        if embed is None:
            raise RuntimeError("La auditoria de subcentros requiere embeddings del probe")

        domains = domain_buckets_from_batch(batch)
        logits, winner, best = prototype_bank.score_details(
            embed, required=required, domain_buckets=domains
        )
        pred = logits.argmax(dim=1)
        idx = torch.arange(pred.numel(), device=device)
        pred_sub = winner[idx, pred]
        gold_sub = winner[idx, y.clamp(min=0, max=winner.size(1) - 1)]
        own_sim = best[idx, y.clamp(min=0, max=best.size(1) - 1)]
        masked = torch.nan_to_num(best.clone(), nan=-float("inf"))
        masked[idx, y.clamp(min=0, max=best.size(1) - 1)] = -float("inf")
        rival_sim, rival_cls = masked.max(dim=1)
        margin = own_sim - rival_sim

        n = int(y.numel())
        part = pd.DataFrame(
            {
                "image_path": [batch.image_path] * n,
                "row": batch.rows.detach().cpu().numpy().astype(int),
                "col": batch.cols.detach().cpu().numpy().astype(int),
                "gold_idx": y.detach().cpu().numpy().astype(int),
                "gold_class": [_class_name(i) for i in y.detach().cpu().numpy().astype(int)],
                "pred_idx": pred.detach().cpu().numpy().astype(int),
                "pred_class": [_class_name(i) for i in pred.detach().cpu().numpy().astype(int)],
                "pred_subcenter": pred_sub.detach().cpu().numpy().astype(int),
                "gold_nearest_subcenter": gold_sub.detach().cpu().numpy().astype(int),
                "own_similarity": own_sim.detach().cpu().numpy().astype(float),
                "nearest_rival_idx": rival_cls.detach().cpu().numpy().astype(int),
                "nearest_rival_class": [
                    _class_name(i) for i in rival_cls.detach().cpu().numpy().astype(int)
                ],
                "nearest_rival_similarity": rival_sim.detach().cpu().numpy().astype(float),
                "subcenter_margin": margin.detach().cpu().numpy().astype(float),
                "correct": (pred == y).detach().cpu().numpy().astype(bool),
            }
        )
        if domains:
            part["domain_bucket"] = domains
        rows.append(part)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return merge_tile_metadata(out, tiles_df)


def write_subcenter_audit(
    *,
    run_dir: Path,
    model: torch.nn.Module,
    tiles_df: pd.DataFrame,
    embed_store: Any,
    prototype_bank: Any,
    device: torch.device,
    batch_size: int = 64,
    split_name: str = "holdout",
) -> dict:
    """Escribe CSV/JSON/MD de salud de subcentros para la corrida."""
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    assignments = collect_subcenter_assignments(
        model,
        tiles_df,
        embed_store,
        prototype_bank,
        device,
        batch_size=batch_size,
    )
    if assignments.empty:
        summary = {"split": split_name, "n_tiles": 0, "status": "empty"}
        (analysis_dir / "subcenter_angular.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    assignments.to_csv(analysis_dir / "subcenter_assignments.csv", index=False)

    occupancy = (
        assignments.groupby(["pred_class", "pred_subcenter"], dropna=False)
        .agg(
            n_tiles=("correct", "size"),
            n_correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            margin_mean=("subcenter_margin", "mean"),
            margin_p05=("subcenter_margin", lambda s: float(np.percentile(s, 5))),
        )
        .reset_index()
    )
    occupancy.to_csv(analysis_dir / "subcenter_occupancy.csv", index=False)

    confusion = (
        assignments.groupby(["gold_class", "pred_class", "pred_subcenter"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["gold_class", "pred_class", "pred_subcenter"])
    )
    confusion.to_csv(analysis_dir / "subcenter_confusion.csv", index=False)

    by_gold = (
        assignments.groupby(["gold_class", "gold_nearest_subcenter"], dropna=False)
        .agg(
            n_tiles=("correct", "size"),
            recall_local=("correct", "mean"),
            margin_mean=("subcenter_margin", "mean"),
        )
        .reset_index()
    )
    by_gold.to_csv(analysis_dir / "subcenter_gold_nearest.csv", index=False)

    domain_metrics = summarize_domain_metrics(assignments)
    if domain_metrics:
        pd.DataFrame(domain_metrics).to_csv(
            analysis_dir / "domain_metrics_summary.csv", index=False
        )
        (analysis_dir / "domain_metrics_summary.json").write_text(
            json.dumps(domain_metrics, indent=2), encoding="utf-8"
        )

    by_domain_sub = None
    if "domain_bucket" in assignments.columns:
        by_domain_sub = (
            assignments.groupby(["domain_bucket", "gold_class", "pred_subcenter"], dropna=False)
            .agg(n_tiles=("correct", "size"), recall=("correct", "mean"))
            .reset_index()
        )
        by_domain_sub.to_csv(analysis_dir / "subcenter_by_domain.csv", index=False)

    by_tile_edge = None
    if "tile_edge" in assignments.columns:
        by_tile_edge = (
            assignments.groupby(["tile_edge", "gold_class"], dropna=False)
            .agg(n_tiles=("correct", "size"), recall=("correct", "mean"))
            .reset_index()
        )
        by_tile_edge.to_csv(analysis_dir / "metrics_by_tile_edge.csv", index=False)

    by_subset = None
    if "subset" in assignments.columns:
        by_subset = (
            assignments.groupby(["subset", "gold_class"], dropna=False)
            .agg(n_tiles=("correct", "size"), recall=("correct", "mean"))
            .reset_index()
        )
        by_subset.to_csv(analysis_dir / "metrics_by_subset.csv", index=False)

    geometry = _prototype_geometry(prototype_bank)
    margin_by_class = (
        assignments.groupby("gold_class")
        .agg(
            n_tiles=("correct", "size"),
            recall=("correct", "mean"),
            margin_mean=("subcenter_margin", "mean"),
            margin_p05=("subcenter_margin", lambda s: float(np.percentile(s, 5))),
            frac_positive_margin=("subcenter_margin", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    k_per = getattr(prototype_bank, "num_subcenters_per_class", (prototype_bank.num_subcenters,))
    summary = {
        "split": split_name,
        "n_tiles": int(len(assignments)),
        "num_subcenters": int(prototype_bank.num_subcenters),
        "num_subcenters_per_class": list(k_per),
        "domain_aware": bool(getattr(prototype_bank, "domain_aware", False)),
        "geometry": geometry,
        "margin_by_class": margin_by_class.to_dict(orient="records"),
        "domain_metrics": domain_metrics,
        "artifacts": {
            "assignments": "analysis/subcenter_assignments.csv",
            "occupancy": "analysis/subcenter_occupancy.csv",
            "confusion": "analysis/subcenter_confusion.csv",
            "gold_nearest": "analysis/subcenter_gold_nearest.csv",
            "angular": "analysis/subcenter_angular.json",
            "report": "analysis/SUBCENTER_AUDIT.md",
            "domain_metrics": "analysis/domain_metrics_summary.json",
            "by_domain": "analysis/subcenter_by_domain.csv",
            "by_tile_edge": "analysis/metrics_by_tile_edge.csv",
            "by_subset": "analysis/metrics_by_subset.csv",
        },
    }
    (analysis_dir / "subcenter_angular.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# SUBCENTER_AUDIT",
        "",
        f"**Split:** `{split_name}`",
        f"**Tiles:** {len(assignments):,}",
        f"**Subcentros por clase:** {list(k_per)}",
        f"**Domain-aware:** {bool(getattr(prototype_bank, 'domain_aware', False))}",
        "",
        "## Geometria",
        "",
        f"- M-/M+ subcentros: {geometry['summary']['mminus_mplus']}",
        f"- Intra-clase: {geometry['summary']['intra_class']}",
        f"- Inter-clase: {geometry['summary']['inter_class']}",
        "",
        "## Margen Por Clase",
        "",
        "| Clase | n_tiles | recall | margen medio | p05 margen | frac margen positivo |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in margin_by_class.to_dict(orient="records"):
        lines.append(
            f"| {row['gold_class']} | {int(row['n_tiles'])} | {row['recall']:.3f} | "
            f"{row['margin_mean']:.4f} | {row['margin_p05']:.4f} | "
            f"{row['frac_positive_margin']:.3f} |"
        )

    if domain_metrics:
        lines.extend(["", "## Metricas Por Dominio", ""])
        lines.append("| domain | clase | n_tiles | recall | margen |")
        lines.append("|---|---|---:|---:|---:|")
        for row in domain_metrics:
            lines.append(
                f"| {row['domain_bucket']} | {row['class']} | {row['n_tiles']} | "
                f"{row['recall']:.3f} | {row['margin_mean']:.4f} |"
            )

    if by_tile_edge is not None and not by_tile_edge.empty:
        lines.extend(["", "## Recall Por tile_edge", ""])
        lines.append("| tile_edge | clase | n_tiles | recall |")
        lines.append("|---:|---|---:|---:|")
        for row in by_tile_edge.to_dict(orient="records"):
            lines.append(
                f"| {row['tile_edge']} | {row['gold_class']} | {int(row['n_tiles'])} | "
                f"{row['recall']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Artefactos",
            "",
            "- `analysis/subcenter_assignments.csv`",
            "- `analysis/subcenter_occupancy.csv`",
            "- `analysis/subcenter_confusion.csv`",
            "- `analysis/subcenter_gold_nearest.csv`",
            "- `analysis/domain_metrics_summary.json`",
            "- `analysis/subcenter_by_domain.csv`",
            "- `analysis/metrics_by_tile_edge.csv`",
            "- `analysis/subcenter_angular.json`",
        ]
    )
    (analysis_dir / "SUBCENTER_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (analysis_dir / "SUBCENTER_DOMAIN_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
