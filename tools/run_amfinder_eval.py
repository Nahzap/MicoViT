"""Evalua el gate AM entrenado sobre AMFinder CNN1 (formato Micorizae).

Genera carpeta de resultados con mapas grid (L0+L1, gold/pred, errores),
metricas globales y por imagen / tile_edge.

Uso:
  python tools/run_amfinder_eval.py
  python tools/run_amfinder_eval.py --tile-edge 256 --force-cache
  python tools/run_amfinder_eval.py --limit 3 --skip-maps
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config as project_config

from micorizae.common.io import read_table, write_table
from micorizae.common.logging_utils import get_logger
from micorizae.common.paths import get_paths
from micorizae.phase_d_stage1 import build_branch_a
from micorizae.phase_d_stage1.gate4.config import gate4_config_from_module
from micorizae.phase_d_stage1.gate_checkpoint_snapshot import _load_classifier_from_checkpoint
from micorizae.phase_d_stage1.gate_classes import GATE_CLASS_NAMES, encode_gate_indices
from micorizae.phase_d_stage1.gate_embed_cache import (
    GateEmbedStore,
    build_gate_embed_cache,
    cache_paths_with_basename,
)
from micorizae.phase_d_stage1.gate_tile_dino import (
    collect_probe_logits,
    evaluate_gate_probe_on_df,
)
from micorizae.phase_d_stage1.gate_train_report import (
    plot_class_recall_bars,
    plot_confusion_matrix,
    render_gate_image_maps,
)
from micorizae.phase_d_stage1.gate_training_protocol import (
    GateTrainProtocol,
    compute_gate_metrics,
    macro_f1_score,
)

log = get_logger("tools.amfinder_eval")

AMFINDER_CACHE_BASENAME = "gate_amfinder_eval_v1"
AMFINDER_TILES_INDEX = ROOT / "Data" / "am" / "am" / "amfinder_eval" / "amfinder_tiles_index"
STAGE1_FROM_CSV = {
    "AMColonised": "Mplus",
    "Uncolonised": "Mminus",
    "Background": "Background",
    "Unreadable": "Unreadable",
}


def _row_stage1(row: pd.Series) -> str:
    for col, stage1 in STAGE1_FROM_CSV.items():
        if int(row.get(col, 0) or 0) == 1:
            return stage1
    return "Unreadable"


def load_amfinder_tiles_df(
    catalog_path: Path,
    *,
    tile_edge: Optional[int] = None,
    limit: int = 0,
    exclude_unreadable: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Construye tiles_df desde catalogo + CSVs convertidos."""
    meta = json.loads(catalog_path.read_text(encoding="utf-8"))
    pairs = list(meta.get("pairs") or [])
    if tile_edge is not None:
        pairs = [p for p in pairs if int(p.get("tile_edge", 0)) == tile_edge]
    if limit > 0:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No hay pares AMFinder tras filtros.")

    rows: list[dict] = []
    for entry in pairs:
        ann = ROOT / str(entry["annotation_path"])
        tile_size = int(entry["tile_edge"])
        image_path = str(entry["image_path"]).replace("\\", "/")
        df = read_table(ann)
        for _, r in df.iterrows():
            stage1 = _row_stage1(r)
            if exclude_unreadable and stage1 == "Unreadable":
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "row": int(r["row"]),
                    "col": int(r["col"]),
                    "tile_size": tile_size,
                    "stage1": stage1,
                    "lineage": "AM",
                    "subset": "amfinder_eval",
                    "split": "test",
                    "tile_edge": tile_size,
                    "stem": entry["stem"],
                }
            )
    tiles_df = pd.DataFrame(rows)
    if tiles_df.empty:
        raise ValueError("tiles_df vacio tras cargar anotaciones.")
    return tiles_df, pairs


def _ensure_embed_cache(
    tiles_df: pd.DataFrame,
    tiles_index_path: Path,
    *,
    force_rebuild: bool = False,
) -> GateEmbedStore:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible para compilar cache AMFinder.")

    paths = get_paths()
    cpaths = cache_paths_with_basename(paths.root, AMFINDER_CACHE_BASENAME)
    if not force_rebuild and cpaths.meta.exists():
        try:
            store = GateEmbedStore(cpaths)
            store.indices_for_sub(tiles_df)
            log.info(f"[AMFinder] Cache HIT -> {cpaths.embed.name}")
            return store
        except KeyError:
            log.info("[AMFinder] Cache incompleta para este subset; recompilando...")

    device = torch.device("cuda")
    backbone = build_branch_a(
        backbone_name=project_config.GATE_BACKBONE,
        num_classes=3,
        freeze_backbone=True,
    ).backbone
    build_gate_embed_cache(
        tiles_df,
        backbone,
        device=device,
        backbone_name=project_config.GATE_BACKBONE,
        dino_input_size=project_config.GATE_DINO_INPUT_SIZE,
        seg_target_size=project_config.GATE_SEG_TARGET_SIZE,
        batch_size=project_config.GATE_CACHE_BATCH_SIZE,
        tiles_index_path=tiles_index_path,
        cache_paths=cpaths,
        force_rebuild=force_rebuild,
        dynamic_batch=project_config.GATE_CACHE_DYNAMIC_BATCH,
        vram_budget_mb=project_config.GATE_CACHE_VRAM_BUDGET_MB,
        cpu_decode_above_mb=project_config.GATE_CACHE_CPU_DECODE_ABOVE_MB,
        empty_cache_every_n_batches=project_config.GATE_CACHE_EMPTY_CACHE_EVERY_N_BATCHES,
        gc_collect_every_n_batches=project_config.GATE_CACHE_GC_COLLECT_EVERY_N_BATCHES,
        memmap_flush_every_n_images=project_config.GATE_CACHE_MEMMAP_FLUSH_EVERY_N_IMAGES,
        mplus_aug_variants=(),
        cache_attention=project_config.GATE_CACHE_ATTENTION,
        attention_layers=project_config.GATE_CACHE_ATTENTION_LAYERS,
        attention_head_reduce=project_config.GATE_CACHE_ATTENTION_HEAD_REDUCE,
    )
    return GateEmbedStore(cpaths)


def _metrics_by_group(pred: pd.DataFrame, group_col: str) -> list[dict]:
    rows: list[dict] = []
    for key, sub in pred.groupby(group_col, sort=False):
        if sub.empty:
            continue
        y_true = encode_gate_indices(sub["stage1_gold"].to_numpy())
        y_pred = sub["gate_pred_idx"].to_numpy()
        rows.append(
            {
                group_col: key,
                "n_tiles": len(sub),
                "acc": float(sub["correct"].mean()),
                "macro_f1": float(macro_f1_score(y_true, y_pred)),
            }
        )
    return rows


def run_eval(
    *,
    catalog_path: Path,
    checkpoint: Path,
    out_root: Optional[Path] = None,
    tile_edge: Optional[int] = None,
    limit: int = 0,
    force_cache: bool = False,
    skip_maps: bool = False,
    downscale: int = 4,
    batch_size: int = 48,
) -> Path:
    import torch
    from sklearn.metrics import classification_report
    from tqdm.auto import tqdm

    paths = get_paths()
    tiles_df, pairs = load_amfinder_tiles_df(
        catalog_path, tile_edge=tile_edge, limit=limit, exclude_unreadable=True
    )
    log.info(f"[AMFinder] {len(pairs)} imagenes, {len(tiles_df):,} tiles")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root or (paths.outputs / f"amfinder_eval_{ts}")
    maps_dir = run_dir / "maps"
    tables_dir = run_dir / "tables"
    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    config_dir = run_dir / "config"
    for d in (maps_dir, tables_dir, metrics_dir, reports_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)

    tiles_index_path = write_table(tiles_df, AMFINDER_TILES_INDEX)

    embed_store = _ensure_embed_cache(
        tiles_df, tiles_index_path, force_rebuild=force_cache
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate4 = gate4_config_from_module(project_config)
    protocol = GateTrainProtocol(loss_type=project_config.GATE_LOSS)
    model, st, proto = _load_classifier_from_checkpoint(
        checkpoint,
        embed_store=embed_store,
        device=device,
        gate4_config=gate4,
    )
    calibration = st.get("calibration")
    slice_ms = protocol.loss_type == "slice_ms_only"

    log.info("[AMFinder] Inferencia desde cache...")
    pred = evaluate_gate_probe_on_df(
        model,
        tiles_df,
        embed_store,
        device,
        batch_size=batch_size,
        calibration=calibration,
        prototype_bank=proto,
        slice_ms_only=slice_ms,
    )
    if pred.empty:
        raise RuntimeError("Inferencia vacia; revisa cache y checkpoint.")

    write_table(pred, tables_dir / "predictions_all_tiles")

    logits, labels_arr, _, _, _ = collect_probe_logits(
        model,
        tiles_df,
        embed_store,
        device,
        batch_size=batch_size,
        prototype_bank=proto,
        slice_ms_only=slice_ms,
    )
    if calibration:
        from micorizae.phase_d_stage1.gate_calibrate import GateCalibration, apply_gate_calibration

        logits = apply_gate_calibration(logits, GateCalibration.from_dict(calibration))

    y_true = labels_arr
    y_pred = pred["gate_pred_idx"].to_numpy()
    g1 = compute_gate_metrics(logits, y_true, protocol=protocol)

    metrics_summary: dict[str, Any] = {
        "dataset": "AMFinder CNN1 eval",
        "catalog": str(catalog_path.relative_to(ROOT)),
        "checkpoint": str(checkpoint),
        "n_images": len(pairs),
        "n_tiles": len(pred),
        "tile_edge_filter": tile_edge,
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

    plot_confusion_matrix(y_true, y_pred, metrics_dir / "confusion_matrix.png")
    if g1.get("per_class_recall"):
        plot_class_recall_bars(g1["per_class_recall"], metrics_dir / "class_recall_bars.png")

    per_image = []
    pred_with_edge = pred.merge(
        tiles_df[["image_path", "row", "col", "tile_edge"]].drop_duplicates(),
        on=["image_path", "row", "col"],
        how="left",
    )
    for rel in pred["image_path"].unique():
        sub = pred_with_edge[pred_with_edge["image_path"] == rel]
        yt = encode_gate_indices(sub["stage1_gold"].to_numpy())
        yp = sub["gate_pred_idx"].to_numpy()
        edge = int(sub["tile_edge"].iloc[0]) if "tile_edge" in sub.columns and sub["tile_edge"].notna().any() else None
        per_image.append(
            {
                "image_path": rel,
                "tile_edge": edge,
                "n_tiles": len(sub),
                "acc": float(sub["correct"].mean()),
                "macro_f1": float(macro_f1_score(yt, yp)),
                "n_mplus_pred": int((sub["stage1_pred"] == "Mplus").sum()),
                "n_mminus_pred": int((sub["stage1_pred"] == "Mminus").sum()),
                "n_bg_pred": int((sub["stage1_pred"] == "Background").sum()),
            }
        )
    write_table(pd.DataFrame(per_image), tables_dir / "per_image_summary")

    by_edge = _metrics_by_group(pred_with_edge, "tile_edge")
    write_table(pd.DataFrame(by_edge), tables_dir / "by_tile_edge_summary")
    metrics_summary["by_tile_edge"] = by_edge
    metrics_summary["per_image"] = per_image

    (metrics_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics_summary, indent=2, default=str), encoding="utf-8"
    )

    map_manifest: list[dict] = []
    if not skip_maps:
        log.info(f"[AMFinder] Generando mapas grid ({len(pairs)} imagenes)...")
        for entry in tqdm(pairs, desc="mapas amfinder", unit="img"):
            rel = str(entry["image_path"]).replace("\\", "/")
            img_path = paths.root / rel
            gold_sub = tiles_df[tiles_df["image_path"] == rel]
            pred_sub = pred[pred["image_path"] == rel]
            if gold_sub.empty or pred_sub.empty:
                continue
            ts_img = int(entry["tile_edge"])
            stem = entry["stem"]
            write_table(pred_sub, tables_dir / f"images/{stem}__gate_probs")
            try:
                rendered = render_gate_image_maps(
                    image_path=img_path,
                    tiles_gold=gold_sub,
                    tiles_pred=pred_sub,
                    maps_dir=maps_dir,
                    device=device,
                    tile_size=ts_img,
                    downscale=downscale,
                    image_stem=stem,
                    decode_cpu=True,
                )
                map_manifest.append(
                    {"image": rel, "tile_edge": ts_img, "maps": {k: v.name for k, v in rendered.items()}}
                )
            except Exception as e:
                log.warning(f"[AMFinder] Mapa omitido {stem}: {e}")

    (config_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "dino_input_size": project_config.GATE_DINO_INPUT_SIZE,
                "backbone": project_config.GATE_BACKBONE,
                "cache_basename": AMFINDER_CACHE_BASENAME,
                "tile_edge_filter": tile_edge,
                "limit": limit,
                "n_images": len(pairs),
                "n_tiles": len(pred),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# AMFinder eval — Gate AM",
        "",
        f"- **Imagenes:** {len(pairs)}",
        f"- **Tiles evaluados:** {len(pred):,}",
        f"- **Checkpoint:** `{checkpoint}`",
        f"- **Acc:** {metrics_summary['acc']:.4f}",
        f"- **Macro F1:** {metrics_summary['macro_f1']:.4f}",
        f"- **Min class recall:** {metrics_summary['min_class_recall']:.4f}",
        "",
        "## Metricas por tile_edge",
        "",
    ]
    for row in by_edge:
        md_lines.append(
            f"- **{row['tile_edge']} px:** n={row['n_tiles']:,}, acc={row['acc']:.3f}, macro_f1={row['macro_f1']:.3f}"
        )
    md_lines.extend(["", "## Mapas generados", ""])
    for item in map_manifest:
        md_lines.append(f"### `{item['image']}` (edge={item['tile_edge']})")
        for k, v in item["maps"].items():
            md_lines.append(f"- `{k}`: maps/{v}")
    md_lines.append("")
    (reports_dir / "AMFINDER_EVAL_REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")

    log.info(f"[AMFinder] Resultados -> {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval gate AM sobre AMFinder CNN1 + mapas grid")
    ap.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "Data" / "am" / "am" / "amfinder_eval" / "amfinder_eval_catalog.json",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "gate_am" / "gate_tile_dino_best.pt",
    )
    ap.add_argument("--out", type=Path, default=None, help="Carpeta salida (default: outputs/amfinder_eval_<ts>)")
    ap.add_argument("--tile-edge", type=int, default=None, help="Filtrar imagenes por tile_edge (40|126|256)")
    ap.add_argument("--limit", type=int, default=0, help="Limitar numero de imagenes (0=todas)")
    ap.add_argument("--force-cache", action="store_true", help="Recompilar cache DINO AMFinder")
    ap.add_argument("--skip-maps", action="store_true", help="Omitir PNGs grid")
    ap.add_argument("--downscale", type=int, default=project_config.GATE_VIS_DOWNSCALE)
    ap.add_argument("--batch-size", type=int, default=project_config.GATE_TRAIN_BATCH_SIZE)
    args = ap.parse_args()

    if not args.catalog.exists():
        raise SystemExit(f"Catalogo no encontrado: {args.catalog}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint no encontrado: {args.checkpoint}")

    run_eval(
        catalog_path=args.catalog.resolve(),
        checkpoint=args.checkpoint.resolve(),
        out_root=args.out.resolve() if args.out else None,
        tile_edge=args.tile_edge,
        limit=args.limit,
        force_cache=args.force_cache,
        skip_maps=args.skip_maps,
        downscale=args.downscale,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
