#!/usr/bin/env python3
"""P0 — Auditoría baseline de priors MEViT: concordancia prior↔weak por clase.

No modifica priors ni el pipeline. Mide, sobre los tiles M+ de val/test, el
desacople entre:

  - ``argmax`` del stack de evidencia continua (``PriorMaps.evidence``)
  - la pseudo-etiqueta weak (``PriorMaps.y_weak``) que realmente supervisa al ViT

Salidas (en ``outputs/stage2_pixel_prior_audit/gate<ID>/``):
  - ``prior_audit_per_tile.parquet``  — métricas por tile
  - ``prior_audit_summary.json``      — matriz de confusión + agreements por clase
  - ``panels/``                       — montajes RGB | weak | prior argmax | desacuerdo

Referencia del plan: Docs/fase 2 relevantes/20260709_223500_PLAN_PRIORS_MORFO_V3_IMPLEMENTACION.md (Fase P0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm

    import config as user_config  # type: ignore
    from micorizae.common.paths import get_paths
    from micorizae.phase_e_stage2.pixel_class_map import (
        NUM_PIXEL_CLASSES,
        PIXEL_CLASS_NAMES,
        PIXEL_CLASS_TO_IDX,
    )
    from micorizae.phase_e_stage2.pixel_data import load_mplus_splits
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams, render_pixel_class_map
    from micorizae.phase_e_stage2.pixel_prior_maps import PRIOR_IMPL_VERSION, compute_prior_maps
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams

    ap = argparse.ArgumentParser(description="Auditoría baseline priors MEViT (P0)")
    ap.add_argument("--max-tiles-per-split", type=int, default=1500,
                    help="cap de tiles por split (val/test); 0 = sin límite")
    ap.add_argument("--panels", type=int, default=24, help="nº de tiles a exportar como panel visual")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    rng = np.random.default_rng(args.seed)

    gate_run_id = str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", ""))
    out_dir = ROOT / "outputs" / "stage2_pixel_prior_audit" / f"gate{gate_run_id}"
    panels_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    morph = PixelMorphParams(
        weak=WeakSegParams(
            vesicle_circularity_min=float(getattr(user_config, "STAGE2_PIXEL_VESICLE_CIRCULARITY_MIN", 0.85)),
            frangi_pctl=float(getattr(user_config, "STAGE2_PIXEL_FRANGI_PCTL", 82.0)),
            arbuscule_pctl=float(getattr(user_config, "STAGE2_PIXEL_ARBUSCULE_PCTL", 93.0)),
        ),
        seam_sigma=0.0,
    )

    _, val_df, test_df, split_info = load_mplus_splits()
    paths = get_paths()

    bg_idx = PIXEL_CLASS_TO_IDX["BG"]
    ih_idx = PIXEL_CLASS_TO_IDX["IH"]
    h_idx = PIXEL_CLASS_TO_IDX["H"]

    # Confusión global (weak filas × prior argmax columnas), restringida a root.
    confusion = np.zeros((NUM_PIXEL_CLASSES, NUM_PIXEL_CLASSES), dtype=np.int64)
    per_tile_rows: list[dict] = []
    panel_candidates: list[dict] = []

    t0 = time.perf_counter()
    n_tiles_done = 0

    for split_name, sub in ("val", val_df), ("test", test_df):
        if sub is None or sub.empty:
            continue
        sub_use = sub
        if args.max_tiles_per_split and len(sub) > args.max_tiles_per_split:
            idx = rng.choice(len(sub), size=args.max_tiles_per_split, replace=False)
            sub_use = sub.iloc[np.sort(idx)].reset_index(drop=True)

        for img_rel, grp in tqdm(
            sub_use.groupby("image_path", sort=False),
            desc=f"prior_audit_{split_name}",
            total=sub_use["image_path"].nunique(),
        ):
            img_path = paths.root / str(img_rel)
            if not img_path.exists():
                continue
            img = np.asarray(Image.open(img_path).convert("RGB"))
            for rec in grp.itertuples(index=False):
                x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
                tile = img[y0:y1, x0:x1]
                if tile.size == 0 or tile.shape[0] < 8 or tile.shape[1] < 8:
                    continue

                pm = compute_prior_maps(tile, morph)
                evidence = pm.evidence  # (5,H,W)
                y_weak = pm.y_weak.astype(np.int64)
                root = pm.root > 0
                if evidence.shape[1:] != y_weak.shape:
                    continue
                prior_arg = evidence.argmax(axis=0).astype(np.int64)

                if not root.any():
                    continue
                wr = y_weak[root]
                pr = prior_arg[root]

                # Confusión por tile (para agregación y métricas locales).
                tile_conf = np.zeros((NUM_PIXEL_CLASSES, NUM_PIXEL_CLASSES), dtype=np.int64)
                np.add.at(tile_conf, (wr, pr), 1)
                confusion += tile_conf

                n_root = int(root.sum())
                agree = int((wr == pr).sum())
                ppa = agree / max(n_root, 1)

                # Speckle IH: entre píxeles weak==H, fracción que el prior dice IH.
                h_px = int(tile_conf[h_idx].sum())
                speckle_ih_in_h = float(tile_conf[h_idx, ih_idx] / h_px) if h_px > 0 else float("nan")

                row = {
                    "split": split_name,
                    "image_path": str(img_rel),
                    "row": int(getattr(rec, "row", -1)),
                    "col": int(getattr(rec, "col", -1)),
                    "stage2_gold": str(getattr(rec, "stage2", "")) if hasattr(rec, "stage2") else "",
                    "n_root_px": n_root,
                    "ppa_tile": float(ppa),
                    "disagreement_pct": float((wr != pr).mean() * 100.0),
                    "speckle_IH_in_H": speckle_ih_in_h,
                }
                for c_name in PIXEL_CLASS_NAMES:
                    ci = PIXEL_CLASS_TO_IDX[c_name]
                    w_c = int(tile_conf[ci].sum())
                    row[f"weak_px_{c_name}"] = w_c
                    row[f"agree_{c_name}"] = (
                        float(tile_conf[ci, ci] / w_c) if w_c > 0 else float("nan")
                    )
                per_tile_rows.append(row)

                panel_candidates.append(
                    {
                        "score": float((wr != pr).mean()),
                        "tile": tile.copy(),
                        "weak": y_weak.copy(),
                        "prior": prior_arg.copy(),
                        "root": root.copy(),
                        "tag": f"{split_name}_{Path(str(img_rel)).stem}_r{getattr(rec, 'row', 0)}_c{getattr(rec, 'col', 0)}",
                    }
                )
                n_tiles_done += 1

    wall = time.perf_counter() - t0
    tiles_per_sec = n_tiles_done / wall if wall > 0 else 0.0

    df = pd.DataFrame(per_tile_rows)
    parquet_path = out_dir / "prior_audit_per_tile.parquet"
    df.to_parquet(parquet_path, index=False)

    # --- Agregados globales desde la matriz de confusión ---
    total = int(confusion.sum())
    trace = int(np.trace(confusion))
    global_ppa = trace / max(total, 1)

    per_class_agreement: dict[str, float] = {}
    per_class_weak_px: dict[str, int] = {}
    for c_name in PIXEL_CLASS_NAMES:
        ci = PIXEL_CLASS_TO_IDX[c_name]
        w_c = int(confusion[ci].sum())
        per_class_weak_px[c_name] = w_c
        per_class_agreement[c_name] = float(confusion[ci, ci] / w_c) if w_c > 0 else None

    h_total = int(confusion[h_idx].sum())
    speckle_ih_in_h_global = float(confusion[h_idx, ih_idx] / h_total) if h_total > 0 else None

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate_run_id": gate_run_id,
        "prior_impl_version": int(PRIOR_IMPL_VERSION),
        "stain_aware": bool(getattr(morph.weak, "stain_aware", True)),
        "split_info": split_info,
        "n_tiles_audited": n_tiles_done,
        "wall_seconds": round(wall, 2),
        "tiles_per_sec_audit": round(tiles_per_sec, 3),
        "agreement_prior_weak_global": round(global_ppa, 4),
        "agreement_prior_weak_by_class": {
            k: (round(v, 4) if v is not None else None) for k, v in per_class_agreement.items()
        },
        "weak_px_by_class": per_class_weak_px,
        "speckle_IH_in_H_global": (round(speckle_ih_in_h_global, 4) if speckle_ih_in_h_global is not None else None),
        "confusion_weak_rows_prior_cols": confusion.tolist(),
        "class_order": list(PIXEL_CLASS_NAMES),
        "notes": "Confusión restringida a root>0. Filas = y_weak (pseudo-GT), columnas = argmax(prior_evidence).",
    }
    summary_path = out_dir / "prior_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Paneles visuales: mezcla de alto desacuerdo + muestra aleatoria ---
    def _panel_image(item: dict) -> np.ndarray:
        tile = item["tile"]
        weak_col = render_pixel_class_map(item["weak"])
        prior_col = render_pixel_class_map(item["prior"])
        disc = np.zeros((*item["weak"].shape, 3), dtype=np.uint8)
        d = (item["weak"] != item["prior"]) & item["root"]
        disc[d] = (255, 60, 60)
        h = tile.shape[0]

        def _fit(a: np.ndarray) -> np.ndarray:
            if a.shape[0] != h:
                a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_NEAREST)
            return a

        panel = np.concatenate([_fit(tile), _fit(weak_col), _fit(prior_col), _fit(disc)], axis=1)
        return panel

    if panel_candidates and args.panels > 0:
        n_panels = min(args.panels, len(panel_candidates))
        n_high = n_panels // 2
        by_score = sorted(panel_candidates, key=lambda x: x["score"], reverse=True)
        chosen = by_score[:n_high]
        remaining = by_score[n_high:]
        if remaining:
            pick = rng.choice(len(remaining), size=min(n_panels - n_high, len(remaining)), replace=False)
            chosen += [remaining[i] for i in pick]
        for item in chosen:
            panel = _panel_image(item)
            cv2.imwrite(str(panels_dir / f"{item['tag']}.png"), panel[:, :, ::-1])

    print("\n=== PRIOR AUDIT P0 — RESUMEN ===", flush=True)
    print(f"Tiles auditados       : {n_tiles_done}", flush=True)
    print(f"Wall / tiles-per-sec  : {wall:.1f}s / {tiles_per_sec:.2f} t/s", flush=True)
    print(f"PRIOR_IMPL_VERSION    : {PRIOR_IMPL_VERSION}", flush=True)
    print(f"Agreement global      : {global_ppa:.4f}", flush=True)
    for c_name in PIXEL_CLASS_NAMES:
        v = per_class_agreement[c_name]
        vs = f"{v:.4f}" if v is not None else "n/a"
        print(f"  agree[{c_name:>2}]         : {vs}  (weak_px={per_class_weak_px[c_name]})", flush=True)
    print(f"Speckle IH en H       : {speckle_ih_in_h_global}", flush=True)
    print(f"Parquet               : {parquet_path}", flush=True)
    print(f"Summary JSON          : {summary_path}", flush=True)
    print(f"Panels                : {panels_dir}", flush=True)


if __name__ == "__main__":
    main()
