#!/usr/bin/env python3
"""E2-P0 — Auditoría baseline morfología píxel (weak) en val/test."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import numpy as np
    import pandas as pd

    import config as user_config  # type: ignore
    from micorizae.common.paths import get_paths
    from micorizae.phase_e_stage2.pixel_data import load_mplus_splits
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams, quantize_segment, segment_tile_pixel_morph
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams

    gate_run_id = str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", ""))
    out_dir = ROOT / "outputs" / "stage2_pixel_audit" / f"gate{gate_run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    rows = []

    from PIL import Image
    from tqdm.auto import tqdm

    Image.MAX_IMAGE_PIXELS = None

    for split_name, sub in ("val", val_df), ("test", test_df):
        for img_rel, grp in tqdm(
            sub.groupby("image_path", sort=False),
            desc=f"baseline_{split_name}",
            total=sub["image_path"].nunique(),
        ):
            img_path = paths.root / str(img_rel)
            if not img_path.exists():
                continue
            img = np.asarray(Image.open(img_path).convert("RGB"))
            for rec in grp.itertuples(index=False):
                x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
                tile = img[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                seg = segment_tile_pixel_morph(tile, morph)
                q = quantize_segment(seg)
                rows.append(
                    {
                        "split": split_name,
                        "image_path": str(img_rel),
                        "row": int(rec.row),
                        "col": int(rec.col),
                        "stage2_gold": str(getattr(rec, "stage2", "")),
                        **q,
                    }
                )

    df = pd.DataFrame(rows)
    parquet_path = out_dir / "stage2_pixel_baseline_all.parquet"
    df.to_parquet(parquet_path, index=False)

    summary = {
        "generated": datetime.now().isoformat(),
        "gate_run_id": gate_run_id,
        "split_info": split_info,
        "n_tiles": int(len(df)),
        "g_px1_colony_cov_gt0": float((df["pct_colonized"] > 0).mean()) if len(df) else 0.0,
        "mean_pct_colonized": float(df["pct_colonized"].mean()) if len(df) else 0.0,
    }
    if "stage2_gold" in df.columns:
        hy = df[df["stage2_gold"] == "Hybrid"]["morph_entropy"]
        am = df[df["stage2_gold"] == "AMColonised"]["morph_entropy"]
        if len(hy) > 2 and len(am) > 2:
            from scipy.stats import mannwhitneyu

            _, p = mannwhitneyu(hy, am, alternative="greater")
            summary["g_px2_hybrid_entropy_gt_am_pvalue"] = float(p)
            summary["g_px2_hybrid_mean_entropy"] = float(hy.mean())
            summary["g_px2_am_mean_entropy"] = float(am.mean())

    (out_dir / "STAGE2_PIXEL_BASELINE_AUDIT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        f"# Stage2 pixel baseline audit — gate `{gate_run_id}`",
        "",
        f"- Generado: {summary['generated']}",
        f"- Tiles auditados: {summary['n_tiles']}",
        f"- G-PX.1 (% tiles con colonia>0): **{summary['g_px1_colony_cov_gt0']:.3f}**",
        f"- Media % colonizado: **{summary['mean_pct_colonized']:.2f}**",
    ]
    if "g_px2_hybrid_entropy_gt_am_pvalue" in summary:
        md.append(
            f"- G-PX.2 Hybrid entropy > AMColonised (Mann-Whitney p): **{summary['g_px2_hybrid_entropy_gt_am_pvalue']:.4f}**"
        )
    (out_dir / "STAGE2_PIXEL_BASELINE_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
