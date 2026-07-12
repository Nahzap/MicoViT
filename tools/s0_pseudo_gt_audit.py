#!/usr/bin/env python3
"""S0 — Auditoría pseudo-GT v2 + stain stats (plan STAGE2-PLAN-20260710-v4).

Genera paneles RGB | weakseg | class counts y stain_stats.json para ABS710 + val.
"""

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
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm

    from micorizae.common.io import read_table
    from micorizae.common.paths import get_paths
    from micorizae.phase_e_stage2.atlas_log import density_norm_in_mask, stain_residual
    from micorizae.phase_e_stage2.pixel_class_map import PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX
    from micorizae.phase_e_stage2.pixel_morph import (
        PixelMorphParams,
        render_pixel_class_map,
        segment_tile_pixel_morph,
    )
    from micorizae.phase_e_stage2.pixel_prior_maps import PRIOR_IMPL_VERSION, compute_prior_maps
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams, _stain_maps, segment_tile

    paths = get_paths()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = paths.outputs / f"{stamp}_s0_pseudo_gt_audit"
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    # Prefer ABS710 images + a few other test images
    test_dir = paths.data / "am" / "am" / "test"
    abs710 = sorted(test_dir.glob("ABS710*.jpg"))
    others = [p for p in sorted(test_dir.glob("*.jpg")) if "ABS710" not in p.name][:3]
    images = abs710[:2] + others
    if not images:
        raise SystemExit("No hay imágenes test")

    # Load M+ tiles from tiles_index if available
    try:
        tiles = read_table(paths.manifests / "tiles_index")
    except FileNotFoundError:
        tiles = read_table(paths.manifests / "manifest_labels")

    weak = WeakSegParams()
    # Apply config overrides if present
    try:
        import config as user_config

        weak.ves_max_radius = int(getattr(user_config, "STAGE2_PIXEL_VESICLE_MAX_RADIUS", 0))
        weak.ves_max_sigma = float(getattr(user_config, "STAGE2_PIXEL_VESICLE_MAX_SIGMA", 40.0))
    except Exception:
        pass
    morph = PixelMorphParams(weak=weak, seam_sigma=0.0)

    Image.MAX_IMAGE_PIXELS = None
    stain_rows = []
    class_rows = []
    agreement_rows = []

    for img_path in tqdm(images, desc="S0 audit"):
        rel = img_path.resolve().relative_to(paths.root).as_posix()
        sub = tiles[tiles["image_path"] == rel].copy() if "image_path" in tiles.columns else tiles.iloc[0:0]
        if "stage1" in sub.columns:
            sub = sub[sub["stage1"].astype(str) == "Mplus"]
        if sub.empty:
            print(f"  skip (no M+): {img_path.name}")
            continue

        rgb_full = np.asarray(Image.open(img_path).convert("RGB"))
        # Sample up to 12 M+ tiles for panels
        sample = sub.sample(n=min(12, len(sub)), random_state=0) if len(sub) > 12 else sub
        for _, row in sample.iterrows():
            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            tile = rgb_full[y0:y1, x0:x1].copy()
            if tile.size == 0:
                continue
            masks = segment_tile(tile, weak)
            seg = segment_tile_pixel_morph(tile, morph, masks=masks, ambiguous_to_h_dense=True)
            priors = compute_prior_maps(tile, morph)
            prior_argmax = priors.evidence.argmax(axis=0).astype(np.uint8)

            dens = masks.get("density")
            if dens is not None:
                root = masks["root"] > 0
                dn = density_norm_in_mask(dens, root)
                res = stain_residual(dens)
                stain_rows.append(
                    {
                        "image": img_path.name,
                        "row": int(row["row"]),
                        "col": int(row["col"]),
                        "density_mean": float(dn[root].mean()) if root.any() else 0.0,
                        "density_p95": float(np.percentile(dn[root], 95)) if root.any() else 0.0,
                        "residual_mean": float(res[root].mean()) if root.any() else 0.0,
                        "ambiguous_frac": float((masks.get("ambiguous", np.zeros_like(root)) > 0).mean()),
                        "pct_V": float((seg == PIXEL_CLASS_TO_IDX["V"]).mean() * 100),
                        "pct_A": float((seg == PIXEL_CLASS_TO_IDX["A"]).mean() * 100),
                        "pct_IH": float((seg == PIXEL_CLASS_TO_IDX["IH"]).mean() * 100),
                        "pct_H": float((seg == PIXEL_CLASS_TO_IDX["H"]).mean() * 100),
                    }
                )

            counts = {c: int((seg == PIXEL_CLASS_TO_IDX[c]).sum()) for c in PIXEL_CLASS_NAMES}
            class_rows.append({"image": img_path.name, "row": int(row["row"]), "col": int(row["col"]), **counts})

            agree = float((prior_argmax == seg).mean())
            agreement_rows.append(
                {
                    "image": img_path.name,
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                    "prior_weak_agree": agree,
                    "prior_impl": PRIOR_IMPL_VERSION,
                }
            )

            # Panel
            weak_rgb = render_pixel_class_map(seg)
            prior_rgb = render_pixel_class_map(prior_argmax)
            disagree = (prior_argmax != seg).astype(np.uint8) * 255
            disagree_rgb = np.stack([disagree, np.zeros_like(disagree), np.zeros_like(disagree)], axis=-1)
            panel = np.concatenate([tile, weak_rgb, prior_rgb, disagree_rgb], axis=1)
            fname = f"{img_path.stem}_r{int(row['row'])}_c{int(row['col'])}.png"
            Image.fromarray(panel).save(panels / fname)

    summary = {
        "timestamp": stamp,
        "prior_impl_version": PRIOR_IMPL_VERSION,
        "n_images": len(images),
        "n_tiles_audited": len(class_rows),
        "ves_max_radius": weak.ves_max_radius,
        "ves_max_sigma": weak.ves_max_sigma,
        "mean_prior_weak_agree": float(np.mean([r["prior_weak_agree"] for r in agreement_rows])) if agreement_rows else 0.0,
        "mean_pct_V": float(np.mean([r["pct_V"] for r in stain_rows])) if stain_rows else 0.0,
        "mean_pct_A": float(np.mean([r["pct_A"] for r in stain_rows])) if stain_rows else 0.0,
        "images": [p.name for p in images],
    }
    (out / "stain_stats.json").write_text(json.dumps({"summary": summary, "tiles": stain_rows}, indent=2), encoding="utf-8")
    (out / "class_counts.json").write_text(json.dumps(class_rows, indent=2), encoding="utf-8")
    (out / "prior_agreement.json").write_text(json.dumps({"summary": summary, "tiles": agreement_rows}, indent=2), encoding="utf-8")
    (out / "ACTA_S0.md").write_text(
        f"# Acta S0 — {stamp}\n\n"
        f"- Prior impl: v{PRIOR_IMPL_VERSION}\n"
        f"- Tiles auditados: {summary['n_tiles_audited']}\n"
        f"- Agree prior↔weak: {summary['mean_prior_weak_agree']:.3f}\n"
        f"- mean %V: {summary['mean_pct_V']:.2f} | mean %A: {summary['mean_pct_A']:.2f}\n"
        f"- ves_max_radius={weak.ves_max_radius} (0=adaptativo) ves_max_sigma={weak.ves_max_sigma}\n"
        f"- Paneles: `{panels}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
