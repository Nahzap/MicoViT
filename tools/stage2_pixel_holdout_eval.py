#!/usr/bin/env python3
"""E2-P2 — Eval holdout píxel: tablas val/test + métricas G-PX."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch

    import config as user_config  # type: ignore
    from micorizae.common.paths import get_paths
    from micorizae.phase_d_stage1.gate_tile_dino import infer_image_gate_probe_gpu, load_gate_probe_bundle
    from micorizae.phase_e_stage2.infer_pixel_gpu import infer_image_pixel_morph
    from micorizae.phase_e_stage2.pixel_data import load_mplus_splits
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams, quantize_segment
    from micorizae.phase_e_stage2.pixel_vit_model import build_pixel_morph_vit
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams

    gate_run_id = str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", getattr(user_config, "STAGE2_GATE_RUN_ID", "")))
    backend = str(getattr(user_config, "STAGE2_PIXEL_BACKEND", "vit"))
    vit_name = str(getattr(user_config, "STAGE2_PIXEL_VIT_MODEL", "dinov2_vits14"))
    input_size = int(getattr(user_config, "STAGE2_PIXEL_INPUT_SIZE", 224))
    paths = get_paths()
    rep_dir = ROOT / "outputs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    morph = PixelMorphParams(
        weak=WeakSegParams(
            vesicle_circularity_min=float(getattr(user_config, "STAGE2_PIXEL_VESICLE_CIRCULARITY_MIN", 0.85)),
            frangi_pctl=float(getattr(user_config, "STAGE2_PIXEL_FRANGI_PCTL", 82.0)),
            arbuscule_pctl=float(getattr(user_config, "STAGE2_PIXEL_ARBUSCULE_PCTL", 93.0)),
        ),
    )

    device = torch.device("cuda")
    pixel_model = None
    ckpt = paths.root / "models" / "checkpoints" / "stage2_am" / "stage2_pixel_vit_best.pt"
    if backend in ("vit", "ensemble") and ckpt.exists():
        pixel_model = build_pixel_morph_vit(backbone_name=vit_name, freeze_backbone=True)
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        pixel_model.load_state_dict(st["model_state_dict"])
        pixel_model.to(device).eval()
    elif backend != "weak":
        backend = "weak"

    _, val_df, test_df, split_info = load_mplus_splits()
    gate = load_gate_probe_bundle(device=device, gate_run_id=gate_run_id)

    def _eval_split(name: str, tiles_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for img_rel in sorted(tiles_df["image_path"].unique()):
            img_path = paths.root / img_rel
            if not img_path.exists():
                continue
            s1 = infer_image_gate_probe_gpu(img_path, gate, batch_size=8)
            if s1.empty:
                continue
            full_seg, tile_table, _ = infer_image_pixel_morph(
                img_path,
                s1,
                backend=backend,  # type: ignore
                model=pixel_model,
                device=device,
                morph_params=morph,
                input_size=input_size,
            )
            iq = quantize_segment(full_seg)
            rows.append({"split": name, "image_path": img_rel, "level": "image", **iq})
            if not tile_table.empty:
                tt = tile_table.copy()
                tt["split"] = name
                tt["level"] = "tile"
                rows.extend(tt.to_dict("records"))
        return pd.DataFrame(rows)

    val_out = _eval_split("val", val_df)
    test_out = _eval_split("test", test_df)
    val_out.to_parquet(rep_dir / f"stage2_pixel_val__gate{gate_run_id}.parquet", index=False)
    test_out.to_parquet(rep_dir / f"stage2_pixel_test__gate{gate_run_id}.parquet", index=False)

    tile_test = test_out[test_out["level"] == "tile"].copy()
    summary = {
        "generated": datetime.now().isoformat(),
        "gate_run_id": gate_run_id,
        "backend": backend,
        "split_info": split_info,
        "g_px1_colony_cov_gt0": float((tile_test["pct_colonized"] > 0).mean()) if len(tile_test) else 0.0,
        "mean_pct_colonized_test": float(tile_test["pct_colonized"].mean()) if len(tile_test) else 0.0,
        "n_test_tiles": int(len(tile_test)),
        "n_val_tiles": int((val_out["level"] == "tile").sum()),
    }
    if "stage2_gold" in tile_test.columns:
        hy = tile_test[tile_test["stage2_gold"] == "Hybrid"]["morph_entropy"]
        am = tile_test[tile_test["stage2_gold"] == "AMColonised"]["morph_entropy"]
        if len(hy) > 1 and len(am) > 1:
            from scipy.stats import mannwhitneyu

            _, p = mannwhitneyu(hy, am, alternative="greater")
            summary["g_px2_pvalue"] = float(p)
            summary["hybrid_entropy_mean"] = float(hy.mean())
            summary["amcolonised_entropy_mean"] = float(am.mean())

    explain_json = rep_dir / f"stage2_pixel_explain_audit__gate{gate_run_id}.json"
    if explain_json.exists():
        ex = json.loads(explain_json.read_text(encoding="utf-8"))
        summary["g_px_ex"] = ex

    out_json = rep_dir / f"stage2_pixel_holdout__gate{gate_run_id}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
