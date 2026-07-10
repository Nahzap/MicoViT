#!/usr/bin/env python3
"""E2-EX — Auditoría métricas explicabilidad MEViT (G-PX-EX)."""

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
    import torch

    import config as user_config  # type: ignore
    from micorizae.common.paths import get_paths
    from micorizae.phase_e_stage2.infer_pixel_gpu import (
        PixelMorphInferResult,
        build_tile_explain_bundles,
        infer_image_pixel_morph,
    )
    from micorizae.phase_e_stage2.pixel_data import load_mplus_splits
    from micorizae.phase_e_stage2.stage2_gate_infer import (
        assert_gate_embed_cache_ready,
        infer_image_gate_stage2,
        load_stage2_gate_bundle,
    )
    from micorizae.phase_e_stage2.pixel_explain_metrics import (
        expected_calibration_error,
        prior_conditional_accuracy,
    )
    from micorizae.phase_e_stage2.pixel_prior_maps import prior_argmax_agreement
    from micorizae.phase_e_stage2.pixel_explainability import explain_quant_table, image_explain_summary
    from micorizae.phase_e_stage2.pixel_class_map import PIXEL_CLASS_TO_IDX
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams
    from micorizae.phase_e_stage2.pixel_vit_model import build_pixel_morph_vit
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams

    gate_run_id = str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", ""))
    gate_strict = bool(getattr(user_config, "STAGE2_GATE_INFERENCE_STRICT", True))
    require_gate_cache = bool(getattr(user_config, "STAGE2_POSTTRAIN_REQUIRE_GATE_CACHE", True))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_model = None
    ckpt = paths.root / "models" / "checkpoints" / "stage2_am" / "stage2_pixel_vit_best.pt"
    if backend in ("vit", "ensemble") and ckpt.exists():
        pixel_model = build_pixel_morph_vit(backbone_name=vit_name, freeze_backbone=True)
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        pixel_model.load_state_dict(st["model_state_dict"])
        pixel_model.to(device).eval()
    elif backend != "weak":
        backend = "weak"

    _, _, test_df, split_info = load_mplus_splits()
    if require_gate_cache:
        assert_gate_embed_cache_ready(
            test_df[["image_path"]].drop_duplicates(),
            cfg=user_config,
        )
    gate = load_stage2_gate_bundle(device=device, gate_run_id=gate_run_id, cfg=user_config)

    ppa_vals: list[float] = []
    ece_vals: list[float] = []
    pca_v_vals: list[float] = []
    rows: list[dict] = []

    for img_rel in sorted(test_df["image_path"].unique())[:12]:
        img_path = paths.root / img_rel
        if not img_path.exists():
            continue
        s1 = infer_image_gate_stage2(img_path, gate, batch_size=64, strict=gate_strict)
        if s1.empty:
            continue
        out = infer_image_pixel_morph(
            img_path,
            s1,
            backend=backend,  # type: ignore
            model=pixel_model,
            device=device,
            morph_params=morph,
            input_size=input_size,
            with_explain=True,
        )
        if not isinstance(out, PixelMorphInferResult):
            continue
        bundles = build_tile_explain_bundles(out)
        explain_df = explain_quant_table(s1, bundles, out.tile_table)
        if not explain_df.empty:
            rows.extend(explain_df.to_dict("records"))
            ppa_vals.extend(explain_df["ppa_tile"].tolist())
            ece_vals.extend(explain_df["ece_tile"].tolist())
        for key, bundle in bundles.items():
            root = bundle.prior.root
            pca_v = prior_conditional_accuracy(
                bundle.seg_vit,
                bundle.prior.circularity,
                PIXEL_CLASS_TO_IDX["V"],
                tau=0.5,
                root=root,
            )
            if not np.isnan(pca_v):
                pca_v_vals.append(pca_v)
            ppa_vals.append(
                prior_argmax_agreement(bundle.seg_vit, bundle.seg_weak, root)
            )

    summary = {
        "generated": datetime.now().isoformat(),
        "gate_run_id": gate_run_id,
        "backend": backend,
        "split_info": split_info,
        "g_px_ex_1_ppa_mean": float(np.mean(ppa_vals)) if ppa_vals else 0.0,
        "g_px_ex_3_ece_mean": float(np.mean(ece_vals)) if ece_vals else 0.0,
        "g_px_ex_4_pca_v_mean": float(np.mean(pca_v_vals)) if pca_v_vals else float("nan"),
        "n_tiles_explain": len(rows),
    }
    out_json = rep_dir / f"stage2_pixel_explain_audit__gate{gate_run_id}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        pd.DataFrame(rows).to_parquet(
            rep_dir / f"stage2_pixel_explain_tiles__gate{gate_run_id}.parquet",
            index=False,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
