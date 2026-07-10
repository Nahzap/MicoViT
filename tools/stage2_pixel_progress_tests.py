#!/usr/bin/env python3
"""Tests MEViT E2-EX — priors, losses, explicabilidad."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_imports() -> None:
    from micorizae.phase_e_stage2.pixel_prior_maps import compute_prior_maps  # noqa: F401
    from micorizae.phase_e_stage2.pixel_prior_loss import combine_prior_losses  # noqa: F401
    from micorizae.phase_e_stage2.pixel_explainability import write_explicabilidad_pixel_md  # noqa: F401
    from micorizae.phase_e_stage2.pixel_attention import extract_pixel_vit_attention  # noqa: F401


def test_segment_tile() -> None:
    import numpy as np

    from micorizae.phase_e_stage2.pixel_morph import quantize_segment, segment_tile_pixel_morph

    tile = np.random.randint(40, 200, (252, 252, 3), dtype=np.uint8)
    seg = segment_tile_pixel_morph(tile)
    assert seg.shape == (252, 252)
    q = quantize_segment(seg)
    assert "pct_colonized" in q


def test_prior_maps_agreement() -> None:
    import numpy as np

    from micorizae.phase_e_stage2.pixel_morph import segment_tile_pixel_morph
    from micorizae.phase_e_stage2.pixel_prior_maps import compute_prior_maps, prior_argmax_agreement

    tile = np.random.randint(50, 180, (128, 128, 3), dtype=np.uint8)
    y_weak = segment_tile_pixel_morph(tile)
    pm = compute_prior_maps(tile)
    y_prior = pm.evidence.argmax(axis=0).astype(np.uint8)
    ppa = prior_argmax_agreement(y_prior, y_weak, pm.root)
    assert ppa >= 0.0


def test_prior_loss_shapes() -> None:
    import torch

    from micorizae.phase_e_stage2.pixel_prior_loss import PriorLossWeights, combine_prior_losses

    b, c, h, w = 2, 5, 32, 32
    logits = torch.randn(b, c, h, w)
    prior = torch.rand(b, c, h, w)
    ves = (torch.rand(b, h, w) > 0.9).float()
    bd = combine_prior_losses(logits, prior, ves, weights=PriorLossWeights())
    assert bd.total.ndim == 0
    assert float(bd.total.item()) >= 0.0


def test_explain_metrics() -> None:
    import numpy as np

    from micorizae.phase_e_stage2.pixel_explain_metrics import expected_calibration_error

    pred = np.zeros((16, 16), dtype=np.uint8)
    ref = pred.copy()
    probs = np.zeros((5, 16, 16), dtype=np.float32)
    probs[0] = 1.0
    ece = expected_calibration_error(probs, pred, ref)
    assert 0.0 <= ece <= 1.0


def test_baseline_audit_exists() -> None:
    import config as user_config  # type: ignore

    gate = str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", ""))
    p = ROOT / "outputs" / "stage2_pixel_audit" / f"gate{gate}" / "STAGE2_PIXEL_BASELINE_AUDIT.json"
    assert p.exists(), f"Ejecutar stage2_pixel_baseline_audit.py — falta {p}"


def test_train_checkpoint() -> None:
    p = ROOT / "models" / "checkpoints" / "stage2_am" / "stage2_pixel_vit_best.pt"
    assert p.exists(), "Ejecutar train-stage2-pixel --full (ViT DINOv2)"


def main() -> None:
    required = [
        ("imports", test_imports),
        ("segment_tile", test_segment_tile),
        ("prior_maps_agreement", test_prior_maps_agreement),
        ("prior_loss_shapes", test_prior_loss_shapes),
        ("explain_metrics", test_explain_metrics),
    ]
    optional = [
        ("baseline_audit", test_baseline_audit_exists),
        ("train_checkpoint", test_train_checkpoint),
    ]
    ok, fail = [], []
    req_names = {t[0] for t in required}
    for name, fn in required + optional:
        try:
            fn()
            ok.append(name)
        except AssertionError as e:
            fail.append({"test": name, "error": str(e)})
        except Exception as e:
            if name in req_names:
                fail.append({"test": name, "error": str(e)})
            else:
                ok.append(f"{name}(skip)")

    out = {"ok": ok, "fail": fail, "passed": len([f for f in fail if f["test"] in req_names]) == 0}
    print(json.dumps(out, indent=2))
    if any(f["test"] in req_names for f in fail):
        sys.exit(1)


if __name__ == "__main__":
    main()
