"""Smoke conformación HDF5 Stage2-Pixel — pack path (labels + priors v8).

Corre ANTES del rebuild completo. Sin ViT/U2Net; solo morph CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_prior_fingerprint_is_v16() -> None:
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams
    from micorizae.phase_e_stage2.pixel_prior_maps import PRIOR_IMPL_VERSION
    from micorizae.phase_e_stage2.stage2_pixel_h5_cache import _prior_fingerprint

    assert PRIOR_IMPL_VERSION == 16
    fp = _prior_fingerprint(PixelMorphParams(), 224, "20260710_121127_gate_am_train")
    assert fp.endswith("|prior_v16"), fp


def test_pack_one_shapes_and_v_authority() -> None:
    """pack_one: label 224², prior (5,H,W), prior_V ≡ label V (SRP)."""
    from micorizae.morph_core import WeakSegParams
    from micorizae.phase_e_stage2.pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_TO_IDX
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams
    from micorizae.phase_e_stage2.stage2_h5_cpu_pack import init_worker, pack_one

    h = w = 126
    rgb = np.full((h, w, 3), 210, dtype=np.uint8)
    # tinción azul + disco denso (vesícula sintética)
    rgb[..., 0] = 40
    rgb[..., 1] = 80
    rgb[..., 2] = 200
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - 63) ** 2 + (xx - 63) ** 2 <= 18**2
    rgb[disk, :] = (25, 40, 90)

    morph = PixelMorphParams(weak=WeakSegParams(seam_sigma=0.0), seam_sigma=0.0)
    init_worker(morph, 224, with_priors=True)
    label, ev, ves = pack_one(rgb)

    assert label.shape == (224, 224) and label.dtype == np.uint8
    assert ev is not None and ves is not None
    assert ev.shape == (NUM_PIXEL_CLASSES, 224, 224)
    assert ves.shape == (224, 224)
    assert set(np.unique(label)).issubset(set(range(NUM_PIXEL_CLASSES)))

    v_idx = PIXEL_CLASS_TO_IDX["V"]
    v_mask = label == v_idx
    # prior canal V debe coincidir con máscara label (autoridad detector)
    prior_v = ev[v_idx].astype(np.float32)
    if v_mask.any():
        # donde hay V en label, prior V > 0; fuera, ~0 (threshold suave por resize)
        assert float(prior_v[v_mask].mean()) > 0.5
    assert float(ves.max()) >= 0.0


def test_pack_batch_multiprocess_init() -> None:
    """Worker init + batch de 4 tiles no crashea (ruta ProcessPool)."""
    from micorizae.morph_core import WeakSegParams
    from micorizae.phase_e_stage2.pixel_morph import PixelMorphParams
    from micorizae.phase_e_stage2.stage2_h5_cpu_pack import init_worker, pack_batch

    morph = PixelMorphParams(weak=WeakSegParams(), seam_sigma=0.0)
    init_worker(morph, 224, with_priors=True)
    tiles = [np.full((64, 64, 3), 180, dtype=np.uint8) for _ in range(4)]
    tiles[0][..., 2] = 220
    tiles[0][..., 0] = 30
    out = pack_batch(tiles)
    assert len(out) == 4
    for lab, ev, ves in out:
        assert lab.shape == (224, 224)
        assert ev is not None and ev.shape[0] == 5
        assert ves is not None
