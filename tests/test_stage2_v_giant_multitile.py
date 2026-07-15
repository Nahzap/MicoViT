"""Smoke V ATLAS multi-tile — contornos reales (sin discos Hough)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _paint_pale_vesicle_tile(
    h: int,
    w: int,
    *,
    cy_g: float,
    cx_g: float,
    r: float,
    row: int,
    col: int,
    tile_size: int,
    band: float = 5.0,
) -> np.ndarray:
    """Colonia densa + vesícula pálida esférica (anillo)."""
    rgb = np.full((h, w, 3), 35, dtype=np.uint8)
    rgb[:] = (35, 70, 170)
    y0 = row * tile_size
    x0 = col * tile_size
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy + y0 - cy_g) ** 2 + (xx + x0 - cx_g) ** 2)
    interior = dist <= r * 0.88
    ring = np.abs(dist - r) <= band
    rgb[interior] = (180, 200, 230)
    rgb[ring] = (20, 45, 110)
    return rgb


def test_atlas_rejects_colony_smear() -> None:
    """Tinción densa alargada (colonia) no debe marcarse como V."""
    from micorizae.phase_e_stage2.detectors.v_vesicle import detect_vesicle_masks_for_tiles

    ts = 126
    rgb = np.full((ts, ts, 3), 35, dtype=np.uint8)
    rgb[:] = (35, 70, 170)
    yy, xx = np.ogrid[:ts, :ts]
    smear = (np.abs(yy - xx) < 18) & (yy > 20) & (yy < 100)
    rgb[smear] = (20, 40, 100)
    masks = detect_vesicle_masks_for_tiles([rgb], [0], [0], ts)
    assert int(masks[0].sum()) < 200


def test_atlas_rejects_empty_cortex() -> None:
    """Corteza / tejido claro sin vesícula: 0 px V."""
    from micorizae.phase_e_stage2.detectors.v_vesicle import detect_vesicle_masks_for_tiles

    ts = 126
    rgb = np.full((ts, ts, 3), 210, dtype=np.uint8)
    rgb[:] = (190, 205, 220)
    yy, xx = np.ogrid[:ts, :ts]
    walls = (np.abs((yy - xx) % 18) < 2) | (np.abs((yy + xx) % 22) < 2)
    rgb[walls] = (120, 150, 190)
    masks = detect_vesicle_masks_for_tiles([rgb], [0], [0], ts)
    assert int(masks[0].sum()) < 150, f"empty cortex FP: {int(masks[0].sum())} px"


def test_atlas_multitile_pale_sphere() -> None:
    """Vesícula pálida esférica que cruza dos tiles (anillo real)."""
    from micorizae.phase_e_stage2.detectors.v_vesicle import detect_vesicle_masks_for_tiles

    ts = 126
    cy, cx, r = 63.0, 70.0, 55.0
    t00 = _paint_pale_vesicle_tile(ts, ts, cy_g=cy, cx_g=cx, r=r, row=0, col=0, tile_size=ts)
    t01 = _paint_pale_vesicle_tile(ts, ts, cy_g=cy, cx_g=cx, r=r, row=0, col=1, tile_size=ts)
    masks = detect_vesicle_masks_for_tiles([t00, t01], [0, 0], [0, 1], ts)
    total = int(masks[0].sum()) + int(masks[1].sum())
    assert total > 200, f"pale sphere too small: {total} px"


def test_apply_v_overrides_h_label() -> None:
    from micorizae.phase_e_stage2.detectors.v_vesicle import apply_v_masks_to_label_and_priors
    from micorizae.phase_e_stage2.pixel_class_map import PIXEL_CLASS_TO_IDX

    lab = np.full((64, 64), PIXEL_CLASS_TO_IDX["H"], dtype=np.uint8)
    vmask = np.zeros((64, 64), dtype=bool)
    vmask[10:40, 10:40] = True
    out, _, _ = apply_v_masks_to_label_and_priors(
        lab, None, None, vmask, input_size=64, v_idx=PIXEL_CLASS_TO_IDX["V"]
    )
    assert int((out[vmask] == PIXEL_CLASS_TO_IDX["V"]).sum()) == int(vmask.sum())


def test_atlas_dense_sphere_single_tile() -> None:
    """Vesícula densa azul oscuro dentro de un tile."""
    from micorizae.phase_e_stage2.detectors.v_vesicle import detect_vesicle_masks_for_tiles

    ts = 126
    rgb = np.full((ts, ts, 3), 200, dtype=np.uint8)
    rgb[:] = (180, 200, 220)
    cy, cx, r = 70.0, 75.0, 42.0
    yy, xx = np.ogrid[:ts, :ts]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rgb[dist <= r * 0.92] = (20, 40, 120)
    rgb[np.abs(dist - r) <= 3.5] = (10, 25, 80)
    masks = detect_vesicle_masks_for_tiles([rgb], [0], [0], ts)
    assert int(masks[0].sum()) > 400, f"dense V too small: {int(masks[0].sum())}"


def test_pseudo_gt_version_v5_atlas() -> None:
    import config as user_config

    assert "atlas_multitile" in str(user_config.STAGE2_PIXEL_PSEUDO_GT_VERSION)
    from micorizae.phase_e_stage2.pixel_prior_maps import PRIOR_IMPL_VERSION

    assert PRIOR_IMPL_VERSION == 16


def test_giant_api_redirects_to_atlas() -> None:
    """Compat: detect_giant_* debe devolver lo mismo que ATLAS (no Hough)."""
    from micorizae.phase_e_stage2.detectors.v_giant_multitile import (
        detect_giant_vesicle_masks_for_tiles,
    )
    from micorizae.phase_e_stage2.detectors.v_vesicle import detect_vesicle_masks_for_tiles

    ts = 64
    rgb = np.full((ts, ts, 3), 200, dtype=np.uint8)
    a = detect_vesicle_masks_for_tiles([rgb], [0], [0], ts)
    b = detect_giant_vesicle_masks_for_tiles([rgb], [0], [0], ts)
    assert int(a[0].sum()) == int(b[0].sum())
