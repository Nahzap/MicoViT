"""Tests SRP — un módulo = una técnica por clase."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_spherical_filter_only_in_atlas_log() -> None:
    root = _SRC / "micorizae"
    leaks = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        if "filter_vesicle_mask_spherical(" not in text:
            continue
        if p.name == "atlas_log.py":
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
                continue
            if "filter_vesicle_mask_spherical(" in s:
                leaks.append(f"{p}:{i}:{s}")
    assert not leaks, f"SRP leak spherical: {leaks}"


def test_detectors_package_exports() -> None:
    from micorizae.phase_e_stage2.detectors import (
        detect_arbuscules,
        detect_colony,
        detect_hyphae,
        detect_root,
        detect_vesicles,
    )

    assert callable(detect_root)
    assert callable(detect_hyphae)
    assert callable(detect_vesicles)
    assert callable(detect_arbuscules)
    assert callable(detect_colony)


def test_production_does_not_import_legacy_grayscale() -> None:
    root = _SRC / "micorizae"
    leaks = []
    for p in root.rglob("*.py"):
        if p.name == "legacy_grayscale.py":
            continue
        text = p.read_text(encoding="utf-8")
        if "legacy_grayscale" not in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if "legacy_grayscale" in s and ("import" in s or "from" in s):
                leaks.append(f"{p}:{i}:{s}")
    assert not leaks, f"production imports legacy_grayscale: {leaks}"


def test_segment_tile_forces_stain_aware() -> None:
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams, segment_tile

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = segment_tile(np.zeros((24, 24, 3), dtype=np.uint8), WeakSegParams(stain_aware=False))
    assert any("stain_aware" in str(w.message) for w in caught)
    assert "vesicle" in out and "density" in out


def test_compose_v_retention_synthetic() -> None:
    from micorizae.phase_e_stage2.morph_pipeline import compose_pixel_map
    from micorizae.phase_e_stage2.pixel_class_map import PIXEL_CLASS_TO_IDX
    from micorizae.phase_i_weakseg.pipeline import WeakSegParams

    h, w = 64, 64
    vesicle = np.zeros((h, w), dtype=bool)
    vesicle[20:35, 20:35] = True
    root = np.ones((h, w), dtype=bool)
    masks = {
        "root": root,
        "hyphae": np.zeros((h, w), dtype=bool),
        "vesicle": vesicle,
        "arbuscule": np.zeros((h, w), dtype=bool),
        "ambiguous": np.zeros((h, w), dtype=bool),
        "stain": np.full((h, w), 0.5, dtype=np.float32),
        "density": np.full((h, w), 0.5, dtype=np.float32),
    }
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    seg = compose_pixel_map(rgb, weak=WeakSegParams(), seam_sigma=0.85, masks=masks)
    v_idx = PIXEL_CLASS_TO_IDX["V"]
    assert int((seg == v_idx).sum()) == int(vesicle.sum())
    assert np.array_equal(seg == v_idx, vesicle)
