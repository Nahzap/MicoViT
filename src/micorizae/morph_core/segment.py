"""Tile segmentation orchestration — stain maps + SRP detectors (I↔E)."""

from __future__ import annotations

import numpy as np

from .params import WeakSegParams
from .stain import ambiguous_stain_mask, stain_maps


def segment_tile_stain_aware(tile_rgb: np.ndarray, params: WeakSegParams) -> dict[str, np.ndarray]:
    """Una pasada de detectores SRP: BG → V → IH → A (+ arb_score cache)."""
    from micorizae.phase_e_stage2.atlas_log import _frangi_vesselness_tile, density_norm_in_mask
    from micorizae.phase_e_stage2.detectors import (
        detect_arbuscules,
        detect_hyphae,
        detect_root,
        detect_vesicles,
    )

    m = stain_maps(tile_rgb)
    root = detect_root(m, params).mask
    ih_sigmas = tuple(float(s) for s in params.ih_frangi_sigmas)
    m["frangi_ih"] = _frangi_vesselness_tile(m["density"], root, sigmas=ih_sigmas)
    m["frangi_ves"] = _frangi_vesselness_tile(
        density_norm_in_mask(m["density"], root),
        root,
        sigmas=(1.0, 2.0, 3.0),
    )
    # Orden SRP: V primero, IH excluye V, A excluye IH+V (un solo arb-score)
    ves_res = detect_vesicles(m["density"], root, params, frangi_map=m["frangi_ves"])
    vesicle = ves_res.mask
    ih_res = detect_hyphae(
        m["density"], root, params, exclude=vesicle, frangi_vessel=m["frangi_ih"]
    )
    hyphae = ih_res.mask
    arb_res = detect_arbuscules(
        m["density"], root, params, hyphae=hyphae, vesicle=vesicle, thr=0.35
    )
    arbuscule = arb_res.mask
    ambiguous = ambiguous_stain_mask(m, root, params)
    out = {
        "root": root,
        "hyphae": hyphae,
        "vesicle": vesicle,
        "arbuscule": arbuscule,
        "stain": m["stain"],
        "density": m["density"],
        "ambiguous": ambiguous,
        "frangi_ih": m["frangi_ih"],
        "frangi_ves": m["frangi_ves"],
    }
    if arb_res.score is not None:
        out["arb_score"] = arb_res.score
    if ves_res.score is not None:
        out["vesicle_score"] = ves_res.score
    if ih_res.score is not None:
        out["ih_score"] = ih_res.score
    return out


def segment_tile(tile_rgb: np.ndarray, params: WeakSegParams) -> dict[str, np.ndarray]:
    """Segmentación por tile — **solo** ruta stain-aware (detectores SRP).

    Si ``params.stain_aware=False``, se emite DeprecationWarning y se fuerza
    la ruta canónica. El grayscale vive en ``legacy_grayscale`` (no producción).
    """
    if not getattr(params, "stain_aware", True):
        import warnings

        warnings.warn(
            "WeakSegParams.stain_aware=False is deprecated; forcing stain-aware detectors.",
            DeprecationWarning,
            stacklevel=2,
        )
    return segment_tile_stain_aware(tile_rgb, params)
