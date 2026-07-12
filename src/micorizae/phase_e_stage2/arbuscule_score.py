"""Arb-score Gallaud — evidencia continua de arbúsculos (plan STAGE2 §3.6).

Criterio biológico (Gallaud 1905; McGonigle 1990; PROMETHEUS):
  ramificación dicotómica, ancho decreciente, textura fina, densidad alta.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.filters import frangi
from skimage.morphology import skeletonize

from micorizae.morph_core.params import WeakSegParams


def _normalize_in_mask(field: np.ndarray, mask: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = np.zeros_like(field, dtype=np.float32)
    m = mask.astype(bool)
    if not m.any():
        return out
    vals = field[m]
    vmax = float(np.max(vals)) + eps
    out[m] = (vals / vmax).astype(np.float32)
    return out


def _branch_density(mask: np.ndarray) -> np.ndarray:
    """Densidad local de bifurcaciones del esqueleto (proxy dicotomía Gallaud)."""
    m = mask.astype(bool)
    out = np.zeros(m.shape, dtype=np.float32)
    if m.sum() < 20:
        return out
    sk = skeletonize(m)
    if not sk.any():
        return out
    sk_u8 = sk.astype(np.uint8)
    # vecinos 8-conectados del esqueleto
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0.0
    neigh = cv2.filter2D(sk_u8.astype(np.float32), -1, k)
    branch_pts = sk & (neigh >= 3)
    # densidad local de puntos de rama
    dens = cv2.GaussianBlur(branch_pts.astype(np.float32), (0, 0), sigmaX=3.0)
    return _normalize_in_mask(dens, m)


def _width_gradient(mask: np.ndarray) -> np.ndarray:
    """Variación local del radio (distance transform) — ancho decreciente."""
    m = mask.astype(bool)
    out = np.zeros(m.shape, dtype=np.float32)
    if m.sum() < 20:
        return out
    dist = distance_transform_edt(m).astype(np.float32)
    # gradiente de radio: alta variación → estructura ramificada tipica A
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    return _normalize_in_mask(gmag, m)


def compute_arbuscule_score(
    density: np.ndarray,
    root: np.ndarray,
    *,
    hyphae: Optional[np.ndarray] = None,
    vesicle: Optional[np.ndarray] = None,
    params: Optional[WeakSegParams] = None,
    thr: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (arb_score continuo 0..1, arb_mask booleana).

    arb_score = 0.25·E_fine + 0.30·E_branch + 0.20·E_width
              + 0.15·E_dense + 0.10·(1−Frangi)
    """
    p = params or WeakSegParams()
    dens = density.astype(np.float32)
    root_b = root.astype(bool)
    hyphae_b = np.zeros_like(root_b) if hyphae is None else hyphae.astype(bool)
    vesicle_b = np.zeros_like(root_b) if vesicle is None else vesicle.astype(bool)
    valid = root_b & ~hyphae_b & ~vesicle_b
    if not valid.any():
        z = np.zeros_like(dens, dtype=np.float32)
        return z, z.astype(bool)

    # E_fine — textura alta frecuencia
    low = gaussian_filter(dens, sigma=3.0)
    fine = np.abs(dens - low)
    fine = gaussian_filter(fine, sigma=1.0)
    e_fine = _normalize_in_mask(fine, valid)

    # E_dense — densidad alta
    dens_thr = float(np.percentile(dens[root_b], p.arb_density_pctl)) if root_b.any() else 0.0
    dgate = 1.0 / (1.0 + np.exp(-(dens - dens_thr) / 0.05))
    e_dense = _normalize_in_mask(dgate.astype(np.float32), valid)

    # Candidato binario para skeleton (textura × densidad)
    fine_thr = float(np.percentile(fine[valid], p.arb_fine_pctl)) if valid.any() else 0.0
    cand = (fine >= fine_thr) & (dens >= dens_thr) & valid
    k = np.ones((3, 3), np.uint8)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k) > 0

    e_branch = _branch_density(cand)
    e_width = _width_gradient(cand)

    # (1 − Frangi): no tubular
    vessel = frangi(dens, sigmas=list(p.ih_frangi_sigmas), black_ridges=False).astype(np.float32)
    vessel_n = _normalize_in_mask(vessel, root_b)
    e_notube = np.clip(1.0 - vessel_n, 0.0, 1.0).astype(np.float32)

    score = (
        0.25 * e_fine
        + 0.30 * e_branch
        + 0.20 * e_width
        + 0.15 * e_dense
        + 0.10 * e_notube
    ).astype(np.float32)
    score = _normalize_in_mask(score, valid)

    # máscara: score alto y mayor que evidencia tubular/vesicular local
    ih_local = vessel_n
    v_local = vesicle_b.astype(np.float32)
    arb_mask = (score > float(thr)) & (score >= ih_local) & (score >= v_local) & valid
    if int(arb_mask.sum()) < int(getattr(p, "arb_min_area", 40)):
        # si el área total es pequeña, aún devolver score continuo
        pass
    # limpiar componentes pequeños
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(arb_mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(arb_mask)
    min_a = int(getattr(p, "arb_min_area", 40))
    for i in range(1, nlab):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_a:
            cleaned[labels == i] = True
    return score, cleaned
