"""Detección de vesículas por contorno cerrado + LoG multi-escala acotado.

Las vesículas micorrícicas son estructuras esféricas/elipsoidales con contorno
cerrado visible en tinción (McGonigle et al.; Basset ATLAS IEEE TIP 2015).
No se generan discos sintéticos: solo contornos extraídos de la imagen que
superan esfericidad, solidez y contraste anular.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def adaptive_ves_max_radius(shape: tuple[int, int], configured: int = 0) -> int:
    """Radio máximo: 0 → adaptativo min(min(H,W)//2, 80)."""
    if configured and configured > 0:
        return int(configured)
    h, w = int(shape[0]), int(shape[1])
    return int(min(max(h, w) // 2, 80))


def adaptive_ves_max_sigma(shape: tuple[int, int], configured: float = 40.0) -> float:
    """σ máximo LoG acotado por el radio adaptativo (r ≈ σ√2)."""
    r_max = adaptive_ves_max_radius(shape, 0)
    sigma_from_r = float(r_max) / np.sqrt(2.0)
    return float(min(max(configured, 16.0), sigma_from_r))


def stain_residual(density: np.ndarray, open_radius: int = 15) -> np.ndarray:
    """Picos locales de tinción: density − morph_open(density)."""
    d = np.clip(density.astype(np.float32), 0.0, 1.0)
    d8 = (d * 255.0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius * 2 + 1, open_radius * 2 + 1))
    opened = cv2.morphologyEx(d8, cv2.MORPH_OPEN, k).astype(np.float32) / 255.0
    return np.clip(d - opened, 0.0, 1.0).astype(np.float32)


def density_norm_in_mask(density: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normaliza OD a [0,1] por percentil 5–95 dentro de máscara de tejido."""
    out = np.zeros_like(density, dtype=np.float32)
    m = mask.astype(bool)
    if not m.any():
        return out
    vals = density[m].astype(np.float32)
    p5, p95 = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    scale = max(p95 - p5, eps)
    out[m] = np.clip((vals - p5) / scale, 0.0, 1.0)
    return out


def _circularity_from_contour(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if area <= 0.0 or peri < 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _solidity_from_contour(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area < 1e-6:
        return 0.0
    return float(area / hull_area)


def _ellipse_aspect(contour: np.ndarray) -> float:
    """minor/major del elipse ajustado; 1.0 = círculo perfecto."""
    if len(contour) < 5:
        return 0.0
    try:
        _, (w, h), _ = cv2.fitEllipse(contour)
        ma, mi = float(max(w, h)), float(min(w, h))
        if ma < 1e-6:
            return 0.0
        return float(mi / ma)
    except cv2.error:
        return 0.0


def _contour_is_closed(contour: np.ndarray, *, tol_px: float = 2.5) -> bool:
    """Contorno cerrado topológico (findContours / área positiva).

    OpenCV no garantiza el primer vértice; CHAIN_APPROX_SIMPLE puede dejar
    extremos lejanos aun en regiones cerradas. Criterio: perímetro+área y
    (extremos cercanos O región rellenable coherente).
    """
    if contour is None or len(contour) < 5:
        return False
    peri = float(cv2.arcLength(contour, True))
    area = abs(float(cv2.contourArea(contour)))
    if peri < 1e-3 or area < 1.0:
        return False
    p0 = contour[0, 0].astype(np.float32)
    p1 = contour[-1, 0].astype(np.float32)
    if float(np.linalg.norm(p0 - p1)) <= float(tol_px):
        return True
    # Contorno de findContours: cerrado al dibujar; extremos pueden no coincidir.
    return area >= 4.0 and peri >= 8.0


def _contour_annulus_contrast(
    density: np.ndarray,
    contour: np.ndarray,
    *,
    ring_scale: float = 1.6,
    contrast_min: float = 0.08,
) -> bool:
    """Contraste anular: V densa (interior > exterior) o V de pared (anillo).

    McGonigle: vesículas rellenas o pálidas con pared teñida. Rechaza placas
    planas sin contraste radial (hifa/colonia aplastada).
    """
    mask = np.zeros(density.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
    area = int(mask.sum())
    if area < 4:
        return False
    ys, xs = np.where(mask > 0)
    cy, cx = float(ys.mean()), float(xs.mean())
    r = float(np.sqrt(area / np.pi))
    r_in = max(r * 0.55, 2.0)
    r_out = max(r * float(ring_scale), r_in + 2.0)
    yy, xx = np.ogrid[: density.shape[0], : density.shape[1]]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    inner = dist2 <= r_in * r_in
    # Anillo exterior (fuera del disco) + anillo de pared (borde del contorno)
    ring_out = (dist2 > r * r) & (dist2 <= r_out * r_out)
    wall = (dist2 > (r * 0.72) ** 2) & (dist2 <= (r * 1.08) ** 2) & (mask > 0)
    if int(inner.sum()) < 2:
        return False
    d_in = float(density[inner].mean())
    # (1) V densa: interior más teñido que exterior
    if int(ring_out.sum()) >= 2:
        if (d_in - float(density[ring_out].mean())) >= float(contrast_min):
            return True
    # (2) V pálida con pared: anillo/pared más denso que núcleo
    if int(wall.sum()) >= 2:
        if (float(density[wall].mean()) - d_in) >= float(contrast_min) * 0.85:
            return True
    return False


def _root_bbox(mask: np.ndarray, *, pad: int = 0) -> Optional[tuple[int, int, int, int]]:
    """BBox (y0, y1, x0, x1) del tejido; None si vacío."""
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = mask.shape
    if pad > 0:
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h, y1 + pad)
        x1 = min(w, x1 + pad)
    return y0, y1, x0, x1


def _frangi_vesselness_tile(
    density: np.ndarray,
    root: np.ndarray,
    *,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> np.ndarray:
    """Frangi (1998) vesselness una vez por tile — misma función, sin recomputar por semilla."""
    from skimage.filters import frangi

    root_b = root.astype(bool)
    out = np.zeros(density.shape, dtype=np.float32)
    bbox = _root_bbox(root_b, pad=int(max(sigmas) * 3) + 4)
    if bbox is None:
        return out
    y0, y1, x0, x1 = bbox
    patch = density[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 9:
        return out
    vessel = frangi(patch, sigmas=tuple(float(s) for s in sigmas), black_ridges=False).astype(np.float32)
    out[y0:y1, x0:x1] = vessel
    return out


def _blob_is_tubular(
    density: np.ndarray,
    y: int,
    x: int,
    radius: int,
    *,
    tubular_ratio: float = 1.35,
    frangi_map: Optional[np.ndarray] = None,
) -> bool:
    """Rechaza picos LoG alineados a crestas tubulares (hifas), no esferas."""
    from skimage.filters import frangi

    h, w = density.shape
    pad = max(int(radius * 2), 8)
    y0, y1 = max(0, y - pad), min(h, y + pad + 1)
    x0, x1 = max(0, x - pad), min(w, x + pad + 1)
    patch = density[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 9:
        return False
    cy, cx = y - y0, x - x0
    cy = int(np.clip(cy, 0, patch.shape[0] - 1))
    cx = int(np.clip(cx, 0, patch.shape[1] - 1))
    if frangi_map is not None:
        fr = float(frangi_map[y, x])
    else:
        vessel = frangi(patch, sigmas=(1.0, 2.0, 3.0), black_ridges=False)
        fr = float(vessel[cy, cx])
    sigma = max(float(radius) / np.sqrt(2.0), 1.0)
    from scipy.ndimage import gaussian_laplace

    log_resp = float(-gaussian_laplace(patch, sigma=sigma)[cy, cx] * (sigma**2))
    if log_resp <= 1e-6:
        return fr > 0.06
    return fr / log_resp > float(tubular_ratio)


def _contour_on_tubular_ridge(
    density: np.ndarray,
    contour: np.ndarray,
    *,
    tubular_ratio: float = 1.25,
    frangi_map: Optional[np.ndarray] = None,
) -> bool:
    """Rechaza contornos alargados sobre crestas tubulares (hifas)."""
    from skimage.filters import frangi

    mask = np.zeros(density.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return True
    cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
    r = int(max(4, round(float(np.sqrt(mask.sum() / np.pi)))))
    h, w = density.shape
    pad = max(r * 2, 8)
    y0, y1 = max(0, cy - pad), min(h, cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(w, cx + pad + 1)
    patch = density[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 9:
        return False
    lcy, lcx = cy - y0, cx - x0
    lcy = int(np.clip(lcy, 0, patch.shape[0] - 1))
    lcx = int(np.clip(lcx, 0, patch.shape[1] - 1))
    if frangi_map is not None:
        fr = float(frangi_map[cy, cx])
    else:
        vessel = frangi(patch, sigmas=(1.0, 2.0, 3.0), black_ridges=False)
        fr = float(vessel[lcy, lcx])
    sigma = max(float(r) / np.sqrt(2.0), 1.0)
    from scipy.ndimage import gaussian_laplace

    log_resp = float(-gaussian_laplace(patch, sigma=sigma)[lcy, lcx] * (sigma**2))
    if log_resp <= 1e-6:
        return fr > 0.06
    return fr / log_resp > float(tubular_ratio)


def filter_vesicle_mask_spherical(
    mask: np.ndarray,
    *,
    min_area: int = 30,
    roundness_min: float = 0.76,
    solidity_min: float = 0.85,
    aspect_min: float = 0.58,
) -> np.ndarray:
    """Filtra CC: contorno cerrado + esfericidad + solidez + elipse."""
    m = mask.astype(np.uint8)
    if int(m.sum()) < int(min_area):
        return np.zeros_like(mask, dtype=bool)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if not _contour_is_closed(c):
            continue
        circ = _circularity_from_contour(c)
        sol = _solidity_from_contour(c)
        asp = _ellipse_aspect(c)
        if circ >= float(roundness_min) and sol >= float(solidity_min) and asp >= float(aspect_min):
            cv2.drawContours(out, [c], -1, 1, thickness=-1)
    return out > 0


def _collect_closed_contours_at_scale(
    dens_n: np.ndarray,
    root: np.ndarray,
    *,
    blur_sigma: float,
    density_pctl: float,
    open_radius: int,
) -> list[np.ndarray]:
    """Contornos cerrados externos en una escala de suavizado."""
    blur = gaussian_filter(dens_n.astype(np.float32), sigma=float(blur_sigma))
    root_b = root.astype(bool)
    vals = blur[root_b]
    if vals.size == 0:
        return []
    thr = float(np.percentile(vals, float(density_pctl)))
    binary = ((blur >= thr) & root_b).astype(np.uint8)
    if open_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_radius * 2 + 1, open_radius * 2 + 1)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if _contour_is_closed(c)]


def _nms_blob_centers(
    blobs: np.ndarray,
    *,
    min_dist_ratio: float = 0.55,
) -> np.ndarray:
    """Non-maximum suppression sobre centros LoG."""
    if blobs.size == 0:
        return blobs.reshape(0, 3)
    order = np.argsort(-blobs[:, 2])
    kept: list[tuple[float, float, float]] = []
    for idx in order:
        y, x, sigma = blobs[idx]
        r = float(sigma) * np.sqrt(2.0)
        ok = True
        for y2, x2, sigma2 in kept:
            r2 = float(sigma2) * np.sqrt(2.0)
            dist = float(np.hypot(y - y2, x - x2))
            if dist < float(min_dist_ratio) * min(r, r2):
                ok = False
                break
        if ok:
            kept.append((float(y), float(x), float(sigma)))
    if not kept:
        return blobs.reshape(0, 3)
    return np.asarray(kept, dtype=np.float64)


def _extract_contour_at_log_seed(
    dens_n: np.ndarray,
    residual: np.ndarray,
    root: np.ndarray,
    y: int,
    x: int,
    sigma: float,
    *,
    roundness_min: float,
    solidity_min: float,
    aspect_min: float,
    contrast_min: float,
    min_area: int,
) -> Optional[np.ndarray]:
    """Refina semilla LoG → contorno cerrado real en parche local (sin disco sintético)."""
    h, w = dens_n.shape
    r = max(3, int(round(float(sigma) * np.sqrt(2.0))))
    pad = max(int(r * 2.5), 14)
    y0, y1 = max(0, y - pad), min(h, y + pad + 1)
    x0, x1 = max(0, x - pad), min(w, x + pad + 1)
    d_patch = dens_n[y0:y1, x0:x1]
    r_patch = residual[y0:y1, x0:x1]
    root_p = root[y0:y1, x0:x1].astype(bool)
    if d_patch.size == 0 or not root_p.any():
        return None

    field = np.clip(0.55 * d_patch + 0.45 * r_patch, 0.0, 1.0)
    blur = gaussian_filter(field, sigma=max(0.8, float(sigma) * 0.25))
    vals = blur[root_p]
    if vals.size == 0:
        return None
    thr = float(np.percentile(vals, 72.0))
    binary = ((blur >= thr) & root_p).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, r // 2) * 2 + 1, max(3, r // 2) * 2 + 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cy, cx = y - y0, x - x0
    best: Optional[np.ndarray] = None
    best_score = -1.0
    for c in cnts:
        if not _contour_is_closed(c):
            continue
        area = float(cv2.contourArea(c))
        # Cap: ~disco de radio 1.45·r (morph close puede hinchar el blob)
        area_cap = float(min(pad * pad * 0.85, np.pi * (float(r) * 1.45) ** 2))
        if area < float(min_area) or area > area_cap:
            continue
        if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) < 0:
            continue
        circ = _circularity_from_contour(c)
        sol = _solidity_from_contour(c)
        asp = _ellipse_aspect(c)
        req_circ = float(roundness_min) if area < 400 else max(float(roundness_min), 0.78)
        if circ < req_circ or circ < 0.68:
            continue
        if sol < float(solidity_min) or asp < float(aspect_min):
            continue
        if not _contour_annulus_contrast(d_patch, c, contrast_min=float(contrast_min) * 0.75):
            continue
        score = float(circ * sol * asp * np.sqrt(area))
        if score > best_score:
            best_score = score
            best = c + np.array([[x0, y0]], dtype=c.dtype)

    return best


def _collect_log_seed_contours(
    dens_n: np.ndarray,
    residual: np.ndarray,
    root: np.ndarray,
    *,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    nms_dist_ratio: float,
    roundness_min: float,
    solidity_min: float,
    aspect_min: float,
    contrast_min: float,
    min_area: int,
    reject_tubular: bool,
    frangi_map: Optional[np.ndarray] = None,
) -> list[tuple[np.ndarray, float]]:
    """Semillas LoG multi-escala → contornos cerrados locales."""
    from skimage.feature import blob_log

    root_b = root.astype(bool)
    dens = dens_n.astype(np.float32) * root_b
    if dens.max() <= 1e-6:
        return []

    r_max = adaptive_ves_max_radius(dens.shape, 0)
    sigma_max = adaptive_ves_max_sigma(dens.shape, max_sigma)
    pad = int(float(sigma_max) * 3.0) + 4
    bbox = _root_bbox(root_b, pad=pad)
    if bbox is None:
        return []
    y0, y1, x0, x1 = bbox
    dens_roi = dens[y0:y1, x0:x1]
    blobs = blob_log(
        dens_roi,
        min_sigma=float(min_sigma),
        max_sigma=float(sigma_max),
        num_sigma=int(num_sigma),
        threshold=float(threshold),
    )
    if blobs.size == 0:
        return []
    if blobs.ndim == 1:
        blobs = blobs.reshape(1, 3)
    blobs[:, 0] += float(y0)
    blobs[:, 1] += float(x0)

    out: list[tuple[np.ndarray, float]] = []
    for y, x, sigma in _nms_blob_centers(blobs, min_dist_ratio=float(nms_dist_ratio)):
        yy, xx = int(round(y)), int(round(x))
        rr = int(round(float(sigma) * np.sqrt(2.0)))
        if rr < 2 or rr > r_max:
            continue
        if not root_b[yy, xx]:
            continue
        if reject_tubular and _blob_is_tubular(
            dens_n, yy, xx, rr, tubular_ratio=1.45, frangi_map=frangi_map
        ):
            continue
        c = _extract_contour_at_log_seed(
            dens_n,
            residual,
            root_b,
            yy,
            xx,
            float(sigma),
            roundness_min=float(roundness_min),
            solidity_min=float(solidity_min),
            aspect_min=float(aspect_min),
            contrast_min=float(contrast_min),
            min_area=int(min_area),
        )
        if c is None:
            continue
        area = float(cv2.contourArea(c))
        circ = _circularity_from_contour(c)
        sol = _solidity_from_contour(c)
        asp = _ellipse_aspect(c)
        score = float(circ * sol * asp * np.sqrt(area))
        out.append((c, score))
    return out


def _contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        pts = contour.reshape(-1, 2)
        return float(pts[:, 1].mean()), float(pts[:, 0].mean())
    return float(m["m01"] / m["m00"]), float(m["m10"] / m["m00"])


def _nms_contours(
    accepted: list[tuple[np.ndarray, float]],
    *,
    min_dist_ratio: float = 0.65,
) -> list[np.ndarray]:
    """Suprime contornos duplicados multi-escala por proximidad de centroides."""
    if not accepted:
        return []
    accepted.sort(key=lambda x: -x[1])
    kept: list[np.ndarray] = []
    kept_meta: list[tuple[float, float, float]] = []
    for c, _score in accepted:
        area = float(cv2.contourArea(c))
        r = float(np.sqrt(max(area, 1.0) / np.pi))
        cy, cx = _contour_centroid(c)
        ok = True
        for cy2, cx2, r2 in kept_meta:
            if float(np.hypot(cy - cy2, cx - cx2)) < float(min_dist_ratio) * min(r, r2):
                ok = False
                break
        if ok:
            kept.append(c)
            kept_meta.append((cy, cx, r))
    return kept


def _contour_boundary_strength(density: np.ndarray, contour: np.ndarray, *, min_mean: float = 0.035) -> bool:
    """Contorno cerrado con borde visible (gradiente morfológico a lo largo del perímetro)."""
    mask = np.zeros(density.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(
        (np.clip(density, 0, 1) * 255).astype(np.uint8), cv2.MORPH_GRADIENT, k
    ).astype(np.float32) / 255.0
    edge = np.zeros_like(mask)
    cv2.drawContours(edge, [contour], -1, 1, thickness=2)
    vals = grad[edge > 0]
    if vals.size < 8:
        return False
    return float(vals.mean()) >= float(min_mean)


def _roundness_required(area: float, *, base: float, large_area: float = 400.0, large_min: float = 0.82) -> float:
    """Vesículas grandes parciales en tile deben ser más circulares (evita hifas gruesas)."""
    if area >= float(large_area):
        return max(float(base), float(large_min))
    return float(base)


def detect_vesicles_closed_contour(
    density: np.ndarray,
    root: np.ndarray,
    *,
    min_area: int = 30,
    max_area: int = 0,
    roundness_min: float = 0.72,
    solidity_min: float = 0.82,
    aspect_min: float = 0.52,
    contrast_min: float = 0.06,
    density_pctl: float = 76.0,
    reject_tubular: bool = True,
    max_area_frac: float = 0.55,
    large_roundness_min: float = 0.80,
    min_contour_score: float = 4.5,
    max_instances: int = 20,
    min_sigma: float = 2.0,
    max_sigma: float = 40.0,
    num_sigma: int = 12,
    log_threshold: float = 0.038,
    nms_dist_ratio: float = 0.55,
    check_boundary: bool = False,
    frangi_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vesículas = contornos cerrados reales.

    Ruta principal: semilla LoG multi-escala → contorno local (sin discos sintéticos).
    Ruta secundaria (solo si faltan candidatos): 2 escalas density para vesículas grandes.
    """
    root_b = root.astype(bool)
    dens = density.astype(np.float32)
    if dens.max() <= 1e-6 or not root_b.any():
        return np.zeros_like(root_b, dtype=bool)

    dens_n = density_norm_in_mask(dens, root_b)
    res_n = density_norm_in_mask(stain_residual(dens, open_radius=11), root_b)
    if frangi_map is None and reject_tubular:
        frangi_map = _frangi_vesselness_tile(dens_n, root_b)
    elif frangi_map is not None:
        frangi_map = frangi_map.astype(np.float32)
    r_max = adaptive_ves_max_radius(dens.shape, 0)
    # Vesículas grandes (~0.45·tile): π·r_max²; frac acota respecto al tejido.
    area_from_r = int(np.pi * float(r_max) * float(r_max))
    area_from_frac = int(root_b.sum() * float(max_area_frac))
    area_cap = int(max_area) if max_area > 0 else int(min(area_from_r, area_from_frac))
    area_cap = max(area_cap, int(min_area))

    candidates: list[tuple[np.ndarray, float]] = _collect_log_seed_contours(
        dens_n,
        res_n,
        root_b,
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        num_sigma=int(num_sigma),
        threshold=float(log_threshold),
        nms_dist_ratio=float(nms_dist_ratio),
        roundness_min=float(roundness_min),
        solidity_min=float(solidity_min),
        aspect_min=float(aspect_min),
        contrast_min=float(contrast_min),
        min_area=int(min_area),
        reject_tubular=bool(reject_tubular),
        frangi_map=frangi_map,
    )

    def _try_contour(c: np.ndarray, field: np.ndarray) -> None:
        area = float(cv2.contourArea(c))
        if area < float(min_area) or area > float(area_cap):
            return
        if float(np.sqrt(area / np.pi)) > float(r_max):
            return
        if not _contour_is_closed(c):
            return
        req_circ = _roundness_required(
            area, base=float(roundness_min), large_min=float(large_roundness_min)
        )
        circ = _circularity_from_contour(c)
        sol = _solidity_from_contour(c)
        asp = _ellipse_aspect(c)
        req_asp = float(aspect_min) if area < 400.0 else max(float(aspect_min), 0.60)
        if circ < req_circ or circ < 0.65:
            return
        if sol < float(solidity_min) or asp < req_asp:
            return
        if not _contour_annulus_contrast(field, c, contrast_min=float(contrast_min)):
            return
        if check_boundary and not _contour_boundary_strength(field, c, min_mean=0.022):
            return
        if reject_tubular and _contour_on_tubular_ridge(
            field, c, tubular_ratio=1.35, frangi_map=frangi_map
        ):
            return
        score = float(circ * sol * asp * np.sqrt(area))
        if score >= float(min_contour_score):
            candidates.append((c, score))

    # Secundario: vesículas grandes que LoG puede perder (solo si hay pocos candidatos)
    if len(candidates) < 3:
        for blur_sigma, open_r, pctl in (
            (2.0, 3, float(density_pctl)),
            (5.5, 7, float(density_pctl) - 4.0),
        ):
            for c in _collect_closed_contours_at_scale(
                dens_n, root_b, blur_sigma=blur_sigma, density_pctl=pctl, open_radius=open_r
            ):
                _try_contour(c, dens_n)

    if not candidates:
        return np.zeros_like(root_b, dtype=bool)

    candidates.sort(key=lambda x: -x[1])
    if int(max_instances) > 0:
        candidates = candidates[: int(max_instances)]

    out = np.zeros(dens.shape, dtype=np.uint8)
    for c in _nms_contours(candidates, min_dist_ratio=float(nms_dist_ratio)):
        cv2.drawContours(out, [c], -1, 1, thickness=-1)

    filtered = filter_vesicle_mask_spherical(
        out > 0,
        min_area=int(min_area),
        roundness_min=float(roundness_min),
        solidity_min=float(solidity_min),
        aspect_min=float(aspect_min),
    )
    if not filtered.any():
        return filtered & root_b

    nlab, _, stats, _ = cv2.connectedComponentsWithStats(filtered.astype(np.uint8), connectivity=8)
    if nlab > 1:
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, nlab)]
        total_a = int(sum(areas))
        if len(areas) >= 8 and total_a < 500 and max(areas) < 80:
            return np.zeros_like(root_b, dtype=bool)

    return filtered & root_b


def detect_vesicle_blobs_atlas(
    density: np.ndarray,
    root: np.ndarray,
    *,
    min_sigma: float = 2.0,
    max_sigma: Optional[float] = None,
    num_sigma: int = 12,
    threshold: float = 0.038,
    contrast_min: float = 0.06,
    bg_max: float = 0.45,
    min_area: int = 30,
    max_radius: int = 0,
    roundness_min: float = 0.72,
    solidity_min: float = 0.82,
    nms_dist_ratio: float = 0.55,
    reject_tubular: bool = True,
    density_pctl: float = 76.0,
    max_instances: int = 20,
    frangi_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """API estable → híbrido contorno + semillas LoG refinadas (sin discos sintéticos)."""
    _ = bg_max, max_radius
    return detect_vesicles_closed_contour(
        density,
        root,
        min_area=int(min_area),
        roundness_min=float(roundness_min),
        solidity_min=float(solidity_min),
        contrast_min=float(contrast_min),
        density_pctl=float(density_pctl),
        reject_tubular=bool(reject_tubular),
        max_instances=int(max_instances),
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma if max_sigma is not None else 40.0),
        num_sigma=int(num_sigma),
        log_threshold=float(threshold),
        nms_dist_ratio=float(nms_dist_ratio),
        frangi_map=frangi_map,
    )


def vesicle_prior_from_mask(
    vesicle_mask: np.ndarray,
    root: np.ndarray,
) -> np.ndarray:
    """Prior V acotado a contornos detectados — sin propagación global."""
    from scipy.ndimage import distance_transform_edt

    m = (vesicle_mask > 0) & (root.astype(bool))
    out = np.zeros_like(vesicle_mask, dtype=np.float32)
    if not m.any():
        return out
    dist = distance_transform_edt(m)
    vmax = float(dist[m].max()) + 1e-9
    out[m] = (dist[m] / vmax).astype(np.float32)
    return out


def continuous_log_response(
    density: np.ndarray,
    root: np.ndarray,
    *,
    min_sigma: float = 2.0,
    max_sigma: float = 40.0,
    num_sigma: int = 8,
    vesicle_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mapa LoG acotado a máscara vesicular (nunca global sobre contorno azul)."""
    if vesicle_mask is None:
        return vesicle_prior_from_mask(
            detect_vesicles_closed_contour(density, root), root
        )
    return vesicle_prior_from_mask(vesicle_mask, root)


def _tile_grid_components(
    rows: list[int],
    cols: list[int],
) -> list[list[int]]:
    """Componentes conexas en la grilla (Chebyshev ≤ 1 = vecinas 8-conectadas)."""
    n = len(rows)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if max(abs(int(rows[i]) - int(rows[j])), abs(int(cols[i]) - int(cols[j]))) <= 1:
                union(i, j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def detect_vesicle_masks_atlas_for_tiles(
    densities: list[np.ndarray],
    roots: list[np.ndarray],
    rows: list[int],
    cols: list[int],
    tile_size: int,
    *,
    min_sigma: float = 2.0,
    max_sigma: Optional[float] = None,
    num_sigma: int = 14,
    threshold: float = 0.038,
    contrast_min: float = 0.06,
    min_area: int = 30,
    roundness_min: float = 0.72,
    solidity_min: float = 0.82,
    nms_dist_ratio: float = 0.55,
    density_pctl: float = 76.0,
    max_instances: int = 40,
    max_mosaic_px: int = 6_000_000,
) -> list[np.ndarray]:
    """V multi-tile: ATLAS LoG sobre mosaicos locales (contornos reales).

    No Hough. Agrupa tiles en componentes conexas de la grilla para no montar
    un bbox enorme casi vacío entre tiles lejanos (Frangi/LoG se congelan).
    """
    n = len(densities)
    assert n == len(roots) == len(rows) == len(cols)
    if n == 0:
        return []
    ts = int(tile_size)
    out = [np.zeros(densities[i].shape[:2], dtype=bool) for i in range(n)]

    # σ grande para vesículas que cruzan bordes (~0.45·tile)
    sigma_cap = float(max_sigma) if max_sigma is not None else max(40.0, 0.45 * float(ts))
    sigma_cap = float(min(sigma_cap, 0.55 * float(ts)))

    atlas_kw = dict(
        min_sigma=float(min_sigma),
        max_sigma=sigma_cap,
        num_sigma=int(num_sigma),
        threshold=float(threshold),
        contrast_min=float(contrast_min),
        min_area=int(min_area),
        max_radius=int(0.55 * ts),
        roundness_min=float(roundness_min),
        solidity_min=float(solidity_min),
        nms_dist_ratio=float(nms_dist_ratio),
        reject_tubular=True,
        density_pctl=float(density_pctl),
        max_instances=int(max_instances),
    )

    def _atlas_one(i: int) -> np.ndarray:
        return detect_vesicle_blobs_atlas(densities[i], roots[i], **atlas_kw)

    for comp in _tile_grid_components(rows, cols):
        if len(comp) == 1:
            out[comp[0]] = _atlas_one(comp[0])
            continue

        r_comp = [int(rows[i]) for i in comp]
        c_comp = [int(cols[i]) for i in comp]
        rmin, rmax = min(r_comp), max(r_comp)
        cmin, cmax = min(c_comp), max(c_comp)
        n_r = rmax - rmin + 1
        n_c = cmax - cmin + 1
        mosaic_h = n_r * ts
        mosaic_w = n_c * ts
        fill_ratio = float(len(comp)) / float(max(n_r * n_c, 1))
        # Bbox hueco o demasiado grande → ATLAS por tile (sin congelar Frangi)
        if mosaic_h * mosaic_w > int(max_mosaic_px) or fill_ratio < 0.35:
            for i in comp:
                out[i] = _atlas_one(i)
            continue

        mosa_d = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
        mosa_r = np.zeros((mosaic_h, mosaic_w), dtype=bool)
        for i in comp:
            dens = densities[i]
            h, w = dens.shape[:2]
            y0 = (int(rows[i]) - rmin) * ts
            x0 = (int(cols[i]) - cmin) * ts
            hh, ww = min(h, ts), min(w, ts)
            mosa_d[y0 : y0 + hh, x0 : x0 + ww] = dens[:hh, :ww].astype(np.float32)
            mosa_r[y0 : y0 + hh, x0 : x0 + ww] = roots[i].astype(bool)[:hh, :ww]

        if not mosa_r.any():
            continue

        mosa_mask = detect_vesicle_blobs_atlas(mosa_d, mosa_r, **atlas_kw)
        for i in comp:
            h, w = densities[i].shape[:2]
            y0 = (int(rows[i]) - rmin) * ts
            x0 = (int(cols[i]) - cmin) * ts
            hh, ww = min(h, ts), min(w, ts)
            tile_m = np.zeros((h, w), dtype=bool)
            tile_m[:hh, :ww] = mosa_mask[y0 : y0 + hh, x0 : x0 + ww]
            out[i] = tile_m
    return out
