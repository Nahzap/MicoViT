"""V gigantes multi-tile — evidencia de tinción, no discos Hough sintéticos.

v4g (esta versión): los FP del preview eran círculos naranjas perfectos sobre tejido
sin vesícula. Causa: Hough propone un círculo y el raster **pintaba el disco geométrico**.

Regla v4g:
1. Hough/Kasa solo proponen (cy,cx,r) — nunca son la máscara final
2. Máscara = body de densidad (densa) o anillo+núcleo pálido real (pálida)
3. Rechazo duro si no hay contraste de tinción (tejido vacío / paredes celulares)

Fundamento: vesículas AMF esféricas/elipsoidales con pared/tinción propia
(McGonigle; Biermann & Linderman) — no arcos de textura de corteza.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from micorizae.morph_core.stain import stain_maps

TECHNIQUE = "giant_multitile_stain_body"
TECHNIQUE_VERSION = 6  # v4g: no synthetic Hough disk fill


@dataclass
class _ArcHyp:
    tile_i: int
    pts_local: np.ndarray  # (N,2) y,x
    cy_g: float
    cx_g: float
    r: float
    rmse: float
    cov: float


def _fit_circle_kasa(yx: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    pts = np.asarray(yx, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 16:
        return None
    y = pts[:, 0]
    x = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if r2 <= 1.0:
        return None
    r = float(np.sqrt(r2))
    if not np.isfinite(r) or r < 12.0:
        return None
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - r) ** 2)))
    return cy, cx, r, rmse


def _angular_coverage(yx: np.ndarray, cy: float, cx: float, n_bins: int = 36) -> float:
    pts = np.asarray(yx, dtype=np.float64)
    if pts.shape[0] < 8:
        return 0.0
    ang = np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)
    bins = np.floor((ang + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int32) % n_bins
    return float(len(np.unique(bins))) / float(n_bins)


def _mask_circularity(mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    if int(m.sum()) < 20:
        return 0.0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _mask_aspect(mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    if int(m.sum()) < 20:
        return 0.0
    ys, xs = np.where(m > 0)
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    return float(min(h, w) / max(h, w))


def _mask_touches_border(mask: np.ndarray, margin: int = 1) -> bool:
    m = mask.astype(bool)
    if not m.any():
        return False
    return bool(
        m[:margin, :].any()
        or m[-margin:, :].any()
        or m[:, :margin].any()
        or m[:, -margin:].any()
    )


def _to_global(pts_local: np.ndarray, row: int, col: int, tile_size: int) -> np.ndarray:
    pts = np.asarray(pts_local, dtype=np.float64)
    gy = pts[:, 0] + float(row) * float(tile_size)
    gx = pts[:, 1] + float(col) * float(tile_size)
    return np.column_stack([gy, gx])


def _edge_map(density: np.ndarray, root: np.ndarray, *, edge_pctl: float = 88.0) -> np.ndarray:
    dens = np.clip(density.astype(np.float32), 0.0, 1.0)
    root_b = root.astype(bool)
    if not root_b.any():
        return np.zeros_like(dens, dtype=np.uint8)
    d8 = (dens * 255.0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blur = cv2.GaussianBlur(d8, (0, 0), 1.4)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, k)
    thr = float(np.percentile(grad[root_b], edge_pctl))
    edge = ((grad >= max(thr, 8.0)) & root_b).astype(np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k, iterations=1)
    return edge


def _ring_support_points(
    density: np.ndarray,
    root: np.ndarray,
    cy: float,
    cx: float,
    r: float,
    *,
    edge_pctl: float = 82.0,
    tol_frac: float = 0.14,
) -> np.ndarray:
    """Puntos de gradiente locales cerca del anillo (y,x local)."""
    edge = _edge_map(density, root, edge_pctl=edge_pctl)
    ys, xs = np.where(edge > 0)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    dist = np.sqrt((ys.astype(np.float64) - cy) ** 2 + (xs.astype(np.float64) - cx) ** 2)
    keep = np.abs(dist - r) <= max(3.0, float(r) * float(tol_frac))
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([ys[keep], xs[keep]]).astype(np.float32)


def _fill_binary_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Rellena huecos interiores (vesícula moteada → óvalo sólido)."""
    m = (mask_u8 > 0).astype(np.uint8)
    if not m.any():
        return m
    h, w = m.shape
    ff = m.copy()
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 2)
    holes = ff == 0
    out = m.copy()
    out[holes] = 1
    return out


def _dense_sphere_body(
    dens: np.ndarray,
    root: np.ndarray,
    cy: float,
    cx: float,
    r: float,
) -> np.ndarray:
    """Máscara densa sólida dentro del disco: umbral suave + close + fill.

    Evita circularidad ~0 en bodies moteados (ABS710) y el fill de colonia entera.
    """
    h, w = dens.shape
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    root_b = root.astype(bool)
    disk = (dist2 <= (r * 1.02) ** 2) & root_b
    if int(disk.sum()) < 40:
        return np.zeros((h, w), dtype=bool)
    dens_f = dens.astype(np.float32)
    outside = root_b & (dist2 >= (r * 1.08) ** 2) & (dist2 <= (r * 1.55) ** 2)
    if int(outside.sum()) < 20:
        outside = root_b & ~disk
    out_med = float(np.median(dens_f[outside])) if outside.any() else float(np.median(dens_f[disk]))
    thr = max(out_med * 1.05, float(np.percentile(dens_f[disk], 35.0)))
    seed = (disk & (dens_f >= thr)).astype(np.uint8)
    ksz = max(7, int(round(min(15.0, r * 0.22))) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k, iterations=3)
    seed = _fill_binary_holes(seed)
    seed = seed.astype(bool) & disk
    # quedarse con el CC más grande que toque el núcleo
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    if nlab <= 1:
        return np.zeros((h, w), dtype=bool)
    core = (dist2 <= (r * 0.45) ** 2) & root_b
    best_i = 0
    best_area = 0
    for i in range(1, nlab):
        comp = lab == i
        if core.any() and not (comp & core).any():
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_i = i
    if best_i == 0:
        best_i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        best_area = int(stats[best_i, cv2.CC_STAT_AREA])
    if best_area < 60:
        return np.zeros((h, w), dtype=bool)
    return lab == best_i


def _extract_dense_blob_hypotheses(
    density: np.ndarray,
    root: np.ndarray,
    *,
    tile_i: int,
    row: int,
    col: int,
    tile_size: int,
    r_min: float,
    r_max: float,
) -> list[_ArcHyp]:
    """Vesículas densas: Hough en density + body relleno elipsoidal (ABS710/AFF756).

    No usa el disco Hough completo (pinta colonia). El body denso relleno ES la máscara.
    """
    dens = np.clip(density.astype(np.float32), 0.0, 1.0)
    root_b = root.astype(bool)
    if not root_b.any():
        return []
    h, w = dens.shape
    y0g = float(row) * float(tile_size)
    x0g = float(col) * float(tile_size)
    hyps: list[_ArcHyp] = []
    seen: list[tuple[float, float, float]] = []

    d8 = (dens * 255.0).astype(np.uint8)
    blur = cv2.GaussianBlur(d8, (0, 0), 1.4)
    blur = np.where(root_b, blur, 0).astype(np.uint8)
    candidates: list[tuple[float, float, float]] = []
    for param2 in (26, 20):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(22.0, float(tile_size) * 0.26),
            param1=40,
            param2=param2,
            minRadius=max(12, int(r_min * 0.85)),
            maxRadius=max(int(r_min) + 1, int(min(r_max, tile_size * 0.95))),
        )
        if circles is None:
            continue
        for x, y, rr in circles[0]:
            candidates.append((float(y), float(x), float(rr)))

    # fallback: CCs compactos tras open (sintéticos / AFF aisladas)
    thr = float(np.percentile(dens[root_b], 72.0))
    thr = max(thr, float(np.median(dens[root_b])) * 1.12)
    blob = ((dens >= thr) & root_b).astype(np.uint8)
    for ksz in (9, 15):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        op = cv2.morphologyEx(blob, cv2.MORPH_OPEN, k, iterations=1)
        op = cv2.morphologyEx(op, cv2.MORPH_CLOSE, k, iterations=2)
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
        area_root = float(root_b.sum())
        for i in range(1, nlab):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < max(120.0, 0.008 * area_root) or area > 0.55 * area_root:
                continue
            comp = (lab == i).astype(np.uint8)
            circ = _mask_circularity(comp)
            asp = _mask_aspect(comp)
            if circ < 0.42 or asp < 0.48:
                continue
            ys, xs = np.where(comp > 0)
            candidates.append((float(ys.mean()), float(xs.mean()), float(np.sqrt(area / np.pi))))

    for cy, cx, r in candidates:
        if r < r_min * 0.80 or r > r_max:
            continue
        if np.pi * r * r > 0.50 * h * w:
            continue
        if any(np.hypot(cy - sy, cx - sx) < 0.40 * max(r, sr) for sy, sx, sr in seen):
            continue
        body = _dense_sphere_body(dens, root_b, cy, cx, r)
        if int(body.sum()) < 80:
            continue
        circ = _mask_circularity(body.astype(np.uint8))
        asp = _mask_aspect(body.astype(np.uint8))
        if circ < 0.40 or asp < 0.48:
            continue
        cover = float(body.sum()) / float(h * w)
        if cover > 0.42:
            continue
        yy, xx = np.ogrid[:h, :w]
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2 <= (r * 1.02) ** 2) & root_b
        disk_frac = float(body.sum()) / float(max(1, int(disk.sum())))
        if disk_frac < 0.28 or disk_frac > 0.98:
            continue
        ys, xs = np.where(body)
        cy_b = float(ys.mean())
        cx_b = float(xs.mean())
        r_b = float(np.sqrt(float(body.sum()) / np.pi))
        # prefer body-equivalent radius (evita discos Hough demasiado grandes)
        rr = min(r, max(r_b, r * 0.75))
        cnts, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        pts = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        yx_l = np.column_stack([pts[:, 1], pts[:, 0]]).astype(np.float64)
        if yx_l.shape[0] > 400:
            idx = np.linspace(0, yx_l.shape[0] - 1, 400).astype(np.int32)
            yx_fit = yx_l[idx]
        else:
            yx_fit = yx_l
        yx_g = _to_global(yx_fit, row, col, tile_size)
        cy_g, cx_g = y0g + cy_b, x0g + cx_b
        fit = _fit_circle_kasa(yx_g)
        rmse = 0.05 * rr
        if fit is not None:
            fcy, fcx, fr, frmse = fit
            if r_min * 0.75 <= fr <= r_max and frmse / max(fr, 1.0) <= 0.16:
                cy_g, cx_g, rr, rmse = fcy, fcx, fr, frmse
        # contraste suave: núcleo vs anillo exterior (colonia puede ser densa)
        dist2 = (yy - cy_b) ** 2 + (xx - cx_b) ** 2
        core = (dist2 <= (rr * 0.50) ** 2) & root_b
        outside = root_b & (dist2 >= (rr * 1.08) ** 2) & (dist2 <= (rr * 1.50) ** 2)
        if int(core.sum()) < 20 or int(outside.sum()) < 20:
            continue
        if float(dens[core].mean()) < float(dens[outside].mean()) * 1.06:
            continue
        cov = _angular_coverage(yx_g, cy_g, cx_g)
        seen.append((cy_b, cx_b, rr))
        hyps.append(
            _ArcHyp(
                tile_i=tile_i,
                pts_local=yx_l.astype(np.float32),
                cy_g=float(cy_g),
                cx_g=float(cx_g),
                r=float(rr),
                rmse=float(rmse),
                cov=max(float(cov), 0.40),
            )
        )
    return hyps


def _extract_arc_hypotheses_in_tile(
    density: np.ndarray,
    root: np.ndarray,
    *,
    tile_i: int,
    row: int,
    col: int,
    tile_size: int,
    r_min: float,
    r_max: float,
    rmse_frac: float,
    min_arc_px: int = 28,
    min_partial_cov: float = 0.12,
) -> list[_ArcHyp]:
    """Hough (pálidas / arcos) + blobs densos redondos (AFF756/ABS710)."""
    _ = min_arc_px
    dens = np.clip(density.astype(np.float32), 0.0, 1.0)
    root_b = root.astype(bool)
    if not root_b.any():
        return []
    d8 = (dens * 255.0).astype(np.uint8)
    blur = cv2.GaussianBlur(d8, (0, 0), 1.6)
    gimg = cv2.morphologyEx(
        blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    gimg = np.where(root_b, gimg, 0).astype(np.uint8)

    candidates: list[tuple[float, float, float]] = []
    for param2 in (28, 22):
        circles = cv2.HoughCircles(
            gimg,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(24.0, float(tile_size) * 0.28),
            param1=40,
            param2=param2,
            minRadius=max(12, int(r_min)),
            maxRadius=max(int(r_min) + 1, int(min(r_max, tile_size * 1.2))),
        )
        if circles is None:
            continue
        for x, y, rr in circles[0]:
            candidates.append((float(y), float(x), float(rr)))

    hyps: list[_ArcHyp] = []
    seen: list[tuple[float, float, float]] = []
    y0g = float(row) * float(tile_size)
    x0g = float(col) * float(tile_size)

    for cy, cx, r in candidates:
        if r < r_min or r > r_max:
            continue
        if any(np.hypot(cy - sy, cx - sx) < 0.40 * max(r, sr) for sy, sx, sr in seen):
            continue
        # pálidas multi-tile: r puede superar el área del tile; no filtrar por πr²
        ring_pts = _ring_support_points(dens, root_b, cy, cx, r)
        if ring_pts.shape[0] < 18:
            continue
        yx_g = _to_global(ring_pts, row, col, tile_size)
        fit = _fit_circle_kasa(yx_g)
        if fit is None:
            cy_g, cx_g, rr, rmse = y0g + cy, x0g + cx, r, 0.08 * r
        else:
            cy_g, cx_g, rr, rmse = fit
            if rr < r_min or rr > r_max:
                continue
            if rmse / max(rr, 1.0) > float(rmse_frac):
                continue
        cov = _angular_coverage(yx_g, cy_g, cx_g)
        if cov < float(min_partial_cov):
            continue
        cy_l, cx_l = cy_g - y0g, cx_g - x0g
        geom_ok = _sphere_geometry_ok(dens, root_b, cy_l, cx_l, rr)
        h, w = dens.shape
        center_inside = (0.0 <= cy_l < h) and (0.0 <= cx_l < w)
        if not geom_ok:
            if center_inside:
                continue
            if cov < max(float(min_partial_cov), 0.14):
                continue
        seen.append((cy_l, cx_l, rr))
        hyps.append(
            _ArcHyp(
                tile_i=tile_i,
                pts_local=ring_pts,
                cy_g=cy_g,
                cx_g=cx_g,
                r=float(rr),
                rmse=float(rmse),
                cov=float(cov),
            )
        )

    # semillas densas (blob redondo) — complementan Hough
    for hdense in _extract_dense_blob_hypotheses(
        dens, root_b, tile_i=tile_i, row=row, col=col, tile_size=tile_size, r_min=r_min, r_max=r_max
    ):
        cy_l = hdense.cy_g - y0g
        cx_l = hdense.cx_g - x0g
        if any(np.hypot(cy_l - sy, cx_l - sx) < 0.40 * max(hdense.r, sr) for sy, sx, sr in seen):
            continue
        seen.append((cy_l, cx_l, hdense.r))
        hyps.append(hdense)
    return hyps


def _hypotheses_compatible(a: _ArcHyp, b: _ArcHyp, *, center_tol: float, r_tol: float) -> bool:
    dc = float(np.hypot(a.cy_g - b.cy_g, a.cx_g - b.cx_g))
    dr = abs(a.r - b.r)
    return dc <= center_tol and dr <= r_tol


def _sphere_geometry_ok(
    dens: np.ndarray,
    root: np.ndarray,
    cy: float,
    cx: float,
    r: float,
    *,
    ring_width_frac: float = 0.12,
) -> bool:
    """V real: pared/anillo + interior pálido O blob denso contrastado.

    Rechaza discos Hough sobre corteza vacía / textura (FP preview).
    """
    h, w = dens.shape
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    r2 = r * r
    root_b = root.astype(bool)
    disk = (dist2 <= r2 * 0.92) & root_b
    if int(disk.sum()) < 40:
        return False
    rw = max(2.0, float(r) * float(ring_width_frac))
    ring = (dist2 >= (r - rw) ** 2) & (dist2 <= (r + rw) ** 2) & root_b
    core = (dist2 <= (r * 0.55) ** 2) & root_b
    if int(ring.sum()) < 16 or int(core.sum()) < 16:
        return False
    dens_f = dens.astype(np.float32)
    d8 = (np.clip(dens_f, 0, 1) * 255).astype(np.uint8)
    grad = cv2.morphologyEx(
        d8, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    g = grad.astype(np.float32) / 255.0
    ring_g = float(g[ring].mean())
    core_g = float(g[core].mean()) if core.any() else 0.0
    core_d = float(dens_f[core].mean())
    ring_d = float(dens_f[ring].mean())
    outside = root_b & (dist2 >= (r * 1.05) ** 2) & (dist2 <= (r * 1.45) ** 2)
    if int(outside.sum()) < 20:
        outside = root_b & ~disk
    out_d = float(dens_f[outside].mean()) if outside.any() else core_d

    # (A) pálida: pared oscura clara + núcleo más claro que pared y entorno
    pale_ok = (
        (ring_d >= core_d * 1.18)
        and (ring_d >= out_d * 1.12)
        and (core_d <= out_d * 0.98)
        and (ring_d - core_d >= 0.07)
        and (ring_g >= max(0.035, 1.35 * core_g))
    )
    # (B) densa: núcleo claramente más teñido que entorno
    dense_ok = (
        (core_d >= out_d * 1.18)
        and (core_d >= 0.45)
        and (core_d >= ring_d * 0.88)
        and (core_d - out_d >= 0.08)
    )
    if not (pale_ok or dense_ok):
        return False

    if pale_ok:
        body = disk & (dens_f <= max(ring_d * 0.88, out_d * 0.95))
        # exigir anillo oscuro real en la máscara de evidencia
        dark_ring = ring & (dens_f >= max(ring_d * 0.85, core_d * 1.10))
        if int(dark_ring.sum()) < max(24, int(0.25 * float(ring.sum()))):
            return False
        body_u8 = (body | dark_ring).astype(np.uint8)
    else:
        body = _dense_sphere_body(dens_f, root_b, cy, cx, r)
        body_u8 = body.astype(np.uint8)
    if int(body_u8.sum()) < 40:
        return False
    body_frac = float(body_u8.sum()) / float(max(1, disk.sum()))
    if body_frac < 0.18:
        return False

    # evidencia vs fondo: la máscara no puede ser “igual” al resto del tile
    if outside.any():
        in_d = float(dens_f[body_u8.astype(bool)].mean())
        if abs(in_d - out_d) < 0.06:
            return False

    center_outside = not ((0.0 <= cy < h) and (0.0 <= cx < w))
    asp = _mask_aspect(body_u8)
    circ = _mask_circularity(body_u8)
    tile_cover = float(body_u8.sum()) / float(h * w)
    if tile_cover > 0.40 and dense_ok:
        return False
    if dense_ok and circ < 0.42:
        return False

    if center_outside or _mask_touches_border(disk.astype(np.uint8), margin=2):
        if asp < 0.45 or body_frac < 0.22:
            return False
    else:
        min_circ = 0.45 if dense_ok else 0.42
        min_asp = 0.50 if dense_ok else 0.55
        if asp < min_asp or circ < min_circ:
            return False
        if pale_ok and float(dens_f[core].std()) > 0.12:
            return False
    return True


def _ellipse_fill_from_mask(mask: np.ndarray, *, max_grow: float = 1.12) -> np.ndarray:
    """Suaviza contorno del blob real; no inventa elipse sin soporte de tinción."""
    m = mask.astype(np.uint8)
    if int(m.sum()) < 40:
        return m.astype(bool)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return m.astype(bool)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return m.astype(bool)
    try:
        ell = cv2.fitEllipse(c)
    except cv2.error:
        return m.astype(bool)
    (_cx, _cy), (ma, mb), _ang = ell
    area0 = float(m.sum())
    area_ell = float(np.pi * (ma * 0.5) * (mb * 0.5))
    # poco crecimiento: la máscara debe seguir el blob, no un disco Hough
    if area_ell > max_grow * area0 or area_ell < 0.70 * area0:
        return m.astype(bool)
    out = np.zeros_like(m)
    cv2.ellipse(out, ell, 1, thickness=-1)
    # solo píxeles que tocan el blob original (dilatado leve)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near = cv2.dilate(m, k, iterations=2) > 0
    return (out.astype(bool) & near)


def _refine_circle_to_dense_centroid(
    densities: list[np.ndarray],
    roots: list[np.ndarray],
    rows: list[int],
    cols: list[int],
    tile_size: int,
    cy: float,
    cx: float,
    r: float,
) -> tuple[float, float, float]:
    """Recentra (cy,cx,r) al centroide del body denso compacto.

    Evita ABS710: Hough/Kasa se ancla a arcos en el *borde del tile*
    (máscara naranja fuera del centroide). No expandir a colonia entera.
    """
    ts = float(tile_size)
    best: Optional[tuple[float, float, float, float]] = None  # score, cy, cx, r
    for ti, dens in enumerate(densities):
        h, w = dens.shape
        y0 = float(rows[ti]) * ts
        x0 = float(cols[ti]) * ts
        cy_l = cy - y0
        cx_l = cx - x0
        if cy + r < y0 or cy - r > y0 + h or cx + r < x0 or cx - r > x0 + w:
            continue
        root_b = roots[ti].astype(bool)
        dens_f = dens.astype(np.float32)
        # radios acotados: no explorar discos de colonia
        for scale in (0.85, 1.0, 1.12, 1.25):
            rr = float(r) * scale
            if np.pi * rr * rr > 0.38 * h * w:
                continue
            body = _dense_sphere_body(dens_f, root_b, cy_l, cx_l, rr)
            area = int(body.sum())
            if area < 100:
                continue
            cover = float(area) / float(h * w)
            # óvalo V típico; rechazar colonia que llena el tile
            if cover < 0.035:
                continue
            touches = int(body[0, :].any()) + int(body[-1, :].any()) + int(body[:, 0].any()) + int(
                body[:, -1].any()
            )
            if touches >= 3:
                continue
            ys, xs = np.where(body)
            cy_b = float(ys.mean())
            cx_b = float(xs.mean())
            r_b = float(np.sqrt(float(area) / np.pi))
            if r_b > 0.48 * float(min(h, w)):
                continue
            yy, xx = np.ogrid[:h, :w]
            dist2 = (yy - cy_b) ** 2 + (xx - cx_b) ** 2
            core = (dist2 <= (r_b * 0.50) ** 2) & root_b
            outside = root_b & (dist2 >= (r_b * 1.10) ** 2) & (dist2 <= (r_b * 1.55) ** 2)
            if int(core.sum()) < 20 or int(outside.sum()) < 20:
                continue
            core_d = float(dens_f[core].mean())
            out_d = float(dens_f[outside].mean())
            contrast = core_d / max(out_d, 1e-6)
            if contrast < 1.06:
                continue
            # cover holgado solo si contraste alto y no toca bordes (sintético / V aislada)
            max_cover = 0.38 if (contrast >= 1.20 and touches <= 1) else 0.28
            if cover > max_cover:
                continue
            circ = _mask_circularity(body.astype(np.uint8))
            asp = _mask_aspect(body.astype(np.uint8))
            if circ < 0.45 or asp < 0.48:
                continue
            margin = 0.10 * float(min(h, w))
            edge_pen = 1.0
            if cy_b < margin or cy_b > h - margin or cx_b < margin or cx_b > w - margin:
                edge_pen = 0.45
            size_fit = 1.0 - abs(cover - 0.12) / 0.20
            score = circ * asp * max(0.15, size_fit) * edge_pen * min(contrast, 2.0)
            cand = (score, y0 + cy_b, x0 + cx_b, float(r_b * 1.05))
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        return cy, cx, r
    return float(best[1]), float(best[2]), float(best[3])


def _rasterize_sphere_local(
    h: int,
    w: int,
    cy_g: float,
    cx_g: float,
    r: float,
    row: int,
    col: int,
    tile_size: int,
    dens: np.ndarray,
    root: np.ndarray,
) -> np.ndarray:
    """Rasteriza V desde tinción real — nunca un disco Hough sintético vacío."""
    y0 = float(row) * float(tile_size)
    x0 = float(col) * float(tile_size)
    cy = cy_g - y0
    cx = cx_g - x0
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    root_b = root.astype(bool)
    disk = (dist2 <= (r * 0.97) ** 2) & root_b
    if not disk.any():
        return np.zeros((h, w), dtype=bool)

    dens_f = dens.astype(np.float32)
    rw = max(2.0, r * 0.11)
    ring = (dist2 >= (r - rw) ** 2) & (dist2 <= (r + rw * 0.55) ** 2) & root_b
    outside = root_b & (dist2 >= (r * 1.05) ** 2)
    out_med = (
        float(np.median(dens_f[outside])) if outside.any() else float(np.median(dens_f[root_b]))
    )
    core = (dist2 <= (r * 0.55) ** 2) & root_b
    core_d = float(dens_f[core].mean()) if core.any() else out_med
    dense = (core_d >= out_med * 1.18) and (core_d - out_med >= 0.08)

    if dense:
        r_cap = 0.40 * float(min(h, w))
        candidates_r = []
        for scale in (0.75, 0.85, 1.0):
            rr = min(float(r) * scale, r_cap)
            if rr >= 20.0:
                candidates_r.append(rr)
        tried = []
        for rr in sorted(set(round(x, 1) for x in candidates_r)):
            body = _dense_sphere_body(dens_f, root_b, cy, cx, float(rr))
            if int(body.sum()) < 80:
                continue
            # contraste del body vs exterior
            if outside.any() and float(dens_f[body].mean()) < out_med * 1.15:
                continue
            body = _ellipse_fill_from_mask(body, max_grow=1.10)
            circ = _mask_circularity(body.astype(np.uint8))
            asp = _mask_aspect(body.astype(np.uint8))
            cover = float(body.sum()) / float(h * w)
            touches = (
                int(body[0, :].any())
                + int(body[-1, :].any())
                + int(body[:, 0].any())
                + int(body[:, -1].any())
            )
            max_cover = 0.28
            if touches <= 1 and core_d >= out_med * 1.30:
                max_cover = 0.36
            if circ < 0.45 or asp < 0.50 or cover > max_cover or touches >= 3:
                continue
            tried.append((circ * asp * (1.0 - abs(cover - 0.14)), body))
        if not tried:
            return np.zeros((h, w), dtype=bool)
        tried.sort(key=lambda t: t[0], reverse=True)
        return tried[0][1].astype(bool)

    # pálida: NUNCA disco geométrico completo — solo anillo oscuro + núcleo pálido
    if not _sphere_geometry_ok(dens, root_b, cy, cx, r):
        return np.zeros((h, w), dtype=bool)
    if int(ring.sum()) < 20:
        return np.zeros((h, w), dtype=bool)
    ring_med = float(np.median(dens_f[ring])) if ring.any() else 1.0
    if ring_med < out_med * 1.10 or ring_med - core_d < 0.07:
        return np.zeros((h, w), dtype=bool)
    dark_ring = ring & (dens_f >= max(ring_med * 0.80, core_d * 1.12))
    pale_core = disk & (dens_f <= min(ring_med * 0.88, out_med * 0.95))
    mask = pale_core | dark_ring
    if int(dark_ring.sum()) < 24 or int(pale_core.sum()) < 30:
        return np.zeros((h, w), dtype=bool)
    if int(mask.sum()) < max(40, int(0.12 * float(disk.sum()))):
        return np.zeros((h, w), dtype=bool)
    if not _mask_touches_border(mask, margin=2):
        if _mask_circularity(mask) < 0.42 or _mask_aspect(mask) < 0.50:
            return np.zeros((h, w), dtype=bool)
    # veto tejido vacío: mask dens ~ outside
    if outside.any() and abs(float(dens_f[mask].mean()) - out_med) < 0.05:
        return np.zeros((h, w), dtype=bool)
    return mask.astype(bool)


def _collect_ring_points(
    densities: list[np.ndarray],
    roots: list[np.ndarray],
    rows: list[int],
    cols: list[int],
    tile_size: int,
    cy: float,
    cx: float,
    r: float,
    *,
    ring_tol_frac: float = 0.14,
) -> np.ndarray:
    """Puntos de gradiente cerca del anillo teórico en todos los tiles que intersectan."""
    pts: list[np.ndarray] = []
    tol = max(3.0, float(r) * float(ring_tol_frac))
    for ti, dens in enumerate(densities):
        h, w = dens.shape
        y0 = float(rows[ti]) * float(tile_size)
        x0 = float(cols[ti]) * float(tile_size)
        if cy + r + tol < y0 or cy - r - tol > y0 + h or cx + r + tol < x0 or cx - r - tol > x0 + w:
            continue
        edge = _edge_map(dens, roots[ti], edge_pctl=86.0)
        ys, xs = np.where(edge > 0)
        if ys.size == 0:
            continue
        gy = ys.astype(np.float64) + y0
        gx = xs.astype(np.float64) + x0
        dist = np.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
        keep = np.abs(dist - r) <= tol
        if not np.any(keep):
            continue
        pts.append(np.column_stack([gy[keep], gx[keep]]))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    out = np.vstack(pts)
    if out.shape[0] > 900:
        idx = np.linspace(0, out.shape[0] - 1, 900).astype(np.int32)
        out = out[idx]
    return out


def detect_giant_vesicle_masks_for_tiles(
    tiles_hwc: list[np.ndarray],
    rows: list[int],
    cols: list[int],
    tile_size: int,
    *,
    border_width: int = 10,
    r_min_frac: float = 0.22,
    r_max_tiles: float = 2.2,
    rmse_frac: float = 0.10,
    min_angular_cov: float = 0.28,
    min_tiles_span: int = 1,
    max_instances: int = 24,
) -> list[np.ndarray]:
    """DEPRECATED: redirige a ATLAS multi-tile (sin Hough/disco sintético).

    Conserva la firma para tests/callers legacy. Ignora kwargs Hough.
    """
    _ = (
        border_width,
        r_min_frac,
        r_max_tiles,
        rmse_frac,
        min_angular_cov,
        min_tiles_span,
        max_instances,
    )
    from .v_vesicle import detect_vesicle_masks_for_tiles

    return detect_vesicle_masks_for_tiles(tiles_hwc, rows, cols, tile_size)


def apply_giant_v_to_label_and_priors(
    label: np.ndarray,
    prior_e: Optional[np.ndarray],
    prior_v: Optional[np.ndarray],
    giant_native: np.ndarray,
    *,
    input_size: int,
    v_idx: int = 2,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """DEPRECATED alias → ``apply_v_masks_to_label_and_priors``."""
    from .v_vesicle import apply_v_masks_to_label_and_priors

    return apply_v_masks_to_label_and_priors(
        label, prior_e, prior_v, giant_native, input_size=input_size, v_idx=v_idx
    )


# --- compat exports usados por tests / preview ---
def _extract_border_arcs(density, root, *, border_width=10, edge_pctl=82.0, min_arc_px=22):
    """Compat: arcos solo en franja de borde (tests legacy / debug)."""
    dens = np.clip(density.astype(np.float32), 0.0, 1.0)
    root_b = root.astype(bool)
    if not root_b.any():
        return []
    edge = _edge_map(dens, root_b, edge_pctl=edge_pctl)
    h, w = dens.shape
    bw = max(2, int(border_width))
    borders = {
        "top": np.zeros((h, w), dtype=bool),
        "bottom": np.zeros((h, w), dtype=bool),
        "left": np.zeros((h, w), dtype=bool),
        "right": np.zeros((h, w), dtype=bool),
    }
    borders["top"][:bw, :] = True
    borders["bottom"][-bw:, :] = True
    borders["left"][:, :bw] = True
    borders["right"][:, -bw:] = True
    arcs = []
    for name, bmask in borders.items():
        strip = (edge.astype(bool) & bmask).astype(np.uint8)
        if int(strip.sum()) < min_arc_px:
            continue
        nlab, lab = cv2.connectedComponents(strip, connectivity=8)
        for i in range(1, nlab):
            ys, xs = np.where(lab == i)
            if ys.size < min_arc_px:
                continue
            touched = {name}
            for other, om in borders.items():
                if other != name and om[ys, xs].any():
                    touched.add(other)
            arcs.append((np.column_stack([ys, xs]).astype(np.float32), frozenset(touched)))
    return arcs


def _extract_closed_sphere_arcs(density, root, **kwargs):
    """Compat: reusa hipótesis internas filtrando cobertura alta."""
    _ = kwargs
    hyps = _extract_arc_hypotheses_in_tile(
        density,
        root,
        tile_i=0,
        row=0,
        col=0,
        tile_size=max(density.shape),
        r_min=28.0,
        r_max=max(density.shape) * 1.2,
        rmse_frac=0.12,
        min_partial_cov=0.28,
    )
    return [h.pts_local for h in hyps if h.cov >= 0.28]
