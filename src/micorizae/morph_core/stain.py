"""Stain / color-deconvolution maps shared by morph detectors (I↔E)."""

from __future__ import annotations

import cv2
import numpy as np

from .params import WeakSegParams


def stain_maps(tile_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Separa señal de tinción (azul de tripano) del tejido y el fondo.

    Fundamento: en tinción azul, el colorante fúngico absorbe fuertemente la luz
    roja (densidad óptica de rojo alta) y es más azul que el tejido pálido
    (Ruifrok & Johnston 2001, color deconvolution). El fondo es blanco (R,G,B altos,
    baja saturación).

    Devuelve:
      density  — densidad de tinción 0..1 (OD de rojo, alto = estructura densa)
      blueness — (B-R) normalizado 0..1
      stain    — evidencia combinada density*blueness 0..1
      tissue   — máscara booleana de tejido (no fondo blanco)
    """
    rgb = tile_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    sat = (maxc - minc) / (maxc + 1e-3)
    # Densidad óptica del rojo: -log10((R+1)/256). Alto donde R es bajo (tinción densa).
    od_r = -np.log10((R + 1.0) / 256.0)
    density = np.clip(od_r / np.log10(256.0), 0.0, 1.0)
    blueness = np.clip((B - R) / 255.0, 0.0, 1.0)
    stain = np.clip(density * (0.5 + 0.5 * (blueness > 0.05)), 0.0, 1.0) * (blueness > 0.02)
    return {
        "density": density.astype(np.float32),
        "blueness": blueness.astype(np.float32),
        "stain": stain.astype(np.float32),
        "sat": sat.astype(np.float32),
        "maxc": maxc.astype(np.float32),
    }


def _stain_maps(tile_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Backward-compat alias for ``stain_maps``."""
    return stain_maps(tile_rgb)


def ambiguous_stain_mask(m: dict[str, np.ndarray], root: np.ndarray, p: WeakSegParams) -> np.ndarray:
    """Regiones GRANDES de azul saturado = estructura no resoluble.

    Cuando la tinción satura (density y blueness altas) sobre un área extensa, no
    se distinguen estructuras discretas; forzar V/IH/A/H ahí mete ruido. Se marca
    como ambiguo (→ ignore, fuera de la loss). Una vesícula o hifa real es un
    punto/cresta LOCAL pequeño: la apertura morfológica lo elimina y solo
    sobreviven las regiones saturadas grandes.
    """
    dens = m["density"].astype(np.float32)
    blue = m["blueness"].astype(np.float32)
    sat = ((dens >= p.amb_density) & (blue >= p.amb_blueness) & root).astype(np.float32)
    if sat.sum() < 1:
        return np.zeros_like(root, dtype=bool)
    win = int(p.amb_win)
    frac = cv2.blur(sat, (win, win))  # fracción local saturada (robusto a la forma)
    return (frac >= p.amb_frac) & root


def _ambiguous_stain_mask(
    m: dict[str, np.ndarray], root: np.ndarray, p: WeakSegParams
) -> np.ndarray:
    """Backward-compat alias for ``ambiguous_stain_mask``."""
    return ambiguous_stain_mask(m, root, p)
