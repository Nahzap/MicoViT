"""Fase C — Construcción de vistas por tile.

Por cada tile genera 3 vistas espacialmente alineadas:
    - RGB normalizada (input para DINOv2)
    - Segmentación (input pre-procesado para U2Net)
    - Frecuencia (FFT log-magnitud, opcional wavelet Haar)

Las vistas se computan **on-demand** (no se materializan al disco para los
174 671 tiles del corpus). Sólo el módulo `preview` guarda PNGs para auditoría.
"""

from .transforms import (
    rgb_view,
    seg_view,
    freq_view,
    freq_reconstruct,
    freq_reconstruct_from_logmag_phase,
    freq_reconstruct_magnitude_only,
    freq_reconstruct_phase_only,
    build_views,
    ViewBundle,
    FreqRepresentation,
)

__all__ = [
    "rgb_view",
    "seg_view",
    "freq_view",
    "freq_reconstruct",
    "freq_reconstruct_from_logmag_phase",
    "freq_reconstruct_magnitude_only",
    "freq_reconstruct_phase_only",
    "build_views",
    "ViewBundle",
    "FreqRepresentation",
]
