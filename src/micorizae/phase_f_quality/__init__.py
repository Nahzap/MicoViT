"""Fase F — Control de calidad automático.

Criterios de aceptación de pseudo-etiqueta:
    1) max(p) >= tau_hi
    2) entropía < umbral
    3) consistencia TTA
    4) acuerdo inter-rama (JS divergence < umbral)
    5) plausibilidad morfológica

Salidas:
    - manifests/pseudo_labels_accepted.parquet
    - manifests/pseudo_labels_rejected.parquet

Implementación: pendiente (E4).
"""
