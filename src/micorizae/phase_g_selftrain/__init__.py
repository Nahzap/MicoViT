"""Fase G — Self-training por rondas.

Loop teacher-student:
    1) inferir,
    2) filtrar etiquetas aceptadas (Fase F),
    3) reentrenar student,
    4) actualizar teacher (EMA),
    5) reevaluar en conjunto congelado.

Condición de continuidad: mejora real en métricas objetivo, no solo volumen
de pseudo-etiquetas. Sincroniza la Capa L10 con cambios entre rondas.

Implementación: pendiente (E6).
"""
