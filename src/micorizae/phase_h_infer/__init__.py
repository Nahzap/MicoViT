"""Fase H — Inferencia final y cuantificación.

Pipeline:
    1) Stage1 sobre toda la imagen
    2) Stage2 sólo sobre tiles con M+ confiable
    3) Reensamble espacial (capas L2..L9)
    4) Cálculo de colonización y distribuciones

Salidas por `run_id`:
    outputs/<run_id>/maps/...
    outputs/<run_id>/tables/...
    outputs/<run_id>/reports/run_report.md

Implementación: pendiente (E7).
"""
