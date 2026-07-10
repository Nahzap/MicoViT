#!/usr/bin/env python3
"""Genera informe final Stage2-Pixel con mapas, tablas y métricas."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-run-id", default="20260624_012932_gate_am_train")
    args = parser.parse_args()
    gate_run_id = args.gate_run_id

    rep = ROOT / "outputs" / "reports"
    train_meta = sorted((ROOT / "outputs").glob(f"*stage2_pixel_train*gate{gate_run_id}*/STAGE2_PIXEL_RUN_META.json"))
    train_json = rep / f"stage2_pixel_train__gate{gate_run_id}.json"
    if train_meta:
        train_json.write_text(train_meta[-1].read_text(encoding="utf-8"), encoding="utf-8")
    holdout = rep / f"stage2_pixel_holdout__gate{gate_run_id}.json"
    baseline = ROOT / "outputs" / "stage2_pixel_audit" / f"gate{gate_run_id}" / "STAGE2_PIXEL_BASELINE_AUDIT.json"

    sections = [
        f"# Informe Stage2-Pixel — gate `{gate_run_id}`",
        "",
        f"Generado: {datetime.now().isoformat()}",
        "",
        "## 1. Entrenamiento (20 épocas ViT DINOv2 morfológico + MEViT prior losses)",
        "",
    ]
    if train_json.exists():
        tr = json.loads(train_json.read_text(encoding="utf-8"))
        h = tr.get("history", {})
        sections += [
            f"- Run: `{tr.get('run_id', '')}`",
            f"- Best epoch: **{h.get('best_epoch', '?')}**",
            f"- Best val mIoU: **{h.get('best_val_miou', 0):.4f}**",
            f"- Checkpoint: `{tr.get('checkpoint', '')}`",
            "",
            "### Curvas",
            "",
            "| epoch | train_loss | val_loss | val_miou | val_acc |",
            "|------:|-----------:|---------:|---------:|--------:|",
        ]
        for i, ep in enumerate(h.get("epochs", [])):
            sections.append(
                f"| {ep} | {h['train_loss'][i]:.4f} | {h['val_loss'][i]:.4f} | "
                f"{h['val_miou'][i]:.4f} | {h['val_acc'][i]:.4f} |"
            )
    else:
        sections.append("_Sin meta de entrenamiento._")

    sections += ["", "## 2. Baseline weak (E2-P0)", ""]
    if baseline.exists():
        b = json.loads(baseline.read_text(encoding="utf-8"))
        sections.append(f"- G-PX.1: {b.get('g_px1_colony_cov_gt0', 0):.3f}")
        sections.append(f"- Media % colonizado: {b.get('mean_pct_colonized', 0):.2f}")
        if "g_px2_hybrid_entropy_gt_am_pvalue" in b:
            sections.append(f"- G-PX.2 p-value: {b['g_px2_hybrid_entropy_gt_am_pvalue']:.4f}")

    sections += ["", "## 3. Holdout val/test (E2-P2)", ""]
    val_pq = rep / f"stage2_pixel_val__gate{gate_run_id}.parquet"
    test_pq = rep / f"stage2_pixel_test__gate{gate_run_id}.parquet"
    if val_pq.exists() and test_pq.exists():
        import pandas as pd

        v = pd.read_parquet(val_pq)
        t = pd.read_parquet(test_pq)
        vt = v[v["level"] == "tile"]
        tt = t[t["level"] == "tile"]
        sections += [
            "### Val (tiles M+)",
            "",
            f"- n tiles: {len(vt)}",
            f"- media % colonizado: {vt['pct_colonized'].mean():.2f}",
            f"- media % IH: {vt['pct_IH'].mean():.2f}",
            "",
            "### Test (tiles M+)",
            "",
            f"- n tiles: {len(tt)}",
            f"- media % colonizado: {tt['pct_colonized'].mean():.2f}",
            f"- media % IH: {tt['pct_IH'].mean():.2f}",
            f"- media % A: {tt['pct_A'].mean():.2f}",
            f"- media % V: {tt['pct_V'].mean():.2f}",
            "",
            "### Comparación val vs test (% colonizado)",
            "",
            "| split | mean pct_colonized | std |",
            "|-------|-------------------:|----:|",
            f"| val | {vt['pct_colonized'].mean():.2f} | {vt['pct_colonized'].std():.2f} |",
            f"| test | {tt['pct_colonized'].mean():.2f} | {tt['pct_colonized'].std():.2f} |",
        ]
    if holdout.exists():
        ho = json.loads(holdout.read_text(encoding="utf-8"))
        sections += ["", "### G-PX holdout JSON", "", "```json", json.dumps(ho, indent=2), "```"]

    explain_audit = rep / f"stage2_pixel_explain_audit__gate{gate_run_id}.json"
    sections += ["", "## 4. Explicabilidad MEViT (E2-EX)", ""]
    if explain_audit.exists():
        ex = json.loads(explain_audit.read_text(encoding="utf-8"))
        sections += [
            f"- G-PX-EX.1 PPA medio: **{ex.get('g_px_ex_1_ppa_mean', 0):.3f}**",
            f"- G-PX-EX.3 ECE medio: **{ex.get('g_px_ex_3_ece_mean', 0):.4f}**",
            f"- G-PX-EX.4 PCA_V medio: **{ex.get('g_px_ex_4_pca_v_mean', float('nan')):.3f}**",
            f"- Tiles auditados: {ex.get('n_tiles_explain', 0)}",
        ]
    else:
        sections.append("_Ejecutar tools/stage2_pixel_explain_audit.py_")

    sections += [
        "",
        "## 5. Mapas de inferencia (test)",
        "",
        "Por imagen: `outputs/<run>_stage2_infer__AM__gate.../maps/`",
        "",
        "| Archivo | Contenido |",
        "|---------|-----------|",
        "| `*_L9_morph_classes.png` | IH/A/V/H/ROOT píxel |",
        "| `*_L9_colony_binary.png` | Colonias binarias |",
        "| `*_L9_smoothness.png` | Entropía local argmax (legacy: L9_confidence) |",
        "| `*_L9_prior_frangi*.png` | Evidencia Frangi |",
        "| `*_L9_prob_*.png` | Softmax por clase |",
        "| `*_L9_disagreement.png` | ViT vs weak |",
        "| `*_pixel_explain_quant.parquet` | PPA/ECE/narrativa por tile |",
        "| `reports/EXPLICABILIDAD_PIXEL.md` | Narrativa corrida |",
        "| `*_L9_diagnostico.png` | Overlay diagnóstico |",
        "| `*_pixel_seg.npz` | Mapa uint8 H×W |",
        "| `*_pixel_morph_quant.parquet` | Cuantificación por tile |",
    ]

    infer_runs = sorted(
        (ROOT / "outputs").glob(f"*stage2_infer__AM__gate{gate_run_id}__*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seen = set()
    sections += ["", "### Runs inferencia recientes", ""]
    for run in infer_runs[:12]:
        stem = run.name.split("__")[-1][:40]
        if stem in seen:
            continue
        seen.add(stem)
        maps = list((run / "maps").glob("*L9*.png")) if (run / "maps").exists() else []
        sections.append(f"- `{run.name}` — {len(maps)} mapas L9")

    out = rep / f"STAGE2_PIXEL_INFORME__gate{gate_run_id}.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Informe -> {out}")


if __name__ == "__main__":
    main()
