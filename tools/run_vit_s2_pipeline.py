#!/usr/bin/env python3
"""Pipeline completo ViT-S2 — audit, train, eval, infer test (plan 20260628_225244)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {' '.join(cmd)} ===\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando falló ({proc.returncode}): {' '.join(cmd)}")


def inventory_outputs(gate_run_id: str) -> dict:
    out_root = ROOT / "outputs"
    runs = sorted(
        [p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    gate_runs = [p.name for p in runs if "gate_am_train" in p.name]
    stage2_runs = [p.name for p in runs if "stage2" in p.name]
    inv = {
        "generated": datetime.now().isoformat(),
        "gate_run_reference": gate_run_id,
        "total_run_dirs": len(runs),
        "gate_runs_count": len(gate_runs),
        "stage2_runs_count": len(stage2_runs),
        "latest_gate_run": gate_runs[0] if gate_runs else None,
        "best_last_gate": "20260622_180955_gate_am_train_BEST_LAST",
        "latest_e7_gate": "20260624_012932_gate_am_train",
        "gate_runs_recent": gate_runs[:15],
    }
    rep_dir = out_root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / f"OUTPUTS_INVENTORY__gate{gate_run_id}.md"
    lines = [
        f"# Inventario outputs — gate `{gate_run_id}`",
        "",
        f"- Generado: {inv['generated']}",
        f"- Runs gate AM: {inv['gate_runs_count']}",
        f"- Runs stage2: {inv['stage2_runs_count']}",
        f"- Último gate: `{inv['latest_gate_run']}`",
        f"- Referencia activa config: `{gate_run_id}`",
        "",
        "## Runs gate recientes",
        "",
    ]
    for r in gate_runs[:20]:
        marker = " **← REF**" if gate_run_id in r else ""
        lines.append(f"- `{r}`{marker}")
    path.write_text("\n".join(lines), encoding="utf-8")
    (rep_dir / f"OUTPUTS_INVENTORY__gate{gate_run_id}.json").write_text(
        json.dumps(inv, indent=2), encoding="utf-8"
    )
    return inv


def main() -> None:
    import config as user_config  # type: ignore

    gate_run_id = str(getattr(user_config, "STAGE2_GATE_RUN_ID", "20260624_012932_gate_am_train"))
    log_path = ROOT / "outputs" / f"vit_s2_pipeline__gate{gate_run_id}.log"
    log_path.write_text(f"ViT-S2 pipeline start {datetime.now().isoformat()}\n", encoding="utf-8")

    print("[1/5] Inventario outputs...")
    inventory_outputs(gate_run_id)

    print("[2/5] Auditoría datos S0...")
    _run(
        [str(PY), str(ROOT / "tools" / "stage2_data_audit.py"),
         "--gate-run-id", gate_run_id,
         "--out", str(ROOT / "outputs" / "stage2_audit" / f"gate{gate_run_id}")],
        log_path,
    )

    print("[3/5] Entrenamiento ViT-S2 formal...")
    _run([str(PY), str(ROOT / "run.py"), "train-stage2", "--full"], log_path)

    print("[4/5] Holdout eval...")
    _run(
        [str(PY), str(ROOT / "tools" / "stage2_holdout_eval.py"), "--gate-run-id", gate_run_id],
        log_path,
    )

    test_dir = ROOT / "Data" / "am" / "am" / "test"
    images = sorted(test_dir.glob("*.jpg"))
    print(f"[5/5] Inferencia acoplada {len(images)} imágenes test...")
    for img in images:
        _run(
            [str(PY), str(ROOT / "run.py"), "infer-stage2", "--image", str(img), "--lineage", "AM"],
            log_path,
        )

    # Informe final
    holdout_path = ROOT / "outputs" / "reports" / f"stage2_holdout__gate{gate_run_id}.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.exists() else {}
    stage2_runs = sorted(
        [p.name for p in (ROOT / "outputs").iterdir() if p.is_dir() and p.name.startswith("20") and "stage2_train" in p.name],
        reverse=True,
    )
    train_run = stage2_runs[0] if stage2_runs else "unknown"
    infer_runs = sorted(
        [p.name for p in (ROOT / "outputs").iterdir() if p.is_dir() and "stage2_infer" in p.name and gate_run_id in p.name],
        reverse=True,
    )

    final = ROOT / "outputs" / "reports" / f"STAGE2_VIT_S2_INFORME__gate{gate_run_id}.md"
    final.write_text(
        "\n".join([
            f"# Informe ViT-S2 — gate `{gate_run_id}`",
            "",
            f"- **Fecha:** {datetime.now().isoformat()}",
            f"- **Plan:** `Docs/20260628_225244_PLAN_VIT_STAGE2_DISCRIMINADOR_MPLUS.md`",
            f"- **Run train Stage2:** `{train_run}`",
            f"- **Checkpoint:** `models/checkpoints/stage2_am/stage2_am_branch_a_best.pt`",
            f"- **Gate checkpoint:** `models/checkpoints/gate_am/gate_tile_dino_best.pt`",
            "",
            "## Holdout test (gold M+)",
            "",
            f"- macro F1: **{holdout.get('macro_f1', 'N/A')}**",
            f"- n tiles: {holdout.get('n_tiles', 'N/A')}",
            "",
            "## Inferencias test",
            "",
            *[f"- `{r}`" for r in infer_runs[:12]],
            "",
            "## Artefactos",
            "",
            f"- `{holdout_path}`",
            f"- `outputs/stage2_audit/gate{gate_run_id}/STAGE2_DATA_AUDIT.md`",
            f"- `outputs/reports/OUTPUTS_INVENTORY__gate{gate_run_id}.md`",
            f"- Log pipeline: `{log_path.name}`",
        ]),
        encoding="utf-8",
    )
    print(f"\nPipeline OK. Informe: {final}")


if __name__ == "__main__":
    main()
