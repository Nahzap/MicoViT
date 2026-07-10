#!/usr/bin/env python3
"""Pipeline completo Stage2-Pixel E2-P0 → E2-EX + E2-P4."""

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _run(cmd: list[str], log_path: Path) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {' '.join(cmd)} ===\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"Falló ({r.returncode}): {' '.join(cmd)}")


def _latest_stage2_run_id(gate_run_id: str) -> str:
    pattern = f"*stage2_pixel_train__AM__gate{gate_run_id}*"
    runs = sorted((ROOT / "outputs").glob(pattern))
    if not runs:
        raise FileNotFoundError(f"No hay corridas Stage2-Pixel para gate {gate_run_id!r}")
    return runs[-1].name


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Stage2-Pixel + MEViT explain")
    parser.add_argument("--explain", choices=["off", "on", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    import config as user_config  # type: ignore

    gate_run_id = str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", ""))
    epochs = args.epochs or int(getattr(user_config, "STAGE2_PIXEL_EPOCHS", 20))
    log_path = ROOT / "outputs" / f"stage2_pixel_pipeline__gate{gate_run_id}.log"
    log_path.write_text(f"Stage2-Pixel pipeline {datetime.now().isoformat()}\n", encoding="utf-8")

    print("[0/8] Gate embed cache (prerrequisito Modelo 1)...")
    _run([str(PY), str(ROOT / "run.py"), "build-gate-cache"], log_path)

    print("[1/8] E2-P0 baseline audit...")
    _run([str(PY), str(ROOT / "tools" / "stage2_pixel_baseline_audit.py")], log_path)

    print(f"[2/8] E2-P3 train {epochs} epochs (MEViT prior losses)...")
    train_cmd = [str(PY), str(ROOT / "run.py"), "train-stage2-pixel", "--full", "--epochs", str(epochs)]
    _run(train_cmd, log_path)

    print("[3/8] E2-EX progress tests...")
    _run([str(PY), str(ROOT / "tools" / "stage2_pixel_progress_tests.py")], log_path)

    print("[4/8] E2-P2 holdout val/test...")
    _run([str(PY), str(ROOT / "tools" / "stage2_pixel_holdout_eval.py")], log_path)

    if args.explain in ("on", "full"):
        print("[5/8] E2-EX explain audit...")
        _run([str(PY), str(ROOT / "tools" / "stage2_pixel_explain_audit.py")], log_path)
    else:
        print("[5/8] E2-EX explain audit — omitido")

    print("[6/8] infer-stage2 test images...")
    test_dir = ROOT / "Data" / "am" / "am" / "test"
    ok, fail = [], []
    for img in sorted(test_dir.glob("*.jpg")):
        cmd = [str(PY), str(ROOT / "run.py"), "infer-stage2", "--image", str(img), "--lineage", "AM"]
        with open(log_path, "a", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            ok.append(img.name)
        else:
            fail.append(img.name)

    print("[7/8] posttrain report (Gate→Stage2 full-image)...")
    run_id = _latest_stage2_run_id(gate_run_id)
    _run(
        [
            str(PY),
            str(ROOT / "run.py"),
            "stage2-pixel-posttrain-report",
            "--run-id",
            run_id,
            "--force",
            "--fullimage",
        ],
        log_path,
    )

    print("[8/8] informe final...")
    _run([str(PY), str(ROOT / "tools" / "stage2_pixel_write_report.py"), "--gate-run-id", gate_run_id], log_path)

    summary = {
        "gate_run_id": gate_run_id,
        "epochs": epochs,
        "explain": args.explain,
        "infer_ok": len(ok),
        "infer_fail": len(fail),
        "log": str(log_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
