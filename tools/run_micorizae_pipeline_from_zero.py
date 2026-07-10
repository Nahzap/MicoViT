#!/usr/bin/env python3
"""Pipeline MicorizaeVision — 6 pasos secuenciales (1 terminal).

  1. Conformar database (manifest + tiles_index)
  2. Database → HDF5 Gate (luma + labels tile M+/M-/BG)
  3. Entrenar Gate (DINOv2 + probe): clasificador tile para cualquier imagen AM
  4. HDF5 Stage2 (tiles M+ → rgb, weak IH/V/A/H, priors v3)
  5. Entrenar Stage2-Pixel ViT
  6. Resultados + full-image (divide en tiles, mapas píxel densos)

Pasos 2–3 se ejecutan juntos en ``train-gate-am`` (orquestación interna Gate SSD).
"""

from __future__ import annotations

import argparse
import json
import os
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
    """Ejecuta un subproceso transmitiendo su salida EN VIVO a la consola y al log.

    Lee stdout/stderr línea a línea (PYTHONUNBUFFERED) y hace tee a:
      - la consola del orquestador (para que el desarrollo sea autoexplicativo), y
      - ``log_path`` (evidencia persistente completa).
    Así no se pierde ningún mensaje de avance ni métricas.
    """
    line = " ".join(cmd)
    header = f"\n{'=' * 88}\n=== [{datetime.now().strftime('%H:%M:%S')}] {line}\n{'=' * 88}"
    print(header, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(header + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            sys.stdout.write(raw)
            sys.stdout.flush()
            f.write(raw)
            f.flush()
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"Falló ({rc}): {line}")


def _gate_run_id() -> str:
    import config as user_config  # type: ignore

    return str(getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", "") or getattr(user_config, "STAGE2_GATE_RUN_ID", ""))


def _latest_stage2_run_id(gate_run_id: str) -> str:
    runs = sorted((ROOT / "outputs").glob(f"*stage2_pixel_train__AM__gate{gate_run_id}*"))
    if not runs:
        raise FileNotFoundError(f"No hay corrida Stage2 para gate {gate_run_id!r}")
    return runs[-1].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Micorizae 6 pasos")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-gate-train", action="store_true", help="Solo build-gate-cache; no train-gate-am")
    parser.add_argument("--gate-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--wipe-stage2-h5", action="store_true")
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 7))
    args = parser.parse_args()

    import config as user_config  # type: ignore

    gate_epochs = args.gate_epochs or int(getattr(user_config, "GATE_EPOCHS", 20))
    stage2_epochs = args.stage2_epochs or int(getattr(user_config, "STAGE2_PIXEL_EPOCHS", 20))
    log_path = ROOT / "outputs" / f"micorizae_pipeline_6step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text(f"Pipeline 6 pasos {datetime.now().isoformat()}\n", encoding="utf-8")

    plan: list[tuple[int, str, list[str]]] = []

    if not args.skip_ingest:
        plan.append((1, "ingest + tiles-index", [str(PY), str(ROOT / "run.py"), "ingest"]))
        plan.append((1, "tiles-index", [str(PY), str(ROOT / "run.py"), "tiles-index"]))

    if args.skip_gate_train:
        plan.append((2, "embed cache Gate (sin re-entrenar)", [str(PY), str(ROOT / "run.py"), "build-gate-cache", "--force"]))
    else:
        plan.append(
            (
                2,
                "Gate SSD: HDF5 tiles + embed cache + train probe (pasos 2+3)",
                [str(PY), str(ROOT / "run.py"), "train-gate-am", "--full", "--epochs", str(gate_epochs)],
            )
        )

    if args.wipe_stage2_h5:
        for name in ("stage2_pixel_mplus_v1.h5", "stage2_pixel_mplus_v1.meta.json", "stage2_pixel_mplus_v1.lookup.parquet"):
            (ROOT / "cache" / name).unlink(missing_ok=True)

    plan.append(
        (4, "HDF5 Stage2 M+", [str(PY), str(ROOT / "run.py"), "build-stage2-pixel-cache", "--force-rebuild"])
    )
    plan.append(
        (5, "train Stage2-Pixel ViT (+ posttrain auto si config)", [str(PY), str(ROOT / "run.py"), "train-stage2-pixel", "--epochs", str(stage2_epochs)])
    )
    # Paso 6: redundante si STAGE2_PIXEL_POSTTRAIN_FULL_REPORT=True en train; útil para regenerar solo full-image
    plan.append((6, "posttrain full-image (regenerar si hace falta)", []))

    current_step = 0
    for step_num, label, cmd in plan:
        if step_num < args.from_step:
            continue
        if step_num == 6:
            run_id = _latest_stage2_run_id(_gate_run_id())
            cmd = [
                str(PY),
                str(ROOT / "run.py"),
                "stage2-pixel-posttrain-report",
                "--run-id",
                run_id,
                "--force",
                "--fullimage",
            ]
        if step_num != current_step:
            current_step = step_num
        print(f"[{step_num}/6] {label}", flush=True)
        _run(cmd, log_path)

    print(json.dumps({"log": str(log_path), "gate_run_id": _gate_run_id()}, indent=2))


if __name__ == "__main__":
    main()
