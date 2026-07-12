"""M6: force-rebuild Stage2 H5 then formal train — progreso en TERMINAL + log."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_python() -> str:
    """Prefer repo .venv — system Python + run.py os.execve crashes on Win (0xC0000005)."""
    if sys.platform == "win32":
        venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _run_live(cmd: list[str], log_path: Path) -> int:
    """Corre cmd con stdout en vivo en la terminal y copia al log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("MICORIZAE_SKIP_VENV", "1")

    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        logf.write(f"CMD={' '.join(cmd)}\n")
        logf.flush()
        print(f"[m6] >>> {' '.join(cmd)}", flush=True)
        print(f"[m6] log={log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        return int(proc.wait())


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_log = ROOT / "outputs" / f"h5_rebuild_v8_srp_{ts}.log"
    train_log = ROOT / "outputs" / f"train_stage2_v8_srp_{ts}.log"
    status = ROOT / "outputs" / f"m6_pipeline_{ts}.status.txt"
    py = _project_python()
    status.write_text(
        f"START={datetime.now().isoformat()}\nPY={py}\nH5_LOG={h5_log}\nTRAIN_LOG={train_log}\n",
        encoding="utf-8",
    )
    print(f"[m6] python={py}", flush=True)
    print("[m6] paso 1/2: build-stage2-pixel-cache --force-rebuild", flush=True)
    ec = _run_live(
        [py, "-u", str(ROOT / "run.py"), "build-stage2-pixel-cache", "--force-rebuild"],
        h5_log,
    )
    with status.open("a", encoding="utf-8") as sf:
        sf.write(f"H5_EXIT={ec} {datetime.now().isoformat()}\n")
    if ec != 0:
        print(f"[m6] H5 FAIL exit={ec}", flush=True)
        return ec
    print("[m6] H5 OK", flush=True)
    print("[m6] paso 2/2: train-stage2-pixel --epochs 20", flush=True)
    ec = _run_live(
        [py, "-u", str(ROOT / "run.py"), "train-stage2-pixel", "--epochs", "20"],
        train_log,
    )
    with status.open("a", encoding="utf-8") as sf:
        sf.write(f"TRAIN_EXIT={ec} {datetime.now().isoformat()}\n")
    print(f"[m6] TRAIN_EXIT={ec}", flush=True)
    return ec


if __name__ == "__main__":
    raise SystemExit(main())
