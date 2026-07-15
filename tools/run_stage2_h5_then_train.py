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
            safe = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            )
            try:
                sys.stdout.write(safe)
                sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.buffer.write(safe.encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            logf.write(line)
            logf.flush()
        return int(proc.wait())


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import config as user_config  # type: ignore

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_log = ROOT / "outputs" / f"stage2_label_preview_{ts}.log"
    h5_log = ROOT / "outputs" / f"h5_rebuild_v8_srp_{ts}.log"
    train_log = ROOT / "outputs" / f"train_stage2_v8_srp_{ts}.log"
    status = ROOT / "outputs" / f"m6_pipeline_{ts}.status.txt"
    py = _project_python()
    preview_n = int(getattr(user_config, "STAGE2_PIXEL_LABEL_PREVIEW_N", 25))
    do_preview = bool(getattr(user_config, "STAGE2_PIXEL_LABEL_PREVIEW_BEFORE_H5", True))
    status.write_text(
        f"START={datetime.now().isoformat()}\nPY={py}\n"
        f"PREVIEW_LOG={preview_log}\nH5_LOG={h5_log}\nTRAIN_LOG={train_log}\n",
        encoding="utf-8",
    )
    print(f"[m6] python={py}", flush=True)

    step = 0
    total = 3 if do_preview else 2
    if do_preview:
        step += 1
        print(
            f"[m6] paso {step}/{total}: preview-stage2-pixel-labels --n-tiles {preview_n}",
            flush=True,
        )
        ec = _run_live(
            [
                py,
                "-u",
                str(ROOT / "run.py"),
                "preview-stage2-pixel-labels",
                "--n-tiles",
                str(preview_n),
            ],
            preview_log,
        )
        with status.open("a", encoding="utf-8") as sf:
            sf.write(f"PREVIEW_EXIT={ec} {datetime.now().isoformat()}\n")
        if ec != 0:
            print(f"[m6] PREVIEW FAIL exit={ec}", flush=True)
            return ec
        print("[m6] PREVIEW OK — revisar outputs/stage2_label_preview_*/panels/", flush=True)

    step += 1
    print(f"[m6] paso {step}/{total}: build-stage2-pixel-cache --force-rebuild", flush=True)
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

    step += 1
    print(f"[m6] paso {step}/{total}: train-stage2-pixel --epochs 20", flush=True)
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
