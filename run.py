"""Punto de entrada unico para MicorizaeVision.

Uso principal:
    python run.py                    # pipeline gate AM: cache + entrenamiento (config.py)

Subcomandos avanzados:
    python run.py build-gate-cache
    python run.py build-gate-attention-cache
    python run.py train-gate-am
    python run.py recover-gate-am-report
    python run.py gate-am-external-eval [--run-id ...] [--max-images 10]
    python run.py infer-gate-am --image <ruta>
    python run.py --help
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_RUN_LOCK = _ROOT / "outputs" / ".gate_run.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_singleton_lock() -> None:
    """Un solo run.py de entrenamiento a la vez (evita procesos duplicados)."""
    _RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if _RUN_LOCK.is_file():
        try:
            old_pid = int(_RUN_LOCK.read_text(encoding="ascii").strip())
        except ValueError:
            old_pid = 0
        if _pid_alive(old_pid) and old_pid != os.getpid():
            raise SystemExit(
                f"Ya hay un run activo (pid={old_pid}). "
                f"Termina ese proceso antes de iniciar otro."
            )
    _RUN_LOCK.write_text(str(os.getpid()), encoding="ascii")


def _release_singleton_lock() -> None:
    try:
        if _RUN_LOCK.is_file() and int(_RUN_LOCK.read_text(encoding="ascii").strip()) == os.getpid():
            _RUN_LOCK.unlink()
    except (OSError, ValueError):
        pass


def _on_final_interpreter() -> bool:
    """True solo en el intérprete final (post re-exec venv)."""
    if os.environ.get("MICORIZAE_VENV_REEXEC") == "1":
        return True
    if sys.platform == "win32":
        venv_py = _ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = _ROOT / ".venv" / "bin" / "python"
    if not venv_py.is_file():
        return True
    try:
        return Path(sys.executable).resolve() == venv_py.resolve()
    except OSError:
        return True


def _is_train_entrypoint() -> bool:
    return len(sys.argv) == 1 or (len(sys.argv) >= 2 and sys.argv[1] == "train-gate-am")


def _reexec_in_project_venv() -> None:
    """Relanza con .venv del repo si el intérprete actual no es el del proyecto."""
    if os.environ.get("MICORIZAE_SKIP_VENV") == "1":
        return
    if os.environ.get("MICORIZAE_VENV_REEXEC") == "1":
        return
    if sys.platform == "win32":
        venv_py = _ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = _ROOT / ".venv" / "bin" / "python"
    if not venv_py.is_file():
        return
    try:
        same = Path(sys.executable).resolve() == venv_py.resolve()
    except OSError:
        same = False
    if same:
        return
    env = os.environ.copy()
    env["MICORIZAE_VENV_REEXEC"] = "1"
    os.execve(str(venv_py), [str(venv_py), *sys.argv], env)


def _bootstrap_src_path() -> None:
    """Asegura que `src/` este en sys.path para importar `micorizae`."""
    src = _ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_config():
    try:
        import config as user_config  # type: ignore

        return user_config
    except Exception as e:
        raise SystemExit(f"No se pudo cargar config.py: {e}") from e


def _validate_infer_defaults(cfg) -> None:
    image = getattr(cfg, "DEFAULT_IMAGE", None)
    if image is None:
        raise SystemExit(
            "config.py: DEFAULT_IMAGE es None. Define una ruta valida para inferencia."
        )
    p = Path(image)
    if not p.exists():
        raise SystemExit(f"config.py: DEFAULT_IMAGE no existe: {p}")


def main() -> None:
    _reexec_in_project_venv()
    _bootstrap_src_path()
    os.environ["MICORIZAE_ENTRYPOINT"] = "run.py"

    if _is_train_entrypoint() and _on_final_interpreter():
        _acquire_singleton_lock()
        atexit.register(_release_singleton_lock)

    if len(sys.argv) == 1:
        cfg = _load_config()
        from micorizae.gate_runflow import run_gate_pipeline

        run_gate_pipeline(cfg)
        return

    if len(sys.argv) >= 2 and sys.argv[1] in {"infer-gate-am", "infer-stage2", "inspect-labels", "weakseg-morph"}:
        _validate_infer_defaults(_load_config())

    from micorizae.cli import app

    app()


if __name__ == "__main__":
    main()
