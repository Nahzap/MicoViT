#!/usr/bin/env python3
"""Watchdog Stage2-Pixel — detecta entrenamiento muerto / congelado y ALARMA.

Modos de fallo que cubre (la muerte silenciosa NO puede autopostearse desde el
proceso muerto — Windows TDR / Taskkill / OOM killer / cierre del terminal):

1. Heartbeat stale (sin escribirse > --stale-sec)
2. PID del train ya no existe mientras el run NO está marcado complete
3. Excepción legible en train_warnings.jsonl recién escrita (solo notifica)

Lanza alarma ruidosa a stderr + beeps + append a train_warnings.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    pid = int(pid)
    if os.name == "nt":
        try:
            import ctypes

            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            # ERROR_ACCESS_DENIED (5) ⇒ proceso existe pero sin permiso
            err = ctypes.windll.kernel32.GetLastError()
            return err == 5
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False


def _alarm(code: str, message: str, warn_path: Path, *, beeps: int = 5) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    bar = "=" * 72
    text = (
        f"\n{bar}\n"
        f"  !! Stage2-Pixel WATCHDOG ALARMA  code={code}  @ {ts}\n"
        f"  {message}\n"
        f"{bar}\n\n"
    )
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        print(text, flush=True)
    except Exception:
        pass
    try:
        warn_path.parent.mkdir(parents=True, exist_ok=True)
        with open(warn_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": ts,
                        "level": "error",
                        "code": code,
                        "message": message,
                        "source": "watchdog",
                    },
                    default=str,
                )
                + "\n"
            )
    except Exception:
        pass
    try:
        if os.name == "nt":
            import winsound

            for _ in range(beeps):
                winsound.Beep(1100, 400)
        else:
            for _ in range(beeps):
                sys.stdout.write("\a")
                sys.stdout.flush()
    except Exception:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


def _run_complete(run_dir: Path, hb: dict | None = None, train_pid: int | None = None) -> bool:
    if (run_dir / "TRAIN_COMPLETE.flag").is_file():
        return True
    # Señales de fin normal
    for rel in (
        "training_state.json",
        "../posttrain/best_metrics.json",
        "../STAGE2_PIXEL_RUN_META.json",
    ):
        p = (run_dir / rel).resolve()
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        interrupted = data.get("interrupted")
        if interrupted is False:
            return True
        status = str(data.get("status", "")).lower()
        if status in {"complete", "completed", "ok", "done"}:
            return True
    # Inferencia: checkpoint de la última época existe y el proceso ya no vive
    # (cubre runs lanzados antes del flag TRAIN_COMPLETE).
    epochs_total = None
    if hb and hb.get("epochs_total") is not None:
        try:
            epochs_total = int(hb["epochs_total"])
        except Exception:
            epochs_total = None
    if epochs_total is None:
        try:
            metrics = run_dir / "training_metrics.csv"
            if metrics.is_file():
                # no requiere pandas
                lines = [ln for ln in metrics.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if len(lines) >= 2:
                    # max epoch in first column
                    epochs_seen = []
                    for ln in lines[1:]:
                        try:
                            epochs_seen.append(int(ln.split(",")[0]))
                        except Exception:
                            pass
                    if epochs_seen:
                        # si metrics tiene N filas y existe epoch_N.pt, y pid muerto → ok
                        pass
        except Exception:
            pass
    if epochs_total is not None:
        final_ckpt = run_dir / "checkpoints" / f"epoch_{epochs_total:03d}.pt"
        if final_ckpt.is_file() and (train_pid is None or not _pid_alive(train_pid)):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Watchdog Stage2-Pixel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stale-sec",
        type=int,
        default=180,
        help="Segundos sin heartbeat = congelado (default 180)",
    )
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument(
        "--train-pid",
        type=int,
        default=0,
        help="PID del proceso de entrenamiento (si muere ⇒ alarma inmediata)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Una sola pasada (útil para tests)",
    )
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / args.run_id / "pretrain"
    hb_path = run_dir / "train_heartbeat.json"
    warn_path = run_dir / "train_warnings.jsonl"
    train_pid = int(args.train_pid) or None

    print(
        f"[Watchdog] run={args.run_id} stale>{args.stale_sec}s "
        f"interval={args.interval}s train_pid={train_pid or 'from-heartbeat'}",
        flush=True,
    )

    alarmed_dead = False
    alarmed_stale = False
    last_hb_ts = ""

    while True:
        now = datetime.now()
        hb: dict = {}
        if hb_path.is_file():
            try:
                hb = json.loads(hb_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[{now.strftime('%H:%M:%S')}] heartbeat ilegible: {exc}", flush=True)

        hb_pid = int(hb["pid"]) if hb.get("pid") else None
        monitor_pid = train_pid or hb_pid

        if _run_complete(run_dir, hb=hb, train_pid=monitor_pid):
            print(f"[{now.strftime('%H:%M:%S')}] TRAIN_COMPLETE — watchdog exit OK", flush=True)
            return

        if monitor_pid and not _pid_alive(monitor_pid) and not alarmed_dead:
            # Dar margen: puede estar escribiendo posttrain / último ckpt
            time.sleep(5)
            if _run_complete(run_dir, hb=hb, train_pid=monitor_pid):
                print(f"[{now.strftime('%H:%M:%S')}] TRAIN_COMPLETE tras fin de PID — OK", flush=True)
                return
            if not _pid_alive(monitor_pid):
                msg = (
                    f"PROCESO TRAIN MUERTO (pid={monitor_pid}). "
                    f"Último heartbeat: ep={hb.get('epoch')} phase={hb.get('phase')} "
                    f"batch={hb.get('batch')} ts={hb.get('ts')}. "
                    f"Causa probable: TDR GPU / Taskkill / OOM / cierre de terminal. "
                    f"Reanudar: python run.py train-stage2-pixel --full --epochs N "
                    f"--run-id {args.run_id} --resume"
                )
                _alarm("TRAIN_PROCESS_DEAD", msg, warn_path)
                alarmed_dead = True

        if hb:
            try:
                ts = datetime.fromisoformat(str(hb["ts"]))
                age = (now - ts).total_seconds()
            except Exception:
                age = -1
            if hb.get("ts") != last_hb_ts:
                last_hb_ts = str(hb.get("ts"))
                alarmed_stale = False  # se recuperó → reset
            line = (
                f"[{now.strftime('%H:%M:%S')}] ep={hb.get('epoch')} "
                f"phase={hb.get('phase')} batch={hb.get('batch')} "
                f"age={age:.0f}s pid={monitor_pid} alive={_pid_alive(monitor_pid) if monitor_pid else '?'} "
                f"best_mIoU={hb.get('best_miou', '?')}"
            )
            print(line, flush=True)
            if age > args.stale_sec and monitor_pid and _pid_alive(monitor_pid) and not alarmed_stale:
                msg = (
                    f"HEARTBEAT STALE {age:.0f}s (umbral {args.stale_sec}s) — "
                    f"proceso vivo (pid={monitor_pid}) pero CONGELADO. "
                    f"Última fase={hb.get('phase')} ep={hb.get('epoch')} "
                    f"batch={hb.get('batch')}."
                )
                _alarm("HEARTBEAT_STALE", msg, warn_path)
                alarmed_stale = True
        else:
            print(f"[{now.strftime('%H:%M:%S')}] sin heartbeat aún", flush=True)

        if args.once:
            return
        # Si ya alarmó muerte de proceso, salir (no spamear)
        if alarmed_dead:
            print("[Watchdog] alarma TRAIN_PROCESS_DEAD emitida — exit", flush=True)
            return
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
