"""Refleja el canvas Gate AM desde el JSON en disco (espejo a 2 Hz).

El canvas SOLO copia el resultado del entrenamiento: este watcher lee el JSON
que escribe run.py y parchea el snapshot del canvas. No tiene ninguna relacion
con el loop de entrenamiento (proceso independiente).

Uso tipico (2 Hz):
    python tools/sync_gate_canvas.py --watch 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from micorizae.phase_d_stage1.gate_run_live import (  # noqa: E402
    _pipeline_live_path,
    _read_json_if_exists,
    refresh_gate_live_canvas,
)


def _heartbeat() -> str:
    live = _read_json_if_exists(_pipeline_live_path())
    ep = live.get("epoch_current")
    et = live.get("epochs_total")
    b = live.get("batch_current")
    bt = live.get("batches_train_this_epoch")
    loss = live.get("train_loss_avg")
    loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
    return f"ep {ep}/{et} batch {b}/{bt} loss={loss_s}"


def main() -> None:
    p = argparse.ArgumentParser(description="Espejo del canvas Gate AM desde disco")
    p.add_argument("--run-id", default=None, help="Run id (default: ultimo gate_am_train)")
    p.add_argument("--watch", type=float, default=0.5, help="Intervalo de refresco (s); 2 Hz=0.5")
    p.add_argument("--log-every", type=float, default=5.0, help="Heartbeat cada N s")
    args = p.parse_args()
    interval = max(0.05, float(args.watch))

    print(f"[canvas-mirror] reflejando a {1/interval:.1f} Hz (cada {interval}s)", flush=True)
    last_log = 0.0
    while True:
        try:
            refresh_gate_live_canvas(args.run_id)
        except Exception as e:  # noqa: BLE001
            print(f"[canvas-mirror] error: {e}", flush=True)
        now = time.time()
        if now - last_log >= args.log_every:
            last_log = now
            try:
                print(f"[canvas-mirror] {_heartbeat()}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(interval)


if __name__ == "__main__":
    main()
