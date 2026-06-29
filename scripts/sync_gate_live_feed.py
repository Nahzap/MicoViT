"""DEPRECADO: el canvas live se sincroniza desde run.py durante el entrenamiento.

No ejecutes este script en paralelo. Usa solo:

    python run.py
    python run.py train-gate-am

Si necesitas re-espejar metricas de una corrida ya terminada (una vez):

    python scripts/sync_gate_live_feed.py outputs/<run_id>/reports/live_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from micorizae.phase_d_stage1.gate_run_live import _mirror_canvas_live_feed, _sanitize_json_nan


def _load_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").replace(": NaN", ": null")
    return _sanitize_json_nan(json.loads(text))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Uso puntual: python scripts/sync_gate_live_feed.py "
            "outputs/<run_id>/reports/live_metrics.json\n"
            "Durante entrenamiento usa solo: python run.py"
        )
    src = Path(sys.argv[1])
    if not src.is_file():
        raise SystemExit(f"No existe: {src}")
    payload = _load_metrics(src)
    _mirror_canvas_live_feed(payload)
    print(f"[sync] snapshot ep {payload.get('last_epoch')} @ {payload.get('updated_at')}")


if __name__ == "__main__":
    main()
