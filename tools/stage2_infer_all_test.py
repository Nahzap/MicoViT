#!/usr/bin/env python3
"""Inferencia acoplada gate+Stage2 en todas las imágenes am/test."""

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


def main() -> None:
    import config as user_config  # type: ignore

    gate_run_id = str(getattr(user_config, "STAGE2_GATE_RUN_ID", ""))
    test_dir = ROOT / "Data" / "am" / "am" / "test"
    images = sorted(test_dir.glob("*.jpg"))
    log = ROOT / "outputs" / f"stage2_infer_all_test__gate{gate_run_id}.log"
    ok, fail = [], []
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"start {datetime.now().isoformat()}\n")
        for img in images:
            cmd = [str(PY), str(ROOT / "run.py"), "infer-stage2", "--image", str(img), "--lineage", "AM"]
            f.write(f"\n=== {img.name} ===\n")
            f.flush()
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if r.returncode == 0:
                ok.append(img.name)
            else:
                fail.append(img.name)
        summary = {"gate_run_id": gate_run_id, "ok": ok, "fail": fail, "n_ok": len(ok)}
        f.write(f"\n{json.dumps(summary, indent=2)}\n")
    print(json.dumps({"ok": len(ok), "fail": len(fail), "log": str(log)}, indent=2))


if __name__ == "__main__":
    main()
