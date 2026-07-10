#!/usr/bin/env python3
"""Perfil de conformación cache/stage2_pixel_mplus_v1.h5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from micorizae.phase_e_stage2.stage2_pixel_h5_profile import build_h5_conformation_profile


def main() -> None:
    import config as user_config  # type: ignore

    v_min = int(getattr(user_config, "STAGE2_PIXEL_V_MIN_PX", 30))
    profile = build_h5_conformation_profile(v_min_px=v_min)
    print(json.dumps({"conformation_ok": profile["conformation_ok"], "path": profile["h5_path"]}, indent=2))


if __name__ == "__main__":
    main()
