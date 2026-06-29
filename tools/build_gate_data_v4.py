"""Prepara datos gate AM v4: split AMFinder + manifests + multi-densidad.

Uso:
  python tools/build_gate_data_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config as project_config
from micorizae.common.io import read_table
from micorizae.common.logging_utils import get_logger
from micorizae.gate_data_splits import split_gate_am_v4
from micorizae.phase_a_ingest import build_manifests
from micorizae.phase_b_tiling import build_tiles_index

log = get_logger("tools.build_gate_data_v4")


def main() -> int:
    import subprocess

    prep = ROOT / "tools" / "prepare_amfinder_split.py"
    subprocess.run([sys.executable, str(prep)], check=True, cwd=str(ROOT))

    log.info("Fase A: build_manifests...")
    build_manifests()

    tiers = tuple(
        t.strip()
        for t in str(project_config.GATE_MULTIDENSITY_TIERS).split(",")
        if t.strip()
    )
    apply_md = bool(project_config.GATE_MULTIDENSITY_ENABLED)
    log.info(f"Fase B: tiles_index (multidensity={apply_md}, tiers={tiers})...")
    build_tiles_index(
        apply_multidensity=apply_md,
        multidensity_tiers=tiers,
        base_tile_size=int(project_config.GATE_TILE_SIZE_BASE),
        dense_tile_size=int(project_config.GATE_TILE_SIZE_DENSE),
        coarse_tile_size=int(project_config.GATE_TILE_SIZE_COARSE),
    )

    train_df, val_df, external_df, info = split_gate_am_v4()
    log.info(
        f"Split v4: train={info['n_train_tiles']:,} val={info['n_val_tiles']:,} "
        f"external={info['n_external_tiles']:,}"
    )
    log.info(f"  train density: {info.get('train_by_density')}")
    log.info(f"  train subset: {info.get('train_by_subset')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
