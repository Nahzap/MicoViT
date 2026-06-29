#!/usr/bin/env python3
"""Wrapper CLI → run_full_embed_analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    import pandas as pd

    from micorizae.phase_d_stage1 import split_by_image
    from micorizae.phase_d_stage1.gate_embed_analysis import run_full_embed_analysis

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "gate_am" / "gate_tile_dino_best.pt",
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    args = parser.parse_args()

    train_df, val_df, _ = split_by_image(
        lineages=["AM"],
        split_mode="fixed",
        train_splits=("train",),
        val_splits=("test",),
        exclude_unreadable=True,
    )
    summary = run_full_embed_analysis(
        run_dir=args.run,
        checkpoint=args.checkpoint,
        cache_dir=args.cache,
        val_df=val_df,
        train_df=train_df,
    )
    print(f"[extract_probe_embeddings] bottleneck={summary.get('bottleneck')} -> {summary.get('embed_report')}")


if __name__ == "__main__":
    main()
