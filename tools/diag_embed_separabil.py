#!/usr/bin/env python3
"""Diagnóstico separabilidad embeddings DINO crudos (cache) — Fase 2C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from micorizae.phase_d_stage1 import split_by_image
    from micorizae.phase_d_stage1.gate_embed_analysis import _logreg_f1, _normalize_rows
    from micorizae.phase_d_stage1.gate_embed_cache import GateEmbedStore, embed_cache_paths

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "cache" / "dino_separability_diag.json")
    args = parser.parse_args()

    train_df, val_df, _ = split_by_image(
        lineages=["AM"], split_mode="fixed", train_splits=("train",), val_splits=("test",), exclude_unreadable=True
    )
    store = GateEmbedStore(embed_cache_paths(args.cache))
    lookup = store.lookup
    val_keys = set(zip(val_df["image_path"].astype(str), val_df["row"], val_df["col"]))
    hold = lookup[
        lookup.apply(lambda r: (str(r["image_path"]), int(r["row"]), int(r["col"])) in val_keys, axis=1)
    ].sample(n=min(5000, len(lookup)), random_state=0)
    idx = hold["embed_idx"].to_numpy()
    X = np.stack([store.embed[i].astype(np.float32) for i in idx])
    y = store.labels[idx].astype(np.int64)
    f1 = _logreg_f1(X, y)
    report = {"dino_holdout_logreg_macro_f1": f1, "n_tiles": len(y), "embed_dim": store.embed_dim}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[diag_embed_separabil] DINO logreg macro F1={f1:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
