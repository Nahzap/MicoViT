#!/usr/bin/env python3
"""Auditoría datos Stage2 M+ (fase S0 plan ViT-S2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def run_audit(*, lineage: str, out_dir: Path, gate_run_id: str) -> dict:
    import config as user_config  # type: ignore
    import pandas as pd
    from sklearn.metrics import f1_score

    from micorizae.common.io import read_table
    from micorizae.common.paths import get_paths
    from micorizae.phase_e_stage2.gpu_pipeline import filter_mplus_stage2, split_by_image_mplus

    paths = get_paths()
    df = read_table(paths.manifests / "tiles_index")
    mplus = filter_mplus_stage2(df, lineage)

    subsets_raw = str(getattr(user_config, "STAGE2_TRAIN_SUBSETS", "am_train") or "")
    subsets = [s.strip() for s in subsets_raw.split(",") if s.strip()] or None

    train_df, val_df, split_info, class_map = split_by_image_mplus(
        lineage=lineage, val_fraction=float(getattr(user_config, "STAGE2_VAL_FRACTION", 0.2)),
        subsets=subsets,
    )

    test_df = filter_mplus_stage2(df[df["subset"] == "am_test"], lineage)

    def counts(sub: pd.DataFrame) -> dict:
        return {str(k): int(v) for k, v in sub["stage2"].value_counts().to_dict().items()}

    n_train = len(train_df)
    n_hybrid = int((train_df["stage2"] == "Hybrid").sum())
    y_true = ["AMColonised"] * n_train
    y_pred = ["AMColonised"] * n_train
    trivial_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = {
        "lineage": lineage,
        "gate_run_id": gate_run_id,
        "plan_doc": "Docs/20260628_225244_PLAN_VIT_STAGE2_DISCRIMINADOR_MPLUS.md",
        "classes": list(class_map.classes),
        "total_mplus_stage2": int(len(mplus)),
        "counts_all": counts(mplus),
        "counts_train": counts(train_df),
        "counts_val": counts(val_df),
        "counts_test_holdout": counts(test_df),
        "split_info": split_info,
        "trivial_majority_f1": trivial_f1,
        "hybrid_ratio_train": float(n_hybrid / max(n_train, 1)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_counts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        f"# Stage2 data audit — {lineage}",
        "",
        f"- **gate_run_id:** `{gate_run_id}`",
        f"- **classes:** {', '.join(class_map.classes)}",
        f"- **total M+ Stage2:** {report['total_mplus_stage2']:,}",
        "",
        "## Conteos",
        "",
        "| split | " + " | ".join(class_map.classes) + " |",
        "|---|" + "|".join(["---:"] * len(class_map.classes)) + "|",
    ]
    for label, key in [
        ("all", "counts_all"),
        ("train", "counts_train"),
        ("val", "counts_val"),
        ("test", "counts_test_holdout"),
    ]:
        row = report[key]
        md.append("| " + label + " | " + " | ".join(str(row.get(c, 0)) for c in class_map.classes) + " |")
    md.extend([
        "",
        f"- trivial macro F1 (mayoría AMColonised): **{trivial_f1:.4f}**",
        "",
    ])
    (out_dir / "STAGE2_DATA_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> None:
    import config as user_config  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", default=str(getattr(user_config, "STAGE2_LINEAGE_DEFAULT", "AM")))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "stage2_audit")
    parser.add_argument("--gate-run-id", default=str(getattr(user_config, "STAGE2_GATE_RUN_ID", "")))
    args = parser.parse_args()
    rep = run_audit(lineage=args.lineage.upper(), out_dir=args.out, gate_run_id=args.gate_run_id)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
