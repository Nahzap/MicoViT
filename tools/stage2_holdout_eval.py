#!/usr/bin/env python3
"""Evaluación holdout Stage2 ViT-S2 (gold M+ en am_test)."""

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


def eval_holdout(
    *,
    lineage: str,
    ckpt_path: Path,
    gate_run_id: str,
    batch_size: int = 4,
) -> dict:
    import torch
    from sklearn.metrics import classification_report, f1_score

    from micorizae.common.io import read_table
    from micorizae.common.paths import get_paths
    from micorizae.phase_e_stage2.class_map import load_stage2_class_map
    from micorizae.phase_e_stage2.gpu_pipeline import filter_mplus_stage2, iter_image_batches_stage2, plan_epoch_stage2
    from micorizae.phase_e_stage2.models import build_branch_a_mc

    paths = get_paths()
    device = torch.device("cuda")
    df = read_table(paths.manifests / "tiles_index")
    test_df = filter_mplus_stage2(df[df["subset"] == "am_test"], lineage)
    if test_df.empty:
        raise RuntimeError("Sin tiles M+ Stage2 en am_test")

    cmap = load_stage2_class_map(lineage, only_present_in=test_df)
    test_df = test_df[test_df["stage2"].isin(cmap.classes)].copy().reset_index(drop=True)
    test_df["stage2_idx"] = cmap.encode(test_df["stage2"]).astype(int)

    model = build_branch_a_mc(cmap.num_classes, backbone_name="dinov2_vits14", freeze_backbone=True)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.to(device).eval()

    test_plan = plan_epoch_stage2(test_df, shuffle_images=False, shuffle_tiles_within_image=False)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for image_rel in sorted(test_df["image_path"].unique()):
            sub = test_df[test_df["image_path"] == image_rel]
            img_plan = plan_epoch_stage2(sub, shuffle_images=False, shuffle_tiles_within_image=False)
            for batch in iter_image_batches_stage2(
                img_plan, cmap, batch_size=batch_size, device=device,
            ):
                logits = model(batch.rgb)
                y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
                y_true.extend(batch.labels.cpu().tolist())
                del batch, logits
            torch.cuda.empty_cache()

    rep = classification_report(
        y_true, y_pred, target_names=list(cmap.classes), output_dict=True, zero_division=0,
    )
    recalls = {c: float(rep.get(c, {}).get("recall", 0.0)) for c in cmap.classes}
    return {
        "lineage": lineage,
        "gate_run_id": gate_run_id,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": st.get("epoch"),
        "checkpoint_val_f1": st.get("f1_macro"),
        "per_class_recall_checkpoint": st.get("per_class_recall", {}),
        "n_tiles": int(len(test_df)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": recalls,
        "per_class": {c: rep.get(c, {}) for c in cmap.classes},
        "classification_report": rep,
    }


def main() -> None:
    import config as user_config  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", default="AM")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gate-run-id", default=str(getattr(user_config, "STAGE2_GATE_RUN_ID", "")))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    ckpt = args.checkpoint or (
        ROOT / "models" / "checkpoints" / f"stage2_{args.lineage.lower()}"
        / f"stage2_{args.lineage.lower()}_branch_a_best.pt"
    )
    result = eval_holdout(
        lineage=args.lineage.upper(),
        ckpt_path=ckpt,
        gate_run_id=args.gate_run_id,
        batch_size=args.batch_size,
    )
    out_path = args.out or (ROOT / "outputs" / "reports" / f"stage2_holdout__gate{args.gate_run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
