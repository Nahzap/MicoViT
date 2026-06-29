"""Compara estrategias de pooling DINOv2 para separar M+/M-/Bg (sin U2Net).

Gate AM usa mean-pool DINO sobre tile RGB (saliency_mask_pooling=false).
Este script mide separabilidad lineal (logreg balanceada, split 50/50) sobre
holdout, decodificando JPEG on-the-fly:

  1. mean    : mean de patch tokens (= cache actual)
  2. cls     : token CLS de DINOv2
  3. maxpool : max sobre patch tokens

Si todas dan M+ recall bajo -> el backbone congelado no codifica colonizacion
y conviene --finetune. Si mean/cls/maxpool separan bien -> el cuello esta en
el probe Slice-MS, no en el pooling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from micorizae.common.paths import get_paths  # noqa: E402
from micorizae.common.io import read_table  # noqa: E402
from micorizae.phase_b_tiling.tile_cutter import open_image_rgb, crop_tile_from_array  # noqa: E402
from micorizae.phase_c_views.gpu_transforms import build_views_gpu  # noqa: E402
from micorizae.phase_d_stage1 import build_branch_a  # noqa: E402

CLASSES = ["Background", "Mminus", "Mplus"]
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PER_CLASS = 500


def _dino_sizes():
    try:
        import importlib.util

        cfg_path = ROOT / "config.py"
        spec = importlib.util.spec_from_file_location("project_config", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return int(getattr(mod, "GATE_DINO_INPUT_SIZE", 280)), int(
            getattr(mod, "GATE_SEG_TARGET_SIZE", 400)
        )
    except Exception:  # noqa: BLE001
        return 280, 400


def _sample(device):
    paths = get_paths()
    tiles = read_table(paths.manifests / "tiles_index")
    tiles = tiles[tiles["lineage"] == "AM"] if "lineage" in tiles.columns else tiles
    ho = tiles[tiles["split"] == "test"].copy()
    ho = ho[ho["stage1"].isin(CLASSES)]
    parts = []
    for c in CLASSES:
        pool = ho[ho["stage1"] == c]
        n = min(PER_CLASS, len(pool))
        parts.append(pool.sample(n=n, random_state=0))
    sub = pd.concat(parts).sample(frac=1.0, random_state=1).reset_index(drop=True)
    print(f"Subconjunto holdout: {len(sub)} tiles  " +
          str({c: int((sub['stage1'] == c).sum()) for c in CLASSES}))
    return sub


@torch.inference_mode()
def _extract(sub: pd.DataFrame, device):
    dino_in, seg_in = _dino_sizes()
    print(f"DINO input={dino_in}px  (Gate AM: mean-pool, sin U2Net)")
    branch = build_branch_a(backbone_name="dinov2_vits14", num_classes=4, freeze_backbone=True)
    dino = branch.backbone.to(device).eval()
    raw = dino.model

    feats = {k: [] for k in ("mean", "cls", "maxpool")}
    labels = []
    root = get_paths().root
    sub_sorted = sub.sort_values("image_path").reset_index(drop=True)
    cur_ip = None
    arr = None
    for i in range(len(sub_sorted)):
        rec = sub_sorted.iloc[i]
        ip = str(rec["image_path"])
        if ip != cur_ip:
            arr = open_image_rgb(root / ip)
            cur_ip = ip
        ts = int(rec["tile_size"])
        crop = crop_tile_from_array(arr, row=int(rec["row"]), col=int(rec["col"]), tile_size=ts, pad_value=255)
        t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
        views = build_views_gpu(t, target_size=dino_in, seg_target_size=seg_in)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = raw.forward_features(views.rgb)
        patch = out["x_norm_patchtokens"].float()
        cls = out["x_norm_clstoken"].float()

        feats["mean"].append(patch.mean(1).cpu().numpy()[0])
        feats["cls"].append(cls.cpu().numpy()[0])
        feats["maxpool"].append(patch.max(1).values.cpu().numpy()[0])
        labels.append(CLS_TO_IDX[str(rec["stage1"])])
        if (i + 1) % 200 == 0:
            print(f"  extraido {i + 1}/{len(sub_sorted)}", flush=True)
    y = np.array(labels)
    return {k: np.stack(v) for k, v in feats.items()}, y


def _eval(X, y, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = len(idx) // 2
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="multinomial")
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    rec = {}
    for c, i in CLS_TO_IDX.items():
        m = y[te] == i
        rec[c] = float((pred[m] == i).mean()) if m.any() else float("nan")
    acc = float((pred == y[te]).mean())
    print(f"[{tag:12s}] acc={acc:.3f}  " + " ".join(f"{c}={rec[c]:.3f}" for c in CLASSES))


def main():
    device = torch.device("cuda")
    sub = _sample(device)
    feats, y = _extract(sub, device)
    print("\n--- Separabilidad lineal DINO (within-holdout, split 50/50) ---")
    for k in ("mean", "cls", "maxpool"):
        _eval(feats[k], y, k)


if __name__ == "__main__":
    main()
