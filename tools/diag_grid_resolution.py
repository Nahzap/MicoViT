"""Verificacion empirica: resolucion DINO + pooling espacial (grid/patch) vs mean-pool.

Pregunta del experimento
------------------------
El cache actual guarda SOLO el mean-pool de los patch tokens DINO (384-d) a
280 px (grilla 20x20). La distincion M+ (colonizada) vs M- (no colonizada)
depende de estructuras fungicas internas pequenas y dispersas (arbusculos,
vesiculas, hifas). El mean-pool promedia toda la espacialidad del tile y puede
"diluir" esa senal local.

Este script comprueba, sobre el PROTOCOLO REAL (logreg entrenada en imagenes
train -> evaluada en holdout, sin fuga entre imagenes), si:
  (a) subir la resolucion de entrada DINO (mas tokens), y/o
  (b) preservar localidad via grid-pooling o estadisticos (mean+max+std),
mejoran el recall M+/M- frente al mean-pool actual.

No toca el pipeline; opera decodificando JPEG y corriendo DINO congelado.
"""

from __future__ import annotations

import sys
import time
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

# Resoluciones DINO a comparar (multiplos de 14). 280 = produccion actual.
RESOLUTIONS = [224, 280, 392, 560]
SEG_FACTOR = 400 / 280  # mantiene la relacion seg/dino de produccion

TRAIN_PER_CLASS = 500
HOLDOUT_PER_CLASS = 350


def _grid_pool(patch_hw: np.ndarray, gy: int, gx: int) -> np.ndarray:
    """patch_hw: (H, W, C) -> concat de medias por celda (gy*gx*C,)."""
    rows = np.array_split(patch_hw, gy, axis=0)
    cells = []
    for rb in rows:
        for cb in np.array_split(rb, gx, axis=1):
            cells.append(cb.reshape(-1, cb.shape[-1]).mean(0))
    return np.concatenate(cells)


def _poolings(patch: np.ndarray, cls: np.ndarray, gh: int, gw: int) -> dict[str, np.ndarray]:
    """patch: (N, C) tokens; cls: (C,). Devuelve dict de vectores."""
    hw = patch.reshape(gh, gw, patch.shape[-1])
    mean = patch.mean(0)
    mx = patch.max(0)
    std = patch.std(0)
    return {
        "mean(384) [cache actual]": mean,
        "cls(384)": cls,
        "max(384)": mx,
        "mean+max+std(1152)": np.concatenate([mean, mx, std]),
        "grid2x2 mean(1536)": _grid_pool(hw, 2, 2),
        "grid3x3 mean(3456)": _grid_pool(hw, 3, 3),
    }


def _sample(tiles: pd.DataFrame, split: str, per_class: int, seed: int) -> pd.DataFrame:
    sub = tiles[tiles["split"] == split]
    sub = sub[sub["stage1"].isin(CLASSES)]
    if "density_tier" in sub.columns:
        sub = sub[sub["density_tier"].astype(str).isin(["base", "nan", "None"]) | sub["density_tier"].isna()]
    parts = []
    for c in CLASSES:
        pool = sub[sub["stage1"] == c]
        n = min(per_class, len(pool))
        if n:
            parts.append(pool.sample(n=n, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return out


@torch.inference_mode()
def _extract(sub: pd.DataFrame, dino, device) -> dict[int, dict[str, np.ndarray]]:
    """Para cada resolucion: matriz de features por pooling. Devuelve {res: {pool: X}}."""
    root = get_paths().root
    sub_sorted = sub.sort_values("image_path").reset_index(drop=True)
    n = len(sub_sorted)
    feats: dict[int, dict[str, list]] = {r: {} for r in RESOLUTIONS}
    cur_ip, arr = None, None
    t0 = time.perf_counter()
    for i in range(n):
        rec = sub_sorted.iloc[i]
        ip = str(rec["image_path"])
        if ip != cur_ip:
            arr = open_image_rgb(root / ip)
            cur_ip = ip
        ts = int(rec["tile_size"])
        crop = crop_tile_from_array(arr, row=int(rec["row"]), col=int(rec["col"]), tile_size=ts, pad_value=255)
        t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
        for res in RESOLUTIONS:
            seg = int(round(res * SEG_FACTOR))
            views = build_views_gpu(t, target_size=res, seg_target_size=seg)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = dino.forward_features(views.rgb)
            patch = out["x_norm_patchtokens"].float().cpu().numpy()[0]
            cls = out["x_norm_clstoken"].float().cpu().numpy()[0]
            g = res // 14
            pools = _poolings(patch, cls, g, g)
            for k, v in pools.items():
                feats[res].setdefault(k, []).append(v)
        if (i + 1) % 200 == 0:
            dt = time.perf_counter() - t0
            print(f"  extraido {i + 1}/{n}  ({dt:.0f}s, {(i+1)/dt:.1f} tiles/s)", flush=True)
    return {r: {k: np.stack(v) for k, v in d.items()} for r, d in feats.items()}


def _eval(Xtr, ytr, Xte, yte) -> tuple[float, dict, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, multi_class="multinomial")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    rec = {}
    for c, i in CLS_TO_IDX.items():
        m = yte == i
        rec[c] = float((pred[m] == i).mean()) if m.any() else float("nan")
    acc = float((pred == yte).mean())
    # macro F1
    from sklearn.metrics import f1_score
    mf1 = float(f1_score(yte, pred, average="macro"))
    return acc, rec, mf1


def _binary_mpm(Xtr, ytr, Xte, yte) -> tuple[float, float]:
    """M+ vs M- cross. Devuelve (recall M+, recall M-)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pos, neg = CLS_TO_IDX["Mplus"], CLS_TO_IDX["Mminus"]
    tr = np.isin(ytr, [pos, neg])
    te = np.isin(yte, [pos, neg])
    sc = StandardScaler().fit(Xtr[tr])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xtr[tr]), (ytr[tr] == pos).astype(int))
    pred = clf.predict(sc.transform(Xte[te]))
    yb = (yte[te] == pos).astype(int)
    rp = float((pred[yb == 1] == 1).mean()) if (yb == 1).any() else float("nan")
    rn = float((pred[yb == 0] == 0).mean()) if (yb == 0).any() else float("nan")
    return rp, rn


def main():
    device = torch.device("cuda")
    paths = get_paths()
    tiles = read_table(paths.manifests / "tiles_index")
    if "lineage" in tiles.columns:
        tiles = tiles[tiles["lineage"] == "AM"]
    train = _sample(tiles, "train", TRAIN_PER_CLASS, seed=0)
    holdo = _sample(tiles, "test", HOLDOUT_PER_CLASS, seed=10)
    print(f"Train sample: {len(train)}  " + str({c: int((train['stage1'] == c).sum()) for c in CLASSES}))
    print(f"Holdout sample: {len(holdo)}  " + str({c: int((holdo['stage1'] == c).sum()) for c in CLASSES}))

    branch = build_branch_a(backbone_name="dinov2_vits14", num_classes=4, freeze_backbone=True)
    dino = branch.backbone.model.to(device).eval()

    print("\nExtrayendo features train...", flush=True)
    Ftr = _extract(train, dino, device)
    ytr = train.sort_values("image_path")["stage1"].map(CLS_TO_IDX).to_numpy()
    print("Extrayendo features holdout...", flush=True)
    Fho = _extract(holdo, dino, device)
    yho = holdo.sort_values("image_path")["stage1"].map(CLS_TO_IDX).to_numpy()

    pool_names = list(next(iter(Ftr.values())).keys())
    print("\n" + "=" * 96)
    print("PROTOCOLO REAL: logreg cross train->holdout (sin fuga entre imagenes)")
    print("=" * 96)
    header = f"{'resolucion':>10} | {'pooling':<26} | {'acc':>5} {'mF1':>5} | {'Bg':>5} {'M-':>5} {'M+':>5} | {'M+vsM- M+':>9} {'M-':>5}"
    print(header)
    print("-" * len(header))
    results = []
    for res in RESOLUTIONS:
        g = res // 14
        for pn in pool_names:
            Xtr, Xte = Ftr[res][pn], Fho[res][pn]
            acc, rec, mf1 = _eval(Xtr, ytr, Xte, yho)
            rp, rn = _binary_mpm(Xtr, ytr, Xte, yho)
            results.append((res, pn, acc, mf1, rec, rp, rn))
            print(
                f"{res:>7}px | {pn:<26} | {acc:>5.3f} {mf1:>5.3f} | "
                f"{rec['Background']:>5.3f} {rec['Mminus']:>5.3f} {rec['Mplus']:>5.3f} | "
                f"{rp:>9.3f} {rn:>5.3f}"
            )
        print("-" * len(header))

    best = max(results, key=lambda r: r[3])
    base = next(r for r in results if r[0] == 280 and r[1].startswith("mean(384)"))
    print(f"\nBaseline produccion (280px mean-pool): macro_f1={base[3]:.3f}  M+={base[4]['Mplus']:.3f}  M-={base[4]['Mminus']:.3f}")
    print(f"Mejor config: {best[0]}px {best[1]}  macro_f1={best[3]:.3f}  M+={best[4]['Mplus']:.3f}  M-={best[4]['Mminus']:.3f}")
    print(f"Delta macro_f1: {best[3] - base[3]:+.3f}   Delta M+ recall: {best[4]['Mplus'] - base[4]['Mplus']:+.3f}")


if __name__ == "__main__":
    main()
