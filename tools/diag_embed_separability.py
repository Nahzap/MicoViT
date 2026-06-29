"""Diagnostico de separabilidad de la cache de embeddings gate AM.

Pregunta central: el bajo recall de M+ en holdout, ?es por embeddings no
discriminativos o por domain shift entre imagenes train vs holdout?

Experimentos (clases Bg/M-/M+, sin Unknown):
  A. Within-holdout: logreg entrenada y evaluada en holdout (split aleatorio
     por TILE). Mide separabilidad maxima de los embeddings holdout.
  B. Within-holdout por IMAGEN: train/test split por imagen dentro del holdout.
     Mide transferencia entre imagenes (domain shift puro).
  C. Cross train->holdout: logreg entrenada en imagenes train, evaluada en
     holdout. Replica el protocolo real.
  D. kNN (k=15) cruzado train->holdout como referencia no-lineal.

Sin dependencia del modelo entrenado; opera directo sobre el memmap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from micorizae.common.paths import get_paths  # noqa: E402
from micorizae.common.io import read_table  # noqa: E402

CLASSES = ["Background", "Mminus", "Mplus"]
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _load_cache():
    cache_dir = ROOT / "cache"
    meta = json.loads((cache_dir / "gate_am_embeds_v3.meta.json").read_text(encoding="utf-8"))
    n, d = int(meta["n_tiles"]), int(meta["embed_dim"])
    embed = np.memmap(cache_dir / "gate_am_embeds_v3.embed.npy", dtype=np.float16, mode="r", shape=(n, d))
    labels = np.memmap(cache_dir / "gate_am_embeds_v3.label.npy", dtype=np.int8, mode="r", shape=(n,))
    lookup = pd.read_parquet(cache_dir / "gate_am_embeds_v3.lookup.parquet")
    return np.asarray(embed, dtype=np.float32), np.asarray(labels, dtype=np.int64), lookup, meta


def _split_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    paths = get_paths()
    tiles = read_table(paths.manifests / "tiles_index")
    tiles = tiles[["image_path", "row", "col", "split"]].copy()
    tiles["image_path"] = tiles["image_path"].astype(str)
    lk = lookup.copy()
    lk["image_path"] = lk["image_path"].astype(str)
    merged = lk.merge(tiles, on=["image_path", "row", "col"], how="left")
    return merged


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for c, i in CLS_TO_IDX.items():
        m = y_true == i
        out[c] = float((y_pred[m] == i).mean()) if m.any() else float("nan")
    return out


def _fit_eval(Xtr, ytr, Xte, yte, tag: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, multi_class="multinomial")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    rec = _per_class_recall(yte, pred)
    acc = float((pred == yte).mean())
    print(f"\n[{tag}] logreg  acc={acc:.3f}  recall " + " ".join(f"{c}={rec[c]:.3f}" for c in CLASSES))
    return rec


def _knn_eval(Xtr, ytr, Xte, yte, tag: str, k: int = 15):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(normalize(Xtr), ytr)
    pred = clf.predict(normalize(Xte))
    rec = _per_class_recall(yte, pred)
    acc = float((pred == yte).mean())
    print(f"[{tag}] knn{k}   acc={acc:.3f}  recall " + " ".join(f"{c}={rec[c]:.3f}" for c in CLASSES))
    return rec


def _balance(idx: np.ndarray, y: np.ndarray, per_class: int, rng) -> np.ndarray:
    keep = []
    for i in range(len(CLASSES)):
        pool = idx[y[idx] == i]
        if len(pool) == 0:
            continue
        n = min(per_class, len(pool))
        keep.append(rng.choice(pool, size=n, replace=False))
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def main():
    embed, labels, lookup, meta = _load_cache()
    print(f"Cache: {meta['n_tiles']} tiles, dim={meta['embed_dim']}, aug={meta.get('mplus_aug_variants')}")
    merged = _split_lookup(lookup)

    aug_id = merged["aug_id"].to_numpy() if "aug_id" in merged.columns else np.zeros(len(merged), dtype=int)
    split = merged["split"].astype(str).to_numpy()
    emb_idx = merged["embed_idx"].to_numpy()

    valid_lbl = np.isin(labels[emb_idx], [0, 1, 2])
    base = aug_id == 0  # solo tiles base para evaluacion limpia

    train_mask = (split == "train") & base & valid_lbl
    holdo_mask = (split == "test") & base & valid_lbl

    tr_rows = emb_idx[train_mask]
    ho_rows = emb_idx[holdo_mask]
    Xtr_all, ytr_all = embed[tr_rows], labels[tr_rows]
    Xho_all, yho_all = embed[ho_rows], labels[ho_rows]

    print(f"\nTrain tiles (base, 3-clase): {len(ytr_all):,}  | Holdout: {len(yho_all):,}")
    for name, y in [("train", ytr_all), ("holdout", yho_all)]:
        c = {cls: int((y == i).sum()) for cls, i in CLS_TO_IDX.items()}
        print(f"  {name}: {c}")

    rng = np.random.default_rng(0)

    # --- A. within-holdout, split aleatorio por tile (separabilidad maxima) ---
    ho_idx = np.arange(len(yho_all))
    rng.shuffle(ho_idx)
    cut = int(len(ho_idx) * 0.5)
    a_tr, a_te = ho_idx[:cut], ho_idx[cut:]
    a_tr_bal = _balance(a_tr, yho_all, 1000, rng)
    _fit_eval(Xho_all[a_tr_bal], yho_all[a_tr_bal], Xho_all[a_te], yho_all[a_te],
              "A within-holdout (tile split)")

    # --- B. within-holdout, split por IMAGEN (transferencia entre imagenes) ---
    ho_imgs = merged.loc[holdo_mask, "image_path"].to_numpy()
    uniq = sorted(set(ho_imgs))
    rng.shuffle(uniq)
    half = set(uniq[: max(1, len(uniq) // 2)])
    b_tr = np.array([i for i in range(len(yho_all)) if ho_imgs[i] in half])
    b_te = np.array([i for i in range(len(yho_all)) if ho_imgs[i] not in half])
    b_tr_bal = _balance(b_tr, yho_all, 1000, rng)
    print(f"\n  B imgs train={len(half)} test={len(uniq) - len(half)}")
    _fit_eval(Xho_all[b_tr_bal], yho_all[b_tr_bal], Xho_all[b_te], yho_all[b_te],
              "B within-holdout (image split)")

    # --- C. cross train->holdout (protocolo real) ---
    c_tr_bal = _balance(np.arange(len(ytr_all)), ytr_all, 3000, rng)
    _fit_eval(Xtr_all[c_tr_bal], ytr_all[c_tr_bal], Xho_all, yho_all,
              "C cross train->holdout")
    _knn_eval(Xtr_all[c_tr_bal], ytr_all[c_tr_bal], Xho_all, yho_all,
              "C cross train->holdout")

    # --- M+ vs M- binario cruzado (la distincion biologica clave) ---
    def _binary(Xtr, ytr, Xte, yte, pos, neg, tag):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        tr = np.isin(ytr, [pos, neg]); te = np.isin(yte, [pos, neg])
        if tr.sum() == 0 or te.sum() == 0:
            return
        sc = StandardScaler().fit(Xtr[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Xtr[tr]), (ytr[tr] == pos).astype(int))
        pred = clf.predict(sc.transform(Xte[te]))
        yb = (yte[te] == pos).astype(int)
        rec_pos = float((pred[yb == 1] == 1).mean()) if (yb == 1).any() else float("nan")
        rec_neg = float((pred[yb == 0] == 0).mean()) if (yb == 0).any() else float("nan")
        acc = float((pred == yb).mean())
        print(f"[{tag}] acc={acc:.3f} recall pos={rec_pos:.3f} neg={rec_neg:.3f}")

    print("\n--- Binario M+ vs M- (cross train->holdout) ---")
    _binary(Xtr_all[c_tr_bal], ytr_all[c_tr_bal], Xho_all, yho_all,
            CLS_TO_IDX["Mplus"], CLS_TO_IDX["Mminus"], "M+ vs M- cross")


if __name__ == "__main__":
    main()
