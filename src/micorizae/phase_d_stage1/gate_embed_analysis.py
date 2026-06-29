"""Análisis post-run de embeddings (hiperesfera Slice-MS) — pipeline publicable.

Genera figuras, JSON y EMBED_REPORT.md desde checkpoint + cache DINO.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..common.logging_utils import get_logger
from .gate_classes import GATE_CLASS_NAMES, decode_gate_indices
from .gate_metric_inference import ClassPrototypeBank, knn_predict

log = get_logger("phase_d.gate_embed_analysis")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _angular_deg(a: np.ndarray, b: np.ndarray) -> float:
    sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(sim)))


# ---------------------------------------------------------------------------
# Análisis angular pesado de la hiperesfera (radianes, coseno, muestra real)
# ---------------------------------------------------------------------------

# Solo clases operativas (Unknown se omite: sin tiles en cache v3).
_OPERATIVE_CLASSES = (0, 1, 2)


def _class_centroids(X_norm: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    """Centroide L2-normalizado por clase operativa (prototipo empírico en S^{d-1})."""
    cents: dict[int, np.ndarray] = {}
    for c in _OPERATIVE_CLASSES:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        v = X_norm[idx].mean(axis=0)
        nv = float(np.linalg.norm(v))
        if nv < 1e-8:
            continue
        cents[c] = v / nv
    return cents


def _descriptive_stats(sample: np.ndarray) -> dict[str, float]:
    """Estadística descriptiva de una muestra angular (radianes)."""
    if sample.size == 0:
        return {}
    return {
        "n": int(sample.size),
        "mean_rad": float(np.mean(sample)),
        "std_rad": float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0,
        "median_rad": float(np.median(sample)),
        "p05_rad": float(np.percentile(sample, 5)),
        "p25_rad": float(np.percentile(sample, 25)),
        "p75_rad": float(np.percentile(sample, 75)),
        "p95_rad": float(np.percentile(sample, 95)),
        "min_rad": float(np.min(sample)),
        "max_rad": float(np.max(sample)),
        "mean_deg": float(np.degrees(np.mean(sample))),
        "std_deg": float(np.degrees(np.std(sample, ddof=1))) if sample.size > 1 else 0.0,
        "median_deg": float(np.degrees(np.median(sample))),
    }


def _bootstrap_ci_mean(sample: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """IC 95% de la media por bootstrap (radianes)."""
    if sample.size < 2:
        return {}
    rng = np.random.default_rng(seed)
    n = sample.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = sample[rng.integers(0, n, size=n)].mean()
    return {
        "ci95_low_rad": float(np.percentile(means, 2.5)),
        "ci95_high_rad": float(np.percentile(means, 97.5)),
        "ci95_low_deg": float(np.degrees(np.percentile(means, 2.5))),
        "ci95_high_deg": float(np.degrees(np.percentile(means, 97.5))),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Tamaño de efecto (d de Cohen) entre dos muestras angulares."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / pooled)


def _sample_pairwise_angles(
    X_norm: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    n_pairs: int,
    seed: int,
    same: bool,
) -> np.ndarray:
    """Muestra n_pairs ángulos (radianes) entre filas de dos grupos de índices."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ia = idx_a[rng.integers(0, len(idx_a), size=n_pairs)]
    ib = idx_b[rng.integers(0, len(idx_b), size=n_pairs)]
    if same:
        valid = ia != ib
        ia, ib = ia[valid], ib[valid]
        if ia.size == 0:
            return np.empty(0, dtype=np.float64)
    sims = np.einsum("ij,ij->i", X_norm[ia], X_norm[ib])
    sims = np.clip(sims, -1.0, 1.0)
    return np.arccos(sims)


def _angular_separation_sample(
    X_norm: np.ndarray,
    y: np.ndarray,
    *,
    max_pairs_per_group: int = 40000,
    seed: int = 0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Separación angular intra/inter clase como muestra estadística real (radianes).

    Compara la distribución de distancias angulares dentro de cada clase (intra)
    contra entre clases distintas (inter) y cuantifica la separación con tamaño de
    efecto e índice de separación (margen entre medias / dispersión combinada).
    """
    by_class = {c: np.where(y == c)[0] for c in _OPERATIVE_CLASSES if np.any(y == c)}
    intra_all: list[np.ndarray] = []
    intra_per_class: dict[str, dict] = {}
    for c, idx in by_class.items():
        s = _sample_pairwise_angles(
            X_norm, idx, idx, n_pairs=max_pairs_per_group, seed=seed + c, same=True
        )
        if s.size:
            intra_all.append(s)
            stats = _descriptive_stats(s)
            stats.update(_bootstrap_ci_mean(s, seed=seed + c))
            intra_per_class[decode_gate_indices(np.array([c]))[0]] = stats

    inter_all: list[np.ndarray] = []
    inter_per_pair: dict[str, dict] = {}
    cls_list = sorted(by_class.keys())
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            ca, cb = cls_list[i], cls_list[j]
            s = _sample_pairwise_angles(
                X_norm, by_class[ca], by_class[cb],
                n_pairs=max_pairs_per_group, seed=seed + 100 * ca + cb, same=False,
            )
            if s.size:
                inter_all.append(s)
                name = f"{decode_gate_indices(np.array([ca]))[0]}__{decode_gate_indices(np.array([cb]))[0]}"
                stats = _descriptive_stats(s)
                stats.update(_bootstrap_ci_mean(s, seed=seed + 100 * ca + cb))
                inter_per_pair[name] = stats

    intra = np.concatenate(intra_all) if intra_all else np.empty(0)
    inter = np.concatenate(inter_all) if inter_all else np.empty(0)

    result: dict[str, Any] = {
        "units": "radians (mean_deg/median_deg en grados)",
        "intra_class": intra_per_class,
        "inter_class_pairs": inter_per_pair,
        "intra_pooled": _descriptive_stats(intra),
        "inter_pooled": _descriptive_stats(inter),
    }
    if intra.size and inter.size:
        result["intra_pooled"].update(_bootstrap_ci_mean(intra, seed=seed))
        result["inter_pooled"].update(_bootstrap_ci_mean(inter, seed=seed + 1))
        cohen = _cohens_d(intra, inter)
        denom = np.std(intra, ddof=1) + np.std(inter, ddof=1)
        sep_index = float((np.mean(inter) - np.mean(intra)) / denom) if denom > 1e-12 else float("nan")
        result["separation"] = {
            "cohens_d_intra_vs_inter": cohen,
            "separation_index": sep_index,
            "mean_gap_rad": float(np.mean(inter) - np.mean(intra)),
            "mean_gap_deg": float(np.degrees(np.mean(inter) - np.mean(intra))),
        }
        try:
            from scipy.stats import mannwhitneyu

            u, p = mannwhitneyu(intra, inter, alternative="less")
            result["separation"]["mannwhitney_u"] = float(u)
            result["separation"]["mannwhitney_p"] = float(p)
        except Exception:
            pass
    return result, intra, inter


def _prototype_geometry(
    X_norm: np.ndarray,
    y: np.ndarray,
    proto: Optional[ClassPrototypeBank],
) -> dict[str, Any]:
    """Geometría de prototipos: matriz angular (rad) + distancia coseno entre anclas.

    Usa los prototipos reales del modelo si existen; si no, centroides empíricos.
    """
    if proto is not None and proto.is_ready(list(_OPERATIVE_CLASSES)):
        class_idx = list(_OPERATIVE_CLASSES)
        rep = proto.representative_prototypes().detach().cpu().numpy()[class_idx]
        P = _normalize_rows(rep)
        source = "model_prototypes"
    else:
        cents = _class_centroids(X_norm, y)
        if len(cents) < 2:
            return {}
        class_idx = [c for c in _OPERATIVE_CLASSES if c in cents]
        P = np.stack([cents[c] for c in class_idx])
        source = "empirical_centroids"
    names = [decode_gate_indices(np.array([c]))[0] for c in class_idx]

    sim = np.clip(P @ P.T, -1.0, 1.0)
    ang_rad = np.arccos(sim)
    cos_dist = 1.0 - sim
    return {
        "source": source,
        "classes": names,
        "class_idx": class_idx,
        "angular_matrix_rad": ang_rad.tolist(),
        "angular_matrix_deg": np.degrees(ang_rad).tolist(),
        "cosine_distance_matrix": cos_dist.tolist(),
        "cosine_similarity_matrix": sim.tolist(),
        "_P": P,
    }


def _angular_margin_to_prototypes(
    X_norm: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
    proto_classes: list[int],
) -> dict[str, Any]:
    """Margen angular por punto: ángulo a su prototipo vs prototipo rival más cercano.

    margin > 0 => el punto está angularmente más cerca de su clase (bien orientado).
    """
    sims = np.clip(X_norm @ P.T, -1.0, 1.0)  # (N, C)
    ang = np.arccos(sims)  # ángulo a cada prototipo
    out: dict[str, Any] = {"units": "radians", "per_class": {}}
    margins_all = []
    for ci, c in enumerate(proto_classes):
        mask = y == c
        if not mask.any():
            continue
        own = ang[mask, ci]
        other = np.delete(ang[mask], ci, axis=1)
        nearest_other = other.min(axis=1)
        margin = nearest_other - own  # >0 bien clasificado angularmente
        margins_all.append(margin)
        out["per_class"][decode_gate_indices(np.array([c]))[0]] = {
            "n": int(mask.sum()),
            "angle_to_own_mean_rad": float(own.mean()),
            "angle_to_own_mean_deg": float(np.degrees(own.mean())),
            "nearest_rival_mean_rad": float(nearest_other.mean()),
            "margin_mean_rad": float(margin.mean()),
            "margin_mean_deg": float(np.degrees(margin.mean())),
            "margin_median_rad": float(np.median(margin)),
            "frac_positive_margin": float((margin > 0).mean()),
        }
    if margins_all:
        allm = np.concatenate(margins_all)
        out["pooled"] = {
            "margin_mean_rad": float(allm.mean()),
            "margin_mean_deg": float(np.degrees(allm.mean())),
            "frac_positive_margin": float((allm > 0).mean()),
        }
    return out


def _cluster_separation_metrics(X_norm: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calidad de clusters en la hiperesfera (coseno)."""
    out: dict[str, float] = {}
    mask = np.isin(y, _OPERATIVE_CLASSES)
    Xm, ym = X_norm[mask], y[mask]
    if len(np.unique(ym)) < 2:
        return out
    if len(Xm) > 6000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(Xm), size=6000, replace=False)
        Xm, ym = Xm[sel], ym[sel]
    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        out["silhouette_cosine"] = float(silhouette_score(Xm, ym, metric="cosine"))
        out["davies_bouldin"] = float(davies_bouldin_score(Xm, ym))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, ym))
    except Exception as e:
        log.warning(f"[Gate embed] cluster metrics skip: {e}")
    return out


def _confusion_matrices(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Matriz de confusión cruda + normalizada por fila (recall) sobre clases operativas."""
    classes = list(_OPERATIVE_CLASSES)
    names = [decode_gate_indices(np.array([c]))[0] for c in classes]
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    cidx = {c: i for i, c in enumerate(classes)}
    for yt, yp in zip(y, pred):
        if yt in cidx and yp in cidx:
            cm[cidx[yt], cidx[yp]] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum > 0)
    return {
        "classes": names,
        "raw": cm.tolist(),
        "normalized_recall": cm_norm.tolist(),
    }


def _plot_confusion(cm_data: dict, out_png: Path) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    names = cm_data["classes"]
    cm = np.array(cm_data["normalized_recall"])
    raw = np.array(cm_data["raw"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real (gold)")
    ax.set_title("Matriz de confusión holdout (norm. por recall)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j, i, f"{cm[i, j]:.2f}\n({raw[i, j]})",
                ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_angular_distribution(intra: np.ndarray, inter: np.ndarray, out_png: Path) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.pi, 90)
    if intra.size:
        ax.hist(intra, bins=bins, density=True, alpha=0.55, color="#4C78A8",
                label=f"intra-clase (μ={np.degrees(intra.mean()):.1f}°)")
        ax.axvline(intra.mean(), color="#26456b", ls="--", lw=1.2)
    if inter.size:
        ax.hist(inter, bins=bins, density=True, alpha=0.55, color="#F58518",
                label=f"inter-clase (μ={np.degrees(inter.mean()):.1f}°)")
        ax.axvline(inter.mean(), color="#9c5410", ls="--", lw=1.2)
    ax.set_xlabel("Distancia angular (radianes)")
    ax.set_ylabel("Densidad")
    ax.set_title("Separación angular en la hiperesfera (muestra estadística)")
    sec = ax.secondary_xaxis("top", functions=(np.degrees, np.radians))
    sec.set_xlabel("grados")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_prototype_matrix(geom: dict, out_png: Path) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    names = geom["classes"]
    ang = np.array(geom["angular_matrix_deg"])
    cosd = np.array(geom["cosine_distance_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, title, fmt, cmap in (
        (axes[0], ang, "Separación angular prototipos (grados)", ".1f", "viridis"),
        (axes[1], cosd, "Distancia coseno entre prototipos", ".3f", "magma"),
    ):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticklabels(names)
        ax.set_title(title, fontsize=10)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                        color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Geometría de anclas ({geom.get('source', '?')})", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_tsne(df: pd.DataFrame, out_png: Path, *, max_points: int = 4000) -> bool:
    """t-SNE coseno: clusters por clase + panel de aciertos/errores."""
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=0)
    X = _normalize_rows(np.stack(df["embed"].apply(np.array).to_numpy()))
    y = df["label"].to_numpy()
    correct = (df["label_idx"].to_numpy() == df["pred_idx"].to_numpy())
    try:
        from sklearn.manifold import TSNE

        perplexity = float(min(30, max(5, len(X) // 100)))
        xy = TSNE(
            n_components=2, metric="cosine", init="random",
            perplexity=perplexity, random_state=0,
        ).fit_transform(X)
    except Exception as e:
        log.warning(f"[Gate embed] t-SNE skip: {e}")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for cls in sorted(set(y)):
        m = y == cls
        axes[0].scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.6, label=cls)
    axes[0].set_title("t-SNE (coseno) — dominios de clase en S^127")
    axes[0].legend(markerscale=2, fontsize=8)
    axes[1].scatter(xy[correct, 0], xy[correct, 1], s=9, alpha=0.5, color="#54A24B", label="acierto")
    axes[1].scatter(xy[~correct, 0], xy[~correct, 1], s=12, alpha=0.7, color="#E45756", label="error")
    axes[1].set_title("t-SNE — aciertos vs errores del prototipo")
    axes[1].legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def _load_probe_and_store(
    checkpoint: Path,
    cache_dir: Path,
    cfg: Any,
    device: torch.device,
):
    import importlib.util

    from .gate4.config import gate4_config_from_module
    from .gate4.probe_model import build_gate_slice_probe
    from .gate_embed_cache import GateEmbedStore, cache_paths_for_root
    from .gate_probe_input import probe_in_dim_from_attention_meta

    if cfg is None:
        from ..common.paths import get_paths

        cfg_path = get_paths().root / "config.py"
        spec = importlib.util.spec_from_file_location("project_config", cfg_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)

    g4 = gate4_config_from_module(cfg)
    root = Path(cache_dir)
    if root.name == "cache":
        root = root.parent
    cpaths = cache_paths_for_root(root)
    store = GateEmbedStore(cpaths)
    in_dim = probe_in_dim_from_attention_meta(store.meta.get("attention"), store.embed_dim)
    use_attn = in_dim > store.embed_dim

    model = build_gate_slice_probe(
        in_dim=in_dim,
        embed_dim=g4.embed_dim,
        num_slices=g4.num_slices,
        num_classes=len(GATE_CLASS_NAMES),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    proto = None
    if "prototype_bank" in ckpt:
        from .gate_metric_inference import prototype_bank_from_gate4

        proto = prototype_bank_from_gate4(g4, num_classes=len(GATE_CLASS_NAMES), device=device)
        proto.load_state_dict(ckpt["prototype_bank"])

    return model, store, g4, proto, use_attn, ckpt


def _filter_lookup(lookup: pd.DataFrame, val_df: pd.DataFrame | None, train_df: pd.DataFrame | None, split: str) -> pd.DataFrame:
    if split == "holdout" and val_df is not None:
        keys = set(zip(val_df["image_path"].astype(str), val_df["row"].astype(int), val_df["col"].astype(int)))
        mask = lookup.apply(
            lambda r: (str(r["image_path"]), int(r["row"]), int(r["col"])) in keys, axis=1
        )
        return lookup[mask]
    if split == "train" and train_df is not None:
        keys = set(zip(train_df["image_path"].astype(str), train_df["row"].astype(int), train_df["col"].astype(int)))
        mask = lookup.apply(
            lambda r: (str(r["image_path"]), int(r["row"]), int(r["col"])) in keys, axis=1
        )
        return lookup[mask]
    return lookup


def extract_embedding_table(
    *,
    model: torch.nn.Module,
    store: Any,
    lookup: pd.DataFrame,
    device: torch.device,
    proto: Optional[ClassPrototypeBank],
    use_attn: bool,
    max_tiles: int = 0,
) -> pd.DataFrame:
    if max_tiles > 0 and len(lookup) > max_tiles:
        lookup = lookup.sample(n=max_tiles, random_state=0)

    rows: list[dict] = []
    batch_size = 256
    indices = lookup["embed_idx"].to_numpy(dtype=np.int64)
    dino_dim = store.embed_dim

    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        feat, labels = store.read_batch(idx, device)
        attn = store.read_attn_batch(idx, device)
        with torch.no_grad():
            if use_attn and attn is not None:
                pooled = attn.mean(dim=(2, 3)) if attn.ndim == 4 else attn
                x = torch.cat([feat, pooled.float()], dim=-1)
            else:
                x = feat
            embed = model.encode(x)
            if proto is not None and proto.is_ready():
                logits = proto.logits(embed)
            else:
                logits = model.gate_head(embed)
            dino_np = feat.float().cpu().numpy()

        embed_np = embed.float().cpu().numpy()
        pred = logits.argmax(dim=-1).cpu().numpy()
        conf = torch.softmax(logits.float(), dim=-1).max(dim=-1).values.cpu().numpy()

        for i, eidx in enumerate(idx):
            lk = lookup[lookup["embed_idx"] == eidx].iloc[0]
            e = embed_np[i]
            chunks = embed[i].chunk(4, dim=-1)
            slice_vecs = [F.normalize(c, dim=-1).float().cpu().numpy() for c in chunks]
            rows.append(
                {
                    "embed_idx": int(eidx),
                    "image_path": str(lk.get("image_path", "")),
                    "row": int(lk.get("row", 0)),
                    "col": int(lk.get("col", 0)),
                    "aug_id": str(lk.get("aug_id", "")),
                    "label_idx": int(labels[i].item()),
                    "label": decode_gate_indices(np.array([labels[i].item()]))[0],
                    "pred_idx": int(pred[i]),
                    "pred": decode_gate_indices(np.array([pred[i]]))[0],
                    "confidence": float(conf[i]),
                    "embed": e.tolist(),
                    "dino_embed": dino_np[i].tolist(),
                    "slice_0": slice_vecs[0].tolist(),
                    "slice_1": slice_vecs[1].tolist(),
                    "slice_2": slice_vecs[2].tolist(),
                    "slice_3": slice_vecs[3].tolist(),
                }
            )
    return pd.DataFrame(rows)


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    active = sorted(set(y.tolist()))
    return float(f1_score(y, pred, average="macro", labels=active, zero_division=0))


def _logreg_f1(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    if len(np.unique(y)) < 2:
        return 0.0
    Xn = _normalize_rows(X.astype(np.float32))
    lr = LogisticRegression(max_iter=800, multi_class="multinomial")
    lr.fit(Xn, y)
    pred = lr.predict(Xn)
    active = sorted(set(y.tolist()))
    return float(f1_score(y, pred, average="macro", labels=active, zero_division=0))


def _pca_intrinsic_dim(X: np.ndarray, threshold: float = 0.95) -> dict:
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    evals = evals[evals > 1e-9]
    if len(evals) == 0:
        return {"dims_95pct": 0, "dims_99pct": 0, "n_components": 0}
    cum = np.cumsum(evals) / evals.sum()
    d95 = int(np.searchsorted(cum, threshold) + 1)
    d99 = int(np.searchsorted(cum, 0.99) + 1)
    return {"dims_95pct": d95, "dims_99pct": d99, "n_components": len(evals)}


def _angular_margins(df: pd.DataFrame) -> dict:
    X = np.stack(df["embed"].apply(np.array).to_numpy())
    X = _normalize_rows(X)
    y = df["label_idx"].to_numpy()
    margins: dict[str, list[float]] = {"intra": [], "inter_mminus_mplus": [], "inter_mplus_bg": []}
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        cent = _normalize_rows(X[idx].mean(axis=0, keepdims=True))[0]
        for i in idx:
            margins["intra"].append(_angular_deg(X[i], cent))
    mminus, mplus, bg = 1, 2, 0
    idx_m = np.where(y == mminus)[0]
    idx_p = np.where(y == mplus)[0]
    idx_b = np.where(y == bg)[0]
    if len(idx_m) and len(idx_p):
        for i in idx_p[: min(500, len(idx_p))]:
            j = idx_m[i % len(idx_m)]
            margins["inter_mminus_mplus"].append(_angular_deg(X[i], X[j]))
    if len(idx_b) and len(idx_p):
        for i in idx_p[: min(500, len(idx_p))]:
            j = idx_b[i % len(idx_b)]
            margins["inter_mplus_bg"].append(_angular_deg(X[i], X[j]))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} if v else {} for k, v in margins.items()}


def _plot_umap(df: pd.DataFrame, out_png: Path, *, max_points: int = 3000) -> bool:
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=0)
    X = _normalize_rows(np.stack(df["embed"].apply(np.array).to_numpy()))
    y = df["label"].to_numpy()
    method = "PCA"
    try:
        import umap

        xy = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=0,
            n_jobs=1,
        ).fit_transform(X)
        method = "UMAP"
    except Exception:
        try:
            from sklearn.decomposition import PCA

            xy = PCA(n_components=2, random_state=0).fit_transform(X)
            method = "PCA"
        except Exception as e:
            log.warning(f"[Gate embed] UMAP/PCA skip: {e}")
            return False

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in sorted(set(y)):
        m = y == cls
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, label=cls)
    ax.set_title(f"{method} — probe embed S^127 (holdout)")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def _plot_angular_hist(margins: dict, out_png: Path) -> None:
    from .gate_pretrain_viz import ensure_matplotlib_agg

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    labels, data = [], []
    for k in ("intra", "inter_mminus_mplus", "inter_mplus_bg"):
        if k in margins and margins[k]:
            labels.append(k)
            data.append(margins[k].get("mean", 0))
    if data:
        ax.bar(labels, data, color=["#4C78A8", "#F58518", "#72B7B2"])
        ax.set_ylabel("Mean angle (deg)")
        ax.set_title("Margen angular (probe embed)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _per_image_recall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for img, sub in df.groupby("image_path"):
        y = sub["label_idx"].to_numpy()
        pred = sub["pred_idx"].to_numpy()
        row: dict = {"image_path": img, "n_tiles": len(sub)}
        for cls in ("Background", "Mminus", "Mplus"):
            from .gate_classes import GATE_CLASS_TO_IDX

            ci = GATE_CLASS_TO_IDX[cls]
            mask = y == ci
            row[f"recall_{cls}"] = float((pred[mask] == ci).mean()) if mask.any() else float("nan")
        row["macro_f1"] = _macro_f1(y, pred)
        rows.append(row)
    return pd.DataFrame(rows)


def _bottleneck_verdict(dino_f1: float, probe_f1: float, lr_f1: float, per_image_std: float) -> dict:
    verdict = "probe"
    if dino_f1 >= probe_f1 - 0.02:
        verdict = "dino"
    if per_image_std > 0.12:
        verdict = "domain_shift"
    if lr_f1 - probe_f1 > 0.05:
        verdict = "calibration"
    return {
        "bottleneck": verdict,
        "dino_logreg_macro_f1": dino_f1,
        "probe_macro_f1": probe_f1,
        "probe_logreg_macro_f1": lr_f1,
        "per_image_recall_std": per_image_std,
    }


def run_full_embed_analysis(
    *,
    run_dir: Path,
    checkpoint: Path,
    cache_dir: Path,
    val_df: pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
    cfg: Any = None,
    max_holdout: int = 8000,
    max_train: int = 4000,
) -> dict[str, Any]:
    """Pipeline completo Fase 2 → analysis/ + figures/."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    analysis = run_dir / "analysis"
    figures = analysis / "figures"
    analysis.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    model, store, g4, proto, use_attn, ckpt = _load_probe_and_store(checkpoint, cache_dir, cfg, device)
    lookup_all = store.lookup

    df_hold = extract_embedding_table(
        model=model,
        store=store,
        lookup=_filter_lookup(lookup_all, val_df, train_df, "holdout"),
        device=device,
        proto=proto,
        use_attn=use_attn,
        max_tiles=max_holdout,
    )
    df_train = extract_embedding_table(
        model=model,
        store=store,
        lookup=_filter_lookup(lookup_all, val_df, train_df, "train"),
        device=device,
        proto=proto,
        use_attn=use_attn,
        max_tiles=max_train,
    )

    pq_h = analysis / "embeddings_holdout.parquet"
    pq_t = analysis / "embeddings_train.parquet"
    df_hold.drop(columns=["embed", "dino_embed", "slice_0", "slice_1", "slice_2", "slice_3"]).to_parquet(pq_h, index=False)
    df_train.drop(columns=["embed", "dino_embed", "slice_0", "slice_1", "slice_2", "slice_3"]).to_parquet(pq_t, index=False)
    np.save(analysis / "embeddings_holdout_vectors.npy", np.stack(df_hold["embed"].apply(np.array)))
    np.save(analysis / "embeddings_train_vectors.npy", np.stack(df_train["embed"].apply(np.array)))

    y = df_hold["label_idx"].to_numpy()
    pred = df_hold["pred_idx"].to_numpy()
    probe_f1 = _macro_f1(y, pred)

    X_probe = np.stack(df_hold["embed"].apply(np.array))
    X_dino = np.stack(df_hold["dino_embed"].apply(np.array))
    lr_probe_f1 = _logreg_f1(X_probe, y)
    dino_f1 = _logreg_f1(X_dino, y)

    slice_metrics = {}
    for s in range(4):
        Xs = np.stack(df_hold[f"slice_{s}"].apply(np.array))
        slice_metrics[f"slice_{s}"] = {"logreg_macro_f1": _logreg_f1(Xs, y)}

    Xn = _normalize_rows(X_probe)
    ref_X = _normalize_rows(np.stack(df_train["embed"].apply(np.array)))
    ref_y = df_train["label_idx"].to_numpy()
    knn_pred = knn_predict(Xn, ref_X, ref_y, k=5)
    knn_f1 = _macro_f1(y, knn_pred)

    intrinsic = _pca_intrinsic_dim(X_probe)
    margins = _angular_margins(df_hold)
    per_img = _per_image_recall(df_hold)
    per_img.to_csv(analysis / "per_image_recall.csv", index=False)
    per_image_std = float(per_img[[c for c in per_img.columns if c.startswith("recall_")]].stack().std())

    # --- Análisis angular pesado de la hiperesfera (radianes, coseno, muestra real) ---
    Xn_hold = _normalize_rows(X_probe)
    sep_report, intra_sample, inter_sample = _angular_separation_sample(Xn_hold, y)
    geom = _prototype_geometry(Xn_hold, y, proto)
    margin_report = (
        _angular_margin_to_prototypes(Xn_hold, y, geom["_P"], geom["class_idx"]) if geom else {}
    )
    cluster_metrics = _cluster_separation_metrics(Xn_hold, y)
    confusion = _confusion_matrices(y, pred)

    hypersphere = {
        "n_holdout_tiles": int(len(df_hold)),
        "embed_dim": int(X_probe.shape[1]),
        "angular_separation_sample": sep_report,
        "angular_margin_to_prototypes": margin_report,
        "cluster_separation": cluster_metrics,
        "confusion_matrix": confusion,
    }
    if geom:
        geom_out = {k: v for k, v in geom.items() if k != "_P"}
        hypersphere["prototype_geometry"] = geom_out
    (analysis / "hypersphere_angular.json").write_text(
        json.dumps(hypersphere, indent=2), encoding="utf-8"
    )

    pd.DataFrame(confusion["raw"], index=confusion["classes"], columns=confusion["classes"]).to_csv(
        analysis / "confusion_matrix_holdout.csv"
    )

    _plot_umap(df_hold, figures / "umap_probe_holdout.png")
    _plot_angular_hist(margins, figures / "angular_margin_hist.png")
    _plot_angular_distribution(intra_sample, inter_sample, figures / "angular_distance_distribution.png")
    _plot_confusion(confusion, figures / "confusion_matrix_holdout.png")
    if geom:
        _plot_prototype_matrix(geom, figures / "prototype_geometry_matrix.png")
    tsne_ok = _plot_tsne(df_hold, figures / "tsne_probe_holdout.png")

    bottleneck = _bottleneck_verdict(dino_f1, probe_f1, lr_probe_f1, per_image_std)
    bottleneck["slice_metrics"] = slice_metrics
    bottleneck["knn_macro_f1"] = knn_f1
    bottleneck["intrinsic_dim"] = intrinsic
    bottleneck["angular_margins"] = margins
    (analysis / "embed_bottleneck.json").write_text(json.dumps(bottleneck, indent=2), encoding="utf-8")
    (analysis / "intrinsic_dim_report.json").write_text(json.dumps(intrinsic, indent=2), encoding="utf-8")
    (analysis / "slice_separability.json").write_text(json.dumps(slice_metrics, indent=2), encoding="utf-8")

    dino_diag = {
        "dino_logreg_macro_f1": dino_f1,
        "probe_macro_f1": probe_f1,
        "probe_logreg_macro_f1": lr_probe_f1,
        "knn_macro_f1": knn_f1,
        "loss_type": ckpt.get("loss_type", "unknown"),
        "metric_inference": ckpt.get("metric_inference", "prototype"),
    }
    (analysis / "dino_separability.json").write_text(json.dumps(dino_diag, indent=2), encoding="utf-8")

    md_lines = [
        f"# EMBED_REPORT — {run_dir.name}",
        "",
        f"**Generado:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Checkpoint:** `{checkpoint}`",
        f"**Loss (ckpt):** {ckpt.get('loss_type', '?')} | **Inferencia:** {ckpt.get('metric_inference', 'prototype')}",
        "",
        "## Métricas holdout",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| probe macro F1 (prototipos) | {probe_f1:.4f} |",
        f"| logreg probe embed | {lr_probe_f1:.4f} |",
        f"| logreg DINO 384-d | {dino_f1:.4f} |",
        f"| kNN (train→holdout, k=5) | {knn_f1:.4f} |",
        "",
        "## Dimensión intrínseca (PCA probe)",
        "",
        f"- dims 95% varianza: **{intrinsic['dims_95pct']}**",
        f"- dims 99% varianza: **{intrinsic['dims_99pct']}**",
        "",
        "## Separabilidad por slice (logreg in-sample)",
        "",
    ]
    for sk, sv in slice_metrics.items():
        md_lines.append(f"- {sk}: macro F1 = {sv['logreg_macro_f1']:.4f}")

    # --- Sección hiperesfera ---
    md_lines += [
        "",
        "## Hiperesfera S^{d-1} — análisis angular",
        "",
    ]
    if geom:
        gnames = geom["classes"]
        md_lines += [
            f"**Geometría de anclas** (`{geom['source']}`) — separación angular entre prototipos:",
            "",
            "| | " + " | ".join(gnames) + " |",
            "|" + "---|" * (len(gnames) + 1),
        ]
        ang_deg = geom["angular_matrix_deg"]
        for i, nm in enumerate(gnames):
            md_lines.append(
                "| **" + nm + "** | " + " | ".join(f"{ang_deg[i][j]:.1f}°" for j in range(len(gnames))) + " |"
            )
        md_lines.append("")
        md_lines.append("Distancia coseno entre prototipos:")
        md_lines.append("")
        md_lines.append("| | " + " | ".join(gnames) + " |")
        md_lines.append("|" + "---|" * (len(gnames) + 1))
        cosd = geom["cosine_distance_matrix"]
        for i, nm in enumerate(gnames):
            md_lines.append(
                "| **" + nm + "** | " + " | ".join(f"{cosd[i][j]:.3f}" for j in range(len(gnames))) + " |"
            )
        md_lines.append("")

    sep = sep_report.get("separation", {})
    ip = sep_report.get("intra_pooled", {})
    xp = sep_report.get("inter_pooled", {})
    if ip and xp:
        md_lines += [
            "**Separación angular como muestra estadística** (intra vs inter clase):",
            "",
            "| Distribución | Media (rad) | Media (°) | σ (rad) | Mediana (°) | n |",
            "|--------------|-------------|-----------|---------|-------------|---|",
            f"| intra-clase | {ip.get('mean_rad', 0):.4f} | {ip.get('mean_deg', 0):.2f} | {ip.get('std_rad', 0):.4f} | {ip.get('median_deg', 0):.2f} | {ip.get('n', 0):,} |",
            f"| inter-clase | {xp.get('mean_rad', 0):.4f} | {xp.get('mean_deg', 0):.2f} | {xp.get('std_rad', 0):.4f} | {xp.get('median_deg', 0):.2f} | {xp.get('n', 0):,} |",
            "",
            f"- Brecha de medias: **{sep.get('mean_gap_rad', 0):.4f} rad ({sep.get('mean_gap_deg', 0):.2f}°)**",
            f"- d de Cohen (intra vs inter): **{sep.get('cohens_d_intra_vs_inter', float('nan')):.3f}**",
            f"- Índice de separación: **{sep.get('separation_index', float('nan')):.3f}**",
        ]
        if "mannwhitney_p" in sep:
            md_lines.append(f"- Mann–Whitney U (intra < inter): p = {sep['mannwhitney_p']:.2e}")
        md_lines.append("")

    pooled_margin = margin_report.get("pooled", {}) if margin_report else {}
    if pooled_margin:
        md_lines += [
            "**Margen angular a prototipos** (ángulo al rival más cercano − ángulo al propio):",
            "",
            f"- Margen medio: **{pooled_margin.get('margin_mean_rad', 0):.4f} rad ({pooled_margin.get('margin_mean_deg', 0):.2f}°)**",
            f"- Fracción con margen positivo (bien orientados): **{pooled_margin.get('frac_positive_margin', 0):.3f}**",
            "",
        ]

    if cluster_metrics:
        md_lines += [
            "**Calidad de clusters en la hiperesfera:**",
            "",
            f"- Silhouette (coseno): **{cluster_metrics.get('silhouette_cosine', float('nan')):.4f}**",
            f"- Davies–Bouldin (↓ mejor): **{cluster_metrics.get('davies_bouldin', float('nan')):.4f}**",
            f"- Calinski–Harabasz (↑ mejor): **{cluster_metrics.get('calinski_harabasz', float('nan')):.1f}**",
            "",
        ]

    cm_names = confusion["classes"]
    md_lines += [
        "**Matriz de confusión holdout** (norm. por recall, conteo crudo entre paréntesis):",
        "",
        "| gold \\ pred | " + " | ".join(cm_names) + " |",
        "|" + "---|" * (len(cm_names) + 1),
    ]
    cm_norm = confusion["normalized_recall"]
    cm_raw = confusion["raw"]
    for i, nm in enumerate(cm_names):
        md_lines.append(
            "| **" + nm + "** | "
            + " | ".join(f"{cm_norm[i][j]:.2f} ({cm_raw[i][j]})" for j in range(len(cm_names)))
            + " |"
        )
    md_lines.append("")

    md_lines += [
        f"## Veredicto cuello de botella",
        "",
        f"**`{bottleneck['bottleneck']}`** — ver `embed_bottleneck.json`",
        "",
        "## Figuras",
        "",
        "- `figures/umap_probe_holdout.png` — proyección UMAP",
        "- `figures/tsne_probe_holdout.png` — t-SNE coseno (dominios + aciertos/errores)" + ("" if tsne_ok else " (omitida)"),
        "- `figures/angular_distance_distribution.png` — histograma separación angular intra/inter",
        "- `figures/prototype_geometry_matrix.png` — matriz angular + coseno entre prototipos",
        "- `figures/confusion_matrix_holdout.png` — matriz de confusión",
        "- `figures/angular_margin_hist.png` — margen angular (legacy)",
        "",
        "## Artefactos JSON",
        "",
        "- `hypersphere_angular.json` — separación angular, margen, clusters, confusión",
        "- `confusion_matrix_holdout.csv`",
        "",
        "## Por imagen",
        "",
        f"- `per_image_recall.csv` ({len(per_img)} imágenes)",
        "",
    ]
    report_path = analysis / "EMBED_REPORT.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "run_id": run_dir.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "loss_type": ckpt.get("loss_type"),
        "probe_macro_f1": probe_f1,
        "bottleneck": bottleneck["bottleneck"],
        "embed_report": str(report_path),
        "best_epoch": ckpt.get("epoch"),
        "hypersphere": {
            "intra_mean_deg": sep_report.get("intra_pooled", {}).get("mean_deg"),
            "inter_mean_deg": sep_report.get("inter_pooled", {}).get("mean_deg"),
            "separation_index": sep_report.get("separation", {}).get("separation_index"),
            "cohens_d": sep_report.get("separation", {}).get("cohens_d_intra_vs_inter"),
            "silhouette_cosine": cluster_metrics.get("silhouette_cosine"),
            "frac_positive_margin": (margin_report.get("pooled", {}) or {}).get("frac_positive_margin"),
        },
    }
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"[Gate embed analysis] OK -> {report_path}")
    return summary
