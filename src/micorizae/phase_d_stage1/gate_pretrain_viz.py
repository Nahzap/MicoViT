"""Visualizaciones pre-entrenamiento gate AM (distribución + muestras de tiles)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.jpeg_streaming import crop_tile_u8_from_file
from .gate_classes import GATE_CLASS_NAMES

log = get_logger("phase_d.gate_pretrain_viz")

# Cache acotada: como mucho 2 panorámicas AM en RAM (~2–3 GB).
_image_rgb_cache: dict[str, np.ndarray] = {}
_IMAGE_CACHE_MAX = 2


def _load_tile_rgb_u8(rec: pd.Series, paths_root: Path) -> np.ndarray | None:
    """Recorte tile RGB uint8; evita DecompressionBomb PIL y loguea decode lento."""
    img_path = paths_root / str(rec["image_path"])
    if not img_path.exists():
        return None
    ts = int(rec["tile_size"]) if "tile_size" in rec else 252
    row, col = int(rec["row"]), int(rec["col"])
    key = str(img_path.resolve())
    try:
        if key not in _image_rgb_cache:
            while len(_image_rgb_cache) >= _IMAGE_CACHE_MAX:
                evict = next(iter(_image_rgb_cache))
                del _image_rgb_cache[evict]
            log.info(f"[Gate viz] decodificando JPEG {img_path.name} (puede tardar en panorámicas AM)...")
            print(f"[Gate viz] decodificando {img_path.name}...", flush=True)
        return crop_tile_u8_from_file(
            img_path,
            row,
            col,
            ts,
            image_arr_cache=_image_rgb_cache,
        )
    except Exception as e:
        log.debug(f"[Gate viz] tile load fail {img_path}: {e}")
        return None


def _clear_tile_viz_cache() -> None:
    _image_rgb_cache.clear()


def ensure_matplotlib_agg() -> None:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")


def legend_if_labeled(ax, **kwargs) -> None:
    """Solo añade leyenda si hay series etiquetadas (evita warnings de matplotlib)."""
    _handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def _sample_tiles_for_viz(
    pool: pd.DataFrame,
    n_per_class: int,
    rng: np.random.Generator,
    *,
    max_unique_images: int = 4,
) -> pd.DataFrame:
    """Muestra tiles para PNG pretrain limitando JPEGs únicos (panorámicas AM son lentas)."""
    if pool.empty or n_per_class <= 0:
        return pool.iloc[:0]
    imgs = pool["image_path"].astype(str).unique()
    rng.shuffle(imgs)
    chosen = list(imgs[: max(1, min(max_unique_images, len(imgs)))])
    per_img = max(1, int(np.ceil(n_per_class / len(chosen))))
    parts: list[pd.DataFrame] = []
    for img in chosen:
        if len(parts) >= n_per_class:
            break
        sub = pool[pool["image_path"].astype(str) == img]
        need = min(per_img, n_per_class - sum(len(p) for p in parts), len(sub))
        if need <= 0:
            continue
        idx = rng.choice(sub.index.to_numpy(), size=need, replace=len(sub) < need)
        parts.append(sub.loc[idx])
    if not parts:
        return pool.iloc[:0]
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n_per_class:
        rest = pool.drop(out.index, errors="ignore")
        if not rest.empty:
            extra = min(n_per_class - len(out), len(rest))
            idx = rng.choice(rest.index.to_numpy(), size=extra, replace=len(rest) < extra)
            out = pd.concat([out, rest.loc[idx]], ignore_index=True)
    return out.iloc[:n_per_class]


def plot_tile_samples_by_class(
    tiles_df: pd.DataFrame,
    out_png: Path,
    *,
    n_per_class: int = 12,
    seed: int = 0,
    title: str = "Muestra tiles por clase (gold stage1)",
) -> Optional[Path]:
    """Grid PNG: n tiles aleatorios por clase, recorte JPEG desde manifest."""
    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    _clear_tile_viz_cache()
    required = {"image_path", "row", "col", "stage1"}
    if tiles_df.empty or not required.issubset(tiles_df.columns):
        log.warning("[Gate viz] plot_tile_samples: dataframe vacio o sin columnas requeridas")
        return None

    rng = np.random.default_rng(seed)
    paths_root = get_paths().root
    stage1_map = {"Unknown": "Unreadable"}
    order = [c for c in GATE_CLASS_NAMES if c != "Unknown"]

    fig, axes = plt.subplots(len(order), n_per_class, figsize=(n_per_class * 1.2, len(order) * 1.2))
    if len(order) == 1:
        axes = np.array([axes])
    if n_per_class == 1:
        axes = axes.reshape(len(order), 1)

    for row_i, cls in enumerate(order):
        key = stage1_map.get(cls, cls)
        pool = tiles_df[tiles_df["stage1"].astype(str) == key]
        if pool.empty:
            pool = tiles_df[tiles_df["stage1"].astype(str) == cls]
        if pool.empty:
            for col_i in range(n_per_class):
                axes[row_i, col_i].axis("off")
            axes[row_i, 0].set_ylabel(cls, fontsize=9)
            continue
        n = min(n_per_class, len(pool))
        sample = _sample_tiles_for_viz(pool, n, rng)
        # Ordenar por imagen: maximiza hits de cache y reduce thrashing RAM.
        if not sample.empty and "image_path" in sample.columns:
            sample = sample.sort_values("image_path").reset_index(drop=True)

        for col_i in range(n_per_class):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(cls, fontsize=9)
            if col_i >= len(sample):
                continue
            rec = sample.iloc[col_i]
            tile = _load_tile_rgb_u8(rec, paths_root)
            if tile is None:
                ax.text(0.5, 0.5, "missing" if not (paths_root / str(rec["image_path"])).exists() else "err", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(tile)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    _clear_tile_viz_cache()
    return out_png


def _stage1_to_gate_label(stage1: str) -> str:
    from .gate_classes import stage1_to_gate_label

    return stage1_to_gate_label(str(stage1))


def _tile_basename(image_path: str, max_len: int = 22) -> str:
    from pathlib import PurePath

    name = PurePath(str(image_path)).name
    if len(name) <= max_len:
        return name
    stem, suffix = PurePath(name).stem, PurePath(name).suffix
    keep = max(8, max_len - len(suffix) - 1)
    return f"{stem[:keep]}…{suffix}" if len(stem) > keep else name[:max_len]


def _format_tile_caption(rec: pd.Series, *, split: str, etapa: str) -> str:
    cls = _stage1_to_gate_label(rec.get("stage1", ""))
    pos = f"r{int(rec['row'])}c{int(rec['col'])}"
    aug = ""
    if "aug_id" in rec.index and pd.notna(rec["aug_id"]):
        try:
            if int(rec["aug_id"]) > 0:
                aug = f" aug={int(rec['aug_id'])}"
        except (TypeError, ValueError):
            pass
    return f"{split} | {cls}\n{_tile_basename(rec.get('image_path', ''))}\n{pos}{aug} | {etapa}"


def plot_tile_samples_annotated(
    tiles_df: pd.DataFrame,
    out_png: Path,
    *,
    split: str = "train",
    etapa: str = "pre_train",
    n_per_class: int = 12,
    seed: int = 0,
    title: str = "Tiles anotados (split | clase | imagen | pos | aug | etapa)",
    manifest_csv: Optional[Path] = None,
) -> Optional[Path]:
    """Grid PNG con metadatos por tile + CSV manifest opcional."""
    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    _clear_tile_viz_cache()
    required = {"image_path", "row", "col", "stage1"}
    if tiles_df.empty or not required.issubset(tiles_df.columns):
        log.warning("[Gate viz] plot_tile_samples_annotated: dataframe vacio o sin columnas")
        return None

    rng = np.random.default_rng(seed)
    paths_root = get_paths().root
    stage1_map = {"Unknown": "Unreadable"}
    order = [c for c in GATE_CLASS_NAMES if c != "Unknown"]
    manifest_rows: list[dict] = []

    fig, axes = plt.subplots(len(order), n_per_class, figsize=(n_per_class * 1.65, len(order) * 1.85))
    if len(order) == 1:
        axes = np.array([axes])
    if n_per_class == 1:
        axes = axes.reshape(len(order), 1)

    for row_i, cls in enumerate(order):
        key = stage1_map.get(cls, cls)
        pool = tiles_df[tiles_df["stage1"].astype(str) == key]
        if pool.empty:
            pool = tiles_df[tiles_df["stage1"].astype(str) == cls]
        if pool.empty:
            for col_i in range(n_per_class):
                axes[row_i, col_i].axis("off")
            axes[row_i, 0].set_ylabel(cls, fontsize=9)
            continue
        n = min(n_per_class, len(pool))
        sample = _sample_tiles_for_viz(pool, n, rng)
        # Ordenar por imagen: maximiza hits de cache y reduce thrashing RAM.
        if not sample.empty and "image_path" in sample.columns:
            sample = sample.sort_values("image_path").reset_index(drop=True)

        for col_i in range(n_per_class):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(cls, fontsize=9)
            if col_i >= len(sample):
                continue
            rec = sample.iloc[col_i]
            ts = int(rec["tile_size"]) if "tile_size" in rec else 252
            img_path = paths_root / str(rec["image_path"])
            caption = _format_tile_caption(rec, split=split, etapa=etapa)
            tile = _load_tile_rgb_u8(rec, paths_root)
            if tile is None:
                ax.text(
                    0.5,
                    0.5,
                    "missing" if not img_path.exists() else "err",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
            else:
                ax.imshow(tile)
            ax.set_title(caption, fontsize=6.5, pad=4, linespacing=1.15)
            manifest_rows.append(
                {
                    "split": split,
                    "etapa": etapa,
                    "gate_class": cls,
                    "stage1": str(rec.get("stage1", "")),
                    "image_path": str(rec.get("image_path", "")),
                    "row": int(rec["row"]),
                    "col": int(rec["col"]),
                    "aug_id": rec.get("aug_id", ""),
                    "tile_size": ts,
                }
            )

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.92, hspace=0.55, wspace=0.25)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _clear_tile_viz_cache()

    if manifest_csv is not None:
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    return out_png


def write_pre_training_index(out_dir: Path) -> Path:
    """INDEX.md legible para artefactos pre-training."""
    content = """# Pre-training — índice de artefactos

| Archivo | Descripción |
|---------|-------------|
| `class_distribution.png` | Conteo tiles por clase train vs holdout |
| `train_val_balance.png` | Balance train/val por imagen |
| `stratified_tile_samples_annotated.png` | Muestra train/sampler con metadatos por tile |
| `holdout_tile_samples_annotated.png` | Muestra holdout natural con metadatos |
| `tile_manifest_stratified.csv` | CSV: split, clase, imagen, row/col, aug_id, etapa |
| `tile_manifest_holdout.csv` | Idem para holdout |
| `dataset_stats.json` | Stats split + stage1 |
| `baseline_metrics.json` | Métricas ep0 (probe sin entrenar) |

**Leyenda por tile:** `split | clase | imagen | rXcY | aug=… | etapa`
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "INDEX.md"
    path.write_text(content, encoding="utf-8")
    return path


def plot_pretrain_dimension_panels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: Path,
    *,
    stratified_batch_df: pd.DataFrame | None = None,
    n_per: int = 6,
    seed: int = 0,
) -> None:
    """Fase 1.2: PNG por split/clase, sampler batch, holdout por imagen."""
    from ..common.logging_utils import get_logger
    from .gate_classes import stage1_to_gate_label

    viz_log = get_logger("phase_d.gate_pretrain_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [
            train_df.assign(_split="train", _etapa="pre_train"),
            val_df.assign(_split="holdout", _etapa="pre_train"),
        ],
        ignore_index=True,
    )
    viz_log.info("[Gate viz] panel by_split_class (JPEG)...")
    plot_tile_samples_annotated(
        combined,
        out_dir / "by_split_class.png",
        split="train+holdout",
        etapa="pre_train",
        n_per_class=n_per,
        seed=seed,
        title="Por split y clase (train + holdout)",
        manifest_csv=out_dir / "tile_manifest_by_split_class.csv",
    )
    if stratified_batch_df is not None and not stratified_batch_df.empty:
        viz_log.info("[Gate viz] panel sampler_batch_example (JPEG)...")
        plot_tile_samples_annotated(
            stratified_batch_df,
            out_dir / "sampler_batch_example.png",
            split="train",
            etapa="pre_train_sampler",
            n_per_class=min(n_per, len(stratified_batch_df) // 3 + 1),
            seed=seed,
            title="Ejemplo batch g1_stratified",
            manifest_csv=out_dir / "tile_manifest_sampler_batch.csv",
        )
    if not val_df.empty and "image_path" in val_df.columns:
        imgs = val_df["image_path"].astype(str).unique()
        if len(imgs):
            img = imgs[seed % len(imgs)]
            sub = val_df[val_df["image_path"].astype(str) == img].head(n_per * 3)
            viz_log.info(f"[Gate viz] panel holdout_by_image: {Path(img).name} (JPEG)...")
            plot_tile_samples_annotated(
                sub,
                out_dir / "holdout_by_image.png",
                split="holdout",
                etapa="pre_train",
                n_per_class=min(n_per, len(sub)),
                seed=seed,
                title=f"Holdout imagen: {Path(img).name}",
                manifest_csv=out_dir / "tile_manifest_holdout_by_image.csv",
            )
