"""Informe post-entrenamiento Stage2-Pixel: métricas, tablas y visualizaciones."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from .pixel_class_map import MORPH_STRUCTURE_CLASSES, PIXEL_CLASS_NAMES
from .pixel_data import load_mplus_splits
from .pixel_morph import (
    PixelMorphParams,
    quantize_segment,
    render_diagnostic_overlay,
    render_pixel_class_map,
)
from .pixel_vit_model import build_pixel_morph_vit
from .stage2_pixel_h5_cache import Stage2PixelH5Store, _cache_paths
from .stage2_pixel_run_layout import (
    Stage2PixelRunLayout,
    atomic_write_text,
    plot_confusion_matrix,
    plot_val_miou_curve,
    resolve_best_checkpoint,
    resolve_run_layout,
)
from .stage2_pixel_train_report import (
    confusion_from_pred_labels,
    macro_iou,
    morph_iou_summary,
    per_class_iou,
    phase_log,
)
log = get_logger("phase_e.pixel_posttrain")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_matplotlib_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def _denorm_rgb(rgb_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) float normalizado ImageNet → (H,W,3) uint8."""
    x = rgb_chw.transpose(1, 2, 0).astype(np.float32)
    x = x * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _hstack_images(left: np.ndarray, right: np.ndarray, gap: int = 4) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + gap + right.shape[1]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] + gap :] = right
    return out


def _hstack_many(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    out = images[0]
    for img in images[1:]:
        out = _hstack_images(out, img, gap=gap)
    return out


def _merge_gate_tile_coords(
    gate_df: pd.DataFrame,
    image_rel: str,
    *,
    gate_scope: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Une predicciones gate con coordenadas de la malla Gate (252px base)."""
    key_cols = ["row", "col"]
    coord_cols = ["row", "col", "x0", "y0", "x1", "y1", "tile_size"]

    if gate_scope is not None:
        scope = gate_scope[gate_scope["image_path"].astype(str) == image_rel].copy()
        if not scope.empty and all(c in scope.columns for c in coord_cols):
            coords = scope[coord_cols].drop_duplicates(subset=key_cols)
            merged = gate_df.merge(coords, on=key_cols, how="left", suffixes=("_g", ""))
            for c in ("x0", "y0", "x1", "y1", "tile_size"):
                cg, cm = f"{c}_g", c
                if cg in merged.columns:
                    if cm in merged.columns:
                        merged[cm] = merged[cm].fillna(merged[cg])
                    else:
                        merged[cm] = merged[cg]
                    merged.drop(columns=[cg], inplace=True)
            return merged.dropna(subset=["x0", "y0", "x1", "y1"]).copy()

    paths = get_paths()
    manifest = pd.read_parquet(paths.manifests / "tiles_index.parquet")
    sub = manifest[manifest["image_path"] == image_rel][coord_cols]
    if "tile_size" in sub.columns:
        sub = sub[sub["tile_size"] == 252]
    sub = sub.drop_duplicates(subset=key_cols)
    merged = gate_df.merge(sub, on=key_cols, how="left")
    return merged.dropna(subset=["x0", "y0", "x1", "y1"]).copy()


def _bbox_tile_union(df: pd.DataFrame, *, pad: int = 32) -> tuple[int, int, int, int] | None:
    if df.empty:
        return None
    x0 = max(0, int(df["x0"].min()) - pad)
    y0 = max(0, int(df["y0"].min()) - pad)
    x1 = int(df["x1"].max()) + pad
    y1 = int(df["y1"].max()) + pad
    return x0, y0, x1, y1


def _crop_rgb(img: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return img[y0:y1, x0:x1].copy()


def _crop_seg(seg: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    return _crop_rgb(seg, bbox)


def _render_gate_tile_grid(
    image_shape: tuple[int, int],
    gate_df: pd.DataFrame,
    *,
    mplus_keys: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Overlay rejilla de tiles coloreada por clase gate (Bg/M-/M+). Cian = procesado Stage2."""
    import cv2

    h, w = image_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    gate_colors = {
        "Background": (200, 200, 200),
        "Mminus": (255, 220, 80),
        "Mplus": (80, 220, 120),
        "Unknown": (180, 140, 255),
    }
    for rec in gate_df.itertuples(index=False):
        x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
        label = str(getattr(rec, "stage1_pred", getattr(rec, "stage1", "Background")))
        color = gate_colors.get(label, (160, 160, 160))
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, thickness=2)
        if mplus_keys and (int(rec.row), int(rec.col)) in mplus_keys:
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (0, 255, 255), thickness=2)
    return canvas


def _render_stage2_coverage_mask(
    image_shape: tuple[int, int],
    gate_df: pd.DataFrame,
    mplus_df: pd.DataFrame,
    *,
    gold_mplus_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Verde = M+ Gate procesado Stage2; naranja = M+ gold no detectado; gris = resto malla."""
    import cv2

    h, w = image_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mplus_keys = {(int(r.row), int(r.col)) for r in mplus_df.itertuples(index=False)}
    gold_keys: set[tuple[int, int]] = set()
    if gold_mplus_df is not None and not gold_mplus_df.empty:
        gold_keys = {(int(r.row), int(r.col)) for r in gold_mplus_df.itertuples(index=False)}
    for rec in gate_df.itertuples(index=False):
        x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
        key = (int(rec.row), int(rec.col))
        if key in mplus_keys:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 80), thickness=-1)
        elif key in gold_keys:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 140, 255), thickness=-1)
        else:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), thickness=-1)
    return canvas


def _tile_pixel_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    cm = confusion_from_pred_labels(pred, gt)
    return macro_iou(per_class_iou(cm))


def plot_per_class_iou_bars(per_class: dict[str, float], out_png: Path, *, title: str) -> Optional[Path]:
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    names = [n for n in PIXEL_CLASS_NAMES if n != "BG"]
    vals = [per_class.get(n, float("nan")) for n in names]
    colors = ["#4C78A8" if not np.isnan(v) and v > 0.05 else "#E45756" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, [0.0 if np.isnan(v) else v for v in vals], color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_train_loss_curve(history: dict, out_png: Path) -> Optional[Path]:
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not epochs:
        return None
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, marker="o", label="train loss", color="#72B7B2")
    if val_loss:
        ax.plot(epochs, val_loss, marker="s", label="val loss", color="#F58518")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("Stage2-Pixel ViT — training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_per_class_iou_curves(history: dict, out_png: Path) -> Optional[Path]:
    epochs = history.get("epochs", [])
    per_epoch = history.get("val_per_class_iou") or []
    if not epochs or not per_epoch:
        return None
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    for cls in MORPH_STRUCTURE_CLASSES:
        ys = [pc.get(cls, float("nan")) for pc in per_epoch[: len(epochs)]]
        ax.plot(epochs[: len(ys)], ys, marker=".", label=cls)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val IoU")
    ax.set_title("Stage2-Pixel ViT — per-class val IoU")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


@torch.no_grad()
def evaluate_split_tiles(
    model: torch.nn.Module,
    tiles_df: pd.DataFrame,
    h5_store: Stage2PixelH5Store,
    *,
    device: torch.device,
    batch_size: int = 8,
    collect_tile_rows: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evalúa split tile-a-tile vía HDF5; devuelve métricas agregadas + tabla por tile."""
    model.eval()
    cm = np.zeros((len(PIXEL_CLASS_NAMES), len(PIXEL_CLASS_NAMES)), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    n = len(tiles_df)
    if n == 0:
        return {"loss": 0.0, "miou": 0.0, "acc": 0.0, "per_class_iou": {}, "confusion": cm.tolist()}, pd.DataFrame()

    all_idx = h5_store.indices_for_sub(tiles_df)
    total_loss, total_px, correct_px = 0.0, 0, 0
    import torch.nn.functional as F

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_idx = all_idx[start:end]
        chunk_df = tiles_df.iloc[start:end]
        rgb, labels = h5_store.read_batch(chunk_idx, device)
        logits = model(rgb)
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        pred_np = pred.cpu().numpy()
        lab_np = labels.cpu().numpy()
        cm += confusion_from_pred_labels(pred_np, lab_np)
        correct_px += int((pred == labels).sum().item())
        total_px += int(labels.numel())

        if collect_tile_rows:
            rgb_np = rgb.cpu().numpy()
            for j in range(bs):
                rec = chunk_df.iloc[j]
                pred_seg = pred_np[j]
                gt_seg = lab_np[j]
                tile_iou = _tile_pixel_iou(pred_seg, gt_seg)
                q_pred = quantize_segment(pred_seg)
                q_gt = quantize_segment(gt_seg)
                rows.append(
                    {
                        "split": "",
                        "level": "tile",
                        "image_path": str(rec["image_path"]),
                        "row": int(rec["row"]),
                        "col": int(rec["col"]),
                        "stage2_gold": str(rec.get("stage2", "")),
                        "tile_miou": tile_iou,
                        **{f"pred_{k}": v for k, v in q_pred.items()},
                        **{f"gt_{k}": v for k, v in q_gt.items()},
                    }
                )

    pc_iou = per_class_iou(cm)
    metrics = {
        "loss": total_loss / max(n, 1),
        "miou": macro_iou(pc_iou),
        "acc": float(correct_px / max(total_px, 1)),
        "per_class_iou": pc_iou,
        "confusion": cm.tolist(),
        "n_tiles": n,
    }
    return metrics, pd.DataFrame(rows)


def plot_tile_morph_audit_grid(
    model: torch.nn.Module,
    tiles_df: pd.DataFrame,
    h5_store: Stage2PixelH5Store,
    *,
    device: torch.device,
    out_png: Path,
    n_per_class: int = 4,
    seed: int = 0,
) -> Optional[Path]:
    """Grid: RGB | weak GT | pred | overlay — una fila por clase morfológica dominante."""
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    morph_classes = list(MORPH_STRUCTURE_CLASSES)
    rng = np.random.default_rng(seed)
    model.eval()

    fig, axes = plt.subplots(len(morph_classes), n_per_class, figsize=(n_per_class * 2.2, len(morph_classes) * 2.0))
    if len(morph_classes) == 1:
        axes = np.array([axes])
    if n_per_class == 1:
        axes = axes.reshape(len(morph_classes), 1)

    for row_i, cls in enumerate(morph_classes):
        cls_idx = PIXEL_CLASS_NAMES.index(cls)
        for col_i in range(n_per_class):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(cls, fontsize=9)

        pool_idx = []
        for start in range(0, len(tiles_df), 64):
            sub = tiles_df.iloc[start : start + 64]
            if sub.empty:
                continue
            h5_idx = h5_store.indices_for_sub(sub)
            _, labels = h5_store.read_batch(h5_idx, device)
            lab_np = labels.cpu().numpy()
            for j, lab in enumerate(lab_np):
                if (lab == cls_idx).sum() > lab.size * 0.02:
                    pool_idx.append(start + j)
        if not pool_idx:
            continue
        pick = rng.choice(pool_idx, size=min(n_per_class, len(pool_idx)), replace=False)

        for col_i, ti in enumerate(pick):
            ax = axes[row_i, col_i]
            rec = tiles_df.iloc[int(ti)]
            h5_idx = h5_store.indices_for_sub(pd.DataFrame([rec]))
            rgb, labels = h5_store.read_batch(h5_idx, device)
            with torch.no_grad():
                pred = model(rgb).argmax(dim=1)
            rgb_u8 = _denorm_rgb(rgb[0].cpu().numpy())
            gt_seg = labels[0].cpu().numpy().astype(np.uint8)
            pred_seg = pred[0].cpu().numpy().astype(np.uint8)
            gt_color = render_pixel_class_map(gt_seg)
            pred_color = render_pixel_class_map(pred_seg)
            overlay = render_diagnostic_overlay(rgb_u8, pred_seg, alpha=0.45)
            panel = _hstack_many([rgb_u8, gt_color, pred_color, overlay], gap=2)
            ax.imshow(panel)
            tiou = _tile_pixel_iou(pred_seg, gt_seg)
            ax.set_title(f"r{rec['row']}c{rec['col']}\nmIoU={tiou:.2f}", fontsize=6)

    fig.suptitle("Stage2-Pixel — tile audit (RGB | weak GT | pred | overlay)", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _prior_channel_heatmap(prior_ch: np.ndarray) -> np.ndarray:
    """Canal prior (H,W) float → (H,W,3) uint8 colormap."""
    import cv2

    x = prior_ch.astype(np.float32)
    denom = float(x.max()) + 1e-9
    gray = (np.clip(x / denom, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def render_sample_tile_panels(
    model: torch.nn.Module,
    tiles_df: pd.DataFrame,
    h5_store: Stage2PixelH5Store,
    *,
    device: torch.device,
    out_dir: Path,
    n_tiles: int = 12,
    seed: int = 42,
    split_name: str = "val",
    with_explain_panels: bool = False,
    attention_layers: str = "last",
    input_size: int = 224,
) -> list[Path]:
    """Guarda PNGs individuales RGB|weak|pred|overlay [+| prior IH/A | atención DINO]."""
    from .pixel_attention import extract_pixel_vit_attention, render_attention_heatmap
    from .pixel_class_map import PIXEL_CLASS_TO_IDX
    from .pixel_vit_model import PixelMorphViT

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if tiles_df.empty:
        return []
    n = min(n_tiles, len(tiles_df))
    pick = rng.choice(tiles_df.index.to_numpy(), size=n, replace=False)
    written: list[Path] = []
    model.eval()
    vit_model = model if isinstance(model, PixelMorphViT) else None

    for ti in pick:
        rec = tiles_df.loc[ti]
        h5_idx = h5_store.indices_for_sub(pd.DataFrame([rec]))
        if with_explain_panels and h5_store.has_priors:
            rgb, labels, prior_e, _ = h5_store.read_training_tensors(h5_idx, device, load_priors=True)
        else:
            rgb, labels = h5_store.read_batch(h5_idx, device)
            prior_e = None
        with torch.no_grad():
            pred = model(rgb).argmax(dim=1)
        rgb_u8 = _denorm_rgb(rgb[0].cpu().numpy())
        gt_seg = labels[0].cpu().numpy().astype(np.uint8)
        pred_seg = pred[0].cpu().numpy().astype(np.uint8)
        panels = [
            rgb_u8,
            render_pixel_class_map(gt_seg),
            render_pixel_class_map(pred_seg),
            render_diagnostic_overlay(rgb_u8, pred_seg, alpha=0.45),
        ]
        if with_explain_panels and prior_e is not None:
            pe = prior_e[0].cpu().numpy()
            panels.append(_prior_channel_heatmap(pe[PIXEL_CLASS_TO_IDX["IH"]]))
            panels.append(_prior_channel_heatmap(pe[PIXEL_CLASS_TO_IDX["A"]]))
        if with_explain_panels and vit_model is not None:
            attn = extract_pixel_vit_attention(
                vit_model, rgb, layers_spec=attention_layers, input_size=input_size
            )
            attn_rgb = render_attention_heatmap(attn)[:, :, ::-1]  # BGR→RGB
            panels.append(_hstack_images(rgb_u8, attn_rgb, gap=2))
        panel = _hstack_many(panels, gap=3)
        stem = Path(str(rec["image_path"])).stem
        out = out_dir / f"{split_name}__{stem}__r{int(rec['row'])}c{int(rec['col'])}.png"
        from PIL import Image

        Image.fromarray(panel).save(out)
        written.append(out)
    return written


def render_fullimage_overlays(
    *,
    image_paths: list[str],
    gate_run_id: str,
    model: torch.nn.Module,
    device: torch.device,
    morph_params: PixelMorphParams,
    input_size: int,
    out_dir: Path,
    split_name: str = "test",
    gate_strict: bool = True,
    crop_to_tiles: bool = True,
) -> list[Path]:
    """Mapas full-image secuenciales: Gate (M1) → Stage2 (M2) solo en tiles M+.

    Paneles por imagen:
    - ``__pipeline_overview.png``: RGB | rejilla gate | cobertura M+ | colonia pred
    - ``__weak_vs_pred.png``: weak vs ViT (solo zona M+ procesada)
    - ``__L9_pred_vs_diag.png``: mapa clases vs overlay diagnóstico
    """
    from ..phase_b_tiling.gpu_io import decode_jpeg_gpu
    from .infer_pixel_gpu import infer_image_pixel_morph
    from .pixel_morph import render_colony_binary
    from .stage2_gate_infer import gate_infer_tile_scope, infer_image_gate_stage2, load_stage2_gate_bundle

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = get_paths()
    gate = load_stage2_gate_bundle(device=device, gate_run_id=gate_run_id)
    gate_scope = gate_infer_tile_scope()
    written: list[Path] = []

    for img_rel in image_paths:
        img_path = paths.root / img_rel
        if not img_path.is_file():
            continue

        # Paso 1 — Modelo 1 (Gate) sobre todos los tiles
        s1 = infer_image_gate_stage2(
            img_path, gate, batch_size=64, strict=gate_strict, tile_scope_df=gate_scope
        )
        if s1.empty:
            continue
        gate_coords = _merge_gate_tile_coords(s1, img_rel, gate_scope=gate_scope)
        mplus_df = gate_coords[gate_coords["is_mplus"] == 1].copy()
        mplus_keys = {(int(r.row), int(r.col)) for r in mplus_df.itertuples(index=False)}
        scope_img = gate_scope[gate_scope["image_path"].astype(str) == img_rel]
        gold_mplus_df = scope_img[scope_img["stage1"].astype(str) == "Mplus"].copy()

        # Paso 2 — Modelo 2 (ViT morfológico) solo en tiles M+ del Gate
        full_seg, _, _ = infer_image_pixel_morph(
            img_path,
            s1,
            backend="vit",
            model=model,
            device=device,
            morph_params=morph_params,
            input_size=input_size,
        )
        full_weak, _, _ = infer_image_pixel_morph(
            img_path,
            s1,
            backend="weak",
            model=None,
            device=device,
            morph_params=morph_params,
            input_size=input_size,
        )

        gimg = decode_jpeg_gpu(img_path, device=device)
        img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        h, w = img_np.shape[:2]
        del gimg

        bbox = _bbox_tile_union(gate_coords, pad=48) if crop_to_tiles else None
        img_c = _crop_rgb(img_np, bbox)
        seg_c = _crop_seg(full_seg, bbox)
        weak_c = _crop_seg(full_weak, bbox)

        gate_grid = _render_gate_tile_grid((h, w), gate_coords, mplus_keys=mplus_keys)
        coverage = _render_stage2_coverage_mask(
            (h, w), gate_coords, mplus_df, gold_mplus_df=gold_mplus_df
        )
        gate_grid_c = _crop_rgb(gate_grid, bbox)
        coverage_c = _crop_rgb(coverage, bbox)

        pred_color = render_pixel_class_map(seg_c)
        weak_color = render_pixel_class_map(weak_c)
        colony = render_colony_binary(seg_c)
        diag = render_diagnostic_overlay(img_c, seg_c, alpha=0.45)

        stem = img_path.stem
        from PIL import Image

        overview = _hstack_many(
            [
                img_c,
                _hstack_images(gate_grid_c, coverage_c, gap=4),
                colony,
                diag,
            ],
            gap=6,
        )
        p0 = out_dir / f"{stem}__pipeline_overview.png"
        Image.fromarray(overview).save(p0)
        written.append(p0)

        p1 = out_dir / f"{stem}__L9_pred_vs_diag.png"
        Image.fromarray(_hstack_images(pred_color, diag, gap=8)).save(p1)
        written.append(p1)

        p2 = out_dir / f"{stem}__weak_vs_pred.png"
        Image.fromarray(_hstack_images(weak_color, pred_color, gap=8)).save(p2)
        written.append(p2)

        n_mplus = len(mplus_keys)
        n_gold_mplus = len(gold_mplus_df)
        n_gate = len(gate_coords)
        phase_log(
            f"[posttrain viz] {split_name} {stem}: malla Gate {n_gate} tiles (252px), "
            f"M+ Gate={n_mplus} / gold={n_gold_mplus} -> Stage2 | {p0.name}"
        )

    return written


def _metrics_table_md(metrics: dict[str, Any], *, split: str) -> list[str]:
    lines = [
        f"# Stage2-Pixel — informe {split}",
        "",
        f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Métricas holdout (weak pseudo-labels como referencia)",
        "",
        "| Métrica | Valor |",
        "|---------|------:|",
        f"| mIoU | **{metrics.get('miou', 0):.4f}** |",
        f"| pixel accuracy | {metrics.get('acc', 0):.4f} |",
        f"| CE loss | {metrics.get('loss', 0):.4f} |",
        f"| n tiles | {metrics.get('n_tiles', 0)} |",
        "",
        "## IoU por clase",
        "",
        "| Clase | IoU |",
        "|-------|----:|",
    ]
    for cls in PIXEL_CLASS_NAMES:
        v = metrics.get("per_class_iou", {}).get(cls, float("nan"))
        if np.isnan(v):
            lines.append(f"| {cls} | — |")
        else:
            lines.append(f"| {cls} | {v:.4f} |")
    lines += [
        "",
        "## Interpretación",
        "",
        "- **H (hifas/colonia)** suele ser la clase dominante; IoU ~0.4 indica aprendizaje parcial del tejido colonizado.",
        "- **IH / V / A** requieren señal morfológica fina; IoU ≈ 0 implica que el ViT no separa estructuras intra-coloniales.",
        "- Las pseudo-labels weak (Frangi/vesículas) son ruidosas: métricas bajas pueden reflejar límite del supervisor, no solo del modelo.",
        "",
        "## Artefactos visuales",
        "",
        f"- `posttrain/visuals/tile_audit_best_ckpt.png` — grid tile RGB|weak|pred|overlay",
        f"- `posttrain/visuals/{split}/` — muestras individuales",
        f"- `posttrain/visuals/{split}/fullimage/` — mapas por imagen (weak vs pred)",
        f"- `posttrain/plots/` — curvas de entrenamiento y barras IoU",
        "",
    ]
    return lines


def write_run_status_md(
    layout: Stage2PixelRunLayout,
    *,
    history: dict,
    best_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> Path:
    be = history.get("best_epoch", "?")
    bm = history.get("best_val_miou", 0)
    morph = morph_iou_summary(best_metrics.get("per_class_iou_at_best", {}))
    lines = [
        "# Estado Stage2-Pixel ViT",
        "",
        f"- **Estado:** `completed` ({len(history.get('epochs', []))} épocas)",
        f"- **Mejor checkpoint:** época {be}, val mIoU={bm:.4f}",
        f"- **IoU morfológico (best):** {morph}",
        "",
        "## Holdout val / test (best ckpt, weak GT)",
        "",
        f"- Val mIoU: **{val_metrics.get('miou', 0):.4f}** ({val_metrics.get('n_tiles', 0)} tiles)",
        f"- Test mIoU: **{test_metrics.get('miou', 0):.4f}** ({test_metrics.get('n_tiles', 0)} tiles)",
        "",
        "## Diagnóstico rápido",
        "",
        "Si IH/V/A IoU ≈ 0 pero H IoU > 0.3: el modelo aprende **presencia de colonia** pero no **descomposición morfológica**.",
        "Revisar `posttrain/visuals/tile_audit_best_ckpt.png` para confirmar visualmente.",
        "",
        "## Artefactos",
        "",
        "- `posttrain/report/SUMMARY.md` — índice del informe",
        "- `posttrain/report/training_report_val.md` / `training_report_test.md`",
        "- `posttrain/tables/` — parquet + CSV holdout",
        "- `posttrain/visuals/` — overlays y auditoría",
        "",
    ]
    path = layout.posttrain / "report" / "RUN_STATUS.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def write_summary_md(
    layout: Stage2PixelRunLayout,
    *,
    meta: dict,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
    artifact_paths: dict[str, str],
) -> Path:
    h = history
    lines = [
        "# Stage2-Pixel ViT — Informe post-entrenamiento",
        "",
        f"- **run_id:** `{meta.get('run_id', layout.run_root.name)}`",
        f"- **gate_run_id:** `{meta.get('gate_run_id', '')}`",
        f"- **backbone:** {meta.get('backbone', 'dinov2_vits14')}",
        f"- **épocas:** {meta.get('epochs_completed', len(h.get('epochs', [])))} / {meta.get('epochs', 40)}",
        f"- **best epoch:** {h.get('best_epoch')} | **best val mIoU:** {h.get('best_val_miou', 0):.4f}",
        "",
        "## Split",
        "",
    ]
    si = meta.get("split_info") or {}
    lines += [
        f"- Train: {si.get('n_train_tiles', '?')} tiles / {si.get('n_train_images', '?')} imágenes",
        f"- Val: {si.get('n_val_tiles', '?')} tiles / {si.get('n_val_images', '?')} imágenes",
        f"- Test: {si.get('n_test_tiles', '?')} tiles / {si.get('n_test_images', '?')} imágenes",
        "",
        "## Resultados holdout (checkpoint best, GT = weak pseudo-labels)",
        "",
        "| split | mIoU | acc | n tiles |",
        "|-------|-----:|----:|--------:|",
        f"| val | {val_metrics.get('miou', 0):.4f} | {val_metrics.get('acc', 0):.4f} | {val_metrics.get('n_tiles', 0)} |",
        f"| test | {test_metrics.get('miou', 0):.4f} | {test_metrics.get('acc', 0):.4f} | {test_metrics.get('n_tiles', 0)} |",
        "",
        "### IoU por clase (val)",
        "",
    ]
    for cls in PIXEL_CLASS_NAMES:
        v = val_metrics.get("per_class_iou", {}).get(cls, float("nan"))
        if not np.isnan(v):
            lines.append(f"- **{cls}:** {v:.4f}")
    lines += ["", "### IoU por clase (test)", ""]
    for cls in PIXEL_CLASS_NAMES:
        v = test_metrics.get("per_class_iou", {}).get(cls, float("nan"))
        if not np.isnan(v):
            lines.append(f"- **{cls}:** {v:.4f}")

    lines += [
        "",
        "## Curvas de entrenamiento",
        "",
        "| epoch | train_loss | val_loss | val_mIoU | val_acc |",
        "|------:|-----------:|---------:|---------:|--------:|",
    ]
    for i, ep in enumerate(h.get("epochs", [])):
        lines.append(
            f"| {ep} | {h['train_loss'][i]:.4f} | {h['val_loss'][i]:.4f} | "
            f"{h['val_miou'][i]:.4f} | {h['val_acc'][i]:.4f} |"
        )

    lines += ["", "## Artefactos generados", ""]
    for k, v in sorted(artifact_paths.items()):
        lines.append(f"- **{k}:** `{v}`")

    lines += [
        "",
        "## Lectura honesta",
        "",
        f"Val mIoU **{val_metrics.get('miou', 0):.4f}**; test mIoU **{test_metrics.get('miou', 0):.4f}** "
        f"(checkpoint best, pseudo-labels weak).",
        "IoU por clase en val: "
        + ", ".join(
            f"**{cls}**={val_metrics.get('per_class_iou', {}).get(cls, float('nan')):.3f}"
            for cls in PIXEL_CLASS_NAMES
            if not np.isnan(val_metrics.get("per_class_iou", {}).get(cls, float("nan")))
        )
        + ".",
        "IH/V/A siguen siendo las clases más difíciles; H domina el mIoU agregado.",
        "Overlays en `posttrain/visuals/` usan pipeline secuencial Gate→Stage2 (sin fallback gold).",
        "",
    ]
    path = layout.posttrain / "report" / "SUMMARY.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def _write_table(df: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    pq_path = base.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    return csv_path, pq_path


def generate_stage2_pixel_posttrain_report(
    run_root: Path | str,
    *,
    max_fullimage_val: int = 6,
    max_fullimage_test: int = 6,
    n_sample_tiles: int = 12,
    n_audit_per_class: int = 4,
    batch_size: int = 8,
    skip_fullimage: bool = False,
    with_explain_panels: bool = True,
    attention_layers: str = "last",
    gate_strict: bool = True,
    require_gate_cache: bool = True,
    force: bool = False,
) -> Path:
    """Genera informe completo posttrain/ (report, tables, visuals, plots)."""
    layout = resolve_run_layout(run_root)
    report_dir = layout.posttrain / "report"
    tables_dir = layout.posttrain / "tables"
    visuals_dir = layout.posttrain / "visuals"
    plots_dir = layout.posttrain / "plots"

    marker = report_dir / "SUMMARY.md"
    if marker.is_file() and not force:
        phase_log(f"Informe ya existe -> {marker} (usa force=True para regenerar)")
        return report_dir

    for d in (report_dir, tables_dir, visuals_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths = get_paths()
    train_cfg_path = layout.config_dir / "train_config.json"
    train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8")) if train_cfg_path.is_file() else {}
    gate_run_id = str(train_cfg.get("gate_run_id", ""))
    vit_name = str(train_cfg.get("vit_name", "dinov2_vits14"))
    input_size = int(train_cfg.get("input_size", 224))

    hist_path = layout.posttrain / "training_history.json"
    if not hist_path.is_file():
        hist_path = layout.pretrain / "training_state.json"
        state = json.loads(hist_path.read_text(encoding="utf-8"))
        history = state.get("history", {})
    else:
        history = json.loads(hist_path.read_text(encoding="utf-8"))

    meta_path = layout.posttrain / "STAGE2_PIXEL_RUN_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {"run_id": layout.run_root.name}

    best_path = layout.posttrain / "best_metrics.json"
    best_metrics = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else {}

    global_ckpt = paths.root / "models" / "checkpoints" / "stage2_am"
    ckpt = resolve_best_checkpoint(layout, global_ckpt_dir=global_ckpt)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint en {layout.posttrain}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pixel_morph_vit(backbone_name=vit_name, freeze_backbone=True)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.to(device).eval()

    h5_path, _, lookup_path = _cache_paths(paths.root)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Falta HDF5 Stage2-Pixel: {h5_path}")
    h5_store = Stage2PixelH5Store(h5_path, lookup_path)

    _, val_df, test_df, split_info = load_mplus_splits()
    meta["split_info"] = meta.get("split_info") or split_info

    morph_params = PixelMorphParams()
    artifact_paths: dict[str, str] = {}

    phase_log("Evaluando val holdout...")
    val_metrics, val_tiles = evaluate_split_tiles(model, val_df, h5_store, device=device, batch_size=batch_size)
    val_tiles["split"] = "val"
    val_metrics["split"] = "val"

    phase_log("Evaluando test holdout...")
    test_metrics, test_tiles = evaluate_split_tiles(model, test_df, h5_store, device=device, batch_size=batch_size)
    test_tiles["split"] = "test"
    test_metrics["split"] = "test"

    # Tablas holdout
    v_csv, v_pq = _write_table(val_tiles, tables_dir / "tile_predictions_val")
    t_csv, t_pq = _write_table(test_tiles, tables_dir / "tile_predictions_test")
    artifact_paths["tile_predictions_val"] = str(v_pq)
    artifact_paths["tile_predictions_test"] = str(t_pq)

    val_summary = pd.DataFrame([{k: v for k, v in val_metrics.items() if k != "confusion"}])
    test_summary = pd.DataFrame([{k: v for k, v in test_metrics.items() if k != "confusion"}])
    _write_table(val_summary, tables_dir / "holdout_val_summary")
    _write_table(test_summary, tables_dir / "holdout_test_summary")

    pc_val = pd.DataFrame([val_metrics.get("per_class_iou", {})])
    pc_test = pd.DataFrame([test_metrics.get("per_class_iou", {})])
    _write_table(pc_val, tables_dir / "per_class_iou_val")
    _write_table(pc_test, tables_dir / "per_class_iou_test")

    eval_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": layout.run_root.name,
        "checkpoint": str(ckpt),
        "val": {k: v for k, v in val_metrics.items() if k != "confusion"},
        "test": {k: v for k, v in test_metrics.items() if k != "confusion"},
    }
    _write_json(report_dir / "evaluation_metrics_summary.json", eval_json)
    _write_json(layout.run_root / "evaluation_metrics_summary_val.json", val_metrics)
    _write_json(layout.run_root / "evaluation_metrics_summary_test.json", test_metrics)

    # Plots
    plot_val_miou_curve(history, plots_dir / "val_miou_curve.png")
    plot_train_loss_curve(history, plots_dir / "train_loss_curve.png")
    plot_per_class_iou_curves(history, plots_dir / "per_class_iou_curves.png")
    plot_per_class_iou_bars(
        val_metrics.get("per_class_iou", {}),
        plots_dir / "per_class_iou_val_best.png",
        title="Per-class IoU — val (best ckpt)",
    )
    plot_per_class_iou_bars(
        test_metrics.get("per_class_iou", {}),
        plots_dir / "per_class_iou_test_best.png",
        title="Per-class IoU — test (best ckpt)",
    )
    if val_metrics.get("confusion"):
        plot_confusion_matrix(val_metrics["confusion"], plots_dir / "confusion_matrix_val.png", title="Confusion — val")
    if test_metrics.get("confusion"):
        plot_confusion_matrix(test_metrics["confusion"], plots_dir / "confusion_matrix_test.png", title="Confusion — test")
    if best_metrics.get("per_class_iou_at_best"):
        plot_per_class_iou_bars(
            best_metrics["per_class_iou_at_best"],
            visuals_dir / "per_class_iou_best.png",
            title=f"Per-class IoU @ best epoch {best_metrics.get('best_epoch')}",
        )

    # Visualizaciones tile-level
    phase_log("Generando tile audit grid...")
    audit_png = visuals_dir / "tile_audit_best_ckpt.png"
    plot_tile_morph_audit_grid(
        model, val_df, h5_store, device=device, out_png=audit_png, n_per_class=n_audit_per_class
    )
    artifact_paths["tile_audit"] = str(audit_png)

    phase_log("Generando muestras tile val/test...")
    val_samples = render_sample_tile_panels(
        model,
        val_df,
        h5_store,
        device=device,
        out_dir=visuals_dir / "val",
        n_tiles=n_sample_tiles,
        split_name="val",
        with_explain_panels=with_explain_panels,
        attention_layers=attention_layers,
        input_size=input_size,
    )
    test_samples = render_sample_tile_panels(
        model,
        test_df,
        h5_store,
        device=device,
        out_dir=visuals_dir / "test",
        n_tiles=n_sample_tiles,
        split_name="test",
        with_explain_panels=with_explain_panels,
        attention_layers=attention_layers,
        input_size=input_size,
    )
    artifact_paths["val_tile_samples"] = str(visuals_dir / "val")
    artifact_paths["test_tile_samples"] = str(visuals_dir / "test")

    if not skip_fullimage and gate_run_id:
        val_imgs = sorted(val_df["image_path"].unique())[:max_fullimage_val]
        test_imgs = sorted(test_df["image_path"].unique())[:max_fullimage_test]
        if require_gate_cache:
            from .stage2_gate_infer import assert_gate_embed_cache_ready

            gate_images = pd.DataFrame(
                {"image_path": list(val_imgs) + list(test_imgs)}
            )
            phase_log(
                "Verificando gate embed cache (prerrequisito Modelo 1 secuencial)..."
            )
            assert_gate_embed_cache_ready(gate_images)
        phase_log(
            f"Generando mapas full-image secuenciales Gate->Stage2 "
            f"({len(val_imgs)} val + {len(test_imgs)} test)..."
        )
        render_fullimage_overlays(
            image_paths=list(val_imgs),
            gate_run_id=gate_run_id,
            model=model,
            device=device,
            morph_params=morph_params,
            input_size=input_size,
            out_dir=visuals_dir / "val" / "fullimage",
            split_name="val",
            gate_strict=gate_strict,
        )
        render_fullimage_overlays(
            image_paths=list(test_imgs),
            gate_run_id=gate_run_id,
            model=model,
            device=device,
            morph_params=morph_params,
            input_size=input_size,
            out_dir=visuals_dir / "test" / "fullimage",
            split_name="test",
            gate_strict=gate_strict,
        )
        artifact_paths["val_fullimage"] = str(visuals_dir / "val" / "fullimage")
        artifact_paths["test_fullimage"] = str(visuals_dir / "test" / "fullimage")

    h5_store.close()

    # Markdown reports
    atomic_write_text(report_dir / "training_report_val.md", "\n".join(_metrics_table_md(val_metrics, split="val")))
    atomic_write_text(report_dir / "training_report_test.md", "\n".join(_metrics_table_md(test_metrics, split="test")))
    write_run_status_md(layout, history=history, best_metrics=best_metrics, val_metrics=val_metrics, test_metrics=test_metrics)
    summary_path = write_summary_md(
        layout, meta=meta, history=history, val_metrics=val_metrics, test_metrics=test_metrics, artifact_paths=artifact_paths
    )

    atomic_write_text(layout.run_root / "RUN_STATUS.md", (layout.posttrain / "report" / "RUN_STATUS.md").read_text(encoding="utf-8"))

    phase_log(f"Informe posttrain OK -> {summary_path}")
    return report_dir
