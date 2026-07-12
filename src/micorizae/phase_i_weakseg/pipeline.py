"""Fase I — Segmentacion semantica debilmente supervisada (morfologica).

Convierte etiquetas por tile (AMFinder / manifest) en mascaras continuas
basadas en forma y textura, priorizando estructura sobre color.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from micorizae.morph_core import (
    CLASS_ARBUSCULE,
    CLASS_BG,
    CLASS_COLONY,
    CLASS_COLORS,
    CLASS_HYPHAE,
    CLASS_NAMES,
    CLASS_ROOT,
    CLASS_VESICLE,
    N_CLASSES,
    WeakSegParams,
    segment_tile,
    segment_tile_stain_aware,
)
from micorizae.morph_core.stain import (
    ambiguous_stain_mask as _ambiguous_stain_mask,
    stain_maps as _stain_maps,
)

from ..common.io import read_table, write_table
from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..common.run_outputs import RunOutputs

log = get_logger("phase_i_weakseg")

TILE_LABEL_COLORS = {
    "AMColonised": (0, 200, 255),
    "Hybrid": (255, 140, 40),
    "Uncolonised": (210, 180, 140),
    "MainRoot": (170, 140, 120),
    "DSE": (255, 80, 180),
    "Background": (70, 70, 70),
    "Unreadable": (180, 100, 200),
    "Mplus": (0, 200, 255),
    "Mminus": (210, 180, 140),
}

STAGE1_TILE_COLORS = {
    "Mplus": (0, 200, 255),
    "Mminus": (210, 180, 140),
    "Background": (70, 70, 70),
    "Unreadable": (180, 100, 200),
}

STAGE2_TILE_COLORS = {
    "AMColonised": (0, 200, 255),
    "Hybrid": (255, 140, 40),
    "BlueCoils": (60, 120, 255),
    "BrownCoils": (150, 80, 40),
    "TypeTwo": (0, 220, 140),
    "HybridErm": (255, 160, 0),
    "HybridDse": (220, 80, 180),
}

RAW_PRIORITY = [
    "AMColonised",
    "Hybrid",
    "BlueCoils",
    "BrownCoils",
    "TypeTwo",
    "HybridErm",
    "HybridDse",
    "DSE",
    "MainRoot",
    "Uncolonised",
    "Background",
    "Unreadable",
]

STAGE2_KNOWN = {"AMColonised", "Hybrid", "BlueCoils", "BrownCoils", "TypeTwo", "HybridErm", "HybridDse"}


def _load_image_rgb(image_path: Path) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(image_path).convert("RGB"))


def _safe_slice(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=img.dtype)
    return img[y0:y1, x0:x1]


def _image_size(path: Path) -> tuple[int, int]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def _is_positive_row(row: pd.Series) -> bool:
    if "stage1" in row:
        return str(row["stage1"]).strip().lower() == "mplus"
    if "positive" in row:
        v = row["positive"]
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y", "mplus", "positive"}
        return bool(v)
    return False


def _is_annotated_root_row(row: pd.Series) -> bool:
    if "stage1" in row:
        s = str(row["stage1"]).strip().lower()
        return s in {"mplus", "mminus"}
    if "positive" in row:
        return _is_positive_row(row)
    return False


def _build_stage1_from_legacy_columns(df: pd.DataFrame) -> pd.Series:
    bg = df.get("Background", 0)
    unr = df.get("Unreadable", 0)
    mplus = 0
    for c in ("AMColonised", "Hybrid", "BlueCoils", "BrownCoils", "TypeTwo", "HybridErm", "HybridDse"):
        if c in df.columns:
            mplus = mplus + df[c].fillna(0).astype(int)
    mminus = 0
    for c in ("Uncolonised", "MainRoot", "DSE"):
        if c in df.columns:
            mminus = mminus + df[c].fillna(0).astype(int)
    stage1 = np.full(len(df), "Background", dtype=object)
    stage1[(mminus > 0).to_numpy()] = "Mminus"
    stage1[(mplus > 0).to_numpy()] = "Mplus"
    stage1[bg.fillna(0).astype(int).to_numpy() > 0] = "Background"
    stage1[unr.fillna(0).astype(int).to_numpy() > 0] = "Unreadable"
    return pd.Series(stage1, index=df.index)


def _build_raw_label_from_legacy_columns(df: pd.DataFrame) -> pd.Series:
    labels = np.full(len(df), "Unknown", dtype=object)
    for cls in RAW_PRIORITY:
        if cls in df.columns:
            on = df[cls].fillna(0).astype(int).to_numpy() > 0
            labels[on] = cls
    return pd.Series(labels, index=df.index)


def _derive_stage2_label(raw_label: str, stage2_value: object) -> str | None:
    if stage2_value is not None and str(stage2_value).strip() and str(stage2_value).lower() != "nan":
        return str(stage2_value)
    rl = str(raw_label).strip()
    if rl in STAGE2_KNOWN:
        return rl
    return None


def _infer_tile_size_from_annotation_grid(image_path: Path, df: pd.DataFrame, fallback: int) -> int:
    if "row" not in df.columns or "col" not in df.columns or df.empty:
        return int(fallback)
    n_rows = int(df["row"].max()) + 1
    n_cols = int(df["col"].max()) + 1
    if n_rows <= 0 or n_cols <= 0:
        return int(fallback)
    full_cells = n_rows * n_cols
    dense = (
        df["row"].nunique() >= max(1, int(0.9 * n_rows))
        and df["col"].nunique() >= max(1, int(0.9 * n_cols))
        and len(df) >= int(0.8 * full_cells)
    )
    if not dense:
        return int(fallback)
    w, h = _image_size(image_path)
    ts_x = w / float(n_cols)
    ts_y = h / float(n_rows)
    ts = int(round((ts_x + ts_y) / 2.0))
    if ts < 16:
        return int(fallback)
    if abs(ts - int(fallback)) >= max(8, int(0.1 * max(1, fallback))):
        log.warning(
            f"tile_size override por grilla de anotacion: user={fallback} -> inferido={ts} "
            f"(image={w}x{h}, grid={n_rows}x{n_cols})"
        )
    return ts


def _load_positive_tiles_from_annotations(
    *,
    image_path: Path,
    image_rel: str,
    annotations_path: Path,
    tile_size: int,
) -> pd.DataFrame:
    if annotations_path.suffix.lower() == ".xml":
        import xml.etree.ElementTree as et

        root = et.parse(annotations_path).getroot()
        rows = []
        for node in root.findall(".//*"):
            row = node.attrib.get("row")
            col = node.attrib.get("col")
            positive = node.attrib.get("positive", node.attrib.get("label", ""))
            if row is None or col is None:
                continue
            is_pos = str(positive).strip().lower() in {"1", "true", "yes", "mplus", "positive"}
            rows.append({"row": int(row), "col": int(col), "positive": is_pos})
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(annotations_path)

    if "row" not in df.columns or "col" not in df.columns:
        raise ValueError(f"Anotacion sin columnas row/col: {annotations_path}")
    tile_size = _infer_tile_size_from_annotation_grid(image_path=image_path, df=df, fallback=tile_size)
    # Conservamos todas las anotaciones de raiz (M+ y M-) para reconstruir plano de raiz.
    if "positive" not in df.columns and "stage1" not in df.columns:
        if any(c in df.columns for c in ("AMColonised", "Uncolonised", "Background", "Unreadable")):
            df["stage1"] = _build_stage1_from_legacy_columns(df)
            df["raw_label"] = _build_raw_label_from_legacy_columns(df)
        else:
            raise ValueError("Anotacion requiere columna 'positive', 'stage1' o esquema AMColonised/Uncolonised")

    out = df[["row", "col"]].copy().reset_index(drop=True)
    out["row"] = out["row"].astype(int)
    out["col"] = out["col"].astype(int)
    out["image_path"] = image_rel
    out["tile_size"] = int(tile_size)
    out["x0"] = out["col"] * int(tile_size)
    out["y0"] = out["row"] * int(tile_size)
    out["x1"] = out["x0"] + int(tile_size)
    out["y1"] = out["y0"] + int(tile_size)
    if "stage1" in df.columns:
        out = out.join(df[["stage1"]].reset_index(drop=True))
    else:
        out["stage1"] = np.where(
            df["positive"].reset_index(drop=True).astype(bool), "Mplus", "Mminus"
        )
    if "stage2" in df.columns:
        out = out.join(df[["stage2"]].reset_index(drop=True))
    if "raw_label" in df.columns:
        out = out.join(df[["raw_label"]].reset_index(drop=True))
    elif "stage2" in out.columns:
        out["raw_label"] = out["stage2"].where(out["stage2"].notna(), out["stage1"])
    else:
        out["raw_label"] = out["stage1"]
    out["stage2_label"] = [
        _derive_stage2_label(r, s if "stage2" in out.columns else None)
        for r, s in zip(out["raw_label"].astype(str), out.get("stage2", pd.Series([None] * len(out))))
    ]
    return out


def load_annotated_tiles_index(
    *,
    image_path: Path,
    manifests_dir: Path,
    tile_size: int,
    annotations_path: Path | None = None,
) -> pd.DataFrame:
    root = get_paths().root
    try:
        image_rel = image_path.resolve().relative_to(root).as_posix()
    except ValueError:
        image_rel = image_path.as_posix()
    if annotations_path is not None:
        return _load_positive_tiles_from_annotations(
            image_path=image_path,
            image_rel=image_rel,
            annotations_path=annotations_path,
            tile_size=tile_size,
        )

    try:
        base = read_table(manifests_dir / "tiles_index")
    except FileNotFoundError:
        base = read_table(manifests_dir / "manifest_labels")

    if "image_path" not in base.columns:
        raise ValueError("Manifest sin columna image_path")

    sub = base[base["image_path"] == image_rel].copy()
    if sub.empty:
        raise ValueError(f"No hay filas para {image_rel} en manifests")

    if "x0" not in sub.columns or "y0" not in sub.columns:
        sub["tile_size"] = int(tile_size)
        sub["x0"] = sub["col"].astype(int) * int(tile_size)
        sub["y0"] = sub["row"].astype(int) * int(tile_size)
        sub["x1"] = sub["x0"] + int(tile_size)
        sub["y1"] = sub["y0"] + int(tile_size)

    ann = sub.copy()
    if "raw_label" not in ann.columns:
        if "stage2" in ann.columns:
            ann["raw_label"] = ann["stage2"].where(ann["stage2"].notna(), ann["stage1"])
        else:
            ann["raw_label"] = ann["stage1"]
    ann["stage2_label"] = [
        _derive_stage2_label(r, s if "stage2" in ann.columns else None)
        for r, s in zip(ann["raw_label"].astype(str), ann.get("stage2", pd.Series([None] * len(ann))))
    ]
    return ann.reset_index(drop=True)


def _tile_root_mask_stain(m: dict[str, np.ndarray], p: WeakSegParams) -> np.ndarray:
    """Tejido = no fondo blanco. Delega a ``detectors.bg_root``."""
    from micorizae.phase_e_stage2.detectors import detect_root

    return detect_root(m, p).mask


def _place_mask(canvas: np.ndarray, patch: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    h, w = canvas.shape
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return
    ph = y1 - y0
    pw = x1 - x0
    if patch.shape[0] != ph or patch.shape[1] != pw:
        patch = patch[:ph, :pw]
    canvas[y0:y1, x0:x1] |= patch.astype(canvas.dtype)


def consolidate_with_precedence(
    root: np.ndarray,
    colony: np.ndarray,
    hyphae: np.ndarray,
    vesicle: np.ndarray,
    arbuscule: np.ndarray,
) -> np.ndarray:
    seg = np.zeros_like(root, dtype=np.uint8)
    seg[root > 0] = CLASS_ROOT
    seg[colony > 0] = CLASS_COLONY
    seg[hyphae > 0] = CLASS_HYPHAE
    seg[vesicle > 0] = CLASS_VESICLE
    seg[arbuscule > 0] = CLASS_ARBUSCULE
    return seg


def smooth_seams(seg_map: np.ndarray, sigma: float) -> np.ndarray:
    probs = np.stack([(seg_map == i).astype(np.float32) for i in range(N_CLASSES)], axis=0)
    for i in range(1, N_CLASSES):
        probs[i] = gaussian_filter(probs[i], sigma=sigma)
    return np.argmax(probs, axis=0).astype(np.uint8)


def render_overlay(image_rgb: np.ndarray, seg_map: np.ndarray, alpha: float) -> np.ndarray:
    color = np.zeros_like(image_rgb, dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color[seg_map == cls] = np.array(rgb, dtype=np.uint8)
    out = cv2.addWeighted(image_rgb, 1.0 - alpha, color, alpha, gamma=0.0)
    return out


def render_segmentation_color(seg_map: np.ndarray) -> np.ndarray:
    color = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color[seg_map == cls] = np.array(rgb, dtype=np.uint8)
    return color


def render_binary_plane_rgba(mask: np.ndarray, color: tuple[int, int, int], alpha: int = 170) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    on = mask > 0
    rgba[on, 0] = color[0]
    rgba[on, 1] = color[1]
    rgba[on, 2] = color[2]
    rgba[on, 3] = alpha
    return rgba


def render_labeled_tiles(
    *,
    shape: tuple[int, int],
    tiles_df: pd.DataFrame,
    label_col: str,
    color_map: dict[str, tuple[int, int, int]],
    alpha: int = 150,
) -> np.ndarray:
    h, w = shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for rec in tiles_df.itertuples(index=False):
        label = str(getattr(rec, label_col, "Unknown"))
        if not label or label.lower() == "nan":
            continue
        color = color_map.get(label)
        if color is None:
            continue
        x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        rgba[y0:y1, x0:x1, 0] = color[0]
        rgba[y0:y1, x0:x1, 1] = color[1]
        rgba[y0:y1, x0:x1, 2] = color[2]
        rgba[y0:y1, x0:x1, 3] = alpha
    return rgba


def render_tile_label_overlay(
    *,
    image_rgb: np.ndarray,
    tiles_df: pd.DataFrame,
    alpha: float = 0.32,
) -> np.ndarray:
    color = np.zeros_like(image_rgb, dtype=np.uint8)
    for rec in tiles_df.itertuples(index=False):
        label = str(getattr(rec, "raw_label", getattr(rec, "stage1", "Unknown")))
        x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
        rgb = TILE_LABEL_COLORS.get(label, (120, 120, 120))
        y0c, y1c = max(0, y0), min(image_rgb.shape[0], y1)
        x0c, x1c = max(0, x0), min(image_rgb.shape[1], x1)
        if y1c > y0c and x1c > x0c:
            color[y0c:y1c, x0c:x1c] = np.array(rgb, dtype=np.uint8)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, color, alpha, gamma=0.0)


def _downscale_preview(img: np.ndarray, max_w: int = 1800, max_h: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if s >= 1.0:
        return img
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _legend_strip(items: list[tuple[str, tuple[int, int, int], str]], width: int) -> np.ndarray:
    row_h = 24
    h = 6 + len(items) * row_h + 6
    img = np.full((h, width, 3), 255, dtype=np.uint8)
    for i, (abbr, color, desc) in enumerate(items):
        y = 6 + i * row_h
        cv2.rectangle(img, (8, y + 3), (24, y + 19), color, thickness=-1)
        cv2.rectangle(img, (8, y + 3), (24, y + 19), (0, 0, 0), thickness=1)
        cv2.putText(img, f"{abbr} = {desc}", (32, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
    return img


def _compose_rgba_on_rgb(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    ov = overlay_rgba[..., :3].astype(np.float32)
    a = (overlay_rgba[..., 3:4].astype(np.float32) / 255.0)
    out = ov * a + base * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_legend_inset(
    image_rgb: np.ndarray,
    *,
    title: str,
    items: list[tuple[str, tuple[int, int, int], str]],
) -> np.ndarray:
    out = image_rgb.copy()
    x0, y0 = 14, 14
    row_h = 24
    pad = 10
    box_w = 640
    box_h = 36 + row_h * len(items) + pad
    x1 = min(out.shape[1] - 8, x0 + box_w)
    y1 = min(out.shape[0] - 8, y0 + box_h)
    cv2.rectangle(out, (x0, y0), (x1, y1), (250, 250, 250), thickness=-1)
    cv2.rectangle(out, (x0, y0), (x1, y1), (20, 20, 20), thickness=1)
    cv2.putText(out, title, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    yy = y0 + 36
    for abbr, color, desc in items:
        cv2.rectangle(out, (x0 + 10, yy - 12), (x0 + 26, yy + 4), color, thickness=-1)
        cv2.rectangle(out, (x0 + 10, yy - 12), (x0 + 26, yy + 4), (0, 0, 0), thickness=1)
        cv2.putText(out, f"{abbr}: {desc}", (x0 + 34, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 1, cv2.LINE_AA)
        yy += row_h
    return out


def build_comparison_panel(
    *,
    image_rgb: np.ndarray,
    overlay_morph: np.ndarray,
    seg_color: np.ndarray,
    overlay_tiles: np.ndarray,
) -> np.ndarray:
    a = _downscale_preview(image_rgb)
    b = _downscale_preview(overlay_morph)
    c = _downscale_preview(seg_color)
    d = _downscale_preview(overlay_tiles)
    h = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
    top = np.hstack([a, b])
    bot = np.hstack([c, d])
    panel = np.vstack([top, bot])
    cv2.putText(panel, "Original", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Overlay Morfologico", (w + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Mascara Color (0..5)", (12, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Overlay Etiqueta Tile", (w + 12, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel

def export_patch_dataset(
    *,
    image_rgb: np.ndarray,
    seg_map: np.ndarray,
    run: RunOutputs,
    patch_size: int = 518,
    stride: int = 518,
    min_fg_ratio: float = 0.002,
) -> Path:
    root = run.root / "dino_patches"
    img_dir = root / "images"
    msk_dir = root / "masks"
    msk_vis_dir = root / "masks_vis"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    msk_vis_dir.mkdir(parents=True, exist_ok=True)

    h, w = seg_map.shape
    rows = []
    idx = 0
    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            y1 = min(y + patch_size, h)
            x1 = min(x + patch_size, w)
            if (y1 - y) < patch_size or (x1 - x) < patch_size:
                continue
            img_patch = image_rgb[y:y1, x:x1]
            msk_patch = seg_map[y:y1, x:x1]
            fg_ratio = float((msk_patch > 0).mean())
            if fg_ratio < float(min_fg_ratio):
                continue
            stem = f"p{idx:06d}_x{x}_y{y}"
            Image.fromarray(img_patch).save(img_dir / f"{stem}.png")
            Image.fromarray(msk_patch).save(msk_dir / f"{stem}.png")
            Image.fromarray(render_segmentation_color(msk_patch)).save(msk_vis_dir / f"{stem}.png")
            rows.append({"patch_id": stem, "x0": x, "y0": y, "x1": x1, "y1": y1, "fg_ratio": fg_ratio})
            idx += 1

    patch_index = pd.DataFrame(rows)
    path = write_table(patch_index, run.tables / "dino_patch_index")
    return path


def run_weakseg_pipeline(
    *,
    image_path: Path,
    manifests_dir: Path,
    params: WeakSegParams,
    annotations_path: Path | None = None,
    export_dino_patches: bool = True,
    patch_size: int = 518,
    patch_stride: int = 518,
) -> RunOutputs:
    run = RunOutputs.create("weakseg_morph", suffix=image_path.stem)
    image_rgb = _load_image_rgb(image_path)
    h, w = image_rgb.shape[:2]

    ann_tiles = load_annotated_tiles_index(
        image_path=image_path,
        manifests_dir=manifests_dir,
        tile_size=params.tile_size,
        annotations_path=annotations_path,
    )

    if "tile_size" in ann_tiles.columns and not ann_tiles["tile_size"].isna().all():
        params.tile_size = int(ann_tiles["tile_size"].iloc[0])

    proc_tiles = ann_tiles[ann_tiles.apply(_is_annotated_root_row, axis=1)].copy()
    if proc_tiles.empty:
        log.warning("No hay tiles M+/M- para esculpido morfologico; usando todos los tiles como fallback.")
        proc_tiles = ann_tiles.copy()

    root_canvas = np.zeros((h, w), dtype=np.uint8)
    colony_canvas = np.zeros((h, w), dtype=np.uint8)
    hyphae_canvas = np.zeros((h, w), dtype=np.uint8)
    vesicle_canvas = np.zeros((h, w), dtype=np.uint8)
    arbuscule_canvas = np.zeros((h, w), dtype=np.uint8)

    rows = []
    for rec in proc_tiles.itertuples(index=False):
        x0 = int(getattr(rec, "x0"))
        y0 = int(getattr(rec, "y0"))
        x1 = int(getattr(rec, "x1"))
        y1 = int(getattr(rec, "y1"))
        stage1 = str(getattr(rec, "stage1", "Mminus"))
        stage2 = str(getattr(rec, "stage2", "")) if hasattr(rec, "stage2") else ""
        raw_label = str(getattr(rec, "raw_label", stage2 if stage2 else stage1))
        is_mplus = stage1.strip().lower() == "mplus"
        tile = _safe_slice(image_rgb, x0, y0, x1, y1)
        if tile.size == 0:
            continue
        m = segment_tile(tile, params)
        _place_mask(root_canvas, m["root"].astype(np.uint8), x0, y0, x1, y1)
        if is_mplus:
            _place_mask(colony_canvas, m["root"].astype(np.uint8), x0, y0, x1, y1)
            _place_mask(hyphae_canvas, m["hyphae"].astype(np.uint8), x0, y0, x1, y1)
            _place_mask(vesicle_canvas, m["vesicle"].astype(np.uint8), x0, y0, x1, y1)
            _place_mask(arbuscule_canvas, m["arbuscule"].astype(np.uint8), x0, y0, x1, y1)
        rows.append(
            {
                "row": int(getattr(rec, "row")),
                "col": int(getattr(rec, "col")),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "stage1": stage1,
                "stage2": stage2,
                "raw_label": raw_label,
                "root_cov": float(m["root"].mean()),
                "colony_cov": float(m["root"].mean() if is_mplus else 0.0),
                "hyphae_cov": float(m["hyphae"].mean()),
                "vesicle_cov": float(m["vesicle"].mean()),
                "arbuscule_cov": float(m["arbuscule"].mean()),
            }
        )

    seg_raw = consolidate_with_precedence(
        root_canvas,
        colony_canvas,
        hyphae_canvas,
        vesicle_canvas,
        arbuscule_canvas,
    )
    seg_smooth = smooth_seams(seg_raw, sigma=params.seam_sigma)
    overlay = render_overlay(image_rgb, seg_smooth, alpha=params.alpha_overlay)
    seg_color = render_segmentation_color(seg_smooth)
    tile_overlay = render_tile_label_overlay(image_rgb=image_rgb, tiles_df=ann_tiles, alpha=0.30)
    panel = build_comparison_panel(
        image_rgb=image_rgb,
        overlay_morph=overlay,
        seg_color=seg_color,
        overlay_tiles=tile_overlay,
    )

    np.savez_compressed(
        run.tables / f"{image_path.stem}__weakseg_masks.npz",
        root=root_canvas.astype(np.uint8),
        colony=colony_canvas.astype(np.uint8),
        hyphae=hyphae_canvas.astype(np.uint8),
        vesicle=vesicle_canvas.astype(np.uint8),
        arbuscule=arbuscule_canvas.astype(np.uint8),
        seg_raw=seg_raw.astype(np.uint8),
        seg_smooth=seg_smooth.astype(np.uint8),
    )
    Image.fromarray(seg_smooth).save(run.maps / f"{image_path.stem}__seg_morph.png")
    Image.fromarray((seg_smooth * (255 // (N_CLASSES - 1))).astype(np.uint8)).save(
        run.maps / f"{image_path.stem}__seg_morph_gray.png"
    )
    Image.fromarray(seg_color).save(run.maps / f"{image_path.stem}__seg_morph_color.png")
    Image.fromarray(overlay).save(run.maps / f"{image_path.stem}__overlay_alpha.png")
    Image.fromarray(tile_overlay).save(run.maps / f"{image_path.stem}__tile_labels_overlay.png")
    Image.fromarray(panel).save(run.maps / f"{image_path.stem}__comparison_panel.png")

    Image.fromarray((root_canvas * 255).astype(np.uint8)).save(run.maps / f"{image_path.stem}__plane_root.png")
    Image.fromarray((colony_canvas * 255).astype(np.uint8)).save(run.maps / f"{image_path.stem}__plane_colony.png")
    Image.fromarray((hyphae_canvas * 255).astype(np.uint8)).save(run.maps / f"{image_path.stem}__plane_hyphae.png")
    Image.fromarray((vesicle_canvas * 255).astype(np.uint8)).save(run.maps / f"{image_path.stem}__plane_vesicle.png")
    Image.fromarray((arbuscule_canvas * 255).astype(np.uint8)).save(run.maps / f"{image_path.stem}__plane_arbuscule.png")
    Image.fromarray(render_binary_plane_rgba(root_canvas, CLASS_COLORS[CLASS_ROOT], alpha=165)).save(
        run.maps / f"{image_path.stem}__plane_root_rgba.png"
    )
    Image.fromarray(render_binary_plane_rgba(colony_canvas, CLASS_COLORS[CLASS_COLONY], alpha=165)).save(
        run.maps / f"{image_path.stem}__plane_colony_rgba.png"
    )
    Image.fromarray(render_binary_plane_rgba(hyphae_canvas, CLASS_COLORS[CLASS_HYPHAE], alpha=185)).save(
        run.maps / f"{image_path.stem}__plane_hyphae_rgba.png"
    )
    Image.fromarray(render_binary_plane_rgba(vesicle_canvas, CLASS_COLORS[CLASS_VESICLE], alpha=190)).save(
        run.maps / f"{image_path.stem}__plane_vesicle_rgba.png"
    )
    Image.fromarray(render_binary_plane_rgba(arbuscule_canvas, CLASS_COLORS[CLASS_ARBUSCULE], alpha=190)).save(
        run.maps / f"{image_path.stem}__plane_arbuscule_rgba.png"
    )

    stage1_rgba = render_labeled_tiles(
        shape=(h, w),
        tiles_df=ann_tiles,
        label_col="stage1",
        color_map=STAGE1_TILE_COLORS,
        alpha=130,
    )
    stage2_tiles = ann_tiles[ann_tiles["stage2_label"].notna()].copy() if "stage2_label" in ann_tiles.columns else ann_tiles.iloc[0:0].copy()
    stage2_rgba = render_labeled_tiles(
        shape=(h, w),
        tiles_df=stage2_tiles,
        label_col="stage2_label",
        color_map=STAGE2_TILE_COLORS,
        alpha=145,
    )
    Image.fromarray(stage1_rgba).save(run.maps / f"{image_path.stem}__stage1_tiles_rgba.png")
    Image.fromarray(stage2_rgba).save(run.maps / f"{image_path.stem}__stage2_tiles_rgba.png")

    seg_items = [
        ("0 BG", CLASS_COLORS[CLASS_BG], "fondo"),
        ("1 ROOT", CLASS_COLORS[CLASS_ROOT], "tejido radicular"),
        ("2 COLONY", CLASS_COLORS[CLASS_COLONY], "M+"),
        ("3 IH", CLASS_COLORS[CLASS_HYPHAE], "hifa intraradical"),
        ("4 V", CLASS_COLORS[CLASS_VESICLE], "vesicula"),
        ("5 A", CLASS_COLORS[CLASS_ARBUSCULE], "arbusculo"),
    ]
    stage1_items = [
        ("Mplus", STAGE1_TILE_COLORS["Mplus"], "tile colonizado"),
        ("Mminus", STAGE1_TILE_COLORS["Mminus"], "tile no colonizado"),
        ("Background", STAGE1_TILE_COLORS["Background"], "fondo"),
        ("Unreadable", STAGE1_TILE_COLORS["Unreadable"], "ilegible"),
    ]
    stage2_items = [
        ("AMColonised", STAGE2_TILE_COLORS["AMColonised"], "subclase AM"),
        ("Hybrid", STAGE2_TILE_COLORS["Hybrid"], "subclase AM"),
        ("BlueCoils", STAGE2_TILE_COLORS["BlueCoils"], "subclase ERM"),
        ("BrownCoils", STAGE2_TILE_COLORS["BrownCoils"], "subclase ERM"),
        ("TypeTwo", STAGE2_TILE_COLORS["TypeTwo"], "subclase ERM"),
        ("HybridErm", STAGE2_TILE_COLORS["HybridErm"], "subclase ERM"),
        ("HybridDse", STAGE2_TILE_COLORS["HybridDse"], "subclase ERM"),
    ]

    seg_color_labeled = _draw_legend_inset(seg_color, title="Segmentacion morfologica", items=seg_items)
    Image.fromarray(seg_color_labeled).save(run.maps / f"{image_path.stem}__seg_morph_color_con_leyenda.png")

    stage1_preview = _compose_rgba_on_rgb(image_rgb, stage1_rgba)
    stage1_labeled = _draw_legend_inset(stage1_preview, title="Stage 1 tiles", items=stage1_items)
    Image.fromarray(stage1_labeled).save(run.maps / f"{image_path.stem}__stage1_tiles_con_leyenda.png")

    stage2_preview = _compose_rgba_on_rgb(image_rgb, stage2_rgba)
    stage2_labeled = _draw_legend_inset(stage2_preview, title="Stage 2 tiles (solo M+)", items=stage2_items)
    Image.fromarray(stage2_labeled).save(run.maps / f"{image_path.stem}__stage2_tiles_con_leyenda.png")

    plane_defs = [
        ("root_tejido_radicular", root_canvas, CLASS_COLORS[CLASS_ROOT], "Raiz"),
        ("colony_colonia_mplus", colony_canvas, CLASS_COLORS[CLASS_COLONY], "Colonia M+"),
        ("hyphae_hifa_intrarradical", hyphae_canvas, CLASS_COLORS[CLASS_HYPHAE], "Hifa intraradical"),
        ("vesicle_vesicula", vesicle_canvas, CLASS_COLORS[CLASS_VESICLE], "Vesicula"),
        ("arbuscule_arbusculo", arbuscule_canvas, CLASS_COLORS[CLASS_ARBUSCULE], "Arbusculo"),
    ]
    for slug, plane, color, label in plane_defs:
        preview = _compose_rgba_on_rgb(image_rgb, render_binary_plane_rgba(plane, color, alpha=185))
        labeled = _draw_legend_inset(preview, title=f"Plano {label}", items=[(label, color, "mascara de clase")])
        Image.fromarray(labeled).save(run.maps / f"{image_path.stem}__plane_{slug}_con_leyenda.png")

    per_tile = pd.DataFrame(rows)
    write_table(per_tile, run.tables / f"{image_path.stem}__tile_metrics")

    patch_index = None
    if export_dino_patches:
        patch_index = export_patch_dataset(
            image_rgb=image_rgb,
            seg_map=seg_smooth,
            run=run,
            patch_size=patch_size,
            stride=patch_stride,
            min_fg_ratio=params.min_patch_fg_ratio,
        )

    counts = {int(i): int((seg_smooth == i).sum()) for i in range(N_CLASSES)}
    raw_counts = per_tile["raw_label"].value_counts().to_dict() if not per_tile.empty else {}
    stage1_counts = ann_tiles["stage1"].value_counts().to_dict() if "stage1" in ann_tiles.columns else {}
    stage2_counts = (
        ann_tiles["stage2_label"].dropna().value_counts().to_dict()
        if "stage2_label" in ann_tiles.columns
        else {}
    )
    lines = [
        f"# WeakSeg Morfologico — {run.run_id}",
        "",
        "## Entrada",
        f"- Imagen: `{image_path}`",
        f"- Manifests: `{manifests_dir}`",
        f"- Tiles totales etiquetados: `{len(ann_tiles)}`",
        f"- Tiles M+/M- procesados morfologicamente: `{len(rows)}`",
        "",
        "## Parametros clave",
        f"- tile_size: `{params.tile_size}`",
        f"- canny: `({params.canny_low}, {params.canny_high})`",
        f"- frangi_pctl: `{params.frangi_pctl}`",
        f"- vesicle_circularity_min: `{params.vesicle_circularity_min}`",
        f"- arbuscule_pctl: `{params.arbuscule_pctl}`",
        f"- seam_sigma: `{params.seam_sigma}`",
        "",
        "## Cobertura por clase (pixeles)",
        f"- bg(0): `{counts[0]}`",
        f"- root(1): `{counts[1]}`",
        f"- colony(2): `{counts[2]}`",
        f"- hyphae(3): `{counts[3]}`",
        f"- vesicle(4): `{counts[4]}`",
        f"- arbuscule(5): `{counts[5]}`",
        "",
        "## Conteo Stage1",
    ]
    for k, v in stage1_counts.items():
        lines.append(f"- {k}: `{int(v)}`")
    lines.extend(
        [
            "",
            "## Conteo Stage2",
        ]
    )
    if stage2_counts:
        for k, v in stage2_counts.items():
            lines.append(f"- {k}: `{int(v)}`")
    else:
        lines.append("- Sin etiquetas Stage2 detectables en esta anotacion.")
    lines.extend(
        [
            "",
        "## Conteo de tiles por acronimo origen",
        ]
    )
    for k, v in raw_counts.items():
        lines.append(f"- {k}: `{int(v)}`")
    lines.extend(
        [
            "",
            "## Salidas",
            f"- `maps/{image_path.stem}__seg_morph.png` (IDs 0..5 para entrenamiento)",
            f"- `maps/{image_path.stem}__seg_morph_gray.png` (visual escalado 0..255)",
            f"- `maps/{image_path.stem}__seg_morph_color.png` (visual por clase)",
            f"- `maps/{image_path.stem}__seg_morph_color_con_leyenda.png` (leyenda embebida)",
            f"- `maps/{image_path.stem}__overlay_alpha.png`",
            f"- `maps/{image_path.stem}__stage1_tiles_rgba.png` + `maps/{image_path.stem}__stage2_tiles_rgba.png` (transparentes)",
            f"- `maps/{image_path.stem}__stage1_tiles_con_leyenda.png` + `maps/{image_path.stem}__stage2_tiles_con_leyenda.png`",
            f"- `maps/{image_path.stem}__tile_labels_overlay.png` (acronimo por tile)",
            f"- `maps/{image_path.stem}__comparison_panel.png` (comparativo 2x2)",
            f"- `maps/{image_path.stem}__plane_root.png` / `__plane_colony.png` / `__plane_hyphae.png` / `__plane_vesicle.png` / `__plane_arbuscule.png`",
            f"- `maps/{image_path.stem}__plane_root_tejido_radicular_con_leyenda.png` (y equivalentes por clase)",
            f"- `maps/{image_path.stem}__plane_*_rgba.png` (capas transparentes)",
            f"- `tables/{image_path.stem}__weakseg_masks.npz`",
            f"- `tables/{image_path.stem}__tile_metrics.parquet|csv`",
        ]
    )
    if patch_index is not None:
        lines.append("- `dino_patches/images/*.png` + `dino_patches/masks/*.png`")
        lines.append(f"- `{patch_index.relative_to(run.root).as_posix()}`")
    (run.reports / "run_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return run

