"""Inferencia morfológica píxel Fase 2 — DINOv2 ViT + weak bootstrap + MEViT explain."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.gpu_io import decode_jpeg_gpu
from ..phase_c_views.gpu_transforms import rgb_view_gpu
from .pixel_attention import extract_pixel_vit_attention
from .pixel_class_map import PIXEL_CLASS_TO_IDX
from .pixel_explainability import TileExplainBundle
from .pixel_morph import (
    PixelMorphParams,
    load_tile_rgb_from_image,
    quantize_tiles_table,
    segment_tile_pixel_morph,
    stitch_pixel_map_from_tiles,
)
from .pixel_prior_maps import PriorMaps, compute_prior_maps
from .pixel_vit_model import PixelMorphViT

log = get_logger("phase_e.infer_pixel")

Backend = Literal["weak", "vit", "ensemble"]


@dataclass
class PixelMorphInferResult:
    seg_map: np.ndarray
    tile_table: pd.DataFrame
    tile_segments: dict[tuple[int, int], np.ndarray]
    tile_probs: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    tile_weak: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    tile_priors: dict[tuple[int, int], PriorMaps] = field(default_factory=dict)
    tile_attention: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


def _is_mplus_row(row: pd.Series) -> bool:
    if "is_mplus" in row and int(row["is_mplus"]) == 1:
        return True
    if "stage1_pred" in row and pd.notna(row["stage1_pred"]):
        sp = row["stage1_pred"]
        if isinstance(sp, str):
            return sp == "Mplus"
        return int(sp) == 1
    if "stage1" in row:
        return str(row["stage1"]).strip() == "Mplus"
    return False


def _merge_manifest_coords(mplus_df: pd.DataFrame, image_rel: str) -> pd.DataFrame:
    paths = get_paths()
    manifest = pd.read_parquet(paths.manifests / "tiles_index.parquet")
    key_cols = ["row", "col"]
    if "tile_size" in mplus_df.columns and mplus_df["tile_size"].notna().any():
        key_cols.append("tile_size")
    sub = manifest[manifest["image_path"] == image_rel][
        key_cols + ["x0", "y0", "x1", "y1", "stage2"]
    ].drop_duplicates(subset=key_cols)
    merged = mplus_df.merge(sub, on=key_cols, how="left", suffixes=("", "_m"))
    for c in ("x0", "y0", "x1", "y1"):
        if f"{c}_m" in merged.columns:
            merged[c] = merged[c].fillna(merged[f"{c}_m"])
            merged.drop(columns=[f"{c}_m"], inplace=True)
    if "stage2_m" in merged.columns:
        if "stage2" in merged.columns:
            merged["stage2"] = merged["stage2"].fillna(merged["stage2_m"])
        else:
            merged["stage2"] = merged["stage2_m"]
        merged.drop(columns=["stage2_m"], errors="ignore")
    merged["image_path"] = image_rel
    return merged.drop_duplicates(subset=key_cols, keep="first")


@torch.no_grad()
def _predict_tiles_vit_batch(
    model: PixelMorphViT,
    tiles_rgb: list[np.ndarray],
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 8,
    *,
    return_probs: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    segs: list[np.ndarray] = []
    probs_out: list[np.ndarray] | None = [] if return_probs else None
    for i in range(0, len(tiles_rgb), batch_size):
        chunk = tiles_rgb[i : i + batch_size]
        sizes = [(t.shape[0], t.shape[1]) for t in chunk]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)
        tensors = []
        for t in chunk:
            th, tw = t.shape[:2]
            if th == max_h and tw == max_w:
                tensors.append(torch.from_numpy(t).permute(2, 0, 1).contiguous())
            else:
                padded = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
                padded[:th, :tw] = t
                tensors.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
        batch_u8 = torch.stack(tensors, dim=0)
        rgb_in = rgb_view_gpu(batch_u8, target_size=input_size, normalize_imagenet=True)
        rgb_in = rgb_in.to(device)
        logits = model(rgb_in)
        if return_probs:
            prob = F.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        for j, (h, w) in enumerate(sizes):
            seg = pred[j]
            if seg.shape != (h, w):
                seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            segs.append(seg.astype(np.uint8))
            if return_probs and probs_out is not None:
                pr = prob[j]
                if pr.shape[-2:] != (h, w):
                    pr = np.stack(
                        [
                            cv2.resize(pr[c].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                            for c in range(pr.shape[0])
                        ],
                        axis=0,
                    )
                probs_out.append(pr.astype(np.float32))
    return segs, probs_out


def infer_image_pixel_morph(
    image_path: Path,
    mplus_tiles_df: pd.DataFrame,
    *,
    backend: Backend = "vit",
    model: Optional[PixelMorphViT] = None,
    device: Optional[torch.device] = None,
    morph_params: Optional[PixelMorphParams] = None,
    input_size: int = 224,
    with_explain: bool = False,
    attention_layers: str = "last",
) -> tuple[np.ndarray, pd.DataFrame, dict[tuple[int, int], np.ndarray]] | PixelMorphInferResult:
    """Devuelve (seg_map, tabla tiles, segmentos) o PixelMorphInferResult si with_explain=True."""
    result = _infer_image_pixel_morph_core(
        image_path,
        mplus_tiles_df,
        backend=backend,
        model=model,
        device=device,
        morph_params=morph_params,
        input_size=input_size,
        with_explain=with_explain,
        attention_layers=attention_layers,
    )
    if with_explain:
        return result
    return result.seg_map, result.tile_table, result.tile_segments


def _infer_image_pixel_morph_core(
    image_path: Path,
    mplus_tiles_df: pd.DataFrame,
    *,
    backend: Backend = "vit",
    model: Optional[PixelMorphViT] = None,
    device: Optional[torch.device] = None,
    morph_params: Optional[PixelMorphParams] = None,
    input_size: int = 224,
    with_explain: bool = False,
    attention_layers: str = "last",
) -> PixelMorphInferResult:
    paths = get_paths()
    rel = Path(image_path).resolve().relative_to(paths.root).as_posix()
    sub = mplus_tiles_df[mplus_tiles_df["image_path"] == rel].copy() if "image_path" in mplus_tiles_df.columns else mplus_tiles_df.copy()
    sub = sub[sub.apply(_is_mplus_row, axis=1)].copy()
    if sub.empty:
        gimg = decode_jpeg_gpu(image_path, device=device or torch.device("cuda"))
        h, w = gimg.height, gimg.width
        del gimg
        return PixelMorphInferResult(
            seg_map=np.zeros((h, w), dtype=np.uint8),
            tile_table=pd.DataFrame(),
            tile_segments={},
        )

    sub = _merge_manifest_coords(sub, rel)
    sub = sub.dropna(subset=["x0", "y0", "x1", "y1"]).copy()
    if sub.empty:
        raise ValueError(f"Sin coordenadas manifest para M+ en {rel}")

    dev = device or torch.device("cuda")
    morph_params = morph_params or PixelMorphParams()
    gimg = decode_jpeg_gpu(image_path, device=dev)
    img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
    del gimg

    tile_segments: dict[tuple[int, int], np.ndarray] = {}
    tile_probs: dict[tuple[int, int], np.ndarray] = {}
    tile_weak: dict[tuple[int, int], np.ndarray] = {}
    tile_priors: dict[tuple[int, int], PriorMaps] = {}
    tile_attention: dict[tuple[int, int], np.ndarray] = {}

    use_vit = backend in ("vit", "ensemble") and model is not None
    if use_vit:
        model.eval().to(dev)

    records = list(sub.itertuples(index=False))
    tiles_rgb: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    for rec in records:
        x0, y0, x1, y1 = int(rec.x0), int(rec.y0), int(rec.x1), int(rec.y1)
        tile = load_tile_rgb_from_image(img_np, x0, y0, x1, y1)
        if tile.size == 0:
            continue
        keys.append((int(rec.row), int(rec.col)))
        tiles_rgb.append(tile)

    seg_weak_list = [segment_tile_pixel_morph(t, morph_params) for t in tiles_rgb]
    if with_explain:
        prior_list = [compute_prior_maps(t, morph_params) for t in tiles_rgb]
    else:
        prior_list = [None] * len(tiles_rgb)

    seg_vit_list: list[np.ndarray] | None = None
    probs_list: list[np.ndarray] | None = None
    if use_vit:
        seg_vit_list, probs_list = _predict_tiles_vit_batch(
            model,
            tiles_rgb,
            dev,
            input_size=input_size,
            return_probs=with_explain,
        )

    for idx, key in enumerate(keys):
        seg_w = seg_weak_list[idx]
        tile_weak[key] = seg_w
        if with_explain and prior_list[idx] is not None:
            tile_priors[key] = prior_list[idx]

        if use_vit and seg_vit_list is not None:
            seg_v = seg_vit_list[idx]
            if backend == "ensemble":
                colony_w = np.isin(seg_w, [PIXEL_CLASS_TO_IDX[c] for c in ("IH", "V", "A", "H")])
                seg = seg_v.copy()
                seg[~colony_w] = seg_w[~colony_w]
            else:
                seg = seg_v
            if with_explain and probs_list is not None:
                tile_probs[key] = probs_list[idx]
                if model is not None:
                    t_u8 = torch.from_numpy(tiles_rgb[idx]).permute(2, 0, 1).unsqueeze(0)
                    rgb_in = rgb_view_gpu(t_u8, target_size=input_size, normalize_imagenet=True).to(dev)
                    tile_attention[key] = extract_pixel_vit_attention(
                        model,
                        rgb_in,
                        layers_spec=attention_layers,
                        input_size=input_size,
                    )
        else:
            seg = seg_w
            if with_explain:
                one_hot = np.eye(5, dtype=np.float32)[seg]
                tile_probs[key] = np.transpose(one_hot, (2, 0, 1))

        tile_segments[key] = seg

    h, w = img_np.shape[:2]
    full_seg = stitch_pixel_map_from_tiles((h, w), sub, tile_segments)
    from .instance_merge import merge_vesicle_instances

    full_seg = merge_vesicle_instances(full_seg, min_area=30, circularity_min=0.65)
    tile_table = quantize_tiles_table(sub, tile_segments)
    return PixelMorphInferResult(
        seg_map=full_seg,
        tile_table=tile_table,
        tile_segments=tile_segments,
        tile_probs=tile_probs,
        tile_weak=tile_weak,
        tile_priors=tile_priors,
        tile_attention=tile_attention,
    )


def build_tile_explain_bundles(result: PixelMorphInferResult) -> dict[tuple[int, int], TileExplainBundle]:
    bundles: dict[tuple[int, int], TileExplainBundle] = {}
    for key, seg in result.tile_segments.items():
        weak = result.tile_weak.get(key)
        prior = result.tile_priors.get(key)
        probs = result.tile_probs.get(key)
        if weak is None or prior is None or probs is None:
            continue
        bundles[key] = TileExplainBundle(
            seg_vit=seg,
            seg_weak=weak,
            probs=probs,
            prior=prior,
            attention=result.tile_attention.get(key),
        )
    return bundles
