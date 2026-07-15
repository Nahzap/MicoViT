"""Preview de pseudo-labels Stage2 (~N tiles) antes de conformar el HDF5 completo.

Misma lógica que el pack H5: ``segment_tile_pixel_morph`` + V ATLAS multi-tile.
Decode CPU. Muestreo por defecto: **N tiles aleatorias** del pool M+.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from ..phase_b_tiling.jpeg_streaming import crop_tile_u8_from_file
from .detectors.v_vesicle import (
    apply_v_masks_to_label_and_priors,
    detect_vesicle_masks_for_tiles,
)
from .pixel_class_map import PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX
from .pixel_morph import (
    PixelMorphParams,
    render_diagnostic_overlay,
    render_pixel_class_map,
    segment_tile_pixel_morph,
)
from .stage2_pixel_h5_cache import collect_mplus_tiles_for_h5

log = get_logger("stage2.preview")


def select_preview_tiles(
    df: pd.DataFrame,
    n_tiles: int = 25,
    seed: int = 42,
    *,
    priority_substrings: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Selecciona ``n_tiles`` al azar (uniforme) del pool M+.

    ``priority_substrings`` queda por compatibilidad de API; por defecto vacío
    (sin sesgo ACH/ABS). Si se pasa una lista no vacía, se priorizan esos hits
    y el resto se completa al azar.
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame vacío: no hay tiles M+ para preview.")
    work = df.reset_index(drop=True).copy()
    n_tiles = max(1, min(int(n_tiles), len(work)))
    rng = np.random.default_rng(int(seed))

    if not priority_substrings:
        idx = rng.choice(work.index.to_numpy(), size=n_tiles, replace=False)
        return work.loc[idx].reset_index(drop=True)

    chosen_idx: list[int] = []
    used: set[int] = set()
    paths = work["image_path"].astype(str)
    for sub in priority_substrings:
        hits = work.index[paths.str.contains(sub, regex=False)].tolist()
        rng.shuffle(hits)
        for i in hits:
            ii = int(i)
            if ii in used:
                continue
            chosen_idx.append(ii)
            used.add(ii)
            if len(chosen_idx) >= n_tiles:
                break
        if len(chosen_idx) >= n_tiles:
            break
    if len(chosen_idx) < n_tiles:
        remain = [int(i) for i in work.index.tolist() if int(i) not in used]
        rng.shuffle(remain)
        for i in remain:
            chosen_idx.append(i)
            if len(chosen_idx) >= n_tiles:
                break
    return work.loc[chosen_idx].reset_index(drop=True).head(n_tiles)


def _resize_label(seg: np.ndarray, target_size: int) -> np.ndarray:
    if seg.shape[0] == target_size and seg.shape[1] == target_size:
        return seg.astype(np.uint8)
    return cv2.resize(seg, (target_size, target_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _panel_rgb_label_overlay(rgb: np.ndarray, label: np.ndarray) -> Image.Image:
    lab_rgb = render_pixel_class_map(label)
    ov = render_diagnostic_overlay(rgb, label, alpha=0.48)
    h, w = rgb.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = rgb
    canvas[:, w : 2 * w] = lab_rgb
    canvas[:, 2 * w : 3 * w] = ov
    return Image.fromarray(canvas)


def run_stage2_pixel_label_preview(
    *,
    n_tiles: int = 25,
    seed: int = 42,
    out_dir: Optional[Path] = None,
    morph: Optional[PixelMorphParams] = None,
    val_fraction: float = 0.2,
    gate_run_id: str = "",
    input_size: int = 224,
) -> dict[str, Any]:
    """Genera preview de pseudo-GT (misma lógica que H5 pack + V ATLAS multi-tile)."""
    from micorizae.morph_core import WeakSegParams

    from .pixel_data import load_mplus_splits
    import config as user_config  # type: ignore

    if morph is None:
        morph = PixelMorphParams(
            weak=WeakSegParams(
                vesicle_circularity_min=float(
                    getattr(user_config, "STAGE2_PIXEL_VESICLE_CIRCULARITY_MIN", 0.85)
                ),
                frangi_pctl=float(getattr(user_config, "STAGE2_PIXEL_FRANGI_PCTL", 82.0)),
                arbuscule_pctl=float(getattr(user_config, "STAGE2_PIXEL_ARBUSCULE_PCTL", 93.0)),
                ves_max_radius=int(getattr(user_config, "STAGE2_PIXEL_VESICLE_MAX_RADIUS", 0)),
                ves_max_sigma=float(getattr(user_config, "STAGE2_PIXEL_VESICLE_MAX_SIGMA", 40.0)),
            ),
            seam_sigma=0.0,
        )

    pseudo_gt = str(getattr(user_config, "STAGE2_PIXEL_PSEUDO_GT_VERSION", "v4b_giant_spherical_arc"))
    train_df, val_df, test_df, split_info = load_mplus_splits(val_fraction=val_fraction)
    combined = collect_mplus_tiles_for_h5(train_df, val_df, test_df)
    sample = select_preview_tiles(combined, n_tiles=n_tiles, seed=seed)

    ts_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        out_dir = Path("outputs") / f"stage2_label_preview_{ts_stamp}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    paths = get_paths()
    root = paths.root
    v_idx = int(PIXEL_CLASS_TO_IDX["V"])
    img_cache: dict[str, np.ndarray] = {}

    by_img: dict[str, list[int]] = defaultdict(list)
    for i, row in sample.iterrows():
        by_img[str(row["image_path"])].append(int(i))

    records: list[dict[str, Any]] = []
    giant_px_total = 0

    for img_rel, idxs in by_img.items():
        img_path = root / img_rel
        if not img_path.exists():
            log.warning(f"imagen ausente: {img_path}")
            continue

        sample_rows = [sample.loc[i] for i in idxs]
        default_ts = int(sample_rows[0]["tile_size"]) if "tile_size" in sample.columns else 252
        need_keys: set[tuple[int, int]] = set()
        for r in sample_rows:
            tr, tc = int(r["row"]), int(r["col"])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    need_keys.add((tr + dr, tc + dc))

        img_all = combined[combined["image_path"].astype(str) == img_rel]
        neighbor_rows: list[Any] = []
        for _, nrow in img_all.iterrows():
            key = (int(nrow["row"]), int(nrow["col"]))
            if key in need_keys:
                neighbor_rows.append(nrow)

        tiles_hwc: list[np.ndarray] = []
        rows_l: list[int] = []
        cols_l: list[int] = []
        labels_native: list[np.ndarray] = []
        key_to_i: dict[tuple[int, int], int] = {}

        for nrow in neighbor_rows:
            r, c = int(nrow["row"]), int(nrow["col"])
            ts = int(nrow["tile_size"]) if "tile_size" in nrow.index else default_ts
            try:
                rgb = crop_tile_u8_from_file(
                    img_path, r, c, ts, image_arr_cache=img_cache
                )
            except Exception as exc:
                log.warning(f"skip crop {img_rel} r{r}c{c}: {exc}")
                continue
            lab = segment_tile_pixel_morph(rgb, morph)
            key_to_i[(r, c)] = len(tiles_hwc)
            tiles_hwc.append(rgb)
            rows_l.append(r)
            cols_l.append(c)
            labels_native.append(lab)

        if not tiles_hwc:
            continue

        log.info(
            f"[preview] {Path(img_rel).name}: ATLAS multi-tile sobre "
            f"{len(tiles_hwc)} tiles vecindario ({len(idxs)} sample)..."
        )
        v_masks = detect_vesicle_masks_for_tiles(
            tiles_hwc, rows_l, cols_l, default_ts
        )
        n_v = int(sum(int(m.sum()) for m in v_masks))
        giant_px_total += n_v
        log.info(f"[preview] {Path(img_rel).name}: atlas_v={n_v} px")

        for i in idxs:
            row = sample.loc[i]
            key = (int(row["row"]), int(row["col"]))
            if key not in key_to_i:
                continue
            ti = key_to_i[key]
            rgb = tiles_hwc[ti]
            lab = labels_native[ti]
            vmask = v_masks[ti]
            lab_r = _resize_label(lab, input_size)
            lab2, _, _ = apply_v_masks_to_label_and_priors(
                lab_r, None, None, vmask, input_size=input_size, v_idx=v_idx
            )
            rgb_r = rgb
            if rgb.shape[0] != input_size or rgb.shape[1] != input_size:
                rgb_r = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)

            stem = Path(img_rel).stem
            panel_name = f"{stem}_r{key[0]}c{key[1]}.png"
            panel_path = panels_dir / panel_name
            _panel_rgb_label_overlay(rgb_r, lab2).save(panel_path)

            hist = {
                PIXEL_CLASS_NAMES[c]: int(np.count_nonzero(lab2 == c))
                for c in range(len(PIXEL_CLASS_NAMES))
            }
            records.append(
                {
                    "image_path": img_rel,
                    "row": key[0],
                    "col": key[1],
                    "panel": str(panel_path.relative_to(out_dir)).replace("\\", "/"),
                    "class_hist": hist,
                    "atlas_v_px": int(np.count_nonzero(vmask)),
                }
            )

        # liberar cache de esta imagen (panorámicas grandes)
        img_cache.clear()

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir.resolve()),
        "pseudo_gt_version": pseudo_gt,
        "n_requested": int(n_tiles),
        "n_rendered": len(records),
        "seed": int(seed),
        "input_size": int(input_size),
        "gate_run_id": str(
            gate_run_id or getattr(user_config, "STAGE2_PIXEL_GATE_RUN_ID", "")
        ),
        "split_info": {
            k: split_info.get(k)
            for k in (
                "n_train_tiles",
                "n_val_tiles",
                "n_test_tiles",
                "n_train_images",
                "n_val_images",
                "n_test_images",
            )
            if isinstance(split_info, dict)
        },
        "atlas_v_px_total": int(giant_px_total),
        "legend": "panel = RGB | pseudo-label | overlay (V=naranja ATLAS, H=verde, A=magenta, IH=cian)",
        "tiles": records,
    }
    (out_dir / "preview_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        f"# Stage2 label preview ({meta['n_rendered']} tiles)",
        "",
        f"- pseudo_gt: `{meta['pseudo_gt_version']}`",
        f"- atlas_v_px_total: {giant_px_total}",
        "",
        "| # | tile | V | A | H | IH | atlas_V | panel |",
        "|---|------|---|---|---|----|---------|-------|",
    ]
    for i, rec in enumerate(records):
        h = rec["class_hist"]
        name = f"{Path(rec['image_path']).stem}_r{rec['row']}c{rec['col']}"
        lines.append(
            f"| {i + 1} | `{name}` | {h.get('V', 0)} | {h.get('A', 0)} | "
            f"{h.get('H', 0)} | {h.get('IH', 0)} | {rec.get('atlas_v_px', rec.get('giant_v_px', 0))} | {rec['panel']} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log.info(f"[preview] {len(records)} paneles -> {out_dir}")
    print(
        f"[Stage2-Pixel] label-preview LISTO -> {out_dir} | "
        f"n={len(records)} | giant_px={giant_px_total} | ver={pseudo_gt}",
        flush=True,
    )
    return meta
