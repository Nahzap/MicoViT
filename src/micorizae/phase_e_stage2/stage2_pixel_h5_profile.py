"""Perfil de conformación HDF5 Stage2-Pixel — validación post-build."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX
from .pixel_data import load_mplus_splits
from .stage2_pixel_h5_cache import Stage2PixelH5Store, _cache_paths
from .stage2_pixel_run_layout import atomic_write_text

log = get_logger("phase_e.pixel_h5_profile")


def _scan_label_stats(labels: np.ndarray) -> dict[str, Any]:
    flat = labels.reshape(-1)
    counts = np.bincount(flat, minlength=NUM_PIXEL_CLASSES)
    total = int(counts.sum())
    pixel_pct = {
        c: float(100.0 * counts[i] / total) if total else 0.0
        for i, c in enumerate(PIXEL_CLASS_NAMES)
    }
    n_tiles = labels.shape[0]
    tile_presence = {
        c: float(100.0 * (labels == i).any(axis=(1, 2)).mean())
        for i, c in enumerate(PIXEL_CLASS_NAMES)
    }
    v_idx = PIXEL_CLASS_TO_IDX["V"]
    v_px_per_tile = (labels == v_idx).sum(axis=(1, 2))
    v_pos = v_px_per_tile >= 30
    return {
        "n_tiles": n_tiles,
        "n_pixels": total,
        "pixel_pct": pixel_pct,
        "tile_presence_pct": tile_presence,
        "v_pos_tiles_ge30px": int(v_pos.sum()),
        "v_pos_tile_pct": float(100.0 * v_pos.mean()),
        "mean_v_px_per_tile": float(v_px_per_tile.mean()),
    }


def build_h5_conformation_profile(
    *,
    h5_path: Optional[Path] = None,
    lookup_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    val_fraction: float = 0.2,
    v_min_px: int = 30,
) -> dict[str, Any]:
    """Escanea HDF5 y escribe perfil JSON + MD de conformación."""
    paths = get_paths()
    h5_path, meta_path_d, lookup_path_d = _cache_paths(paths.root)
    h5_path = Path(h5_path) if h5_path is None else Path(h5_path)
    lookup_path = lookup_path or lookup_path_d
    meta_path = meta_path or meta_path_d
    out_dir = out_dir or (paths.root / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 no encontrado: {h5_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    train_df, val_df, test_df, split_info = load_mplus_splits(val_fraction=val_fraction)
    lookup = pd.read_parquet(lookup_path)

    def idx_for(df: pd.DataFrame) -> np.ndarray:
        m = lookup.merge(df[["image_path", "row", "col"]], on=["image_path", "row", "col"])
        return m["h5_idx"].values.astype(int)

    with h5py.File(h5_path, "r") as hf:
        label_ds = hf["label"]
        rgb_shape = tuple(hf["rgb"].shape)
        label_shape = tuple(label_ds.shape)

        n_total = label_shape[0]
        chunk = 512
        parts = []
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            parts.append(label_ds[s:e])
        global_labels = np.concatenate(parts, axis=0)

        split_stats = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            idx = idx_for(df)
            split_stats[name] = _scan_label_stats(label_ds[idx])

    global_stats = _scan_label_stats(global_labels)

    checks = {
        "schema_5_classes": meta.get("classes") == list(PIXEL_CLASS_NAMES),
        "cache_version_v2": int(meta.get("cache_version", 0)) >= 2,
        "no_root_pixels": global_stats["pixel_pct"].get("BG", 0) >= 0 and "ROOT" not in global_stats["pixel_pct"],
        "v_present": global_stats["pixel_pct"].get("V", 0) > 0.01,
        "ih_present": global_stats["pixel_pct"].get("IH", 0) > 1.0,
        "h_present": global_stats["pixel_pct"].get("H", 0) > 5.0,
        "v_pos_train_tiles": split_stats["train"]["v_pos_tiles_ge30px"] > 0,
    }
    ok = all(checks.values())

    profile: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "h5_path": str(h5_path),
        "meta_path": str(meta_path),
        "rgb_shape": list(rgb_shape),
        "label_shape": list(label_shape),
        "meta": meta,
        "split_info": split_info,
        "global": global_stats,
        "splits": split_stats,
        "checks": checks,
        "conformation_ok": ok,
        "v_min_px_threshold": v_min_px,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"stage2_pixel_h5_profile_{ts}.json"
    md_path = out_dir / f"stage2_pixel_h5_profile_{ts}.md"
    latest_json = out_dir / "stage2_pixel_h5_profile_latest.json"
    latest_md = out_dir / "stage2_pixel_h5_profile_latest.md"

    payload = json.dumps(profile, indent=2, default=str)
    atomic_write_text(json_path, payload)
    atomic_write_text(latest_json, payload)
    atomic_write_text(md_path, _profile_md(profile))
    atomic_write_text(latest_md, _profile_md(profile))

    msg = f"Perfil HDF5 -> {md_path} | conformación={'OK' if ok else 'REVISAR'}"
    log.info(msg)
    print(f"[Stage2-Pixel H5] {msg}", flush=True)
    return profile


def _profile_md(profile: dict[str, Any]) -> str:
    g = profile["global"]
    checks = profile["checks"]
    lines = [
        f"# Perfil conformación HDF5 Stage2-Pixel",
        "",
        f"**Generado:** {profile['generated_at']}",
        f"**Estado:** {'OK' if profile['conformation_ok'] else 'REVISAR'}",
        "",
        f"- HDF5: `{profile['h5_path']}`",
        f"- Shape rgb: {profile['rgb_shape']}",
        f"- Shape label: {profile['label_shape']}",
        "",
        "## Distribución global de píxeles",
        "",
        "| Clase | % píxeles | % tiles con clase |",
        "|-------|----------:|------------------:|",
    ]
    for c in PIXEL_CLASS_NAMES:
        lines.append(
            f"| {c} | {g['pixel_pct'][c]:.4f} | {g['tile_presence_pct'][c]:.1f} |"
        )
    lines += [
        "",
        f"- Tiles V≥30px: **{g['v_pos_tiles_ge30px']}** ({g['v_pos_tile_pct']:.1f}%)",
        f"- Media px V/tile: {g['mean_v_px_per_tile']:.1f}",
        "",
        "## Por split",
        "",
    ]
    for split, st in profile["splits"].items():
        lines.append(f"### {split} ({st['n_tiles']} tiles)")
        for c in PIXEL_CLASS_NAMES:
            lines.append(f"- {c}: {st['pixel_pct'][c]:.3f}% px | presente en {st['tile_presence_pct'][c]:.1f}% tiles")
        lines.append(f"- V≥30px tiles: {st['v_pos_tiles_ge30px']}")
        lines.append("")

    lines += ["## Checks", ""]
    for k, v in checks.items():
        lines.append(f"- [{'x' if v else ' '}] {k}")
    return "\n".join(lines)
