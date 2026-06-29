"""Visualizaciones post-entrenamiento: tiles gold vs predicción (best checkpoint)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from .gate_classes import GATE_CLASS_NAMES

log = get_logger("phase_d.gate_posttrain_viz")


def plot_tile_audit_best_ckpt(
    val_df: pd.DataFrame,
    *,
    model: torch.nn.Module,
    embed_store: object,
    prototype_bank: Optional[object],
    use_attn: bool,
    out_png: Path,
    n_per_class: int = 8,
    seed: int = 0,
    device: torch.device,
) -> Optional[Path]:
    """Grid: gold vs pred + confianza por tile holdout."""
    from PIL import Image

    from .gate_pretrain_viz import ensure_matplotlib_agg
    from .gate_probe_input import probe_features_from_batch
    from .gpu_pipeline import GPUImageBatch

    ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    required = {"image_path", "row", "col", "stage1"}
    if val_df.empty or not required.issubset(val_df.columns):
        return None

    rng = np.random.default_rng(seed)
    stage1_map = {"Unknown": "Unreadable"}
    order = [c for c in GATE_CLASS_NAMES if c != "Unknown"]
    paths_root = get_paths().root

    fig, axes = plt.subplots(len(order), n_per_class, figsize=(n_per_class * 1.5, len(order) * 1.6))
    if len(order) == 1:
        axes = np.array([axes])
    if n_per_class == 1:
        axes = axes.reshape(len(order), 1)

    model.eval()
    with torch.no_grad():
        for row_i, cls in enumerate(order):
            key = stage1_map.get(cls, cls)
            pool = val_df[val_df["stage1"].astype(str) == key]
            if pool.empty:
                continue
            n = min(n_per_class, len(pool))
            idx = rng.choice(pool.index.to_numpy(), size=n, replace=len(pool) < n)
            sample = pool.loc[idx]

            for col_i in range(n_per_class):
                ax = axes[row_i, col_i]
                ax.axis("off")
                if col_i == 0:
                    ax.set_ylabel(cls, fontsize=9)
                if col_i >= len(sample):
                    continue
                rec = sample.iloc[col_i]
                ts = int(rec.get("tile_size", 252))
                x0, y0 = int(rec["col"]) * ts, int(rec["row"]) * ts
                img_path = paths_root / str(rec["image_path"])

                try:
                    sub = pd.DataFrame([rec])
                    eidx = embed_store.indices_for_sub(sub)
                    feat, lab = embed_store.read_batch(eidx, device)
                    attn = embed_store.read_attn_batch(eidx, device)
                    domain = None
                    if "domain_bucket" in rec.index and pd.notna(rec.get("domain_bucket")):
                        domain = str(rec["domain_bucket"])
                    batch = GPUImageBatch(
                        rgb=feat,
                        seg=feat,
                        freq=feat,
                        labels=lab,
                        label_mode="gate",
                        rows=torch.tensor([rec["row"]], device=device),
                        cols=torch.tensor([rec["col"]], device=device),
                        image_path=str(rec["image_path"]),
                        features=feat,
                        vit_attention=attn,
                        domain_buckets=[domain] if domain else None,
                    )
                    feats = probe_features_from_batch(batch)
                    embed = model.encode(feats)
                    if prototype_bank is not None and prototype_bank.is_ready():
                        from ..gate_domain_buckets import domain_buckets_from_batch

                        logits = prototype_bank.logits(
                            embed, domain_buckets=domain_buckets_from_batch(batch)
                        )
                    else:
                        logits = model.gate_head(embed)
                    pred_i = int(logits.argmax(dim=-1).item())
                    conf = float(torch.softmax(logits.float(), dim=-1).max().item())
                    from .gate_classes import decode_gate_indices

                    pred_name = decode_gate_indices(np.array([pred_i]))[0]
                    ok = pred_name == cls
                    color = "#2ca02c" if ok else "#d62728"

                    if img_path.exists():
                        with Image.open(img_path) as im:
                            tile = im.convert("RGB").crop((x0, y0, x0 + ts, y0 + ts))
                            ax.imshow(np.asarray(tile))
                    ax.set_title(
                        f"gold={cls}\npred={pred_name}\nconf={conf:.2f}",
                        fontsize=5,
                        color=color,
                        pad=2,
                    )
                except Exception as e:
                    ax.text(0.5, 0.5, "err", ha="center", fontsize=6)
                    log.debug(f"[Gate post viz] {e}")

    fig.suptitle("Post-train audit — holdout (gold vs pred, best ckpt)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
