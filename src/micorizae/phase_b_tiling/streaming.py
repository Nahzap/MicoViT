"""DEPRECATED / QUARANTINE — legacy ImageWindowDataset streaming.

Production tiling uses ``jpeg_streaming``. Do not use for new work.

Streaming de tiles imagen-por-imagen.

Filosofía (recordatorio del usuario):
    "pequeñas ventanas de observación, recorriendo todo el lienzo".

Concretamente:
    - El loader abre UNA imagen.
    - Itera sobre TODOS sus tiles relevantes (en raster o aleatorio interno).
    - Suelta la imagen.
    - Pasa a la siguiente.

Esto evita mantener panorámicas multi-GB en RAM y elimina las reaperturas
repetidas que dominaban el wall-clock cuando el dataset era random-access.

Soporta `WorkerInfo` para repartir las imágenes entre workers de DataLoader.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..common.logging_utils import get_logger
from ..common.paths import get_paths
from .tile_cutter import crop_tile_from_array, open_image_rgb

log = get_logger("phase_b.streaming")

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalize_imagenet(arr: np.ndarray) -> torch.Tensor:
    """uint8 HWC numpy -> float32 CHW ImageNet-normalized tensor."""
    x = torch.from_numpy(np.asarray(arr, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


def plan_image_order(
    tiles_df: pd.DataFrame,
    *,
    max_neg_per_image: Optional[int] = None,
    interleave_by_lineage: bool = True,
    shuffle_images: bool = True,
    shuffle_tiles_within_image: bool = True,
    seed: int = 0,
    pos_class: str = "Mplus",
) -> pd.DataFrame:
    """Devuelve `tiles_df` reordenado en bloques por imagen, listo para streaming.

    - Filtra cada imagen para conservar todos sus M+ + `max_neg_per_image` negativos.
    - Mezcla las imágenes (opcional) e intercala linajes para que cada época vea
      tanto AM como ERM, no en bloques separados.
    - Dentro de una imagen mezcla tiles para evitar sesgo posicional, sin perder
      la propiedad clave: todos los tiles de la imagen son CONTIGUOS en la salida.
    """
    rng = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []
    for image_path, grp in tiles_df.groupby("image_path", sort=False):
        if max_neg_per_image is not None:
            pos = grp[grp["stage1"] == pos_class]
            neg = grp[grp["stage1"] != pos_class]
            if len(neg) > max_neg_per_image:
                idx = rng.choice(neg.index.to_numpy(), size=max_neg_per_image, replace=False)
                neg = neg.loc[idx]
            grp = pd.concat([pos, neg])
        if shuffle_tiles_within_image:
            grp = grp.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        chunks.append(grp)

    if shuffle_images and chunks:
        rng.shuffle(chunks)

    if interleave_by_lineage and chunks:
        # Buckets por linaje y round-robin.
        buckets: dict[str, list[pd.DataFrame]] = {}
        for c in chunks:
            lin = str(c["lineage"].iloc[0])
            buckets.setdefault(lin, []).append(c)
        interleaved: list[pd.DataFrame] = []
        while any(buckets.values()):
            for lin in list(buckets.keys()):
                if buckets[lin]:
                    interleaved.append(buckets[lin].pop(0))
        chunks = interleaved

    if not chunks:
        return tiles_df.iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


@dataclass
class StreamingTile:
    rgb: torch.Tensor    # (3, 224, 224) float32 normalizada ImageNet
    seg: torch.Tensor    # (3, 320, 320) float32 normalizada ImageNet
    freq: torch.Tensor   # (3, 224, 224) float32 (log|F|, cosφ, sinφ)
    label: torch.Tensor  # () float32 in {0,1}
    meta: dict           # image_path, row, col, stage1, lineage


def iter_tiles_for_image(
    image_path: Path,
    tiles_subdf: pd.DataFrame,
    *,
    target_size: int = 224,
    seg_target_size: int = 320,
    pos_class: str = "Mplus",
) -> Iterator[StreamingTile]:
    """Yields un `StreamingTile` por fila de `tiles_subdf` desde UNA imagen.

    La imagen se decodifica una sola vez al entrar al generador y se libera al
    salir mediante `gc.collect()` explícito en el `finally`.
    """
    from ..phase_c_views.transforms import build_views

    image_arr = open_image_rgb(image_path)
    try:
        for _, row in tiles_subdf.iterrows():
            tile_size = int(row["tile_size"])
            tile = crop_tile_from_array(
                image_arr, row=int(row["row"]), col=int(row["col"]), tile_size=tile_size
            )
            bundle = build_views(tile, target_size=target_size, seg_target_size=seg_target_size)
            rgb = _normalize_imagenet(bundle.rgb)
            seg = _normalize_imagenet(bundle.seg_input)
            freq = torch.from_numpy(bundle.freq.features_3ch).float()
            label = torch.tensor(1.0 if row["stage1"] == pos_class else 0.0, dtype=torch.float32)
            meta = {
                "image_path": row["image_path"],
                "row": int(row["row"]),
                "col": int(row["col"]),
                "tile_size": tile_size,
                "stage1": row["stage1"],
                "lineage": row["lineage"],
            }
            yield StreamingTile(rgb=rgb, seg=seg, freq=freq, label=label, meta=meta)
    finally:
        del image_arr
        gc.collect()


class ImageWindowDataset(IterableDataset):
    """IterableDataset: streaming por imagen.

    Cada época abre cada imagen UNA vez. Los tiles se emiten en raster local.
    Soporta multi-worker repartiendo imágenes (no tiles).
    """

    def __init__(
        self,
        tiles_df: pd.DataFrame,
        *,
        target_size: int = 224,
        seg_target_size: int = 320,
        epoch_seed: int = 0,
    ):
        super().__init__()
        required = {"image_path", "row", "col", "tile_size", "stage1", "lineage"}
        missing = required - set(tiles_df.columns)
        if missing:
            raise ValueError(f"tiles_df incompleto: faltan {missing}")
        self.df = tiles_df.reset_index(drop=True)
        self.target_size = target_size
        self.seg_target_size = seg_target_size
        self.epoch_seed = epoch_seed
        self.paths = get_paths()

    def __len__(self) -> int:
        return len(self.df)

    def _image_paths_for_this_worker(self) -> list[str]:
        all_paths = self.df["image_path"].drop_duplicates().tolist()
        info = get_worker_info()
        if info is None:
            return all_paths
        return all_paths[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        my_paths = self._image_paths_for_this_worker()
        order = {p: i for i, p in enumerate(self.df["image_path"].unique())}
        my_paths.sort(key=lambda p: order[p])

        for img_path in my_paths:
            sub = self.df[self.df["image_path"] == img_path]
            full_path = self.paths.root / img_path
            try:
                for st in iter_tiles_for_image(
                    full_path,
                    sub,
                    target_size=self.target_size,
                    seg_target_size=self.seg_target_size,
                ):
                    yield st.rgb, st.seg, st.freq, st.label
            except Exception as e:
                log.warning(f"[stream] saltando {img_path}: {e}")
                continue
