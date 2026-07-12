"""Mapas de evidencia morfológica MEViT — priors continuos Frangi / circularidad / entropía."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch
from skimage.filters import frangi

from micorizae.morph_core import WeakSegParams, segment_tile
from micorizae.morph_core.stain import stain_maps as _stain_maps
from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX
from .pixel_morph import PixelMorphParams, segment_tile_pixel_morph

# Versión de implementación de priors — cambiar fuerza re-materialización en HDF5
# sin re-etiquetar labels.
#   v2 = priors stain-aware (density) alineados a weakseg.
#   v3 = evidencia continua = MISMOS gates que los detectores binarios del weakseg:
#        IH = Frangi × top-hat (mata speckle en azul uniforme, brecha G1);
#        A  = textura fina × compuerta suave de densidad (brecha G2).
#   v4 = V LoG continuo multi-σ (ATLAS) + A arb-score Gallaud + H_stain acotado.
#   v5 = V prior acotado a discos esféricos (sin LoG global sobre contorno azul).
#   v6 = V solo contornos cerrados reales (sin discos sintéticos LoG).
#   v7 = híbrido: contorno + semilla LoG refinada + residual local.
#   v8 = SRP: arb_score cache + prior V ≡ detector; sin ruta grayscale.
PRIOR_IMPL_VERSION = 8  # SRP: arb_score cache + prior V ≡ detector mask

def _imagenet_mean_std():
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return mean, std
_TILE_PRIOR_POOL: Optional[ThreadPoolExecutor] = None
_TILE_PRIOR_POOL_WORKERS = 0


def _tile_prior_pool(workers: int) -> ThreadPoolExecutor:
    """Pool global de tiles — evita ThreadPoolExecutor anidado (deadlock en Windows)."""
    global _TILE_PRIOR_POOL, _TILE_PRIOR_POOL_WORKERS
    if _TILE_PRIOR_POOL is None or _TILE_PRIOR_POOL_WORKERS != workers:
        if _TILE_PRIOR_POOL is not None:
            _TILE_PRIOR_POOL.shutdown(wait=False, cancel_futures=True)
        _TILE_PRIOR_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prior_tile")
        _TILE_PRIOR_POOL_WORKERS = workers
    return _TILE_PRIOR_POOL


@dataclass
class PriorMaps:
    """Evidencia matemática por píxel (§7 MEViT methods)."""

    frangi: np.ndarray
    circularity: np.ndarray
    entropy: np.ndarray
    root: np.ndarray
    evidence: np.ndarray
    y_weak: np.ndarray
    masks: dict[str, np.ndarray]


def _normalize_in_mask(field: np.ndarray, mask: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = np.zeros_like(field, dtype=np.float32)
    if not mask.any():
        return out
    vals = field[mask]
    vmax = float(np.max(vals)) + eps
    out[mask] = (vals / vmax).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Priors stain-aware (operan sobre density = OD rojo), alineados a detectors.*
# ---------------------------------------------------------------------------


def _stain_frangi_continuous(
    density: np.ndarray,
    root: np.ndarray,
    p: WeakSegParams,
    *,
    frangi_vessel: Optional[np.ndarray] = None,
    ih_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tubularidad (IH) = Frangi × top-hat; reusa ``ih_score`` del detector si existe."""
    if ih_score is not None:
        return np.asarray(ih_score, dtype=np.float32)
    d = density.astype(np.float32)
    if frangi_vessel is not None:
        vessel = frangi_vessel.astype(np.float32)
    else:
        vessel = frangi(d, sigmas=list(p.ih_frangi_sigmas), black_ridges=False).astype(np.float32)
    d8 = (np.clip(d, 0.0, 1.0) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.ih_tophat_disk, p.ih_tophat_disk))
    tophat = cv2.morphologyEx(d8, cv2.MORPH_TOPHAT, k).astype(np.float32) / 255.0
    vessel_n = _normalize_in_mask(vessel, root > 0)
    tophat_n = _normalize_in_mask(tophat, root > 0)
    return _normalize_in_mask(vessel_n * tophat_n, root > 0)


def _stain_vesicle_continuous(
    density: np.ndarray, vesicle_mask: np.ndarray, root: np.ndarray
) -> np.ndarray:
    """Evidencia V = prior radial del módulo V (no re-detecta)."""
    _ = density
    from .detectors.v_vesicle import prior_from_mask

    return prior_from_mask(vesicle_mask, root)


def _stain_fine_texture_continuous(
    density: np.ndarray,
    root: np.ndarray,
    hyphae: np.ndarray,
    vesicle: np.ndarray,
    p: WeakSegParams,
    *,
    arb_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evidencia A = arb-score del detector (readonly). No re-ejecuta A si hay cache."""
    if arb_score is not None:
        return np.asarray(arb_score, dtype=np.float32)
    # Solo si segment_tile no cacheó arb_score (legado / tests)
    from .detectors import detect_arbuscules

    return detect_arbuscules(
        density, root, p, hyphae=hyphae, vesicle=vesicle, thr=0.35
    ).score.astype(np.float32)


def _build_evidence_stack_stain(
    frangi_e: np.ndarray,
    circ_e: np.ndarray,
    ent_e: np.ndarray,
    stain: np.ndarray,
    root: np.ndarray,
    hyphae: np.ndarray,
    vesicle: np.ndarray,
    arbuscule: np.ndarray,
) -> np.ndarray:
    """Stack de evidencia (5,H,W) con canal H explícito basado en tinción.

    Cierra el gap "no hay canal E_H stain-aware": la evidencia de colonización
    residual (H) es la densidad de tinción presente en tejido sin firma
    específica IH/V/A.
    """
    e = np.zeros((NUM_PIXEL_CLASSES, *frangi_e.shape), dtype=np.float32)
    root_b = root > 0
    struct = (hyphae > 0) | (vesicle > 0) | (arbuscule > 0)
    ih = PIXEL_CLASS_TO_IDX["IH"]
    v = PIXEL_CLASS_TO_IDX["V"]
    a = PIXEL_CLASS_TO_IDX["A"]
    h = PIXEL_CLASS_TO_IDX["H"]
    bg = PIXEL_CLASS_TO_IDX["BG"]

    stain_n = _normalize_in_mask(stain.astype(np.float32), root_b)
    e[ih] = frangi_e
    e[v] = circ_e
    e[a] = ent_e
    # H = tinción colonizada sin estructura específica (evidencia stain-aware)
    e[h] = np.where(struct, 0.0, stain_n).astype(np.float32) * root_b
    struct_max = np.maximum(np.maximum(np.maximum(frangi_e, circ_e), ent_e), e[h])
    e[bg] = np.clip(1.0 - struct_max, 0.0, 1.0)
    return e


def _stain_prior_targets(
    tile_rgb: np.ndarray,
    p: PixelMorphParams,
    *,
    masks: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evidencia (5,H,W) + máscara vesicular stain-aware, alineadas al weakseg."""
    if masks is None:
        masks = segment_tile(tile_rgb, p.weak)
    density = masks.get("density")
    if density is None:  # tolerancia si segment_tile no expuso density
        density = _stain_maps(tile_rgb)["density"]
    density = density.astype(np.float32)
    root = masks["root"].astype(bool)
    hyphae = masks["hyphae"].astype(bool)
    vesicle = masks["vesicle"].astype(bool)
    arbuscule = masks["arbuscule"].astype(bool)
    stain = masks.get("stain", density).astype(np.float32)

    frangi_e = _stain_frangi_continuous(
        density,
        root,
        p.weak,
        frangi_vessel=masks.get("frangi_ih"),
        ih_score=masks.get("ih_score"),
    )
    circ_e = _stain_vesicle_continuous(density, vesicle, root)
    ent_e = _stain_fine_texture_continuous(
        density, root, hyphae, vesicle, p.weak, arb_score=masks.get("arb_score")
    )
    evidence = _build_evidence_stack_stain(
        frangi_e, circ_e, ent_e, stain, root, hyphae, vesicle, arbuscule
    )
    return evidence, vesicle.astype(np.float32)


def compute_prior_training_targets(
    tile_rgb: np.ndarray,
    params: Optional[PixelMorphParams] = None,
    *,
    masks: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evidencia + máscara vesicular para prior loss (ruta stain-aware única)."""
    p = params or PixelMorphParams()
    if not getattr(p.weak, "stain_aware", True):
        import warnings

        warnings.warn(
            "stain_aware=False deprecated for priors; forcing stain-aware path.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _stain_prior_targets(tile_rgb, p, masks=masks)


def compute_prior_maps(tile_rgb: np.ndarray, params: Optional[PixelMorphParams] = None) -> PriorMaps:
    """Calcula priors continuos + pseudo-GT weak para un tile RGB uint8."""
    p = params or PixelMorphParams()
    masks = segment_tile(tile_rgb, p.weak)
    y_weak = segment_tile_pixel_morph(tile_rgb, p, masks=masks)
    density = masks.get("density")
    if density is None:
        density = _stain_maps(tile_rgb)["density"]
    density = density.astype(np.float32)
    root = masks["root"].astype(np.uint8)
    hyphae = masks["hyphae"].astype(bool)
    vesicle = masks["vesicle"].astype(bool)
    arbuscule = masks["arbuscule"].astype(bool)
    stain = masks.get("stain", density).astype(np.float32)
    frangi_c = _stain_frangi_continuous(
        density,
        root > 0,
        p.weak,
        frangi_vessel=masks.get("frangi_ih"),
        ih_score=masks.get("ih_score"),
    )
    circ = _stain_vesicle_continuous(density, vesicle, root > 0)
    ent = _stain_fine_texture_continuous(
        density,
        root > 0,
        hyphae,
        vesicle,
        p.weak,
        arb_score=masks.get("arb_score"),
    )
    evidence = _build_evidence_stack_stain(
        frangi_c, circ, ent, stain, root > 0, hyphae, vesicle, arbuscule
    )
    return PriorMaps(
        frangi=frangi_c,
        circularity=circ,
        entropy=ent,
        root=root,
        evidence=evidence,
        y_weak=y_weak,
        masks={
            "root": np.asarray(root, dtype=np.uint8),
            "hyphae": masks["hyphae"].astype(np.uint8),
            "vesicle": masks["vesicle"].astype(np.uint8),
            "arbuscule": masks["arbuscule"].astype(np.uint8),
        },
    )


def _denormalize_imagenet(rgb: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(rgb.device, dtype=rgb.dtype)
    std = _IMAGENET_STD.to(rgb.device, dtype=rgb.dtype)
    return (rgb * std + mean).clamp(0.0, 1.0)


def _prior_training_batch_cpu(
    rgb_cpu: np.ndarray,
    morph_params: Optional[PixelMorphParams],
    max_workers: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tiles = [rgb_cpu[i] for i in range(rgb_cpu.shape[0])]
    workers = max(1, min(max_workers, len(tiles), os.cpu_count() or 4))

    def _one(tile_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return compute_prior_training_targets(tile_rgb, morph_params)

    if workers == 1:
        out = [_one(t) for t in tiles]
    else:
        out = list(_tile_prior_pool(workers).map(_one, tiles))
    evidence = [ev for ev, _ in out]
    vesicle = [vm for _, vm in out]
    return evidence, vesicle


def prior_training_batch_from_normalized(
    rgb_normalized: "torch.Tensor",
    morph_params: Optional[PixelMorphParams] = None,
    *,
    device: Optional["torch.device"] = None,
    max_workers: int = 4,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """(B,3,H,W) ImageNet norm → evidencia (B,5,H,W) + vesícula (B,H,W) en device."""
    import torch

    dev = device or rgb_normalized.device
    rgb01 = _denormalize_imagenet(rgb_normalized)
    b, _, h, w = rgb01.shape
    rgb_cpu = (rgb01 * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    evidence_list, vesicle_list = _prior_training_batch_cpu(rgb_cpu, morph_params, max_workers)
    evidence = torch.stack([torch.from_numpy(ev) for ev in evidence_list], dim=0).to(dev, non_blocking=True)
    vesicle = torch.stack([torch.from_numpy(vm) for vm in vesicle_list], dim=0).to(dev, non_blocking=True)
    return evidence, vesicle


class PriorPrefetcher:
    """Precalcula priors del siguiente batch en CPU mientras la GPU entrena el actual."""

    def __init__(
        self,
        morph_params: Optional[PixelMorphParams],
        device: "torch.device",
        *,
        max_workers: int = 4,
    ) -> None:
        self._morph_params = morph_params
        self._device = device
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prior_prefetch")
        self._future = None

    def prime(self, model_input: torch.Tensor) -> None:
        self._future = self._pool.submit(
            prior_training_batch_from_normalized,
            model_input,
            self._morph_params,
            device=self._device,
            max_workers=self._max_workers,
        )

    def get(self) -> tuple["torch.Tensor", "torch.Tensor"]:
        if self._future is None:
            raise RuntimeError("PriorPrefetcher.get() sin prime() previo")
        return self._future.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


def prior_evidence_batch_from_normalized(
    rgb_normalized: "torch.Tensor",
    morph_params: Optional[PixelMorphParams] = None,
    *,
    device: Optional["torch.device"] = None,
    max_workers: int = 4,
) -> "torch.Tensor":
    """(B,3,H,W) ImageNet norm → (B,5,H,W) evidencia en device."""
    evidence, _ = prior_training_batch_from_normalized(
        rgb_normalized, morph_params, device=device, max_workers=max_workers
    )
    return evidence


def vesicle_mask_batch_from_normalized(
    rgb_normalized: "torch.Tensor",
    morph_params: Optional[PixelMorphParams] = None,
    *,
    max_workers: int = 4,
) -> "torch.Tensor":
    """Máscara binaria vesicular weak (B,H,W) float32."""
    _, vesicle = prior_training_batch_from_normalized(
        rgb_normalized, morph_params, device=rgb_normalized.device, max_workers=max_workers
    )
    return vesicle


def prior_argmax_agreement(y_pred: np.ndarray, y_weak: np.ndarray, root: Optional[np.ndarray] = None) -> float:
    """PPA tile-level entre predicción y weak."""
    if root is not None:
        mask = root > 0
    else:
        mask = np.ones_like(y_pred, dtype=bool)
    if not mask.any():
        return 1.0
    return float(np.mean(y_pred[mask] == y_weak[mask]))


def dominant_class_name(seg: np.ndarray) -> str:
    mask = seg > PIXEL_CLASS_TO_IDX["BG"]
    if not mask.any():
        return "BG"
    sub = seg[mask]
    counts = np.bincount(sub, minlength=NUM_PIXEL_CLASSES)
    idx = int(np.argmax(counts))
    return PIXEL_CLASS_NAMES[idx]
