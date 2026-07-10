"""Mapas de evidencia morfológica MEViT — priors continuos Frangi / circularidad / entropía."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from ..phase_i_weakseg.pipeline import (
    WeakSegParams,
    _circularity,
    _stain_maps,
    _tile_root_mask,
    _tile_vesicle_mask,
    segment_tile,
)
from .pixel_class_map import NUM_PIXEL_CLASSES, PIXEL_CLASS_NAMES, PIXEL_CLASS_TO_IDX
from .pixel_morph import PixelMorphParams, segment_tile_pixel_morph

# Versión de implementación de priors — cambiar fuerza re-materialización en HDF5
# sin re-etiquetar labels.
#   v2 = priors stain-aware (density) alineados a weakseg.
#   v3 = evidencia continua = MISMOS gates que los detectores binarios del weakseg:
#        IH = Frangi × top-hat (mata speckle en azul uniforme, brecha G1);
#        A  = textura fina × compuerta suave de densidad (brecha G2).
PRIOR_IMPL_VERSION = 3

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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


def _frangi_continuous(gray: np.ndarray, root_mask: np.ndarray) -> np.ndarray:
    img = gray.astype(np.float32) / 255.0
    vessel = frangi(img, sigmas=range(1, 4), black_ridges=False)
    return _normalize_in_mask(vessel.astype(np.float32), root_mask > 0)


def _circularity_continuous(gray: np.ndarray, root_mask: np.ndarray, p: WeakSegParams) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(gray, dtype=np.float32)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < p.vesicle_min_area:
            continue
        circ = _circularity(c)
        if circ < p.vesicle_circularity_min * 0.5:
            continue
        score = float(np.clip(circ, 0.0, 1.0))
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, color=255, thickness=-1)
        m = (mask > 0) & root_mask
        out[m] = np.maximum(out[m], score)
    return out


def _entropy_continuous(gray: np.ndarray, root_mask: np.ndarray, p: WeakSegParams) -> np.ndarray:
    ent = entropy(img_as_ubyte(gray), disk(p.entropy_radius)).astype(np.float32)
    return _normalize_in_mask(ent, root_mask > 0)


# ---------------------------------------------------------------------------
# Priors stain-aware (operan sobre el canal density = OD del rojo), alineados
# con los detectores del weakseg stain-aware (Ruifrok & Johnston 2001; Frangi
# et al. 1998). Sustituyen los priors grayscale para cerrar el desacoplamiento
# teoría↔implementación descrito en la bitácora 2026-07-08.
# ---------------------------------------------------------------------------


def _stain_frangi_continuous(density: np.ndarray, root: np.ndarray, p: WeakSegParams) -> np.ndarray:
    """Tubularidad (IH) = Frangi × top-hat SOBRE density (v3, brecha G1).

    El detector binario `_tile_hyphae_stain` exige DOS condiciones: vesselness alta
    (Frangi) Y prominencia de cresta alta (top-hat blanco). El prior v2 usaba solo
    Frangi, que en tinción azul uniforme dispara evidencia IH espuria (speckle) donde
    el weakseg NO etiqueta hifa. v3 multiplica ambas señales normalizadas: el top-hat
    ~0 en azul plano anula la vesselness, alineando el prior con la pseudo-etiqueta.
    """
    d = density.astype(np.float32)
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
    """Evidencia V continua = densidad de tinción dentro de las vesículas detectadas.

    En vez de circularidad sobre Otsu grayscale, la evidencia se ancla a los
    blobs stain-aware (LoG + contraste disco-anillo, gate `ves_bg_max`). El prior
    V queda así consistente con la etiqueta V y con la clase que el ViT debe
    aprender.
    """
    m = (vesicle_mask > 0) & (root > 0)
    if not m.any():
        return np.zeros_like(density, dtype=np.float32)
    return _normalize_in_mask(density.astype(np.float32) * m, m)


def _stain_fine_texture_continuous(
    density: np.ndarray,
    root: np.ndarray,
    hyphae: np.ndarray,
    vesicle: np.ndarray,
    p: WeakSegParams,
) -> np.ndarray:
    """Evidencia A = textura fina × compuerta suave de densidad (v3, brecha G2).

    El detector binario `_tile_arbuscule_stain` exige textura fina alta Y densidad
    alta (`arb_density_pctl`). El prior v2 exportaba solo la textura fina, dando
    evidencia A en tejido poco teñido. v3 pondera la energía de alta frecuencia por
    una sigmoide centrada en el percentil de densidad, replicando el gate de densidad
    de forma diferenciable (sin umbral duro), restringida a tejido no tubular / no
    vesicular.
    """
    dens = density.astype(np.float32)
    low = gaussian_filter(dens, sigma=3.0)
    fine = np.abs(dens - low)
    fine = gaussian_filter(fine, sigma=1.0)
    valid = (root > 0) & ~(hyphae > 0) & ~(vesicle > 0)
    root_b = root > 0
    dens_thr = float(np.percentile(dens[root_b], p.arb_density_pctl)) if root_b.any() else 0.0
    # sigmoide suave alrededor del umbral de densidad (escala ~0.05 del rango OD 0..1)
    dgate = 1.0 / (1.0 + np.exp(-(dens - dens_thr) / 0.05))
    return _normalize_in_mask(fine * dgate.astype(np.float32), valid)


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


def _build_evidence_stack(
    frangi: np.ndarray,
    circ: np.ndarray,
    ent: np.ndarray,
    root: np.ndarray,
) -> np.ndarray:
    e = np.zeros((NUM_PIXEL_CLASSES, *frangi.shape), dtype=np.float32)
    root_b = root > 0
    ih = PIXEL_CLASS_TO_IDX["IH"]
    v = PIXEL_CLASS_TO_IDX["V"]
    a = PIXEL_CLASS_TO_IDX["A"]
    h = PIXEL_CLASS_TO_IDX["H"]
    bg = PIXEL_CLASS_TO_IDX["BG"]

    e[ih] = frangi
    e[v] = circ
    e[a] = ent
    struct_max = np.maximum(np.maximum(frangi, circ), ent)
    e[h] = root_b.astype(np.float32) * np.clip(1.0 - struct_max, 0.0, 1.0)
    e[bg] = np.clip(1.0 - np.max(e[ih:], axis=0), 0.0, 1.0)
    return e


def _stain_prior_targets(
    tile_rgb: np.ndarray, p: PixelMorphParams
) -> tuple[np.ndarray, np.ndarray]:
    """Evidencia (5,H,W) + máscara vesicular stain-aware, alineadas al weakseg."""
    masks = segment_tile(tile_rgb, p.weak)  # ya es stain-aware (dispatch en pipeline)
    density = masks.get("density")
    if density is None:  # tolerancia si segment_tile no expuso density
        density = _stain_maps(tile_rgb)["density"]
    density = density.astype(np.float32)
    root = masks["root"].astype(bool)
    hyphae = masks["hyphae"].astype(bool)
    vesicle = masks["vesicle"].astype(bool)
    arbuscule = masks["arbuscule"].astype(bool)
    stain = masks.get("stain", density).astype(np.float32)

    frangi_e = _stain_frangi_continuous(density, root, p.weak)
    circ_e = _stain_vesicle_continuous(density, vesicle, root)
    ent_e = _stain_fine_texture_continuous(density, root, hyphae, vesicle, p.weak)
    evidence = _build_evidence_stack_stain(
        frangi_e, circ_e, ent_e, stain, root, hyphae, vesicle, arbuscule
    )
    return evidence, vesicle.astype(np.float32)


def compute_prior_training_targets(
    tile_rgb: np.ndarray, params: Optional[PixelMorphParams] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Evidencia + máscara vesicular para prior loss (sin weak-seg completo)."""
    p = params or PixelMorphParams()
    if getattr(p.weak, "stain_aware", True):
        return _stain_prior_targets(tile_rgb, p)
    # --- Ruta legacy grayscale (solo si stain_aware=False) ---
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    root = _tile_root_mask(gray, p.weak).astype(np.uint8)
    frangi = _frangi_continuous(gray, root)
    circ = _circularity_continuous(gray, root > 0, p.weak)
    ent = _entropy_continuous(gray, root > 0, p.weak)
    evidence = _build_evidence_stack(frangi, circ, ent, root)
    ves = _tile_vesicle_mask(gray, root, p.weak).astype(np.float32)
    return evidence, ves


def compute_prior_maps(tile_rgb: np.ndarray, params: Optional[PixelMorphParams] = None) -> PriorMaps:
    """Calcula priors continuos + pseudo-GT weak para un tile RGB uint8."""
    p = params or PixelMorphParams()
    masks = segment_tile(tile_rgb, p.weak)
    y_weak = segment_tile_pixel_morph(tile_rgb, p)
    if getattr(p.weak, "stain_aware", True):
        density = masks.get("density")
        if density is None:
            density = _stain_maps(tile_rgb)["density"]
        density = density.astype(np.float32)
        root = masks["root"].astype(np.uint8)
        hyphae = masks["hyphae"].astype(bool)
        vesicle = masks["vesicle"].astype(bool)
        arbuscule = masks["arbuscule"].astype(bool)
        stain = masks.get("stain", density).astype(np.float32)
        frangi_c = _stain_frangi_continuous(density, root > 0, p.weak)
        circ = _stain_vesicle_continuous(density, vesicle, root > 0)
        ent = _stain_fine_texture_continuous(density, root > 0, hyphae, vesicle, p.weak)
        evidence = _build_evidence_stack_stain(
            frangi_c, circ, ent, stain, root > 0, hyphae, vesicle, arbuscule
        )
    else:
        gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        root = _tile_root_mask(gray, p.weak).astype(np.uint8)
        frangi_c = _frangi_continuous(gray, root)
        circ = _circularity_continuous(gray, root > 0, p.weak)
        ent = _entropy_continuous(gray, root > 0, p.weak)
        evidence = _build_evidence_stack(frangi_c, circ, ent, root)
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


@torch.no_grad()
def prior_training_batch_from_normalized(
    rgb_normalized: torch.Tensor,
    morph_params: Optional[PixelMorphParams] = None,
    *,
    device: Optional[torch.device] = None,
    max_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,3,H,W) ImageNet norm → evidencia (B,5,H,W) + vesícula (B,H,W) en device."""
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
        device: torch.device,
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

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._future is None:
            raise RuntimeError("PriorPrefetcher.get() sin prime() previo")
        return self._future.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


def prior_evidence_batch_from_normalized(
    rgb_normalized: torch.Tensor,
    morph_params: Optional[PixelMorphParams] = None,
    *,
    device: Optional[torch.device] = None,
    max_workers: int = 4,
) -> torch.Tensor:
    """(B,3,H,W) ImageNet norm → (B,5,H,W) evidencia en device."""
    evidence, _ = prior_training_batch_from_normalized(
        rgb_normalized, morph_params, device=device, max_workers=max_workers
    )
    return evidence


def vesicle_mask_batch_from_normalized(
    rgb_normalized: torch.Tensor,
    morph_params: Optional[PixelMorphParams] = None,
    *,
    max_workers: int = 4,
) -> torch.Tensor:
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
