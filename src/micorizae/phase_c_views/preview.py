"""Preview de un tile — pipeline 100% CUDA, transfiere a CPU SOLO para PIL save.

Genera el panel:
    Fila 1: [ raw | RGB+resize | seg-prep | log|F| | phase | recon(real,imag) ]
    Fila 2: [ recon 3-canal | recon SOLO |F| (random phase) | recon SOLO fase ]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..common.logging_utils import get_logger
from ..phase_b_tiling.gpu_io import crop_tile_gpu, decode_jpeg_gpu
from ..phase_c_views.gpu_transforms import (
    FreqRepGPU,
    build_views_gpu,
    freq_reconstruct_gpu,
    freq_view_gpu,
    rgb_view_gpu,
    seg_view_gpu,
)

log = get_logger("phase_c.preview_gpu")


def _to_cpu_uint8(t: torch.Tensor) -> np.ndarray:
    """Tensor cuda (1, C, H, W) o (C, H, W) -> ndarray (H, W, C) uint8."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach()
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255).round().to(torch.uint8) if t.dtype.is_floating_point else t.to(torch.uint8)
    if t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0).contiguous()
    return t.cpu().numpy()


def _viridis_gpu_to_rgb(arr2d_gpu: torch.Tensor) -> np.ndarray:
    """Aplica viridis (CPU via matplotlib) tras transferir a CPU. Para visualización."""
    import matplotlib.cm as cm

    arr = arr2d_gpu.detach().clamp(0, 1).cpu().numpy()
    rgb = (cm.get_cmap("viridis")(arr)[..., :3] * 255).astype(np.uint8)
    return rgb


def _hsv_phase_gpu_to_rgb(cos_t: torch.Tensor, sin_t: torch.Tensor) -> np.ndarray:
    import matplotlib.colors as mcolors

    cos_np = cos_t.detach().cpu().numpy()
    sin_np = sin_t.detach().cpu().numpy()
    phase = np.arctan2(sin_np, cos_np)
    hue = (phase + np.pi) / (2 * np.pi)
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def _hstack_with_labels(panels: list[np.ndarray], labels: list[str], pad: int = 6) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        ratio = h / p.shape[0]
        new_w = max(1, int(p.shape[1] * ratio))
        resized.append(np.array(Image.fromarray(p).resize((new_w, h), Image.BILINEAR)))
    total_w = sum(p.shape[1] for p in resized) + pad * (len(panels) - 1)
    canvas = np.full((h + 28, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for p in resized:
        canvas[28 : 28 + h, x : x + p.shape[1]] = p
        x += p.shape[1] + pad
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = None
    for name in ("arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 14)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    x = 0
    for p, label in zip(resized, labels):
        max_chars = max(8, p.shape[1] // 7)
        if len(label) > max_chars:
            label = label[: max_chars - 1] + "…"
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
        x += p.shape[1] + pad
    return np.array(pil)


def _vstack(rows: list[np.ndarray], pad: int = 8) -> np.ndarray:
    max_w = max(r.shape[1] for r in rows)
    h_total = sum(r.shape[0] for r in rows) + pad * (len(rows) - 1)
    out = np.full((h_total, max_w, 3), 255, dtype=np.uint8)
    y = 0
    for r in rows:
        out[y : y + r.shape[0], 0 : r.shape[1]] = r
        y += r.shape[0] + pad
    return out


def preview_tile(
    image_path: Path,
    row: int,
    col: int,
    tile_size: int = 252,
    target_size: int = 224,
    out_path: Path = Path("tile_preview.png"),
    device: str | torch.device = "cuda",
) -> Path:
    """Genera el panel de auditoría de Fase C — todo en CUDA salvo render PIL."""
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. preview_tile es GPU-only.")
    dev = torch.device(device)

    gimg = decode_jpeg_gpu(image_path, device=dev)
    try:
        tile = crop_tile_gpu(gimg, row=row, col=col, tile_size=tile_size)  # (3, T, T) uint8 cuda
    finally:
        del gimg
        torch.cuda.empty_cache()

    raw_np = _to_cpu_uint8(tile)

    rgb_t = rgb_view_gpu(tile, target_size=target_size, normalize_imagenet=False)
    seg_t = seg_view_gpu(tile, target_size=320, blur_radius=1, normalize_imagenet=False)
    freq = freq_view_gpu(tile, target_size=target_size)

    log_mag_rgb = _viridis_gpu_to_rgb(freq.log_mag_norm[0, 0])
    phase_rgb = _hsv_phase_gpu_to_rgb(freq.cos_phase[0, 0], freq.sin_phase[0, 0])

    # Reconstrucciones
    recon_full = freq_reconstruct_gpu(freq)               # (1, 1, H, W) float [0,1]
    eps = 1e-6
    log_mag = freq.log_mag_norm * (freq.log_mag_max - freq.log_mag_min).view(-1, 1, 1, 1) + freq.log_mag_min.view(-1, 1, 1, 1)
    mag = torch.expm1(log_mag) - eps
    phase_t = torch.atan2(freq.sin_phase, freq.cos_phase)
    F_3ch = mag * torch.exp(1j * phase_t)
    F_3ch = F_3ch.squeeze(1)
    f_uns = torch.fft.ifftshift(F_3ch, dim=(-2, -1))
    recon_3ch = torch.fft.ifft2(f_uns).real.unsqueeze(1) + freq.dc_mean.view(-1, 1, 1, 1)
    recon_3ch = recon_3ch.clamp(0, 1)

    # Sólo magnitud (random phase)
    torch.manual_seed(0)
    random_phase = (torch.rand_like(freq.real) - 0.5) * 2 * torch.pi
    F_mag_only = mag * torch.exp(1j * random_phase.squeeze(1))
    F_mag_only_u = torch.fft.ifftshift(F_mag_only, dim=(-2, -1))
    recon_mag_only = torch.fft.ifft2(F_mag_only_u).real.unsqueeze(1) + freq.dc_mean.view(-1, 1, 1, 1)
    recon_mag_only = recon_mag_only.clamp(0, 1)

    # Sólo fase (|F|=1)
    F_phase_only = torch.exp(1j * phase_t.squeeze(1))
    F_phase_u = torch.fft.ifftshift(F_phase_only, dim=(-2, -1))
    recon_phase_only_t = torch.fft.ifft2(F_phase_u).real.unsqueeze(1)
    # normalizar a [0,1]
    rmin = recon_phase_only_t.amin(dim=(-1, -2), keepdim=True)
    rmax = recon_phase_only_t.amax(dim=(-1, -2), keepdim=True)
    recon_phase_only = (recon_phase_only_t - rmin) / (rmax - rmin + 1e-6)

    # PSNR (devolvemos los floats CPU)
    gray = freq.gray
    mse_full = ((gray - recon_full) ** 2).mean().item()
    mse_3ch = ((gray - recon_3ch) ** 2).mean().item()
    psnr_full = 20 * np.log10(1.0 / max(np.sqrt(mse_full), 1e-6)) if mse_full > 0 else float("inf")
    psnr_3ch = 20 * np.log10(1.0 / max(np.sqrt(mse_3ch), 1e-6)) if mse_3ch > 0 else float("inf")
    log.info(f"[freq-gpu] PSNR full={psnr_full:.1f} dB, 3ch={psnr_3ch:.1f} dB")

    # Convertir a numpy uint8 para render PIL
    def to_gray_rgb(t):
        return np.stack([(t.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)

    rgb_np = _to_cpu_uint8(rgb_t)
    seg_np = _to_cpu_uint8(seg_t)
    recon_full_np = to_gray_rgb(recon_full)
    recon_3ch_np = to_gray_rgb(recon_3ch)
    recon_mag_np = to_gray_rgb(recon_mag_only)
    recon_phase_np = to_gray_rgb(recon_phase_only)

    row1 = _hstack_with_labels(
        [raw_np, rgb_np, seg_np, log_mag_rgb, phase_rgb, recon_full_np],
        [
            "raw tile",
            f"RGB @{target_size}",
            "seg-prep @320",
            "log|F| (viz)",
            "phase angle (HSV)",
            f"recon real+imag  PSNR {psnr_full:.0f} dB",
        ],
    )
    row1_w = row1.shape[1]
    pad = 8
    per_w = (row1_w - pad * 2) // 3
    target_h = row1.shape[0] - 28
    from PIL import Image as _PIL
    row2_panels = [
        np.array(_PIL.fromarray(p).resize((per_w, target_h), _PIL.BILINEAR))
        for p in [recon_3ch_np, recon_mag_np, recon_phase_np]
    ]
    row2 = _hstack_with_labels(
        row2_panels,
        [
            f"recon 3ch  PSNR {psnr_3ch:.0f} dB",
            "recon ONLY |F| (random phase)",
            "recon ONLY phase (|F|=1)",
        ],
        pad=pad,
    )
    panel = _vstack([row1, row2])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out_path)
    log.info(f"Panel guardado en {out_path}")
    return out_path
