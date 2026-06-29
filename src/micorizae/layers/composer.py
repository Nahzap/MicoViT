"""Composición de capas visuales auditables L0..L10.

Diseño:
    - Cada capa expone una función `render(ctx) -> ndarray (H,W,3) uint8`.
    - `compose([L0, L2, L3], alphas=[1.0, 0.5, 0.4])` apila por alpha-blending.
    - `LayerContext` lleva la imagen base, dimensiones de tile, manifests parciales,
      predicciones por etapa, etc. Las capas no leen del disco directamente.

La implementación inicial sólo cubre L0 (base) y L1 (grid). Las demás capas
están registradas en `_REGISTRY` pero levantan `NotImplementedError` con
mensaje informativo hasta que las fases correspondientes existan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


LayerName = str
RenderFn = Callable[["LayerContext"], np.ndarray]


@dataclass
class LayerContext:
    image: np.ndarray  # (H, W, 3) uint8
    tile_size: int = 252
    tile_offset: tuple[int, int] = (0, 0)  # (row0, col0) en pixeles
    tiles: Optional[object] = None  # pandas.DataFrame con (row,col,...)
    stage1_mask: Optional[np.ndarray] = None
    stage1_proba: Optional[np.ndarray] = None
    seg_mask: Optional[np.ndarray] = None
    freq_map: Optional[np.ndarray] = None
    consensus: Optional[np.ndarray] = None
    stage2_classes: Optional[np.ndarray] = None
    stage2_entropy: Optional[np.ndarray] = None
    morphology: Optional[dict] = None
    diff_map: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


# ---------------------- Implementaciones iniciales ---------------------- #

def render_L0(ctx: LayerContext) -> np.ndarray:
    """Capa L0 — Imagen base (RGB)."""
    if ctx.image is None:
        raise ValueError("LayerContext.image es obligatorio para L0")
    return ctx.image.copy()


def render_L1(ctx: LayerContext) -> np.ndarray:
    """Capa L1 — Grid/tile overlay (row, col).

    Dibuja una grilla de cada `tile_size` píxeles sobre la imagen base.
    El grosor se elige proporcional al tile para sobrevivir a downscales.
    Útil para auditar geometría antes de Fase B.
    """
    base = render_L0(ctx)
    h, w = base.shape[:2]
    ts = ctx.tile_size
    color = np.array([255, 220, 0], dtype=np.uint8)  # amarillo
    thickness: int = int(ctx.extra.get("grid_thickness", max(2, ts // 64)))

    out = base.copy()
    for x in range(0, w + 1, ts):
        x0 = min(x, w - thickness)
        out[:, x0 : x0 + thickness] = color
    for y in range(0, h + 1, ts):
        y0 = min(y, h - thickness)
        out[y0 : y0 + thickness, :] = color
    return out


# Colores canónicos del master plan (sección 5.3).
STAGE1_COLORS: dict[str, tuple[int, int, int]] = {
    "Mplus": (0, 200, 255),        # cian
    "Mminus": (210, 180, 140),     # beige
    "Background": (110, 110, 110), # gris
    "Unreadable": (160, 80, 200),  # morado tenue
    "Unknown": (160, 80, 200),     # alias gate4
}


def render_L2(ctx: LayerContext) -> np.ndarray:
    """Capa L2 — Máscara Stage1 (M+/M-/Bg/Unr) coloreada por tile.

    Soporta dos fuentes:
      A) `ctx.stage1_mask` (ndarray HxW de strings o de int categórico).
      B) `ctx.tiles` (DataFrame con columnas `row, col, stage1`) — fuente sintética
         desde anotaciones. Útil como ground truth visual antes de entrenar.

    La capa renderiza cada celda al color canónico definido en STAGE1_COLORS.
    Si la capa se compone con L0, usar `compose(['L0','L2'], alphas=[1.0, 0.5])`.
    """
    h, w = ctx.image.shape[:2]
    ts = ctx.tile_size
    out = np.zeros((h, w, 3), dtype=np.uint8)

    if ctx.tiles is not None:
        df = ctx.tiles
        # Pintar por (row,col) — vectorizar sería más rápido pero el tamaño es modesto
        for row_, col_, stage1 in zip(df["row"].values, df["col"].values, df["stage1"].values):
            color = STAGE1_COLORS.get(str(stage1), (0, 0, 0))
            y0 = int(row_) * ts
            x0 = int(col_) * ts
            y1 = min(y0 + ts, h)
            x1 = min(x0 + ts, w)
            if y0 >= h or x0 >= w:
                continue
            out[y0:y1, x0:x1] = color
        return out

    raise ValueError(
        "L2 necesita `ctx.tiles` (DataFrame con row,col,stage1) o `ctx.stage1_mask`."
    )


def _not_yet(name: str, phase: str) -> RenderFn:
    def _fn(ctx: LayerContext) -> np.ndarray:
        raise NotImplementedError(
            f"Capa {name}: pendiente de implementación en la {phase}. "
            f"Consulta MASTER_PLAN_CAPAS_VISUALES_AMF.md sección 5."
        )

    return _fn


def _viridis_color(value: float) -> tuple[int, int, int]:
    """Colormap viridis aproximado en uint8 sin matplotlib (para evitar import en hot path)."""
    try:
        import matplotlib.cm as cm

        r, g, b, _ = cm.get_cmap("viridis")(float(np.clip(value, 0, 1)))
        return (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        v = int(np.clip(value, 0, 1) * 255)
        return (v, v, v)


def render_L3(ctx: LayerContext) -> np.ndarray:
    """Capa L3 — Heatmap continuo p(M+) por tile.

    Requiere `ctx.tiles` con columna `p_fused` (output de infer.py).
    Pinta cada tile con `viridis(p_fused)` y dibuja contornos en 0.5, 0.7, 0.9
    como bordes blanco/amarillo/rojo respectivamente.
    """
    if ctx.tiles is None or "p_fused" not in ctx.tiles.columns:
        raise ValueError("L3 requiere ctx.tiles con columna 'p_fused' (corre infer_image primero)")

    h, w = ctx.image.shape[:2]
    ts = ctx.tile_size
    out = np.zeros((h, w, 3), dtype=np.uint8)
    df = ctx.tiles

    for r_, c_, p in zip(df["row"].values, df["col"].values, df["p_fused"].values):
        y0 = int(r_) * ts
        x0 = int(c_) * ts
        y1 = min(y0 + ts, h); x1 = min(x0 + ts, w)
        if y0 >= h or x0 >= w:
            continue
        color = _viridis_color(float(p))
        out[y0:y1, x0:x1] = color

    # Contornos por umbral (borde de 2 px alrededor de los tiles que pasan el umbral)
    contour_specs = [
        (0.5, (255, 255, 255)),
        (0.7, (255, 220, 0)),
        (0.9, (255, 60, 60)),
    ]
    th = max(2, ts // 64)
    for tau, color in contour_specs:
        for r_, c_, p in zip(df["row"].values, df["col"].values, df["p_fused"].values):
            if p < tau:
                continue
            y0 = int(r_) * ts
            x0 = int(c_) * ts
            y1 = min(y0 + ts, h); x1 = min(x0 + ts, w)
            if y0 >= h or x0 >= w:
                continue
            # solo bordes
            out[y0:y0 + th, x0:x1] = color
            out[max(y1 - th, y0):y1, x0:x1] = color
            out[y0:y1, x0:x0 + th] = color
            out[y0:y1, max(x1 - th, x0):x1] = color
    return out


def render_L4(ctx: LayerContext) -> np.ndarray:
    """Capa L4 — Segmentación U2Net (probabilidad/máscara) sobre la imagen.

    Usa `ctx.seg_mask` (HxW float en [0,1]). Si no existe, devuelve capa vacía.
    """
    h, w = ctx.image.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if ctx.seg_mask is None:
        return out

    seg = np.asarray(ctx.seg_mask)
    if seg.ndim == 3:
        seg = seg[..., 0]
    if seg.shape != (h, w):
        # Evita romper composición si el mapa no coincide por downscale.
        return out

    mask = np.clip(seg.astype(np.float32), 0.0, 1.0)
    # Cian para estructuras detectadas, intensidad proporcional a probabilidad.
    out[..., 0] = (40.0 * mask).astype(np.uint8)
    out[..., 1] = (220.0 * mask).astype(np.uint8)
    out[..., 2] = (255.0 * mask).astype(np.uint8)
    return out


def render_L6(ctx: LayerContext) -> np.ndarray:
    """Capa L6 — Consenso inter-rama.

    Verde si `consensus >= 0.85`, amarillo si en [0.6, 0.85), rojo si < 0.6.
    Requiere `ctx.tiles` con columna `consensus` ∈ [0, 1].
    """
    if ctx.tiles is None or "consensus" not in ctx.tiles.columns:
        raise ValueError("L6 requiere ctx.tiles con columna 'consensus'")

    h, w = ctx.image.shape[:2]
    ts = ctx.tile_size
    out = np.zeros((h, w, 3), dtype=np.uint8)
    df = ctx.tiles

    for r_, c_, cv in zip(df["row"].values, df["col"].values, df["consensus"].values):
        if cv >= 0.85:
            color = (60, 200, 60)
        elif cv >= 0.6:
            color = (255, 200, 0)
        else:
            color = (220, 50, 50)
        y0 = int(r_) * ts
        x0 = int(c_) * ts
        y1 = min(y0 + ts, h); x1 = min(x0 + ts, w)
        if y0 >= h or x0 >= w:
            continue
        out[y0:y1, x0:x1] = color
    return out


STAGE2_COLORS: dict[str, tuple[int, int, int]] = {
    "AMColonised": (0, 180, 255),
    "Hybrid": (255, 120, 0),
    "BlueCoils": (80, 120, 255),
    "BrownCoils": (160, 90, 40),
    "TypeTwo": (180, 60, 220),
    "HybridErm": (255, 200, 60),
    "HybridDse": (120, 220, 120),
}


def render_L7(ctx: LayerContext) -> np.ndarray:
    """Capa L7 — Subclases Stage2 (solo tiles con predicción/anotación Stage2)."""
    col = "stage2_pred" if ctx.tiles is not None and "stage2_pred" in ctx.tiles.columns else "stage2"
    if ctx.tiles is None or col not in ctx.tiles.columns:
        raise ValueError("L7 requiere ctx.tiles con columna 'stage2_pred' o 'stage2'")

    h, w = ctx.image.shape[:2]
    ts = ctx.tile_size
    out = np.zeros((h, w, 3), dtype=np.uint8)
    df = ctx.tiles.dropna(subset=[col])

    for r_, c_, cls in zip(df["row"].values, df["col"].values, df[col].values):
        color = STAGE2_COLORS.get(str(cls), (200, 200, 200))
        y0 = int(r_) * ts
        x0 = int(c_) * ts
        y1 = min(y0 + ts, h)
        x1 = min(x0 + ts, w)
        if y0 >= h or x0 >= w:
            continue
        out[y0:y1, x0:x1] = color
    return out


def render_L8(ctx: LayerContext) -> np.ndarray:
    """Capa L8 — Incertidumbre Stage2 (entropía normalizada por tile)."""
    h, w = ctx.image.shape[:2]
    ts = ctx.tile_size
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if ctx.tiles is None or "stage2_entropy" not in ctx.tiles.columns:
        return out

    df = ctx.tiles.dropna(subset=["stage2_entropy"])

    for r_, c_, ent in zip(df["row"].values, df["col"].values, df["stage2_entropy"].values):
        y0 = int(r_) * ts
        x0 = int(c_) * ts
        y1 = min(y0 + ts, h)
        x1 = min(x0 + ts, w)
        if y0 >= h or x0 >= w:
            continue
        color = _viridis_color(float(ent))
        out[y0:y1, x0:x1] = color
    return out


_REGISTRY: dict[LayerName, RenderFn] = {
    "L0": render_L0,
    "L1": render_L1,
    "L2": render_L2,
    "L3": render_L3,
    "L4": render_L4,
    "L5": _not_yet("L5 (mapa frecuencia)", "Fase D"),
    "L6": render_L6,
    "L7": render_L7,
    "L8": render_L8,
    "L9": _not_yet("L9 (morfología derivada)", "Fase H"),
    "L10": _not_yet("L10 (diff entre rondas)", "Fase G"),
}


def render_layer(name: LayerName, ctx: LayerContext) -> np.ndarray:
    name = name.upper()
    if name not in _REGISTRY:
        raise KeyError(f"Capa desconocida: {name}. Disponibles: {sorted(_REGISTRY)}")
    return _REGISTRY[name](ctx)


def compose(
    layers: Iterable[LayerName],
    ctx: LayerContext,
    alphas: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Apila capas por alpha-blending en orden dado.

    Implementación memory-aware: NUNCA expande a float32 si la imagen excede
    ~200 MP. En su lugar usa blending entero por chunks via numpy.lerp.
    """
    names = list(layers)
    if alphas is None:
        alphas = [1.0] * len(names)
    alphas = list(alphas)
    if len(alphas) != len(names):
        raise ValueError("len(alphas) debe coincidir con len(layers)")

    if not names:
        return render_L0(ctx)

    out = render_layer(names[0], ctx).copy()  # uint8
    for name, alpha in zip(names[1:], alphas[1:]):
        layer = render_layer(name, ctx)  # uint8
        if alpha <= 0.0:
            continue
        if alpha >= 1.0:
            out[:] = layer
            continue
        # blend uint8 in-place sin float32 global:
        # out = (1-a)*out + a*layer ≈ out + a*(layer-out)
        a16 = int(round(alpha * 256))
        diff = layer.astype(np.int16) - out.astype(np.int16)
        out = (out.astype(np.int16) + ((diff * a16) >> 8)).clip(0, 255).astype(np.uint8)
    return out


def downscale_context(ctx: LayerContext, factor: int) -> LayerContext:
    """Devuelve un nuevo LayerContext con imagen y tile_size escalados por `factor`."""
    if factor is None or factor <= 1:
        return ctx
    from PIL import Image

    h, w = ctx.image.shape[:2]
    new_w, new_h = max(1, w // factor), max(1, h // factor)
    img_small = np.array(Image.fromarray(ctx.image).resize((new_w, new_h), Image.BILINEAR))
    new_ts = max(1, ctx.tile_size // factor)
    def _resize_opt_2d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            return arr
        im = Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8))
        return np.array(im.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0

    return LayerContext(
        image=img_small,
        tile_size=new_ts,
        tile_offset=ctx.tile_offset,
        tiles=ctx.tiles,
        stage1_mask=None if ctx.stage1_mask is None else ctx.stage1_mask,
        stage1_proba=None if ctx.stage1_proba is None else ctx.stage1_proba,
        seg_mask=_resize_opt_2d(ctx.seg_mask),
        freq_map=_resize_opt_2d(ctx.freq_map),
        consensus=None if ctx.consensus is None else ctx.consensus,
        stage2_classes=None if ctx.stage2_classes is None else ctx.stage2_classes,
        stage2_entropy=None if ctx.stage2_entropy is None else ctx.stage2_entropy,
        morphology=ctx.morphology,
        diff_map=None if ctx.diff_map is None else ctx.diff_map,
        extra=ctx.extra,
    )


def save_png(arr: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path
