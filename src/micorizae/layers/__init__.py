"""Capas visuales L0..L10 del master plan.

Cada capa es una función pura `render(ctx) -> np.ndarray (H,W,3) uint8` y
puede componerse mediante `compose([...], alphas=[...])`.
"""

from .composer import LayerContext, compose, render_layer, save_png, downscale_context, STAGE1_COLORS

__all__ = [
    "LayerContext",
    "compose",
    "render_layer",
    "save_png",
    "downscale_context",
    "STAGE1_COLORS",
]
