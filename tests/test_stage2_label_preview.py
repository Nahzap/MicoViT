"""Unit tests para selección de tiles del preview Stage2."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_select_preview_tiles_random_count() -> None:
    from micorizae.phase_e_stage2.stage2_pixel_label_preview import select_preview_tiles

    rows = [
        {
            "image_path": f"data/IMG_{i // 5}/img.jpg",
            "row": i,
            "col": 0,
            "tile_size": 252,
        }
        for i in range(40)
    ]
    df = pd.DataFrame(rows)
    out = select_preview_tiles(df, n_tiles=25, seed=7)
    assert len(out) == 25
    # misma semilla → misma muestra
    out2 = select_preview_tiles(df, n_tiles=25, seed=7)
    assert list(out["row"]) == list(out2["row"])
    # otra semilla → distinta (con alta probabilidad)
    out3 = select_preview_tiles(df, n_tiles=25, seed=99)
    assert list(out["row"]) != list(out3["row"])
