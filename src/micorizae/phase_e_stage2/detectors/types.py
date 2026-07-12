"""Contrato común de detectores por clase (SRP Stage2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class DetectResult:
    """Salida de un detector de clase: máscara + score opcional + meta."""

    mask: np.ndarray
    score: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool))
        if self.score is not None:
            object.__setattr__(self, "score", np.asarray(self.score, dtype=np.float32))
