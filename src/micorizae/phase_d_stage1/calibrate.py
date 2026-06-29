"""Temperature Scaling (Guo et al., ICML 2017).

Una sola escalar T > 0 que recalibra p(M+) sobre validación, manteniendo
ordenamiento pero ajustando confianzas. Crucial antes de fusionar 3 ramas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init)))))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> float:
        """Optimiza T minimizando BCE sobre (logits, targets) fijos.

        Devuelve la temperatura final.
        """
        logits = logits.detach().float()
        targets = targets.detach().float()
        opt = torch.optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter)

        def _closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(logits), targets)
            loss.backward()
            return loss

        opt.step(_closure)
        return self.temperature
