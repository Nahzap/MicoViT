"""U2-Net architecture (Qin et al. 2020, arXiv:2005.09007).

Implementación RSU/U2NET adaptada para saliencia en tiles AM.
"""

from .model_def import U2NET, U2NETP, RSU4, RSU4F, RSU5, RSU6, RSU7

__all__ = ["U2NET", "U2NETP", "RSU4", "RSU4F", "RSU5", "RSU6", "RSU7"]
