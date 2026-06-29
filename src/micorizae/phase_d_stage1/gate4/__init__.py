"""Gate 4 clases (M+/M-/Background/Unknown) + Slice Multi-Similarity Loss."""

from .config import Gate4SliceMSConfig, gate4_config_from_module
from .probe_model import GateSliceProbeModel, build_gate_slice_probe
from .slice_encoder import SliceMSEncoder
from .slice_ms_loss import MultiSimilarityLoss, SliceMultiSimilarityLoss
from .training import Gate4CombinedLoss, Gate4SliceMSLossOnly

__all__ = [
    "Gate4CombinedLoss",
    "Gate4SliceMSLossOnly",
    "Gate4SliceMSConfig",
    "GateSliceProbeModel",
    "MultiSimilarityLoss",
    "SliceMSEncoder",
    "SliceMultiSimilarityLoss",
    "build_gate_slice_probe",
    "gate4_config_from_module",
]
