"""Fase D — Entrenamiento Stage1 (gate M+/M-) multi-rama.

Ramas:
    A) DINOv2 ViT-S/14 + cabezal binario (semántica global).
    B) ResNet18 + cabezal binario (detalle local, diversidad de familia).
    C) FreqNet  custom 3ch -> (log|F|, cosφ, sinφ) (frecuencia/textura).

Fusión calibrada (Temperature scaling) ponderada con
w_A=0.45, w_B=0.35, w_C=0.20 (master plan §9).
"""

from .gpu_pipeline import (
    EpochPlan,
    GPUImageBatch,
    iter_image_batches,
    plan_epoch,
    split_by_image,
)
from .train_gpu import GPUTrainHistory, evaluate_branch_gpu, train_branch_gpu
from .infer_gpu import StageOneEnsembleGPU, infer_image_gpu
from .models import build_branch_a, build_branch_b, build_branch_c, count_parameters
from .losses import FocalLossBCE
from .calibrate import TemperatureScaler
from .fusion import fuse_probabilities, js_divergence, js_divergence_multiclass
from .gate_multiclass import (
    GateEnsembleGPU,
    build_gate_ensemble_from_trained,
    infer_image_gate_gpu,
    load_gate_am_ensemble,
    train_gate_branch_gpu,
)
from .gate_tile_dino import (
    GateTileDinoGPU,
    infer_image_gate_tile_dino_gpu,
    load_gate_tile_dino,
    train_gate_tile_dino_gpu,
)
from .gate_classes import GATE_CLASS_NAMES

__all__ = [
    "EpochPlan",
    "GPUImageBatch",
    "iter_image_batches",
    "plan_epoch",
    "split_by_image",
    "GPUTrainHistory",
    "evaluate_branch_gpu",
    "train_branch_gpu",
    "StageOneEnsembleGPU",
    "infer_image_gpu",
    "build_branch_a",
    "build_branch_b",
    "build_branch_c",
    "count_parameters",
    "FocalLossBCE",
    "TemperatureScaler",
    "fuse_probabilities",
    "js_divergence",
    "js_divergence_multiclass",
    "GATE_CLASS_NAMES",
    "GateEnsembleGPU",
    "GateTileDinoGPU",
    "build_gate_ensemble_from_trained",
    "infer_image_gate_gpu",
    "infer_image_gate_tile_dino_gpu",
    "load_gate_am_ensemble",
    "load_gate_tile_dino",
    "train_gate_branch_gpu",
    "train_gate_tile_dino_gpu",
]
