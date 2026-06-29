"""Fase E — Entrenamiento Stage2 (subclases en M+).



Entrada: solo tiles cuyo Stage1 == M+ con subclase anotada.

Modelos: DINOv2 multiclase, U2Net subestructura, FrequencyNet multiclase.

"""



from .class_map import Stage2ClassMap, load_stage2_class_map

from .fusion import DEFAULT_WEIGHTS, entropy, fuse_probabilities_mc, js_divergence_mc

from .gpu_pipeline import (

    GPUStage2Batch,

    filter_mplus_stage2,

    iter_image_batches_stage2,

    plan_epoch_stage2,

    split_by_image_mplus,

)

from .infer_gpu import StageTwoEnsembleGPU, infer_image_stage2_gpu, infer_tiles_stage2_gpu

from .models import (

    BranchFrequencyMC,

    BranchSegmentationMC,

    BranchSemanticMC,

    build_branch_a_mc,

    build_branch_b_mc,

    build_branch_c_mc,

    count_parameters,

)

from .train_gpu import GPUTrainHistoryS2, evaluate_branch_gpu_s2, train_branch_gpu_s2



__all__ = [

    "Stage2ClassMap",

    "load_stage2_class_map",

    "DEFAULT_WEIGHTS",

    "entropy",

    "fuse_probabilities_mc",

    "js_divergence_mc",

    "GPUStage2Batch",

    "filter_mplus_stage2",

    "iter_image_batches_stage2",

    "plan_epoch_stage2",

    "split_by_image_mplus",

    "StageTwoEnsembleGPU",

    "infer_image_stage2_gpu",

    "infer_tiles_stage2_gpu",

    "BranchSemanticMC",

    "BranchSegmentationMC",

    "BranchFrequencyMC",

    "build_branch_a_mc",

    "build_branch_b_mc",

    "build_branch_c_mc",

    "count_parameters",

    "GPUTrainHistoryS2",

    "evaluate_branch_gpu_s2",

    "train_branch_gpu_s2",

]

