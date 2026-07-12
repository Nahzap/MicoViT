"""Configuracion central de MicorizaeVision.

Edita SOLO este archivo para cambiar parametros del pipeline gate AM.
Los subcomandos CLI (`python run.py train-gate-am`, etc.) leen estos valores
cuando no pasas flags explicitos.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# run.py — pipeline interactivo (`python run.py` sin argumentos)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gate AM v4 — multi-densidad + AMFinder
# ---------------------------------------------------------------------------

GATE_MULTIDENSITY_ENABLED: bool = False  # base 252px ~126k tiles; build HDF5 ~12 GB en minutos
GATE_TILE_SIZE_BASE: int = 252
GATE_TILE_SIZE_DENSE: int = 189
GATE_TILE_SIZE_COARSE: int = 336
GATE_MULTIDENSITY_TIERS: str = "dense,coarse"

GATE_AMFINDER_TRAIN_ENABLED: bool = True  # incluye subset amfinder_train en train Gate
# Post-train: benchmark holdout externo (NO Stage2). Compila embed auxiliar + evalua probe Gate.
GATE_AMFINDER_EXTERNAL_EVAL: bool = True
# 0 = todas las imagenes holdout (~29 imgs / ~161k tiles). 10 = subconjunto reproducible (seed=0).
GATE_AMFINDER_EXTERNAL_MAX_IMAGES: int = 10
# Embed auxiliar AMFinder: batch alto (RTX 3070 8GB suele usar <2 GB VRAM en este paso).
GATE_AMFINDER_EXTERNAL_CACHE_BATCH_SIZE: int = 48
GATE_AMFINDER_EXTERNAL_VRAM_BUDGET_MB: float = 7600.0
GATE_AMFINDER_EXTERNAL_CPU_DECODE_WORKERS: int = 4  # crops paralelos en RAM (flatbed scans)
GATE_AMFINDER_EXTERNAL_MEMMAP_FLUSH_EVERY_N_IMAGES: int = 10
GATE_AMFINDER_EXTERNAL_FUSED_ATTENTION: bool = True  # 1 forward ViT (embed+attn), ~2x vs doble pass
GATE_CACHE_BASENAME: str = "gate_am_embeds_v5"

# Pipeline sin prompts (ejecucion automatica)
INTERACTIVE_PROMPTS: bool = False

# Comportamiento sin terminal interactiva (CI / redireccion):
AUTO_BUILD_CACHE_IF_MISSING: bool = True
AUTO_USE_VALID_CACHE: bool = True
AUTO_REBUILD_CACHE: bool = True

# Pipeline velocidad SSD: HDF5 (RGB+saliencia) -> embeddings DINO -> train desde cache.
GATE_H5_CACHE_ENABLED: bool = True
GATE_EMBED_BUILD_FROM_H5: bool = True  # compilar embeddings leyendo HDF5 (sin JPEG)
# Paso 1 HDF5 — materialización ligera (sin inferencia)
GATE_H5_BATCH_SIZE: int = 256  # sin U2Net en build; mas tiles/batch en RTX 3070 8GB
GATE_H5_CHUNK_TILES: int = 64  # chunks HDF5
GATE_H5_COMPRESSION: str = "none"  # max velocidad escritura SSD (lzf ralentiza CPU+disco)
GATE_H5_GRAYSCALE: bool = True  # 1 canal luma (tinción monocromática); ~3x vs RGB
GATE_H5_STORE_SALIENCY: bool = False  # no guardar saliency (U2Net solo en train)
GATE_H5_SKIP_U2NET: bool = True  # sin U2Net en Paso 1
GATE_H5_U2NET_BG_ONLY: bool = True  # legacy; ignorado si GATE_H5_SKIP_U2NET=True
GATE_H5_STREAMING_DECODE: bool = True  # pyvips/cv2 por ventanas; evita PIL 2GB RGB (E)
GATE_H5_RESUME: bool = True  # reanudar build interrumpido sin truncar (G)
# False = no bloquear en Paso 1; E6 LoRA entrena desde JPEG si HDF5 no esta listo.
# Reanudar build luego: GATE_H5_BUILD_BEFORE_TRAIN=True (resume desde progreso parcial).
GATE_H5_BUILD_BEFORE_TRAIN: bool = True
# E6 LoRA: entrena desde HDF5; omitir Paso 2 embed cache principal (F)
GATE_E6_SKIP_EMBED_CACHE: bool = False  # E7 probe requiere embed cache

# ---------------------------------------------------------------------------
# Gate AM — cache de embeddings v2
# ---------------------------------------------------------------------------

GATE_BACKBONE: str = "dinov2_vits14"
# DINO: tiles AM 252px nativos, resize a 280px -> grilla 20x20 tokens @ patch14 (280/14=20).
# Antes 252px -> 18x18=324 tokens. Requiere recompilar cache si cambias este valor.
GATE_DINO_INPUT_SIZE: int = 280
GATE_SEG_TARGET_SIZE: int = 400  # escala con DINO (280 * 320/224)
GATE_CACHE_BATCH_SIZE: int = 12  # regimen alta velocidad 8GB VRAM (sin spill a RAM compartida)
GATE_CACHE_DYNAMIC_BATCH: bool = True
GATE_CACHE_VRAM_BUDGET_MB: float = 6200.0  # dejar ~2GB para U2Net/DINO sin spill
# Imagenes >300 MB RGB: decode CPU, solo mini-batches a GPU (evita VRAM llena).
GATE_CACHE_CPU_DECODE_ABOVE_MB: float = 300.0

# Limpieza GPU/RAM: 0 = solo al terminar cada imagen (mas rapido).
# Evitar gc.collect() cada batch — ralentiza mucho en imagenes grandes.
GATE_CACHE_EMPTY_CACHE_EVERY_N_BATCHES: int = 0
GATE_CACHE_GC_COLLECT_EVERY_N_BATCHES: int = 0
GATE_CACHE_MEMMAP_FLUSH_EVERY_N_IMAGES: int = 5

# Legacy (ignorado si EMPTY_CACHE_EVERY_N_BATCHES > 0):
GATE_CACHE_GPU_CLEANUP_EACH_BATCH: bool = False

# Cache de mapas de atencion ViT (CLS -> patch, por capa). ~0.9 GB @ 120k tiles.
GATE_CACHE_ATTENTION: bool = True
GATE_CACHE_ATTENTION_LAYERS: str = "all"  # all | last | 0,5,11
GATE_CACHE_ATTENTION_HEAD_REDUCE: str = "mean"  # mean | none (guarda todas las heads)
# Concatena mean-pool attn (L dims) al embed en el probe Slice-MS.
GATE_PROBE_USE_ATTENTION: bool = True

# E7: U2Net solo en tiles Background al compilar cache (M+/M- mean-pool sin mascara).
GATE_EMBED_POOLING_MODE: str = "bg_only"  # none | bg_only

# ---------------------------------------------------------------------------
# Gate AM — entrenamiento (linear probe desde cache)
# ---------------------------------------------------------------------------

# Gate4 + Slice Multi-Similarity Loss (plan G4 reducido)
GATE4_SLICE_MS_ENABLED: bool = True
GATE4_INCLUDE_UNKNOWN: bool = False  # True: incluir Unreadable (requiere recompilar cache)
GATE_MS_NUM_SLICES: int = 4
GATE_MS_EMBED_DIM: int = 128
GATE_MS_ALPHA: float = 2.0
GATE_MS_BETA: float = 50.0
GATE_MS_BASE: float = 0.5
GATE_MS_LOSS_WEIGHT: float = 1.0
GATE_MS_WARMUP_EPOCHS: int = 0  # MS-pura: sin rampa CE

# --- Pair-aware mining Slice-MS (frontera M-/M+ + ancla BG) ---
# (1) Semi-hard Wang 2019 + banda OR para pares confundibles (no perder sim ~0.45-0.60).
GATE_MS_HARD_MINING: bool = True
GATE_MS_MINING_MARGIN: float = 0.1
# Pares: clinico M-/M+ + guardia BG/M+ (sin BG/M-: empuja M- hacia fondo y sube BG->M-).
GATE_MS_CONFUSABLE_PAIRS: str = "Mminus:Mplus,Background:Mplus"
GATE_MS_CONFUSABLE_NEG_WEIGHT: float = 1.1
GATE_MS_CONFUSABLE_GUARD_NEG_WEIGHT: float = 1.1
GATE_MS_CONFUSABLE_BASE: float = 0.48  # < base=0.5 endurece negativos confundibles
GATE_MS_CONFUSABLE_BAND_LOW: float = 0.40
GATE_MS_CONFUSABLE_BAND_HIGH: float = 0.62
# E1: mining asimétrico M-/M+ + curriculum banda OR desde ep6.
GATE_MS_CONFUSABLE_DIRECTED_WEIGHTS: str = "Mminus:Mplus:1.5,Mplus:Mminus:1.1,Background:Mplus:1.1"
GATE_MS_BAND_START_EPOCH: int = 13
# (4) Sub-prototipos: K=1 mientras optimizamos mining (subcentros OFF).
GATE_PROTO_SUBCENTERS: int = 1  # fallback k_max si BY_CLASS vacio
GATE_PROTO_SUBCENTERS_BY_CLASS: str = ""  # E7: K=1 todas (recipe BEST_LAST)
# OFF por defecto: K asimetrico modela modos intra-clase; domain-aware reparte slots
# por dominio y en holdout nativo solo usa 1 slot efectivo. Activar solo con
# eval externa estratificada y diseno K×dominio validado.
GATE_DOMAIN_AWARE_SUBCENTERS: bool = False
GATE_TRAIN_DOMAIN_STRATIFIED: bool = False  # E7: alinear con BEST_LAST
# Tier B: incluir min_recall en tile_edge al seleccionar checkpoint (peso compuesto).
GATE_CHECKPOINT_TILE_EDGE_WEIGHT: float = 0.2
GATE_CHECKPOINT_TILE_EDGE_TARGETS: str = "126,252"

GATE_PROBE: bool = True  # E7: probe+cache (recipe BEST_LAST); LoRA pausado
# E6/E6b: partial_lora = LoRA en bloques 9-11 + unfreeze ultimos N bloques ViT.
# Adaptacion de dominio AM (Cell-DINO 2025 style): LoRA supervisado sobre tiles nativos.
GATE_FINETUNE_MODE: str = "partial_lora"
GATE_FINETUNE_LAST_N_BLOCKS: int = 2  # bloques 10-11 full; bloque 9 solo LoRA (ExPLoRA)
GATE_LORA_ENABLED: bool = True
GATE_LORA_RANK: int = 8
GATE_LORA_ALPHA: float = 16.0
GATE_LORA_BLOCKS: str = "9,10,11"
GATE_LORA_TARGET: str = "qkv"
GATE_FINETUNE_BACKBONE_LR_FACTOR: float = 0.1
GATE_EPOCHS: int = 40  # E7: igual BEST_LAST + early stop
GATE_MIN_EPOCHS: int = 5
GATE_EARLY_STOP_PATIENCE: int = 20
GATE_EARLY_STOP_MIN_DELTA: float = 0.005
GATE_TRAIN_BATCH_SIZE: int = 48  # E7 probe+cache (BEST_LAST)
# E6 JPEG: decode en CPU (cv2); GPU solo recibe mini-batches de tiles (no imagen completa en VRAM).
GATE_TRAIN_FORCE_CPU_DECODE: bool = True
GATE_TRAIN_CPU_DECODE_ABOVE_MB: float = 300.0  # si FORCE=False, umbral como embed cache
GATE_MAX_BG_PER_IMAGE: int = 50
GATE_FULL_DATASET: bool = True  # False = modo rapido (pocos batches/epoca)

# Protocolo Evangelisti G1 + buenas practicas (IEEE/ML): balanceo y checkpoint.
# g1_stratified: batches mixtos Bg/M-/M+ (mejor para Slice-MS y recall M+/M-).
# balanced_4class: balance por imagen (legacy).
GATE_BALANCE_MODE: str = "g1_stratified"
GATE_STRATIFIED_SAMPLES_PER_CLASS: int = 3840
GATE_STRATIFIED_MIN_PER_BATCH: int = 0  # 0 = auto batch_size // 3
# (3) Cuota por clase en el batch. Vacio = uniforme (baseline).
GATE_STRATIFIED_CLASS_WEIGHTS: str = ""  # E7: uniforme como BEST_LAST
# E3: fracción de batches solo M-/M+ (sin BG) para pares duros.
GATE_STRATIFIED_ROOT_ONLY_BATCH_RATIO: float = 0.5
# E4: oversample tiles de imágenes difíciles (ABS710 local).
GATE_HARD_IMAGES: str = "ABS710,AFF756"
GATE_HARD_IMAGE_WEIGHT: float = 3.0
GATE_EVAL_BALANCE_MODE: str = "dual"  # g1_stratified (rapido) | dual (strat + holdout)
GATE_EVAL_STRATIFIED_SAMPLES_PER_CLASS: int = 384
# 0 = holdout 29k solo al finalizar (ahorra ~1822 batches x DINO cada epoca)
GATE_HOLDOUT_EVAL_EVERY_N_EPOCHS: int = 5
GATE_CHECKPOINT_EVAL: str = "natural"  # E7: holdout natural como BEST_LAST
GATE_CHECKPOINT_METRIC: str = "min_class_recall"  # mAP | macro_f1 | min_class_recall | evangelisti_g1
# E2: score compuesto 0.7*min_recall + 0.3*bg_p10 al seleccionar checkpoint.
GATE_CHECKPOINT_COMPOSITE_BG_WEIGHT: float = 0.3
# slice_ms_only = unica perdida academica (Wang et al. 2019 + slices Melivision)
GATE_FORMAL_TRAIN: bool = True  # True: bloquea CE/focal; pipeline listo para publicacion
GATE_LOSS: str = "slice_ms_only"  # slice_ms_only | ce | focal_ce (legacy, solo si FORMAL=False)
# prototype: sub-centros EMA (Deng ECCV 2020); knn: solo analisis post-train (embed_analysis)
GATE_METRIC_INFERENCE: str = "prototype"
GATE_FOCAL_GAMMA: float = 2.0  # legacy; ignorado con slice_ms_only
GATE_CALIBRATE_POST_TRAIN: bool = False  # MS-pura: inferencia por prototipos (sin T-scaling ad-hoc)
GATE_PROBE_LR: float = 2e-4  # mas bajo + warmup: estabiliza probe Slice-MS (evita mode-flip)
GATE_PROBE_LR_WARMUP_EPOCHS: int = 5
GATE_PROBE_LR_MIN_FACTOR: float = 0.1  # cosine decay hasta 10% del LR pico
GATE_FINETUNE_LR: float = 1e-5  # E6: ultimos bloques ViT
GATE_MPLUS_OVERSAMPLE_FACTOR: float = 1.0  # 1.0 = pareo M+/M- sin oversample (boost >1 colapsa a M+)
GATE_MPLUS_FOCAL_BOOST: float = 1.0  # 1.0 = sin peso extra; usar aug en cache para M+
GATE_CACHE_MPLUS_AUGMENT: bool = True  # views extra M+ al compilar cache (embeddings distintos)
GATE_CACHE_MPLUS_AUG_VARIANTS: str = "hflip,vflip"  # hflip | vflip | rot90 (solo train M+)
GATE_USE_CLASS_WEIGHTS_TRAIN: bool = False  # True solo con cap_bg sin balance fuerte
GATE_EVAL_UNWEIGHTED_LOSS: bool = True

# Umbrales G1 (Evangelisti et al. 2021 — holdout am_test).
GATE_RECALL_THRESH_MPLUS: float = 0.90
GATE_RECALL_THRESH_MMINUS: float = 0.90
GATE_RECALL_THRESH_BACKGROUND: float = 0.85
GATE_RECALL_THRESH_UNKNOWN: float = 0.80
GATE_SPEC_THRESH_MPLUS: float = 0.90
GATE_SPEC_THRESH_MMINUS: float = 0.90
GATE_SPEC_THRESH_BACKGROUND: float = 0.90
GATE_SPEC_THRESH_UNKNOWN: float = 0.85

# Solo si GATE_FULL_DATASET = False:
GATE_FAST_MAX_TRAIN_BATCHES: int = 25
GATE_FAST_MAX_VAL_BATCHES: int = 12
GATE_FAST_MAX_BG_PER_IMAGE: int = 15

GATE_RENDER_MAPS: bool = True
GATE_VIS_ALL_TEST: bool = True  # las 10 imagenes holdout (~3 s/img, decode CPU)
GATE_MAX_VIS_IMAGES: int | None = None  # cap opcional (ej. 3); None = todas si VIS_ALL_TEST
GATE_VIS_DOWNSCALE: int = 4

# Transparencia y velocidad (evita re-inferencia U2Net+DINO sobre 29k tiles).
GATE_REPORT_FROM_CACHE: bool = True  # eval rapida post-train desde cache
GATE_SAVE_LIVE_SNAPSHOTS: bool = True  # live_metrics.json + live_curves.png cada epoca
# Reentrenamientos iterativos (mismo manifest/tiles): saltar pasos lentos de arranque.
GATE_SPATIAL_AUDIT: bool = False  # ~40s; activar solo tras cambio de manifest/splits
GATE_SPATIAL_AUDIT_SAMPLE: int = 5  # PNGs de muestra solo si hay duda visual
GATE_SKIP_PRETRAIN_VIZ: bool = False  # E7: stats + muestras pre-train

# ---------------------------------------------------------------------------
# Inferencia (`python run.py infer-gate-am`)
# ---------------------------------------------------------------------------

DEFAULT_IMAGE: Path | None = (
    PROJECT_ROOT / "Data" / "am" / "am" / "train" / "10E_2L_E_Default_Extended.jpg"
)
DEFAULT_TAU_S1: float = 0.65
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LINEAGE: str = "AM"

# ---------------------------------------------------------------------------
# ViT-S2 — Stage2 discriminador subclases en tiles M+ (plan 20260628_225244)
# ---------------------------------------------------------------------------

STAGE2_GATE_RUN_ID: str = "20260624_012932_gate_am_train"
STAGE2_LINEAGE_DEFAULT: str = "AM"
STAGE2_BACKBONE: str = "dinov2_vits14"
STAGE2_DINO_INPUT_SIZE: int = 224
STAGE2_HEAD_HIDDEN: int = 256
STAGE2_DROPOUT: float = 0.2
STAGE2_EPOCHS: int = 30
STAGE2_BATCH_SIZE: int = 16
STAGE2_LR: float = 1e-4
STAGE2_BACKBONE_LR_FACTOR: float = 0.1
STAGE2_VAL_FRACTION: float = 0.2
STAGE2_FREEZE_BACKBONE: bool = False
STAGE2_LOSS: str = "focal"  # ce | focal
STAGE2_FOCAL_GAMMA: float = 2.0
STAGE2_HYBRID_OVERSAMPLE: int = 8
STAGE2_TRAIN_SUBSETS: str = "am_train"
STAGE2_BRANCHES: str = "A"
STAGE2_CHECKPOINT_METRIC: str = "f1_macro"
STAGE2_TAU_GATE_MPLUS: float = 0.65
STAGE2_USE_AMP: bool = True
STAGE2_SKIP_B: bool = True
STAGE2_SKIP_C: bool = True

# ---------------------------------------------------------------------------
# Fase 2 píxel — morfología IH/A/V/H (plan 20260629_134156)
# ---------------------------------------------------------------------------

STAGE2_PIXEL_ENABLED: bool = True
STAGE2_PIXEL_BACKEND: str = "vit"  # weak | vit | ensemble
STAGE2_PIXEL_VIT_MODEL: str = "dinov2_vits14"
STAGE2_PIXEL_INPUT_SIZE: int = 224
STAGE2_PIXEL_FREEZE_BACKBONE: bool = True  # 8 GB VRAM: solo decoder, sin memoria compartida
STAGE2_PIXEL_EPOCHS: int = 20
STAGE2_PIXEL_BATCH_SIZE: int = 128  # óptimo ~tiles/s; 256↑VRAM pero I/O HDF5 limita (~16 batches/ep)
STAGE2_PIXEL_GRAD_ACCUM_STEPS: int = 1
STAGE2_PIXEL_LR: float = 1e-4
STAGE2_PIXEL_BACKBONE_LR_FACTOR: float = 0.1
STAGE2_PIXEL_VAL_FRACTION: float = 0.2
STAGE2_PIXEL_USE_AMP: bool = True
STAGE2_PIXEL_GATE_RUN_ID: str = "20260710_121127_gate_am_train"
STAGE2_PIXEL_VESICLE_CIRCULARITY_MIN: float = 0.72
STAGE2_PIXEL_FRANGI_PCTL: float = 82.0
STAGE2_PIXEL_ARBUSCULE_PCTL: float = 93.0
# Pseudo-GT v3: híbrido contorno cerrado + semilla LoG refinada (sin discos sintéticos)
STAGE2_PIXEL_PSEUDO_GT_VERSION: str = "v3_hybrid_contour_log"
STAGE2_PIXEL_VESICLE_MAX_RADIUS: int = 0  # 0 = adaptativo (min(tile/2, 80))
STAGE2_PIXEL_VESICLE_MAX_SIGMA: float = 40.0  # ATLAS multi-escala (antes 16)
STAGE2_PIXEL_AMBIGUOUS_TO_H_DENSE: bool = True  # saturación homogénea → H, no ignore
STAGE2_PIXEL_H5_ENABLED: bool = True
STAGE2_PIXEL_H5_COMPRESSION: str = "none"  # build: none = max velocidad; train usa RAM cache
STAGE2_PIXEL_H5_FORCE_REBUILD: bool = False   # one-shot completado 2026-07-08
STAGE2_PIXEL_H5_STORE_STAIN_CHANNELS: bool = True  # density_norm + stain_residual en H5
STAGE2_PIXEL_H5_PROFILE_ON_TRAIN: bool = False  # skip ~15s scan en cada arranque si cache hit
STAGE2_PIXEL_H5_STORE_PRIORS: bool = True  # materializa Frangi/entropía en HDF5 (one-shot, train solo GPU)
STAGE2_PIXEL_H5_BATCH_PREFETCH: bool = True  # precarga siguiente batch HDF5 mientras GPU entrena
STAGE2_PIXEL_H5_RAM_CACHE: bool = True  # ~4GB RAM: carga H5 una vez, evita lzf/batch (~3-5x tiles/s)
STAGE2_PIXEL_H5_BUILD_BATCH: int = 128  # más tiles por pool.map → satura 14 workers
STAGE2_PIXEL_H5_BUILD_WORKERS: int = 14  # i7-11800H: 16 lógicos − 2 (main + GPU decode)
STAGE2_PIXEL_H5_BUILD_INFLIGHT: int = 4  # batches CPU en vuelo mientras GPU decodifica
# Flip oversample tiles con vesículas (train only)
STAGE2_PIXEL_FLIP_OVERSAMPLE_V: bool = True
STAGE2_PIXEL_V_MIN_PX: int = 30
STAGE2_PIXEL_RARE_FLIP_VARIANTS: int = 4
# Legacy alias (usar STAGE2_PIXEL_INPUT_SIZE)
STAGE2_PIXEL_SEG_SIZE: int = 224

# --- Slice Multi-Similarity metric loss (Wang et al. 2019) sobre embeddings de píxel ---
STAGE2_PIXEL_MS_LOSS_ENABLED: bool = True   # metric learning aux; estructura el espacio, robusto a ruido de label
STAGE2_PIXEL_MS_WEIGHT: float = 0.5
STAGE2_PIXEL_MS_SLICES: int = 4
STAGE2_PIXEL_MS_K_PER_CLASS: int = 256
STAGE2_PIXEL_MS_ALPHA: float = 2.0
STAGE2_PIXEL_MS_BETA: float = 50.0
STAGE2_PIXEL_MS_BASE: float = 0.5

# --- Loss principal densa ---
STAGE2_PIXEL_LOSS_TYPE: str = "ce"               # "ce" | "focal" (focal era parche anti-colapso; con labels stain-aware + MS, CE es más limpia)
STAGE2_PIXEL_FOCAL_GAMMA: float = 2.0            # gamma para focal loss
STAGE2_PIXEL_CLASS_WEIGHTS: str = "inv_freq"     # "uniform" | "inv_freq" | "inv_sqrt"
STAGE2_PIXEL_DECODER_TYPE: str = "multiscale"    # "simple" | "multiscale"
STAGE2_PIXEL_AUX_ENTROPY_LOSS: bool = True       # loss auxiliar stage2 gold
STAGE2_PIXEL_AUX_ENTROPY_WEIGHT: float = 0.1     # peso de la loss auxiliar
STAGE2_PIXEL_AUGMENT_H5: bool = True             # augmentación en path H5
STAGE2_PIXEL_UNFREEZE_LAST_N: int = 2            # descongelar últimos N bloques backbone
# Augmentations foco/tinción (UnMICST / AMFinder / PSF-Net)
STAGE2_PIXEL_AUG_PSF_SIGMAS: str = "0.5,1.0,1.5,2.0,2.5"
STAGE2_PIXEL_AUG_PSF_PROB: float = 0.35
STAGE2_PIXEL_AUG_STAIN_SCALE_LO: float = 0.7
STAGE2_PIXEL_AUG_STAIN_SCALE_HI: float = 1.3
# Decoder + canales stain (Ruifrok) — concat al fuse, no al backbone DINO
STAGE2_PIXEL_DECODER_STAIN_CHANNELS: bool = True
STAGE2_PIXEL_STAIN_AUX_LOSS_WEIGHT: float = 0.02

# --- MEViT explicabilidad (E2-EX) ---
STAGE2_PIXEL_EXPLAIN_ENABLED: bool = True
STAGE2_PIXEL_EXPLAIN_ATTENTION_LAYERS: str = "last"  # all | last | 0,5,11
STAGE2_PIXEL_EXPLAIN_EXPORT_PROBS: bool = True
STAGE2_PIXEL_EXPLAIN_EXPORT_PRIORS: bool = True
STAGE2_PIXEL_EXPLAIN_NARRATIVE: bool = True

# --- Posttrain automático tras cada entrenamiento ---
STAGE2_PIXEL_POSTTRAIN_FULL_REPORT: bool = True
STAGE2_PIXEL_POSTTRAIN_SKIP_FULLIMAGE: bool = False  # mapas full-image secuenciales Gate→Stage2
STAGE2_PIXEL_POSTTRAIN_MAX_FULLIMAGE_VAL: int = 6
STAGE2_PIXEL_POSTTRAIN_MAX_FULLIMAGE_TEST: int = 6
STAGE2_PIXEL_POSTTRAIN_EXPLAIN_PANELS: bool = True
STAGE2_GATE_INFERENCE_STRICT: bool = True  # Modelo 1: sin fallback a gold stage1 del manifest
STAGE2_POSTTRAIN_REQUIRE_GATE_CACHE: bool = True  # exige build-gate-cache antes de full-image

# --- MEViT prior losses (E2-EX.2) ---
STAGE2_PIXEL_PRIOR_LOSS_ENABLED: bool = True
STAGE2_PIXEL_PRIOR_LOSS_IH_WEIGHT: float = 0.05
STAGE2_PIXEL_PRIOR_LOSS_V_WEIGHT: float = 0.08  # v4: ancla vesiculas multi-escala
STAGE2_PIXEL_PRIOR_LOSS_PREC_WEIGHT: float = 0.02
STAGE2_PIXEL_PRIOR_LOSS_A_WEIGHT: float = 0.05  # v4: arb-score Gallaud
STAGE2_PIXEL_PRIOR_LOSS_H_STAIN_WEIGHT: float = 0.03  # v4: separa H de estructura
STAGE2_PIXEL_PRIOR_LOSS_WORKERS: int = 4  # solo si priors NO están en HDF5 (fallback runtime)

