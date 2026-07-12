"""Shared WeakSeg / morph detector parameters (I↔E contract)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeakSegParams:
    tile_size: int = 126
    canny_low: int = 40
    canny_high: int = 110
    close_kernel: int = 5
    frangi_pctl: float = 82.0
    vesicle_circularity_min: float = 0.85
    vesicle_min_area: int = 30
    entropy_radius: int = 2
    arbuscule_pctl: float = 93.0
    seam_sigma: float = 0.85
    alpha_overlay: float = 0.45
    root_min_cov: float = 0.02
    root_max_cov: float = 0.70
    min_patch_fg_ratio: float = 0.002
    # --- Detector stain-aware (color deconvolution + estructura) ---
    stain_aware: bool = True
    stain_bg_maxc: int = 232          # canal max > umbral ⇒ fondo blanco
    stain_bg_sat: float = 0.10        # saturación mínima para considerar tejido
    stain_pctl: float = 60.0          # percentil de densidad para "hay tinción" (colonización)
    ves_min_sigma: float = 2.0        # LoG blob vesículas (px)
    ves_max_sigma: float = 40.0       # ATLAS multi-escala (plan v4; antes 16)
    ves_num_sigma: int = 12           # escalas LoG (Lindeberg 1998; skimage blob_log)
    ves_blob_thr: float = 0.038       # umbral LoG para semillas (refinadas a contorno)
    ves_solidity_min: float = 0.82
    ves_roundness_min: float = 0.72
    ves_aspect_min: float = 0.52
    ves_stain_z: float = 0.45
    ves_contrast_min: float = 0.06
    ves_bg_max: float = 0.45
    ves_density_pctl: float = 76.0
    ves_max_area_frac: float = 0.15
    ves_max_instances: int = 20
    ves_nms_dist_ratio: float = 0.55
    arb_fine_pctl: float = 88.0       # textura fina alta (energía alta-frecuencia)
    arb_density_pctl: float = 80.0    # densidad de tinción alta
    arb_min_area: int = 40            # área mínima de arbúsculo (px)
    ves_max_radius: int = 0           # 0 = adaptativo min(tile/2, 80); antes 30 descartaba gigantes
    ih_frangi_sigmas: tuple = (1.0, 2.0, 3.0, 4.0)
    ih_tophat_disk: int = 9           # top-hat prominencia de cresta (hifa fina)
    ih_prom_pctl: float = 65.0        # percentil de prominencia (piso relativo)
    ih_tophat_abs: float = 0.055      # piso ABSOLUTO de prominencia — mata speckle en azul uniforme
    ih_min_area: int = 18             # elimina fragmentos IH diminutos (px)
    # --- Ambigüedad / ignore: azul saturado en área grande (estructura no resoluble) ---
    amb_density: float = 0.60         # OD alto = tinción intensa
    amb_blueness: float = 0.60        # azuleza alta
    amb_win: int = 48                 # ventana px para fracción local saturada
    amb_frac: float = 0.55            # fracción saturada en la ventana ⇒ región grande no resoluble
