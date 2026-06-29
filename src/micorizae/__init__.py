"""micorizae — pipeline multi-fase y multi-capa para identificación de AM/ERM.

Estructura por fases (alineada al MASTER_PLAN_CAPAS_VISUALES_AMF):
    - phase_a_ingest    : ingesta + manifests (E1)
    - phase_b_tiling    : tiling + sincronización espacial
    - phase_c_views     : vistas RGB / seg / freq por tile
    - phase_d_stage1    : entrenamiento gate (M+/M-)
    - phase_e_stage2    : entrenamiento subclases en M+
    - phase_f_quality   : fusión + filtros de pseudo-etiqueta
    - phase_g_selftrain : self-training por rondas
    - phase_h_infer     : inferencia final + cuantificación
    - phase_i_weakseg   : GT morfologico weakly-supervised + puente DINOv2

Capas visuales auditables L0..L10 en `micorizae.layers`.
"""

__version__ = "0.1.0"

# tqdm: robustez en ejecución no interactiva (logs redirigidos / background).
#  1) monitor_interval=0 desactiva el hilo TMonitor que en Windows provoca un
#     deadlock al cerrar barras (MainThread join() vs monitor esperando el lock
#     global de tqdm), congelando el run sin CPU ni I/O.
#  2) Si stderr no es TTY (redirección `*>`, captura por pipe, background),
#     dynamic_ncols consulta la API de consola de Windows sobre un handle no-consola
#     y lanza OSError [Errno 22] Invalid argument; además las barras escriben miles
#     de líneas \r al log. En ese caso se desactivan las barras. En terminal real
#     (TTY) se mantienen intactas.
try:  # pragma: no cover - defensivo ante entornos sin tqdm
    import sys as _sys

    import tqdm as _tqdm

    _tqdm.tqdm.monitor_interval = 0

    _stderr = getattr(_sys, "stderr", None)
    _is_tty = bool(getattr(_stderr, "isatty", lambda: False)())
    if not _is_tty:
        _orig_tqdm_init = _tqdm.std.tqdm.__init__

        def _safe_tqdm_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            kwargs["dynamic_ncols"] = False
            kwargs.setdefault("ncols", 100)
            kwargs.setdefault("mininterval", 1.0)
            if kwargs.get("disable") is None:
                kwargs["disable"] = True
            _orig_tqdm_init(self, *args, **kwargs)

        _tqdm.std.tqdm.__init__ = _safe_tqdm_init
except Exception:  # noqa: BLE001
    pass
