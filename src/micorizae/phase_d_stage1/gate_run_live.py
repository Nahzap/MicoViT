"""Publicación incremental de artefactos durante el entrenamiento gate AM.

Escribe en ``outputs/<run_id>/`` en cada época (no solo al finalize final).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..common.logging_utils import get_logger
from ..common.run_outputs import RunOutputs
from .gate_classes import GATE_CLASS_NAMES
from .gate_run_layout import GateRunLayout, layout_for_run, plot_class_distribution, plot_loss_curves, plot_train_val_balance
from .gate_tile_dino import CHECKPOINT_NAME, PROGRESS_NAME
from .gate_training_protocol import format_g1_status
from .train_gpu import GPUTrainHistory

log = get_logger("phase_d.gate_live")
CANVAS_VERSION = 16
PIPELINE_PHASES_MARKER = "pipeline_phases"
_HIST_METRIC_KEYS = (
    "epochs_done",
    "train_loss",
    "val_loss",
    "val_macro_f1",
    "val_acc",
    "val_min_class_recall",
    "g1_pass_by_epoch",
    "epoch_details",
    "best_epoch",
    "best_score",
    "best_checkpoint_score",
    "last_per_class_recall",
    "last_per_class_spec",
    "last_g1_pass",
    "last_min_class_recall",
    "recall_targets",
    "spec_targets",
    "tiles_train_per_epoch",
)


def _default_recall_spec_targets() -> tuple[dict[str, float], dict[str, float]]:
    try:
        import config as cfg

        recall = {
            "Mplus": float(cfg.GATE_RECALL_THRESH_MPLUS),
            "Mminus": float(cfg.GATE_RECALL_THRESH_MMINUS),
            "Background": float(cfg.GATE_RECALL_THRESH_BACKGROUND),
            "Unknown": float(cfg.GATE_RECALL_THRESH_UNKNOWN),
        }
        spec = {
            "Mplus": float(cfg.GATE_SPEC_THRESH_MPLUS),
            "Mminus": float(cfg.GATE_SPEC_THRESH_MMINUS),
            "Background": float(cfg.GATE_SPEC_THRESH_BACKGROUND),
            "Unknown": float(cfg.GATE_SPEC_THRESH_UNKNOWN),
        }
        return recall, spec
    except Exception:
        return {}, {}


def _run_root_for_id(run_id: str) -> Path:
    try:
        from ..common.paths import get_paths

        return get_paths().root / "outputs" / run_id
    except Exception:
        return Path.cwd() / "outputs" / run_id


def _load_run_metrics(run_id: str) -> dict:
    run_root = _run_root_for_id(run_id)
    for rel in ("reports/live_metrics.json", "logs/live_metrics.json"):
        data = _read_json_if_exists(run_root / rel)
        if data:
            return data
    return {}


def _enrich_canvas_payload(payload: dict) -> dict:
    """Fusiona progreso en vivo + historial de epocas antes de parchear el canvas."""
    out = dict(payload)
    pipeline = _read_json_if_exists(_pipeline_live_path())
    if isinstance(pipeline.get("phases"), list) and pipeline["phases"]:
        out["phases"] = pipeline["phases"]
    if pipeline.get("pipeline_phase") is not None:
        out["pipeline_phase"] = pipeline["pipeline_phase"]
    if pipeline.get("run_id") and not out.get("run_id"):
        out["run_id"] = pipeline["run_id"]
    run_id = str(out.get("run_id") or "")
    metrics = _load_run_metrics(run_id) if run_id else {}

    for k in _HIST_METRIC_KEYS:
        if k in metrics:
            out[k] = metrics[k]
    for k in (
        "epochs_done",
        "train_loss",
        "val_loss",
        "val_macro_f1",
        "val_acc",
        "epoch_details",
        "best_epoch",
        "best_score",
        "best_checkpoint_score",
    ):
        if k in metrics:
            out[k] = metrics[k]

    for k in (
        "epoch_current",
        "batch_current",
        "batches_train_this_epoch",
        "batches_test_this_epoch",
        "train_loss_avg",
        "eta_epoch",
        "eta_total",
        "status",
        "phases",
        "updated_at",
        "tiles_train_per_epoch",
        "checkpoint_metric",
        "epochs_total",
        "run_id",
        "pipeline_phase",
    ):
        if payload.get(k) is not None:
            out[k] = payload[k]

    if out.get("best_epoch") is None and metrics.get("best_epoch") is not None:
        out["best_epoch"] = metrics["best_epoch"]
    if out.get("best_checkpoint_score") is None:
        out["best_checkpoint_score"] = metrics.get("best_checkpoint_score") or metrics.get(
            "best_score"
        )
    if out.get("best_score") is None:
        out["best_score"] = metrics.get("best_score")

    details = out.get("epoch_details")
    if isinstance(details, list) and details:
        last = details[-1]
        out.setdefault("last_per_class_recall", last.get("per_class_recall"))
        out.setdefault("last_per_class_spec", last.get("per_class_specificity"))
        out.setdefault("last_g1_pass", last.get("evangelisti_g1_pass"))
        out.setdefault("last_min_class_recall", last.get("min_class_recall"))
        if not out.get("val_min_class_recall"):
            out["val_min_class_recall"] = [
                float(d["min_class_recall"])
                for d in details
                if d.get("min_class_recall") is not None
            ]
        if not out.get("g1_pass_by_epoch"):
            out["g1_pass_by_epoch"] = [bool(d.get("evangelisti_g1_pass")) for d in details]

    if not out.get("recall_targets"):
        recall, spec = _default_recall_spec_targets()
        if recall:
            out["recall_targets"] = recall
        if spec:
            out["spec_targets"] = spec

    out["canvas_version"] = CANVAS_VERSION
    return out


_VM_CLASS_ORDER = ("Background", "Mminus", "Mplus", "Unknown")
_VM_CLASS_SHORT = {"Background": "Bg", "Mminus": "M-", "Mplus": "M+", "Unknown": "Unr"}


def _vm_num(x: Any, decimals: int = 4) -> str:
    """Formatea numero -> string para el canvas (— si falta/NaN)."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if xf != xf:  # NaN
        return "—"
    return f"{xf:.{decimals}f}"


def _vm_fdata(values: list, decimals: int = 4) -> list:
    """Lista de floats para charts (None/NaN -> 0.0, ChartSeries.data es number[])."""
    out: list = []
    for v in values:
        if v is None:
            out.append(0.0)
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            out.append(0.0)
            continue
        out.append(0.0 if vf != vf else round(vf, decimals))
    return out


def _vm_tone(status: Optional[str]) -> str:
    if status == "skipped":
        return "warning"
    if status in ("done", "ready"):
        return "success"
    if status == "running":
        return "info"
    return "neutral"


def _vm_status_label(status: Optional[str]) -> str:
    return "omitido" if status == "skipped" else (status or "pending")


def _vm_pct_bar(pct: Any, status: Optional[str]) -> Optional[float]:
    if status == "skipped" or pct is None:
        return None
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return max(0.0, min(100.0, v))


def _vm_phase(p: dict) -> dict:
    status = p.get("status")
    is_skipped = status == "skipped"
    bar = _vm_pct_bar(p.get("pct"), status)
    pct_label = "—" if bar is None else (f"{bar:.2f}%" if 0 < bar < 10 else f"{round(bar)}%")
    td, tt = p.get("tiles_done"), p.get("tiles_total")
    tiles_label = (
        f"{int(td):,} / {int(tt):,}" if (td is not None and tt is not None) else "—"
    )
    tps = p.get("tiles_per_second")
    speed_label = "—" if (is_skipped or tps is None) else f"{tps} t/s"
    return {
        "id": str(p.get("id") or ""),
        "label": str(p.get("label") or ""),
        "status_label": _vm_status_label(status),
        "tone": _vm_tone(status),
        "pct_label": pct_label,
        "pct_bar": bar if (bar is not None and bar > 0) else None,
        "eta_label": "—" if is_skipped else (p.get("eta") or "—"),
        "tiles_label": tiles_label,
        "speed_label": speed_label,
        "detail": p.get("detail") or "",
    }


def build_canvas_view_model(enriched: dict) -> dict:
    """Construye el view-model COMPLETO. El canvas solo mapea esto a componentes.

    Todo calculo/formateo (series, tablas, OK/FAIL, redondeo, labels) ocurre aqui.
    """
    m = enriched
    is_cache = int(m.get("pipeline_phase", 0) or 0) == 0
    ckpt_metric = m.get("checkpoint_metric") or "mAP"

    phases_vm = [_vm_phase(p) for p in (m.get("phases") or []) if isinstance(p, dict)]

    ep = list(m.get("epochs_done") or [])
    loss = list(m.get("train_loss") or [])
    val_loss = list(m.get("val_loss") or [])
    f1 = list(m.get("val_macro_f1") or [])
    val_acc = list(m.get("val_acc") or [])
    details = list(m.get("epoch_details") or [])

    min_recall = list(m.get("val_min_class_recall") or [])
    if not min_recall and details:
        min_recall = [d.get("min_class_recall") for d in details]
    g1pass = list(m.get("g1_pass_by_epoch") or [])
    if not g1pass and details:
        g1pass = [bool(d.get("evangelisti_g1_pass")) for d in details]
    maps = [d.get("mAP") for d in details]
    ckpt_scores = [d.get("checkpoint_score") for d in details]

    best_epoch = m.get("best_epoch")
    best_score = m.get("best_score")
    if best_score is None:
        best_score = m.get("best_checkpoint_score")

    last_recall = m.get("last_per_class_recall") or {}
    last_spec = m.get("last_per_class_spec") or {}
    last_g1 = m.get("last_g1_pass")
    last_min_recall = m.get("last_min_class_recall")
    recall_t = m.get("recall_targets") or {}
    spec_t = m.get("spec_targets") or {}

    # --- train stats ---
    epoch_current = m.get("epoch_current")
    epochs_total = m.get("epochs_total")
    batch_current = m.get("batch_current")
    batches_train = m.get("batches_train_this_epoch")
    batch_label = "—"
    if batch_current and batches_train:
        pct = round(100 * int(batch_current) / int(batches_train))
        batch_label = f"{batch_current}/{batches_train} ({pct}%)"
    loss_train = m.get("train_loss_avg")
    if loss_train is None and loss:
        loss_train = loss[-1]
    best_label = (
        f"{_vm_num(best_score)} @ ep{best_epoch} ({ckpt_metric})"
        if best_score is not None
        else "—"
    )
    train_stats = [
        {"label": "Epoca", "value": f"{epoch_current or 0}/{epochs_total or '—'}"},
        {"label": "Batch", "value": batch_label},
        {"label": "Loss train", "value": _vm_num(loss_train)},
        {"label": f"Best ({ckpt_metric})", "value": best_label},
    ]

    eta_stats: list = []
    if not is_cache:
        tiles_ep = m.get("tiles_train_per_epoch")
        carga = (
            f"{int(tiles_ep):,} tiles · {batches_train or '?'} train + "
            f"{m.get('batches_test_this_epoch') or '?'} val"
            if tiles_ep
            else "—"
        )
        eta_stats = [
            {"label": "ETA epoca", "value": m.get("eta_epoch") or "—"},
            {"label": f"ETA total ({epochs_total or 60} ep max)", "value": m.get("eta_total") or "—"},
            {"label": "Carga/epoca", "value": carga},
        ]

    g1_count = (
        f"{sum(1 for x in g1pass if x)}/{len(g1pass)}" if g1pass else "—"
    )
    target_stats = [
        {"label": "Objetivo checkpoint", "value": ckpt_metric},
        {"label": "Ult. min_recall", "value": _vm_num(last_min_recall)},
        {"label": "G1 Evangelisti", "value": "—" if last_g1 is None else ("PASS" if last_g1 else "FAIL")},
        {"label": "Epocas G1 PASS", "value": g1_count},
    ]

    # --- epoch curves ---
    epoch_curves = None
    if ep:
        series = [{"name": "train_loss", "data": _vm_fdata(loss)}]
        if any(v is not None for v in val_loss):
            series.append({"name": "val_loss", "data": _vm_fdata(val_loss)})
        series.append({"name": "macro_f1", "data": _vm_fdata(f1)})
        if any(v is not None for v in val_acc):
            series.append({"name": "val_acc", "data": _vm_fdata(val_acc)})
        if any(v is not None for v in maps):
            series.append({"name": "mAP", "data": _vm_fdata(maps)})
        if any(v is not None for v in min_recall):
            series.append({"name": "min_recall", "data": _vm_fdata(min_recall)})
        epoch_curves = {
            "categories": [str(e) for e in ep],
            "series": series,
            "caption": "train/val loss · macro_f1 · val_acc · mAP · min_recall",
        }

    # --- recall bars ---
    recall_bars = None
    if last_recall:
        cats = [c for c in _VM_CLASS_ORDER if (c in recall_t or c in last_recall)]
        recall_bars = {
            "categories": [_VM_CLASS_SHORT[c] for c in cats],
            "series": [
                {"name": "recall actual", "data": _vm_fdata([last_recall.get(c, 0) for c in cats], 4)},
                {"name": "umbral G1", "data": _vm_fdata([recall_t.get(c, 0) for c in cats], 4)},
            ],
            "caption": "verde=recall medido · azul=umbral objetivo (Evangelisti G1)",
        }

    # --- targets table ---
    targets_table = None
    if recall_t or spec_t:
        rows = []
        for cls in _VM_CLASS_ORDER:
            r, s = last_recall.get(cls), last_spec.get(cls)
            rt, st = recall_t.get(cls), spec_t.get(cls)
            ok = "—"
            if None not in (r, s, rt, st):
                ok = "si" if (float(r) >= float(rt) and float(s) >= float(st)) else "no"
            rows.append([
                _VM_CLASS_SHORT[cls],
                _vm_num(rt, 2),
                _vm_num(st, 2),
                _vm_num(r, 3),
                _vm_num(s, 3),
                ok,
            ])
        targets_table = {
            "columns": ["clase", "recall obj.", "spec obj.", "recall act.", "spec act.", "OK?"],
            "rows": rows,
        }

    # --- history table ---
    history_table = None
    if ep:
        rows = []
        for i, e in enumerate(ep):
            det = details[i] if i < len(details) else {}
            pcr = (det.get("per_class_recall") or {}) if isinstance(det, dict) else {}
            rows.append([
                str(e),
                _vm_num(loss[i] if i < len(loss) else None),
                _vm_num(val_loss[i] if i < len(val_loss) else None),
                _vm_num(f1[i] if i < len(f1) else None),
                _vm_num(val_acc[i] if i < len(val_acc) else None),
                _vm_num(maps[i] if i < len(maps) else None),
                _vm_num(min_recall[i] if i < len(min_recall) else None),
                _vm_num(ckpt_scores[i] if i < len(ckpt_scores) else None),
                _vm_num(pcr.get("Background"), 3),
                _vm_num(pcr.get("Mminus"), 3),
                _vm_num(pcr.get("Mplus"), 3),
                ("—" if i >= len(g1pass) or g1pass[i] is None else ("PASS" if g1pass[i] else "FAIL")),
            ])
        history_table = {
            "columns": ["ep", "train_loss", "val_loss", "macro_f1", "val_acc", "mAP",
                        "min_rec", "ckpt", "Bg", "M-", "M+", "G1"],
            "rows": rows,
        }

    eta_note = None
    if ep:
        eta_note = (
            f"ETA total = epocas restantes x (train + val_strat). Max {epochs_total or 60} ep; "
            "early stop paciencia 25 tras min 5 ep. Sin U2Net en train (mean-pool puro)."
        )

    return {
        "canvas_version": CANVAS_VERSION,
        "header": {
            "run_id": m.get("run_id") or "—",
            "status": m.get("status") or "—",
            "fase_label": "Phase 0 · cache" if is_cache else "Phase 1 · train",
            "fase_tone": "warning" if is_cache else "success",
            "fase_pill": "cache activa" if is_cache else "train activo",
            "is_cache": is_cache,
            "synced_at": str(m.get("updated_at") or ""),
            "checkpoint_metric": ckpt_metric,
        },
        "phases": phases_vm,
        "show_train": not is_cache,
        "train_stats": train_stats,
        "eta_stats": eta_stats,
        "target_stats": target_stats,
        "epoch_curves": epoch_curves,
        "recall_bars": recall_bars,
        "targets_table": targets_table,
        "history_table": history_table,
        "eta_note": eta_note,
    }


def refresh_gate_live_canvas(run_id: str | None = None) -> Optional[Path]:
    """Sincroniza canvas desde JSON en disco (util si el run usa codigo antiguo)."""
    progress = _read_json_if_exists(_pipeline_live_path())
    if run_id:
        progress = {**progress, "run_id": run_id}
    elif not progress.get("run_id"):
        try:
            from ..common.paths import get_paths

            outputs = get_paths().root / "outputs"
            runs = sorted(outputs.glob("*_gate_am_train"), key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                progress = {**progress, "run_id": runs[0].name}
        except Exception:
            pass
    run_id = str(progress.get("run_id") or "")
    if run_id:
        live = _read_json_if_exists(_run_root_for_id(run_id) / "reports" / "live_progress.json")
        progress = {**progress, **{k: v for k, v in live.items() if v is not None}}
        metrics = _read_json_if_exists(_run_root_for_id(run_id) / "reports" / "live_metrics.json")
        progress = {**progress, **{k: v for k, v in metrics.items() if v is not None}}
    # Actualiza plantilla si esta obsoleta (v10 -> v13 panel scroll propio).
    ensure_gate_live_canvas()
    _mirror_canvas_live_feed(progress)
    for canvases in _canvas_paths():
        p = canvases / "gate-am-live-metrics.canvas.tsx"
        if p.is_file():
            return p
    return None


def _sanitize_json_nan(obj: Any) -> Any:
    """Reemplaza float nan por None para JSON valido (canvas live feed)."""
    import math

    if isinstance(obj, dict):
        return {k: _sanitize_json_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_json_nan(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)


def _canvas_paths() -> list[Path]:
    home = Path.home()
    out: list[Path] = []
    for slug in ("f-MicorizaeVision", "MicorizaeVision"):
        out.append(home / ".cursor" / "projects" / slug / "canvases")
    out.append(Path.cwd() / "canvases")
    return out


def ensure_gate_live_canvas(*, force: bool = False) -> Optional[Path]:
    """Crea gate-am-live-metrics.canvas.tsx si falta (unico punto: run.py train)."""
    from .gate_live_canvas_template import GATE_LIVE_CANVAS_TSX

    name = "gate-am-live-metrics.canvas.tsx"
    for canvases in _canvas_paths():
        try:
            canvases.mkdir(parents=True, exist_ok=True)
            path = canvases / name
            needs_write = force or not path.is_file()
            if path.is_file() and not needs_write:
                try:
                    text = path.read_text(encoding="utf-8")
                    needs_write = (
                        PIPELINE_PHASES_MARKER not in text
                        or f"CANVAS_TEMPLATE_VERSION = {CANVAS_VERSION}" not in text
                        or "canvas_layout_v16" not in text
                    )
                except OSError:
                    needs_write = True
            if needs_write:
                path.write_text(GATE_LIVE_CANVAS_TSX, encoding="utf-8")
            return path
        except OSError:
            continue
    return None


def ensure_gate_live_canvas_exists() -> Optional[Path]:
    """Solo crea el canvas si no existe (nunca reescribe uno desplegado)."""
    for canvases in _canvas_paths():
        path = canvases / "gate-am-live-metrics.canvas.tsx"
        if path.is_file():
            return path
    return ensure_gate_live_canvas(force=True)


def print_train_console_banner(
    *,
    run_id: str,
    run_root: Path,
    mode_label: str,
    pooling_mode: str,
    batch_size: int,
    epochs: int,
    canvas_path: Optional[Path],
) -> None:
    """Mensajes clasicos de terminal (flush) al arrancar entrenamiento."""
    print("=" * 62, flush=True)
    print(f"Gate AM entrenamiento  run={run_id}", flush=True)
    print(f"  modo={mode_label}  pooling={pooling_mode}  batch={batch_size}  epochs<={epochs}", flush=True)
    print(f"  salida: {run_root}", flush=True)
    print(f"  live:   {run_root / 'reports' / 'live_metrics.json'}", flush=True)
    if canvas_path:
        print(f"  canvas: {canvas_path}", flush=True)
    print("  (canvas + metricas se actualizan desde run.py, sin scripts externos)", flush=True)
    print("=" * 62, flush=True)


def _pipeline_live_path() -> Path:
    try:
        from ..common.paths import get_paths

        return get_paths().root / "cache" / "gate_pipeline_live.json"
    except Exception:
        return Path.cwd() / "cache" / "gate_pipeline_live.json"


def _read_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _default_pipeline_phases() -> list[dict[str, Any]]:
    return [
        {
            "id": "hdf5_tiles",
            "label": "Paso 1: HDF5 tiles (luma+label)",
            "status": "pending",
            "pct": 0,
            "eta": None,
            "detail": "Pendiente",
        },
        {
            "id": "embed_v5",
            "label": "Paso 2: Embeddings v5 (DINO bg_only)",
            "status": "pending",
            "pct": 0,
            "eta": None,
            "detail": "Pendiente",
        },
        {
            "id": "train_slice_ms",
            "label": "Paso 3: Entrenamiento Slice-MS",
            "status": "pending",
            "pct": 0,
            "eta": None,
            "detail": "Pendiente",
        },
    ]


def _merge_pipeline_phases(base: list[dict], updates: list[dict]) -> list[dict]:
    base_by_id = {str(p.get("id")): dict(p) for p in base if isinstance(p, dict)}
    order = [str(p.get("id")) for p in base if isinstance(p, dict) and p.get("id")]
    for phase in updates:
        if not isinstance(phase, dict):
            continue
        pid = str(phase.get("id") or "")
        if not pid:
            continue
        prev = base_by_id.get(pid, {})
        merged = {**prev, **phase}
        base_by_id[pid] = merged
        if pid not in order:
            order.append(pid)
    return [base_by_id[k] for k in order if k in base_by_id]


def publish_pipeline_panel(**kwargs: Any) -> dict:
    """Publica estado de pipeline (fase 0 cache / fase 1 train) para canvas live."""
    ensure_gate_live_canvas_exists()
    pipeline_path = _pipeline_live_path()
    base = _read_json_if_exists(pipeline_path)
    phases_base = base.get("phases")
    if not isinstance(phases_base, list) or not phases_base:
        phases_base = _default_pipeline_phases()
    phases_updates = kwargs.get("phases")
    if isinstance(phases_updates, list):
        phases = _merge_pipeline_phases(phases_base, phases_updates)
    else:
        phases = phases_base

    pipeline_phase = kwargs.get("pipeline_phase", base.get("pipeline_phase", 0))
    payload = {
        **base,
        **{k: v for k, v in kwargs.items() if k != "phases"},
        "canvas_version": CANVAS_VERSION,
        "pipeline_phase": int(pipeline_phase),
        "phases": phases,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(pipeline_path, payload)
    # El canvas NO se toca aqui: el watcher 2 Hz lo refleja desde disco.
    return payload


def _mirror_canvas_live_feed(payload: dict) -> None:
    """Parchea snapshot en .tsx (datos vivos) + sidecar. Diff-skip si nada cambio."""
    ensure_gate_live_canvas()
    enriched = _enrich_canvas_payload(payload)
    view_model = build_canvas_view_model(enriched)
    feed_name = "gate-live-metrics-feed.json"
    for canvases in _canvas_paths():
        if not canvases.is_dir():
            continue
        canvas_path = canvases / "gate-am-live-metrics.canvas.tsx"
        try:
            _write_json(canvases / feed_name, view_model)
            _patch_canvas_live_stats(canvas_path, view_model)
            _patch_canvas_snapshot(canvas_path, view_model)
            _patch_canvas_data(canvas_path, view_model)
            return
        except OSError:
            continue


_CANVAS_SNAPSHOT_BEGIN = "// LIVE_METRICS_BEGIN"
_CANVAS_SNAPSHOT_END = "// LIVE_METRICS_END"
_CANVAS_TICK_BEGIN = "// LIVE_TICK_BEGIN"
_CANVAS_TICK_END = "// LIVE_TICK_END"
_CANVAS_STATS_BEGIN = "// LIVE_STATS_BEGIN"
_CANVAS_STATS_END = "// LIVE_STATS_END"
_CANVAS_DATA_KEY = "liveMetrics"
# Firmas separadas: tsx estable, tick, data.json
_LAST_PATCH_SIG: dict[str, str] = {}


def _canvas_signature(clean: dict) -> str:
    """Serializa el view-model ignorando timestamps volatiles."""
    import copy

    snap = copy.deepcopy(clean)
    if isinstance(snap.get("header"), dict):
        snap["header"]["synced_at"] = ""
    snap.pop("updated_at", None)
    return json.dumps(snap, ensure_ascii=False, sort_keys=True)


def _canvas_stable_signature(clean: dict) -> str:
    """Firma sin batch/loss intra-epoca (van en LIVE_TICK)."""
    import copy

    snap = copy.deepcopy(clean)
    snap.pop("train_stats", None)
    snap.pop("eta_stats", None)
    if isinstance(snap.get("header"), dict):
        snap["header"]["synced_at"] = ""
    snap.pop("updated_at", None)
    return json.dumps(snap, ensure_ascii=False, sort_keys=True)


def _extract_live_stats(view_model: dict) -> dict:
    header = view_model.get("header") if isinstance(view_model.get("header"), dict) else {}
    train = view_model.get("train_stats") if isinstance(view_model.get("train_stats"), list) else []
    eta = view_model.get("eta_stats") if isinstance(view_model.get("eta_stats"), list) else []
    return {
        "train": train,
        "eta": eta,
        "synced_at": str(header.get("synced_at") or ""),
    }


def _canvas_data_path(canvas_path: Path) -> Path:
    """gate-am-live-metrics.canvas.tsx -> gate-am-live-metrics.canvas.data.json"""
    return canvas_path.with_name(canvas_path.stem + ".data.json")


def _migrate_canvas_scroll_keys(existing: dict) -> None:
    """Unifica claves de scroll legacy en hostScrollTop."""
    if "hostScrollTop" in existing:
        return
    for legacy in ("panelScrollTop", "scrollTop"):
        val = existing.get(legacy)
        if isinstance(val, (int, float)) and val > 0:
            existing["hostScrollTop"] = val
            return


def _patch_canvas_data(canvas_path: Path, view_model: dict) -> None:
    """Copia de respaldo en sidecar (preserva hostScrollTop del usuario)."""
    if not canvas_path.is_file():
        return
    data_path = _canvas_data_path(canvas_path)
    sig = _canvas_signature(view_model)
    key = f"data:{data_path}"
    if _LAST_PATCH_SIG.get(key) == sig:
        return

    existing = _read_json_if_exists(data_path)
    if not isinstance(existing, dict):
        existing = {}
    _migrate_canvas_scroll_keys(existing)
    existing[_CANVAS_DATA_KEY] = view_model
    _write_json(data_path, existing)
    _LAST_PATCH_SIG[key] = sig


def _patch_canvas_live_stats(canvas_path: Path, view_model: dict) -> None:
    """Parchea batch/loss/ETA en bloque pequeno (~2 Hz) sin tocar graficos."""
    if not canvas_path.is_file():
        return
    stats = _extract_live_stats(view_model)
    sig = json.dumps(stats, ensure_ascii=False, sort_keys=True)
    key = f"stats:{canvas_path}"
    if _LAST_PATCH_SIG.get(key) == sig:
        return

    text = canvas_path.read_text(encoding="utf-8")
    if _CANVAS_STATS_BEGIN not in text or _CANVAS_STATS_END not in text:
        return
    body = json.dumps(stats, ensure_ascii=False, indent=2)
    block = (
        f"{_CANVAS_STATS_BEGIN}\n"
        f"const LIVE_STATS: LiveStats | null = {body};\n"
        f"{_CANVAS_STATS_END}"
    )
    pre, rest = text.split(_CANVAS_STATS_BEGIN, 1)
    _, post = rest.split(_CANVAS_STATS_END, 1)
    canvas_path.write_text(pre + block + post, encoding="utf-8")
    _LAST_PATCH_SIG[key] = sig


def _patch_canvas_snapshot(canvas_path: Path, payload: dict) -> None:
    """Parchea .tsx solo al cerrar epoca (cambian graficos/tablas). Intra-epoca: sin tocar tsx."""
    if not canvas_path.is_file():
        return
    clean = _sanitize_json_nan(payload)
    stable_sig = _canvas_stable_signature(clean)
    key = f"stable:{canvas_path}"
    if _LAST_PATCH_SIG.get(key) == stable_sig:
        return

    text = canvas_path.read_text(encoding="utf-8")
    if _CANVAS_SNAPSHOT_BEGIN not in text or _CANVAS_SNAPSHOT_END not in text:
        return
    synced_at = str(clean.get("header", {}).get("synced_at") or datetime.now().isoformat(timespec="seconds"))
    body = json.dumps(clean, ensure_ascii=False)
    metrics_block = (
        f"{_CANVAS_SNAPSHOT_BEGIN}\n"
        f"const LIVE_METRICS_SNAPSHOT: ViewModel | null = {body} as ViewModel;\n"
        f'const LIVE_METRICS_SYNCED_AT = "{synced_at}";\n'
        f"{_CANVAS_SNAPSHOT_END}"
    )
    pre, rest = text.split(_CANVAS_SNAPSHOT_BEGIN, 1)
    _, post = rest.split(_CANVAS_SNAPSHOT_END, 1)
    merged = pre + metrics_block + post

    if _CANVAS_STATS_BEGIN in merged and _CANVAS_STATS_END in merged:
        stats = _extract_live_stats(clean)
        stats_body = json.dumps(stats, ensure_ascii=False, indent=2)
        stats_block = (
            f"{_CANVAS_STATS_BEGIN}\n"
            f"const LIVE_STATS: LiveStats | null = {stats_body};\n"
            f"{_CANVAS_STATS_END}"
        )
        spre, srest = merged.split(_CANVAS_STATS_BEGIN, 1)
        _, spost = srest.split(_CANVAS_STATS_END, 1)
        merged = spre + stats_block + spost

    canvas_path.write_text(merged, encoding="utf-8")
    _LAST_PATCH_SIG[key] = stable_sig
    _LAST_PATCH_SIG[f"tsx:{canvas_path}"] = _canvas_signature(clean)


@dataclass
class GateRunLivePublisher:
    """Escribe métricas, curvas y checkpoint en la carpeta de corrida mientras entrena."""

    run: RunOutputs
    layout: GateRunLayout
    train_config: dict[str, Any]
    info_split: dict
    protocol: Any
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    ckpt_dir: Path
    embed_store: Any = None
    gate4_config: Any = None
    device: Any = None
    skip_pretrain_viz: bool = False
    canvas_path: Optional[Path] = None
    _last_snapshot_epoch: int = 0
    _last_live: Optional[dict] = None

    @classmethod
    def create(
        cls,
        *,
        run: RunOutputs,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        info_split: dict,
        train_config: dict[str, Any],
        protocol: Any,
        ckpt_dir: Path,
        embed_store: Any = None,
        gate4_config: Any = None,
        device: Any = None,
        skip_pretrain_viz: bool = False,
    ) -> GateRunLivePublisher:
        canvas_path = ensure_gate_live_canvas()
        return cls(
            run=run,
            layout=layout_for_run(run),
            train_config=train_config,
            info_split=info_split,
            protocol=protocol,
            train_df=train_df,
            val_df=val_df,
            ckpt_dir=ckpt_dir,
            embed_store=embed_store,
            gate4_config=gate4_config,
            device=device,
            skip_pretrain_viz=skip_pretrain_viz,
            canvas_path=canvas_path,
        )

    def publish_progress(self, progress: dict) -> None:
        """Heartbeat batch/epoca -> JSON (el canvas lo refleja un watcher externo)."""
        base = _read_json_if_exists(_pipeline_live_path())
        if not base:
            base = dict(self._last_live or {})
        # Evita que el historial de un run anterior contamine este run.
        if base.get("run_id") and base.get("run_id") != self.run.run_id:
            for _k in _HIST_METRIC_KEYS:
                base.pop(_k, None)
        ep = progress.get("epoch_current")
        batch = progress.get("batch_current")
        total = progress.get("batches_train_this_epoch")
        epochs_total = progress.get("epochs_total")
        step3_pct = 0
        if ep and batch and total and epochs_total:
            step3_pct = round(
                ((int(ep) - 1) + int(batch) / int(total)) / int(epochs_total) * 100,
                2,
            )
        step3_detail = progress.get("detail")
        if not step3_detail and ep and batch and total:
            loss = progress.get("train_loss_avg")
            loss_s = f" loss={loss:.4f}" if isinstance(loss, (int, float)) else ""
            step3_detail = f"Ep {ep}/{epochs_total} batch {batch}/{total}{loss_s}"
        phases_base = base.get("phases")
        if not isinstance(phases_base, list) or not phases_base:
            phases_base = _default_pipeline_phases()
        phases = _merge_pipeline_phases(
            phases_base,
            [
                {
                    "id": "train_slice_ms",
                    "label": "Paso 3: Entrenamiento Slice-MS",
                    "status": "running",
                    "pct": step3_pct,
                    "detail": step3_detail or "Entrenando",
                }
            ],
        )
        metrics_hist = _read_json_if_exists(self.run.reports / "live_metrics.json") or {}
        live = {
            **base,
            "run_id": self.run.run_id,
            "pipeline_phase": 1,
            "phases": phases,
            "checkpoint_metric": progress.get("checkpoint_metric")
            or self.train_config.get("checkpoint_metric"),
            "status": progress.get("status", "running"),
            "epochs_total": epochs_total,
            "epoch_current": ep,
            "batch_current": batch,
            "batches_train_this_epoch": total,
            "batches_test_this_epoch": progress.get("batches_test_this_epoch"),
            "train_loss_avg": progress.get("train_loss_avg"),
            "eta_epoch": progress.get("eta_epoch"),
            "eta_total": progress.get("eta_total"),
            "best_epoch": metrics_hist.get("best_epoch"),
            "best_checkpoint_score": metrics_hist.get("best_checkpoint_score")
            or metrics_hist.get("best_score"),
            "best_score": metrics_hist.get("best_score"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "recall_targets": dict(self.protocol.recall_thresh),
            "spec_targets": dict(getattr(self.protocol, "spec_thresh", {})),
        }
        for k in _HIST_METRIC_KEYS:
            if k in metrics_hist:
                live[k] = metrics_hist[k]
        prev = self._last_live or {}
        if progress.get("tiles_train_per_epoch"):
            live["tiles_train_per_epoch"] = progress.get("tiles_train_per_epoch")
        elif prev.get("tiles_train_per_epoch"):
            live["tiles_train_per_epoch"] = prev.get("tiles_train_per_epoch")
        self._last_live = live
        _write_json(self.run.reports / "live_progress.json", live)
        _write_json(self.ckpt_dir / "live_progress.json", live)
        _write_json(_pipeline_live_path(), {**base, **live, "canvas_version": CANVAS_VERSION})
        if ep and batch and total and (batch == 1 or batch % 20 == 0 or batch >= total):
            loss = progress.get("train_loss_avg")
            loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "?"
            pct = round(100 * int(batch) / int(total)) if total else 0
            eta_ep = progress.get("eta_epoch") or "?"
            eta_tot = progress.get("eta_total") or "?"
            tiles_ep = progress.get("tiles_train_per_epoch")
            elapsed = progress.get("elapsed_epoch_s")
            tps_s = ""
            if tiles_ep and elapsed and elapsed > 0:
                done = tiles_ep * int(batch) / int(total)
                tps_s = f" | {round(done / elapsed)} tiles/s"
            print(
                f"[Gate train] ep {ep}/{progress.get('epochs_total', '?')} | "
                f"batch {batch}/{total} ({pct}%) | loss={loss_s}{tps_s} | "
                f"ETA_ep {eta_ep} | ETA_tot {eta_tot}",
                flush=True,
            )

    def publish_startup(
        self,
        *,
        stratified_sample_df: Optional[pd.DataFrame] = None,
        skip_viz: Optional[bool] = None,
    ) -> None:
        """Pre-training: distribución, config y estado inicial."""
        if skip_viz if skip_viz is not None else self.skip_pretrain_viz:
            self._publish_startup_minimal()
            return
        import time

        from .gate_pretrain_viz import (
            plot_tile_samples_annotated,
            plot_tile_samples_by_class,
            plot_pretrain_dimension_panels,
            write_pre_training_index,
        )

        out = self.layout.pre_training
        n_samples = int(self.train_config.get("pretrain_tile_samples", 12))
        sample_df = stratified_sample_df if stratified_sample_df is not None else self.train_df
        steps_total = 8

        log.info(
            f"[Gate AM] Pre-training -> {out} "
            f"({steps_total} pasos; lectura JPEG + PNG, puede tardar unos minutos)"
        )

        def step(n: int, msg: str) -> None:
            log.info(f"[Gate AM] Pre-training ({n}/{steps_total}): {msg}")

        t0 = time.perf_counter()
        step(1, "distribución de clases train/holdout")
        plot_class_distribution(
            self.train_df,
            self.val_df,
            out / "class_distribution.png",
            out / "class_distribution.csv",
        )
        step(2, "balance train vs holdout")
        plot_train_val_balance(self.info_split, out / "train_val_balance.png")
        step(3, f"tiles anotados sampler ({n_samples}/clase, JPEG)")
        plot_tile_samples_annotated(
            sample_df,
            out / "stratified_tile_samples_annotated.png",
            split="train",
            etapa="pre_train",
            n_per_class=n_samples,
            title="Train/sampler — tiles anotados",
            manifest_csv=out / "tile_manifest_stratified.csv",
        )
        step(4, f"tiles anotados holdout ({n_samples}/clase, JPEG)")
        plot_tile_samples_annotated(
            self.val_df,
            out / "holdout_tile_samples_annotated.png",
            split="holdout",
            etapa="pre_train",
            n_per_class=n_samples,
            title="Holdout natural — tiles anotados",
            manifest_csv=out / "tile_manifest_holdout.csv",
        )
        step(5, "muestra tiles gold (train/sampler)")
        plot_tile_samples_by_class(
            sample_df,
            out / "stratified_tile_samples.png",
            n_per_class=n_samples,
            title="Muestra tiles gold (train/sampler)",
        )
        step(6, "muestra tiles gold (holdout)")
        plot_tile_samples_by_class(
            self.val_df,
            out / "holdout_tile_samples.png",
            n_per_class=n_samples,
            title="Muestra tiles gold (holdout natural)",
        )
        step(7, "paneles split/sampler/holdout + índice")
        write_pre_training_index(out)
        plot_pretrain_dimension_panels(
            self.train_df,
            self.val_df,
            out,
            stratified_batch_df=stratified_sample_df,
            n_per=n_samples // 2 or 6,
        )
        step(8, "config.json, dataset_stats, training_status")
        cfg = dict(self.train_config)
        cfg["class_names"] = list(GATE_CLASS_NAMES)
        cfg["run_id"] = self.run.run_id
        _write_json(self.layout.config_dir / "config.json", cfg)
        _write_json(
            out / "dataset_stats.json",
            {
                "split_info": self.info_split,
                "train_stage1": self.train_df["stage1"].value_counts().to_dict()
                if "stage1" in self.train_df.columns
                else {},
                "val_stage1": self.val_df["stage1"].value_counts().to_dict()
                if "stage1" in self.val_df.columns
                else {},
            },
        )
        _write_json(
            self.run.root / "training_status.json",
            {
                "status": "running",
                "run_id": self.run.run_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "epochs_total": self.train_config.get("epochs_max"),
            },
        )
        log.info(
            f"[Gate AM] Pre-training listo en {time.perf_counter() - t0:.1f}s -> {out}"
        )

    def _publish_startup_minimal(self) -> None:
        """Solo metadatos de corrida (sin leer JPEG/PNG). Para reentrenamientos iterativos."""
        import time

        t0 = time.perf_counter()
        out = self.layout.pre_training
        out.mkdir(parents=True, exist_ok=True)
        cfg = dict(self.train_config)
        cfg["class_names"] = list(GATE_CLASS_NAMES)
        cfg["run_id"] = self.run.run_id
        cfg["pretrain_viz_skipped"] = True
        _write_json(self.layout.config_dir / "config.json", cfg)
        _write_json(
            out / "dataset_stats.json",
            {
                "split_info": self.info_split,
                "pretrain_viz_skipped": True,
                "train_stage1": self.train_df["stage1"].value_counts().to_dict()
                if "stage1" in self.train_df.columns
                else {},
                "val_stage1": self.val_df["stage1"].value_counts().to_dict()
                if "stage1" in self.val_df.columns
                else {},
            },
        )
        _write_json(
            self.run.root / "training_status.json",
            {
                "status": "running",
                "run_id": self.run.run_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "epochs_total": self.train_config.get("epochs_max"),
                "pretrain_viz_skipped": True,
            },
        )
        log.info(
            f"[Gate AM] Pre-training omitido (GATE_SKIP_PRETRAIN_VIZ) en "
            f"{time.perf_counter() - t0:.1f}s -> config + dataset_stats"
        )

    def publish_baseline(self, pretrain: dict) -> None:
        _write_json(self.layout.pre_training / "baseline_metrics.json", pretrain)

    def publish_eval_phase(self, phase: str, epoch: int, epochs_total: int) -> None:
        """Heartbeat: indica que el proceso sigue vivo durante eval larga."""
        _write_json(
            self.run.root / "training_status.json",
            {
                "status": "evaluating",
                "phase": phase,
                "run_id": self.run.run_id,
                "epoch_current": epoch,
                "epochs_total": epochs_total,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def publish_epoch(
        self,
        *,
        epoch: int,
        epochs_total: int,
        history: GPUTrainHistory,
        epoch_details: list[dict],
        pretrain: Optional[dict],
        ckpt_metric: str,
        improved: bool,
        last_val: dict,
        last_val_ckpt: Optional[dict] = None,
        last_val_strat: Optional[dict] = None,
    ) -> None:
        """Snapshot tras cada época: métricas, curvas, progreso y mejor checkpoint."""
        hist = history.to_dict()
        hist["checkpoint_metric"] = ckpt_metric
        hist["epoch_details"] = epoch_details
        hist["pretrain_baseline"] = pretrain

        base = _read_json_if_exists(_pipeline_live_path())
        if not base:
            base = dict(self._last_live or {})
        if base.get("run_id") and base.get("run_id") != self.run.run_id:
            for _k in _HIST_METRIC_KEYS:
                base.pop(_k, None)
        live = {
            **base,
            "pipeline_phase": int(base.get("pipeline_phase", 1) or 1),
            "checkpoint_metric": ckpt_metric,
            "best_epoch": history.best_epoch,
            "best_score": history.best_val_auroc,
            "epochs_done": history.epochs,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_macro_f1": history.val_f1,
            "val_acc": history.val_acc,
            "val_min_class_recall": [
                float(d["min_class_recall"])
                for d in epoch_details
                if d.get("min_class_recall") is not None
            ],
            "g1_pass_by_epoch": [bool(d.get("evangelisti_g1_pass")) for d in epoch_details],
            "recall_targets": dict(self.protocol.recall_thresh),
            "spec_targets": dict(getattr(self.protocol, "spec_thresh", {})),
            "pretrain_baseline": pretrain,
            "epoch_details": epoch_details,
            "last_epoch": epoch,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if epoch_details:
            last = epoch_details[-1]
            live["last_per_class_recall"] = last.get("per_class_recall")
            live["last_per_class_spec"] = last.get("per_class_specificity")
            live["last_g1_pass"] = bool(last.get("evangelisti_g1_pass"))
            live["last_min_class_recall"] = last.get("min_class_recall")
        _write_json(self.run.reports / "live_metrics.json", live)
        _write_json(self.layout.logs / "live_metrics.json", live)
        self._last_live = live
        # El canvas NO se toca aqui: el watcher 2 Hz lo refleja desde disco.

        metrics_csv = self.run.root / "training_metrics.csv"
        curves_png = self.layout.post_training / "loss_curves.png"
        plot_loss_curves(hist, curves_png, metrics_csv)
        from .gate_run_layout import plot_recall_curves

        plot_recall_curves(
            epoch_details,
            self.layout.post_training / "recall_curves.png",
            self.layout.post_training / "recall_curves.csv",
        )
        val_curves = self.layout.post_val / "loss_curves.png"
        val_curves.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(curves_png, val_curves)
        shutil.copy2(metrics_csv, self.layout.post_val / "loss_curves.csv")

        if epoch_details:
            pd.DataFrame(epoch_details).to_csv(
                self.run.tables / "epoch_metrics.csv", index=False
            )
            pd.DataFrame(epoch_details).to_csv(
                self.run.root / "training_metrics_epochs.csv", index=False
            )

        val_summary = {
            "updated_at": live["updated_at"],
            "epoch_current": epoch,
            "epochs_total": epochs_total,
            "checkpoint_metric": ckpt_metric,
            "checkpoint_eval": self.protocol.checkpoint_eval,
            "eval_balance_mode": self.protocol.eval_balance_mode,
            "best_epoch": history.best_epoch,
            "best_score": history.best_val_auroc,
            "last_holdout": {
                "acc": last_val.get("acc"),
                "macro_f1": last_val.get("macro_f1"),
                "balanced_accuracy": last_val.get("balanced_accuracy"),
                "min_class_recall": last_val.get("min_class_recall"),
                "evangelisti_g1_pass": last_val.get("evangelisti_g1_pass"),
                "g1_status": format_g1_status(last_val, self.protocol),
                "per_class_recall": last_val.get("per_class_recall"),
                "per_class_specificity": last_val.get("per_class_specificity"),
                "diagnostics": last_val.get("diagnostics"),
                "per_image_bg": last_val.get("per_image_bg"),
            },
        }
        ckpt = last_val_ckpt or last_val
        val_summary["last_checkpoint"] = {
            "acc": ckpt.get("acc"),
            "macro_f1": ckpt.get("macro_f1"),
            "min_class_recall": ckpt.get("min_class_recall"),
            "per_class_recall": ckpt.get("per_class_recall"),
            "diagnostics": ckpt.get("diagnostics"),
            "per_image_bg": ckpt.get("per_image_bg"),
        }
        if last_val_strat is not None:
            val_summary["last_val_stratified"] = {
                "acc": last_val_strat.get("acc"),
                "macro_f1": last_val_strat.get("macro_f1"),
                "min_class_recall": last_val_strat.get("min_class_recall"),
                "per_class_recall": last_val_strat.get("per_class_recall"),
            }
        _write_json(self.run.root / "evaluation_metrics_summary_val.json", val_summary)

        progress_src = self.ckpt_dir / PROGRESS_NAME
        if progress_src.exists():
            dst = self.run.reports / PROGRESS_NAME
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(progress_src, dst)
            shutil.copy2(progress_src, self.layout.logs / PROGRESS_NAME)

        live_png = self.ckpt_dir / "live_curves.png"
        if live_png.exists():
            shutil.copy2(live_png, self.layout.post_val / "live_curves.png")
            shutil.copy2(live_png, self.run.maps / "training_live_curves.png")

        if improved:
            ckpt_src = self.ckpt_dir / CHECKPOINT_NAME
            if ckpt_src.exists():
                dst_ckpt = self.layout.checkpoints / CHECKPOINT_NAME
                shutil.copy2(ckpt_src, dst_ckpt)

        from .gate_checkpoint_snapshot import write_run_status_md

        write_run_status_md(
            self.run.root,
            status="running" if epoch < epochs_total else "epoch_done",
            epoch_current=epoch,
            epochs_total=epochs_total,
            best_epoch=history.best_epoch,
            best_score=float(history.best_val_auroc or 0),
            protocol=self.protocol,
            last_holdout=val_summary.get("last_holdout"),
            post_training_ready=False,
        )

        _write_json(
            self.run.root / "training_status.json",
            {
                "status": "running" if epoch < epochs_total else "epoch_done",
                "run_id": self.run.run_id,
                "epoch_current": epoch,
                "epochs_total": epochs_total,
                "best_epoch": history.best_epoch,
                "best_score": history.best_val_auroc,
                "updated_at": live["updated_at"],
            },
        )
        log.info(
            f"[Gate AM] Ep {epoch}/{epochs_total} guardado -> "
            f"{self.run.root} (best ep{history.best_epoch})"
        )

    def publish_completed(self, *, history: GPUTrainHistory, ckpt_metric: str) -> None:
        _write_json(
            self.run.root / "training_status.json",
            {
                "status": "completed",
                "run_id": self.run.run_id,
                "best_epoch": history.best_epoch,
                "best_score": history.best_val_auroc,
                "checkpoint_metric": ckpt_metric,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        ckpt_src = self.ckpt_dir / CHECKPOINT_NAME
        if ckpt_src.exists():
            shutil.copy2(ckpt_src, self.layout.checkpoints / CHECKPOINT_NAME)
        if self.embed_store is not None and self.device is not None:
            from .gate_checkpoint_snapshot import publish_eval_snapshot, write_run_status_md

            publish_eval_snapshot(
                run_root=self.run.root,
                val_df=self.val_df,
                embed_store=self.embed_store,
                checkpoint=ckpt_src,
                device=self.device,
                protocol=self.protocol,
                gate4_config=self.gate4_config,
                best_epoch=history.best_epoch,
                best_score=float(history.best_val_auroc or 0),
            )
            write_run_status_md(
                self.run.root,
                status="completed",
                epoch_current=history.epochs[-1] if history.epochs else 0,
                epochs_total=int(self.train_config.get("epochs_max", 0)),
                best_epoch=history.best_epoch,
                best_score=float(history.best_val_auroc or 0),
                protocol=self.protocol,
                post_training_ready=True,
            )
