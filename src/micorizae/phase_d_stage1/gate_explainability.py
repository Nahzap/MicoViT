"""Artefactos de explicabilidad post-entrenamiento gate AM."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from ..common.logging_utils import get_logger
from ..common.run_outputs import RunOutputs
from .gate_classes import GATE_CLASS_NAMES

log = get_logger("phase_d.gate_explain")


def resolve_vis_image_paths(
    val_image_paths: list[Path],
    *,
    render_maps: bool,
    vis_all_test: bool,
    max_vis_images: Optional[int],
) -> list[Path]:
    if not render_maps:
        return []
    paths = list(val_image_paths)
    if vis_all_test:
        if max_vis_images is not None and max_vis_images > 0:
            return paths[:max_vis_images]
        return paths
    n = max(1, max_vis_images or 1)
    return paths[:n]


def copy_checkpoint_live_artifacts(ckpt_dir: Path, run: RunOutputs) -> list[str]:
    """Copia curvas/metricas en vivo al folder de la corrida."""
    copied: list[str] = []
    for name in ("live_curves.png", "live_metrics.json", "training_progress.json"):
        src = ckpt_dir / name
        if not src.exists():
            continue
        dest_dir = run.maps if name.endswith(".png") else run.reports
        dest = dest_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(dest.relative_to(run.root)))
    return copied


def plot_class_specificity_bars(per_class_spec: dict[str, float], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    names = list(per_class_spec.keys())
    vals = [per_class_spec.get(n, 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#666666", "#D2B48C", "#00C8FF"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Especificidad")
    ax.set_title("Especificidad por clase — test holdout")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_g1_combined_bars(
    recall: dict[str, float],
    specificity: dict[str, float],
    recall_thresh: dict[str, float],
    spec_thresh: dict[str, float],
    out_path: Path,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(GATE_CLASS_NAMES))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    rec_vals = [recall.get(c, 0) for c in GATE_CLASS_NAMES]
    spec_vals = [specificity.get(c, 0) for c in GATE_CLASS_NAMES]
    ax.bar(x - w, rec_vals, w, label="Sensibilidad", color="#4C78A8")
    ax.bar(x, spec_vals, w, label="Especificidad", color="#F58518")
    ax.bar(x + w, [recall_thresh.get(c, 0) for c in GATE_CLASS_NAMES], w, label="Umbral Sens", alpha=0.35, color="#4C78A8")
    ax.bar(x + 2 * w, [spec_thresh.get(c, 0) for c in GATE_CLASS_NAMES], w, label="Umbral Spec", alpha=0.35, color="#F58518")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(GATE_CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_title("Evangelisti G1 — Sensibilidad vs Especificidad")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_map_gallery_md(map_manifest: list[dict], out_path: Path, run: RunOutputs) -> Path:
    from ..layers.composer import STAGE1_COLORS

    lines = [
        "# Galería test holdout — gold vs predicción",
        "",
        "Mapas generados **desde predicciones en cache** (sin re-ejecutar DINO por tile).",
        "Un decode JPEG por imagen + composición de capas L0/L1/L2.",
        "",
        "## Leyenda gate4",
        "",
        "| Clase | Color |",
        "|-------|-------|",
    ]
    for name in GATE_CLASS_NAMES:
        vis = "Unreadable" if name == "Unknown" else name
        rgb = STAGE1_COLORS.get(vis, STAGE1_COLORS.get(name, (0, 0, 0)))
        lines.append(f"| {name} | RGB{rgb} |")
    lines += [
        "",
        "**Panel recomendado:** `*__L0_gold_pred_audit.png` = foto | gold | pred.",
        "",
    ]
    for entry in map_manifest:
        rel = entry["image"]
        stem = Path(rel).stem
        lines += [f"## `{Path(rel).name}`", ""]
        maps = entry.get("maps", {})
        for key in maps:
            lines.append(f"- **{key}:** `images/post_training/test/fullimage/{maps[key]}`")
            lines.append(f"- **{key} (legacy):** `maps/test_images/{maps[key]}`")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_explainability_md(
    *,
    run: RunOutputs,
    metrics_summary: dict[str, Any],
    history: dict,
    train_config: dict,
    map_manifest: list[dict],
    live_artifacts: list[str],
    protocol: Any,
) -> Path:
    th = metrics_summary.get("test_holdout", {})
    pb = history.get("pretrain_baseline") or {}
    best_ep = history.get("best_epoch", "?")
    ckpt_metric = history.get("checkpoint_metric") or train_config.get("checkpoint_metric", "macro_f1")
    best_score = history.get("best_val_auroc", 0)
    elapsed = history.get("elapsed_s") or []
    train_s = float(sum(elapsed)) if elapsed else 0.0
    n_epochs = len(elapsed)
    embed_cache = train_config.get("embed_cache", True)

    lines = [
        "# Explicabilidad — corrida gate AM",
        "",
        f"**Run:** `{run.run_id}`",
        "",
        "## Qué se hizo",
        "",
        "1. **Cache embeddings** (DINO mean-pool precomputado) → entrenamiento rápido del probe Slice-MS.",
        "2. **Linear probe** sobre embeddings; evaluación test desde memmap (~segundos).",
        "3. **Mapas visuales:** predicciones ya calculadas + 1 lectura JPEG/imagen (CPU).",
        "",
        "## Duración del entrenamiento",
        "",
        f"- Épocas ejecutadas: **{n_epochs}** (early stop posible antes de `{train_config.get('epochs_max', '?')}`)",
        f"- Tiempo acumulado train+val: **{train_s:.1f} s** (~{train_s / max(n_epochs, 1):.1f} s/época)",
        "- Esto es **esperado** en modo `FULL_PROBE` + cache: solo entrena el probe Slice-MS",
        "  leyendo vectores DINO precomputados; **no** re-ejecuta DINO por tile.",
        "- Para entrenamiento más lento (minutos, backbone en GPU): `GATE_PROBE=False` en config.py.",
        "",
        "## Configuración",
        "",
        f"- Modo: `{train_config.get('mode', '?')}`",
        f"- Cache embeddings en train: `{embed_cache}`",
        f"- DINO input: `{train_config.get('dino_input_size', 252)}` px (tiles nativos AM)",
        f"- Balance train: `{train_config.get('balance_mode', '?')}`",
        f"- Checkpoint por: `{ckpt_metric}`",
        f"- Loss: `{train_config.get('loss_type', 'ce')}`",
        f"- Class weights train: `{train_config.get('use_class_weights_train', False)}`",
        "",
        "## Baseline vs mejor modelo",
        "",
        f"| Etapa | {ckpt_metric} (checkpoint) | Balanced acc |",
        "|-------|----------|--------------|",
        f"| Ep 0 (sin entrenar) | {pb.get('macro_f1', 0):.4f} | {pb.get('balanced_accuracy', 0):.4f} |",
        f"| Mejor ep {best_ep} | {best_score:.4f} | {th.get('balanced_accuracy', 0):.4f} |",
        f"| Eval final holdout | {th.get('macro_f1', 0):.4f} | {th.get('balanced_accuracy', 0):.4f} |",
        "",
        "## Interpretación rápida",
        "",
    ]

    if th:
        from .gate_training_protocol import g1_metric_display

        g1 = "CUMPLE" if th.get("evangelisti_g1_pass") else "NO CUMPLE"
        lines += [
            f"- Evangelisti G1: **{g1}**",
            f"- Accuracy global **{th.get('acc', 0):.1%}** — alta porque ~83% del test es Background;",
            "  no usar acc como métrica principal en holdout desbalanceado.",
            f"- **Macro F1** y **balanced accuracy** son las métricas de selección de checkpoint.",
            "",
            "### Por clase (test holdout)",
            "",
            "| Clase | Sens | Spec | Umbral Sens | Umbral Spec |",
            "|-------|------|------|-------------|-------------|",
        ]
        for cls in GATE_CLASS_NAMES:
            rec = th.get("per_class_recall", {}).get(cls, float("nan"))
            spec = th.get("per_class_specificity", {}).get(cls, float("nan"))
            rt = protocol.recall_thresh.get(cls, 0)
            st = protocol.spec_thresh.get(cls, 0)
            rec_txt, _ = g1_metric_display(rec, rt)
            spec_txt, _ = g1_metric_display(spec, st)
            lines.append(f"| {cls} | {rec_txt} | {spec_txt} | {rt:.2f} | {st:.2f} |")

    lines += [
        "",
        "## Layout pre/post entrenamiento",
        "",
        "| Ruta | Contenido |",
        "|------|-----------|",
        "| `images/pre_training/class_distribution.png` | Distribución clases train vs holdout |",
        "| `images/pre_training/baseline_metrics.json` | Métricas ep0 (pre-train) |",
        "| `images/post_training/loss_curves.png` | Loss y F1 por época |",
        "| `images/post_training/val/` | Validación durante entrenamiento |",
        "| `images/post_training/test/fullimage/` | Mapas por imagen (foto, gold, pred) |",
        "| `training_protocol.md` | Hiperparámetros y split |",
        "| `training_report_test.md` | Informe evaluación final holdout |",
        "| `training_report_val.md` | Informe mejor época (val) |",
        "| `evaluation_metrics_summary_test.json` | Métricas JSON test |",
        "",
        "## Artefactos visuales (legacy maps/)",
        "",
        "| Archivo | Contenido |",
        "|---------|-----------|",
        "| `maps/train_curves_all_branches.png` | Loss / acc / F1 por época |",
        "| `maps/training_live_curves.png` | Curvas en vivo (copia checkpoint) |",
        "| `maps/confusion_matrix_test.png` | Matriz confusión test |",
        "| `maps/recall_by_class_test.png` | Sensibilidad por clase |",
        "| `maps/specificity_by_class_test.png` | Especificidad por clase |",
        "| `maps/g1_sens_spec_bars.png` | Panel G1 combinado |",
        "| `maps/test_images/*__L0_gold_pred_audit.png` | Foto | gold | pred |",
        "| `maps/test_images/*__L0_L2_errors.png` | Tiles mal clasificados (morado) |",
        "",
        "**Sobre mapas de error:** cada cuadrado morado = un tile 252×252 mal clasificado.",
        "No es segmentación pixel a pixel; el modelo decide por tile entero.",
        "Mucho morado en raíces = el modelo predice Background donde gold dice M−/M+.",
        "",
        f"**Imágenes test renderizadas:** {len(map_manifest)}",
        "",
        "Ver índice detallado: `reports/map_gallery.md` y `images/post_training/test/fullimage/INDEX.md`",
        "",
        "## Tablas CSV/Parquet",
        "",
        "- `tables/test_predictions_all_tiles` — probabilidades por tile",
        "- `tables/test_per_image_summary` — accuracy por imagen",
        "- `tables/epoch_metrics` — métricas por época",
        "",
        "## Auditoría espacial (inicio de corrida)",
        "",
        "- `reports/spatial_audit/spatial_audit.json`",
        "- `reports/spatial_audit/sample_overlays/` — 5 PNGs muestra",
        "",
    ]
    if live_artifacts:
        lines += ["## Métricas en vivo (durante entrenamiento)", ""]
        for a in live_artifacts:
            lines.append(f"- `{a}`")

    path = run.reports / "EXPLICABILIDAD.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"[Gate tile] Explicabilidad -> {path}")
    return path


def print_run_artifact_index(run: RunOutputs, explain_path: Path) -> None:
    """Resumen en consola al terminar la corrida."""
    root = run.root
    print("\n=== Resultados (explicabilidad) ===", flush=True)
    print(f"  Carpeta corrida: {root}", flush=True)
    print(f"  Guía legible:   {explain_path}", flush=True)
    print(f"  Reporte MD:     {root / 'reports' / 'run_report.md'}", flush=True)
    print(f"  Protocolo:      {root / 'training_protocol.md'}", flush=True)
    print(f"  Informe test:   {root / 'training_report_test.md'}", flush=True)
    print(f"  G1 validación:  Docs/AM_GATE_VALIDATION.md", flush=True)
    pre = root / "images" / "pre_training"
    post = root / "images" / "post_training"
    n_maps = len(list((post / "test" / "fullimage").glob("*.png"))) if (post / "test" / "fullimage").exists() else 0
    print(f"  Pre-train:      {pre}/ (distribución, baseline)", flush=True)
    print(f"  Post-train:     {post}/ (curvas, val, test)", flush=True)
    print(f"  Mapas test:     {n_maps} PNG en images/post_training/test/fullimage/", flush=True)
    print(f"  Curvas train:   {post / 'loss_curves.png'}", flush=True)
    print("", flush=True)
