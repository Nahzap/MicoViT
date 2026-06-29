"""CLI principal del proyecto.

Acceso soportado:
    python run.py --help
    python run.py ingest
    python run.py weakseg-gui
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from .common.logging_utils import get_logger
from .common.paths import get_paths
from .common.run_outputs import RunOutputs

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)
log = get_logger("cli")


@app.callback(invoke_without_command=True)
def _entrypoint_guard():
    """Fuerza el uso de `run.py` como único entrypoint operativo."""
    if os.environ.get("MICORIZAE_ENTRYPOINT") != "run.py":
        typer.echo("Entry point bloqueado: usa exclusivamente `python run.py ...`", err=True)
        raise typer.Exit(code=2)


def _load_user_config():
    """Carga `config.py` del root del proyecto si existe."""
    try:
        import config as user_config  # type: ignore

        return user_config
    except Exception:
        return None


def _cfg(name: str, default):
    cfg = _load_user_config()
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _cfg_path(name: str) -> Optional[Path]:
    v = _cfg(name, None)
    if v is None:
        return None
    return Path(v)


def _load_tiles_for_image(image_path: Path) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Carga el subset de tiles_index/manifest_labels que corresponde a `image_path`."""
    import pandas as pd

    from .common.io import read_table

    paths = get_paths()
    try:
        df = read_table(paths.manifests / "tiles_index")
    except FileNotFoundError:
        df = read_table(paths.manifests / "manifest_labels")
    rel = image_path.resolve().relative_to(paths.root).as_posix()
    sub = df[df["image_path"] == rel].copy()
    if sub.empty:
        log.warning(f"[yellow]No hay tiles para {rel} en el manifest[/yellow]")
    return sub


def _infer_tile_size_from_manifest(image_path: Path, fallback: int = 252) -> int:
    """Lee `tile_size` del tiles_index si está disponible; si no, usa fallback."""
    sub = _load_tiles_for_image(image_path)
    if sub.empty or "tile_size" not in sub.columns:
        return fallback
    return int(sub["tile_size"].iloc[0])


def _write_report(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _save_step_png(
    *,
    arr,
    out_path: Path,
    step_code: str,
    title: str,
    legend_items: list[tuple[str, tuple[int, int, int], str]] | None = None,
) -> Path:
    """Guarda PNG con cabecera + leyenda incrustada para lectura secuencial."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    base = np.asarray(arr).copy()
    h, w = base.shape[:2]
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    items = legend_items or []
    header_h = 28 + (16 * len(items))
    box_h = min(max(header_h, 28), max(28, int(h * 0.45)))
    draw.rectangle([(0, 0), (w - 1, box_h)], fill=(255, 255, 255))
    draw.text((8, 6), f"{step_code} | {title}", fill=(0, 0, 0), font=font)

    y = 24
    for label, color, desc in items:
        draw.rectangle([(8, y), (20, y + 10)], fill=tuple(color), outline=(0, 0, 0))
        draw.text((26, y - 1), f"{label}: {desc}", fill=(0, 0, 0), font=font)
        y += 14

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _save_stage_sequence(
    *,
    run: RunOutputs,
    image_stem: str,
    step_code: str,
    slug: str,
    arr,
    title: str,
    legend_items: list[tuple[str, tuple[int, int, int], str]] | None = None,
) -> Path:
    name = f"{step_code}_{image_stem}__{slug}.png"
    out = _save_step_png(
        arr=arr,
        out_path=run.maps / name,
        step_code=step_code,
        title=title,
        legend_items=legend_items,
    )
    log.info(f"[SEQ] {step_code} -> {out.name}")
    return out


def _write_stage1_run_report(
    *,
    run: RunOutputs,
    image: Path,
    tau_s1: float,
    n_pos: int,
    n_tot: int,
    seg_cov_pct: float,
) -> Path:
    p = run.reports / "run_report.md"
    lines = [
        f"# Run Report — {run.run_id}",
        "",
        "## Contexto",
        f"- Imagen: `{image}`",
        "- Fase: Stage1 (gate M+/M- + segmentación L4)",
        f"- Tau Stage1 (`tau_s1`): `{tau_s1}`",
        f"- Tiles predichos M+: `{n_pos}/{n_tot}` ({(100.0 * n_pos / max(n_tot, 1)):.2f}%)",
        f"- Cobertura segmentación L4: `{seg_cov_pct:.2f}%` de píxeles",
        "",
        "## Nomenclatura de capas",
        "- `L0`: imagen base RGB.",
        "- `L2`: máscara por tile del gate (M+, M-, Background, Unreadable).",
        "- `L3`: confianza continua `p_fused(M+)` por tile.",
        "- `L4`: segmentación U2Net (cian intenso = mayor probabilidad).",
        "- `L6`: consenso entre ramas A/B/C (verde alto, amarillo medio, rojo bajo).",
        "",
        "## Archivos clave",
        f"- `maps/{image.stem}__L0_L3.png`: confianza del gate (L3).",
        f"- `maps/{image.stem}__L0_L4.png`: segmentación U2Net (L4).",
        f"- `maps/{image.stem}__L0_L2_L4.png`: gate + segmentación.",
        f"- `maps/{image.stem}__L0_L3_L4_L6.png`: panel diagnóstico Stage1.",
        f"- `tables/{image.stem}__stage1_probs.csv`: probabilidades por tile.",
        f"- `tables/{image.stem}__stage1_segmentation_l4.npz`: mapas `seg_prob` y `seg_bin`.",
        "",
        "## Secuencia unitaria (recomendada para entender el flujo)",
        f"- `maps/01_{image.stem}__L0_input.png`",
        f"- `maps/02_{image.stem}__L2_gate_pred.png`",
        f"- `maps/03_{image.stem}__L3_confidence.png`",
        f"- `maps/04_{image.stem}__L4_segmentation.png`",
        f"- `maps/05_{image.stem}__L6_consensus.png`",
        f"- `maps/06_{image.stem}__L0_L2_L4_overlay.png`",
        f"- `maps/07_{image.stem}__L0_L3_L4_L6_diagnostic.png`",
        "",
        "## Lectura rápida recomendada",
        "- Si L4 se superpone a raíz y evita fondo, la segmentación está razonable.",
        "- Si L6 aparece muy uniforme, revisar diversidad/calibración de ramas.",
        "- Si casi todo sale M+ o M-, ajustar `tau_s1` y verificar checkpoints.",
    ]
    return _write_report(p, lines)


def _write_stage2_run_report(
    *,
    run: RunOutputs,
    image: Path,
    lineage: str,
    tau_s1: float,
    n_s2: int,
    n_tot: int,
    seg_cov_pct: float,
    stage2_counts: dict[str, int],
) -> Path:
    p = run.reports / "run_report.md"
    counts_str = ", ".join([f"{k}: {v}" for k, v in stage2_counts.items()]) if stage2_counts else "sin clases predichas"
    lines = [
        f"# Run Report — {run.run_id}",
        "",
        "## Contexto",
        f"- Imagen: `{image}`",
        f"- Linaje Stage2: `{lineage}`",
        "- Fase: Stage2 (subclases en tiles M+)",
        f"- Tau Stage1 (`tau_s1`): `{tau_s1}`",
        f"- Tiles con predicción Stage2: `{n_s2}/{n_tot}` ({(100.0 * n_s2 / max(n_tot, 1)):.2f}%)",
        f"- Cobertura segmentación L4: `{seg_cov_pct:.2f}%` de píxeles",
        f"- Distribución Stage2: {counts_str}",
        "",
        "## Nomenclatura de capas",
        "- `L4`: segmentación U2Net (estructura fina).",
        "- `L7`: clase Stage2 por tile (color por subclase).",
        "- `L8`: incertidumbre Stage2 (entropía; más brillante = más incierto).",
        "",
        "## Archivos clave",
        f"- `maps/{image.stem}__L0_L4.png`: segmentación base.",
        f"- `maps/{image.stem}__L0_L2_L4_L7.png`: gate + segmentación + clase Stage2.",
        f"- `maps/{image.stem}__L0_L4_L7_L8.png`: segmentación + clase + incertidumbre.",
        f"- `maps/{image.stem}__diagnostico_stage2.png`: panel diagnóstico completo.",
        f"- `tables/{image.stem}__stage2_probs.csv`: probabilidades y predicción Stage2 por tile.",
        f"- `tables/{image.stem}__stage2_segmentation_l4.npz`: mapas `seg_prob` y `seg_bin`.",
        "",
        "## Secuencia unitaria (recomendada para entender el flujo)",
        f"- `maps/01_{image.stem}__L0_input.png`",
        f"- `maps/02_{image.stem}__L2_gate_pred.png`",
        f"- `maps/03_{image.stem}__L4_segmentation.png`",
        f"- `maps/04_{image.stem}__L7_stage2_classes.png`",
        f"- `maps/05_{image.stem}__L8_stage2_uncertainty.png`",
        f"- `maps/06_{image.stem}__L0_L2_L4_L7_overlay.png`",
        f"- `maps/07_{image.stem}__L0_L4_L7_L8_overlay.png`",
        f"- `maps/08_{image.stem}__diagnostico_stage2.png`",
        "",
        "## Lectura rápida recomendada",
        "- L7 debe concentrarse en regiones de raíz, no en fondo.",
        "- L8 alto en zonas complejas es normal; L8 alto global sugiere modelo inmaduro.",
        "- Si L7 pinta fondo, revisar `tau_s1` y/o checkpoints Stage1/Stage2.",
    ]
    return _write_report(p, lines)


@app.command()
def ingest(
    data_root: Optional[Path] = typer.Option(None, help="Directorio raíz con las imágenes (default: ./Data)"),
    out: Optional[Path] = typer.Option(None, help="Directorio de manifests (default: ./manifests)"),
    schema_map: Optional[Path] = typer.Option(None, help="Ruta a schema_map.yaml"),
    datasets: Optional[Path] = typer.Option(None, help="Ruta a datasets.yaml"),
):
    """Fase A — Construye los manifests parquet desde Data/."""
    from .phase_a_ingest import build_manifests

    res = build_manifests(
        data_root=data_root,
        out_dir=out,
        datasets_config_path=datasets,
        schema_map_path=schema_map,
    )
    log.info(f"[bold green]Fase A OK[/bold green] — {res.report['n_images_ok']} imágenes ingestadas")


@app.command()
def layer(
    image: Path = typer.Option(..., help="Ruta a la imagen RGB de entrada"),
    layers: list[str] = typer.Option(["L0", "L1"], help="Lista ordenada de capas a componer"),
    alphas: Optional[list[float]] = typer.Option(
        None, help="Opacidades por capa. Si se omite usa 1.0 para la primera y 0.5 para el resto."
    ),
    tile_size: Optional[int] = typer.Option(
        None, help="Tamaño de tile en píxeles. Si se omite se infiere del manifest (recomendado)."
    ),
    downscale: int = typer.Option(
        4, help="Factor de downscale (>=1). Se aplica ANTES de componer para evitar MemoryError."
    ),
    out: Optional[Path] = typer.Option(None, help="Ruta de salida .png"),
):
    """Compone capas L0..L10 sobre una imagen y guarda el PNG resultante."""
    import numpy as np
    from PIL import Image

    from .layers import LayerContext, compose, downscale_context, save_png, render_layer

    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(image).convert("RGB"))

    needs_tiles = any(l.upper() in {"L2"} for l in layers)
    tiles = _load_tiles_for_image(image) if needs_tiles else None
    ts = tile_size or _infer_tile_size_from_manifest(image, fallback=252)

    ctx = LayerContext(image=img, tile_size=ts, tiles=tiles)
    ctx = downscale_context(ctx, downscale)

    if alphas is None:
        alphas = [1.0] + [0.5] * (len(layers) - 1)
    composed = compose(layers, ctx, alphas=alphas)

    paths = get_paths()
    run = RunOutputs.create("layers_preview")
    out = out or (run.maps / f"{image.stem}__{'_'.join(layers)}.png")
    save_png(composed, out)
    log.info(f"[bold green]Capas compuestas:[/bold green] {out} (run={run.run_id}, tile_size={ts}, downscale={downscale})")


@app.command(name="tiles-index")
def tiles_index_cmd(
    tile_size: Optional[int] = typer.Option(None, help="Override del tile_size (default: configs/datasets.yaml)"),
    manifests: Optional[Path] = typer.Option(None, help="Directorio de manifests"),
):
    """Fase B — Construye tiles_index con bbox absoluto a partir del manifest de labels."""
    from .phase_b_tiling import build_tiles_index

    path = build_tiles_index(manifests_dir=manifests, tile_size=tile_size)
    log.info(f"[bold green]Fase B OK[/bold green] -> {path}")


@app.command()
def gate(
    image: Path = typer.Option(..., help="Ruta a la imagen base"),
    alpha: float = typer.Option(0.55, help="Opacidad de la capa L2 sobre L0"),
    tile_size: Optional[int] = typer.Option(None, help="Override del tile_size; se infiere del manifest si se omite"),
    downscale: int = typer.Option(4, help="Downscale ANTES de componer (evita MemoryError en panorámicas)"),
    out: Optional[Path] = typer.Option(None, help="Ruta del PNG resultante"),
):
    """Compone L0 + L2 (gate sintético desde anotaciones) sobre una imagen."""
    import numpy as np
    from PIL import Image

    from .layers import LayerContext, compose, downscale_context, save_png, render_layer

    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(image).convert("RGB"))
    tiles = _load_tiles_for_image(image)
    ts = tile_size or _infer_tile_size_from_manifest(image, fallback=252)

    ctx = LayerContext(image=img, tile_size=ts, tiles=tiles)
    ctx = downscale_context(ctx, downscale)
    composed = compose(["L0", "L2"], ctx, alphas=[1.0, alpha])

    paths = get_paths()
    run = RunOutputs.create("gate_preview")
    out = out or (run.maps / f"{image.stem}__L0_L2.png")
    save_png(composed, out)
    log.info(f"[bold green]Gate compuesto:[/bold green] {out} (run={run.run_id}, tile_size={ts}, downscale={downscale})")


@app.command(name="tile-view")
def tile_view_cmd(
    image: Path = typer.Option(..., help="Ruta a la imagen"),
    row: int = typer.Option(..., help="Coordenada row del tile (manifest)"),
    col: int = typer.Option(..., help="Coordenada col del tile (manifest)"),
    tile_size: Optional[int] = typer.Option(None, help="Tamaño del tile; se infiere del manifest si se omite"),
    target_size: int = typer.Option(224, help="Tamaño de la vista normalizada"),
    out: Optional[Path] = typer.Option(None, help="PNG de salida"),
):
    """Fase C preview — Decode CUDA, crop CUDA, build_views CUDA, PNG final."""
    import torch as _torch

    from .phase_c_views.preview import preview_tile

    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. preview es GPU-only.")
    ts = tile_size or _infer_tile_size_from_manifest(image, fallback=252)
    paths = get_paths()
    run = RunOutputs.create("tile_view")
    out = out or (run.maps / f"{image.stem}__r{row}_c{col}__rgb_seg_freq.png")
    preview_tile(
        image_path=image,
        row=row,
        col=col,
        tile_size=ts,
        target_size=target_size,
        out_path=out,
        device="cuda",
    )
    log.info(f"[bold green]Tile view guardado:[/bold green] {out} (run={run.run_id}, tile_size={ts}, GPU)")


@app.command(name="inspect-labels")
def inspect_labels_cmd(
    image: Optional[Path] = typer.Option(None, help="Imagen a inspeccionar (default config.py::DEFAULT_IMAGE)"),
):
    """Representa etiquetas del manifest (ground truth) por capas, sin modelos."""
    import numpy as np
    import torch

    from .common.io import read_table, write_table
    from .layers import LayerContext, compose, downscale_context, render_layer, save_png
    from .phase_b_tiling.gpu_io import decode_jpeg_gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")

    image = image or _cfg_path("DEFAULT_IMAGE")
    if image is None:
        raise typer.BadParameter("Debes indicar --image o definir DEFAULT_IMAGE en config.py")

    paths = get_paths()
    run = RunOutputs.create("labels_inspect", suffix=image.stem)
    device = torch.device("cuda")

    df = read_table(paths.manifests / "tiles_index")
    rel = image.resolve().relative_to(paths.root).as_posix()
    sub = df[df["image_path"] == rel].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No hay tiles en manifest para {rel}")

    sub["stage1_pred"] = (sub["stage1"].astype(str) == "Mplus").astype(int)
    sub["p_fused"] = sub["stage1_pred"].astype(float)
    sub["consensus"] = 1.0
    write_table(sub, run.tables / f"{image.stem}__labels_manifest")

    ts = int(sub["tile_size"].iloc[0]) if "tile_size" in sub.columns else _infer_tile_size_from_manifest(image, fallback=252)
    gimg = decode_jpeg_gpu(image, device=device)
    img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
    del gimg
    torch.cuda.empty_cache()

    ctx = LayerContext(image=img_np, tile_size=ts, tiles=sub)
    ctx = downscale_context(ctx, 4)
    l0 = render_layer("L0", ctx)
    l2 = render_layer("L2", ctx)
    ov_l0_l2 = compose(["L0", "L2"], ctx, alphas=[1.0, 0.55])

    save_png(ov_l0_l2, run.maps / f"{image.stem}__L0_L2_labels.png")
    _save_stage_sequence(
        run=run,
        image_stem=image.stem,
        step_code="01",
        slug="L0_input",
        arr=l0,
        title="Entrada L0",
    )
    _save_stage_sequence(
        run=run,
        image_stem=image.stem,
        step_code="02",
        slug="L2_labels_manifest",
        arr=l2,
        title="Etiquetas Stage1 del manifest (ground truth)",
        legend_items=[
            ("Mplus", (0, 200, 255), "colonizado"),
            ("Mminus", (210, 180, 140), "no colonizado"),
            ("Background", (120, 120, 120), "fondo"),
            ("Unreadable", (180, 100, 200), "ilegible"),
        ],
    )
    _save_stage_sequence(
        run=run,
        image_stem=image.stem,
        step_code="03",
        slug="L0_L2_overlay",
        arr=ov_l0_l2,
        title="Overlay L0 + etiquetas Stage1",
    )

    n_pos = int((sub["stage1_pred"] == 1).sum())
    n_tot = int(len(sub))
    rep = _write_report(
        run.reports / "run_report.md",
        [
            f"# Label Inspection - {run.run_id}",
            "",
            "## Contexto",
            f"- Imagen: `{image}`",
            "- Fuente: `manifests/tiles_index`",
            "- Modo: representacion directa de etiquetas (sin inferencia modelo)",
            f"- Mplus (manifest): `{n_pos}/{n_tot}` ({100.0 * n_pos / max(n_tot, 1):.2f}%)",
            "",
            "## Archivos",
            f"- `maps/01_{image.stem}__L0_input.png`",
            f"- `maps/02_{image.stem}__L2_labels_manifest.png`",
            f"- `maps/03_{image.stem}__L0_L2_overlay.png`",
            f"- `tables/{image.stem}__labels_manifest.csv`",
        ],
    )
    log.info(f"[Inspect Labels] M+ manifest: {n_pos}/{n_tot}")
    log.info(f"[Inspect Labels] Reporte -> {rep}")
    log.info(f"[bold green]OK[/bold green] run={run.run_id} -> {run.root}")


@app.command(name="dataset-prepost-maps")
def dataset_prepost_maps_cmd(
    phase: str = typer.Option("both", help="pre|post|both"),
    lineage: Optional[str] = typer.Option(None, help="Filtrar linaje (AM/ERM)"),
    max_images: int = typer.Option(0, help="Limitar numero de imagenes (0=todas)"),
    downscale: int = typer.Option(4, help="Downscale visual para mapas finales"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size inferencia POST"),
    tau_s1: Optional[float] = typer.Option(None, help="Umbral Stage1 para POST"),
    backbone: Optional[str] = typer.Option(None, help="Backbone Branch A para POST"),
):
    """Genera mapas PRE (etiquetas) y POST (prediccion) por todo el dataset.

    PRE:
      - fuente: manifests/tiles_index (ground truth)
      - overlay principal: L0 + L2 + L7

    POST:
      - fuente: Branch A (DINOv2) entrenada
      - overlay principal: L0 + L2 + L3
    """
    import json
    import numpy as np
    import pandas as pd
    import torch

    from .common.io import read_table
    from .layers import LayerContext, compose, downscale_context, save_png, render_layer
    from .phase_b_tiling.gpu_io import decode_jpeg_gpu
    from .phase_d_stage1 import build_branch_a, iter_image_batches, plan_epoch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")

    phase = phase.lower().strip()
    if phase not in {"pre", "post", "both"}:
        raise typer.BadParameter("phase debe ser pre|post|both")

    paths = get_paths()
    run = RunOutputs.create("dataset_prepost_maps")
    device = torch.device("cuda")
    batch_size = int(batch_size if batch_size is not None else _cfg("DEFAULT_BATCH_SIZE", 32))
    tau_s1 = float(tau_s1 if tau_s1 is not None else _cfg("DEFAULT_TAU_S1", 0.65))
    backbone = backbone or str(_cfg("DEFAULT_BACKBONE", "dinov2_vits14"))

    pre_dir = run.root / "PRE"
    post_dir = run.root / "POST"
    for p in (pre_dir / "maps", pre_dir / "tables", pre_dir / "reports", post_dir / "maps", post_dir / "tables", post_dir / "reports"):
        p.mkdir(parents=True, exist_ok=True)

    tiles = read_table(paths.manifests / "tiles_index")
    images = read_table(paths.manifests / "manifest_images")
    images = images[images["status"] == "ok"].copy()
    if lineage is not None:
        lin = lineage.upper().strip()
        images = images[images["lineage"] == lin].copy()
    if images.empty:
        raise ValueError("No hay imagenes validas para generar mapas.")
    if max_images and max_images > 0:
        images = images.head(max_images).copy()

    # POST: prepara Branch A y temperatura si corresponde.
    branch_a = None
    temp_a = 1.0
    if phase in {"post", "both"}:
        ckpt_dir = paths.root / "models" / "checkpoints" / "gate_am"
        ckpt_a = ckpt_dir / "gate_branch_a_best.pt"
        if not ckpt_a.exists():
            raise FileNotFoundError(
                f"No existe checkpoint del gate AM: {ckpt_a}. "
                "Entrena primero con: python run.py train-gate-am"
            )
        branch_a = build_branch_a(backbone_name=backbone, num_classes=3).to(device).eval()
        st = torch.load(ckpt_a, map_location="cpu", weights_only=False)
        branch_a.load_state_dict(st["model_state_dict"])

    pre_rows = []
    post_rows = []

    @torch.no_grad()
    def _infer_stage1_a(sub_df: "pd.DataFrame") -> "pd.DataFrame":
        if branch_a is None:
            raise RuntimeError("Branch A no cargada.")
        plan = plan_epoch(
            sub_df,
            max_neg_per_image=None,
            interleave_by_lineage=False,
            shuffle_images=False,
            shuffle_tiles_within_image=False,
            seed=0,
        )
        rows_all, cols_all, probs_all = [], [], []
        for batch in iter_image_batches(plan, batch_size=batch_size, device=device, label_mode="gate"):
            logits = branch_a(batch.rgb)
            p = torch.softmax(logits.float(), dim=-1)[:, 2].cpu().numpy()
            probs_all.append(p)
            rows_all.extend(batch.rows.cpu().tolist())
            cols_all.extend(batch.cols.cpu().tolist())
        p_a = np.concatenate(probs_all) if probs_all else np.array([])
        pred_df = pd.DataFrame({"row": rows_all, "col": cols_all, "p_fused": p_a})
        out = sub_df.merge(pred_df, on=["row", "col"], how="left")
        out["p_A"] = out["p_fused"]
        out["consensus"] = 1.0
        out["stage1_pred"] = (out["p_fused"] >= tau_s1).astype(int)
        out["stage1"] = np.where(out["stage1_pred"].to_numpy() == 1, "Mplus", "Mminus")
        return out

    total = len(images)
    for idx, rec in enumerate(images.itertuples(index=False), start=1):
        image_path = paths.root / str(rec.image_path)
        stem = Path(str(rec.image_stem)).stem
        sub = tiles[tiles["image_path"] == str(rec.image_path)].copy().reset_index(drop=True)
        if sub.empty:
            continue
        ts = int(sub["tile_size"].iloc[0]) if "tile_size" in sub.columns else _infer_tile_size_from_manifest(image_path, fallback=252)

        gimg = decode_jpeg_gpu(image_path, device=device)
        img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        del gimg
        torch.cuda.empty_cache()

        if phase in {"pre", "both"}:
            pre_df = sub.copy()
            pre_df["stage1_pred"] = (pre_df["stage1"].astype(str) == "Mplus").astype(int)
            pre_df["p_fused"] = pre_df["stage1_pred"].astype(float)
            pre_df["consensus"] = 1.0
            pre_df["stage2_pred"] = pre_df["stage2"]

            ctx_pre = LayerContext(image=img_np, tile_size=ts, tiles=pre_df)
            ctx_pre = downscale_context(ctx_pre, downscale)
            ov_pre = compose(["L0", "L2", "L7"], ctx_pre, alphas=[1.0, 0.30, 0.45])
            _save_step_png(
                arr=ov_pre,
                out_path=pre_dir / "maps" / f"{stem}__PRE_L0_L2_L7.png",
                step_code="PRE",
                title="Ground truth BD: L0 + L2 + L7",
                legend_items=[
                    ("Mplus", (0, 200, 255), "Stage1 colonizado"),
                    ("Mminus", (210, 180, 140), "Stage1 no colonizado"),
                    ("Background", (110, 110, 110), "fondo"),
                    ("AMColonised", (0, 180, 255), "Stage2 AM"),
                    ("BlueCoils", (80, 120, 255), "Stage2 ERM"),
                    ("BrownCoils", (160, 90, 40), "Stage2 ERM"),
                ],
            )
            pre_rows.append(
                {
                    "image_path": str(rec.image_path),
                    "n_tiles": int(len(pre_df)),
                    "mplus_gt": int((pre_df["stage1_pred"] == 1).sum()),
                }
            )

        if phase in {"post", "both"}:
            post_df = _infer_stage1_a(sub)
            ctx_post = LayerContext(image=img_np, tile_size=ts, tiles=post_df)
            ctx_post = downscale_context(ctx_post, downscale)
            ov_post = compose(["L0", "L2", "L3"], ctx_post, alphas=[1.0, 0.28, 0.42])
            _save_step_png(
                arr=ov_post,
                out_path=post_dir / "maps" / f"{stem}__POST_L0_L2_L3.png",
                step_code="POST",
                title="Prediccion modelo: L0 + L2 + L3",
                legend_items=[
                    ("Mplus", (0, 200, 255), "pred Stage1"),
                    ("Mminus", (210, 180, 140), "pred Stage1"),
                    ("L3 bajo", (70, 70, 70), "p(M+) cercano a 0"),
                    ("L3 alto", (240, 240, 240), "p(M+) cercano a 1"),
                ],
            )
            post_rows.append(
                {
                    "image_path": str(rec.image_path),
                    "n_tiles": int(len(post_df)),
                    "mplus_pred": int((post_df["stage1_pred"] == 1).sum()),
                    "mplus_ratio_pred": float((post_df["stage1_pred"] == 1).mean()),
                }
            )

        if idx % 5 == 0 or idx == total:
            log.info(f"[PRE/POST] procesadas {idx}/{total} imagenes")

    import pandas as pd

    if pre_rows:
        pd.DataFrame(pre_rows).to_csv(pre_dir / "tables" / "pre_summary.csv", index=False)
    if post_rows:
        pd.DataFrame(post_rows).to_csv(post_dir / "tables" / "post_summary.csv", index=False)

    report_lines = [
        f"# Dataset PRE/POST maps - {run.run_id}",
        "",
        "## Configuracion",
        f"- phase: `{phase}`",
        f"- n_images: `{len(images)}`",
        f"- downscale: `{downscale}`",
        f"- tau_s1 (POST): `{tau_s1}`",
        f"- backbone (POST): `{backbone}`",
        "",
        "## Estructura",
        "- `PRE/maps/*__PRE_L0_L2_L7.png`",
        "- `PRE/tables/pre_summary.csv`",
        "- `POST/maps/*__POST_L0_L2_L3.png`",
        "- `POST/tables/post_summary.csv`",
    ]
    _write_report(run.reports / "run_report.md", report_lines)
    log.info(f"[bold green]PRE/POST OK[/bold green] -> {run.root}")


@app.command(name="build-gate-cache")
def build_gate_cache_cmd(
    legacy_h5: bool = typer.Option(
        False,
        "--legacy-h5",
        help="Compilar cache HDF5 legacy RGB+saliencia (~44 GB). Default: embeddings v2 (~150 MB).",
    ),
    backbone: Optional[str] = typer.Option(None, help="Backbone DINOv2 para embeddings"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size GPU por imagen (32 seguro en 8GB VRAM)"),
    force: bool = typer.Option(False, "--force", help="Recompilar aunque la cache sea valida"),
):
    """Precompila cache de embeddings DINOv2 (v2) o HDF5 legacy (--legacy-h5)."""
    import warnings

    import pandas as pd
    import torch

    warnings.filterwarnings("ignore", message=".*xFormers.*")
    warnings.filterwarnings("ignore", message=".*not writable.*")

    from .phase_d_stage1 import split_by_image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")
    paths = get_paths()
    device = torch.device("cuda")
    train_df, val_df, info = split_by_image(
        lineages=["AM"], split_mode="fixed", train_splits=("train",), val_splits=("test",)
    )
    all_tiles = pd.concat([train_df, val_df], ignore_index=True).drop_duplicates(
        subset=["image_path", "row", "col"]
    )
    log.info(f"[Gate cache] {len(all_tiles):,} tiles ({info['n_train_images']}+{info['n_val_images']} imgs)")

    if legacy_h5:
        from .phase_d_stage1.gate_tile_h5_cache import build_gate_tile_h5_cache, load_frozen_u2net_saliency

        u2net = load_frozen_u2net_saliency(paths.root / "models" / "weights" / "u2netp.pth", device)
        path = build_gate_tile_h5_cache(
            all_tiles,
            u2net,
            device=device,
            batch_size=int(_cfg("GATE_CACHE_BATCH_SIZE", 32)),
            empty_cache_every_n_batches=int(_cfg("GATE_CACHE_EMPTY_CACHE_EVERY_N_BATCHES", 0)),
            gc_collect_every_n_batches=int(_cfg("GATE_CACHE_GC_COLLECT_EVERY_N_BATCHES", 0)),
        )
        log.info(f"[bold green]Cache HDF5 legacy listo[/bold green] -> {path}")
        return

    from .gate_runflow import _cache_build_kwargs, _mplus_aug_variants_from_cfg, execute_build_gate_cache

    cfg = _load_user_config()
    build_kw = _cache_build_kwargs(cfg) if cfg is not None else {
        "dynamic_batch": True,
        "vram_budget_mb": 7500.0,
        "cpu_decode_above_mb": 500.0,
        "empty_cache_every_n_batches": 0,
        "gc_collect_every_n_batches": 0,
        "memmap_flush_every_n_images": 5,
    }
    mplus = _mplus_aug_variants_from_cfg(cfg) if cfg is not None else ()

    execute_build_gate_cache(
        backbone=str(backbone or _cfg("GATE_BACKBONE", "dinov2_vits14")),
        batch_size=int(batch_size if batch_size is not None else _cfg("GATE_CACHE_BATCH_SIZE", 32)),
        force_rebuild=force,
        mplus_aug_variants=mplus,
        **build_kw,
    )


@app.command(name="build-gate-attention-cache")
def build_gate_attention_cache_cmd(
    backbone: Optional[str] = typer.Option(None, help="Backbone DINOv2 para atencion ViT"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size GPU por imagen"),
    layers: Optional[str] = typer.Option(
        None,
        "--layers",
        help="Capas ViT: all | last | 0,5,11 (default: config GATE_CACHE_ATTENTION_LAYERS)",
    ),
    head_reduce: Optional[str] = typer.Option(
        None,
        "--head-reduce",
        help="Reduccion heads: mean | none (default: config GATE_CACHE_ATTENTION_HEAD_REDUCE)",
    ),
    force: bool = typer.Option(False, "--force", help="Recompilar attn aunque ya exista"),
):
    """Compila mapas de atencion ViT CLS->patch sin recomputar embeddings existentes."""
    import warnings

    warnings.filterwarnings("ignore", message=".*xFormers.*")
    warnings.filterwarnings("ignore", message=".*not writable.*")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")

    from .gate_runflow import _cache_build_kwargs, _mplus_aug_variants_from_cfg, execute_build_gate_attention_cache

    cfg = _load_user_config()
    build_kw = _cache_build_kwargs(cfg) if cfg is not None else {}
    attn_layers = str(layers or _cfg("GATE_CACHE_ATTENTION_LAYERS", "all"))
    attn_reduce = str(head_reduce or _cfg("GATE_CACHE_ATTENTION_HEAD_REDUCE", "mean"))
    mplus = _mplus_aug_variants_from_cfg(cfg) if cfg is not None else ()
    execute_build_gate_attention_cache(
        backbone=str(backbone or _cfg("GATE_BACKBONE", "dinov2_vits14")),
        batch_size=int(batch_size if batch_size is not None else _cfg("GATE_CACHE_BATCH_SIZE", 32)),
        force_rebuild=force,
        mplus_aug_variants=mplus,
        attention_layers=attn_layers,
        attention_head_reduce=attn_reduce,
        **{
            k: v
            for k, v in build_kw.items()
            if k not in {"cache_attention", "attention_layers", "attention_head_reduce"}
        },
    )


@app.command(name="train-gate-am")
def train_gate_am_cmd(
    full: bool = typer.Option(
        False,
        "--full",
        help="Entrenamiento completo (~horas, 1383 batches/epoca). Sin --full = modo rapido.",
    ),
    finetune: bool = typer.Option(
        False,
        "--finetune",
        help="Fine-tune DINOv2 parcial (on-the-fly JPEG). Default: linear probe desde cache embeddings.",
    ),
    legacy_h5: bool = typer.Option(
        False,
        "--legacy-h5",
        help="Entrenar desde cache HDF5 legacy (solo util con --finetune).",
    ),
    backbone: Optional[str] = typer.Option(None, help="Backbone DINOv2"),
    epochs: Optional[int] = typer.Option(None, help="Epocas (default: config.py GATE_EPOCHS)"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size (default: config.py)"),
    max_bg_per_image: Optional[int] = typer.Option(None, help="Max tiles Background por imagen y epoca"),
    max_train_batches: Optional[int] = typer.Option(None, help="Limite batches/epoca train"),
    max_val_batches: Optional[int] = typer.Option(None, help="Limite batches test"),
    eval_every_s: Optional[float] = typer.Option(None, help="Test rapido cada N segundos (fast=120)"),
    max_vis_images: Optional[int] = typer.Option(None, help="Imagenes test con mapas gold vs pred"),
    vis_downscale: Optional[int] = typer.Option(None, help="Downscale para mapas de test"),
):
    """Gate AM: DINOv2 mean-pool + Slice-MS. Default linear probe (~minutos con cache embeddings)."""
    import pandas as pd
    import torch

    from .gate_runflow import GateTrainParams, execute_train_gate_am, gate_train_params_from_config
    from .phase_d_stage1.gate_training_protocol import GateTrainProtocol
    from .phase_d_stage1 import build_branch_a, count_parameters, split_by_image
    from .phase_d_stage1.gate_tile_dino import (
        GateTileDinoGPU,
        train_gate_tile_dino_gpu,
    )
    from .phase_d_stage1.gate_train_report import finalize_gate_tile_dino_run

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")
    if legacy_h5 and not finetune:
        raise typer.BadParameter("--legacy-h5 solo aplica con --finetune.")

    cfg = _load_user_config()
    if cfg is not None and not finetune and not legacy_h5:
        _cli_overrides = any(
            (
                full,
                epochs is not None,
                batch_size is not None,
                max_bg_per_image is not None,
                max_train_batches is not None,
                max_val_batches is not None,
                eval_every_s is not None,
                max_vis_images is not None,
                vis_downscale is not None,
                backbone is not None,
            )
        )
        if not _cli_overrides:
            from .gate_runflow import run_gate_pipeline

            run_gate_pipeline(cfg)
            return
    base = gate_train_params_from_config(cfg) if cfg is not None else GateTrainParams(
        backbone="dinov2_vits14",
        epochs=30,
        batch_size=64,
        probe=True,
        full_dataset=True,
        max_bg_per_image=50,
        max_train_batches=None,
        max_val_batches=None,
        max_vis_images=None,
        vis_downscale=4,
        spatial_audit=True,
        spatial_audit_sample=5,
        skip_pretrain_viz=False,
        report_from_cache=True,
        save_live_snapshots=True,
        vis_all_test=True,
        render_maps=True,
        dino_input_size=252,
        seg_target_size=360,
        protocol=GateTrainProtocol(),
    )
    backbone = str(backbone or base.backbone)
    if not backbone.startswith("dinov2"):
        raise typer.BadParameter("Este pipeline requiere backbone DINOv2 (ej. dinov2_vits14).")

    if not finetune and not legacy_h5:
        use_full = full if full else base.full_dataset
        if max_train_batches is not None or max_val_batches is not None:
            use_full = False
        if use_full:
            ep = epochs if epochs is not None else base.epochs
            bs = batch_size if batch_size is not None else base.batch_size
            mtrain = max_train_batches
            mval = max_val_batches
            mbg = max_bg_per_image if max_bg_per_image is not None else base.max_bg_per_image
        else:
            ep = epochs if epochs is not None else 4
            bs = batch_size if batch_size is not None else base.batch_size
            mtrain = max_train_batches if max_train_batches is not None else 25
            mval = max_val_batches if max_val_batches is not None else 12
            mbg = max_bg_per_image if max_bg_per_image is not None else 15

        from dataclasses import replace

        proto = replace(
            base.protocol or GateTrainProtocol(),
            max_epochs=int(ep),
            max_bg_per_image=int(mbg),
        )
        use_probe = base.probe if not finetune else False
        params = GateTrainParams(
            backbone=backbone,
            epochs=int(ep),
            batch_size=int(bs),
            probe=use_probe,
            full_dataset=use_full,
            max_bg_per_image=int(mbg),
            max_train_batches=mtrain,
            max_val_batches=mval,
            max_vis_images=max_vis_images if max_vis_images is not None else base.max_vis_images,
            vis_downscale=int(
                vis_downscale if vis_downscale is not None else base.vis_downscale
            ),
            spatial_audit=base.spatial_audit,
            spatial_audit_sample=base.spatial_audit_sample,
            skip_pretrain_viz=base.skip_pretrain_viz,
            report_from_cache=base.report_from_cache,
            save_live_snapshots=base.save_live_snapshots,
            vis_all_test=base.vis_all_test,
            render_maps=base.render_maps,
            dino_input_size=base.dino_input_size,
            seg_target_size=base.seg_target_size,
            mplus_aug_variants=base.mplus_aug_variants,
            cache_attention=base.cache_attention,
            attention_layers=base.attention_layers,
            attention_head_reduce=base.attention_head_reduce,
            use_probe_attention=base.use_probe_attention,
            protocol=proto,
            gate4=base.gate4,
            formal_train=base.formal_train,
            finetune_mode=base.finetune_mode,
            pooling_mode=base.pooling_mode,
        )
        execute_train_gate_am(params, require_cache=True)
        return

    freeze_backbone = False
    if full:
        epochs = epochs if epochs is not None else base.epochs
        batch_size = batch_size if batch_size is not None else 16
        max_bg = max_bg_per_image if max_bg_per_image is not None else base.max_bg_per_image
        eval_interval = eval_every_s
        mode_label = "FULL_FINETUNE"
    else:
        epochs = epochs if epochs is not None else 4
        batch_size = batch_size if batch_size is not None else 32
        max_train_batches = max_train_batches if max_train_batches is not None else 25
        max_val_batches = max_val_batches if max_val_batches is not None else 12
        max_bg = max_bg_per_image if max_bg_per_image is not None else 15
        eval_interval = None
        mode_label = "FAST_FINETUNE"

    paths = get_paths()
    run = RunOutputs.create("gate_am_train")
    device = torch.device("cuda")
    from .phase_d_stage1.gate_classes import GATE_CLASS_NAMES

    num_classes = len(GATE_CLASS_NAMES)
    log.info(
        f"[Gate AM] modo={mode_label} | DINOv2 mean-pool | "
        f"epochs={epochs} bs={batch_size} | device={device} run={run.run_id}"
    )
    if not full and freeze_backbone:
        log.info(
            f"[Gate AM] FAST probe: ~{max_train_batches} batches train + ~{max_val_batches} test/epoca "
            f"(segundos con cache embeddings). Usa --finetune o --full para otros modos."
        )

    train_df, val_df, info_split = split_by_image(
        lineages=["AM"],
        split_mode="fixed",
        train_splits=("train",),
        val_splits=("test",),
    )
    log.info(
        f"[Gate AM] train_imgs={info_split['n_train_images']} test_imgs={info_split['n_val_images']} "
        f"train_tiles={info_split['n_train_tiles']} test_tiles={info_split['n_val_tiles']}"
    )

    ckpt_dir = paths.root / "models" / "checkpoints" / "gate_am"
    weights_dir = paths.root / "models" / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    classifier = build_branch_a(
        backbone_name=backbone, num_classes=num_classes, freeze_backbone=freeze_backbone
    )
    log.info(f"[DINOv2] {backbone} params={count_parameters(classifier):,}")

    h5_store = None
    embed_store = None
    all_tiles = pd.concat([train_df, val_df], ignore_index=True).drop_duplicates(
        subset=["image_path", "row", "col"]
    )

    if freeze_backbone:
        from .phase_d_stage1.gate_embed_cache import ensure_gate_embed_cache

        log.info(f"[Gate AM] Cache embeddings: {len(all_tiles):,} tiles (compila si falta)")
        embed_store = ensure_gate_embed_cache(
            all_tiles,
            classifier.backbone,
            device,
            backbone_name=backbone,
        )
    elif legacy_h5:
        from .phase_d_stage1.gate_tile_h5_cache import ensure_gate_tile_h5_cache, load_frozen_u2net_saliency

        u2net = load_frozen_u2net_saliency(weights_dir / "u2netp.pth", device)
        log.info(f"[Gate AM] Cache HDF5 legacy: {len(all_tiles):,} tiles")
        h5_store = ensure_gate_tile_h5_cache(all_tiles, u2net, device)
    else:
        log.info("[Gate AM] Fine-tune on-the-fly (JPEG + DINO mean-pool en GPU)")

    history = train_gate_tile_dino_gpu(
        model=classifier,
        train_df=train_df,
        val_df=val_df,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=ckpt_dir,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_bg_per_image=max_bg,
        h5_store=h5_store,
        embed_store=embed_store,
        freeze_backbone=freeze_backbone,
        dino_input_size=base.dino_input_size,
        seg_target_size=base.seg_target_size,
    )

    if h5_store is not None:
        h5_store.close()

    gate = GateTileDinoGPU(classifier=classifier, device=device).to(device)
    test_images = sorted({paths.root / p for p in val_df["image_path"].unique()})
    train_config = {
        "mode": mode_label,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_bg_per_image": max_bg,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "eval_every_s": eval_interval,
        "embed_cache": embed_store is not None,
        "h5_cache": h5_store is not None,
        "freeze_backbone": freeze_backbone,
        "backbone": backbone,
        "pipeline": "dinov2_embed_cache+slice_ms",
    }
    report_md = finalize_gate_tile_dino_run(
        run=run,
        history=history.to_dict(),
        info_split=info_split,
        gate=gate,
        train_df=train_df,
        val_df=val_df,
        val_image_paths=test_images,
        backbone=backbone,
        train_config=train_config,
        device=device,
        batch_size=batch_size,
        max_vis_images=max_vis_images,
        ckpt_dir=ckpt_dir,
        downscale=vis_downscale,
    )
    log.info(f"[Gate AM] [bold green]OK[/bold green] -> {report_md}")


@app.command(name="recover-gate-am-report")
def recover_gate_am_report_cmd(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="ID de corrida en outputs/ (ej. 20260618_234151_gate_am_train). Default: ultima incompleta.",
    ),
):
    """Regenera reportes/mapas si el entrenamiento termino sin finalize."""
    from .gate_runflow import recover_gate_am_run_report

    recover_gate_am_run_report(run_id=run_id)


@app.command(name="infer-gate-am")
def infer_gate_am_cmd(
    image: Optional[Path] = typer.Option(None, help="Imagen AM (default config.py)"),
    backbone: Optional[str] = typer.Option(None, help="Backbone DINOv2"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size"),
):
    """Inferencia gate AM: DINOv2 mean-pool por tile + mapas."""
    import torch

    from .phase_d_stage1.gate_tile_dino import infer_image_gate_tile_dino_gpu, load_gate_tile_dino
    from .phase_d_stage1.gate_train_report import render_gate_image_maps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")

    image = image or _cfg_path("DEFAULT_IMAGE")
    if image is None:
        raise typer.BadParameter("Indica --image o DEFAULT_IMAGE en config.py")
    backbone = backbone or str(_cfg("DEFAULT_BACKBONE", "dinov2_vits14"))
    batch_size = int(batch_size if batch_size is not None else _cfg("DEFAULT_BATCH_SIZE", 32))

    run = RunOutputs.create("gate_am_infer", suffix=image.stem)
    device = torch.device("cuda")

    try:
        gate = load_gate_tile_dino(backbone=backbone, device=device)
    except FileNotFoundError as e:
        log.error(str(e))
        raise typer.Exit(code=1) from e

    df = infer_image_gate_tile_dino_gpu(image, gate, batch_size=batch_size)
    if df.empty:
        log.warning("Sin tiles para esta imagen.")
        return

    from .common.io import write_table

    csv_path = write_table(df, run.tables / f"{image.stem}__gate_am_probs")
    log.info(f"[Gate AM] probs -> {csv_path}")

    ts = _infer_tile_size_from_manifest(image, fallback=252)
    gold_tiles = _load_tiles_for_image(image)
    maps_dir = run.maps

    if not gold_tiles.empty:
        render_gate_image_maps(
            image_path=image,
            tiles_gold=gold_tiles,
            tiles_pred=df,
            maps_dir=maps_dir,
            device=device,
            tile_size=ts,
            downscale=4,
            image_stem=image.stem,
        )
    else:
        from .layers import LayerContext, compose, downscale_context, save_png
        from .phase_b_tiling.gpu_io import decode_jpeg_gpu

        gimg = decode_jpeg_gpu(image, device=device)
        img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        del gimg
        torch.cuda.empty_cache()
        pred = df.copy()
        pred["stage1"] = pred["stage1_pred"].astype(str)
        ctx = downscale_context(LayerContext(image=img_np, tile_size=ts, tiles=pred), 4)
        save_png(
            compose(["L0", "L1"], downscale_context(LayerContext(image=img_np, tile_size=ts), 4), alphas=[1.0, 0.85]),
            maps_dir / f"{image.stem}__L0_L1_grid.png",
        )
        save_png(compose(["L0", "L2"], ctx, alphas=[1.0, 0.55]), maps_dir / f"{image.stem}__L0_L2_pred.png")

    n_mplus = int((df["stage1_pred"] == "Mplus").sum())
    n_mminus = int((df["stage1_pred"] == "Mminus").sum())
    n_bg = int((df["stage1_pred"] == "Background").sum())
    n_root = n_mplus + n_mminus
    colon_pct = 100.0 * n_mplus / max(n_root, 1)

    report_lines = [
        f"# Gate AM — {image.name}",
        "",
        f"- run_id: `{run.run_id}`",
        f"- pipeline: DINOv2 mean-pool + Slice-MS",
        f"- tiles totales: **{len(df)}**",
        f"- M+ predichos: **{n_mplus}**",
        f"- M- predichos: **{n_mminus}**",
        f"- Background: **{n_bg}**",
        f"- Colonizacion (M+ / raiz): **{colon_pct:.1f}%**",
        "",
        "## Mapas",
        f"- `maps/{image.stem}__L0_L1_grid.png` — grilla",
        f"- `maps/{image.stem}__L0_L2_pred.png` — prediccion",
    ]
    if not gold_tiles.empty:
        report_lines += [
            f"- `maps/{image.stem}__L0_L2_gold.png` — gold CSV",
            f"- `maps/{image.stem}__gold_vs_pred.png` — comparacion",
            f"- `maps/{image.stem}__L0_L2_errors.png` — errores",
        ]
    _write_report(run.reports / "run_report.md", report_lines)
    log.info(
        f"[Gate AM] M+={n_mplus} M-={n_mminus} Bg={n_bg} colon={colon_pct:.1f}% -> {run.root}"
    )


@app.command(name="train-stage2")
def train_stage2_cmd(
    lineage: str = typer.Option("AM", help="Linaje: AM o ERM"),
    backbone: str = typer.Option("dinov2_vits14", help="Backbone Branch A"),
    epochs: int = typer.Option(3, help="Epocas por rama"),
    batch_size: int = typer.Option(16, help="Batch size"),
    val_fraction: float = typer.Option(0.2, help="Fraccion de imagenes para validacion"),
    max_train_batches: int = typer.Option(40, help="Max batches por epoca (smoke)"),
    max_val_batches: int = typer.Option(15, help="Max batches validacion"),
    skip_a: bool = typer.Option(False, help="Saltar rama A"),
    skip_b: bool = typer.Option(False, help="Saltar rama B"),
    skip_c: bool = typer.Option(False, help="Saltar rama C"),
    use_amp: bool = typer.Option(False, help="AMP"),
):
    """Fase E — Entrena Stage2 multiclas e (solo tiles M+) por linaje, 100% CUDA."""
    import json
    import torch

    from .phase_e_stage2 import (
        build_branch_a_mc,
        build_branch_b_mc,
        build_branch_c_mc,
        count_parameters,
        split_by_image_mplus,
        train_branch_gpu_s2,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")

    lineage = lineage.upper()
    paths = get_paths()
    run = RunOutputs.create("stage2_train", suffix=lineage)
    device = torch.device("cuda")

    train_df, val_df, info_split, class_map = split_by_image_mplus(
        lineage=lineage, val_fraction=val_fraction,
    )
    log.info(
        f"[Fase E] lineage={lineage} classes={list(class_map.classes)} "
        f"train_tiles={info_split['n_train_tiles']} val_tiles={info_split['n_val_tiles']} "
        f"run={run.run_id}"
    )

    ckpt_dir = paths.root / "models" / "checkpoints" / f"stage2_{lineage.lower()}"
    weights_dir = paths.root / "models" / "weights"
    n_cls = class_map.num_classes
    history_all = {}

    branches: list[tuple[str, torch.nn.Module]] = []
    if not skip_a:
        a = build_branch_a_mc(n_cls, backbone_name=backbone)
        log.info(f"[Branch A] {backbone} params={count_parameters(a):,}")
        branches.append(("A", a))
    if not skip_b:
        b = build_branch_b_mc(n_cls, weights_path=weights_dir / "u2netp.pth")
        log.info(f"[Branch B] U2NETP params={count_parameters(b):,}")
        branches.append(("B", b))
    if not skip_c:
        c = build_branch_c_mc(n_cls)
        log.info(f"[Branch C] FreqNet params={count_parameters(c):,}")
        branches.append(("C", c))

    for name, model in branches:
        log.info(f"[Fase E] === Entrenando Branch {name} ({lineage}) ===")
        history = train_branch_gpu_s2(
            model=model,
            train_df=train_df,
            val_df=val_df,
            class_map=class_map,
            branch=name,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_dir=ckpt_dir,
            branch_name=f"stage2_{lineage.lower()}_branch_{name.lower()}",
            use_amp=use_amp,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
        history_all[name] = history.to_dict()

    report_path = run.reports / "stage2_train_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run.run_id,
                "lineage": lineage,
                "classes": list(class_map.classes),
                "backbone_a": backbone,
                "device": str(device),
                "history": history_all,
                "split_info": info_split,
            },
            f, indent=2,
        )
    log.info(f"[Fase E] [bold green]OK[/bold green] -> {report_path}")


@app.command(name="infer-stage2")
def infer_stage2_cmd(
    image: Optional[Path] = typer.Option(None, help="Imagen a inferir (si se omite usa config.py::DEFAULT_IMAGE)"),
    lineage: Optional[str] = typer.Option(None, help="Linaje AM/ERM (si se omite usa config.py::DEFAULT_LINEAGE)"),
    backbone: Optional[str] = typer.Option(None, help="Backbone Branch A (si se omite usa config.py::DEFAULT_BACKBONE)"),
    tau_s1: Optional[float] = typer.Option(None, help="Umbral gate Stage1 (si se omite usa config.py::DEFAULT_TAU_S1)"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size (si se omite usa config.py::DEFAULT_BATCH_SIZE)"),
):
    """Fase E — Stage1 gate + Stage2 subclases en M+ + capas L4/L7/L8."""
    import json
    import torch

    from .common.io import write_table
    from .layers import LayerContext, compose, downscale_context, save_png, render_layer
    from .phase_b_tiling.gpu_io import decode_jpeg_gpu
    from .phase_d_stage1 import infer_image_gate_gpu, load_gate_am_ensemble
    from .phase_d_stage1.segmentation_map import build_segmentation_map_mplus
    from .phase_e_stage2 import (
        StageTwoEnsembleGPU,
        build_branch_a_mc,
        build_branch_b_mc,
        build_branch_c_mc,
        infer_image_stage2_gpu,
        load_stage2_class_map,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Este pipeline es GPU-only sin fallback.")

    image = image or _cfg_path("DEFAULT_IMAGE")
    if image is None:
        raise typer.BadParameter("Debes indicar --image o definir DEFAULT_IMAGE en config.py")
    lineage = str(lineage or _cfg("DEFAULT_LINEAGE", "AM")).upper()
    backbone = backbone or str(_cfg("DEFAULT_BACKBONE", "dinov2_vits14"))
    tau_s1 = float(tau_s1 if tau_s1 is not None else _cfg("DEFAULT_TAU_S1", 0.65))
    batch_size = int(batch_size if batch_size is not None else _cfg("DEFAULT_BATCH_SIZE", 32))

    paths = get_paths()
    run = RunOutputs.create("stage2_infer", suffix=f"{lineage}__{image.stem}")
    device = torch.device("cuda")
    ckpt_s2 = paths.root / "models" / "checkpoints" / f"stage2_{lineage.lower()}"
    weights_dir = paths.root / "models" / "weights"

    try:
        ens1 = load_gate_am_ensemble(backbone=backbone, device=device)
    except FileNotFoundError as e:
        log.error(str(e))
        raise typer.Exit(code=1) from e

    s1_df = infer_image_gate_gpu(image, ens1, batch_size=batch_size)
    if s1_df.empty:
        log.warning("Sin tiles para esta imagen.")
        return

    # Stage2 subclases
    class_map = load_stage2_class_map(lineage)
    n_cls = class_map.num_classes
    a2 = build_branch_a_mc(n_cls, backbone_name=backbone)
    b2 = build_branch_b_mc(n_cls, weights_path=weights_dir / "u2netp.pth")
    c2 = build_branch_c_mc(n_cls)

    def _load_s2(model, name):
        p = ckpt_s2 / f"stage2_{lineage.lower()}_branch_{name.lower()}_best.pt"
        if p.exists():
            st = torch.load(p, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model_state_dict"])
            log.info(f"[Stage2 {name}] checkpoint F1={st.get('f1_macro', 0):.4f}")
        else:
            log.warning(f"[Stage2 {name}] sin checkpoint en {p}")

    _load_s2(a2, "A"); _load_s2(b2, "B"); _load_s2(c2, "C")

    ens2 = StageTwoEnsembleGPU(
        branch_a=a2, branch_b=b2, branch_c=c2, class_map=class_map,
    ).to(device)

    df = infer_image_stage2_gpu(image, ens2, s1_df, batch_size=batch_size)
    csv = write_table(df, run.tables / f"{image.stem}__stage2_probs")
    log.info(f"[Fase E] Probs Stage2 -> {csv}")

    ts = _infer_tile_size_from_manifest(image, fallback=252)
    gimg = decode_jpeg_gpu(image, device=device)
    img_np = gimg.tensor.permute(1, 2, 0).contiguous().cpu().numpy()
    del gimg
    torch.cuda.empty_cache()

    import numpy as np

    df_l2 = df.copy()
    if "stage1_pred" in df_l2.columns:
        df_l2["stage1"] = df_l2["stage1_pred"].astype(str)

    ctx = LayerContext(image=img_np, tile_size=ts, tiles=df_l2)
    seg_prob, seg_bin = build_segmentation_map_mplus(
        image_path=image,
        tiles_df=df_l2,
        seg_branch_model=ens1.branch_b,
        batch_size=batch_size,
        device=device,
        threshold=0.5,
    )
    np.savez_compressed(
        run.tables / f"{image.stem}__stage2_segmentation_l4.npz",
        seg_prob=seg_prob,
        seg_bin=seg_bin,
    )
    ctx.seg_mask = seg_prob
    ctx = downscale_context(ctx, 4)

    l0 = render_layer("L0", ctx)
    l2 = render_layer("L2", ctx)
    l4 = render_layer("L4", ctx)
    l7 = render_layer("L7", ctx)
    l8 = render_layer("L8", ctx)
    ov_l0_l4 = compose(["L0", "L4"], ctx, alphas=[1.0, 0.65])
    ov_l0_l2_l4 = compose(["L0", "L2", "L4"], ctx, alphas=[1.0, 0.25, 0.65])
    ov_l0_l2_l4_l7 = compose(["L0", "L2", "L4", "L7"], ctx, alphas=[1.0, 0.2, 0.55, 0.45])

    # Archivos legacy (compatibilidad)
    save_png(ov_l0_l4, run.maps / f"{image.stem}__L0_L4.png")
    save_png(ov_l0_l2_l4, run.maps / f"{image.stem}__L0_L2_L4.png")
    save_png(ov_l0_l2_l4_l7, run.maps / f"{image.stem}__L0_L2_L4_L7.png")
    if "stage2_entropy" in df.columns and df["stage2_entropy"].notna().any():
        ov_l0_l4_l7_l8 = compose(["L0", "L4", "L7", "L8"], ctx, alphas=[1.0, 0.5, 0.45, 0.35])
        save_png(ov_l0_l4_l7_l8, run.maps / f"{image.stem}__L0_L4_L7_L8.png")
        ov_diag = compose(["L0", "L3", "L4", "L6", "L7", "L8"], ctx, alphas=[1.0, 0.25, 0.45, 0.25, 0.4, 0.35])
        save_png(
            ov_diag, run.maps / f"{image.stem}__diagnostico_stage2.png",
        )
    else:
        ov_l0_l4_l7_l8 = compose(["L0", "L4", "L7"], ctx, alphas=[1.0, 0.55, 0.45])
        ov_diag = compose(["L0", "L4", "L7"], ctx, alphas=[1.0, 0.55, 0.45])
        log.warning("[Fase E] Gate Stage1 no predijo M+ - L8 omitida (entrenar/cargar checkpoints Stage1)")

    # Secuencia unitaria con leyendas
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="01", slug="L0_input",
        arr=l0, title="Entrada L0 (imagen base)",
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="02", slug="L2_gate_pred",
        arr=l2, title="Filtro Stage1 (tiles usados para Stage2)",
        legend_items=[
            ("Mplus", (0, 200, 255), "tile pasa a Stage2"),
            ("Mminus", (210, 180, 140), "tile bloqueado en Stage1"),
        ],
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="03", slug="L4_segmentation",
        arr=ov_l0_l4, title="Segmentacion U2Net (L4)",
        legend_items=[("Cian", (40, 220, 255), "estructura detectada")],
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="04", slug="L7_stage2_classes",
        arr=compose(["L0", "L7"], ctx, alphas=[1.0, 0.5]), title="Subclase Stage2 por tile (L7)",
        legend_items=[
            ("AMColonised", (0, 180, 255), "clase AM"),
            ("Hybrid", (255, 120, 0), "clase AM hibrida"),
        ],
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="05", slug="L8_stage2_uncertainty",
        arr=compose(["L0", "L8"], ctx, alphas=[1.0, 0.55]), title="Incertidumbre Stage2 (L8)",
        legend_items=[
            ("Oscuro", (40, 40, 40), "incertidumbre baja"),
            ("Claro", (220, 220, 220), "incertidumbre alta"),
        ],
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="06", slug="L0_L2_L4_L7_overlay",
        arr=ov_l0_l2_l4_l7, title="Integracion gate + segmentacion + clase",
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="07", slug="L0_L4_L7_L8_overlay",
        arr=ov_l0_l4_l7_l8, title="Clase + incertidumbre sobre segmentacion",
    )
    _save_stage_sequence(
        run=run, image_stem=image.stem, step_code="08", slug="diagnostico_stage2",
        arr=ov_diag, title="Diagnostico Stage2 completo",
    )

    n_s2 = int(df["stage2_pred"].notna().sum()) if "stage2_pred" in df.columns else 0
    seg_cov = float((seg_bin > 0.5).mean() * 100.0)
    stage2_counts = {}
    if "stage2_pred" in df.columns:
        stage2_counts = {str(k): int(v) for k, v in df["stage2_pred"].dropna().value_counts().to_dict().items()}
    rep_path = _write_stage2_run_report(
        run=run,
        image=image,
        lineage=lineage,
        tau_s1=tau_s1,
        n_s2=n_s2,
        n_tot=len(df),
        seg_cov_pct=seg_cov,
        stage2_counts=stage2_counts,
    )
    log.info(f"[Fase E] tiles con Stage2: {n_s2}")
    log.info(f"[Fase E] Cobertura L4 (pixeles segmentados): {seg_cov:.2f}%")
    log.info(f"[Fase E] Reporte -> {rep_path}")
    log.info(f"[bold green]Inferencia Stage2 OK[/bold green] run={run.run_id} -> {run.root}")


@app.command(name="weakseg-morph")
def weakseg_morph_cmd(
    image: Optional[Path] = typer.Option(None, help="Imagen a procesar (default config.py::DEFAULT_IMAGE)"),
    manifests: Optional[Path] = typer.Option(None, help="Directorio manifests (default: ./manifests)"),
    annotations: Optional[Path] = typer.Option(
        None, help="CSV/XML legacy opcional para tiles positivos (si se omite usa manifests)"
    ),
    tile_size: int = typer.Option(126, help="Tile size fallback para parseo legacy"),
    canny_low: int = typer.Option(40, help="Umbral bajo Canny"),
    canny_high: int = typer.Option(110, help="Umbral alto Canny"),
    frangi_pctl: float = typer.Option(82.0, help="Percentil Frangi para hifas"),
    vesicle_circularity_min: float = typer.Option(0.85, help="Circularidad minima de vesiculas"),
    arbuscule_pctl: float = typer.Option(93.0, help="Percentil de entropia para arbusculos"),
    seam_sigma: float = typer.Option(0.85, help="Sigma de suavizado en costuras"),
    alpha_overlay: float = typer.Option(0.45, help="Alpha de auditoria visual overlay"),
    root_min_cov: float = typer.Option(0.02, help="Cobertura minima de raiz por tile"),
    root_max_cov: float = typer.Option(0.70, help="Cobertura maxima de raiz por tile"),
    min_patch_fg_ratio: float = typer.Option(0.002, help="Minimo foreground por parche para exportar"),
    export_dino_patches: bool = typer.Option(True, help="Exportar parches imagen/mascara para DINOv2"),
    patch_size: int = typer.Option(518, help="Tamano de parche para DINOv2"),
    patch_stride: int = typer.Option(518, help="Stride de parche para DINOv2"),
):
    """Fase I — Pipeline morfologico weakly-supervised (Fases 1-4 del plan)."""
    from .phase_i_weakseg import WeakSegParams, run_weakseg_pipeline

    image = image or _cfg_path("DEFAULT_IMAGE")
    if image is None:
        raise typer.BadParameter("Debes indicar --image o definir DEFAULT_IMAGE en config.py")
    manifests = manifests or (get_paths().manifests)

    params = WeakSegParams(
        tile_size=tile_size,
        canny_low=canny_low,
        canny_high=canny_high,
        frangi_pctl=frangi_pctl,
        vesicle_circularity_min=vesicle_circularity_min,
        arbuscule_pctl=arbuscule_pctl,
        seam_sigma=seam_sigma,
        alpha_overlay=alpha_overlay,
        root_min_cov=root_min_cov,
        root_max_cov=root_max_cov,
        min_patch_fg_ratio=min_patch_fg_ratio,
    )
    run = run_weakseg_pipeline(
        image_path=image,
        manifests_dir=manifests,
        params=params,
        annotations_path=annotations,
        export_dino_patches=export_dino_patches,
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    log.info(f"[bold green]WeakSeg morfologico OK[/bold green] -> {run.root}")


@app.command(name="train-dinov2-weakseg")
def train_dinov2_weakseg_cmd(
    patches_root: Path = typer.Option(..., help="Directorio con dino_patches/{images,masks}"),
    model_name: str = typer.Option("facebook/dinov2-small", help="Checkpoint base DINOv2"),
    num_classes: int = typer.Option(6, help="Numero de clases (incluye background)"),
    batch_size: int = typer.Option(2, help="Batch size"),
    epochs: int = typer.Option(2, help="Epocas"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    freeze_backbone: bool = typer.Option(True, help="Congelar backbone y afinar cabeza lineal"),
):
    """Fase I.4 — Fine-tuning de cabeza de segmentacion DINOv2."""
    from .phase_i_weakseg.dinov2 import DinoTrainConfig, train_dinov2_linear_head

    cfg = DinoTrainConfig(
        num_classes=num_classes,
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        freeze_backbone=freeze_backbone,
    )
    run = train_dinov2_linear_head(patches_root=patches_root, cfg=cfg)
    log.info(f"[bold green]DINOv2 weakseg train OK[/bold green] -> {run.root}")


@app.command(name="weakseg-gui")
def weakseg_gui_cmd():
    """Lanza interfaz grafica PyQt5 para ejecutar WeakSeg por imagen."""
    from .phase_i_weakseg.gui import launch_weakseg_gui

    launch_weakseg_gui()


@app.command()
def info():
    """Muestra rutas y estado del proyecto."""
    paths = get_paths()
    log.info(f"root      = {paths.root}")
    log.info(f"data      = {paths.data}")
    log.info(f"configs   = {paths.configs}")
    log.info(f"manifests = {paths.manifests}")
    log.info(f"outputs   = {paths.outputs}")
    import torch
    log.info(f"torch     = {torch.__version__}  cuda={torch.cuda.is_available()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
