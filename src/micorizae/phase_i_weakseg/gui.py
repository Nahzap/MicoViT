"""Interfaz PyQt5 por pestañas para WeakSeg y visualizacion de etiquetas."""

from __future__ import annotations

import logging
import os
import sys
import time
import tempfile
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None  # type: ignore[assignment]

from ..common.paths import get_paths
from .pipeline import WeakSegParams, run_weakseg_pipeline

try:
    from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("PyQt5 no instalado. Ejecuta: pip install PyQt5") from e


LOG = logging.getLogger("micorizae.gui")
LOG.setLevel(logging.DEBUG)
TORCH_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())

LAYER_COLORS = {
    "Mplus": (0, 200, 255),
    "Mminus": (210, 180, 140),
    "Background": (70, 70, 70),
    "Unreadable": (180, 100, 200),
    "AMColonised": (0, 200, 255),
    "Hybrid": (255, 130, 40),
    "DSE": (220, 80, 180),
    "Uncolonised": (210, 180, 140),
    "MainRoot": (170, 140, 120),
    "BlueCoils": (60, 120, 255),
    "BrownCoils": (150, 80, 40),
    "TypeTwo": (0, 220, 140),
    "HybridErm": (255, 160, 0),
    "HybridDse": (220, 80, 180),
}


def _qpix(path: Path) -> QPixmap:
    return QPixmap(str(path))


def _image_to_pixmap(img_rgb: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(img_rgb)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _shift_tensor_xy(tensor, dx: int, dy: int):
    if dx == 0 and dy == 0:
        return tensor
    if torch is None:
        raise RuntimeError("Torch no disponible para traslacion GPU")
    out = torch.zeros_like(tensor)
    h = tensor.shape[0]
    w = tensor.shape[1]
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return out
    if tensor.dim() == 2:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = tensor[src_y0:src_y1, src_x0:src_x1]
    else:
        out[dst_y0:dst_y1, dst_x0:dst_x1, ...] = tensor[src_y0:src_y1, src_x0:src_x1, ...]
    return out


def _cache_key(*, image_name: str, alpha: float, dx: int, dy: int, layers: list[str]) -> str:
    core = f"{image_name}|a={alpha:.3f}|dx={dx}|dy={dy}|layers={'|'.join(sorted(layers))}"
    return hashlib.sha1(core.encode("utf-8")).hexdigest()


class ZoomImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self._zoom = 0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("border: 1px solid #777; background: #1a1a1a;")

    def set_pixmap(self, pix: QPixmap, *, reset_view: bool = False) -> None:
        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        if reset_view:
            self.reset_zoom()

    def reset_zoom(self) -> None:
        self._zoom = 0
        self.resetTransform()
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def has_pixmap(self) -> bool:
        return not self._pix_item.pixmap().isNull()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._pix_item.pixmap().isNull():
            return
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        if self._zoom < -10:
            self._zoom = -10
            return
        if self._zoom > 40:
            self._zoom = 40
            return
        self.scale(factor, factor)


def _infer_tile_size(image_path: Path, df: pd.DataFrame, fallback: int = 126) -> int:
    if "row" not in df.columns or "col" not in df.columns or df.empty:
        return fallback
    Image.MAX_IMAGE_PIXELS = None
    w, h = Image.open(image_path).size
    n_rows = int(df["row"].max()) + 1
    n_cols = int(df["col"].max()) + 1
    if n_rows <= 0 or n_cols <= 0:
        return fallback
    ts = int(round((w / n_cols + h / n_rows) / 2.0))
    return ts if ts >= 16 else fallback


def _derive_stage1(df: pd.DataFrame) -> pd.Series:
    if "stage1" in df.columns:
        return df["stage1"].astype(str)
    bg = df["Background"].fillna(0).astype(int) if "Background" in df.columns else pd.Series(0, index=df.index)
    unr = df["Unreadable"].fillna(0).astype(int) if "Unreadable" in df.columns else pd.Series(0, index=df.index)
    mplus = pd.Series(0, index=df.index)
    for c in ("AMColonised", "Hybrid", "BlueCoils", "BrownCoils", "TypeTwo", "HybridErm", "HybridDse"):
        if c in df.columns:
            mplus += df[c].fillna(0).astype(int)
    mminus = pd.Series(0, index=df.index)
    for c in ("Uncolonised", "MainRoot", "DSE"):
        if c in df.columns:
            mminus += df[c].fillna(0).astype(int)
    out = np.full(len(df), "Background", dtype=object)
    out[mminus.to_numpy() > 0] = "Mminus"
    out[mplus.to_numpy() > 0] = "Mplus"
    out[bg.to_numpy() > 0] = "Background"
    out[unr.to_numpy() > 0] = "Unreadable"
    return pd.Series(out, index=df.index)


class QtLogHandler(logging.Handler):
    def __init__(self, sink: "LogsTab"):
        super().__init__(level=logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.sink.append_record(record.levelno, msg)


class LogsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.level_threshold = logging.DEBUG
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.level_combo = QComboBox()
        self.level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_combo.setCurrentText("DEBUG")
        self.level_combo.currentTextChanged.connect(self._on_level_changed)
        clear = QPushButton("Limpiar logs")
        clear.clicked.connect(self._clear)
        top.addWidget(QLabel("Nivel visible:"))
        top.addWidget(self.level_combo)
        top.addStretch(1)
        top.addWidget(clear)
        lay.addLayout(top)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text, stretch=1)

    def _on_level_changed(self, txt: str) -> None:
        self.level_threshold = getattr(logging, txt, logging.DEBUG)

    def _clear(self) -> None:
        self.text.clear()

    def append_record(self, levelno: int, msg: str) -> None:
        if levelno < self.level_threshold:
            return
        self.text.append(msg)


@dataclass
class GuiConfig:
    image_path: Path
    manifests_dir: Path
    annotations_path: Path | None
    params: WeakSegParams
    export_patches: bool
    patch_size: int
    patch_stride: int


class WeakSegWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: GuiConfig):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            LOG.info("Iniciando weakseg para imagen: %s", self.cfg.image_path)
            run = run_weakseg_pipeline(
                image_path=self.cfg.image_path,
                manifests_dir=self.cfg.manifests_dir,
                params=self.cfg.params,
                annotations_path=self.cfg.annotations_path,
                export_dino_patches=self.cfg.export_patches,
                patch_size=self.cfg.patch_size,
                patch_stride=self.cfg.patch_stride,
            )
            dt = time.perf_counter() - t0
            LOG.info("WeakSeg finalizado: %s (%.2fs)", run.root, dt)
            self.done.emit(str(run.root))
        except Exception as e:  # pragma: no cover
            LOG.exception("WeakSeg fallo")
            self.failed.emit(str(e))


class PipelineTab(QWidget):
    def __init__(self):
        super().__init__()
        paths = get_paths()
        self.worker: WeakSegWorker | None = None
        self.last_output_dir: Path | None = None
        main = QVBoxLayout(self)

        inputs = QGroupBox("Entrada Pipeline")
        form = QFormLayout(inputs)
        self.image_edit = QLineEdit(str(paths.root / "Data" / "am" / "am" / "train" / "10E_2L_E_Default_Extended.jpg"))
        self.manifests_edit = QLineEdit(str(paths.manifests))
        self.annotations_edit = QLineEdit("")
        b_img = QPushButton("Elegir imagen")
        b_img.clicked.connect(self._pick_image)
        b_man = QPushButton("Elegir manifests")
        b_man.clicked.connect(self._pick_manifests)
        b_ann = QPushButton("Elegir anotacion")
        b_ann.clicked.connect(self._pick_annotations)
        r1, r2, r3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        r1.addWidget(self.image_edit); r1.addWidget(b_img)
        r2.addWidget(self.manifests_edit); r2.addWidget(b_man)
        r3.addWidget(self.annotations_edit); r3.addWidget(b_ann)
        form.addRow("Imagen", r1); form.addRow("Manifests", r2); form.addRow("Anotacion", r3)
        main.addWidget(inputs)

        params = QGroupBox("Parametros")
        grid = QGridLayout(params)
        self.tile_size = self._spin_int(126, 64, 1024)
        self.canny_low = self._spin_int(40, 0, 255)
        self.canny_high = self._spin_int(110, 0, 255)
        self.frangi = self._spin_float(90.0, 50.0, 99.9, 0.1)
        self.arb = self._spin_float(97.0, 50.0, 99.9, 0.1)
        self.ves = self._spin_float(0.85, 0.1, 1.0, 0.01)
        self.seam = self._spin_float(0.35, 0.0, 3.0, 0.05)
        self.alpha = self._spin_float(0.30, 0.05, 0.95, 0.05)
        self.min_fg = self._spin_float(0.003, 0.0, 1.0, 0.001)
        self.export = QCheckBox("Exportar parches")
        self.export.setChecked(True)
        self.patch_size = self._spin_int(518, 64, 2048)
        self.patch_stride = self._spin_int(518, 64, 2048)
        rows = [
            ("Tile size", self.tile_size), ("Canny low", self.canny_low), ("Canny high", self.canny_high),
            ("Frangi pctl", self.frangi), ("Arbuscule pctl", self.arb), ("Vesicle circularity", self.ves),
            ("Seam sigma", self.seam), ("Overlay alpha", self.alpha), ("Min patch foreground", self.min_fg),
            ("Patch size", self.patch_size), ("Patch stride", self.patch_stride),
        ]
        for i, (name, w) in enumerate(rows):
            grid.addWidget(QLabel(name), i // 2, (i % 2) * 2)
            grid.addWidget(w, i // 2, (i % 2) * 2 + 1)
        grid.addWidget(self.export, 6, 0, 1, 2)
        main.addWidget(params)

        actions = QHBoxLayout()
        self.run_btn = QPushButton("Ejecutar")
        self.run_btn.clicked.connect(self._run)
        self.open_btn = QPushButton("Abrir salida")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output)
        self.status = QLabel("Listo.")
        actions.addWidget(self.run_btn); actions.addWidget(self.open_btn); actions.addWidget(self.status, 1)
        main.addLayout(actions)

        prev = QGridLayout()
        self.left = QLabel("Preview")
        self.right = QLabel("Comparacion")
        for lbl in (self.left, self.right):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(600, 320)
            lbl.setStyleSheet("border: 1px solid #777;")
        prev.addWidget(QLabel("Overlay"), 0, 0)
        prev.addWidget(QLabel("Panel"), 0, 1)
        prev.addWidget(self.left, 1, 0)
        prev.addWidget(self.right, 1, 1)
        main.addLayout(prev)

    def _spin_int(self, v: int, lo: int, hi: int) -> QSpinBox:
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(v); return s

    def _spin_float(self, v: float, lo: float, hi: float, step: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setValue(v); return s

    def _pick_image(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if p:
            self.image_edit.setText(p)
            LOG.info("Imagen seleccionada: %s", p)

    def _pick_manifests(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Seleccionar manifests")
        if p:
            self.manifests_edit.setText(p)
            LOG.info("Manifests seleccionado: %s", p)

    def _pick_annotations(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Seleccionar anotacion", "", "Annotations (*.csv *.xml)")
        if p:
            self.annotations_edit.setText(p)
            LOG.info("Anotacion seleccionada: %s", p)

    def _set_preview(self, label: QLabel, img_path: Path) -> None:
        pix = _qpix(img_path)
        if pix.isNull():
            label.setText(f"No se pudo abrir {img_path.name}")
            LOG.warning("Preview no disponible: %s", img_path)
            return
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _build_cfg(self) -> GuiConfig:
        image = Path(self.image_edit.text().strip())
        manifests = Path(self.manifests_edit.text().strip())
        ann = self.annotations_edit.text().strip()
        ann_path = Path(ann) if ann else None
        if not image.exists():
            raise ValueError(f"Imagen no encontrada: {image}")
        if not manifests.exists():
            raise ValueError(f"Manifests no encontrado: {manifests}")
        if ann_path is not None and not ann_path.exists():
            raise ValueError(f"Anotacion no encontrada: {ann_path}")
        params = WeakSegParams(
            tile_size=int(self.tile_size.value()),
            canny_low=int(self.canny_low.value()),
            canny_high=int(self.canny_high.value()),
            frangi_pctl=float(self.frangi.value()),
            vesicle_circularity_min=float(self.ves.value()),
            arbuscule_pctl=float(self.arb.value()),
            seam_sigma=float(self.seam.value()),
            alpha_overlay=float(self.alpha.value()),
            min_patch_fg_ratio=float(self.min_fg.value()),
        )
        return GuiConfig(
            image_path=image,
            manifests_dir=manifests,
            annotations_path=ann_path,
            params=params,
            export_patches=self.export.isChecked(),
            patch_size=int(self.patch_size.value()),
            patch_stride=int(self.patch_stride.value()),
        )

    def _run(self) -> None:
        try:
            cfg = self._build_cfg()
        except Exception as e:
            QMessageBox.critical(self, "Configuracion invalida", str(e))
            LOG.error("Configuracion invalida: %s", e)
            return
        self.run_btn.setEnabled(False)
        self.status.setText("Ejecutando...")
        self.worker = WeakSegWorker(cfg)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, out: str) -> None:
        self.run_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.last_output_dir = Path(out)
        stem = Path(self.image_edit.text().strip()).stem
        self.status.setText(f"OK: {out}")
        self._set_preview(self.left, self.last_output_dir / "maps" / f"{stem}__overlay_alpha.png")
        self._set_preview(self.right, self.last_output_dir / "maps" / f"{stem}__comparison_panel.png")

    def _failed(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText("Fallo.")
        QMessageBox.critical(self, "Error", err)
        LOG.error("Pipeline fallo: %s", err)

    def _open_output(self) -> None:
        if self.last_output_dir is None:
            return
        os.startfile(str(self.last_output_dir))  # type: ignore[attr-defined]


class LabelViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        paths = get_paths()
        self.image_rgb: np.ndarray | None = None
        self.image_view_rgb: np.ndarray | None = None
        self.ann_df: pd.DataFrame | None = None
        self.tile_size = 126
        self.layer_names: list[str] = []
        self.layer_grids: dict[str, np.ndarray] = {}
        self.grid_shape: tuple[int, int] = (0, 0)
        self.layer_grids_gpu: dict[str, "torch.Tensor"] = {}
        self.base_image_path: Path | None = None
        self._cache_dir_ctx = tempfile.TemporaryDirectory(prefix="micorizae_label_viewer_")
        self.cache_dir = Path(self._cache_dir_ctx.name)
        self.renders_dir = self.cache_dir / "renders"
        self.layers_dir = self.cache_dir / "layers"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self.render_cache_index: OrderedDict[str, Path] = OrderedDict()
        self.max_cached_renders = 18
        self.base_gpu_f16 = None
        self.x_coords_gpu = None
        self.gpu_device = "cuda"
        self.last_render_ms = 0.0
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render)

        main = QVBoxLayout(self)
        src = QGroupBox("Visualizador de etiquetas")
        form = QFormLayout(src)
        self.image_edit = QLineEdit(str(paths.root / "Data" / "am" / "am" / "train" / "10E_2L_E_Default_Extended.jpg"))
        self.ann_edit = QLineEdit("")
        b_img, b_ann = QPushButton("Elegir imagen"), QPushButton("Elegir anotacion")
        b_img.clicked.connect(self._pick_image); b_ann.clicked.connect(self._pick_ann)
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        r1.addWidget(self.image_edit); r1.addWidget(b_img)
        r2.addWidget(self.ann_edit); r2.addWidget(b_ann)
        form.addRow("Imagen", r1)
        form.addRow("Anotacion CSV/XML", r2)
        main.addWidget(src)

        controls = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(35)
        self.alpha_slider.valueChanged.connect(self._alpha_changed)
        self.alpha_label = QLabel("Transparencia: 0.35")
        self.load_btn = QPushButton("Cargar y visualizar")
        self.load_btn.clicked.connect(self._load_and_render)
        controls.addWidget(self.alpha_label)
        controls.addWidget(self.alpha_slider, 1)
        controls.addWidget(self.load_btn)
        main.addLayout(controls)

        align = QHBoxLayout()
        self.shift_x = QSpinBox()
        self.shift_x.setRange(-3000, 3000)
        self.shift_x.setValue(0)
        self.shift_x.valueChanged.connect(self._on_shift_changed)
        self.shift_y = QSpinBox()
        self.shift_y.setRange(-3000, 3000)
        self.shift_y.setValue(0)
        self.shift_y.valueChanged.connect(self._on_shift_changed)
        self.reset_shift_btn = QPushButton("Reset traslacion")
        self.reset_shift_btn.clicked.connect(self._reset_shift)
        self.shift_label = QLabel("Offset (x,y): (0, 0)")
        align.addWidget(QLabel("Traslacion X(px):"))
        align.addWidget(self.shift_x)
        align.addWidget(QLabel("Traslacion Y(px):"))
        align.addWidget(self.shift_y)
        align.addWidget(self.reset_shift_btn)
        align.addWidget(self.shift_label, 1)
        main.addLayout(align)

        layer_box = QGroupBox("Capas de anotacion (marcar para mostrar)")
        layer_lay = QVBoxLayout(layer_box)
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QListWidget.NoSelection)
        self.layer_list.itemChanged.connect(lambda _item: self._schedule_render())
        layer_lay.addWidget(self.layer_list)
        main.addWidget(layer_box)

        self.canvas = ZoomImageView()
        self.canvas.setMinimumSize(1100, 520)
        main.addWidget(self.canvas, stretch=1)

    def _pick_image(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if p:
            self.image_edit.setText(p)

    def _pick_ann(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Seleccionar anotacion", "", "Annotations (*.csv *.xml)")
        if p:
            self.ann_edit.setText(p)

    def _alpha_changed(self) -> None:
        a = self.alpha_slider.value() / 100.0
        self.alpha_label.setText(f"Transparencia: {a:.2f}")
        self._schedule_render()

    def _on_shift_changed(self) -> None:
        self.shift_label.setText(f"Offset (x,y): ({self.shift_x.value()}, {self.shift_y.value()})")
        self._schedule_render()

    def _reset_shift(self) -> None:
        self.shift_x.setValue(0)
        self.shift_y.setValue(0)
        self.shift_label.setText("Offset (x,y): (0, 0)")
        self._schedule_render(0)
        LOG.info("Alineacion: traslacion reseteada a (0,0)")

    def _schedule_render(self, delay_ms: int = 70) -> None:
        self._render_timer.start(delay_ms)

    def _load_and_render(self) -> None:
        try:
            t0 = time.perf_counter()
            img_path = Path(self.image_edit.text().strip())
            ann_path = Path(self.ann_edit.text().strip())
            if not img_path.exists():
                raise ValueError(f"Imagen no encontrada: {img_path}")
            if not ann_path.exists():
                raise ValueError(f"Anotacion no encontrada: {ann_path}")
            Image.MAX_IMAGE_PIXELS = None
            self.image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
            self.ann_df = pd.read_csv(ann_path) if ann_path.suffix.lower() == ".csv" else pd.DataFrame()
            if self.ann_df.empty:
                raise ValueError("Solo se soporta visualizador CSV por ahora.")
            if "row" not in self.ann_df.columns or "col" not in self.ann_df.columns:
                raise ValueError("CSV sin columnas row/col.")
            self.tile_size = _infer_tile_size(img_path, self.ann_df, fallback=126)
            self._build_view_cache()
            self._populate_layers()
            self.canvas.reset_zoom()
            dt = time.perf_counter() - t0
            LOG.info(
                "Viewer cargado (GPU-only): %s (%d filas, tile_size=%d, native_res=%dx%d, %.2fs)",
                img_path.name, len(self.ann_df), self.tile_size, self.image_view_rgb.shape[1], self.image_view_rgb.shape[0], dt
            )
            self._schedule_render(0)
        except Exception as e:
            QMessageBox.critical(self, "Error viewer", str(e))
            LOG.error("Error viewer: %s", e)

    def _build_view_cache(self) -> None:
        assert self.image_rgb is not None and self.ann_df is not None
        if not TORCH_CUDA_AVAILABLE or torch is None or F is None:
            raise RuntimeError("GPU/CUDA obligatoria no disponible para LABEL VIEWER.")
        self.layer_grids_gpu.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Sin downscale: viewer a resolucion nativa.
        self.image_view_rgb = self.image_rgb.copy()
        self.render_cache_index.clear()
        for p in self.renders_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass

        # Cache de imagen base en disco (memmap) para acceso estable en datasets grandes.
        self.base_image_path = self.cache_dir / "base_image.npy"
        np.save(self.base_image_path, self.image_view_rgb)
        self.image_view_rgb = np.load(self.base_image_path, mmap_mode="r")  # type: ignore[assignment]

        stage1 = _derive_stage1(self.ann_df)
        labels_per_row: list[set[str]] = []
        dynamic_cols = [c for c in self.ann_df.columns if c not in {"row", "col"} and pd.api.types.is_numeric_dtype(self.ann_df[c])]
        all_layers: set[str] = {"Mplus", "Mminus", "Background", "Unreadable"}
        n_rows = int(self.ann_df["row"].max()) + 1
        n_cols = int(self.ann_df["col"].max()) + 1
        self.grid_shape = (n_rows, n_cols)
        for i, rec in enumerate(self.ann_df.itertuples(index=False)):
            active: set[str] = set()
            active.add(str(stage1.iloc[i]))
            for c in dynamic_cols:
                val = getattr(rec, c, 0)
                try:
                    if float(val) > 0:
                        active.add(c)
                except Exception:
                    pass
            labels_per_row.append(active)
            all_layers.update(active)

        self.layer_names = sorted(all_layers)
        self.layer_grids = {name: np.zeros((n_rows, n_cols), dtype=np.uint8) for name in self.layer_names}
        for i, rec in enumerate(self.ann_df.itertuples(index=False)):
            r, c = int(rec.row), int(rec.col)
            if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
                continue
            for layer in labels_per_row[i]:
                if layer in self.layer_grids:
                    self.layer_grids[layer][r, c] = 1
        for name, grid in self.layer_grids.items():
            np.save(self.layers_dir / f"{name}.npy", grid)
        # Pre-carga a GPU: sin fallback CPU.
        self.layer_grids_gpu = {
            name: torch.from_numpy(grid.astype(np.uint8)).to(device=self.gpu_device, dtype=torch.bool)
            for name, grid in self.layer_grids.items()
        }
        # Mantener la imagen base en CPU-memmap para no saturar VRAM.
        self.base_gpu_f16 = None
        h, w = self.image_view_rgb.shape[:2]
        self.x_coords_gpu = torch.arange(w, device=self.gpu_device, dtype=torch.int32)

    def cleanup_cache(self) -> None:
        try:
            self.layer_grids_gpu.clear()
            self.render_cache_index.clear()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._cache_dir_ctx.cleanup()
        except Exception:
            pass

    def _populate_layers(self) -> None:
        if not self.layer_names:
            return
        self.layer_list.clear()
        defaults = {"Mplus", "AMColonised", "Hybrid"}
        for name in self.layer_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in defaults else Qt.Unchecked)
            self.layer_list.addItem(item)
        LOG.debug("Viewer capas cargadas: %d", len(self.layer_names))

    def _selected_layers(self) -> list[str]:
        out: list[str] = []
        for i in range(self.layer_list.count()):
            it = self.layer_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _render(self) -> None:
        if self.image_view_rgb is None or not self.layer_grids_gpu:
            return
        selected = self._selected_layers()
        if not selected:
            LOG.warning("Viewer: no hay capas seleccionadas.")
            return
        if not TORCH_CUDA_AVAILABLE or torch is None or F is None or self.base_gpu_f16 is None:
            # GPU-only: render solo en CUDA, pero base se transmite por chunks desde memmap CPU.
            if not TORCH_CUDA_AVAILABLE or torch is None or F is None:
                raise RuntimeError("GPU/CUDA obligatoria no disponible para render.")
        t0 = time.perf_counter()
        try:
            alpha = float(self.alpha_slider.value()) / 100.0
            h, w = self.image_view_rgb.shape[:2]
            n_rows, n_cols = self.grid_shape
            dx = int(self.shift_x.value())
            dy = int(self.shift_y.value())
            img_name = Path(self.image_edit.text().strip()).name
            ckey = _cache_key(image_name=img_name, alpha=alpha, dx=dx, dy=dy, layers=selected)
            cached_png = self.renders_dir / f"{ckey}.png"
            if cached_png.exists():
                self.canvas.set_pixmap(_qpix(cached_png), reset_view=not self.canvas.has_pixmap())
                dt = time.perf_counter() - t0
                self.last_render_ms = dt * 1000.0
                LOG.debug(
                    "Viewer render GPU cache HIT: alpha=%.2f shift=(%d,%d) capas=%s (%.1fms)",
                    alpha, dx, dy, ",".join(selected), self.last_render_ms
                )
                self.render_cache_index[ckey] = cached_png
                self.render_cache_index.move_to_end(ckey)
                return

            overlay_acc_grid = torch.zeros((n_rows, n_cols, 3), device=self.gpu_device, dtype=torch.float16)
            weight_grid = torch.zeros((n_rows, n_cols), device=self.gpu_device, dtype=torch.float16)

            for layer in selected:
                grid = self.layer_grids_gpu.get(layer)
                if grid is None:
                    continue
                mask = grid
                if not bool(mask.any()):
                    continue
                color = torch.tensor(LAYER_COLORS.get(layer, (120, 120, 120)), device=self.gpu_device, dtype=torch.float16)
                overlay_acc_grid[mask] += color
                weight_grid[mask] += 1.0

            mask_grid = weight_grid > 0
            if not bool(mask_grid.any()):
                LOG.warning("Viewer render: sin pixeles activos para capas seleccionadas.")
                return

            overlay_grid = torch.zeros_like(overlay_acc_grid)
            overlay_grid[mask_grid] = overlay_acc_grid[mask_grid] / weight_grid[mask_grid].unsqueeze(-1)

            # Render chunked en GPU para evitar OOM en imágenes ultra grandes.
            out = np.zeros((h, w, 3), dtype=np.uint8)
            chunk_h = 256
            x_all = self.x_coords_gpu
            if x_all is None:
                x_all = torch.arange(w, device=self.gpu_device, dtype=torch.int32)
                self.x_coords_gpu = x_all
            for y0 in range(0, h, chunk_h):
                y1 = min(h, y0 + chunk_h)
                y = torch.arange(y0, y1, device=self.gpu_device, dtype=torch.int32)[:, None]
                src_y = y - dy
                src_x = x_all[None, :] - dx
                valid = (src_y >= 0) & (src_y < h) & (src_x >= 0) & (src_x < w)

                gy = torch.clamp((src_y * n_rows) // h, 0, n_rows - 1).to(torch.long)
                gx = torch.clamp((src_x * n_cols) // w, 0, n_cols - 1).to(torch.long)
                overlay_chunk = overlay_grid[gy, gx]  # [ch,w,3]
                w_chunk = weight_grid[gy, gx]
                mask_chunk = (w_chunk > 0) & valid

                base_np = np.asarray(self.image_view_rgb[y0:y1, :, :], dtype=np.float16)
                base_chunk = torch.from_numpy(base_np).to(device=self.gpu_device)
                mixed_chunk = base_chunk.clone()
                if bool(mask_chunk.any()):
                    mixed_chunk[mask_chunk] = base_chunk[mask_chunk] * (1.0 - alpha) + overlay_chunk[mask_chunk] * alpha
                out[y0:y1, :, :] = mixed_chunk.clamp(0, 255).to(torch.uint8).cpu().numpy()

            self.canvas.set_pixmap(_image_to_pixmap(out), reset_view=not self.canvas.has_pixmap())
            Image.fromarray(out).save(cached_png)
            self.render_cache_index[ckey] = cached_png
            self.render_cache_index.move_to_end(ckey)
            while len(self.render_cache_index) > self.max_cached_renders:
                old_k, old_p = self.render_cache_index.popitem(last=False)
                try:
                    old_p.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            dt = time.perf_counter() - t0
            self.last_render_ms = dt * 1000.0
            LOG.debug(
                "Viewer render GPU: alpha=%.2f shift=(%d,%d) capas=%s (%.1fms)",
                alpha, dx, dy, ",".join(selected), self.last_render_ms
            )
            if torch.cuda.is_available():
                LOG.debug(
                    "GPU mem: alloc=%.2fGB reserved=%.2fGB",
                    torch.cuda.memory_allocated() / (1024**3),
                    torch.cuda.memory_reserved() / (1024**3),
                )
        except Exception as e:
            LOG.error("Render GPU fallo (sin fallback CPU): %s", e)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise


class WeakSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicorizaeVision - GUI Principal")
        self.resize(1380, 920)
        root = QWidget()
        lay = QVBoxLayout(root)
        self.tabs = QTabWidget()
        self.viewer_tab = LabelViewerTab()
        self.tabs.addTab(self.viewer_tab, "LABEL VIEWER")
        lay.addWidget(self.tabs)
        self.setCentralWidget(root)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.viewer_tab.cleanup_cache()
        super().closeEvent(event)


def launch_weakseg_gui() -> None:
    if not TORCH_CUDA_AVAILABLE or torch is None:
        raise SystemExit("LABEL VIEWER requiere GPU CUDA (modo GPU-only, sin fallback CPU).")
    app = QApplication(sys.argv)
    win = WeakSegWindow()
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.handlers.clear()
    LOG.addHandler(stream)
    LOG.propagate = False
    pkg = logging.getLogger("micorizae")
    pkg.setLevel(logging.DEBUG)
    pkg.handlers.clear()
    pkg.addHandler(stream)
    pkg.propagate = False
    LOG.info("GUI iniciada con unica tab: LABEL VIEWER.")
    win.show()
    app.exec_()

