"""Native, lazy results views; importing this module never loads the pipeline.

Disk reads and numerical preparation belong to the thread pool. Only the GUI
thread constructs widgets, chart canvases and pixmaps.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, QObject, QRunnable, QRectF, QSize,
    QThreadPool, Qt, QUrl, Signal, Slot,
)
from PySide6.QtGui import QColor, QDesktopServices, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHeaderView, QHBoxLayout, QLabel, QLayout, QPushButton, QScrollArea, QSplitter,
    QTableView, QTabWidget, QVBoxLayout, QWidget,
)

EM_DASH = "\N{EM DASH}"
DISPLAY_BUDGET = 64 * 1024 * 1024
# Leave half the decoded display budget for the one active QPixmap.
IMAGE_BUDGET = DISPLAY_BUDGET // 2
BOOLEAN_COLUMNS = {"hollow", "in_stats", "truncated", "touches_exclusion"}
INTEGER_COLUMNS = {"particle_id", "aggregate_id", "n_spots", "area_px"}
PARTICLE_COLUMNS = (
    "particle_id", "panel_id", "paper_id", "d_px", "d_nm", "feret_max_nm",
    "aspect", "area_nm2", "cavity_d_nm", "shell_nm", "core_d_nm", "hollow",
    "truncated", "touches_exclusion", "aggregate_id", "source", "rule_id",
    "flags", "in_stats", "area_px", "feret_max_px", "cavity_d_px", "shell_px",
    "core_d_px", "n_spots",
)


@dataclass(frozen=True)
class PanelRecord:
    panel_id: str
    name: str
    summary: dict
    base_path: Path
    ids_path: Path | None
    source_size: tuple[int, int]
    extent: tuple[float, float]
    rows: tuple[int, ...]


@dataclass(frozen=True)
class RunData:
    folder: Path
    particles: Any  # A single shared pandas DataFrame, imported in the worker.
    panels: tuple[PanelRecord, ...]


@dataclass(frozen=True)
class PanelDisplay:
    panel: PanelRecord
    image: QImage
    counts: tuple[int, ...]
    edges: tuple[float, ...]
    sample_count: int
    unit: str
    warnings: tuple[str, ...]


def _image_size(path: Path) -> tuple[int, int]:
    reader = QImageReader(str(path))
    size = reader.size()
    if not size.isValid():
        raise ValueError(f"Cannot read image {path.name}: {reader.errorString()}")
    return size.width(), size.height()


def _read_run(folder: Path) -> RunData:
    """Read only headers/metadata until an individual panel is requested."""
    import pandas as pd

    folder = folder.resolve()
    measure = folder / "measure"
    # Also support review folders produced directly by the existing pipeline.
    if not measure.is_dir():
        measure = folder
    all_csv = folder / "particles_all.csv"
    if all_csv.is_file():
        particles = pd.read_csv(all_csv)
    else:
        csvs = sorted(measure.glob("*__particles.csv"))
        frames = [pd.read_csv(path) for path in csvs]
        particles = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PARTICLE_COLUMNS)
    for column in BOOLEAN_COLUMNS & set(particles.columns):
        particles[column] = particles[column].map(_bool_value)
    groups: dict[str, tuple[int, ...]] = {}
    if "panel_id" in particles:
        groups = {
            str(panel_id): tuple(int(i) for i in indices)
            for panel_id, indices in particles.groupby("panel_id", sort=False).groups.items()
        }
    manifest = {}
    manifest_path = folder / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest = {row["panel_id"]: row for row in csv.DictReader(handle)}
    image_metadata = {}
    run_info_path = folder / "run_info.json"
    if run_info_path.is_file():
        with run_info_path.open(encoding="utf-8") as handle:
            image_metadata = {str(image["panel_id"]): image for image in json.load(handle).get("images", [])}
    panels = []
    for path in sorted(measure.glob("*__panel.json")):
        panel_id = path.name.removesuffix("__panel.json")
        base = measure / f"{panel_id}__overlay.png"
        if not base.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        base_w, base_h = _image_size(base)
        source = (base_w, base_h)
        metadata = folder / "cleanup" / f"{panel_id}.json"
        if metadata.is_file():
            with metadata.open(encoding="utf-8") as handle:
                cleanup = json.load(handle)
            source = (int(cleanup["width"]), int(cleanup["height"]))
        entry = manifest.get(panel_id, {})
        source_path = Path(entry.get("image_path", ""))
        if not source_path.is_absolute():
            source_path = folder / source_path
        if not metadata.is_file() and source_path.is_file():
            source = _image_size(source_path)
        recorded = image_metadata.get(panel_id, {})
        if not metadata.is_file() and not source_path.is_file() and recorded.get("width") and recorded.get("height"):
            source = (int(recorded["width"]), int(recorded["height"]))
        name = Path(entry["image_path"]).name if entry.get("image_path") else recorded.get("filename", panel_id)
        ids = measure / f"{panel_id}__overlay_ids.png"
        ids = ids if ids.is_file() else None
        base_scale = base_w / source[0]
        extent_height = base_h / base_scale
        if ids is not None:
            ids_w, ids_h = _image_size(ids)
            extent_height = max(extent_height, ids_h / (ids_w / source[0]))
        panels.append(PanelRecord(
            panel_id, name, summary, base, ids, source,
            (float(source[0]), max(float(source[1]), extent_height)),
            groups.get(panel_id, ()),
        ))
    if not panels:
        raise ValueError("No completed image results were found in this folder.")
    return RunData(folder, particles, tuple(panels))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        # pandas' nullable scalars deliberately refuse a boolean conversion.
        return type(value).__name__ in {"NAType", "NaTType"}


def _bool_value(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return {"true": True, "false": False, "1": True, "0": False,
                "yes": True, "no": False}.get(value.strip().lower())
    return bool(value)


def _unit(summary: dict) -> str:
    scale = summary.get("nm_per_px")
    return "nm" if scale is not None and math.isfinite(float(scale)) and float(scale) > 0 else "px"


def _warnings(panel: PanelRecord, frame: Any) -> tuple[str, ...]:
    summary = panel.summary
    flags = set(summary.get("flags", []))
    if "flags" in frame:
        for row in panel.rows:
            value = frame.at[row, "flags"]
            if not _is_missing(value):
                flags.update(str(value).split(";"))
    messages = []
    if _unit(summary) == "px":
        messages.append("No scale bar found. Sizes are in pixels. Enter nm per pixel to get nm.")
    if summary.get("stats_all") is None:
        messages.append("Too few particles for statistics on this image.")
    if "FRAME_PARTICLE" in flags:
        messages.append("A particle fills most of the image; check the result.")
    if "POSSIBLE_CORE_ONLY" in flags:
        messages.append("Some particles may be detected by their core only.")
    if summary.get("n_particles") == 0:
        messages.append("No particles detected in this image.")
    width, height = panel.source_size
    if max(width, height) > 1024:
        messages.append(
            f"This image is large ({width}\N{MULTIPLICATION SIGN}{height}). SAM shrinks it to 1024 px, "
            "so small particles may be missed. For best results, crop to the area you care about."
        )
    return tuple(messages)


class DecodedImageCache:
    """Worker-safe LRU; RGBA bytes, rather than PNG file sizes, count here."""

    def __init__(self, budget: int = IMAGE_BUDGET):
        self.budget = budget
        self.bytes_used = 0
        self._images: OrderedDict[Path, QImage] = OrderedDict()
        self._lock = Lock()

    def read(self, path: Path) -> QImage:
        with self._lock:
            if path in self._images:
                self._images.move_to_end(path)
                return self._images[path]
            reader = QImageReader(str(path))
            size = reader.size()
            if not size.isValid():
                raise ValueError(f"Cannot read {path.name}: {reader.errorString()}")
            ratio = min(1.0, math.sqrt(self.budget / (size.width() * size.height() * 4)))
            if ratio < 1:
                reader.setScaledSize(QSize(max(1, int(size.width() * ratio)), max(1, int(size.height() * ratio))))
            image = reader.read()
            if image.isNull():
                raise ValueError(f"Cannot read {path.name}: {reader.errorString()}")
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
            # Some image plugins do not honour scaled decoding.
            if image.sizeInBytes() > self.budget:
                image = image.scaled(
                    max(1, int(size.width() * ratio)), max(1, int(size.height() * ratio)),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
                )
            cost = image.sizeInBytes()
            while self._images and self.bytes_used + cost > self.budget:
                _, old = self._images.popitem(last=False)
                self.bytes_used -= old.sizeInBytes()
            self._images[path] = image
            self.bytes_used += cost
            return image

    def clear(self) -> None:
        with self._lock:
            self._images.clear()
            self.bytes_used = 0


def _prepare_panel(run: RunData, panel: PanelRecord, cache: DecodedImageCache) -> PanelDisplay:
    import numpy as np

    unit = _unit(panel.summary)
    frame = run.particles
    values = []
    column = f"d_{unit}"
    if column in frame and "in_stats" in frame:
        for row in panel.rows:
            value = frame.at[row, column]
            if _bool_value(frame.at[row, "in_stats"]) and not _is_missing(value) and math.isfinite(float(value)):
                values.append(float(value))
    if values:
        edges = np.histogram_bin_edges(values, bins="auto")
        if len(edges) > 41:
            edges = np.linspace(edges[0], edges[-1], 41)
        counts, edges = np.histogram(values, bins=edges)
        counts, edges = tuple(int(n) for n in counts), tuple(float(n) for n in edges)
    else:
        counts, edges = (), ()
    return PanelDisplay(panel, cache.read(panel.base_path), counts, edges, len(values), unit, _warnings(panel, frame))


class _TaskSignals(QObject):
    result = Signal(object, object)
    failed = Signal(object, str)


class _Task(QRunnable):
    def __init__(self, token: object, work: Callable[[], Any]):
        super().__init__()
        self.token = token
        self.work = work
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            self.signals.result.emit(self.token, self.work())
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc))


class ParticleTableModel(QAbstractTableModel):
    """Indexes the shared raw frame without constructing formatted cell tables."""

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.frame = None
        self.rows: list[int] = []
        self.columns = list(PARTICLE_COLUMNS)
        self.unit = "px"

    def set_panel(self, frame: Any, panel: PanelRecord | None) -> None:
        self.beginResetModel()
        self.frame = frame
        self.rows = list(panel.rows) if panel else []
        self.unit = _unit(panel.summary) if panel else "px"
        first = ["particle_id", f"d_{self.unit}", "hollow", "in_stats",
                 "truncated", "touches_exclusion", "aggregate_id"]
        available = list(frame.columns) if frame is not None else list(PARTICLE_COLUMNS)
        self.columns = first + [column for column in available if column not in first]
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def raw_value(self, row: int, column: int) -> Any:
        name = self.columns[column]
        if self.frame is None or name not in self.frame.columns:
            return None
        return self.frame.at[self.rows[row], name]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        value = self.raw_value(index.row(), index.column())
        name = self.columns[index.column()]
        missing = _is_missing(value)
        if role == Qt.ItemDataRole.UserRole:
            return None if missing else value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            # Booleans display as Yes/No text; pandas stores them as numpy.bool_.
            numeric = name not in BOOLEAN_COLUMNS and (
                name in INTEGER_COLUMNS or (not missing and not isinstance(value, (str, bool))))
            return int((Qt.AlignmentFlag.AlignRight if numeric else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.DisplayRole:
            if missing:
                return EM_DASH
            if name in BOOLEAN_COLUMNS:
                return "Yes" if _bool_value(value) else "No"
            if name in INTEGER_COLUMNS:
                return str(int(value))
            if not isinstance(value, (str, bool)):
                return f"{float(value):.2f}"
            return str(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Vertical:
            return section + 1
        name = self.columns[section]
        labels = {"particle_id": "ID", "hollow": "Hollow", "in_stats": "In statistics",
                  "truncated": "Truncated", "touches_exclusion": "Touches exclusion",
                  "aggregate_id": "Aggregate ID", f"d_{self.unit}": f"Diameter ({self.unit})"}
        return labels.get(name, name)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if not self.rows or column >= len(self.columns):
            return
        name = self.columns[column]
        if name not in self.frame.columns:
            return
        persistent = self.persistentIndexList()
        identities = [(self.rows[index.row()], index.column()) for index in persistent]
        self.layoutAboutToBeChanged.emit()
        def key(row: int):
            value = self.frame.at[row, name]
            missing = _is_missing(value)
            return (missing, None if missing else value)
        # Missing values remain last in both directions.
        nonnull = [row for row in self.rows if not key(row)[0]]
        null = [row for row in self.rows if key(row)[0]]
        nonnull.sort(key=lambda row: key(row)[1], reverse=order == Qt.SortOrder.DescendingOrder)
        self.rows[:] = nonnull + null
        positions = {row: position for position, row in enumerate(self.rows)}
        self.changePersistentIndexList(persistent, [self.index(positions[row], column) for row, column in identities])
        self.layoutChanged.emit()


class FrozenParticleTable(QTableView):
    """Keep ID visible using a native view of the same data and selection."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.frozen_column = QTableView(self)
        self.frozen_column.setAccessibleName("Particle IDs")
        self.frozen_column.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frozen_column.setFrameShape(QTableView.Shape.NoFrame)
        self.frozen_column.setStyleSheet("QTableView { border: 0; }")
        self.frozen_column.setShowGrid(False)
        self.frozen_column.verticalHeader().hide()
        self.frozen_column.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.frozen_column.horizontalHeader().setSortIndicatorShown(True)
        self.frozen_column.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_column.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_column.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.frozen_column.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self.frozen_column.setVerticalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self.horizontalHeader().sectionResized.connect(self._column_resized)
        self.frozen_column.horizontalHeader().sectionResized.connect(self._frozen_column_resized)
        self.verticalHeader().sectionResized.connect(self._row_resized)
        self.horizontalHeader().sortIndicatorChanged.connect(self._sort_indicator_changed)
        self.frozen_column.horizontalHeader().sectionClicked.connect(self._sort_id)
        self.verticalScrollBar().valueChanged.connect(self.frozen_column.verticalScrollBar().setValue)
        self.frozen_column.verticalScrollBar().valueChanged.connect(self.verticalScrollBar().setValue)
        self.frozen_column.clicked.connect(lambda: self.setFocus(Qt.FocusReason.MouseFocusReason))

    def setModel(self, model) -> None:
        super().setModel(model)
        self.frozen_column.setModel(model)
        self.frozen_column.setSelectionModel(self.selectionModel())
        model.modelReset.connect(self._refresh_frozen_column)
        self._refresh_frozen_column()

    def _refresh_frozen_column(self) -> None:
        if self.model() is None:
            return
        for column in range(1, self.model().columnCount()):
            self.frozen_column.setColumnHidden(column, True)
        self.frozen_column.setColumnWidth(0, self.columnWidth(0))
        self.frozen_column.verticalHeader().setDefaultSectionSize(self.verticalHeader().defaultSectionSize())
        self._update_frozen_geometry()
        self.frozen_column.raise_()

    def _column_resized(self, column: int, old_size: int, size: int) -> None:
        if column == 0:
            self.frozen_column.setColumnWidth(0, size)
            self._update_frozen_geometry()

    def _row_resized(self, row: int, old_size: int, size: int) -> None:
        self.frozen_column.setRowHeight(row, size)

    def _frozen_column_resized(self, column: int, old_size: int, size: int) -> None:
        if column == 0:
            self.setColumnWidth(0, size)

    def _sort_indicator_changed(self, column: int, order: Qt.SortOrder) -> None:
        self.frozen_column.horizontalHeader().setSortIndicator(0 if column == 0 else -1, order)

    def _sort_id(self, column: int) -> None:
        header = self.horizontalHeader()
        descending = header.sortIndicatorSection() == 0 and header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
        self.sortByColumn(0, Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder)

    def _update_frozen_geometry(self) -> None:
        header_width = self.verticalHeader().width() if self.verticalHeader().isVisible() else 0
        self.frozen_column.setGeometry(
            header_width + self.frameWidth(), self.frameWidth(), self.columnWidth(0),
            self.viewport().height() + self.horizontalHeader().height(),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "frozen_column"):
            self._update_frozen_geometry()

    def scrollTo(self, index: QModelIndex, hint=QTableView.ScrollHint.EnsureVisible) -> None:
        if index.column() == 0:
            self.frozen_column.scrollTo(index, hint)
        else:
            super().scrollTo(index, hint)
            self._reveal_after_frozen_column(index)

    def moveCursor(self, action, modifiers) -> QModelIndex:
        index = super().moveCursor(action, modifiers)
        if hasattr(self, "frozen_column"):
            self._reveal_after_frozen_column(index)
        return index

    def _reveal_after_frozen_column(self, index: QModelIndex) -> None:
        if index.isValid() and index.column() > 0:
            left = self.visualRect(index).left()
            width = self.columnWidth(0)
            if left < width:
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + left - width)


class RasterView(QGraphicsView):
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QColor("#0B0B0D"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumHeight(220)
        self.setAccessibleName("Result image")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._fit = True
        self._has_image = False

    def display(self, image: QImage, panel: PanelRecord, fit: bool) -> None:
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(1.0)
        self.item.setPixmap(pixmap)
        self.item.setScale(panel.source_size[0] / image.width())
        self.item.setPos(0, 0)
        self.scene().setSceneRect(QRectF(0, 0, *panel.extent))
        self._has_image = True
        self.setAccessibleDescription(
            f"{panel.name}, {panel.source_size[0]} by {panel.source_size[1]} pixels, "
            f"{panel.summary.get('n_particles', 0)} particles"
        )
        if fit:
            self.fit_image()

    def clear_image(self) -> None:
        self.item.setPixmap(QPixmap())
        self._has_image = False

    def fit_image(self) -> None:
        if self._has_image:
            self._fit = True
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(self.sceneRect().center())
            self.zoom_changed.emit(self.transform().m11())

    def actual_size(self) -> None:
        if self._has_image:
            self._fit = False
            self.resetTransform()
            self.centerOn(self.sceneRect().center())
            self.zoom_changed.emit(1.0)

    def wheelEvent(self, event) -> None:
        if not self._has_image or not event.angleDelta().y():
            return super().wheelEvent(event)
        self._fit = False
        factor = 1.15 ** (event.angleDelta().y() / 120)
        target = min(32.0, max(0.005, self.transform().m11() * factor))
        self.scale(target / self.transform().m11(), target / self.transform().m11())
        self.zoom_changed.emit(target)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit:
            self.fit_image()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.fit_image()
            event.accept()
        else:
            super().keyPressEvent(event)


class ResultsWidget(QWidget):
    ready = Signal()
    error = Signal(str)
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.current_run: Path | None = None
        self.run_data: RunData | None = None
        self.current_display: PanelDisplay | None = None
        self.cache = DecodedImageCache()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self._tasks: dict[object, _Task] = {}
        self._sequence = 0
        self._run_token = None
        self._panel_token = None
        self._ids_token = None
        self._busy = False
        self._closed = False
        self._canvas = None
        self._figure = None
        self._axes = None
        self._plotted = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        title = QLabel("Results")
        title.setObjectName("paneHeading")
        title.setProperty("role", "heading")
        self.run_label = QLabel("No completed run")
        self.run_label.setObjectName("secondaryText")
        self.run_label.setTextFormat(Qt.TextFormat.PlainText)
        self.run_label.setProperty("role", "muted")
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.run_label)
        layout.addLayout(heading)
        toolbar = QHBoxLayout()
        image_label = QLabel("Image")
        self.image_picker = QComboBox()
        self.image_picker.setAccessibleName("Processed image")
        self.image_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.image_picker.setMinimumContentsLength(12)
        image_label.setBuddy(self.image_picker)
        self.ids_checkbox = QCheckBox("Show particle IDs")
        self.fit_button = QPushButton("Fit")
        self.actual_button = QPushButton("100%")
        self.zoom_label = QLabel(EM_DASH)
        self.zoom_label.setObjectName("secondaryText")
        self.zoom_label.setProperty("role", "muted")
        for widget in (image_label, self.image_picker, self.ids_checkbox, self.fit_button, self.actual_button, self.zoom_label):
            toolbar.addWidget(widget, 1 if widget is self.image_picker else 0)
        layout.addLayout(toolbar)
        self.loading_label = QLabel("Add images and run an analysis to inspect results.")
        self.loading_label.setObjectName("secondaryText")
        self.loading_label.setProperty("role", "muted")
        self.loading_label.setTextFormat(Qt.TextFormat.PlainText)
        # Reserve one quiet text line so an ID decode never resizes the viewport
        # and shifts the same image's pan position while loading.
        self.loading_label.setMinimumHeight(self.loading_label.fontMetrics().height())
        layout.addWidget(self.loading_label)
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setObjectName("scientificWarning")
        self.warning_label.setTextFormat(Qt.TextFormat.PlainText)
        self.warning_label.setProperty("role", "warning")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)
        self.view = RasterView()
        self.tabs = QTabWidget()
        self.distribution = QWidget()
        distribution_layout = QHBoxLayout(self.distribution)
        distribution_layout.setContentsMargins(8, 12, 8, 8)
        self.distribution_scroll = QScrollArea()
        self.distribution_scroll.setWidgetResizable(True)
        self.distribution_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.distribution_scroll.setAccessibleName("Size distribution and summary")
        distribution_layout.addWidget(self.distribution_scroll)
        self.distribution_scroll.viewport().installEventFilter(self)
        distribution_content = QWidget()
        distribution_content_layout = QHBoxLayout(distribution_content)
        distribution_content_layout.setContentsMargins(0, 0, 0, 0)
        distribution_content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.distribution_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.distribution_splitter.setChildrenCollapsible(False)
        distribution_content_layout.addWidget(self.distribution_splitter)
        self.distribution_scroll.setWidget(distribution_content)
        self.chart_host = QWidget()
        self.chart_host.setMinimumSize(180, 180)
        self.chart_layout = QVBoxLayout(self.chart_host)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_empty = QLabel("A size distribution appears after analysis.")
        self.chart_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_layout.addWidget(self.chart_empty)
        self.summary_host = QWidget()
        self.summary_layout = QFormLayout(self.summary_host)
        self.summary_layout.setContentsMargins(12, 0, 0, 0)
        self.summary_layout.setSpacing(6)
        self.summary_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.summary_values = {}
        for label in ("Detected", "Used in statistics", "Median", "Mean", "SD", "D10", "D90", "Units", "Scale"):
            value = QLabel(EM_DASH)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.summary_layout.addRow(label, value)
            self.summary_values[label] = value
        self.distribution_splitter.addWidget(self.chart_host)
        self.distribution_splitter.addWidget(self.summary_host)
        self.distribution_splitter.setStretchFactor(0, 1)
        self.tabs.addTab(self.distribution, "Distribution")
        self.table = FrozenParticleTable()
        self.table.setAccessibleName("Particles")
        self.particle_model = ParticleTableModel(self)
        self.particle_model.set_panel(None, None)
        self.table.setModel(self.particle_model)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(118)
        self.table.setColumnWidth(0, 60)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table, "Particles")
        self.tabs.setMinimumHeight(160)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([390, 280])
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        self.open_button = QPushButton("Open results folder")
        self.open_button.setToolTip("Open the displayed run's results folder.")
        self.save_button = QPushButton("Save as zip…")
        self.save_button.setToolTip("Save a zip copy of the displayed run's results.")
        actions.addWidget(self.open_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.open_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.image_picker.setEnabled(False)
        self.ids_checkbox.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.actual_button.setEnabled(False)
        self.image_picker.currentIndexChanged.connect(self._select_panel)
        self.ids_checkbox.toggled.connect(self._toggle_ids)
        self.fit_button.clicked.connect(self.fit_image)
        self.actual_button.clicked.connect(self.view.actual_size)
        self.view.zoom_changed.connect(lambda zoom: self.zoom_label.setText(f"{zoom * 100:.0f}%"))
        self.tabs.currentChanged.connect(self._tab_changed)
        self.open_button.clicked.connect(self._open_folder)
        self.save_button.clicked.connect(self.export_requested)

    def _submit(self, kind: str, work: Callable[[], Any]) -> object:
        self._sequence += 1
        token = (kind, self._sequence)
        task = _Task(token, work)
        task.signals.result.connect(self._task_result)
        task.signals.failed.connect(self._task_error)
        self._tasks[token] = task
        self.pool.start(task)
        return token

    def load_run(self, folder: Path) -> None:
        self._panel_token = self._ids_token = None
        self.ids_checkbox.setEnabled(False)
        self.loading_label.setText("Loading results…")
        self.loading_label.show()
        self.image_picker.setEnabled(False)
        self._run_token = self._submit("run", lambda: _read_run(Path(folder)))

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_run_label()

    def _update_run_label(self) -> None:
        self.run_label.setText("Previous run" if self._busy and self.current_run else
                               (self.current_run.name if self.current_run else "No completed run"))
        self.run_label.setToolTip(str(self.current_run or ""))

    @Slot(object, object)
    def _task_result(self, token: object, result: object) -> None:
        self._tasks.pop(token, None)
        if self._closed:
            return
        if token == self._run_token:
            self.run_data = result
            self.current_run = result.folder
            self.current_display = None
            self._panel_token = self._ids_token = None
            self._update_run_label()
            self.image_picker.blockSignals(True)
            self.image_picker.clear()
            for index, panel in enumerate(result.panels, 1):
                self.image_picker.addItem(f"{panel.name}  ·  {index}/{len(result.panels)}")
                self.image_picker.setItemData(index - 1, panel.name, Qt.ItemDataRole.ToolTipRole)
            self.image_picker.blockSignals(False)
            self.image_picker.setEnabled(True)
            self.open_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.tabs.setCurrentIndex(0)
            self._select_panel(0)
        elif token == self._panel_token:
            self.view.display(result.image, result.panel, fit=True)
            # The cache is the only owner retaining decoded QImages. Keeping a
            # second base image here after LRU eviction would exceed the budget.
            self.current_display = replace(result, image=QImage())
            self.particle_model.set_panel(self.run_data.particles, result.panel)
            self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            self.warning_label.setText("\n".join(result.warnings))
            self.warning_label.setVisible(bool(result.warnings))
            self._update_summary(result)
            self.ids_checkbox.setEnabled(result.panel.ids_path is not None)
            self.fit_button.setEnabled(True)
            self.actual_button.setEnabled(True)
            self.loading_label.setText("")
            if self.tabs.currentIndex() == 0:
                self._plot()
            self.ready.emit()
        elif token == self._ids_token and self.current_display is not None:
            self.view.display(result, self.current_display.panel, fit=False)
            self.loading_label.setText("")

    @Slot(object, str)
    def _task_error(self, token: object, message: str) -> None:
        self._tasks.pop(token, None)
        if self._closed or token not in (self._run_token, self._panel_token, self._ids_token):
            return
        self.loading_label.setText(message)
        self.loading_label.show()
        if token == self._run_token:
            self.image_picker.setEnabled(self.run_data is not None)
        elif token == self._ids_token:
            self.ids_checkbox.blockSignals(True)
            self.ids_checkbox.setChecked(False)
            self.ids_checkbox.blockSignals(False)
        self.error.emit(message)

    def _select_panel(self, index: int) -> None:
        if self.run_data is None or index < 0:
            return
        run = self.run_data
        panel = run.panels[index]
        self.image_picker.setToolTip(panel.name)
        self.current_display = None
        self._ids_token = None
        self.ids_checkbox.blockSignals(True)
        self.ids_checkbox.setChecked(False)
        self.ids_checkbox.blockSignals(False)
        self.ids_checkbox.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.actual_button.setEnabled(False)
        self.view.clear_image()
        self.particle_model.set_panel(None, None)
        for value in self.summary_values.values():
            value.setText(EM_DASH)
        self.warning_label.hide()
        if self._canvas is not None:
            self._canvas.hide()
        self.chart_empty.setText("Loading image…")
        self.chart_empty.show()
        self.loading_label.setText("Loading image…")
        self.loading_label.show()
        self._panel_token = self._submit("panel", lambda: _prepare_panel(run, panel, self.cache))

    def _toggle_ids(self, checked: bool) -> None:
        display = self.current_display
        if display is None:
            return
        self._ids_token = None
        if checked and display.panel.ids_path is not None:
            self.loading_label.setText("Loading image…")
            self.loading_label.show()
            path = display.panel.ids_path
            self._ids_token = self._submit("ids", lambda: self.cache.read(path))
        else:
            self.loading_label.setText("Loading image…")
            self.loading_label.show()
            path = display.panel.base_path
            self._ids_token = self._submit("ids", lambda: self.cache.read(path))

    def _update_summary(self, display: PanelDisplay) -> None:
        summary = display.panel.summary
        stats = summary.get("stats_all") or {}
        values = {"Detected": summary.get("n_particles"),
                  "Used in statistics": stats.get("n", summary.get("n_in_stats")),
                  "Units": display.unit}
        for label in ("Median", "Mean", "SD", "D10", "D90"):
            value = stats.get(label.lower())
            values[label] = EM_DASH if value is None else f"{value:.2f}"
        if display.unit == "nm":
            source = "Manual" if summary.get("cal_source") == "manual" else "Automatic"
            values["Scale"] = f"{source} · {summary['nm_per_px']:.6g} nm/px"
        else:
            values["Scale"] = "Not found · sizes in pixels"
        for label, value in values.items():
            self.summary_values[label].setText(EM_DASH if value is None else str(value))

    def _tab_changed(self, index: int) -> None:
        if index == 0 and self.current_display is not None:
            self._plot()

    def _plot(self) -> None:
        display = self.current_display
        if display is None:
            return
        if self._canvas is None:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._figure = Figure(figsize=(4, 2.3), facecolor="#1B1B1F", layout="constrained")
            self._axes = self._figure.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._canvas.setMinimumSize(180, 180)
            self.chart_layout.addWidget(self._canvas)
        self.chart_empty.hide()
        self._canvas.show()
        if self._plotted is display:
            return
        from matplotlib.ticker import MaxNLocator
        axes = self._axes
        axes.clear()
        axes.set_facecolor("#1B1B1F")
        axes.spines[["top", "right"]].set_visible(False)
        for spine in ("bottom", "left"):
            axes.spines[spine].set_color("#44444B")
        axes.tick_params(colors="#B5B5BD", labelsize=9)
        axes.set_xlabel(f"Equivalent diameter ({display.unit})", color="#EDEDED", fontsize=10)
        axes.set_ylabel("Particles", color="#EDEDED", fontsize=10)
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes.grid(axis="y", color="#44444B", linewidth=0.5, alpha=0.5)
        axes.set_axisbelow(True)
        if display.counts:
            axes.stairs(display.counts, display.edges, fill=True, color="#B5B5BD")
            stats = display.panel.summary.get("stats_all") or {}
            median = stats.get("median")
            if median is not None:
                axes.axvline(median, color="#E90802", linewidth=1)
                axes.text(median, 0.96, f" Median {median:.2f}", transform=axes.get_xaxis_transform(),
                          color="#EDEDED", fontsize=9, va="top")
            axes.set_ylim(bottom=0)
        else:
            axes.text(0.5, 0.5, "No particles in statistics", ha="center", va="center",
                      transform=axes.transAxes, color="#B5B5BD", fontsize=10)
            axes.set_ylim(0, 1)
        axes.set_title(f"{display.sample_count} particles in statistics", color="#B5B5BD", fontsize=10, loc="left")
        self._plotted = display
        self._canvas.draw_idle()

    def fit_image(self) -> None:
        self.view.fit_image()

    def _open_folder(self) -> None:
        if self.current_run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_run)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_distribution_orientation()

    def eventFilter(self, watched: QObject, event) -> bool:
        from PySide6.QtCore import QEvent
        if hasattr(self, "distribution_scroll") and watched is self.distribution_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._update_distribution_orientation()
        return super().eventFilter(watched, event)

    def _update_distribution_orientation(self) -> None:
        if hasattr(self, "distribution_splitter"):
            width = self.distribution_scroll.viewport().width()
            orientation = Qt.Orientation.Vertical if width < 720 else Qt.Orientation.Horizontal
            if self.distribution_splitter.orientation() != orientation:
                self.distribution_splitter.setOrientation(orientation)

    def shutdown(self, wait_ms: int = 3000) -> bool:
        """Discard pending handoffs and finish the short-lived display workers."""
        self._closed = True
        self._run_token = self._panel_token = self._ids_token = None
        self.pool.clear()
        finished = self.pool.waitForDone(wait_ms)
        if finished:
            self._tasks.clear()
            self.cache.clear()
            self.current_display = None
            self.view.clear_image()
            if self._figure is not None:
                self._figure.clear()
            self.run_data = None
            self.particle_model.set_panel(None, None)
        return finished
