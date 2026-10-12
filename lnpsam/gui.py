"""Native LNP-SAM desktop workspace. Launch with ``python -m lnpsam.gui``.

Only Qt and the standard library are imported here. Model verification, science,
network requests and file operations are handed to workers after the window opens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import threading
import traceback

from PySide6.QtCore import (
    QAbstractListModel, QLocale, QModelIndex, QRect, QSettings, QSize,
    Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import QDesktopServices, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListView, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSplitter,
    QStyledItemDelegate, QStyle, QStyleOptionViewItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from .gui_results import ResultsWidget
from .theme import apply_theme


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
STAGES = {"cleanup": "Cleanup", "proposals": "SAM proposals", "graph": "Relations",
          "resolve": "Classification", "measure": "Measurement"}


def label(text: str, *, role: str | None = None, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        widget.setProperty("role", role)
    widget.setWordWrap(wrap)
    return widget


def primary(button: QPushButton, enabled: bool = True) -> None:
    button.setProperty("primary", enabled)
    button.style().unpolish(button)
    button.style().polish(button)


@dataclass(frozen=True)
class InputImage:
    path: Path
    width: int
    height: int

    @property
    def warning(self) -> str:
        if max(self.width, self.height) <= 1024:
            return ""
        return (f"This image is large ({self.width}×{self.height}). SAM shrinks it to 1024 px, "
                "so small particles may be missed. For best results, crop to the area you care about.")


class InputModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images: list[InputImage] = []
        self.modality = "TEM"

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.images)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        image = self.images[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return image.path.name
        if role == Qt.ItemDataRole.UserRole:
            return image
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(image.path) + ("\n" + image.warning if image.warning else "")
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return (f"{image.path.name}, {self.modality}, {image.width} by {image.height} pixels"
                    + (", large image" if image.warning else ""))
        return None

    def add(self, images):
        known = {image.path for image in self.images}
        new = [image for image in images if image.path not in known]
        if new:
            self.beginInsertRows(QModelIndex(), len(self.images), len(self.images) + len(new) - 1)
            self.images.extend(new)
            self.endInsertRows()

    def remove(self, rows):
        for row in sorted(set(rows), reverse=True):
            self.beginRemoveRows(QModelIndex(), row, row)
            del self.images[row]
            self.endRemoveRows()

    def set_modality(self, value):
        self.modality = value
        if self.images:
            self.dataChanged.emit(self.index(0), self.index(len(self.images) - 1))


class InputDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(220, max(56, option.fontMetrics.height() * 2 + 16))

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = option.widget.style() if option.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)
        image = index.data(Qt.ItemDataRole.UserRole)
        painter.save()
        rect = option.rect.adjusted(10, 6, -10, -6)
        line_height = option.fontMetrics.height()
        painter.setPen(option.palette.text().color())
        painter.drawText(QRect(rect.x(), rect.y(), rect.width(), line_height),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         option.fontMetrics.elidedText(image.path.name, Qt.TextElideMode.ElideMiddle,
                                                      rect.width()))
        painter.setPen(option.palette.placeholderText().color())
        detail = f"{index.model().modality} · {image.width} × {image.height}"
        if image.warning:
            detail += " · ⚠ Large image"
        painter.drawText(QRect(rect.x(), rect.y() + line_height + 3, rect.width(), line_height),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         option.fontMetrics.elidedText(detail, Qt.TextElideMode.ElideRight, rect.width()))
        painter.restore()


class InputList(QListView):
    files_dropped = Signal(list)
    remove_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.setAccessibleName("Input microscope images")
        self.empty_label = label("Add microscope images\nPNG, JPG or TIF · drag files here", role="muted")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.empty_label.setParent(self.viewport())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.empty_label.setGeometry(self.viewport().rect())

    def _drag_highlight(self, value):
        self.setProperty("dragOver", value)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        if self.isEnabled() and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_highlight(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self.isEnabled() and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drag_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._drag_highlight(False)
        self.files_dropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_requested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class Task(QThread):
    """A short-lived worker; signals are the only path back to widgets."""
    result = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)
    stage = Signal(int, int, str, str)
    message = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.partial_folder = None

    def run(self):
        try:
            self.result.emit(self.function(self))
        except Exception as exc:
            from .app_runtime import RunCancelled
            if isinstance(exc, RunCancelled):
                self.cancelled.emit(str(self.partial_folder or ""))
            else:
                self.failed.emit(str(exc) or type(exc).__name__, traceback.format_exc())
        finally:
            self.function = None


class RemoteSettingsDialog(QDialog):
    def __init__(self, address, token, task_launcher, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remote server")
        self.setMinimumWidth(400)
        self.resize(480, 310)
        self._launch = task_launcher
        self._dialog_closed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        form = QFormLayout()
        form.setVerticalSpacing(12)
        self.address_field = QLineEdit(address)
        self.address_field.setPlaceholderText("http://localhost:8765")
        self.address_field.setAccessibleName("Server address")
        address_label = label("Server address")
        address_label.setBuddy(self.address_field)
        form.addRow(address_label, self.address_field)
        self.token_field = QLineEdit(token)
        self.token_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_field.setAccessibleName("Access token")
        token_label = label("Access token")
        token_label.setBuddy(self.token_field)
        form.addRow(token_label, self.token_field)
        layout.addLayout(form)
        show = QCheckBox("Show token")
        show.toggled.connect(lambda checked: self.token_field.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password))
        layout.addWidget(show)
        self.test_button = QPushButton("Test connection")
        self.test_button.clicked.connect(self.test_connection)
        layout.addWidget(self.test_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.connection_status = label("", role="muted", wrap=True)
        layout.addWidget(self.connection_status)
        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        primary(save)
        save.clicked.connect(self.save)
        row.addWidget(cancel)
        row.addWidget(save)
        layout.addLayout(row)

    @property
    def address(self):
        return self.address_field.text().strip().rstrip("/")

    @property
    def token(self):
        return self.token_field.text()

    def done(self, result):
        # A test may still be in flight when Cancel closes/deletes this dialog.
        self._dialog_closed = True
        super().done(result)

    def save(self):
        from .remote import RemoteClient, RemoteError
        try:
            RemoteClient(self.address, self.token)
        except (ValueError, RemoteError) as exc:
            self.connection_status.setText(str(exc))
            return
        if not self.token:
            self.connection_status.setText("Enter the server's access token.")
            self.token_field.setFocus()
            return
        self.accept()

    def test_connection(self):
        self.test_button.setEnabled(False)
        self.connection_status.setText("Testing connection…")
        address, token = self.address, self.token

        def work(_):
            from .remote import RemoteClient
            return RemoteClient(address, token).info()

        def done(info):
            if self._dialog_closed:
                return
            self.test_button.setEnabled(True)
            device = {"cuda": "CUDA", "mps": "Apple GPU", "cpu": "CPU"}.get(info.get("device"), info.get("device", "Unknown device"))
            self.connection_status.setText(f"Connected · {device} · LNP-SAM {info.get('version', info.get('lnpsam_version', 'unknown'))}")

        def failed(message, _details):
            if self._dialog_closed:
                return
            self.test_button.setEnabled(True)
            self.connection_status.setText(message)

        self._launch(work, done, failed)


class MainWindow(QMainWindow):
    def __init__(self, *, config_path=None, checkpoint=None, device="auto", runs_root=None,
                 settings=None, check_setup=True):
        super().__init__()
        self.setWindowTitle("LNP-SAM")
        self.setMinimumSize(1024, 720)
        self.resize(1280, 840)
        self.config_path = Path(config_path or REPO_ROOT / "config.yaml").resolve()
        self.checkpoint = checkpoint
        self.device = device
        self.runs_root = runs_root
        self.settings = settings if settings is not None else QSettings("LNP-SAM", "LNP-SAM")
        self._threads = set()
        self._active = False
        self._cancel_event = None
        self._local_ready = False
        self._setup_checking = check_setup
        self._setup_error = "Local model has not been checked."
        self._invalid_scale = False
        self._pending_images = 0
        self._exporting = False
        self._close_pending = False
        self._terminal_status = ""
        self._device_label = ""
        self._last_panel = ""
        self._last_stage = ""
        self.remote_address = str(self.settings.value("remote/address", ""))
        self.remote_token = ""  # Session only: never persisted in QSettings or provenance.
        self._upload_confirmed = set()
        self._build()
        self._restore_preferences()
        self._update_ready()
        if check_setup:
            QTimer.singleShot(100, self.check_local_setup)

    def _build(self):
        center = QWidget()
        outer = QVBoxLayout(center)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        toolbar = QWidget()
        toolbar.setObjectName("productToolbar")
        toolbar.setMinimumHeight(44)
        top = QHBoxLayout(toolbar)
        top.setContentsMargins(16, 6, 16, 6)
        product = label("LNP-SAM", role="heading")
        product_font = QFont(product.font())
        product_font.setPointSizeF(product_font.pointSizeF() * 1.35)
        product.setFont(product_font)
        top.addWidget(product)
        top.addStretch()
        self.toolbar_status = label("This computer", role="muted")
        top.addWidget(self.toolbar_status)
        outer.addWidget(toolbar)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setChildrenCollapsible(False)
        rail = QWidget()
        rail.setObjectName("inputRail")
        rail.setMinimumWidth(264)
        rail.setMaximumWidth(360)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(label("Inputs", role="heading"))
        layout.addSpacing(12)
        images_row = QHBoxLayout()
        images_row.addWidget(label("Images", role="heading"))
        images_row.addStretch()
        self.image_count = label("0 added", role="muted")
        images_row.addWidget(self.image_count)
        layout.addLayout(images_row)
        self.add_button = QPushButton("Add images…")
        self.add_button.clicked.connect(self.pick_images)
        layout.addWidget(self.add_button)
        self.input_model = InputModel(self)
        self.image_list = InputList()
        self.image_list.setModel(self.input_model)
        self.image_list.setItemDelegate(InputDelegate(self.image_list))
        self.image_list.setMinimumHeight(110)
        self.image_list.files_dropped.connect(self.add_images)
        self.image_list.remove_requested.connect(self.remove_images)
        layout.addWidget(self.image_list, 1)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_images)
        self.image_list.selectionModel().selectionChanged.connect(self._selection_changed)
        layout.addWidget(self.remove_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.input_error = label("", role="error", wrap=True)
        self.input_error.hide()
        layout.addWidget(self.input_error)
        layout.addSpacing(12)
        layout.addWidget(label("Modality", role="heading"))
        modality = QHBoxLayout()
        self.tem_radio = QRadioButton("TEM")
        self.sem_radio = QRadioButton("SEM")
        self.tem_radio.setChecked(True)
        self.tem_radio.toggled.connect(lambda checked: self._set_modality("TEM") if checked else None)
        self.sem_radio.toggled.connect(lambda checked: self._set_modality("SEM") if checked else None)
        modality.addWidget(self.tem_radio)
        modality.addWidget(self.sem_radio)
        modality.addStretch()
        layout.addLayout(modality)
        layout.addWidget(label("Applies to all images.", role="muted"))
        layout.addSpacing(8)
        scale_label = label("Scale (nm per pixel)", role="heading")
        self.scale_field = QLineEdit()
        self.scale_field.setAccessibleName("Scale in nanometers per pixel; optional")
        scale_label.setBuddy(self.scale_field)
        self.scale_field.editingFinished.connect(self.validate_scale)
        self.scale_field.textEdited.connect(self._scale_edited)
        layout.addWidget(scale_label)
        layout.addWidget(self.scale_field)
        layout.addWidget(label("Leave blank to read the scale bar.\nApplies to all images in this run.", role="muted", wrap=True))
        self.scale_error = label("Enter a positive number, or leave blank for automatic scale detection.", role="error", wrap=True)
        self.scale_error.hide()
        layout.addWidget(self.scale_error)
        layout.addSpacing(12)
        compute_label = label("Compute", role="heading")
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["This computer", "Remote server"])
        self.compute_combo.setAccessibleName("Compute location")
        compute_label.setBuddy(self.compute_combo)
        self.compute_combo.currentIndexChanged.connect(self._compute_changed)
        layout.addWidget(compute_label)
        layout.addWidget(self.compute_combo)
        self.settings_button = QPushButton("Settings…")
        self.settings_button.clicked.connect(self.show_remote_settings)
        layout.addWidget(self.settings_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.compute_helper = label("", role="muted", wrap=True)
        layout.addWidget(self.compute_helper)
        self.setup_message = label("", wrap=True)
        self.setup_message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.setup_message)
        self.check_button = QPushButton("Check again")
        self.check_button.clicked.connect(self.check_local_setup)
        self.check_button.hide()
        layout.addWidget(self.check_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setup_details_button = QPushButton("Setup details…")
        self.setup_details_button.clicked.connect(self.show_setup_details)
        self.setup_details_button.hide()
        layout.addWidget(self.setup_details_button, alignment=Qt.AlignmentFlag.AlignLeft)
        # Keep Run/Cancel reachable even when a setup message or large text scale
        # makes the preparation controls taller than the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(rail)
        input_panel = QWidget()
        input_panel.setObjectName("inputRail")
        input_panel.setMinimumWidth(264)
        input_panel.setMaximumWidth(360)
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)
        input_layout.addWidget(scroll, 1)
        action_panel = QWidget()
        action_panel.setObjectName("inputRail")
        actions = QVBoxLayout(action_panel)
        actions.setContentsMargins(16, 12, 16, 16)
        actions.setSpacing(8)
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("runButton")  # Theme sets its 40 px height.
        primary(self.run_button)
        self.run_button.clicked.connect(self.run_or_cancel)
        actions.addWidget(self.run_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setAccessibleName("Completed images")
        actions.addWidget(self.progress_bar)
        self.progress_status = label("Add images to begin.", role="muted", wrap=True)
        actions.addWidget(self.progress_status)
        input_layout.addWidget(action_panel)
        self.main_splitter.addWidget(input_panel)
        self.results = ResultsWidget()
        self.results.error.connect(self.show_result_error)
        self.results.export_requested.connect(self.save_zip)
        self.results.ready.connect(self._results_ready)
        self.main_splitter.addWidget(self.results)
        self.main_splitter.setSizes([296, 978])
        self.main_splitter.setStretchFactor(1, 1)
        outer.addWidget(self.main_splitter, 1)
        self.setCentralWidget(center)
        self.statusBar().setMinimumHeight(28)
        self.statusBar().showMessage("Add images to begin.")
        self.details_button = QPushButton("Details")
        self.details_button.hide()
        self.details_button.clicked.connect(self.show_details)
        self.statusBar().addPermanentWidget(self.details_button)
        self._details = ""
        self._input_controls = [self.add_button, self.image_list, self.remove_button,
                                self.tem_radio, self.sem_radio, self.scale_field, self.compute_combo]
        self._shortcut(QKeySequence.StandardKey.Open, self.pick_images)
        self._shortcut("Ctrl+Return", self._shortcut_run)
        if sys.platform == "darwin":
            self._shortcut("Meta+Return", self._shortcut_run)
        self._shortcut(QKeySequence.StandardKey.Save, self.save_zip)
        self._shortcut("Escape", self._escape)

    def _shortcut(self, key, callback):
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.activated.connect(callback)
        return shortcut

    def _task(self, function, done, failed=None):
        task = Task(function, self)
        self._threads.add(task)
        task.result.connect(done)
        task.failed.connect(failed or self._failed)
        task.finished.connect(lambda: self._task_finished(task))
        task.start()
        return task

    def _task_finished(self, task):
        self._threads.discard(task)
        task.deleteLater()
        if self._close_pending and not self._threads:
            self.close()

    def check_local_setup(self):
        if self._active or any(getattr(task, "setup", False) for task in self._threads):
            return
        self._setup_checking = True
        self._update_ready()

        def work(_):
            from .app_runtime import check_local_setup
            return check_local_setup(self.config_path, self.checkpoint)

        def done(_):
            self._setup_checking = False
            self._local_ready = True
            self._setup_error = ""
            self._update_ready()

        def failed(message, _details):
            self._setup_checking = False
            self._local_ready = False
            self._setup_error = message
            self._update_ready()

        task = self._task(work, done, failed)
        task.setup = True

    def pick_images(self):
        if self._active:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add microscope images", "",
                                               "Microscope images (*.png *.jpg *.jpeg *.tif *.tiff)")
        self.add_images(paths)

    def add_images(self, paths):
        if self._active or not paths:
            return
        self._pending_images += 1
        self._terminal_status = ""
        self.input_error.hide()
        self._update_ready()

        def work(_):
            from PIL import Image
            images, errors = [], []
            known = set()
            for value in paths:
                path = Path(value).expanduser().resolve()
                if path in known:
                    continue
                known.add(path)
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    errors.append(f"{path.name}: use PNG, JPG or TIF.")
                    continue
                try:
                    with Image.open(path) as source:
                        width, height = source.size
                    images.append(InputImage(path, width, height))
                except Exception:
                    errors.append(f"Can't read {path.name}. Choose a valid image file.")
            return images, errors

        def done(result):
            self._pending_images -= 1
            images, errors = result
            self.input_model.add(images)
            if errors:
                self.input_error.setText("\n".join(errors))
                self.input_error.show()
            self._update_ready()

        def failed(message, details):
            self._pending_images -= 1
            self.input_error.setText(message)
            self.input_error.show()
            self._update_ready()

        self._task(work, done, failed)

    def remove_images(self):
        if self._active:
            return
        self.input_model.remove([index.row() for index in self.image_list.selectedIndexes()])
        self._terminal_status = ""
        self._update_ready()

    def _selection_changed(self):
        self.remove_button.setEnabled(not self._active and bool(self.image_list.selectedIndexes()))

    def _set_modality(self, value):
        self.input_model.set_modality(value)

    def _scale_edited(self):
        self._invalid_scale = False
        self.scale_error.hide()
        self._update_ready()

    def scale_value(self):
        text = self.scale_field.text().strip()
        if not text:
            return None
        value, valid = QLocale().toDouble(text)
        if not valid or not math.isfinite(value) or value <= 0:
            raise ValueError("Enter a positive number, or leave blank for automatic scale detection.")
        return value

    def validate_scale(self):
        try:
            self.scale_value()
            self._invalid_scale = False
        except ValueError:
            self._invalid_scale = True
        self.scale_error.setVisible(self._invalid_scale)
        self._update_ready()
        return not self._invalid_scale

    def _compute_changed(self):
        self._terminal_status = ""
        self._update_ready()

    @property
    def is_remote(self):
        return self.compute_combo.currentIndex() == 1

    def _update_ready(self):
        count = len(self.input_model.images)
        self.image_count.setText(f"{count} added")
        self.image_list.empty_label.setVisible(count == 0)
        self._selection_changed()
        self.settings_button.setEnabled(self.is_remote and not self._active)
        self.compute_helper.setText(
            (f"{self.remote_address}\nImages are sent to this server." if self.remote_address
             else "Choose a server in Settings. Images are sent to this server.")
            if self.is_remote else "Uses an available GPU, otherwise CPU.")
        self.toolbar_status.setText("Remote server" if self.is_remote else "This computer")
        self.check_button.setVisible(not self.is_remote and not self._local_ready and not self._setup_checking)
        self.check_button.setEnabled(not self._active)
        self.setup_details_button.setVisible(not self.check_button.isHidden())
        if self.is_remote:
            self.setup_message.setText("" if self.remote_token else "Set the server address and access token in Settings.")
        else:
            self.setup_message.setText("Checking local model…" if self._setup_checking
                                       else ("" if self._local_ready else self._short_setup_error()))
        self.setup_message.setToolTip(self._setup_error if not self.is_remote else "")
        self.setup_message.setVisible(bool(self.setup_message.text()))
        if self._active:
            return
        ready = bool(self.remote_address and self.remote_token) if self.is_remote else self._local_ready
        self.run_button.setEnabled(count > 0 and ready and not self._pending_images and not self._invalid_scale)
        self.run_button.setText("Run")
        primary(self.run_button)
        if self._terminal_status:
            status = self._terminal_status
        elif self._pending_images:
            status = "Reading image dimensions…"
        elif count == 0:
            status = "Add images to begin."
        elif ready and not self._invalid_scale:
            status = f"Ready · {count} {'image' if count == 1 else 'images'}"
        elif self._invalid_scale:
            status = "Check the scale value."
        else:
            status = "Configure the remote server." if self.is_remote else "Check local setup."
        self._set_status(status)

    def _set_status(self, message):
        self.progress_status.setText(message)
        self.statusBar().showMessage(message)

    def show_remote_settings(self):
        dialog = RemoteSettingsDialog(self.remote_address, self.remote_token, self._task, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.remote_address, self.remote_token = dialog.address, dialog.token
            self.settings.setValue("remote/address", self.remote_address)
            self._update_ready()
        dialog.deleteLater()

    def _short_setup_error(self):
        if "checkpoint_sha256" in self._setup_error:
            return "Set the model SHA-256 in config.yaml."
        if "model file not found" in self._setup_error.lower():
            return "SAM model file not found. Open Setup details to download it."
        if "doesn't match" in self._setup_error:
            return "SAM model file doesn't match. Check the file and configured hash."
        return self._setup_error.split("\n", 1)[0]

    def show_setup_details(self):
        from .app_runtime import SAM_DOWNLOAD_URL
        from PySide6.QtCore import QUrl
        dialog = QDialog(self)
        dialog.setWindowTitle("Local model setup")
        dialog.resize(560, 260)
        layout = QVBoxLayout(dialog)
        message = label(self._setup_error, wrap=True)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(message)
        config = label(f"Config: {self.config_path}", role="muted", wrap=True)
        config.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(config)
        row = QHBoxLayout()
        download = QPushButton("Download SAM model")
        download.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(SAM_DOWNLOAD_URL)))
        row.addWidget(download)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        row.addWidget(close)
        layout.addLayout(row)
        dialog.exec()

    def _shortcut_run(self):
        if self.run_button.isEnabled() and not self._active:
            self.run_or_cancel()

    def _escape(self):
        if QApplication.activeModalWidget() is None and self._active:
            self.request_cancel()

    def run_or_cancel(self):
        if self._active:
            self.request_cancel()
            return
        if not self.validate_scale():
            self.scale_field.setFocus()
            return
        if not self.run_button.isEnabled():
            return
        self._active = True
        self._cancel_event = threading.Event()
        self._terminal_status = ""
        self._details = ""
        self._last_panel = self._last_stage = ""
        self.details_button.hide()
        for widget in self._input_controls:
            widget.setEnabled(False)
        self.settings_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.run_button.setText("Cancel")
        primary(self.run_button, False)
        self.run_button.setEnabled(True)
        self.progress_bar.setRange(0, len(self.input_model.images))
        self.progress_bar.setValue(0)
        self.results.set_busy(True)
        self._set_status("Checking server…" if self.is_remote else "Preparing run…")
        if self.is_remote:
            self._remote_preflight()
        else:
            self._start_processing()

    def _remote_preflight(self):
        address, token = self.remote_address, self.remote_token

        def work(_):
            from .remote import RemoteClient
            from .app_runtime import app_identity
            return RemoteClient(address, token).info(), app_identity(self.config_path, self.checkpoint)

        def done(result):
            if self._cancel_event.is_set():
                self._cancelled("")
                return
            remote, local = result
            differing = []
            server_version = remote.get("version", remote.get("lnpsam_version"))
            local_version = local.get("version", local.get("lnpsam_version"))
            if server_version != local_version:
                differing.append(f"Version: this app {local_version}; server {server_version}.")
            expected = local.get("checkpoint_sha256")
            if expected and remote.get("checkpoint_sha256") != expected:
                differing.append(f"Model: this app {expected}; server {remote.get('checkpoint_sha256', 'unknown')}.")
            if differing and not self._confirm(
                    "Server runs a different LNP-SAM version or model; results may differ.\n\n" + "\n".join(differing), warning=True):
                self._cancelled("")
                return
            if address not in self._upload_confirmed:
                if not self._confirm(f"Your images will be sent to {address}."):
                    self._cancelled("")
                    return
                self._upload_confirmed.add(address)
            self._device_label = f"Remote ({address})"
            self._start_processing()

        def failed(message, details):
            if self._cancel_event.is_set():
                self._cancelled("")
            else:
                self._failed(message, details)

        self._task(work, done, failed)

    def _confirm(self, text, warning=False):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Remote server")
        dialog.setIcon(QMessageBox.Icon.Warning if warning else QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(text)
        proceed = dialog.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        return dialog.clickedButton() is proceed

    def _start_processing(self):
        images = [image.path for image in self.input_model.images]
        modality, scale = self.input_model.modality, self.scale_value()
        remote = self.is_remote
        cancel_event = self._cancel_event
        address, token = self.remote_address, self.remote_token

        def work(task):
            from .app_runtime import prepare_run, run_local
            folder = prepare_run(images, modality, scale, self.runs_root)
            task.partial_folder = folder
            if remote:
                import csv
                from .remote import RemoteClient
                # Upload the same immutable snapshot described by our manifest.
                with (folder / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                    uploaded_images = [Path(row["image_path"]) for row in csv.DictReader(handle)]
                return RemoteClient(address, token).run(
                    uploaded_images, modality, scale, folder, progress=task.stage.emit,
                    status=task.message.emit, cancel_event=cancel_event)
            return run_local(images, modality, scale, self.config_path,
                             checkpoint=self.checkpoint, device=self.device,
                             runs_root=self.runs_root, run_folder=folder,
                             progress=task.stage.emit, status=task.message.emit,
                             cancel_event=cancel_event)

        # Connect signals before starting: a small/faked run can finish immediately.
        task = Task(work, self)
        self._threads.add(task)
        task.result.connect(self._complete)
        task.failed.connect(self._failed)
        task.cancelled.connect(self._cancelled)
        task.stage.connect(self._progress)
        task.message.connect(self._worker_status)
        task.finished.connect(lambda: self._task_finished(task))
        task.start()

    def _worker_status(self, message):
        if not self._active or self._cancel_event.is_set():
            return
        if " · image " in message:
            return  # The stage signal supplies the same update with GUI wording.
        if message in ("Apple GPU", "CPU", "NVIDIA GPU", "CUDA"):
            self._device_label = message
        elif "retrying on CPU" in message:
            self._device_label = "CPU"
        self._set_status(message)

    def _progress(self, index, total, panel, stage):
        if not self._active or self._cancel_event.is_set():
            return
        self.progress_bar.setRange(0, total)
        self._last_panel = panel
        self._last_stage = STAGES.get(stage, stage)
        self.progress_bar.setValue(max(0, index - 1))
        self.progress_bar.setAccessibleDescription(f"{index - 1} of {total} images complete")
        self._set_status(f"{self._device_label or 'This computer'} · image {index}/{total} · {STAGES.get(stage, stage)}")
        self.progress_status.setToolTip(f"{panel} · {STAGES.get(stage, stage)}")

    def request_cancel(self):
        if not self._active or self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self.run_button.setText("Cancelling…")
        self.run_button.setEnabled(False)
        self._set_status("Stopping after the current stage.")

    def _restore_inputs(self):
        self._active = False
        for widget in self._input_controls:
            widget.setEnabled(True)
        self.results.set_busy(False)
        self._update_ready()

    def _complete(self, folder):
        count = len(self.input_model.images)
        self.progress_bar.setValue(count)
        self._terminal_status = f"Complete · {count} {'image' if count == 1 else 'images'}"
        self._restore_inputs()
        self.results.load_run(Path(folder))

    def _cancelled(self, folder):
        if self.is_remote:
            self._terminal_status = "Cancelled · no remote results downloaded."
        else:
            self._terminal_status = f"Cancelled · partial results kept in {folder}" if folder else "Cancelled"
        self._restore_inputs()

    def _failed(self, message, details):
        self._details = details
        self.details_button.setVisible(bool(details))
        context = f"{self._last_panel} · {self._last_stage} · " if self._active and self._last_panel else ""
        self._terminal_status = f"Failed · {context}{message}"
        if self._active:
            self._restore_inputs()
        else:
            self._set_status(self._terminal_status)

    def show_result_error(self, message):
        self._set_status(message)

    def _results_ready(self):
        self.results.save_button.setEnabled(not self._exporting)
        if self._terminal_status:
            self._set_status(self._terminal_status)

    def show_details(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing details")
        dialog.resize(720, 420)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self._details)
        layout.addWidget(text)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def save_zip(self):
        folder = self.results.current_run
        if folder is None or self._exporting:
            return
        destination, _ = QFileDialog.getSaveFileName(self, "Save results as zip", f"{folder.name}.zip", "Zip archive (*.zip)")
        if not destination:
            return
        if not destination.lower().endswith(".zip"):
            destination += ".zip"
        self._exporting = True
        self.results.save_button.setEnabled(False)
        self._set_status("Saving results…")

        def work(_):
            from .app_runtime import export_results_zip
            return export_results_zip(folder, destination)

        def done(_):
            self._exporting = False
            self.results.save_button.setEnabled(self.results.current_run is not None)
            self._set_status(f"Saved results to {destination}")

        def failed(message, details):
            self._exporting = False
            self.results.save_button.setEnabled(self.results.current_run is not None)
            self._details = details
            self.details_button.setVisible(bool(details))
            self._set_status(f"Could not save results · {message}")

        self._task(work, done, failed)

    def _restore_preferences(self):
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        screen = self.screen().availableGeometry()
        size = self.size().boundedTo(screen.size())
        self.resize(size)
        frame = self.frameGeometry()
        if not screen.contains(frame):
            self.move(max(screen.left(), min(self.x(), screen.right() - frame.width() + 1)),
                      max(screen.top(), min(self.y(), screen.bottom() - frame.height() + 1)))
        split = self.settings.value("window/splitter")
        if split is not None:
            self.main_splitter.restoreState(split)
        result_split = self.settings.value("results/splitter")
        if result_split is not None:
            self.results.splitter.restoreState(result_split)
        table_columns = self.settings.value("results/table_columns")
        if table_columns is not None:
            self.results.table.horizontalHeader().restoreState(table_columns)
        self.compute_combo.setCurrentIndex(int(self.settings.value("compute", 0)))

    def closeEvent(self, event):
        if self._threads:
            self._close_pending = True
            if self._active:
                self.request_cancel()
            self._set_status("Closing after current work stops…")
            event.ignore()
            return
        if hasattr(self.results, "shutdown") and not self.results.shutdown(wait_ms=0):
            self._close_pending = True
            QTimer.singleShot(100, self.close)
            event.ignore()
            return
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.main_splitter.saveState())
        self.settings.setValue("results/splitter", self.results.splitter.saveState())
        self.settings.setValue("results/table_columns", self.results.table.horizontalHeader().saveState())
        self.settings.setValue("compute", self.compute_combo.currentIndex())
        self.settings.sync()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description="LNP-SAM desktop app")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--results", type=Path, help="Open an existing results folder without running SAM")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setOrganizationName("LNP-SAM")
    app.setApplicationName("LNP-SAM")
    apply_theme(app)
    window = MainWindow(config_path=args.config, checkpoint=args.checkpoint, device=args.device)
    window.show()
    if args.results:
        QTimer.singleShot(0, lambda: window.results.load_run(args.results))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
