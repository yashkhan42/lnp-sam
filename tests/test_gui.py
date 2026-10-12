"""Desktop behavior, import boundary and a real pipeline with faked SAM/OCR."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
from PIL import Image
from PySide6.QtCore import QLocale, QSettings
from PySide6.QtWidgets import QApplication
import pytest
import yaml

from lnpsam import app_runtime
from lnpsam.gui import MainWindow, RemoteSettingsDialog
from lnpsam.theme import apply_theme


@pytest.fixture
def batch(tmp_path):
    image = tmp_path / "microscope.png"
    pixels = np.full((60, 60, 3), 200, dtype=np.uint8)
    y, x = np.ogrid[:60, :60]
    pixels[(y - 30) ** 2 + (x - 30) ** 2 <= 100] = 60
    Image.fromarray(pixels).save(image)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"GUI checkpoint fixture")
    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    config["sam"]["checkpoint"] = str(checkpoint)
    config["sam"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return image, path


@pytest.fixture
def window(qtbot, tmp_path, batch):
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    apply_theme(QApplication.instance())
    widget = MainWindow(config_path=batch[1], runs_root=tmp_path / "runs", settings=settings)
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(lambda: widget._local_ready)
    yield widget
    qtbot.waitUntil(lambda: not widget._threads, timeout=10000)
    widget.results.shutdown()


def fake_science(monkeypatch):
    from lnpsam import run
    monkeypatch.setattr(run.proposals_mod, "select_device", lambda _: "cpu")

    class Reader:
        def readtext(self, *args, **kwargs):
            return []

    class Generator:
        def generate(self, image):
            particle = image[:, :, 0] < 130
            return [{"segmentation": mask, "predicted_iou": .95,
                     "stability_score": .95, "point_coords": [[30., 30.]]}
                    for mask in (particle, ~particle)]

    monkeypatch.setattr(run.cleanup_mod, "get_reader", Reader)
    monkeypatch.setattr(run.proposals_mod, "load_sam_model", lambda *a, **k: object())
    monkeypatch.setattr(run.proposals_mod, "make_mask_generator", lambda *a, **k: Generator())


def test_window_opens_without_torch_sam_matplotlib_or_local_model(tmp_path):
    script = r'''
import importlib.abc, sys
class ForbidScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'segment_anything', 'easyocr', 'matplotlib'}:
            raise AssertionError('startup imported ' + fullname)
sys.meta_path.insert(0, ForbidScience())
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings, QTimer
from lnpsam.gui import MainWindow
app = QApplication([])
w = MainWindow(config_path=sys.argv[1], settings=QSettings(sys.argv[2], QSettings.Format.IniFormat))
w.show()
assert w.isVisible()
QTimer.singleShot(350, app.quit)
app.exec()
assert not w.run_button.isEnabled()
w.compute_combo.setCurrentIndex(1)
assert w.settings_button.isEnabled()
assert not any(name in sys.modules for name in ('torch','segment_anything','matplotlib'))
w.close()
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "missing.yaml"),
                             str(tmp_path / "prefs.ini")], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_add_run_and_inspect_matches_pipeline(qtbot, window, batch, monkeypatch):
    fake_science(monkeypatch)
    window.add_images([batch[0]])
    qtbot.waitUntil(lambda: window.input_model.rowCount() == 1)
    assert window.run_button.isEnabled()
    window.scale_field.setText("1.5")
    window.run_or_cancel()
    assert not window.scale_field.isEnabled()
    assert window.run_button.text() == "Cancel"
    qtbot.waitUntil(lambda: window.results.current_display is not None, timeout=15000)
    assert not window._active
    assert window.run_button.text() == "Run"
    assert window.progress_bar.value() == 1
    assert window.results.particle_model.rowCount() == 1
    assert window.results.summary_values["Detected"].text() == "1"
    assert window.results.summary_values["Units"].text() == "nm"
    assert window.results.summary_values["Scale"].text() == "Manual · 1.5 nm/px"
    assert "Too few particles" in window.results.warning_label.text()
    folder = window.results.current_run
    info = json.loads((folder / "run_info.json").read_text())
    assert info["device"] == "cpu"
    assert info["images"][0]["nm_per_px"] == 1.5
    saved = pd.read_csv(folder / "particles_all.csv")
    assert window.results.particle_model.frame.iloc[0]["d_nm"] == saved.iloc[0]["d_nm"]
    # Input edits prepare the next run and never change displayed units/statistics.
    window.scale_field.clear()
    window.sem_radio.setChecked(True)
    assert window.results.summary_values["Scale"].text() == "Manual · 1.5 nm/px"
    assert pd.read_csv(folder / "particles_all.csv").equals(saved)


def test_cancel_reachable_and_waits_for_stage_boundary(qtbot, window, batch, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def pipeline(manifest, config, out, **kwargs):
        calls.append(1)
        kwargs["progress"](1, 1, "microscope", "cleanup")
        entered.set()
        assert release.wait(5)
        kwargs["progress"](1, 1, "microscope", "proposals")

    monkeypatch.setattr(app_runtime, "_load_pipeline", lambda: (pipeline, lambda _: "cpu"))
    window.add_images([batch[0]])
    qtbot.waitUntil(lambda: window.run_button.isEnabled())
    window.run_or_cancel()
    qtbot.waitUntil(entered.is_set)
    window.run_or_cancel()
    assert window.run_button.text() == "Cancelling…"
    assert not window.run_button.isEnabled()
    assert not window.add_button.isEnabled()
    assert window._active
    release.set()
    qtbot.waitUntil(lambda: not window._active)
    assert "Cancelled · partial results kept" in window.progress_status.text()
    assert window.run_button.isEnabled()
    assert len(calls) == 1
    assert window.results.current_run is None


def test_optional_scale_uses_locale_and_rejects_invalid(window):
    previous = QLocale()
    try:
        QLocale.setDefault(QLocale("de_DE"))
        window.scale_field.setText("1,25")
        assert window.scale_value() == 1.25
        for value in ("0", "-1", "nan", "inf", "no"):
            window.scale_field.setText(value)
            assert not window.validate_scale()
            assert not window.scale_error.isHidden()
        window.scale_field.clear()
        assert window.validate_scale()
        assert window.scale_value() is None
    finally:
        QLocale.setDefault(previous)


def test_remote_settings_invalid_address_stays_open(qtbot, window):
    dialog = RemoteSettingsDialog("bad-address", "secret", window._task, window)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.save()
    assert dialog.isVisible()
    assert "server address" in dialog.connection_status.text()
    dialog.reject()


def test_tokens_never_persist(qtbot, window):
    window.remote_address = "http://localhost:8765"
    window.remote_token = "a-session-only-secret"
    window.settings.setValue("remote/address", window.remote_address)
    window.compute_combo.setCurrentIndex(1)
    window.close()
    window.settings.sync()
    assert "a-session-only-secret" not in Path(window.settings.fileName()).read_text()


def test_remote_gui_preflight_uploads_snapshot_and_displays_results(qtbot, window, batch, monkeypatch):
    original = batch[0].read_bytes()
    confirmations = []
    uploaded = []

    class Client:
        def __init__(self, address, token):
            assert address == "http://lab-gpu:8765"
            assert token == "session-secret"

        def info(self):
            return {"version": "0.1.0", "device": "cuda",
                    "checkpoint_sha256": app_runtime.app_identity(batch[1])["checkpoint_sha256"]}

        def run(self, images, modality, nm_per_px, folder, **kwargs):
            # An original changed after preparation must not affect the upload.
            batch[0].write_bytes(b"original changed after snapshot")
            uploaded.extend(images)
            assert images[0].parent == folder / "inputs"
            assert images[0].read_bytes() == original
            measure = folder / "measure"
            measure.mkdir()
            Image.new("RGB", (60, 70), "gray").save(measure / "microscope__overlay.png")
            panel = {"panel_id": "microscope", "n_particles": 0, "n_in_stats": 0,
                     "nm_per_px": None, "cal_source": "missing", "stats_all": None}
            (measure / "microscope__panel.json").write_text(json.dumps(panel))
            pd.DataFrame(columns=["panel_id", "particle_id", "d_px", "d_nm", "in_stats"]).to_csv(
                folder / "particles_all.csv", index=False)
            (folder / "panels_all.csv").write_text("panel_id\nmicroscope\n")
            (folder / "run_info.json").write_text(json.dumps({"where": "http://lab-gpu:8765", "device": "cuda"}))
            kwargs["progress"](1, 1, "microscope", "measure")
            return folder

    monkeypatch.setattr("lnpsam.remote.RemoteClient", Client)
    monkeypatch.setattr(window, "_confirm", lambda text, **kwargs: confirmations.append(text) or True)
    window.remote_address = "http://lab-gpu:8765"
    window.remote_token = "session-secret"
    window.compute_combo.setCurrentIndex(1)
    window.add_images([batch[0]])
    qtbot.waitUntil(lambda: window.run_button.isEnabled())
    window.run_or_cancel()
    qtbot.waitUntil(lambda: window.results.current_display is not None, timeout=10000)
    assert len(uploaded) == 1
    assert confirmations == ["Your images will be sent to http://lab-gpu:8765."]
    assert window.results.summary_values["Units"].text() == "px"
    assert window.results.current_display.panel.source_size == (60, 60)
    assert window.progress_status.text() == "Complete · 1 image"
    assert window.results.save_button.isEnabled()
