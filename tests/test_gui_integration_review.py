"""Regressions for independent background tasks and closed-dialog handoffs."""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
import shiboken6

from lnpsam.gui import InputImage, MainWindow, RemoteSettingsDialog


def test_failed_previous_run_export_does_not_end_active_analysis(qtbot, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, check_setup=False, runs_root=tmp_path / "runs")
    qtbot.addWidget(window)
    window._local_ready = True
    window.input_model.add([InputImage(tmp_path / "micrograph.png", 100, 100)])
    window._update_ready()
    # The real scientific worker is independently covered by GUI smoke tests.
    # Keep this run at its first boundary while an unrelated export completes.
    monkeypatch.setattr(window, "_start_processing", lambda: None)
    window.run_or_cancel()
    assert window._active
    original_cancel_event = window._cancel_event
    window.results.current_run = tmp_path / "previous-run"
    monkeypatch.setattr("lnpsam.gui.QFileDialog.getSaveFileName", lambda *args: (str(tmp_path / "results.zip"), ""))
    callbacks = {}

    def capture_task(work, done, failed=None):
        callbacks.update(done=done, failed=failed)

    monkeypatch.setattr(window, "_task", capture_task)
    window.save_zip()
    assert window._exporting
    callbacks["failed"]("The chosen destination is read-only.", "export traceback")

    assert not window._exporting
    assert window._active, "An export error must not restore inputs while SAM is still running"
    assert window._cancel_event is original_cancel_event
    assert window.run_button.text() == "Cancel"
    assert window.run_button.isEnabled()
    assert all(not control.isEnabled() for control in window._input_controls)
    assert window.results._busy
    # Release the simulated analysis before pytest closes the window.
    window._cancelled("")


def test_cancel_during_remote_preflight_wins_over_late_network_failure(qtbot, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings, check_setup=False)
    qtbot.addWidget(window)
    window.remote_address = "http://localhost:8765"
    window.remote_token = "session-token"
    window.compute_combo.setCurrentIndex(1)
    window.input_model.add([InputImage(tmp_path / "micrograph.png", 100, 100)])
    window._update_ready()
    callbacks = {}

    def capture_task(work, done, failed=None):
        callbacks.update(done=done, failed=failed or window._failed)

    monkeypatch.setattr(window, "_task", capture_task)
    window.run_or_cancel()
    assert window._active
    window.request_cancel()
    assert window.run_button.text() == "Cancelling…"
    callbacks["failed"]("Can't reach the server.", "network traceback")
    assert not window._active
    assert window._terminal_status.startswith("Cancelled")
    assert window.run_button.text() == "Run"
    assert window.run_button.isEnabled()


@pytest.mark.parametrize("outcome", ["done", "failed"])
def test_connection_test_handoff_is_discarded_after_settings_dialog_closes(qtbot, outcome):
    callbacks = {}

    def capture_task(work, done, failed=None):
        callbacks.update(done=done, failed=failed)

    dialog = RemoteSettingsDialog("http://localhost:8765", "session-token", capture_task)
    dialog.show()
    dialog.test_connection()
    assert not dialog.test_button.isEnabled()
    dialog.reject()
    # MainWindow deletes the settings dialog as soon as exec() returns. The
    # network worker may complete afterward, while its Python closure survives.
    shiboken6.delete(dialog)
    assert not shiboken6.isValid(dialog)
    if outcome == "done":
        callbacks["done"]({"device": "cpu", "version": "0.1.0"})
    else:
        callbacks["failed"]("Can't reach the server.", "network traceback")
