"""Results views exercise the saved scientific data without importing SAM."""

import csv
import json
import os
import shutil
import subprocess
import sys
import threading

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPersistentModelIndex, Qt
from PySide6.QtGui import QColor, QImage

from lnpsam.gui_results import (
    DecodedImageCache, EM_DASH, ParticleTableModel, ResultsWidget,
    _prepare_panel, _read_run,
)


def _png(path, width, height):
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor("#567890"))
    assert image.save(str(path))


@pytest.fixture
def run_folder(tmp_path):
    measure = tmp_path / "measure"
    cleanup = tmp_path / "cleanup"
    measure.mkdir()
    cleanup.mkdir()
    for name in ("alpha", "empty"):
        _png(measure / f"{name}__overlay.png", 20, 14)
        _png(measure / f"{name}__overlay_ids.png", 60, 50)
        (cleanup / f"{name}.json").write_text(json.dumps({"width": 20, "height": 10}))
    (measure / "alpha__panel.json").write_text(json.dumps({
        "panel_id": "alpha", "nm_per_px": 10, "cal_source": "manual",
        "stats_unit": "nm", "n_particles": 3, "n_in_stats": 2, "flags": [],
        # Deliberately distinct saved numbers to check the GUI does not invent stats.
        "stats_all": {"n": 2, "mean": 63.45, "median": 61.23,
                      "sd": 4.56, "d10": 21.23, "d90": 99.87},
    }))
    (measure / "empty__panel.json").write_text(json.dumps({
        "panel_id": "empty", "nm_per_px": None, "cal_source": "missing",
        "stats_unit": "px", "n_particles": 0, "n_in_stats": 0,
        "stats_all": None, "flags": ["UNCALIBRATED", "TOO_FEW_FOR_STATS"],
    }))
    fields = ["particle_id", "panel_id", "d_px", "d_nm", "hollow", "in_stats",
              "truncated", "touches_exclusion", "aggregate_id", "flags"]
    with (tmp_path / "particles_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for particle_id, diameter, used, aggregate in ((1, 2, True, ""), (2, 10, True, 100), (3, 300, False, "")):
            writer.writerow({
                "particle_id": particle_id, "panel_id": "alpha", "d_px": diameter,
                "d_nm": diameter * 10, "hollow": particle_id == 2,
                "in_stats": used, "truncated": not used, "touches_exclusion": False,
                "aggregate_id": aggregate,
                "flags": "POSSIBLE_CORE_ONLY" if not used else "",
            })
    with (tmp_path / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel_id", "image_path"])
        writer.writeheader()
        writer.writerows([{"panel_id": name, "image_path": f"inputs/{name}.tif"} for name in ("alpha", "empty")])
    return tmp_path


def test_startup_imports_no_science_and_constructs_no_canvas(tmp_path):
    code = """
import sys
from PySide6.QtWidgets import QApplication
from lnpsam.gui_results import ResultsWidget
app = QApplication([])
widget = ResultsWidget()
assert widget._canvas is None
assert not widget.save_button.isEnabled()
assert not widget.open_button.isEnabled()
assert all(name not in sys.modules for name in ('torch', 'segment_anything', 'matplotlib', 'pandas', 'numpy'))
assert widget.shutdown()
"""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLCONFIGDIR=str(tmp_path))
    subprocess.run([sys.executable, "-c", code], check=True, env=environment, timeout=20)


def test_index_uses_source_geometry_and_handles_unequal_legends(run_folder):
    run = _read_run(run_folder)
    panel = run.panels[0]
    assert panel.name == "alpha.tif"
    assert panel.source_size == (20, 10)
    assert panel.extent == pytest.approx((20, 50 / 3))
    assert panel.rows == (0, 1, 2)


def test_exported_run_metadata_retains_source_geometry_and_filename(run_folder, tmp_path):
    exported = tmp_path / "exported"
    shutil.copytree(run_folder / "measure", exported / "measure")
    shutil.copy(run_folder / "particles_all.csv", exported / "particles_all.csv")
    (exported / "run_info.json").write_text(json.dumps({"images": [
        {"panel_id": "alpha", "width": 20, "height": 10, "filename": "microscope-original.tif"},
    ]}))
    run = _read_run(exported)
    assert run.panels[0].source_size == (20, 10)
    assert run.panels[0].name == "microscope-original.tif"
    assert run.panels[0].extent == pytest.approx((20, 50 / 3))


def test_histogram_uses_in_stats_and_retains_saved_stats(run_folder):
    run = _read_run(run_folder)
    display = _prepare_panel(run, run.panels[0], DecodedImageCache())
    assert display.unit == "nm"
    assert display.sample_count == 2
    assert sum(display.counts) == 2
    assert display.edges[0] == 20
    assert display.edges[-1] == 100  # An aggregate member contributes once; excluded d=3000 does not.
    assert display.panel.summary["stats_all"]["median"] == 61.23
    assert "Some particles may be detected by their core only." in display.warnings


def test_empty_uncalibrated_results_preserve_headers_and_warnings(run_folder):
    run = _read_run(run_folder)
    display = _prepare_panel(run, run.panels[1], DecodedImageCache())
    assert display.unit == "px"
    assert not display.counts
    assert display.sample_count == 0
    assert "No particles detected in this image." in display.warnings
    assert "Too few particles for statistics on this image." in display.warnings
    model = ParticleTableModel()
    model.set_panel(run.particles, display.panel)
    assert model.rowCount() == 0
    assert model.headerData(1, Qt.Orientation.Horizontal) == "Diameter (px)"


def test_model_sorts_raw_numbers_and_displays_booleans_nulls(run_folder):
    import pandas as pd
    run = _read_run(run_folder)
    model = ParticleTableModel()
    model.set_panel(run.particles, run.panels[0])
    assert model.frame is run.particles
    assert model.headerData(1, Qt.Orientation.Horizontal) == "Diameter (nm)"
    assert model.data(model.index(0, 2)) == "No"
    assert model.data(model.index(1, 2)) == "Yes"
    assert model.data(model.index(0, 6)) == EM_DASH
    selected = QPersistentModelIndex(model.index(0, 0))
    model.sort(1, Qt.SortOrder.DescendingOrder)
    assert [model.data(model.index(i, 1), Qt.ItemDataRole.UserRole) for i in range(3)] == [3000, 100, 20]
    assert list(run.particles.particle_id) == [1, 2, 3]
    assert selected.row() == 2
    assert selected.data() == "1"
    model.sort(6, Qt.SortOrder.DescendingOrder)
    assert model.data(model.index(0, 6)) == "100"
    assert model.data(model.index(1, 6)) == EM_DASH
    run.particles["flags"] = run.particles["flags"].astype("string")
    run.particles.at[0, "flags"] = pd.NA
    flags_column = model.columns.index("flags")
    row = model.rows.index(0)
    assert model.data(model.index(row, flags_column)) == EM_DASH


def test_decoded_cache_counts_rgba_evicts_and_previews(tmp_path):
    paths = [tmp_path / f"{number}.png" for number in range(3)]
    for path in paths:
        _png(path, 20, 20)
    cache = DecodedImageCache(budget=3200)
    for path in paths:
        cache.read(path)
    assert cache.bytes_used == 3200
    assert paths[0] not in cache._images
    large = tmp_path / "large.png"
    _png(large, 200, 100)
    preview = cache.read(large)
    assert preview.width() < 200
    assert preview.sizeInBytes() <= 3200
    assert cache.bytes_used <= cache.budget


def test_async_load_atomic_views_and_ids_preserve_source_transform(qtbot, run_folder):
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    widget.resize(900, 750)
    widget.show()
    with qtbot.waitSignal(widget.ready, timeout=10000):
        widget.load_run(run_folder)
    assert widget.current_run == run_folder
    assert widget.summary_values["Median"].text() == "61.23"
    assert widget.summary_values["Used in statistics"].text() == "2"
    assert widget._canvas is not None
    canvas = widget._canvas
    assert widget.current_display.panel.ids_path not in widget.cache._images
    widget.view.actual_size()
    widget.view.centerOn(8, 6)
    transform = widget.view.transform()
    center = widget.view.mapToScene(widget.view.viewport().rect().center())
    extent = widget.view.sceneRect()
    widget.ids_checkbox.setChecked(True)
    qtbot.waitUntil(lambda: widget.view.item.pixmap().width() == 60, timeout=5000)
    assert widget.view.item.scale() == pytest.approx(1 / 3)
    assert widget.view.transform() == transform
    assert widget.view.mapToScene(widget.view.viewport().rect().center()) == center
    assert widget.view.sceneRect() == extent
    assert len(widget.view.scene().items()) == 1
    widget.set_busy(True)
    assert widget.run_label.text() == "Previous run"
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.image_picker.setCurrentIndex(1)
    assert widget.summary_values["Units"].text() == "px"
    assert widget.summary_values["Median"].text() == EM_DASH
    assert widget.summary_values["Scale"].text() == "Not found · sizes in pixels"
    assert widget.particle_model.rowCount() == 0
    assert widget._canvas is canvas
    assert "No particles detected" in widget.warning_label.text()
    assert widget.shutdown()


def test_obsolete_panel_load_is_discarded(qtbot, run_folder, monkeypatch):
    import lnpsam.gui_results as module
    original = module._prepare_panel
    gate = threading.Event()
    entered = threading.Event()

    def prepare(run, panel, cache):
        if panel.panel_id == "alpha":
            entered.set()
            assert gate.wait(10)
        return original(run, panel, cache)

    monkeypatch.setattr(module, "_prepare_panel", prepare)
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    widget.load_run(run_folder)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.image_picker.setCurrentIndex(1)
    assert widget.current_display.panel.panel_id == "empty"
    gate.set()
    qtbot.waitUntil(lambda: not widget._tasks, timeout=5000)
    assert widget.current_display.panel.panel_id == "empty"
    assert widget.particle_model.rowCount() == 0
    assert widget.shutdown()


def test_run_and_panel_reads_happen_off_gui_thread(qtbot, run_folder, monkeypatch):
    import lnpsam.gui_results as module
    seen = []
    main_thread = threading.get_ident()
    for name in ("_read_run", "_prepare_panel"):
        original = getattr(module, name)

        def wrapped(*args, _original=original):
            seen.append(threading.get_ident())
            return _original(*args)

        monkeypatch.setattr(module, name, wrapped)
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.load_run(run_folder)
    assert len(seen) == 2
    assert all(thread != main_thread for thread in seen)
    assert widget.shutdown()


def test_loading_new_run_invalidates_inflight_previous_panel(qtbot, run_folder, tmp_path, monkeypatch):
    import lnpsam.gui_results as module
    second = tmp_path / "second"
    shutil.copytree(run_folder / "measure", second / "measure")
    (second / "measure" / "alpha__panel.json").unlink()
    shutil.copy(run_folder / "particles_all.csv", second / "particles_all.csv")
    original = module._prepare_panel
    entered = threading.Event()
    gate = threading.Event()

    def prepare(run, panel, cache):
        if run.folder == run_folder:
            entered.set()
            assert gate.wait(10)
        return original(run, panel, cache)

    monkeypatch.setattr(module, "_prepare_panel", prepare)
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    widget.load_run(run_folder)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.load_run(second)
    assert widget.current_run == second
    assert widget.current_display.panel.panel_id == "empty"
    gate.set()
    qtbot.waitUntil(lambda: not widget._tasks, timeout=5000)
    assert widget.current_run == second
    assert widget.current_display.panel.panel_id == "empty"
    assert widget.shutdown()


def test_obsolete_ids_decode_cannot_replace_base(qtbot, run_folder, monkeypatch):
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.load_run(run_folder)
    original = widget.cache.read
    entered = threading.Event()
    gate = threading.Event()

    def read(path):
        if path.name.endswith("__overlay_ids.png"):
            entered.set()
            assert gate.wait(10)
        return original(path)

    monkeypatch.setattr(widget.cache, "read", read)
    widget.ids_checkbox.setChecked(True)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    widget.ids_checkbox.setChecked(False)
    qtbot.waitUntil(lambda: len(widget._tasks) == 1, timeout=5000)
    gate.set()
    qtbot.waitUntil(lambda: not widget._tasks, timeout=5000)
    assert not widget.ids_checkbox.isChecked()
    assert widget.view.item.pixmap().width() == 20
    assert widget.view.item.scale() == 1
    assert widget.shutdown()


def test_small_analysis_area_stacks_and_scrolls_complete_chart_and_summary(qtbot, run_folder):
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    widget.resize(720, 600)
    widget.show()
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.load_run(run_folder)
    widget.splitter.setSizes([999, 160])
    qtbot.waitUntil(lambda: widget.distribution_scroll.verticalScrollBar().maximum() > 0)
    assert widget.distribution_splitter.orientation() == Qt.Orientation.Vertical
    assert widget._canvas.height() >= 180
    scale = widget.summary_values["Scale"]
    widget.distribution_scroll.ensureWidgetVisible(scale, 0, 0)
    qtbot.waitUntil(lambda: widget.distribution_scroll.viewport().rect().contains(
        scale.mapTo(widget.distribution_scroll.viewport(), scale.rect().center())))
    assert scale.height() >= scale.fontMetrics().height()
    widget.resize(1000, 600)
    qtbot.waitUntil(lambda: widget.distribution_splitter.orientation() == Qt.Orientation.Horizontal)
    assert widget._canvas.height() >= 180
    assert widget.shutdown()


def test_frozen_ids_share_selection_and_remain_visible_during_scroll(qtbot, run_folder):
    csv_path = run_folder / "particles_all.csv"
    with csv_path.open(newline="") as handle:
        original_rows = list(csv.DictReader(handle))
    rows = [dict(row, particle_id=index + 1) for index, row in enumerate(original_rows * 20)]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    widget.resize(900, 650)
    widget.show()
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.load_run(run_folder)
    widget.tabs.setCurrentIndex(1)
    table = widget.table
    frozen = table.frozen_column
    assert frozen.model() is table.model() is widget.particle_model
    assert frozen.selectionModel() is table.selectionModel()
    assert all(frozen.isColumnHidden(column) for column in range(1, widget.particle_model.columnCount()))
    table.setColumnWidth(0, 82)
    qtbot.waitUntil(lambda: frozen.columnWidth(0) == 82)
    table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
    qtbot.waitUntil(lambda: table.visualRect(table.model().index(0, 0)).right() < 0)
    assert frozen.visualRect(frozen.model().index(0, 0)).left() == 0
    assert frozen.visualRect(frozen.model().index(0, 0)).right() < frozen.viewport().width()
    table.verticalScrollBar().setValue(table.verticalScrollBar().maximum())
    qtbot.waitUntil(lambda: frozen.verticalScrollBar().value() == table.verticalScrollBar().value())
    last_index = frozen.model().index(59, 0)
    qtbot.mouseClick(frozen.viewport(), Qt.MouseButton.LeftButton, pos=frozen.visualRect(last_index).center())
    assert table.selectionModel().selectedRows()[0].row() == 59
    assert table.selectionModel().selectedRows()[0].data() == "60"
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.image_picker.setCurrentIndex(1)
    assert table.columnWidth(0) == frozen.columnWidth(0) == 82
    assert widget.shutdown()


def test_default_results_size_fits_horizontal_distribution_without_scroll(qtbot, run_folder):
    widget = ResultsWidget()
    qtbot.addWidget(widget)
    widget.show()
    # Default main window leaves 978 px for results and 768 px below chrome.
    widget.resize(978, 768)
    with qtbot.waitSignal(widget.ready, timeout=5000):
        widget.load_run(run_folder)
    qtbot.waitUntil(lambda: widget.distribution_splitter.orientation() == Qt.Orientation.Horizontal)
    qtbot.waitUntil(lambda: widget.distribution_scroll.verticalScrollBar().maximum() == 0)
    viewport = widget.distribution_scroll.viewport()
    scale = widget.summary_values["Scale"]
    assert viewport.rect().contains(scale.mapTo(viewport, scale.rect().bottomRight()))
    assert widget._canvas.height() >= 180
    assert widget.shutdown()
