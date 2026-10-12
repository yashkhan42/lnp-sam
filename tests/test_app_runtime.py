"""Desktop worker contracts without a GPU, Qt, or real SAM weights."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import zipfile

import pytest
import yaml

from lnpsam import app_runtime as runtime


@pytest.fixture
def setup_files(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test model identity")
    config = {"sam": {"checkpoint": "model.pth", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
              "measure": {"min_particles_for_stats": 5}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    image = tmp_path / "micrograph.png"
    image.write_bytes(b"faked image; tests inject the pipeline")
    return config_path, checkpoint, image


def _outputs(folder, scale=1.5):
    measure = Path(folder) / "measure"
    measure.mkdir(exist_ok=True)
    panel = {"panel_id": "micrograph", "cal_source": "manual" if scale else "missing",
             "nm_per_px": scale, "n_particles": 8}
    (measure / "micrograph__panel.json").write_text(json.dumps(panel))
    (measure / "micrograph__overlay.png").write_bytes(b"faked overlay")
    (Path(folder) / "particles_all.csv").write_text("panel_id,particle_id\nmicrograph,1\n")
    (Path(folder) / "panels_all.csv").write_text("panel_id,n_particles\nmicrograph,8\n")


def test_configuration_and_weights_are_checked_without_science_imports(setup_files):
    config_path, checkpoint, _ = setup_files
    code = """
import importlib.abc
import sys
class ForbidScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'segment_anything', 'PySide6', 'numpy', 'pandas', 'scipy', 'matplotlib'}:
            raise AssertionError('Science import during setup: ' + fullname)
sys.meta_path.insert(0, ForbidScience())
from lnpsam.app_runtime import check_local_setup
result = check_local_setup(sys.argv[1])
assert result['checkpoint'] == sys.argv[2]
assert len(result['checkpoint_sha256']) == 64
"""
    completed = subprocess.run([sys.executable, "-c", code, str(config_path), str(checkpoint)],
                               cwd=runtime.REPO_ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_checkpoint_override_and_mismatch(setup_files):
    config_path, checkpoint, _ = setup_files
    assert runtime.load_app_config(config_path)["sam"]["checkpoint"] == str(checkpoint)
    override = checkpoint.with_name("another.pth")
    assert runtime.load_app_config(config_path, override)["sam"]["checkpoint"] == str(override)
    with pytest.raises(FileNotFoundError, match="SAM model file not found"):
        runtime.check_local_setup(config_path, override)
    checkpoint.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="doesn't match"):
        runtime.check_local_setup(config_path)


def test_remote_identity_does_not_require_local_weights(setup_files):
    config_path, checkpoint, _ = setup_files
    checkpoint.unlink()
    identity = runtime.app_identity(config_path)
    assert identity["version"]
    assert identity["checkpoint_sha256"] == hashlib.sha256(b"test model identity").hexdigest()
    config = yaml.safe_load(config_path.read_text())
    config["sam"]["checkpoint_sha256"] = "<PASTE>"
    config_path.write_text(yaml.safe_dump(config))
    assert runtime.app_identity(config_path)["checkpoint_sha256"] is None
    with pytest.raises(ValueError, match="64 hexadecimal"):
        runtime.check_local_setup(config_path)


def test_prepare_run_copies_distinct_sanitized_images_and_optional_scale(tmp_path):
    first = tmp_path / "one" / "A cell.PNG"
    second = tmp_path / "two" / "A cell.PNG"
    third = tmp_path / "A_cell_2.tif"
    for index, path in enumerate((first, second, third)):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"image {index}".encode())
    root = tmp_path / "runs"
    folder = runtime.prepare_run((first, second, third), "TEM", None, root)
    with (folder / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["panel_id"] for row in rows] == ["A_cell", "A_cell_2", "A_cell_2_2"]
    assert all(row["nm_per_px"] == "" and row["paper_id"] == "app" for row in rows)
    for row, original in zip(rows, (first, second, third)):
        copied = Path(row["image_path"])
        assert copied.parent == folder / "inputs"
        assert copied.read_bytes() == original.read_bytes()
    second_folder = runtime.prepare_run([first], "SEM", 1.25, root)
    assert second_folder != folder
    with (second_folder / "manifest.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert (row["modality"], row["nm_per_px"]) == ("SEM", "1.25")


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_scale_does_not_create_a_run(setup_files, tmp_path, scale):
    _, _, image = setup_files
    runs_root = tmp_path / "runs"
    with pytest.raises(ValueError, match="positive number"):
        runtime.prepare_run([image], "TEM", scale, runs_root)
    assert not runs_root.exists()


def test_run_retries_mps_once_with_same_inputs_and_records_final_metadata(setup_files, tmp_path, monkeypatch):
    config_path, _, image = setup_files
    calls = []
    events = []
    messages = []
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    def fake_pipeline(manifest, config, folder, **kwargs):
        calls.append((manifest, config, folder, kwargs["device"]))
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert all(kwargs[key] for key in ("run_step2", "run_step3", "run_step4", "run_step6"))
        kwargs["progress"](1, 1, "micrograph", "cleanup")
        kwargs["progress"](1, 1, "micrograph", "proposals")
        if kwargs["device"] == "mps":
            raise RuntimeError("Apple GPU failure")
        _outputs(folder)
        cleanup = Path(folder) / "cleanup"
        cleanup.mkdir()
        (cleanup / "micrograph.json").write_text(json.dumps({"width": 768, "height": 512}))

    def load_pipeline():
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        return fake_pipeline, lambda requested: "mps"

    monkeypatch.setattr(runtime, "_load_pipeline", load_pipeline)
    folder = runtime.run_local([image], "TEM", 1.5, config_path, runs_root=tmp_path / "runs",
                               progress=lambda *event: events.append(event), status=messages.append)
    assert [call[3] for call in calls] == ["mps", "cpu"]
    assert calls[0][:3] == calls[1][:3]
    assert "Apple GPU failed, retrying on CPU" in messages
    assert events == [(1, 1, "micrograph", stage) for stage in ("cleanup", "proposals", "cleanup", "proposals")]
    info = json.loads((folder / "run_info.json").read_text())
    assert (info["device"], info["where"], info["fallback_from"]) == ("cpu", "this computer", "mps")
    assert info["config"]["sam"]["checkpoint"] == str(config_path.parent / "model.pth")
    assert info["images"] == [{"panel_id": "micrograph", "cal_source": "manual", "nm_per_px": 1.5,
                               "n_particles": 8, "filename": "micrograph.png", "width": 768, "height": 512}]
    assert "token" not in info
    archive = runtime.export_results_zip(folder, tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as stream:
        exported = json.loads(stream.read("run_info.json"))
        assert exported["images"] == info["images"]
        assert not any(name.startswith(("inputs/", "cleanup/")) for name in stream.namelist())


def test_run_can_prepare_server_destination_and_preserves_prepared_batch(setup_files, tmp_path, monkeypatch):
    config_path, _, image = setup_files
    destination = tmp_path / "job" / "results"

    def pipeline(manifest, config, folder, **kwargs):
        assert Path(manifest).is_file()
        kwargs["progress"](1, 1, "micrograph", "measure")
        _outputs(folder)

    monkeypatch.setattr(runtime, "_load_pipeline", lambda: (pipeline, lambda requested: "cpu"))
    first = runtime.run_local([image], "SEM", 1.5, config_path, run_folder=destination, where="http://lab-gpu:8765")
    manifest_bytes = (first / "manifest.csv").read_bytes()
    assert first == destination
    assert json.loads((first / "run_info.json").read_text())["where"] == "http://lab-gpu:8765"
    image.unlink()
    runtime.run_local([], "TEM", None, config_path, run_folder=destination)
    assert (destination / "manifest.csv").read_bytes() == manifest_bytes


def test_cancellation_propagates_without_retry_and_keeps_partial_folder(setup_files, tmp_path, monkeypatch):
    config_path, _, image = setup_files
    cancel = Event()
    devices = []

    def pipeline(manifest, config, folder, **kwargs):
        devices.append(kwargs["device"])
        (Path(folder) / "partial.txt").write_text("partial result")
        cancel.set()
        kwargs["progress"](1, 1, "micrograph", "proposals")

    monkeypatch.setattr(runtime, "_load_pipeline", lambda: (pipeline, lambda requested: "mps"))
    with pytest.raises(runtime.RunCancelled):
        runtime.run_local([image], "TEM", None, config_path, run_folder=tmp_path / "cancelled", cancel_event=cancel)
    assert devices == ["mps"]
    assert (tmp_path / "cancelled" / "partial.txt").is_file()
    assert not (tmp_path / "cancelled" / "run_info.json").exists()


def test_arbitrary_progress_error_is_not_retried_on_mps(setup_files, tmp_path, monkeypatch):
    config_path, _, image = setup_files
    devices = []

    def pipeline(manifest, config, folder, **kwargs):
        devices.append(kwargs["device"])
        kwargs["progress"](1, 1, "micrograph", "cleanup")

    def fail_progress(*args):
        raise LookupError("consumer failed")

    monkeypatch.setattr(runtime, "_load_pipeline", lambda: (pipeline, lambda requested: "mps"))
    with pytest.raises(LookupError, match="consumer failed"):
        runtime.run_local([image], "TEM", None, config_path, runs_root=tmp_path / "runs", progress=fail_progress)
    assert devices == ["mps"]


def test_status_callback_exception_is_not_retried_on_mps(setup_files, tmp_path, monkeypatch):
    config_path, _, image = setup_files
    devices = []

    def pipeline(manifest, config, folder, **kwargs):
        devices.append(kwargs["device"])

    def fail_status(message):
        if message == "Apple GPU":
            raise LookupError("status consumer failed")

    monkeypatch.setattr(runtime, "_load_pipeline", lambda: (pipeline, lambda requested: "mps"))
    with pytest.raises(LookupError, match="status consumer failed"):
        runtime.run_local([image], "TEM", None, config_path, runs_root=tmp_path / "runs", status=fail_status)
    assert devices == []


def test_cpu_failure_is_propagated_without_retry(setup_files, tmp_path, monkeypatch):
    config_path, _, image = setup_files
    devices = []

    def pipeline(manifest, config, folder, **kwargs):
        devices.append(kwargs["device"])
        raise RuntimeError("scientific pipeline failure")

    monkeypatch.setattr(runtime, "_load_pipeline", lambda: (pipeline, lambda requested: "cpu"))
    with pytest.raises(RuntimeError, match="scientific pipeline failure"):
        runtime.run_local([image], "TEM", None, config_path, runs_root=tmp_path / "runs")
    assert devices == ["cpu"]


def test_export_contains_only_final_results(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    _outputs(folder)
    (folder / "run_info.json").write_text("{}")
    (folder / "inputs").mkdir()
    (folder / "inputs" / "private.png").write_bytes(b"input")
    (folder / "cleanup").mkdir()
    (folder / "cleanup" / "intermediate.json").write_text("{}")
    archive = runtime.export_results_zip(folder, tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == {"particles_all.csv", "panels_all.csv", "run_info.json",
                                          "measure/micrograph__panel.json", "measure/micrograph__overlay.png"}


def test_export_refuses_incomplete_results_and_external_symlinks(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        runtime.export_results_zip(folder, tmp_path / "results.zip")
    _outputs(folder)
    (folder / "run_info.json").write_text("{}")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (folder / "measure" / "link.txt").symlink_to(outside)
    destination = tmp_path / "results.zip"
    destination.write_bytes(b"previous export")
    with pytest.raises(ValueError, match="unsafe file"):
        runtime.export_results_zip(folder, destination)
    assert destination.read_bytes() == b"previous export"
