"""Pipeline progress reports actual stage boundaries and propagates cancellation."""

import hashlib

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from lnpsam import io as lnp_io
from lnpsam import run as run_mod


@pytest.fixture
def batch(tmp_path, monkeypatch):
    pixels = np.full((60, 60, 3), 200, dtype=np.uint8)
    yy, xx = np.ogrid[:60, :60]
    pixels[(yy - 30) ** 2 + (xx - 30) ** 2 < 10 ** 2] = 60
    records = []
    for panel_id in ("first", "second"):
        image = tmp_path / f"{panel_id}.png"
        Image.fromarray(pixels).save(image)
        records.append({"panel_id": panel_id, "paper_id": "app", "image_path": str(image),
                        "modality": "TEM", "nm_per_px": 1.0})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    config = lnp_io.load_config("config.yaml")
    config["sam"]["checkpoint"] = str(tmp_path / "model.pth")
    config["sam"]["checkpoint_sha256"] = hashlib.sha256(b"fake checkpoint").hexdigest()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    class Reader:
        def readtext(self, *args, **kwargs):
            return []

    class Generator:
        def generate(self, image):
            particle = image[:, :, 0] < 130
            return [{"segmentation": mask, "predicted_iou": 0.95, "stability_score": 0.95,
                     "point_coords": [[30.0, 30.0]]} for mask in (particle, ~particle)]

    calls = []
    monkeypatch.setattr(run_mod.cleanup_mod, "get_reader", lambda: Reader())
    monkeypatch.setattr(run_mod.proposals_mod, "select_device", lambda device: "cpu")
    monkeypatch.setattr(run_mod.proposals_mod, "load_sam_model", lambda *args, **kwargs: calls.append("sam"))
    monkeypatch.setattr(run_mod.proposals_mod, "make_mask_generator", lambda *args, **kwargs: Generator())
    return manifest, config_path, tmp_path, calls


def _run(batch, out, **kwargs):
    manifest, config, _, _ = batch
    return run_mod.run_cleanup_stage(manifest, config, out, run_step2=True,
                                    run_step3=True, run_step4=True, run_step6=True,
                                    device="cpu", **kwargs)


def test_progress_for_each_stage_of_each_panel_including_cached_proposals(batch):
    _, _, tmp_path, calls = batch
    expected = [(index, 2, panel_id, stage)
                for index, panel_id in enumerate(("first", "second"), start=1)
                for stage in ("cleanup", "proposals", "graph", "resolve", "measure")]
    events = []
    output = tmp_path / "run"
    _run(batch, output, progress=lambda *args: events.append(args))
    assert events == expected
    assert calls == ["sam"]
    events.clear()
    _run(batch, output, progress=lambda *args: events.append(args))
    assert events == expected
    assert calls == ["sam"]
    assert (output / "particles_all.csv").is_file()


def test_none_progress_preserves_console_output_and_scientific_tables(batch, capsys):
    _, _, tmp_path, _ = batch
    without = tmp_path / "without"
    with_none = tmp_path / "with_none"
    _run(batch, without)
    previous = capsys.readouterr().out
    _run(batch, with_none, progress=None)
    assert capsys.readouterr().out == previous
    for filename in ("particles_all.csv", "panels_all.csv"):
        assert (without / filename).read_bytes() == (with_none / filename).read_bytes()


def test_callback_exception_stops_before_sam_and_propagates(batch):
    _, _, tmp_path, calls = batch
    events = []
    stopped = RuntimeError("cancelled by worker")

    def cancel(index, total, panel, stage):
        events.append((index, total, panel, stage))
        if stage == "proposals":
            raise stopped

    with pytest.raises(RuntimeError) as exc:
        _run(batch, tmp_path / "cancelled", progress=cancel)
    assert exc.value is stopped
    assert events == [(1, 2, "first", "cleanup"), (1, 2, "first", "proposals")]
    assert calls == []
    assert (tmp_path / "cancelled" / "cleanup" / "first.json").is_file()
    assert not (tmp_path / "cancelled" / "particles_all.csv").exists()


def test_cleanup_callback_runs_before_image_is_loaded(batch, monkeypatch):
    _, _, tmp_path, _ = batch

    def forbid_image(*args):
        raise AssertionError("Cancellation must precede image loading")

    def cancel(*args):
        raise InterruptedError("stop at cleanup")

    monkeypatch.setattr(run_mod.lnp_io, "load_image", forbid_image)
    with pytest.raises(InterruptedError, match="stop at cleanup"):
        _run(batch, tmp_path / "cancelled", progress=cancel)
