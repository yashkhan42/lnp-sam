"""CLI-stage integration tests without loading a real SAM checkpoint."""

import json

import numpy as np
import pytest
import pandas as pd
import yaml
from PIL import Image
from skimage.draw import disk

from lnpsam import io as lnp_io
from lnpsam import measure as measure_mod
from lnpsam import run as run_mod


def test_steps_1_to_6_write_outputs_and_skip_sam_on_rerun(tmp_path, monkeypatch):
    # TEM-like toy: one dark particle on a light background.
    pixels = np.full((60, 60, 3), 200, dtype=np.uint8)
    rr, cc = disk((30, 30), 10, shape=(60, 60))
    pixels[rr, cc] = 60
    image_path = tmp_path / "panel.png"
    Image.fromarray(pixels).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "panel_id": "toy_panel",
                "paper_id": "toy",
                "image_path": str(image_path),
                "modality": "TEM",
                "nm_per_px": None,
            }
        ]
    ).to_csv(manifest_path, index=False)

    checkpoint = tmp_path / "sam_vit_b.pth"
    checkpoint.write_bytes(b"checkpoint identity fixture")
    config = lnp_io.load_config("config.yaml")
    config["sam"]["checkpoint"] = str(checkpoint)
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as stream:
        yaml.safe_dump(config, stream)

    class FakeReader:
        def readtext(self, *args, **kwargs):
            return []

    calls = {"load": 0, "generate": 0, "graph": 0}

    class FakeGenerator:
        def generate(self, image):
            calls["generate"] += 1
            particle = image[:, :, 0] < 130
            return [
                {
                    "segmentation": seg,
                    "predicted_iou": 0.95,
                    "stability_score": 0.95,
                    "point_coords": [[30.0, 30.0]],
                }
                for seg in (particle, ~particle)
            ]

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        return object()

    monkeypatch.setattr(run_mod.cleanup_mod, "get_reader", lambda: FakeReader())
    monkeypatch.setattr(run_mod.proposals_mod, "load_sam_model", fake_load)
    monkeypatch.setattr(
        run_mod.proposals_mod,
        "make_mask_generator",
        lambda *args, **kwargs: FakeGenerator(),
    )
    real_run_graph = run_mod.graph_mod.run_graph

    def tracked_run_graph(*args, **kwargs):
        calls["graph"] += 1
        return real_run_graph(*args, **kwargs)

    monkeypatch.setattr(run_mod.graph_mod, "run_graph", tracked_run_graph)

    out_dir = tmp_path / "run"
    run_mod.run_cleanup_stage(
        manifest_path,
        config_path,
        out_dir,
        run_step2=True,
        run_step3=True,
        run_step4=True,
        run_step6=True,
        device="cpu",
    )
    run_mod.run_cleanup_stage(
        manifest_path,
        config_path,
        out_dir,
        run_step2=True,
        run_step3=True,
        run_step4=True,
        run_step6=True,
        device="cpu",
    )

    assert calls == {"load": 1, "generate": 1, "graph": 2}
    assert (out_dir / "proposals" / "toy_panel__nodes.parquet").is_file()
    assert (out_dir / "proposals" / "toy_panel__meta.json").is_file()
    assert (out_dir / "proposals" / "toy_panel__overlay.png").is_file()
    summary = pd.read_csv(out_dir / "proposals" / "proposals_all.csv")
    assert bool(summary.loc[0, "cache_hit"])
    assert summary.loc[0, "device"] == "cpu"
    assert summary.loc[0, "overlay_path"].endswith("toy_panel__overlay.png")
    # Review trace (resolve/): every mask, its role and the all-roles overlays.
    for name in ("roles.csv", "resolve.json", "nodes.parquet", "overlay.png", "overlay_ids.png"):
        assert (out_dir / "resolve" / f"toy_panel__{name}").is_file()
    assert (out_dir / "resolve" / "resolve_all.csv").is_file()
    roles = pd.read_csv(out_dir / "resolve" / "toy_panel__roles.csv")
    assert roles["role"].notna().all() and roles["rule_id"].notna().all()
    nodes = pd.read_parquet(out_dir / "resolve" / "toy_panel__nodes.parquet")
    sam = nodes[~nodes["virtual"]]
    assert set(sam["role"]) == {"particle", "background"}
    # v4: the particle is a hole in the background mask; one member, so C-AGG drops it.
    assert set(nodes.loc[nodes["virtual"], "rule_id"]) == {"R-SYNTH-DROP"}

    # Final results (measure/, spec 10 v3): only what is reported.
    measure = out_dir / "measure"
    for name in ("particles.csv", "panel.json", "particles.tif", "overlay.png", "overlay_ids.png"):
        assert (measure / f"toy_panel__{name}").is_file()
    assert not (measure / "toy_panel__nodes.parquet").exists()
    assert not (measure / "toy_panel__aggregates.tif").exists()  # no aggregates here
    particles = pd.read_csv(measure / "toy_panel__particles.csv")
    assert len(particles) == 1
    assert particles.loc[0, "d_px"] == pytest.approx(2 * np.sqrt(len(rr) / np.pi), rel=0.1)
    assert bool(particles.loc[0, "in_stats"])
    ids, stack = measure_mod.read_mask_stack(measure / "toy_panel__particles.tif")
    assert ids == [int(particles.loc[0, "particle_id"])] and stack.shape == (1, 60, 60)
    assert stack[0].sum() == particles.loc[0, "area_px"]
    panel = json.loads((measure / "toy_panel__panel.json").read_text())
    assert (panel["n_particles"], panel["cal_source"], panel["stats_unit"]) == (1, "missing", "px")
    assert panel["stats_all"] is None  # fewer than measure.min_particles_for_stats
    assert len(pd.read_csv(out_dir / "particles_all.csv")) == 1
    assert len(pd.read_csv(out_dir / "panels_all.csv")) == 1


def test_proposals_from_reuses_nodes_without_sam(tmp_path, monkeypatch):
    """--proposals-from: Steps 1, 3, 4, 6 run on existing Step-2 nodes; SAM never loads."""
    pixels = np.full((60, 60, 3), 200, dtype=np.uint8)
    rr, cc = disk((30, 30), 10, shape=(60, 60))
    pixels[rr, cc] = 60
    image_path = tmp_path / "panel.png"
    Image.fromarray(pixels).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"panel_id": "toy_panel", "paper_id": "toy", "image_path": str(image_path),
                   "modality": "TEM", "nm_per_px": None}]).to_csv(manifest_path, index=False)

    # Step-2 nodes as an HPC run would leave them: <somewhere>/proposals/<id>__nodes.parquet.
    from lnpsam import proposals as proposals_mod

    gray, _ = lnp_io.load_image(image_path)
    particle = gray < 0.5
    nodes = proposals_mod.postprocess_proposals(
        [{"segmentation": m, "predicted_iou": 0.95, "stability_score": 0.95, "point_coords": [[30.0, 30.0]]}
         for m in (particle, ~particle)],
        gray, np.zeros(gray.shape, bool), np.ones(gray.shape, bool), (0, 0, 60, 60), (0, 0),
        lnp_io.load_config("config.yaml")["sam"],
    )
    hpc = tmp_path / "hpc_run" / "toy_panel" / "proposals"
    proposals_mod.save_proposal_cache(hpc, "toy_panel", "key", {}, "cuda", nodes, raw_count=2)

    class FakeReader:
        def readtext(self, *args, **kwargs):
            return []

    def no_sam(*args, **kwargs):
        raise AssertionError("SAM must not load with --proposals-from")

    monkeypatch.setattr(run_mod.cleanup_mod, "get_reader", lambda: FakeReader())
    monkeypatch.setattr(run_mod.proposals_mod, "load_sam_model", no_sam)
    monkeypatch.setattr(run_mod.proposals_mod, "select_device", no_sam)

    out_dir = tmp_path / "local"
    run_mod.main(["--manifest", str(manifest_path), "--config", "config.yaml", "--out", str(out_dir),
                  "--proposals-from", str(tmp_path / "hpc_run")])

    particles = pd.read_csv(out_dir / "particles_all.csv")
    assert len(particles) == 1 and particles.loc[0, "panel_id"] == "toy_panel"
    summary = pd.read_csv(out_dir / "proposals" / "proposals_all.csv")
    assert summary.loc[0, "n_nodes"] == len(nodes) and summary.loc[0, "cache_key"] == "key"
