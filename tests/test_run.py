"""CLI-stage integration tests without loading a real SAM checkpoint."""

import hashlib
import json
import shutil

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
    config["sam"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
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
    assert bool(summary.loc[0, "proposals_validated"])
    assert summary.loc[0, "proposals_source"] == str(out_dir / "proposals" / "toy_panel__nodes.parquet")
    assert summary.loc[0, "step2_seconds"] >= 0
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


@pytest.fixture
def hpc_mac_runs(tmp_path, monkeypatch):
    """Produce a real cache with fake HPC SAM, then forbid Mac SAM/device access."""
    pixels = np.full((60, 60, 3), 200, dtype=np.uint8)
    rr, cc = disk((30, 30), 10, shape=(60, 60))
    pixels[rr, cc] = 60
    image_path = tmp_path / "panel.png"
    Image.fromarray(pixels).save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"panel_id": "toy_panel", "paper_id": "toy", "image_path": str(image_path),
                   "modality": "TEM", "nm_per_px": None}]).to_csv(manifest_path, index=False)

    class FakeReader:
        def readtext(self, *args, **kwargs):
            return []

    class FakeGenerator:
        def generate(self, image):
            particle = image[:, :, 0] < 130
            return [
                {"segmentation": mask, "predicted_iou": 0.95, "stability_score": 0.95,
                 "point_coords": [[30.0, 30.0]]}
                for mask in (particle, ~particle)
            ]

    config = lnp_io.load_config("config.yaml")
    hpc_checkpoint = tmp_path / "A" / "sam.pth"
    config["sam"]["checkpoint"] = str(hpc_checkpoint)
    config["sam"]["checkpoint_sha256"] = hashlib.sha256(b"test SAM checkpoint").hexdigest()
    hpc_config = tmp_path / "hpc_config.yaml"
    with hpc_config.open("w") as stream:
        yaml.safe_dump(config, stream)

    monkeypatch.setattr(run_mod.cleanup_mod, "get_reader", lambda: FakeReader())
    monkeypatch.setattr(run_mod.proposals_mod, "select_device", lambda requested: "cuda")
    monkeypatch.setattr(run_mod.proposals_mod, "load_sam_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_mod.proposals_mod, "make_mask_generator", lambda *args, **kwargs: FakeGenerator())
    hpc = tmp_path / "hpc_run"
    run_mod.run_cleanup_stage(manifest_path, hpc_config, hpc, run_step2=True, device="cuda")

    mac_checkpoint = tmp_path / "B" / "sam.pth"
    config["sam"]["checkpoint"] = str(mac_checkpoint)
    mac_config = tmp_path / "mac_config.yaml"
    with mac_config.open("w") as stream:
        yaml.safe_dump(config, stream)

    def no_sam(*args, **kwargs):
        raise AssertionError("SAM/device access must not occur with --proposals-from")

    monkeypatch.setattr(run_mod.proposals_mod, "load_sam_model", no_sam)
    monkeypatch.setattr(run_mod.proposals_mod, "select_device", no_sam)
    monkeypatch.setattr(run_mod.proposals_mod, "make_mask_generator", no_sam)
    assert not hpc_checkpoint.exists() and not mac_checkpoint.exists()
    return {
        "manifest": manifest_path,
        "config": mac_config,
        "hpc": hpc,
        "out": tmp_path / "local",
        "meta": hpc / "proposals" / "toy_panel__meta.json",
        "nodes": hpc / "proposals" / "toy_panel__nodes.parquet",
        "tmp": tmp_path,
    }


def _run_mac(paths, proposals_from=None, allow_stale=False):
    args = [
        "--manifest", str(paths["manifest"]), "--config", str(paths["config"]),
        "--out", str(paths["out"]), "--proposals-from", str(proposals_from or paths["hpc"]),
    ]
    if allow_stale:
        args.append("--allow-stale-proposals")
    run_mod.main(args)


def _change_mac_sam_config(paths, **changes):
    config = lnp_io.load_config(paths["config"])
    config["sam"].update(changes)
    with paths["config"].open("w") as stream:
        yaml.safe_dump(config, stream)


def _copy_proposals(paths, root, run_name):
    target = root / run_name
    shutil.copytree(paths["hpc"], target)
    return target / "proposals" / "toy_panel__meta.json"


def test_proposals_from_reuses_nodes_without_sam(hpc_mac_runs):
    """The HPC and Mac keys agree despite nonexistent, different model paths."""
    paths = hpc_mac_runs
    source_meta = json.loads(paths["meta"].read_text())
    _run_mac(paths)
    particles = pd.read_csv(paths["out"] / "particles_all.csv")
    assert len(particles) == 1 and particles.loc[0, "panel_id"] == "toy_panel"
    summary = pd.read_csv(paths["out"] / "proposals" / "proposals_all.csv")
    assert summary.loc[0, "n_nodes"] == source_meta["n_nodes"]
    assert summary.loc[0, "cache_key"] == source_meta["cache_key"]
    assert bool(summary.loc[0, "proposals_validated"])
    assert summary.loc[0, "device"] == "cuda"
    assert summary.loc[0, "proposals_source"] == str(paths["nodes"])
    assert summary.loc[0, "step2_seconds"] == pytest.approx(source_meta["timing"]["total_s"])
    assert json.loads(paths["meta"].read_text()) == source_meta


def test_proposals_from_refuses_changed_params(hpc_mac_runs, capsys):
    _change_mac_sam_config(hpc_mac_runs, pred_iou_thresh=0.71)
    with pytest.raises(SystemExit) as exc:
        _run_mac(hpc_mac_runs)
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "toy_panel" in stderr and "step2_params" in stderr
    assert "Traceback" not in stderr


def test_proposals_from_refuses_changed_step1(hpc_mac_runs, monkeypatch, capsys):
    real_cleanup = run_mod.cleanup_mod.run_cleanup

    def changed_cleanup(*args, **kwargs):
        result = real_cleanup(*args, **kwargs)
        result.exclusion[0, 0] = ~result.exclusion[0, 0]
        return result

    monkeypatch.setattr(run_mod.cleanup_mod, "run_cleanup", changed_cleanup)
    with pytest.raises(SystemExit) as exc:
        _run_mac(hpc_mac_runs)
    assert exc.value.code == 2
    assert "exclusion" in capsys.readouterr().err


def test_proposals_from_refuses_legacy_cache(hpc_mac_runs, capsys):
    meta_path = hpc_mac_runs["meta"]
    metadata = json.loads(meta_path.read_text())
    metadata.pop("key_schema")
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit) as exc:
        _run_mac(hpc_mac_runs)
    assert exc.value.code == 2
    assert "older LNP-SAM" in capsys.readouterr().err


def test_proposals_from_picks_matching_run_not_first(hpc_mac_runs):
    paths = hpc_mac_runs
    root = paths["tmp"] / "all_runs"
    stale_meta_path = _copy_proposals(paths, root, "aaa_stale")
    good_meta_path = _copy_proposals(paths, root, "zzz_good")
    metadata = json.loads(stale_meta_path.read_text())
    metadata["cache_key"] = "stale"
    metadata["input_digests"]["step2_params"] = "stale"
    stale_meta_path.write_text(json.dumps(metadata))
    _run_mac(paths, proposals_from=root)
    summary = pd.read_csv(paths["out"] / "proposals" / "proposals_all.csv")
    assert summary.loc[0, "proposals_source"] == str(good_meta_path.with_name("toy_panel__nodes.parquet"))
    assert bool(summary.loc[0, "proposals_validated"])


def test_proposals_from_uses_newest_matching_run(hpc_mac_runs, capsys):
    paths = hpc_mac_runs
    root = paths["tmp"] / "all_runs"
    for name, created in (
        ("aaa_old", "2026-10-11T02:00:00+01:00"),
        ("zzz_new", "2026-10-11T01:30:00Z"),
        ("000_unknown", ["invalid timestamp"]),
    ):
        meta_path = _copy_proposals(paths, root, name)
        metadata = json.loads(meta_path.read_text())
        metadata["created_utc"] = created
        meta_path.write_text(json.dumps(metadata))
    _run_mac(paths, proposals_from=root)
    summary = pd.read_csv(paths["out"] / "proposals" / "proposals_all.csv")
    assert "zzz_new" in summary.loc[0, "proposals_source"]
    stdout = capsys.readouterr().out
    assert "newest matching" in stdout and "zzz_new" in stdout


def test_allow_stale_proposals_one_candidate(hpc_mac_runs, capsys):
    paths = hpc_mac_runs
    _change_mac_sam_config(paths, pred_iou_thresh=0.71)
    _run_mac(paths, allow_stale=True)
    assert (paths["out"] / "particles_all.csv").is_file()
    summary = pd.read_csv(paths["out"] / "proposals" / "proposals_all.csv")
    assert not bool(summary.loc[0, "proposals_validated"])
    assert summary.loc[0, "device"] == "cuda"
    assert "WARNING: toy_panel" in capsys.readouterr().out


def test_allow_stale_proposals_two_candidates(hpc_mac_runs, capsys):
    paths = hpc_mac_runs
    root = paths["tmp"] / "all_runs"
    _copy_proposals(paths, root, "one")
    _copy_proposals(paths, root, "two")
    _change_mac_sam_config(paths, pred_iou_thresh=0.71)
    with pytest.raises(SystemExit) as exc:
        _run_mac(paths, proposals_from=root, allow_stale=True)
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "point DIR at a single run" in stderr
    assert "one/proposals" in stderr and "two/proposals" in stderr


def test_allow_stale_proposals_requires_source(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_mod.main([
            "--manifest", "unused.csv", "--config", "unused.yaml", "--out", str(tmp_path / "unused"),
            "--allow-stale-proposals",
        ])
    assert exc.value.code == 2
    assert "requires --proposals-from" in capsys.readouterr().err
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("contents", [None, "{broken JSON", "[]", "null"])
def test_bad_proposal_metadata_requires_override(hpc_mac_runs, capsys, contents):
    paths = hpc_mac_runs
    if contents is None:
        paths["meta"].unlink()
    else:
        paths["meta"].write_text(contents)
    with pytest.raises(SystemExit) as exc:
        _run_mac(paths)
    assert exc.value.code == 2
    assert "no __meta.json" in capsys.readouterr().err
    _run_mac(paths, allow_stale=True)
    summary = pd.read_csv(paths["out"] / "proposals" / "proposals_all.csv")
    assert not bool(summary.loc[0, "proposals_validated"])
    assert pd.isna(summary.loc[0, "device"])
    assert pd.isna(summary.loc[0, "step2_seconds"])
    assert "WARNING: toy_panel" in capsys.readouterr().out


@pytest.mark.parametrize("allow_stale", [False, True])
def test_proposals_from_missing_candidate(tmp_path, allow_stale):
    # Files from measure/ must not become Step-2 candidates.
    measure = tmp_path / "measure"
    measure.mkdir()
    (measure / "toy_panel__nodes.parquet").touch()
    with pytest.raises(FileNotFoundError, match="no proposals/toy_panel__nodes.parquet"):
        run_mod.select_proposal_nodes(tmp_path, "toy_panel", "expected", {}, allow_stale)
