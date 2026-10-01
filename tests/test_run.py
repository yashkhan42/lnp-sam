"""CLI-stage integration tests without loading a real SAM checkpoint."""

import pandas as pd
import yaml
from PIL import Image

from lnpsam import io as lnp_io
from lnpsam import run as run_mod


def test_steps_1_to_4_write_outputs_and_skip_sam_on_rerun(tmp_path, monkeypatch):
    image_path = tmp_path / "panel.png"
    Image.new("RGB", (40, 40), (128, 128, 128)).save(image_path)
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
            mask = image[:, :, 0] == 128
            return [
                {
                    "segmentation": mask,
                    "predicted_iou": 0.9,
                    "stability_score": 0.9,
                    "point_coords": [[20.0, 20.0]],
                }
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
        device="cpu",
    )
    run_mod.run_cleanup_stage(
        manifest_path,
        config_path,
        out_dir,
        run_step2=True,
        run_step3=True,
        run_step4=True,
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
    for name in (
        "toy_panel__roles.csv",
        "toy_panel__panel.json",
        "toy_panel__overlay.png",
        "toy_panel__overlay_ids.png",
    ):
        assert (out_dir / "resolve" / name).is_file()
    roles = pd.read_csv(out_dir / "resolve" / "toy_panel__roles.csv")
    assert roles["role"].notna().all() and roles["rule_id"].notna().all()
    assert (out_dir / "resolve" / "resolve_all.csv").is_file()
