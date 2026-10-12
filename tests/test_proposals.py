"""Tests for Step 2 proposals (spec 6)."""

from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest
from skimage.draw import disk

from lnpsam import io as lnp_io
from lnpsam import proposals


SAM_CFG = {
    "checkpoint": "/fixture/weights/sam_vit_b.pth",
    "checkpoint_sha256": "0" * 64,
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "box_nms_thresh": 0.95,
    "min_piece_px": 4,
    "excl_drop_frac": 0.5,
    "bg_area_frac": 0.25,
}


def _ann(mask, predicted_iou=0.9, stability_score=0.9, point=(0.0, 0.0)):
    return {
        "segmentation": mask,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
        "point_coords": [list(point)],
    }


def _panel(shape=(40, 50), content_box=None):
    height, width = shape
    if content_box is None:
        content_box = (0, 0, width, height)
    gray = np.full(shape, 0.8, dtype=np.float32)
    exclusion = np.zeros(shape, dtype=bool)
    content = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = content_box
    content[y0:y1, x0:x1] = True
    return gray, exclusion, content, content_box


def test_mask_generator_uses_exact_spec_parameters():
    captured = {}

    class FakeGenerator:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

    model = object()
    proposals.make_mask_generator(model, SAM_CFG, generator_cls=FakeGenerator)
    assert captured == {
        "model": model,
        "points_per_side": 64,
        "points_per_batch": 128,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.85,
        "box_nms_thresh": 0.95,
        "crop_n_layers": 0,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask",
    }


def test_auto_device_prefers_cuda_then_mps_then_cpu():
    def fake_torch(cuda, mps):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )

    assert proposals.select_device("auto", fake_torch(True, True)) == "cuda"
    assert proposals.select_device("auto", fake_torch(False, True)) == "mps"
    assert proposals.select_device("auto", fake_torch(False, False)) == "cpu"


def test_split_fill_holes_and_full_panel_coordinates():
    gray, exclusion, content, cbox = _panel(shape=(30, 40), content_box=(5, 4, 35, 26))
    crop = np.zeros((22, 30), dtype=bool)
    # Largest component: a ring with a >=4-pixel hole.
    rr, cc = disk((10, 10), 6, shape=crop.shape)
    crop[rr, cc] = True
    crop[8:12, 8:12] = False
    # Second retained component and one sub-threshold component.
    crop[17:19, 23:26] = True
    crop[1, 25:28] = True

    nodes = proposals.postprocess_proposals(
        [_ann(crop, point=(10.0, 10.0))],
        gray,
        exclusion,
        content,
        cbox,
        crop_offset=(5, 4),
        sam_config=SAM_CFG,
    )

    assert len(nodes) == 2
    largest, second = nodes
    assert largest["node_id"] == 0
    assert second["node_id"] == 1
    assert largest["n_holes"] == 1
    assert largest["filled"].sum() > largest["mask"].sum()
    assert largest["point_coords"] == [[15.0, 14.0]]
    assert largest["rle"] != largest["filled_rle"]
    assert lnp_io.decode_rle(largest["rle"]).shape == gray.shape
    assert largest["bbox"][0] >= 5 and largest["bbox"][1] >= 4
    assert second["area"] == 6


def test_exclusion_drop_and_touch_flag_are_based_on_filled_area():
    gray, exclusion, content, cbox = _panel(shape=(30, 30))
    mostly_excluded = np.zeros_like(gray, dtype=bool)
    mostly_excluded[5:9, 5:9] = True
    partly_excluded = np.zeros_like(gray, dtype=bool)
    partly_excluded[15:20, 15:20] = True
    exclusion[5:9, 5:13] = True
    exclusion[15, 15] = True
    content &= ~exclusion

    nodes = proposals.postprocess_proposals(
        [_ann(mostly_excluded), _ann(partly_excluded)],
        gray,
        exclusion,
        content,
        cbox,
        crop_offset=(0, 0),
        sam_config=SAM_CFG,
    )

    assert [node["sam_annotation_id"] for node in nodes] == [1]
    assert nodes[0]["touches_exclusion"] is True
    assert nodes[0]["excl_overlap"] == 1 / 25


def test_exact_duplicate_keeps_higher_quality():
    gray, exclusion, content, cbox = _panel()
    mask = np.zeros_like(gray, dtype=bool)
    mask[10:20, 15:25] = True
    nodes = proposals.postprocess_proposals(
        [
            _ann(mask, predicted_iou=0.81, stability_score=0.86),
            _ann(mask, predicted_iou=0.95, stability_score=0.96),
        ],
        gray,
        exclusion,
        content,
        cbox,
        crop_offset=(0, 0),
        sam_config=SAM_CFG,
    )

    assert len(nodes) == 1
    assert nodes[0]["sam_annotation_id"] == 1
    assert nodes[0]["predicted_iou"] == 0.95


def test_border_and_large_masks_are_kept_and_background_is_flagged():
    gray, exclusion, content, cbox = _panel(shape=(20, 20))
    mask = np.zeros_like(gray, dtype=bool)
    mask[:12, :12] = True  # large and touches top + left; neither property drops it
    nodes = proposals.postprocess_proposals(
        [_ann(mask)],
        gray,
        exclusion,
        content,
        cbox,
        crop_offset=(0, 0),
        sam_config=SAM_CFG,
    )

    assert len(nodes) == 1
    assert nodes[0]["touches_border"] is True
    assert nodes[0]["bg_candidate"] is True


def test_run_proposals_uses_cleanup_sam_image_and_injected_generator():
    gray, exclusion, content, cbox = _panel(shape=(20, 20))
    sam_image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    class FakeGenerator:
        def generate(self, image):
            assert image is sam_image
            return [_ann(mask)]

    cleanup = SimpleNamespace(
        sam_image=sam_image,
        exclusion=exclusion,
        content_mask=content,
        content_box=cbox,
        crop_offset=(0, 0),
    )
    nodes = proposals.run_proposals(
        gray, cleanup, {"sam": SAM_CFG}, mask_generator=FakeGenerator()
    )
    assert len(nodes) == 1


def test_proposal_cache_skips_generator_when_upstream_is_unchanged(tmp_path):
    gray, exclusion, content, cbox = _panel(shape=(20, 20))
    sam_image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    cleanup = SimpleNamespace(
        sam_image=sam_image,
        exclusion=exclusion,
        content_mask=content,
        content_box=cbox,
        crop_offset=(0, 0),
    )
    calls = {"factory": 0, "generate": 0}

    class FakeGenerator:
        def generate(self, image):
            calls["generate"] += 1
            return [_ann(mask)]

    def factory():
        calls["factory"] += 1
        return FakeGenerator()

    args = {
        "gray": gray,
        "cleanup_result": cleanup,
        "cfg": {"sam": SAM_CFG},
        "panel_id": "toy_panel",
        "cache_dir": tmp_path / "proposals",
        "device": "cpu",
        "mask_generator_factory": factory,
    }
    first_nodes, first_hit, first_meta = proposals.run_proposals_cached(**args)
    second_nodes, second_hit, second_meta = proposals.run_proposals_cached(**args)

    assert first_hit is False
    assert second_hit is True
    assert calls == {"factory": 1, "generate": 1}
    assert first_meta["cache_key"] == second_meta["cache_key"]
    assert first_nodes[0]["filled_rle"] == second_nodes[0]["filled_rle"]
    assert np.array_equal(first_nodes[0]["filled"], second_nodes[0]["filled"])


def test_proposal_cache_invalidates_when_cleanup_changes():
    gray, exclusion, content, cbox = _panel(shape=(20, 20))
    sam_image = np.zeros((20, 20, 3), dtype=np.uint8)
    cleanup = SimpleNamespace(
        sam_image=sam_image,
        exclusion=exclusion,
        content_mask=content,
        content_box=cbox,
        crop_offset=(0, 0),
    )
    first_key = proposals.proposal_cache_key(gray, cleanup, SAM_CFG)
    cleanup.exclusion[0, 0] = True
    second_key = proposals.proposal_cache_key(gray, cleanup, SAM_CFG)
    assert first_key != second_key


def _cache_inputs():
    gray, exclusion, content, cbox = _panel(shape=(20, 20))
    cleanup = SimpleNamespace(
        sam_image=np.zeros((20, 20, 3), dtype=np.uint8),
        exclusion=exclusion,
        content_mask=content,
        content_box=cbox,
        crop_offset=(0, 0),
    )
    return gray, cleanup


def _cached_run_args(tmp_path):
    gray, cleanup = _cache_inputs()
    mask = np.zeros(gray.shape, dtype=bool)
    mask[5:10, 5:10] = True
    calls = {"factory": 0, "generate": 0}

    class FakeGenerator:
        def generate(self, image):
            calls["generate"] += 1
            return [_ann(mask)]

    def factory():
        calls["factory"] += 1
        return FakeGenerator()

    return {
        "gray": gray,
        "cleanup_result": cleanup,
        "cfg": {"sam": dict(SAM_CFG)},
        "panel_id": "toy_panel",
        "cache_dir": tmp_path / "proposals",
        "device": "cpu",
        "mask_generator_factory": factory,
    }, calls


def test_proposal_key_is_portable_without_checkpoint_or_sam(monkeypatch):
    gray, cleanup = _cache_inputs()
    hpc_config = {**SAM_CFG, "checkpoint": "/hpc/A/sam.pth"}
    mac_config = {**SAM_CFG, "checkpoint": "/mac/B/sam.pth"}

    def forbidden(*args, **kwargs):
        raise AssertionError("portable cache keys must not inspect runtime packages or files")

    monkeypatch.setattr(proposals.importlib.metadata, "version", forbidden)
    monkeypatch.setattr(proposals.Path, "read_bytes", forbidden)
    monkeypatch.setattr(proposals.Path, "stat", forbidden)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "segment_anything", None)
    assert proposals.proposal_cache_key(gray, cleanup, hpc_config) == (
        proposals.proposal_cache_key(gray, cleanup, mac_config)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("points_per_side", 32),
        ("points_per_batch", 64),
        ("pred_iou_thresh", 0.79),
        ("stability_score_thresh", 0.84),
        ("box_nms_thresh", 0.94),
        ("min_piece_px", 5),
        ("excl_drop_frac", 0.6),
        ("bg_area_frac", 0.3),
        ("checkpoint_sha256", "1" * 64),
    ],
)
def test_proposal_key_tracks_sam_inputs(field, value):
    gray, cleanup = _cache_inputs()
    original = proposals.proposal_cache_key(gray, cleanup, SAM_CFG)
    changed = proposals.proposal_cache_key(gray, cleanup, {**SAM_CFG, field: value})
    assert original != changed


@pytest.mark.parametrize(
    "field",
    ["gray", "sam_image", "exclusion", "content_mask", "content_box", "crop_offset"],
)
def test_proposal_key_tracks_step1_inputs(field):
    gray, cleanup = _cache_inputs()
    original_key = proposals.proposal_cache_key(gray, cleanup, SAM_CFG)
    original_digests = proposals.proposal_input_digests(gray, cleanup, SAM_CFG)
    if field == "gray":
        gray[0, 0] = 0.2
    elif field == "sam_image":
        cleanup.sam_image[0, 0, 0] = 17
    elif field == "exclusion":
        cleanup.exclusion[0, 0] = True
    elif field == "content_mask":
        cleanup.content_mask[0, 0] = False
    elif field == "content_box":
        cleanup.content_box = (1, 0, 20, 20)
    else:
        cleanup.crop_offset = (1, 0)
    changed_digests = proposals.proposal_input_digests(gray, cleanup, SAM_CFG)
    expected_field = "geometry" if field in {"content_box", "crop_offset"} else field
    assert {name for name in original_digests if original_digests[name] != changed_digests[name]} == {
        expected_field
    }
    assert proposals.proposal_cache_key(gray, cleanup, SAM_CFG) != original_key


def test_proposal_key_tracks_step2_version(monkeypatch):
    gray, cleanup = _cache_inputs()
    original = proposals.proposal_cache_key(gray, cleanup, SAM_CFG)
    monkeypatch.setattr(proposals, "STEP2_VERSION", proposals.STEP2_VERSION + 1)
    assert proposals.proposal_cache_key(gray, cleanup, SAM_CFG) != original


def test_input_digests_and_key_use_canonical_json():
    gray, cleanup = _cache_inputs()
    digests = proposals.proposal_input_digests(gray, cleanup, SAM_CFG)
    assert set(digests) == {
        "gray", "sam_image", "exclusion", "content_mask", "geometry", "step2_params",
        "checkpoint_sha256", "step2_version",
    }
    params = {key: value for key, value in SAM_CFG.items()
              if key not in {"checkpoint", "checkpoint_sha256"}}
    canonical_params = json.dumps(params, sort_keys=True, separators=(",", ":"))
    assert digests["step2_params"] == hashlib.sha256(canonical_params.encode()).hexdigest()
    geometry = {"content_box": list(cleanup.content_box), "crop_offset": list(cleanup.crop_offset)}
    canonical_geometry = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    assert digests["geometry"] == hashlib.sha256(canonical_geometry.encode()).hexdigest()
    assert digests["checkpoint_sha256"] == SAM_CFG["checkpoint_sha256"]
    assert digests["step2_version"] == str(proposals.STEP2_VERSION)
    key_json = json.dumps({"key_schema": 2, **digests}, sort_keys=True, separators=(",", ":"))
    expected_key = hashlib.sha256(key_json.encode()).hexdigest()
    assert proposals.proposal_cache_key(gray, cleanup, SAM_CFG) == expected_key
    reversed_cfg = dict(reversed(list(SAM_CFG.items())))
    assert proposals.proposal_cache_key(gray, cleanup, reversed_cfg) == expected_key


def test_array_digests_preserve_existing_casts_and_contiguous_layout():
    gray, cleanup = _cache_inputs()
    first = proposals.proposal_input_digests(gray, cleanup, SAM_CFG)
    cleanup.exclusion = cleanup.exclusion.astype(np.uint8)
    cleanup.content_mask = cleanup.content_mask.astype(np.uint8)
    second = proposals.proposal_input_digests(
        np.asfortranarray(gray.astype(np.float64)), cleanup, SAM_CFG
    )
    assert first == second
    for name, array in (
        ("gray", np.asarray(gray, dtype=np.float32)),
        ("sam_image", cleanup.sam_image),
        ("exclusion", np.asarray(cleanup.exclusion, dtype=np.uint8)),
        ("content_mask", np.asarray(cleanup.content_mask, dtype=np.uint8)),
    ):
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
        assert first[name] == digest.hexdigest()


@pytest.mark.parametrize("value", [None, "0" * 63, "g" * 64, "<PASTE>", 123])
def test_checkpoint_sha_validation(value):
    gray, cleanup = _cache_inputs()
    cfg = dict(SAM_CFG)
    if value is None:
        cfg.pop("checkpoint_sha256")
    else:
        cfg["checkpoint_sha256"] = value
    with pytest.raises(ValueError) as error:
        proposals.proposal_cache_key(gray, cleanup, cfg)
    assert "sha256sum" in str(error.value)
    assert "shasum -a 256" in str(error.value)


def test_checkpoint_sha_uppercase_normalizes_to_lowercase():
    gray, cleanup = _cache_inputs()
    lower_cfg = {**SAM_CFG, "checkpoint_sha256": "abcdef01" * 8}
    upper_cfg = {**lower_cfg, "checkpoint_sha256": lower_cfg["checkpoint_sha256"].upper()}
    assert proposals._require_checkpoint_sha256(upper_cfg) == lower_cfg["checkpoint_sha256"]
    assert proposals.proposal_cache_key(gray, cleanup, upper_cfg) == (
        proposals.proposal_cache_key(gray, cleanup, lower_cfg)
    )


def test_verify_checkpoint_matching_and_mismatching_hashes(tmp_path):
    checkpoint = tmp_path / "sam.pth"
    payload = b"checkpoint bytes" * 100_000  # More than one streaming chunk.
    checkpoint.write_bytes(payload)
    actual = hashlib.sha256(payload).hexdigest()
    proposals.verify_checkpoint(checkpoint, actual)
    wrong = "0" * 64
    with pytest.raises(ValueError) as error:
        proposals.verify_checkpoint(checkpoint, wrong)
    message = str(error.value)
    assert str(checkpoint) in message
    assert actual in message
    assert wrong in message


def test_load_sam_verifies_checkpoint_before_constructing_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam.pth"
    payload = b"model fixture"
    checkpoint.write_bytes(payload)
    calls = []
    model = SimpleNamespace(
        to=lambda **kwargs: calls.append(("to", kwargs)),
        eval=lambda: calls.append(("eval",)),
    )

    def factory(**kwargs):
        calls.append(("load", kwargs))
        return model

    monkeypatch.setitem(sys.modules, "segment_anything", SimpleNamespace(
        sam_model_registry={"vit_b": factory}
    ))
    with pytest.raises(ValueError):
        proposals.load_sam_model(checkpoint, "cpu", expected_sha256="0" * 64)
    assert calls == []
    assert proposals.load_sam_model(
        checkpoint, "cpu", expected_sha256=hashlib.sha256(payload).hexdigest()
    ) is model
    assert calls == [
        ("load", {"checkpoint": str(checkpoint)}),
        ("to", {"device": "cpu"}),
        ("eval",),
    ]


def test_local_legacy_cache_misses_even_with_matching_key(tmp_path):
    args, calls = _cached_run_args(tmp_path)
    _, first_hit, first_meta = proposals.run_proposals_cached(**args)
    assert first_hit is False
    _, meta_path = proposals.proposal_cache_paths(args["cache_dir"], args["panel_id"])
    legacy = {key: value for key, value in first_meta.items() if key != "key_schema"}
    meta_path.write_text(json.dumps(legacy))
    assert proposals.load_proposal_cache(
        args["cache_dir"], args["panel_id"], first_meta["cache_key"]
    ) is None
    _, second_hit, second_meta = proposals.run_proposals_cached(**args)
    assert second_hit is False
    assert second_meta["key_schema"] == 2
    assert calls == {"factory": 2, "generate": 2}


def test_step2_output_fingerprint():
    # Pinned using the pre-cache-fix implementation; exercises splitting, hole
    # filling, duplicate removal, exclusion, background flags and crop offsets.
    shape = (30, 40)
    crop_shape = (22, 30)
    crop_offset = (5, 4)
    content_box = (5, 4, 35, 26)
    rows, cols = np.indices(shape)
    gray = ((rows + 2 * cols) / 110.0).astype(np.float32)
    gray[9:18, 9:20] *= np.float32(0.4)
    exclusion = np.zeros(shape, dtype=bool)
    exclusion[4:8, 26:30] = True
    exclusion[21, 8] = True
    content = np.zeros(shape, dtype=bool)
    content[4:26, 5:35] = True
    content &= ~exclusion

    ring = np.zeros(crop_shape, dtype=bool)
    rr, cc = disk((10, 10), 6, shape=crop_shape)
    ring[rr, cc] = True
    ring[8:12, 8:12] = False
    split = ring.copy()
    split[17:19, 23:26] = True
    split[1, 25:28] = True
    dropped = np.zeros(crop_shape, dtype=bool)
    dropped[0:4, 21:25] = True
    background = np.zeros(crop_shape, dtype=bool)
    background[0:14, 0:16] = True
    partly_excluded = np.zeros(crop_shape, dtype=bool)
    partly_excluded[17:20, 3:7] = True
    annotations = [
        _ann(split, 0.93, 0.94, (10.0, 10.0)),
        _ann(ring, 0.81, 0.86, (11.0, 10.0)),
        _ann(dropped, 0.95, 0.96, (23.0, 2.0)),
        _ann(background, 0.85, 0.89, (8.0, 7.0)),
        _ann(partly_excluded, 0.90, 0.91, (5.0, 18.0)),
    ]
    nodes = proposals.postprocess_proposals(
        annotations, gray, exclusion, content, content_box, crop_offset, SAM_CFG
    )
    records = [
        {key: round(value, 6) if isinstance(value, float) else value
         for key, value in proposals._node_to_cache_record(node).items()}
        for node in nodes
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == "e5fcaf938c12ba62bbeb0dd56b543821be6be31bf7f7b62721fdcff0d520761e", (
        "Step-2 output changed. If intended, bump STEP2_VERSION in proposals.py "
        "and update this constant."
    )


_PROVENANCE_FIELDS = {
    "device", "accelerator", "cpu_count", "torch_threads", "torch",
    "segment_anything", "python", "os", "lnpsam_version", "git_commit",
}


def test_cache_metadata_records_provenance_timing_and_utc(tmp_path):
    args, _ = _cached_run_args(tmp_path)
    _, cache_hit, metadata = proposals.run_proposals_cached(**args)
    assert cache_hit is False
    assert metadata["key_schema"] == 2
    assert metadata["step2_version"] == proposals.STEP2_VERSION
    assert metadata["input_digests"] == proposals.proposal_input_digests(
        args["gray"], args["cleanup_result"], args["cfg"]["sam"]
    )
    assert len(metadata["input_digests"]) == 8
    assert metadata["device"] == "cpu"
    assert metadata["checkpoint_path"] == str(proposals.Path(SAM_CFG["checkpoint"]).resolve())
    assert metadata["n_raw_masks"] == metadata["n_nodes"] == 1
    assert set(metadata["provenance"]) == _PROVENANCE_FIELDS
    assert metadata["provenance"]["device"] == "cpu"
    timing = metadata["timing"]
    assert set(timing) == {"model_load_s", "generate_s", "postprocess_s", "total_s"}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in timing.values())
    assert all(round(value, 3) == value for value in timing.values())
    assert timing["total_s"] == pytest.approx(
        timing["model_load_s"] + timing["generate_s"] + timing["postprocess_s"], abs=0.003
    )
    from datetime import datetime, timezone

    assert metadata["created_utc"].endswith("Z")
    created = datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00"))
    assert created.tzinfo == timezone.utc


def test_cache_hit_preserves_source_metadata_on_other_device(tmp_path, monkeypatch):
    args, calls = _cached_run_args(tmp_path)
    args["device"] = "cuda"
    _, _, first_meta = proposals.run_proposals_cached(**args)
    _, meta_path = proposals.proposal_cache_paths(args["cache_dir"], args["panel_id"])
    stored = json.loads(meta_path.read_text())
    args["device"] = "cpu"
    args["cfg"]["sam"]["checkpoint"] = "/another/machine/sam.pth"

    def forbidden(*args, **kwargs):
        raise AssertionError("a cache hit must preserve its source metadata")

    monkeypatch.setattr(proposals, "_collect_provenance", forbidden)
    _, cache_hit, second_meta = proposals.run_proposals_cached(**args)
    assert cache_hit is True
    assert second_meta == stored == first_meta
    assert second_meta["device"] == "cuda"
    assert calls == {"factory": 1, "generate": 1}


def _fake_provenance_environment(monkeypatch, dirty=True):
    cuda_indices = []

    def device_name(index):
        cuda_indices.append(index)
        return "Fixture GPU"

    fake_torch = SimpleNamespace(
        __version__="fixture-torch",
        get_num_threads=lambda: 3,
        cuda=SimpleNamespace(get_device_name=device_name),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(proposals.os, "cpu_count", lambda: 7)
    monkeypatch.setattr(proposals.platform, "processor", lambda: "Fixture CPU")
    monkeypatch.setattr(proposals.platform, "machine", lambda: "fixture-machine")
    monkeypatch.setattr(proposals.platform, "python_version", lambda: "3.fixture")
    monkeypatch.setattr(proposals.platform, "platform", lambda: "Fixture OS")
    monkeypatch.setattr(proposals.importlib.metadata, "version", lambda package: "fixture-sam")

    def git_run(command, **kwargs):
        assert kwargs["timeout"] == 2
        assert kwargs["cwd"] is not None
        stdout = "deadbeef\n" if "rev-parse" in command else (
            " M lnpsam/proposals.py\n" if dirty else ""
        )
        return SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(proposals.subprocess, "run", git_run)
    return fake_torch, cuda_indices


@pytest.mark.parametrize(
    "device,accelerator,index",
    [
        ("cpu", "Fixture CPU", None),
        ("mps", "Apple MPS", None),
        ("cuda", "Fixture GPU", None),
        ("cuda:2", "Fixture GPU", 2),
    ],
)
def test_provenance_identifies_accelerator_and_environment(monkeypatch, device, accelerator, index):
    _, indices = _fake_provenance_environment(monkeypatch)
    provenance = proposals._collect_provenance(device)
    assert set(provenance) == _PROVENANCE_FIELDS
    assert provenance["device"] == device
    assert provenance["accelerator"] == accelerator
    assert indices == ([index] if device.startswith("cuda") else [])
    assert provenance["cpu_count"] == 7
    assert provenance["torch_threads"] == 3
    assert provenance["torch"] == "fixture-torch"
    assert provenance["segment_anything"] == "fixture-sam"
    assert provenance["python"] == "3.fixture"
    assert provenance["os"] == "Fixture OS"
    assert provenance["git_commit"] == "deadbeef+dirty"
    from lnpsam import __version__

    assert provenance["lnpsam_version"] == __version__ == "0.1.0"


def test_provenance_cpu_falls_back_to_machine(monkeypatch):
    _fake_provenance_environment(monkeypatch, dirty=False)
    monkeypatch.setattr(proposals.platform, "processor", lambda: "")
    provenance = proposals._collect_provenance("cpu")
    assert provenance["accelerator"] == "fixture-machine"
    assert provenance["git_commit"] == "deadbeef"


@pytest.mark.parametrize(
    "operation,field",
    [
        ("cpu_count", "cpu_count"),
        ("get_num_threads", "torch_threads"),
        ("get_device_name", "accelerator"),
        ("python_version", "python"),
        ("platform", "os"),
        ("version", "segment_anything"),
        ("run", "git_commit"),
    ],
)
def test_provenance_field_failure_never_raises(monkeypatch, operation, field):
    fake_torch, _ = _fake_provenance_environment(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable environment field")

    owners = {
        "cpu_count": proposals.os,
        "get_num_threads": fake_torch,
        "get_device_name": fake_torch.cuda,
        "python_version": proposals.platform,
        "platform": proposals.platform,
        "version": proposals.importlib.metadata,
        "run": proposals.subprocess,
    }
    monkeypatch.setattr(owners[operation], operation, fail)
    provenance = proposals._collect_provenance("cuda:2")
    assert provenance[field] is None
    assert set(provenance) == _PROVENANCE_FIELDS
    assert provenance["device"] == "cuda:2"
    assert provenance["lnpsam_version"] == "0.1.0"


def test_provenance_cpu_identification_failure_is_null(monkeypatch):
    _fake_provenance_environment(monkeypatch)

    def fail():
        raise RuntimeError("CPU identity unavailable")

    monkeypatch.setattr(proposals.platform, "processor", fail)
    assert proposals._collect_provenance("cpu")["accelerator"] is None


def test_provenance_without_torch_or_sam(monkeypatch):
    _fake_provenance_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)

    def missing(package):
        raise proposals.importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(proposals.importlib.metadata, "version", missing)
    provenance = proposals._collect_provenance("cuda")
    assert provenance["torch"] is None
    assert provenance["torch_threads"] is None
    assert provenance["accelerator"] is None
    assert provenance["segment_anything"] == "missing"
    assert provenance["python"] == "3.fixture"


def test_provenance_outside_git_checkout_is_null(monkeypatch):
    _fake_provenance_environment(monkeypatch)

    def no_git(*args, **kwargs):
        raise proposals.subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(proposals.subprocess, "run", no_git)
    assert proposals._collect_provenance("cpu")["git_commit"] is None
