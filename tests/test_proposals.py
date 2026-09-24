"""Tests for Step 2 proposals (spec 6)."""

from types import SimpleNamespace

import numpy as np
from skimage.draw import disk

from lnpsam import io as lnp_io
from lnpsam import proposals


SAM_CFG = {
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
    checkpoint = tmp_path / "sam_vit_b.pth"
    checkpoint.write_bytes(b"checkpoint identity fixture")
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
        "checkpoint": checkpoint,
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


def test_proposal_cache_invalidates_when_cleanup_changes(tmp_path):
    gray, exclusion, content, cbox = _panel(shape=(20, 20))
    sam_image = np.zeros((20, 20, 3), dtype=np.uint8)
    cleanup = SimpleNamespace(
        sam_image=sam_image,
        exclusion=exclusion,
        content_mask=content,
        content_box=cbox,
        crop_offset=(0, 0),
    )
    checkpoint = tmp_path / "sam_vit_b.pth"
    checkpoint.write_bytes(b"checkpoint identity fixture")

    first_key, _ = proposals.proposal_cache_key(
        gray, cleanup, SAM_CFG, checkpoint, "cpu"
    )
    cleanup.exclusion[0, 0] = True
    second_key, _ = proposals.proposal_cache_key(
        gray, cleanup, SAM_CFG, checkpoint, "cpu"
    )
    assert first_key != second_key
