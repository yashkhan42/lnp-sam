"""Tests for eval/evaluate.py aggregate metrics and criterion-6 tally (spec 12.3, 12.4)."""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from skimage.draw import disk

from baselines.npsam_run import pack_masks, unpack_masks
from eval import evaluate as ev
from lnpsam import measure
from toys import CAVITY, CFG, PARTICLE, SEED_CENTERS, SEED_R, build_panel
from toys import disk_mask as toy_disk

PANEL = "pylypchuk2021_greenchem_04058d__p05_x46__panel00"
SHAPE = (100, 100)


def _disk_mask(center, radius):
    m = np.zeros(SHAPE, dtype=bool)
    rr, cc = disk(center, radius, shape=SHAPE)
    m[rr, cc] = True
    return m


def _write_gt(gt_dir, with_aggregates):
    """GT: 3 spheres inside one aggregate + 1 isolated sphere."""
    panel_dir = gt_dir / PANEL
    panel_dir.mkdir(parents=True)
    particles = np.zeros(SHAPE, dtype=np.uint16)
    for label, center in enumerate([(40, 35), (40, 55), (58, 45), (85, 85)], start=1):
        particles[_disk_mask(center, 9)] = label
    tifffile.imwrite(panel_dir / "particles.tif", particles)
    if with_aggregates:
        aggregates = np.zeros(SHAPE, dtype=np.uint16)
        aggregates[_disk_mask((46, 45), 22)] = 1
        tifffile.imwrite(panel_dir / "aggregates.tif", aggregates)


def test_aggregate_metrics_when_aggregates_tif_present(tmp_path):
    _write_gt(tmp_path, with_aggregates=True)
    gt_agg = _disk_mask((46, 45), 22)
    pred_aggregates = {7: _disk_mask((47, 46), 22), 8: _disk_mask((85, 85), 9)}
    pred_member_counts = {7: 2, 8: 0}

    records = ev.evaluate_panel_aggregates(tmp_path, PANEL, pred_aggregates, pred_member_counts)

    assert len(records) == 1
    rec = records[0]
    assert rec["pred_aggregate"] == 7
    expected_iou = (gt_agg & pred_aggregates[7]).sum() / (gt_agg | pred_aggregates[7]).sum()
    assert rec["aggregate_iou"] == pytest.approx(expected_iou)
    assert rec["n_gt_members"] == 3  # the isolated sphere is not a member
    assert rec["n_pred_members"] == 2
    assert rec["member_count_error"] == pytest.approx(1 / 3)


def test_aggregate_metrics_no_predicted_aggregate(tmp_path):
    _write_gt(tmp_path, with_aggregates=True)
    records = ev.evaluate_panel_aggregates(tmp_path, PANEL, {}, {})
    assert records[0]["aggregate_iou"] == 0.0
    assert records[0]["pred_aggregate"] is None
    assert records[0]["member_count_error"] == pytest.approx(1.0)


def test_no_aggregate_metrics_when_aggregates_tif_absent(tmp_path):
    _write_gt(tmp_path, with_aggregates=False)
    pred_aggregates = {7: _disk_mask((47, 46), 22)}
    assert ev.evaluate_panel_aggregates(tmp_path, PANEL, pred_aggregates, {7: 2}) is None


def _result(f1, err):
    return {"f1": f1, "median_d_err": err}


def test_criterion6_tally_excludes_tracking_panels():
    better = {"lnpsam": _result(0.9, 0.05), "npsam": _result(0.5, 0.20)}
    worse = {"lnpsam": _result(0.4, 0.30), "npsam": _result(0.5, 0.20)}
    # 12 gating panels: 8 better + control better = 9 -> pass.
    results = {f"golden{i}": better for i in range(8)}
    results.update({f"golden_worse{i}": worse for i in range(3)})
    results[ev.CONTROL_PANEL] = better
    results[PANEL] = worse

    tally = ev.criterion6_tally(results)
    assert tally["n_counted"] == 12
    assert tally["excluded"] == [PANEL]
    assert tally["n_better"] == 9
    assert tally["passed"]

    # A better tracking panel must not rescue an 8/12 result.
    results[ev.CONTROL_PANEL] = {"lnpsam": _result(0.5, 0.20), "npsam": _result(0.5, 0.20)}
    results[PANEL] = better
    tally = ev.criterion6_tally(results)
    assert tally["n_better"] == 8
    assert tally["control_not_worse"]
    assert not tally["passed"]


# --------------------------------------------------------------------------- #
# Core metrics (12.3)
# --------------------------------------------------------------------------- #
FULL_BOX = (0, 0, SHAPE[1], SHAPE[0])


def _labels(*masks):
    lab = np.zeros(SHAPE, dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        lab[m] = i
    return lab


def _gt(particles, cavities=None):
    return {"particles": particles, "cavities": cavities, "aggregates": None}


def test_f1_count_error_and_unmatched():
    gt = [_disk_mask((20, 20), 8), _disk_mask((20, 60), 8), _disk_mask((60, 40), 8)]
    preds = [_disk_mask((20, 21), 8), _disk_mask((21, 60), 8), _disk_mask((85, 85), 4)]
    m = ev.panel_metrics(preds, None, _gt(_labels(*gt)), FULL_BOX)
    assert (m["n_pred"], m["n_gt"], m["tp"]) == (3, 3, 2)
    assert m["f1"] == pytest.approx(2 * 2 / 6)
    assert m["count_error"] == 0
    assert m["unmatched"] == [2]


def test_median_diameter_error_skips_truncated():
    inner_gt, edge_gt = _disk_mask((40, 40), 10), _disk_mask((40, 2), 15)
    inner_pred, edge_pred = _disk_mask((40, 40), 12), _disk_mask((40, 2), 5)
    m = ev.panel_metrics([inner_pred, edge_pred], None, _gt(_labels(inner_gt, edge_gt)), FULL_BOX)
    expected = abs(ev.eqdiam(inner_pred) - ev.eqdiam(inner_gt)) / ev.eqdiam(inner_gt)
    assert m["median_d_err"] == pytest.approx(expected)
    assert m["n_gt_truncated"] == 1


def test_hollow_accuracy_and_pore_checks():
    outer, cav = _disk_mask((50, 50), 20), _disk_mask((50, 50), 8)
    gt = _gt(_labels(outer), cavities=_labels(cav))
    good = ev.panel_metrics([outer], [True], gt, FULL_BOX)
    assert good["hollow_accuracy"] == 1.0
    assert good["n_pred_on_gt_cavity"] == 0 and good["n_extra_inside_matched"] == 0
    bad = ev.panel_metrics([outer, cav], [False, False], gt, FULL_BOX)
    assert bad["hollow_accuracy"] == 0.0
    assert bad["n_pred_on_gt_cavity"] == 1  # the pore predicted as a particle (criterion 3)
    assert bad["n_extra_inside_matched"] == 1  # counted twice (criterion 4)


def test_mask_stack_round_trip_keeps_overlaps(tmp_path):
    a, b = _disk_mask((40, 40), 15), _disk_mask((40, 55), 15)  # overlapping
    path = tmp_path / "p__masks.npz"
    np.savez_compressed(path, **pack_masks(np.stack([a, b])))
    stack = unpack_masks(path)
    assert stack.shape == (2, *SHAPE)
    assert (stack[0] == a).all() and (stack[1] == b).all()


def test_npsam_legacy_label_export_is_read(tmp_path):
    labels = _labels(_disk_mask((30, 30), 10), _disk_mask((60, 60), 10))
    (tmp_path / "masks").mkdir()
    np.save(tmp_path / "masks" / "p.npy", labels.astype(np.int32))
    pred = ev.npsam_prediction(tmp_path, "p", SHAPE)
    assert len(pred["masks"]) == 2 and "overlaps lost" in pred["source"]
    assert ev.npsam_prediction(tmp_path, "missing", SHAPE) is None


def test_evaluate_end_to_end(tmp_path):
    """A Step-6 run, GT and NP-SAM masks -> metrics, unmatched list and criteria."""
    outer, cav, dot = toy_disk((60, 60), 30), toy_disk((60, 60), 18), toy_disk((60, 60), 4)
    result, ids = build_panel([(outer, PARTICLE), (cav, CAVITY), (dot, PARTICLE)], [outer, cav, dot])
    measured = measure.run_measure(result, "toy", "toy", 2.0, "auto", CFG)
    run = tmp_path / "run"
    (run / "measure").mkdir(parents=True)
    (run / "cleanup").mkdir()
    measure.write_outputs(measured, str(run / "measure"), "toy")
    (run / "cleanup" / "toy.json").write_text(json.dumps({"content_box": [0, 0, 200, 200]}))

    seeds = [toy_disk(c, SEED_R) for c in SEED_CENTERS]
    gt_dir = tmp_path / "gt"
    (gt_dir / "toy").mkdir(parents=True)
    tifffile.imwrite(gt_dir / "toy" / "particles.tif", _labels_toy(outer, *seeds))
    tifffile.imwrite(gt_dir / "toy" / "cavities.tif", _labels_toy(cav))
    pd.DataFrame([{"panel_id": "toy", "nm_per_px": 2.04}]).to_csv(gt_dir / "calibration.csv", index=False)

    npsam_dir = tmp_path / "npsam"
    npsam_dir.mkdir()
    np.savez_compressed(npsam_dir / "toy__masks.npz", **pack_masks(np.stack([outer, cav, *seeds[:2]])))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"panel_id": "toy"}, {"panel_id": "no_gt_yet"}]).to_csv(manifest, index=False)

    metrics, unmatched, _, criteria, skipped = ev.evaluate(manifest, gt_dir, run, npsam_dir)
    by = metrics.set_index("method")
    assert by.loc["lnpsam", "f1"] == 1.0 and by.loc["lnpsam", "hollow_accuracy"] == 1.0
    assert by.loc["lnpsam", "calibration_error"] == pytest.approx(abs(2.0 - 2.04) / 2.04)
    assert by.loc["npsam", "f1"] == pytest.approx(2 * 3 / (4 + 4))
    assert by.loc["npsam", "n_pred_on_gt_cavity"] == 1
    assert list(unmatched["method"]) == ["npsam"]
    assert skipped == [{"panel_id": "no_gt_yet", "why": "no ground truth"}]
    assert criteria["5_calibration"]["status"] == "pass"
    assert criteria["6_beats_npsam"]["n_counted"] == 1

    ev.main(["--run", str(run), "--gt", str(gt_dir), "--manifest", str(manifest),
             "--npsam", str(npsam_dir)])
    for name in ("metrics.csv", "unmatched.csv", "aggregates.csv", "criteria.json"):
        assert (run / "eval" / name).is_file()


def _labels_toy(*masks):
    lab = np.zeros(masks[0].shape, dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        lab[m] = i
    return lab
