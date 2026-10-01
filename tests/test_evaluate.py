"""Tests for eval/evaluate.py aggregate metrics and criterion-6 tally (spec 12.3, 12.4)."""

import numpy as np
import pytest
import tifffile
from skimage.draw import disk

from eval import evaluate as ev

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
