"""Evaluation against the golden panels (spec 12.3) -- PARTIALLY IMPLEMENTED.

Metrics: F1@0.5 (Hungarian matching on -IoU), count error, median diameter error,
hollow accuracy, calibration error. Compares LNP-SAM vs the NP-SAM baseline against
ground-truth label images (spec 12.2).

Implemented so far:
- Aggregate metrics (12.3), computed only when ``data/gt/<panel_id>/aggregates.tif``
  exists: aggregate IoU and member count error per GT aggregate.
- The criterion-6 tally (12.4), which excludes tracking panels (12.1).

The core per-panel metrics and ``main`` are not implemented yet.
"""

from __future__ import annotations

import os

import numpy as np
import tifffile

# Spec 12.1: the control panel, and tracking panels (reported, excluded from criterion 6).
CONTROL_PANEL = "pylypchuk2020_acssuschemeng_0c05119__p05_x218__panel02"
TRACKING_PANELS = frozenset({"pylypchuk2021_greenchem_04058d__p05_x46__panel00"})

# Spec 12.4 criterion 6: better on at least this many of the 12 gating panels.
CRITERION6_MIN_BETTER = 9

# Spec 12.3: a GT particle is a member of a GT aggregate if this fraction of its
# pixels lies inside the aggregate.
MEMBER_INSIDE_FRAC = 0.5


def _label_masks(label_img):
    """Label image -> ``{label: bool mask}`` (0 is background)."""
    return {int(v): label_img == v for v in np.unique(label_img) if v != 0}


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def aggregate_metrics(gt_aggregates, gt_particles, pred_aggregates, pred_member_counts):
    """Spec 12.3 aggregate IoU and member count error, one record per GT aggregate.

    ``gt_aggregates`` / ``gt_particles`` are label images (``aggregates.tif`` /
    ``particles.tif``). ``pred_aggregates`` maps predicted aggregate id -> filled
    bool mask (full-panel coordinates); ``pred_member_counts`` maps the same id ->
    number of member particles (``aggregate_id`` == that id).
    """
    gt_particle_masks = _label_masks(gt_particles)
    records = []
    for gt_label, gt_mask in _label_masks(gt_aggregates).items():
        best_id, best_iou = None, 0.0
        for pred_id, pred_mask in pred_aggregates.items():
            iou = _iou(gt_mask, pred_mask)
            if iou > best_iou:
                best_id, best_iou = pred_id, iou

        n_gt = sum(
            np.logical_and(m, gt_mask).sum() >= MEMBER_INSIDE_FRAC * m.sum()
            for m in gt_particle_masks.values()
        )
        n_pred = pred_member_counts.get(best_id, 0) if best_id is not None else 0
        records.append(
            {
                "gt_aggregate": gt_label,
                "pred_aggregate": best_id,
                "aggregate_iou": best_iou,
                "n_gt_members": int(n_gt),
                "n_pred_members": int(n_pred),
                "member_count_error": abs(n_pred - n_gt) / n_gt if n_gt else None,
            }
        )
    return records


def evaluate_panel_aggregates(gt_dir, panel_id, pred_aggregates, pred_member_counts):
    """Aggregate metrics for one panel, or ``None`` if it has no ``aggregates.tif``.

    ``aggregates.tif`` is optional (12.2); without it the panel is evaluated
    exactly as before and no aggregate metrics are produced.
    """
    panel_dir = os.path.join(gt_dir, panel_id)
    agg_path = os.path.join(panel_dir, "aggregates.tif")
    if not os.path.exists(agg_path):
        return None
    gt_aggregates = tifffile.imread(agg_path)
    gt_particles = tifffile.imread(os.path.join(panel_dir, "particles.tif"))
    return aggregate_metrics(gt_aggregates, gt_particles, pred_aggregates, pred_member_counts)


def _better(ours, base):
    """LNP-SAM strictly better on both F1@0.5 and median diameter error."""
    if None in (ours["f1"], base["f1"], ours["median_d_err"], base["median_d_err"]):
        return False
    return ours["f1"] > base["f1"] and ours["median_d_err"] < base["median_d_err"]


def _not_worse(ours, base):
    if None in (ours["f1"], base["f1"], ours["median_d_err"], base["median_d_err"]):
        return False
    return ours["f1"] >= base["f1"] and ours["median_d_err"] <= base["median_d_err"]


def criterion6_tally(panel_results, tracking_panels=TRACKING_PANELS, control_panel=CONTROL_PANEL):
    """Spec 12.4 criterion 6 over the original golden panels; tracking panels excluded.

    ``panel_results`` maps panel_id -> ``{"lnpsam": {"f1", "median_d_err"},
    "npsam": {"f1", "median_d_err"}}``.
    """
    counted = {p: r for p, r in panel_results.items() if p not in tracking_panels}
    n_better = sum(_better(r["lnpsam"], r["npsam"]) for r in counted.values())
    control = counted.get(control_panel)
    control_not_worse = control is not None and _not_worse(control["lnpsam"], control["npsam"])
    return {
        "n_better": n_better,
        "n_counted": len(counted),
        "excluded": sorted(set(panel_results) - set(counted)),
        "control_not_worse": control_not_worse,
        "passed": n_better >= CRITERION6_MIN_BETTER and control_not_worse,
    }


def main(*args, **kwargs):
    raise NotImplementedError(
        "The evaluator (spec 12.3) is not implemented yet. It belongs to the "
        "evaluation stage, after the pipeline produces particle tables."
    )


if __name__ == "__main__":
    main()
