"""Evaluation against the golden panels (spec 12.3) and the exit-criteria report (12.4).

    python -m eval.evaluate --run runs/<name> --gt data/gt --manifest data/manifest.csv \\
        [--npsam runs/npsam_base] [--out runs/<name>/eval]

Per panel and method (``lnpsam``, ``npsam``): F1@0.5 (Hungarian matching on
-IoU), count error, median diameter error, hollow accuracy and calibration
error, plus aggregate IoU / member count error where ``aggregates.tif`` exists.
Panels without ``particles.tif`` are skipped (ground truth not annotated yet).

Inputs
- Ground truth (12.2): ``<gt>/<panel_id>/particles.tif`` (one label per
  particle), ``cavities.tif`` (one label per cavity or core) and, for aggregate
  panels, ``aggregates.tif``; hand-measured ``<gt>/calibration.csv``
  (``panel_id, nm_per_px``).
- LNP-SAM: a run directory written by ``lnpsam.run``: its final outputs
  (``measure/<panel_id>__particles.csv``, ``__particles.tif``, ``__aggregates.tif``,
  ``__panel.json``) and Step 1's content box (``cleanup/``).
- NP-SAM (optional): the output of ``baselines/npsam_run.py``
  (``<panel_id>__masks.npz``, lossless), or the older label-image export
  (``masks/<panel_id>.npy``), in which overlapping masks lost shared pixels.

Conventions
- Masks are compared filled (``binary_fill_holes``): GT outlines are filled
  (12.2) and LNP-SAM particles are filled (4.4); NP-SAM's masks are filled too,
  so the comparison is about finding each particle's outline.
- Diameters are equivalent-area diameters in pixels, so the median diameter
  error scores segmentation alone; calibration is scored separately.
- "Truncated" (median diameter error, criterion 3) means within 1 px of the
  content-box edge (6.3), applied the same way to every method and to GT; the
  content box comes from the LNP-SAM run's Step 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from lnpsam import measure as measure_mod

# Spec 12.1: the control panel, and tracking panels (reported, excluded from criterion 6).
CONTROL_PANEL = "pylypchuk2020_acssuschemeng_0c05119__p05_x218__panel02"
TRACKING_PANELS = frozenset({"pylypchuk2021_greenchem_04058d__p05_x46__panel00"})

# Spec 12.1 failure groups used by the criteria 2-4 checks.
_P05 = "pylypchuk2021_greenchem_04058d__p05_x46__"
GAP_PANELS = frozenset({_P05 + "panel02", _P05 + "panel17"})
EDGE_PORE_PANELS = frozenset({_P05 + "panel01", _P05 + "panel04", _P05 + "panel19"})
HOLLOW_CORE_PANELS = EDGE_PORE_PANELS | {"girard2024_acssuschemeng_4c00711__p08_x658__panel06"}

# Spec 12.4 criterion 6: better on at least this many of the 12 gating panels.
CRITERION6_MIN_BETTER = 9
# Spec 12.3 F1@0.5: a match needs IoU >= 0.5. Criterion 5: within 5 %.
MATCH_IOU = 0.5
CALIBRATION_TOLERANCE = 0.05

# Spec 12.3: a GT particle is a member of a GT aggregate if this fraction of its
# pixels lies inside the aggregate. The same fraction assigns a GT cavity to the
# GT particle containing it, and flags an unmatched prediction lying inside an
# already-matched particle (a double count).
MEMBER_INSIDE_FRAC = 0.5

METHODS = ("lnpsam", "npsam")


# --------------------------------------------------------------------------- #
# Aggregates (12.3) and criterion 6 (12.4)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_gt(gt_dir, panel_id):
    """GT label images for a panel, or ``None`` if ``particles.tif`` is missing."""
    panel_dir = os.path.join(gt_dir, panel_id)
    path = os.path.join(panel_dir, "particles.tif")
    if not os.path.exists(path):
        return None
    gt = {"particles": np.asarray(tifffile.imread(path))}
    for name in ("cavities", "aggregates"):
        extra = os.path.join(panel_dir, f"{name}.tif")
        gt[name] = np.asarray(tifffile.imread(extra)) if os.path.exists(extra) else None
    return gt


def load_gt_calibration(gt_dir):
    """Hand-measured ``nm_per_px`` per panel (blank rows are skipped)."""
    path = os.path.join(gt_dir, "calibration.csv")
    if not os.path.exists(path):
        return {}
    frame = pd.read_csv(path)
    return {
        str(r.panel_id): float(r.nm_per_px)
        for r in frame.itertuples()
        if pd.notna(r.nm_per_px)
    }


def lnpsam_prediction(run_dir, panel_id):
    """LNP-SAM particles of one panel, read from the final Step-6 outputs."""
    measure_dir = os.path.join(run_dir, "measure")
    csv_path = os.path.join(measure_dir, f"{panel_id}__particles.csv")
    if not os.path.exists(csv_path):
        return None
    particles = pd.read_csv(csv_path)
    with open(os.path.join(measure_dir, f"{panel_id}__panel.json")) as f:
        panel = json.load(f)
    ids = [int(i) for i in particles["particle_id"]]
    masks = []
    if ids:
        tif_ids, stack = measure_mod.read_mask_stack(os.path.join(measure_dir, f"{panel_id}__particles.tif"))
        if tif_ids != ids:
            raise ValueError(f"{panel_id}: particles.tif pages do not match particles.csv rows")
        masks = list(stack)
    aggregates = {}
    agg_path = os.path.join(measure_dir, f"{panel_id}__aggregates.tif")
    if os.path.exists(agg_path):
        agg_ids, agg_stack = measure_mod.read_mask_stack(agg_path)
        aggregates = dict(zip(agg_ids, agg_stack))
    members = particles["aggregate_id"].dropna().astype(int).value_counts().to_dict()

    content_box = None
    cleanup_json = os.path.join(run_dir, "cleanup", f"{panel_id}.json")
    if os.path.exists(cleanup_json):
        with open(cleanup_json) as f:
            content_box = tuple(json.load(f)["content_box"])
    return {
        "ids": ids,
        "masks": masks,
        "hollow": [bool(h) for h in particles["hollow"]],
        "aggregates": aggregates,
        "member_counts": {int(k): int(v) for k, v in members.items()},
        "nm_per_px": panel.get("nm_per_px"),
        "cal_source": panel.get("cal_source"),
        "content_box": content_box,
        "source": "lnpsam",
    }


def npsam_prediction(npsam_dir, panel_id, shape):
    """NP-SAM masks of one panel (every mask is a predicted particle, 12.3).

    Prefers the lossless ``<panel_id>__masks.npz`` from ``baselines/npsam_run.py``;
    falls back to a label-image export ``masks/<panel_id>.npy`` (overlaps lost).
    Masks are filled (see the module conventions).
    """
    from baselines.npsam_run import unpack_masks

    npz = os.path.join(npsam_dir, f"{panel_id}__masks.npz")
    legacy = os.path.join(npsam_dir, "masks", f"{panel_id}.npy")
    if os.path.exists(npz):
        stack = unpack_masks(npz)
        masks, source = list(stack), "npsam_run (lossless)"
    elif os.path.exists(legacy):
        labels = np.load(legacy)
        masks = [labels == k for k in np.unique(labels) if k != 0]
        source = "label image (overlaps lost)"
    else:
        return None
    if masks and masks[0].shape != tuple(shape):
        raise ValueError(f"NP-SAM masks for {panel_id} are {masks[0].shape}, panel is {shape}")
    masks = [ndimage.binary_fill_holes(m) for m in masks if m.any()]
    return {
        "ids": list(range(len(masks))),
        "masks": masks,
        "hollow": None,  # NP-SAM has no notion of cavities
        "source": source,
    }


# --------------------------------------------------------------------------- #
# Metrics (12.3)
# --------------------------------------------------------------------------- #
def eqdiam(mask):
    return math.sqrt(4.0 * float(np.count_nonzero(mask)) / math.pi)


def touches_edge(mask, content_box):
    """Within 1 px of the content-box edge (the 6.3 ``touches_border`` test)."""
    x0, y0, x1, y1 = content_box
    return bool(
        mask[y0:min(y0 + 2, y1), x0:x1].any()
        or mask[max(y0, y1 - 2):y1, x0:x1].any()
        or mask[y0:y1, x0:min(x0 + 2, x1)].any()
        or mask[y0:y1, max(x0, x1 - 2):x1].any()
    )


def _relabel(label_img):
    """Map labels to 0..K (0 = background); return the map and the original ids."""
    ids, inverse = np.unique(label_img, return_inverse=True)
    inverse = inverse.reshape(label_img.shape)
    if ids[0] != 0:
        ids = np.concatenate([[0], ids])
        inverse = inverse + 1
    return inverse, ids[1:]


def overlap_tables(pred_masks, label_img):
    """Intersections and IoU of every prediction with every label (dense, n_pred x K)."""
    lab, ids = _relabel(label_img)
    k = len(ids)
    label_area = np.bincount(lab.ravel(), minlength=k + 1)[1:]
    inter = np.zeros((len(pred_masks), k))
    for i, m in enumerate(pred_masks):
        inter[i] = np.bincount(lab[m], minlength=k + 1)[1:]
    pred_area = np.array([m.sum() for m in pred_masks], dtype=float).reshape(-1, 1)
    union = pred_area + label_area.reshape(1, -1) - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return inter, iou, ids, lab


def match(iou, threshold=MATCH_IOU):
    """Hungarian matching on -IoU; pairs ``(pred, gt, iou)`` with IoU >= threshold."""
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= threshold]


def _gt_cavity_owner(gt_lab, gt_ids, cavities):
    """For each GT particle id: does it contain a GT cavity (>= half inside it)?"""
    has = {int(g): False for g in gt_ids}
    for cav in _label_masks(cavities).values():
        counts = np.bincount(gt_lab[cav], minlength=len(gt_ids) + 1)[1:]
        if counts.size and counts.max() >= MEMBER_INSIDE_FRAC * cav.sum():
            has[int(gt_ids[int(counts.argmax())])] = True
    return has


def panel_metrics(pred_masks, pred_hollow, gt, content_box):
    """Spec 12.3 metrics of one method on one panel, plus the criteria-3/4 checks.

    Returns the metrics and ``unmatched`` (indices of predictions matching no GT
    particle; candidates for criteria 1-2 review).
    """
    gt_particles = gt["particles"]
    inter, iou, gt_ids, gt_lab = overlap_tables(pred_masks, gt_particles)
    pairs = match(iou)
    n_pred, n_gt, tp = len(pred_masks), len(gt_ids), len(pairs)
    matched_pred = {p for p, _, _ in pairs}
    matched_gt = {g for _, g, _ in pairs}

    gt_masks = [gt_lab == k + 1 for k in range(n_gt)]
    gt_trunc = [touches_edge(m, content_box) for m in gt_masks]
    pred_trunc = [touches_edge(m, content_box) for m in pred_masks]
    d_pred = [eqdiam(m) for m, t in zip(pred_masks, pred_trunc) if not t]
    d_gt = [eqdiam(m) for m, t in zip(gt_masks, gt_trunc) if not t]
    median_err = (
        abs(np.median(d_pred) - np.median(d_gt)) / np.median(d_gt) if d_pred and d_gt else None
    )

    hollow_acc = None
    if pred_hollow is not None and gt["cavities"] is not None and pairs:
        owner = _gt_cavity_owner(gt_lab, gt_ids, gt["cavities"])
        hollow_acc = float(np.mean([pred_hollow[p] == owner[int(gt_ids[g])] for p, g, _ in pairs]))

    # Criterion 3: a prediction that is a GT pore/cavity (IoU >= 0.5).
    n_on_cavity = 0
    if gt["cavities"] is not None and n_pred:
        _, cav_iou, _, _ = overlap_tables(pred_masks, gt["cavities"])
        n_on_cavity = int((cav_iou.max(axis=1, initial=0.0) >= MATCH_IOU).sum())
    # Criterion 4: an unmatched prediction lying inside a counted GT particle.
    pred_area = np.array([m.sum() for m in pred_masks], dtype=float)
    matched_cols = sorted(matched_gt)
    n_extra = 0
    for p in range(n_pred):
        if p in matched_pred or not matched_cols:
            continue
        if inter[p, matched_cols].max() >= MEMBER_INSIDE_FRAC * pred_area[p]:
            n_extra += 1

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "tp": tp,
        "precision": tp / n_pred if n_pred else None,
        "recall": tp / n_gt if n_gt else None,
        "f1": 2.0 * tp / (n_pred + n_gt) if (n_pred + n_gt) else None,
        "count_error": abs(n_pred - n_gt) / n_gt if n_gt else None,
        "median_d_err": float(median_err) if median_err is not None else None,
        "hollow_accuracy": hollow_acc,
        "n_gt_truncated": int(sum(gt_trunc)),
        "n_gt_truncated_matched": int(sum(gt_trunc[g] for g in matched_gt)),
        "n_pred_on_gt_cavity": n_on_cavity,
        "n_extra_inside_matched": n_extra,
        "unmatched": sorted(set(range(n_pred)) - matched_pred),
    }


def calibration_error(pred, gt_nm_per_px):
    """Spec 12.3, for automatic calibrations only."""
    if pred.get("cal_source") != "auto" or gt_nm_per_px is None or not pred.get("nm_per_px"):
        return None
    return abs(pred["nm_per_px"] - gt_nm_per_px) / gt_nm_per_px


# --------------------------------------------------------------------------- #
# Run + exit criteria (12.4)
# --------------------------------------------------------------------------- #
def _status(ok):
    return "pass" if ok else "fail"


def exit_criteria(metrics, aggregates, gt_cal, cal_sources):
    """Criteria 5-7 decided from the metrics; 3-4 checked on their panels; 1-2 need review."""
    lnp = metrics[metrics["method"] == "lnpsam"].set_index("panel_id") if len(metrics) else pd.DataFrame()
    out = {}

    def rows_for(panels):
        return lnp[lnp.index.isin(panels)] if len(lnp) else lnp

    # 1-2: furniture or gaps predicted as particles show up as unmatched predictions.
    out["1_no_particle_on_furniture"] = {
        "status": "review",
        "how": "inspect unmatched.csv (method lnpsam) on the overlays",
        "unmatched_per_panel": {p: int(r["n_pred"] - r["tp"]) for p, r in lnp.iterrows()},
    }
    out["2_no_gap_as_particle"] = {
        "status": "review",
        "how": "inspect unmatched.csv on the gap panels",
        "unmatched_per_panel": {p: int(r["n_pred"] - r["tp"]) for p, r in rows_for(GAP_PANELS).iterrows()},
    }
    # 3: outer particle present (every truncated GT particle matched) and pore not a particle.
    edge = rows_for(EDGE_PORE_PANELS)
    out["3_edge_pore"] = {
        "panels": {
            p: _status(r["n_gt_truncated_matched"] == r["n_gt_truncated"] and r["n_pred_on_gt_cavity"] == 0)
            for p, r in edge.iterrows()
        },
    }
    # 4: counted once (no extra prediction inside a matched particle), cavity/core attached.
    hc = rows_for(HOLLOW_CORE_PANELS)
    out["4_counted_once_with_cavity"] = {
        "panels": {
            p: ("not evaluated (no cavities.tif)" if pd.isna(r["hollow_accuracy"])
                else _status(r["n_extra_inside_matched"] == 0 and r["hollow_accuracy"] == 1.0))
            for p, r in hc.iterrows()
        },
    }
    # 5: calibration within 5 % wherever the scale bar was read automatically.
    cal = {}
    for p, r in lnp.iterrows():
        if pd.notna(r.get("calibration_error")):
            cal[p] = {"error": float(r["calibration_error"]),
                      "status": _status(r["calibration_error"] <= CALIBRATION_TOLERANCE)}
    no_auto = sorted(p for p in gt_cal if p in lnp.index and cal_sources.get(p) != "auto")
    out["5_calibration"] = {
        "panels": cal,
        "no_auto_calibration": no_auto,
        "note": "panels in no_auto_calibration fail if their scale bar is readable",
    }
    # 6: better than NP-SAM on >= 9 of the 12 gating panels, not worse on the control.
    both = metrics.pivot_table(index="panel_id", columns="method",
                               values=["f1", "median_d_err"], aggfunc="first") if len(metrics) else None
    if both is not None and "npsam" in metrics["method"].values:
        results = {}
        for p in both.index:
            vals = {m: {"f1": _none(both.loc[p].get(("f1", m))),
                        "median_d_err": _none(both.loc[p].get(("median_d_err", m)))}
                    for m in METHODS}
            results[p] = vals
        out["6_beats_npsam"] = criterion6_tally(results)
    else:
        out["6_beats_npsam"] = {"status": "not evaluated", "why": "no NP-SAM predictions (--npsam)"}
    # 7: non-gating aggregate tracking.
    out["7_tracking_aggregate"] = {"status": "reported, not required", "records": aggregates}
    for key in ("3_edge_pore", "4_counted_once_with_cavity"):
        statuses = [v for v in out[key]["panels"].values() if v in ("pass", "fail")]
        out[key]["status"] = ("pass" if statuses and all(s == "pass" for s in statuses)
                              else "fail" if statuses else "not evaluated")
    statuses = [v["status"] for v in cal.values()]
    out["5_calibration"]["status"] = ("pass" if statuses and all(s == "pass" for s in statuses)
                                      else "fail" if statuses else "not evaluated")
    return out


def _none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def evaluate(manifest_path, gt_dir, run_dir, npsam_dir=None):
    """Evaluate every manifest panel that has ground truth.

    Returns ``(metrics, unmatched, aggregates, criteria, skipped)``.
    """
    manifest = pd.read_csv(manifest_path)
    gt_cal = load_gt_calibration(gt_dir)
    rows, unmatched, agg_rows, skipped, cal_sources = [], [], [], [], {}
    for panel_id in manifest["panel_id"]:
        gt = load_gt(gt_dir, panel_id)
        lnp = lnpsam_prediction(run_dir, panel_id)
        if gt is None or lnp is None:
            skipped.append({"panel_id": panel_id,
                            "why": "no ground truth" if gt is None else "no LNP-SAM output"})
            continue
        shape = gt["particles"].shape
        content_box = lnp["content_box"] or (0, 0, shape[1], shape[0])
        cal_sources[panel_id] = lnp["cal_source"]
        preds = {"lnpsam": lnp}
        if npsam_dir:
            nps = npsam_prediction(npsam_dir, panel_id, shape)
            if nps is not None:
                preds["npsam"] = nps
        for method, pred in preds.items():
            m = panel_metrics(pred["masks"], pred["hollow"], gt, content_box)
            for i in m.pop("unmatched"):
                ys, xs = np.nonzero(pred["masks"][i])
                unmatched.append({"panel_id": panel_id, "method": method, "pred_id": pred["ids"][i],
                                  "area_px": int(ys.size), "centroid_row": float(ys.mean()),
                                  "centroid_col": float(xs.mean())})
            rows.append({"panel_id": panel_id, "method": method, "source": pred["source"], **m,
                         "calibration_error": calibration_error(pred, gt_cal.get(panel_id))})
        records = evaluate_panel_aggregates(gt_dir, panel_id, lnp["aggregates"], lnp["member_counts"])
        for rec in records or []:
            agg_rows.append({"panel_id": panel_id, **rec})

    metrics = pd.DataFrame(rows)
    criteria = exit_criteria(metrics, agg_rows, gt_cal, cal_sources)
    return metrics, pd.DataFrame(unmatched), pd.DataFrame(agg_rows), criteria, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eval.evaluate", description="LNP-SAM golden-panel evaluation")
    parser.add_argument("--run", required=True, help="LNP-SAM run directory (lnpsam.run --out)")
    parser.add_argument("--gt", default="data/gt", help="ground-truth directory (spec 12.2)")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--npsam", default=None, help="NP-SAM baseline output directory")
    parser.add_argument("--out", default=None, help="report directory (default: <run>/eval)")
    args = parser.parse_args(argv)

    out_dir = args.out or os.path.join(args.run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    metrics, unmatched, aggregates, criteria, skipped = evaluate(
        args.manifest, args.gt, args.run, args.npsam
    )
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    unmatched.to_csv(os.path.join(out_dir, "unmatched.csv"), index=False)
    aggregates.to_csv(os.path.join(out_dir, "aggregates.csv"), index=False)
    with open(os.path.join(out_dir, "criteria.json"), "w") as f:
        json.dump({"criteria": criteria, "skipped": skipped}, f, indent=2, default=str)

    print(f"[eval] {len(metrics['panel_id'].unique()) if len(metrics) else 0} panels evaluated, "
          f"{len(skipped)} skipped -> {out_dir}")
    for s in skipped:
        print(f"  skipped {s['panel_id']}: {s['why']}")
    for key, value in criteria.items():
        print(f"  criterion {key}: {value.get('status', '')}")
    return metrics, criteria


if __name__ == "__main__":
    main()
