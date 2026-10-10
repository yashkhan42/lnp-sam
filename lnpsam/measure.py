"""Step 6: Measure and write (spec 10).

Turns the Step-4 roles into the final outputs, which hold only what is reported
(10.3, v3): one row per particle (``particles.csv``, 10.1), a per-panel summary
(``panel.json``, 10.2), the particle and aggregate masks (``particles.tif``,
``aggregates.tif``) and a clean overlay (:func:`lnpsam.overlay.overlay`).
Cavities and cores are measured into ``particles.csv`` (``hollow``,
``cavity_d``, ``shell``, ``core_d``), not output as masks. The full trace of
every node (``nodes.parquet``, :func:`nodes_table`) is a review output.

Step 5 is deferred (spec 9, v3), so its last rule runs here first: an orphan
that no step recovered keeps its mask but becomes ``unresolved`` (cavity
subtype, flag ``ORPHAN_UNRECOVERED``) or a ``particle`` kept out of the stats
(core subtype, flag ``POSSIBLE_CORE_ONLY``) (9.6).

Units: ``*_px`` columns are always filled; ``*_nm`` columns only when the panel
is calibrated (5.5). Panel statistics use ``d_nm``, or ``d_px`` when
uncalibrated (``stats_unit``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd
import tifffile
from skimage import measure as sk_measure

from .resolve import ROLES

ORPHAN_FALLBACK_RULE = "ORPHAN-FALLBACK"

# Spec 10.1 columns in spec order, then the v3 additions: pixel twins of the nm
# columns (most panels are uncalibrated until nm_per_px is measured) and n_spots.
PARTICLE_COLUMNS = (
    "particle_id", "panel_id", "paper_id", "d_px", "d_nm", "feret_max_nm", "aspect",
    "area_nm2", "cavity_d_nm", "shell_nm", "core_d_nm", "hollow", "truncated",
    "touches_exclusion", "aggregate_id", "source", "rule_id", "flags", "in_stats",
    "area_px", "feret_max_px", "cavity_d_px", "shell_px", "core_d_px", "n_spots",
)

_STAT_KEYS = ("n", "mean", "sd", "median", "d10", "d90")


@dataclass
class MeasureResult:
    """Step-6 products for one panel.

    ``nodes`` are the Step-4 nodes after the 9.6 orphan fallback, each with a
    ``flags`` list. ``particles`` holds the 10.1 table and ``panel`` the 10.2
    summary.
    """

    nodes: list
    particles: pd.DataFrame
    panel: dict


# --------------------------------------------------------------------------- #
# 9.6 fallback for orphans that were never recovered
# --------------------------------------------------------------------------- #
def apply_orphan_fallback(nodes):
    """Spec 9.6: cavity-type orphan -> ``unresolved``; core-type -> ``particle``.

    Both keep their masks and get a flag. A core-type orphan may be a whole
    particle or only its core, so it is counted but kept out of the stats.
    """
    for n in nodes:
        if n.get("role") != "orphan":
            continue
        if n.get("subtype") == "core":
            n["role"] = "particle"
            n["flags"].append("POSSIBLE_CORE_ONLY")
            n["truncated"] = bool(n.get("touches_border", False))
        else:
            n["role"] = "unresolved"
            n["flags"].append("ORPHAN_UNRECOVERED")
        n["rule_id"] = ORPHAN_FALLBACK_RULE
        n["rule_trace"] = list(n.get("rule_trace") or []) + [ORPHAN_FALLBACK_RULE]


# --------------------------------------------------------------------------- #
# 10.1 Per particle
# --------------------------------------------------------------------------- #
def _shape_measures(filled, bbox):
    """Max Feret diameter and major/minor axis ratio of a filled mask (px)."""
    x0, y0, x1, y1 = bbox
    crop = np.asarray(filled[y0:y1, x0:x1], dtype=np.uint8)
    props = sk_measure.regionprops(crop)[0]
    try:
        feret = float(props.feret_diameter_max)
    except Exception:  # degenerate hull (1-px or collinear masks)
        feret = None
    minor = float(props.axis_minor_length)
    aspect = float(props.axis_major_length) / minor if minor > 0 else None
    return feret, aspect


def _scaled(value_px, nm_per_px, power=1):
    if value_px is None or nm_per_px is None:
        return None
    return float(value_px) * nm_per_px**power


def particle_rows(nodes, panel_id, paper_id, nm_per_px):
    """Spec 10.1: one row per ``particle`` node, with its cavity and core."""
    by_parent = {}
    for n in nodes:
        if n.get("role") in ("cavity", "core") and n.get("parent") is not None:
            by_parent.setdefault(n["parent"], []).append(n)

    rows = []
    for n in sorted(nodes, key=lambda node: node["node_id"]):
        if n.get("role") != "particle":
            continue
        inner = by_parent.get(n["node_id"], [])
        cavities = [c for c in inner if c["role"] == "cavity"]
        cores = [c for c in inner if c["role"] == "core"]
        cavity_d = max((c["eqdiam"] for c in cavities), default=None)
        core_d = max((c["eqdiam"] for c in cores), default=None)
        shell = (n["eqdiam"] - cavity_d) / 2.0 if cavity_d is not None else None
        feret, aspect = _shape_measures(n["filled"], n["bbox"])
        flags = list(n["flags"])
        truncated = bool(n.get("truncated", False))
        touches_exclusion = bool(n.get("touches_exclusion", False))
        rows.append(
            {
                "particle_id": int(n["node_id"]),
                "panel_id": panel_id,
                "paper_id": paper_id,
                "d_px": float(n["eqdiam"]),
                "d_nm": _scaled(n["eqdiam"], nm_per_px),
                "feret_max_nm": _scaled(feret, nm_per_px),
                "aspect": aspect,
                "area_nm2": _scaled(n["area"], nm_per_px, power=2),
                "cavity_d_nm": _scaled(cavity_d, nm_per_px),
                "shell_nm": _scaled(shell, nm_per_px),
                "core_d_nm": _scaled(core_d, nm_per_px),
                "hollow": bool(cavities),
                "truncated": truncated,
                "touches_exclusion": touches_exclusion,
                "aggregate_id": n.get("aggregate_id"),
                "source": n.get("source", "sam"),
                "rule_id": n["rule_id"],
                "flags": ";".join(flags),
                "in_stats": not truncated
                and not touches_exclusion
                and "POSSIBLE_CORE_ONLY" not in flags,
                "area_px": int(n["area"]),
                "feret_max_px": feret,
                "cavity_d_px": cavity_d,
                "shell_px": shell,
                "core_d_px": core_d,
                "n_spots": int(n.get("n_spots", 0)),
            }
        )
    frame = pd.DataFrame(rows, columns=PARTICLE_COLUMNS)
    # Keep aggregate_id an integer column with blanks rather than floats.
    frame["aggregate_id"] = frame["aggregate_id"].astype("Int64")
    return frame


# --------------------------------------------------------------------------- #
# 10.2 Per panel
# --------------------------------------------------------------------------- #
def size_stats(values, min_count):
    """Mean, sample SD, median, D10 and D90, or ``None`` below ``min_count``."""
    values = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if values.size < min_count:
        return None
    d10, median, d90 = np.percentile(values, [10, 50, 90])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(median),
        "d10": float(d10),
        "d90": float(d90),
    }


def _aggregate_rows(nodes, particles, nm_per_px):
    members = particles["aggregate_id"].value_counts()
    rows = []
    for n in sorted(nodes, key=lambda node: node["node_id"]):
        if n.get("role") != "aggregate":
            continue
        rows.append(
            {
                "aggregate_id": int(n["node_id"]),
                "d_px": float(n["eqdiam"]),
                "d_nm": _scaled(n["eqdiam"], nm_per_px),
                "area_px": int(n["area"]),
                "area_nm2": _scaled(n["area"], nm_per_px, power=2),
                "n_members": int(members.get(n["node_id"], 0)),
                "touches_border": bool(n.get("touches_border", False)),
                "touches_exclusion": bool(n.get("touches_exclusion", False)),
                "rule_id": n["rule_id"],
            }
        )
    return rows


def panel_summary(nodes, particles, resolve_panel, panel_id, paper_id, nm_per_px,
                  cal_source, min_count):
    """Spec 10.2 summary. Stats are over ``in_stats`` particles, twice: all, and
    without aggregate members; each only if it has ``min_count`` particles."""
    roles = {role: 0 for role in ROLES}
    for n in nodes:
        roles[n["role"]] += 1
    unit = "nm" if nm_per_px is not None else "px"
    size_col = f"d_{unit}"
    in_stats = particles[particles["in_stats"]]
    free = in_stats[in_stats["aggregate_id"].isna()]

    flags = list(resolve_panel.get("flags", []))
    if cal_source == "missing":
        flags.append("UNCALIBRATED")
    if any("ORPHAN_UNRECOVERED" in n["flags"] or "POSSIBLE_CORE_ONLY" in n["flags"] for n in nodes):
        flags.append("ORPHAN_FALLBACK")
    stats_all = size_stats(in_stats[size_col].tolist(), min_count)
    stats_free = size_stats(free[size_col].tolist(), min_count)
    if stats_all is None:
        flags.append("TOO_FEW_FOR_STATS")

    return {
        "panel_id": panel_id,
        "paper_id": paper_id,
        "modality": resolve_panel.get("modality"),
        "nm_per_px": nm_per_px,
        "cal_source": cal_source,
        "n_particles": int(len(particles)),
        "n_in_stats": int(particles["in_stats"].sum()),
        "n_truncated": int(particles["truncated"].sum()),
        "n_touches_exclusion": int(particles["touches_exclusion"].sum()),
        "n_hollow": int(particles["hollow"].sum()),
        "n_aggregates": int(roles.get("aggregate", 0)),
        "role_counts": roles,
        "flags": flags,
        "stats_unit": unit,
        "min_particles_for_stats": int(min_count),
        "stats_all": stats_all,
        "stats_excl_aggregates": stats_free,
        "aggregates": _aggregate_rows(nodes, particles, nm_per_px),
        "resolve": {
            k: resolve_panel.get(k)
            for k in ("mu_bg", "mu_p", "polarity", "n_seeds", "seed_mode", "bg_mode",
                      "d_ref", "d_min", "d_min_source")
        },
    }


def panel_row(panel):
    """Flatten a panel summary into one ``panels_all.csv`` row."""
    row = {k: v for k, v in panel.items()
           if k not in ("role_counts", "stats_all", "stats_excl_aggregates", "aggregates", "resolve", "flags")}
    row["flags"] = ";".join(panel["flags"])
    for name in ("stats_all", "stats_excl_aggregates"):
        stats = panel[name] or {}
        for key in _STAT_KEYS:
            row[f"{name}_{key}"] = stats.get(key)
    for role, count in sorted(panel["role_counts"].items()):
        row[f"n_role_{role}"] = count
    return row


# --------------------------------------------------------------------------- #
# 10.3 nodes.parquet
# --------------------------------------------------------------------------- #
def _cell(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value if not isinstance(value, set) else sorted(value))
    return value


def nodes_table(nodes):
    """Every node with its ids, RLEs, features and decisions (spec 10.3; a review output).

    Masks are stored only as COCO RLE (``rle``, ``filled_rle``; spec 4.4).
    ``rule_trace`` and ``flags`` are joined strings; other lists are JSON.
    """
    records = []
    for n in sorted(nodes, key=lambda node: node["node_id"]):
        rec = {}
        for key, value in n.items():
            if isinstance(value, np.ndarray):
                continue  # 'mask' / 'filled': kept as RLE
            if key == "centroid":
                rec["centroid_row"], rec["centroid_col"] = (float(v) for v in value)
            elif key == "bbox":
                rec["bbox_x0"], rec["bbox_y0"], rec["bbox_x1"], rec["bbox_y1"] = (int(v) for v in value)
            elif key == "rule_trace":
                rec[key] = ">".join(value or [])
            elif key == "flags":
                rec[key] = ";".join(value or [])
            else:
                rec[key] = _cell(value)
        records.append(rec)
    columns = sorted({k for rec in records for k in rec}, key=lambda k: (k != "node_id", k))
    return pd.DataFrame([{k: rec.get(k) for k in columns} for rec in records], columns=columns)


def final_masks(measured):
    """The reported masks: particles in ``particles.csv`` order, then aggregates.

    Returns ``(particles, aggregates)``: lists of ``{"id", "mask"}`` (filled
    masks), particles also with ``in_stats``.
    """
    by_id = {n["node_id"]: n for n in measured.nodes}
    particles = [
        {"id": int(r.particle_id), "mask": by_id[int(r.particle_id)]["filled"], "in_stats": bool(r.in_stats)}
        for r in measured.particles.itertuples()
    ]
    aggregates = [
        {"id": a["aggregate_id"], "mask": by_id[a["aggregate_id"]]["filled"]}
        for a in measured.panel["aggregates"]
    ]
    return particles, aggregates


def write_mask_stack(path, items, kind):
    """``(N, H, W)`` uint8 stack (255 = inside), page k = ``items[k]``; ids in the
    TIFF description. Overlapping masks stay whole (spec 4.4).

    ``photometric="minisblack"`` is required: without it tifffile reads a stack of
    3 or 4 masks as colour samples and writes one RGB page instead of N pages."""
    stack = np.stack([np.asarray(item["mask"], dtype=np.uint8) * 255 for item in items])
    description = json.dumps({f"{kind}_ids": [int(item["id"]) for item in items]})
    tifffile.imwrite(path, stack, photometric="minisblack", compression="zlib",
                     description=description)


def read_mask_stack(path):
    """Inverse of :func:`write_mask_stack`: ``(ids, bool stack)``."""
    with tifffile.TiffFile(path) as tif:
        meta = json.loads(tif.pages[0].description)
        stack = tif.asarray()
    if stack.ndim == 2:
        stack = stack[None]
    (ids,) = meta.values()
    return [int(i) for i in ids], stack > 0


def write_outputs(measured, measure_dir, panel_id):
    """Write the final per-panel files (10.1-10.3, v3).

    ``<panel_id>__particles.csv`` and ``__panel.json``; ``__particles.tif`` (one
    page per particle, in ``particles.csv`` order) and ``__aggregates.tif`` (one
    page per aggregate, in ``panel.json`` order), each only when non-empty. The
    overlays need the image, so the caller draws them.
    """
    prefix = f"{measure_dir}/{panel_id}"
    measured.particles.to_csv(f"{prefix}__particles.csv", index=False)
    with open(f"{prefix}__panel.json", "w") as f:
        json.dump(measured.panel, f, indent=2)
    particles, aggregates = final_masks(measured)
    if particles:
        write_mask_stack(f"{prefix}__particles.tif", particles, "particle")
    if aggregates:
        write_mask_stack(f"{prefix}__aggregates.tif", aggregates, "aggregate")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_measure(resolve_result, panel_id, paper_id, nm_per_px, cal_source, cfg):
    """Measure one resolved panel (spec 10). Returns a :class:`MeasureResult`.

    ``nm_per_px`` is ``None``/NaN when uncalibrated: ``*_nm`` columns stay
    empty and the stats use pixels.
    """
    if nm_per_px is not None and (isinstance(nm_per_px, float) and math.isnan(nm_per_px)):
        nm_per_px = None
    nodes = [dict(n) for n in resolve_result.nodes]
    for n in nodes:
        n["rule_trace"] = list(n.get("rule_trace") or [n["rule_id"]])
        n["flags"] = ["FRAME_PARTICLE"] if n.get("frame_particle") else []
    apply_orphan_fallback(nodes)

    particles = particle_rows(nodes, panel_id, paper_id, nm_per_px)
    panel = panel_summary(
        nodes, particles, resolve_result.panel, panel_id, paper_id, nm_per_px,
        cal_source, int(cfg["measure"]["min_particles_for_stats"]),
    )
    return MeasureResult(nodes=nodes, particles=particles, panel=panel)
