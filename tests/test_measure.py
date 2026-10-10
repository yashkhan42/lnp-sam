"""Tests for Step 6 Measure and write (spec 10), on toy panels (spec 11)."""

import math

import numpy as np
import pandas as pd
import pytest
import tifffile

from skimage.segmentation import find_boundaries

from lnpsam import measure
from lnpsam import io as lnp_io
from lnpsam import overlay as overlay_mod
from toys import CAVITY, CFG, PARTICLE, SEED_CENTERS, build_panel, by_id, disk_mask


def _measure(result, nm_per_px=None, cal_source=None):
    cal_source = cal_source or ("manual" if nm_per_px else "missing")
    return measure.run_measure(result, "toy_panel", "toy", nm_per_px, cal_source, CFG)


def _row(measured, node_id):
    rows = measured.particles[measured.particles["particle_id"] == node_id]
    assert len(rows) == 1
    return rows.iloc[0]


def test_hollow_particle_measured_in_px_and_nm():
    """10.1: d, area, Feret, cavity, shell, hollow; nm = px x nm_per_px."""
    outer, cav, dot = disk_mask((60, 60), 30), disk_mask((60, 60), 18), disk_mask((60, 60), 4)
    result, ids = build_panel([(outer, PARTICLE), (cav, CAVITY), (dot, PARTICLE)], [outer, cav, dot])
    nodes = by_id(result)
    m = _measure(result, nm_per_px=2.0)
    row = _row(m, ids[0])

    assert row["d_px"] == pytest.approx(nodes[ids[0]]["eqdiam"])
    assert row["cavity_d_px"] == pytest.approx(nodes[ids[1]]["eqdiam"])
    assert row["shell_px"] == pytest.approx((row["d_px"] - row["cavity_d_px"]) / 2)
    assert bool(row["hollow"]) and row["n_spots"] == 1
    assert row["feret_max_px"] == pytest.approx(60, abs=3)
    assert row["aspect"] == pytest.approx(1.0, abs=0.05)
    assert row["d_nm"] == pytest.approx(2.0 * row["d_px"])
    assert row["area_nm2"] == pytest.approx(4.0 * row["area_px"])
    assert row["cavity_d_nm"] == pytest.approx(2.0 * row["cavity_d_px"])
    assert row["shell_nm"] == pytest.approx(2.0 * row["shell_px"])
    assert bool(row["in_stats"])
    assert m.panel["n_hollow"] == 1 and m.panel["stats_unit"] == "nm"
    # Only particles get rows: the cavity and the dot do not.
    assert set(m.particles["particle_id"]) == {ids[0]} | {
        n["node_id"] for n in result.nodes if n["role"] == "particle" and n["node_id"] not in ids
    }


def test_uncalibrated_panel_leaves_nm_empty_and_uses_px():
    result, _ = build_panel([], [])
    m = _measure(result)
    assert m.particles["d_nm"].isna().all() and m.particles["area_nm2"].isna().all()
    assert m.particles["d_px"].notna().all()
    assert m.panel["stats_unit"] == "px" and "UNCALIBRATED" in m.panel["flags"]


def test_truncated_and_exclusion_particles_are_out_of_stats():
    """10.1: in_stats = not truncated and not touches_exclusion (and not POSSIBLE_CORE_ONLY)."""
    edge, pore = disk_mask((60, 10), 30), disk_mask((60, 18), 8)
    result, ids = build_panel([(edge, PARTICLE), (pore, CAVITY)], [edge, pore])
    seed_ids = [n["node_id"] for n in result.nodes if n["role"] == "particle" and n["node_id"] != ids[0]]
    by_id(result)[seed_ids[0]]["touches_exclusion"] = True  # as Step 2 sets it next to furniture
    m = _measure(result)

    assert bool(_row(m, ids[0])["truncated"]) and not bool(_row(m, ids[0])["in_stats"])
    assert bool(_row(m, seed_ids[0])["touches_exclusion"]) and not bool(_row(m, seed_ids[0])["in_stats"])
    assert m.panel["n_truncated"] == 1 and m.panel["n_touches_exclusion"] == 1
    assert m.panel["n_in_stats"] == m.panel["n_particles"] - 2


def test_unrecovered_orphans_fall_back():
    """9.6 (run in Step 6 while Step 5 is deferred)."""
    outer, cav = disk_mask((60, 60), 30), disk_mask((60, 60), 18)
    result, ids = build_panel([(outer, PARTICLE), (cav, CAVITY)], [cav])
    assert by_id(result)[ids[0]]["role"] == "orphan"

    m = _measure(result)
    node = next(n for n in m.nodes if n["node_id"] == ids[0])
    assert (node["role"], node["rule_id"]) == ("unresolved", measure.ORPHAN_FALLBACK_RULE)
    assert "ORPHAN_UNRECOVERED" in node["flags"]
    assert ids[0] not in set(m.particles["particle_id"])
    assert "ORPHAN_FALLBACK" in m.panel["flags"] and m.panel["role_counts"]["orphan"] == 0

    by_id(result)[ids[0]]["subtype"] = "core"  # a core-type orphan instead
    m = _measure(result)
    row = _row(m, ids[0])
    assert "POSSIBLE_CORE_ONLY" in row["flags"] and not bool(row["in_stats"])


def test_stats_twice_and_aggregate_report():
    """10.2: stats over in_stats particles, all and without aggregate members."""
    centers = [(50, 50), (50, 85), (50, 120), (85, 67), (85, 102)]
    members = [disk_mask(c, 16) for c in centers]
    blob = np.logical_or.reduce([disk_mask(c, 19) for c in centers])
    free = [disk_mask(c, 10) for c in ((130, 175), (60, 175), (130, 40))]
    draw = [(blob, PARTICLE)] + [(f, PARTICLE) for f in free]
    result, ids = build_panel(draw, [blob] + members + free)
    m = _measure(result)

    agg = m.panel["aggregates"]
    assert len(agg) == 1 and agg[0]["aggregate_id"] == ids[0] and agg[0]["n_members"] == 5
    assert m.panel["n_aggregates"] == 1
    member_rows = m.particles[m.particles["aggregate_id"].notna()]
    assert sorted(member_rows["particle_id"]) == sorted(ids[1:6])

    in_stats = m.particles[m.particles["in_stats"]]
    n_free = len(free) + len(SEED_CENTERS)
    assert m.panel["stats_all"]["n"] == len(in_stats) == 5 + n_free
    assert m.panel["stats_excl_aggregates"]["n"] == n_free
    d = in_stats["d_px"].to_numpy()
    assert m.panel["stats_all"]["median"] == pytest.approx(np.median(d))
    assert m.panel["stats_all"]["sd"] == pytest.approx(d.std(ddof=1))
    assert m.panel["stats_all"]["d10"] == pytest.approx(np.percentile(d, 10))
    assert m.panel["stats_all"]["d90"] == pytest.approx(np.percentile(d, 90))

    row = measure.panel_row(m.panel)
    assert row["stats_all_n"] == 5 + n_free and row["n_role_aggregate"] == 1


def test_too_few_particles_skips_stats():
    one = disk_mask((60, 60), 20)
    result, _ = build_panel([(one, PARTICLE)], [one], with_seeds=False)
    m = _measure(result)
    assert m.panel["n_particles"] == 1
    assert m.panel["stats_all"] is None and m.panel["stats_excl_aggregates"] is None
    assert "TOO_FEW_FOR_STATS" in m.panel["flags"]


def test_size_stats_ignores_missing_values():
    assert measure.size_stats([1.0, None, math.nan, 3.0], min_count=2)["n"] == 2
    assert measure.size_stats([1.0], min_count=2) is None


def test_nodes_table_round_trips_through_parquet(tmp_path):
    """10.3: every node, with RLEs, features, role, rule_id, parent and flags."""
    outer, cav, dot = disk_mask((60, 60), 30), disk_mask((60, 60), 18), disk_mask((60, 60), 4)
    shifted = np.roll(outer, 1, axis=1)  # a duplicate
    result, ids = build_panel([(outer, PARTICLE), (cav, CAVITY), (dot, PARTICLE)], [outer, cav, dot, shifted])
    m = _measure(result)
    path = tmp_path / "nodes.parquet"
    measure.nodes_table(m.nodes).to_parquet(path, index=False)
    table = pd.read_parquet(path)

    assert len(table) == len(m.nodes)
    assert table.columns[0] == "node_id"
    for col in ("rle", "filled_rle", "role", "rule_id", "rule_trace", "parent", "flags",
                "area", "eqdiam", "quality", "centroid_row", "bbox_x0"):
        assert col in table.columns
    row = table[table["node_id"] == ids[1]].iloc[0]
    assert row["role"] == "cavity" and row["parent"] == ids[0]
    assert lnp_io.decode_rle(row["filled_rle"]).sum() == row["area"]
    assert "duplicate" in set(table["role"])


def _aggregate_panel():
    centers = [(50, 50), (50, 85), (50, 120), (85, 67), (85, 102)]
    members = [disk_mask(c, 16) for c in centers]
    blob = np.logical_or.reduce([disk_mask(c, 19) for c in centers])
    return build_panel([(blob, PARTICLE)], [blob] + members)


def test_final_mask_files_hold_only_reported_masks(tmp_path):
    """10.3 (v3): particles.tif = particles.csv rows in order; aggregates.tif = the aggregates."""
    result, ids = _aggregate_panel()
    m = _measure(result)
    measure.write_outputs(m, str(tmp_path), "toy")

    p_ids, p_stack = measure.read_mask_stack(tmp_path / "toy__particles.tif")
    assert p_ids == list(m.particles["particle_id"])
    assert [int(s.sum()) for s in p_stack] == list(m.particles["area_px"])
    a_ids, a_stack = measure.read_mask_stack(tmp_path / "toy__aggregates.tif")
    assert a_ids == [ids[0]] and (a_stack[0] == by_id(result)[ids[0]]["filled"]).all()
    assert not (tmp_path / "toy__nodes.parquet").exists()


@pytest.mark.parametrize("n", [1, 3, 4])
def test_mask_stack_writes_one_grayscale_page_per_mask(tmp_path, n):
    """10.3: page k = mask k. N = 3 or 4 must not be written as one RGB(A) page."""
    items = [{"id": 10 + k, "mask": disk_mask((30, 20 + 10 * k), 6)} for k in range(n)]
    path = tmp_path / "toy__particles.tif"
    measure.write_mask_stack(path, items, "particle")

    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == n
        for page in tif.pages:
            assert page.photometric == tifffile.PHOTOMETRIC.MINISBLACK
            assert page.samplesperpixel == 1
            assert page.shape == items[0]["mask"].shape
    ids, stack = measure.read_mask_stack(path)
    assert ids == [item["id"] for item in items]
    assert all((s == item["mask"]).all() for s, item in zip(stack, items))


def test_hollow_particle_outputs_no_cavity_mask(tmp_path):
    outer, cav, dot = disk_mask((60, 60), 30), disk_mask((60, 60), 18), disk_mask((60, 60), 4)
    result, ids = build_panel([(outer, PARTICLE), (cav, CAVITY), (dot, PARTICLE)], [outer, cav, dot])
    m = _measure(result)
    measure.write_outputs(m, str(tmp_path), "toy")
    p_ids, _ = measure.read_mask_stack(tmp_path / "toy__particles.tif")
    assert ids[0] in p_ids and ids[1] not in p_ids and ids[2] not in p_ids
    assert bool(_row(m, ids[0])["hollow"])  # the cavity survives as a measurement
    assert not (tmp_path / "toy__aggregates.tif").exists()


def test_clean_overlay_draws_particles_but_not_cavities():
    outer, cav, dot = disk_mask((60, 60), 30), disk_mask((60, 60), 18), disk_mask((60, 60), 4)
    result, _ = build_panel([(outer, PARTICLE), (cav, CAVITY), (dot, PARTICLE)], [outer, cav, dot])
    m = _measure(result)
    particles, aggregates = measure.final_masks(m)
    rgb = np.full((*outer.shape, 3), 128, dtype=np.uint8)
    img = overlay_mod.overlay(rgb, particles, aggregates, np.zeros(outer.shape, bool), m.panel)

    drawn = (img[: outer.shape[0]] != rgb).any(axis=2)
    assert drawn[find_boundaries(outer, mode="inner")].mean() > 0.9
    cavity_edge = find_boundaries(cav, mode="inner")
    assert not drawn[cavity_edge].any()
    assert img.shape[0] > outer.shape[0]  # colour key below the image
