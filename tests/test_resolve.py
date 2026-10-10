"""Tests for Step 4 Resolve (spec 8, 11).

Toy TEM-like panels (background 0.8, particle 0.3, cavity 0.6, spec 11) with
hand-built masks. Nodes go through the real Step-2 post-processing and Step-3
graph, so features and relations are computed exactly as in the pipeline.
"""

import numpy as np
from skimage.draw import polygon

from lnpsam import graph, proposals, resolve
from toys import BG, CAVITY, CFG, PARTICLE, SHAPE
from toys import build_panel as _panel
from toys import by_id as _by_id
from toys import cfg_with as _cfg
from toys import disk_mask as _disk
from toys import roles as _roles


def test_seeds_set_panel_levels():
    result, _ = _panel([], [])
    panel = result.panel
    assert panel["n_seeds"] == 3
    assert panel["seed_mode"] == "strict"
    assert abs(panel["mu_bg"] - BG) < 0.02
    assert abs(panel["mu_p"] - PARTICLE) < 0.02
    assert panel["flags"] == []
    assert panel["role_counts"]["particle"] == 3


def test_hollow_particle_outer_cavity_center_dot():
    """Spec 11: outer mask, cavity mask, center dot -> 1 particle, 1 cavity, 1 spot."""
    outer, cav, dot = _disk((60, 60), 30), _disk((60, 60), 18), _disk((60, 60), 4)
    result, ids = _panel([(outer, PARTICLE), (cav, CAVITY), (dot, PARTICLE)], [outer, cav, dot])
    assert _roles(result, ids) == ["particle", "cavity", "spot"]
    nodes = _by_id(result)
    assert nodes[ids[1]]["parent"] == ids[0]
    assert nodes[ids[1]]["rule_id"] == "C-CAVITY"
    assert nodes[ids[0]]["n_spots"] == 1


def test_three_touching_disks_and_gap():
    """Spec 11: three touching disks plus the gap between them -> 3 particle, 1 gap."""
    r = 25
    centers = [(60, 60), (60, 108), (60 + 41.6, 84)]  # pairwise ~48 px < 2r: touching
    disks = [_disk(c, r) for c in centers]
    tri = np.zeros(SHAPE, dtype=bool)
    rr, cc = polygon([c[0] for c in centers], [c[1] for c in centers], shape=SHAPE)
    tri[rr, cc] = True
    gap = tri & ~np.logical_or.reduce(disks)
    result, ids = _panel([(d, PARTICLE) for d in disks], disks + [gap])
    assert _roles(result, ids) == ["particle", "particle", "particle", "gap"]
    assert _by_id(result)[ids[3]]["rule_id"] == "R-GAP"


def test_border_cut_particle_with_pore():
    """Spec 11: particle cut by the border keeps its pore as cavity/spot, not a particle."""
    outer, pore = _disk((60, 10), 30), _disk((60, 18), 8)
    result, ids = _panel([(outer, PARTICLE), (pore, CAVITY)], [outer, pore])
    nodes = _by_id(result)
    particle, pore_node = nodes[ids[0]], nodes[ids[1]]
    assert particle["role"] == "particle"
    assert particle["truncated"] is True
    assert pore_node["role"] in ("cavity", "spot")
    assert pore_node["truncated"] is True  # 8.5: flags propagate to cavity/spot
    assert result.panel["n_truncated"] == 1


def test_orphan_cavity_when_only_cavity_mask_given():
    """Spec 11: hollow particle drawn, only the cavity mask given -> orphan (cavity)."""
    outer, cav = _disk((60, 60), 30), _disk((60, 60), 18)
    result, ids = _panel([(outer, PARTICLE), (cav, CAVITY)], [cav])
    node = _by_id(result)[ids[0]]
    assert node["role"] == "orphan"
    assert node["subtype"] == "cavity"


def test_dumbbell_is_merged_into_two_particles():
    """Spec 11: dumbbell mask containing two disk masks -> merged parent, 2 particles."""
    a, b = _disk((60, 50), 22), _disk((60, 90), 22)
    dumbbell = a | b
    result, ids = _panel([(dumbbell, PARTICLE)], [dumbbell, a, b])
    assert _roles(result, ids) == ["merged", "particle", "particle"]
    nodes = _by_id(result)
    assert nodes[ids[0]]["rule_id"] == "C-SPLIT"
    assert nodes[ids[1]]["parent"] is None
    assert nodes[ids[1]]["split_from"] == ids[0]


def test_blob_with_five_disks_is_aggregate():
    """Spec 11: blob containing 5 disk masks -> aggregate with 5 member particles."""
    centers = [(50, 50), (50, 85), (50, 120), (85, 67), (85, 102)]
    disks = [_disk(c, 16) for c in centers]
    blob = np.logical_or.reduce([_disk(c, 19) for c in centers])
    result, ids = _panel([(blob, PARTICLE)], [blob] + disks)
    nodes = _by_id(result)
    assert nodes[ids[0]]["role"] == "aggregate"
    assert nodes[ids[0]]["rule_id"] == "C-AGG"
    for i in ids[1:]:
        assert nodes[i]["role"] == "particle"
        assert nodes[i]["aggregate_id"] == ids[0]


def test_iou_duplicate_keeps_duplicate_role():
    """Spec 11: two masks with IoU 0.9 -> one representative, one duplicate."""
    a = _disk((60, 60), 25)
    b = np.roll(a, 1, axis=1)
    result, ids = _panel([(a, PARTICLE)], [a, b])
    assert sorted(_roles(result, ids)) == ["duplicate", "particle"]


def test_speck_and_bg_patch_roots():
    speck = np.zeros(SHAPE, dtype=bool)
    speck[20:22, 20:23] = True  # 6 px >= min_piece_px, eqdiam 2.8 < d_min
    patch = _disk((60, 150), 15)  # background-level mask, no particle contact
    result, ids = _panel([], [speck, patch])
    assert _roles(result, ids) == ["speck", "bg_patch"]


def test_no_seeds_falls_back_and_flags_panel():
    edge = _disk((60, 5), 25)
    result, _ = _panel([(edge, PARTICLE)], [edge], with_seeds=False)
    assert "NO_SEEDS" in result.panel["flags"]
    assert result.panel["d_min_source"] == "floor"


def test_calibrated_d_min_uses_min_particle_nm():
    outer = _disk((60, 60), 20)
    gray = np.full(SHAPE, BG, dtype=np.float32)
    gray[outer] = PARTICLE
    content_mask = np.ones(SHAPE, dtype=bool)
    nodes = proposals.postprocess_proposals(
        [{"segmentation": outer, "predicted_iou": 0.95, "stability_score": 0.95,
          "point_coords": [[0.0, 0.0]]}],
        gray, np.zeros(SHAPE, dtype=bool), content_mask, (0, 0, 200, 200), (0, 0), CFG["sam"],
    )
    result = resolve.run_resolve(graph.run_graph(nodes, CFG), gray, content_mask, "TEM", 0.5, CFG)
    assert result.panel["d_min_source"] == "calibrated"
    assert result.panel["d_min"] == CFG["resolve"]["min_particle_nm"] / 0.5


# --------------------------------------------------------------------------- #
# v3 amendments (each also checked with its switch restoring the original rule)
# --------------------------------------------------------------------------- #
def _hollow(hole_centers, hole_r=5, outer_r=30, center=(60, 60)):
    outer = _disk(center, outer_r)
    holes = [_disk(c, hole_r) for c in hole_centers]
    draw = [(outer, PARTICLE)] + [(h, CAVITY) for h in holes]
    return draw, [outer] + holes


def test_single_small_centred_hole_is_cavity():
    """v3 E: one hole near the centre makes the particle hollow, whatever its size."""
    draw, masks = _hollow([(61, 61)])  # hole diameter ~0.17 of the particle's
    result, ids = _panel(draw, masks)
    assert _roles(result, ids) == ["particle", "cavity"]
    original, ids = _panel(draw, masks, cfg=_cfg(cavity={"rule": "size"}))
    assert _roles(original, ids) == ["particle", "spot"]


def test_two_holes_are_spots():
    """v3 E: more than one hole -> no cavity; every hole is a spot."""
    draw, masks = _hollow([(60, 50), (60, 70)])
    result, ids = _panel(draw, masks)
    assert _roles(result, ids) == ["particle", "spot", "spot"]
    assert {_by_id(result)[i]["rule_id"] for i in ids[1:]} == {"C-HOLE-SPOT"}
    assert _by_id(result)[ids[0]]["n_spots"] == 2


def test_off_centre_hole_is_spot():
    draw, masks = _hollow([(60, 80)])  # centroid offset ~0.67 of the radius
    result, ids = _panel(draw, masks)
    assert _roles(result, ids) == ["particle", "spot"]


def test_overlapping_masks_of_one_hole_count_once():
    """SAM often outlines one hole twice; overlapping hole masks are one hole."""
    outer = _disk((60, 60), 30)
    hole_a, hole_b = _disk((60, 58), 6), _disk((60, 62), 6)
    result, ids = _panel(
        [(outer, PARTICLE), (hole_a | hole_b, CAVITY)], [outer, hole_a, hole_b]
    )
    roles = _roles(result, ids)
    assert roles[0] == "particle"
    assert sorted(roles[1:]) == ["cavity", "spot"]


def test_seed_polarity_check_rejects_background_holes():
    """v3 A: bright round holes are not seeds on a TEM panel (particles are dark)."""
    holes = [_disk((40, x), 12) for x in (40, 90, 140)]
    gray_draw = [(h, 0.97) for h in holes]
    result, ids = _panel(gray_draw, holes)
    assert not set(ids) & set(result.panel["seed_ids"])
    assert "POLARITY_MISMATCH" not in result.panel["flags"]
    original, ids = _panel(gray_draw, holes, cfg=_cfg(seed_polarity_check=False))
    assert set(ids) <= set(original.panel["seed_ids"])


def test_rim_dark_particle_with_background_interior_is_particle():
    """v3 B: R-BGPATCH tests pl, so a dark rim with a light interior is a particle."""
    outer, inner = _disk((60, 60), 30), _disk((60, 60), 25)
    draw = [(outer, PARTICLE), (inner, BG)]
    result, ids = _panel(draw, [outer])
    node = _by_id(result)[ids[0]]
    assert node["rho_in"] <= CFG["resolve"]["rho_bg_max"] < node["pl"]
    assert node["role"] == "particle"
    original, ids = _panel(draw, [outer], cfg=_cfg(bg_rules_use="rho_in"))
    assert _roles(original, ids) == ["bg_patch"]


def test_specks_are_not_aggregate_members():
    """v3 C: one particle-sized child plus specks is not an aggregate."""
    outer, inner = _disk((60, 60), 30), _disk((60, 60), 20)
    specks = []
    for y, x in ((40, 75), (78, 45)):
        m = np.zeros(SHAPE, dtype=bool)
        m[y:y + 2, x:x + 2] = True  # 4 px, eqdiam 2.3 < d_min
        specks.append(m)
    draw = [(outer, PARTICLE)]
    result, ids = _panel(draw, [outer, inner] + specks)
    assert _by_id(result)[ids[0]]["role"] == "particle"
    original, ids = _panel(draw, [outer, inner] + specks, cfg=_cfg(group_members_only=False))
    assert _by_id(original)[ids[0]]["role"] == "aggregate"


def test_large_round_mask_is_one_particle():
    """v3 D: a round solid mask above particle.max_area_frac is a particle."""
    big = _disk((70, 100), 62)  # area ~12k px > 25% of the 200x200 panel
    result, ids = _panel([(big, PARTICLE)], [big])
    node = _by_id(result)[ids[0]]
    assert (node["role"], node["rule_id"]) == ("particle", "R-PARTICLE-LARGE")
    original, ids = _panel([(big, PARTICLE)], [big], cfg=_cfg(large_round={"enabled": False}))
    assert _roles(original, ids) == ["aggregate"]


def test_speck_sized_hole_does_not_block_cavity():
    """v3 E: hole-like specks smaller than d_min are spots, not a second hole."""
    draw, masks = _hollow([(61, 61)])
    speck = np.zeros(SHAPE, dtype=bool)
    speck[50:52, 72:74] = True  # 4 px, eqdiam 2.3 < d_min
    result, ids = _panel(draw + [(speck, CAVITY)], masks + [speck])
    assert _roles(result, ids) == ["particle", "cavity", "spot"]


def test_particle_bright_centre_is_not_a_hole():
    """v3 E: a centre darker than the rim but still particle-bright is not a hole.

    (SEM edge brightening makes solid spheres look like this.) The original
    size rule calls it a cavity.
    """
    outer, centre = _disk((60, 60), 30), _disk((60, 60), 18)
    draw = [(outer, PARTICLE), (centre, 0.42)]  # rho ~0.76: particle-like
    result, ids = _panel(draw, [outer, centre])
    assert _roles(result, ids)[1] != "cavity"
    original, ids = _panel(draw, [outer, centre], cfg=_cfg(cavity={"rule": "size"}))
    assert _roles(original, ids)[1] == "cavity"


def test_gap_inside_merged_outline_does_not_block_split():
    """v3 C: a gap enclosed by a merged outline is not a child that blocks C-SPLIT."""
    a, b = _disk((60, 50), 22), _disk((60, 90), 22)
    waist = np.zeros(SHAPE, dtype=bool)
    waist[48:72, 66:74] = True
    dumbbell = a | b | waist
    gap = waist & ~(a | b)  # background-level region inside the outline
    result, ids = _panel([(a | b, PARTICLE)], [dumbbell, a, b, gap])
    roles = _roles(result, ids)
    assert roles[:3] == ["merged", "particle", "particle"]
    assert roles[3] in ("gap", "speck")
    original, ids = _panel([(a | b, PARTICLE)], [dumbbell, a, b, gap], cfg=_cfg(group_members_only=False))
    assert _roles(original, ids)[0] != "merged"


def test_frame_filling_particle_is_not_background():
    """v3 F: a round particle filling the frame (Step 2 flags it bg_candidate) is a
    truncated particle, and the rebuilt graph lets it own its pore."""
    big = _disk((115, 120), 95)  # touches the bottom and right edges; > 25% of the panel
    pore = _disk((115, 115), 12)
    draw = [(big, PARTICLE), (pore, CAVITY)]
    result, ids = _panel(draw, [big, pore], with_seeds=False)
    nodes = _by_id(result)
    sphere = nodes[ids[0]]
    assert sphere["bg_candidate"] is False and sphere["frame_particle"] is True
    assert (sphere["role"], sphere["truncated"]) == ("particle", True)
    assert nodes[ids[1]]["role"] == "cavity" and nodes[ids[1]]["parent"] == ids[0]
    assert "FRAME_PARTICLE" in result.panel["flags"]

    original, ids = _panel(draw, [big, pore], with_seeds=False, cfg=_cfg(frame_particle=False))
    assert _roles(original, ids)[0] == "background"
