"""Tests for Step 3 relation graph (spec 7)."""

import numpy as np

from lnpsam import graph


CFG = {
    "graph": {
        "dup_iou": 0.85,
        "contain_frac": 0.90,
        "contain_area_ratio": 1.10,
        "adjacent_px": 2,
    }
}


def _node(node_id, mask, quality=0.9, bg_candidate=False):
    rows, cols = np.nonzero(mask)
    return {
        "node_id": node_id,
        "filled": mask,
        "bbox": (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        ),
        "area": int(mask.sum()),
        "quality": quality,
        "bg_candidate": bg_candidate,
    }


def _rect(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _by_id(result):
    return {node["node_id"]: node for node in result.nodes}


def test_iou_duplicate_keeps_highest_quality_representative():
    shape = (30, 35)
    left = _rect(shape, 5, 25, 5, 25)
    shifted = _rect(shape, 5, 25, 6, 26)  # IoU = 380 / 420 > 0.9

    result = graph.run_graph(
        [_node(10, left, quality=0.8), _node(11, shifted, quality=0.95)], CFG
    )
    nodes = _by_id(result)

    assert result.representative_ids == [11]
    assert result.duplicate_clusters == [[10, 11]]
    assert nodes[10]["role"] == "duplicate"
    assert nodes[10]["representative_id"] == 11
    assert nodes[11]["is_representative"] is True
    assert result.pairs[0]["iou"] > 0.9


def test_duplicate_union_find_is_transitive():
    shape = (30, 40)
    first = _rect(shape, 5, 25, 5, 25)
    middle = _rect(shape, 5, 25, 6, 26)
    last = _rect(shape, 5, 25, 7, 27)

    result = graph.run_graph(
        [
            _node(1, first, quality=0.8),
            _node(2, middle, quality=0.99),
            _node(3, last, quality=0.9),
        ],
        CFG,
    )

    assert result.representative_ids == [2]
    assert result.duplicate_clusters == [[1, 2, 3]]
    assert _by_id(result)[1]["role"] == "duplicate"
    assert _by_id(result)[3]["role"] == "duplicate"


def test_parent_is_smallest_qualifying_non_background_representative():
    shape = (30, 30)
    child = _rect(shape, 12, 16, 12, 16)
    medium = _rect(shape, 10, 18, 10, 18)
    large = _rect(shape, 7, 22, 7, 22)

    result = graph.run_graph(
        [
            _node(1, child),
            _node(2, medium, bg_candidate=True),
            _node(3, large),
        ],
        CFG,
    )
    nodes = _by_id(result)

    assert nodes[1]["parent"] == 3
    assert nodes[2]["parent"] == 3
    assert nodes[3]["children"] == [1, 2]

    result_without_bg_flag = graph.run_graph(
        [_node(1, child), _node(2, medium), _node(3, large)], CFG
    )
    assert _by_id(result_without_bg_flag)[1]["parent"] == 2


def test_adjacency_is_symmetric_and_excludes_overlapping_masks():
    shape = (20, 30)
    first = _rect(shape, 5, 9, 2, 6)
    near = _rect(shape, 5, 9, 8, 12)  # closest pixel centers are 3 px apart
    overlapping = _rect(shape, 6, 10, 4, 8)
    far = _rect(shape, 5, 9, 20, 24)

    config = {"graph": {**CFG["graph"], "adjacent_px": 3}}
    result = graph.run_graph(
        [
            _node(1, first),
            _node(2, near),
            _node(3, overlapping),
            _node(4, far),
        ],
        config,
    )
    nodes = _by_id(result)

    assert 2 in nodes[1]["adjacent"]
    assert 1 in nodes[2]["adjacent"]
    assert 3 not in nodes[1]["adjacent"]
    assert nodes[4]["adjacent"] == []


def test_parent_coverage_uses_union_of_direct_children():
    shape = (20, 20)
    parent = _rect(shape, 2, 12, 2, 12)  # area 100
    child_a = _rect(shape, 3, 8, 3, 8)  # area 25
    child_b = _rect(shape, 6, 11, 6, 11)  # area 25, overlap 4

    result = graph.run_graph(
        [_node(1, parent), _node(2, child_a), _node(3, child_b)], CFG
    )
    nodes = _by_id(result)

    assert nodes[2]["parent"] == 1
    assert nodes[3]["parent"] == 1
    assert nodes[1]["children"] == [2, 3]
    assert nodes[1]["coverage"] == 46 / 100


def test_graph_does_not_mutate_step_2_nodes():
    mask = _rect((10, 10), 2, 6, 2, 6)
    original = _node(1, mask)

    graph.run_graph([original], CFG)

    assert "parent" not in original
    assert "representative_id" not in original


def test_empty_graph_is_valid():
    result = graph.run_graph([], CFG)

    assert result == graph.GraphResult([], [], [], [], [])
