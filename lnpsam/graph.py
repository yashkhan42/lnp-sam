"""Step 3: relation graph over filled proposal masks (spec 7)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GraphResult:
    """All Step-3 products for one panel.

    ``nodes`` contains every input node, including duplicates. Graph fields are
    added to shallow copies so Step 2's cached nodes are not mutated. ``pairs``
    contains the section 7.1 candidate pairs before duplicate removal, while
    ``representative_pairs`` contains the pairs allowed to continue through
    the rest of Step 3.
    """

    nodes: list[dict]
    pairs: list[dict]
    representative_pairs: list[dict]
    representative_ids: list[int]
    duplicate_clusters: list[list[int]]


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _graph_config(cfg):
    try:
        graph_cfg = cfg["graph"]
        dup_iou = float(graph_cfg["dup_iou"])
        contain_frac = float(graph_cfg["contain_frac"])
        contain_area_ratio = float(graph_cfg["contain_area_ratio"])
        adjacent_px = float(graph_cfg["adjacent_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "cfg.graph must define numeric dup_iou, contain_frac, "
            "contain_area_ratio, and adjacent_px"
        ) from exc

    if not 0.0 <= dup_iou <= 1.0:
        raise ValueError("graph.dup_iou must be in [0, 1]")
    if not 0.0 <= contain_frac <= 1.0:
        raise ValueError("graph.contain_frac must be in [0, 1]")
    if contain_area_ratio < 1.0:
        raise ValueError("graph.contain_area_ratio must be at least 1")
    if adjacent_px < 0.0:
        raise ValueError("graph.adjacent_px must be non-negative")
    return dup_iou, contain_frac, contain_area_ratio, adjacent_px


def _validate_nodes(nodes):
    required = {"node_id", "filled", "bbox", "area", "quality", "bg_candidate"}
    seen_ids = set()
    shape = None
    validated = []

    for position, original in enumerate(nodes):
        missing = required - set(original)
        if missing:
            raise ValueError(f"node {position} is missing keys: {sorted(missing)}")

        node = dict(original)
        node_id = node["node_id"]
        if not isinstance(node_id, (int, np.integer)):
            raise ValueError(f"node_id must be an integer, got {node_id!r}")
        node_id = int(node_id)
        if node_id in seen_ids:
            raise ValueError(f"duplicate node_id: {node_id}")
        seen_ids.add(node_id)
        node["node_id"] = node_id

        filled = np.asarray(node["filled"], dtype=bool)
        if filled.ndim != 2:
            raise ValueError(f"node {node_id} filled mask must be 2-D")
        if shape is None:
            shape = filled.shape
        elif filled.shape != shape:
            raise ValueError("all filled masks must have the same panel shape")
        area = int(filled.sum())
        if area == 0:
            raise ValueError(f"node {node_id} has an empty filled mask")
        if int(node["area"]) != area:
            raise ValueError(
                f"node {node_id} area does not match its filled mask: "
                f"{node['area']} != {area}"
            )

        x0, y0, x1, y1 = (int(value) for value in node["bbox"])
        rows, cols = np.nonzero(filled)
        expected_bbox = (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        )
        if (x0, y0, x1, y1) != expected_bbox:
            raise ValueError(
                f"node {node_id} bbox does not match its filled mask: "
                f"{(x0, y0, x1, y1)} != {expected_bbox}"
            )

        node["filled"] = filled
        node["bbox"] = expected_bbox
        node["area"] = area
        node["quality"] = float(node["quality"])
        node["bg_candidate"] = bool(node["bg_candidate"])
        validated.append(node)

    return validated


def _bbox_distance(left_bbox, right_bbox):
    """Euclidean distance between the closest pixel centers of two boxes."""
    left_x0, left_y0, left_x1, left_y1 = left_bbox
    right_x0, right_y0, right_x1, right_y1 = right_bbox
    dx = max(left_x0 - (right_x1 - 1), right_x0 - (left_x1 - 1), 0)
    dy = max(left_y0 - (right_y1 - 1), right_y0 - (left_y1 - 1), 0)
    return float(math.hypot(dx, dy))


def _candidate_index_pairs(nodes, adjacent_px):
    """Section 7.1 bbox sweep, sorted by x0."""
    order = sorted(
        range(len(nodes)),
        key=lambda index: (nodes[index]["bbox"][0], nodes[index]["node_id"]),
    )
    pairs = []
    for order_position, left_index in enumerate(order):
        left = nodes[left_index]
        left_last_x = left["bbox"][2] - 1
        for right_index in order[order_position + 1 :]:
            right = nodes[right_index]
            if right["bbox"][0] - left_last_x > adjacent_px:
                break
            bbox_distance = _bbox_distance(left["bbox"], right["bbox"])
            if bbox_distance <= adjacent_px:
                if left["node_id"] <= right["node_id"]:
                    first, second = left_index, right_index
                else:
                    first, second = right_index, left_index
                pairs.append((first, second, bbox_distance))
    pairs.sort(key=lambda item: (nodes[item[0]]["node_id"], nodes[item[1]]["node_id"]))
    return pairs


def _pair_metrics(left, right, bbox_distance):
    inter = int(np.logical_and(left["filled"], right["filled"]).sum())
    union = left["area"] + right["area"] - inter
    return {
        "a": left["node_id"],
        "b": right["node_id"],
        "inter": inter,
        "iou": float(inter / union),
        "c_ab": float(inter / left["area"]),
        "c_ba": float(inter / right["area"]),
        "bbox_distance": bbox_distance,
    }


def _compute_pairs(nodes, adjacent_px):
    pairs = []
    for left_index, right_index, bbox_distance in _candidate_index_pairs(
        nodes, adjacent_px
    ):
        pairs.append(
            _pair_metrics(nodes[left_index], nodes[right_index], bbox_distance)
        )
    return pairs


def _duplicate_components(nodes, pairs, dup_iou):
    id_to_index = {node["node_id"]: index for index, node in enumerate(nodes)}
    union_find = _UnionFind(len(nodes))
    for pair in pairs:
        if pair["iou"] >= dup_iou:
            union_find.union(id_to_index[pair["a"]], id_to_index[pair["b"]])

    members_by_root = {}
    for index in range(len(nodes)):
        members_by_root.setdefault(union_find.find(index), []).append(index)

    components = list(members_by_root.values())
    components.sort(key=lambda members: min(nodes[index]["node_id"] for index in members))
    return components


def _select_representatives(nodes, components):
    representative_indices = []
    duplicate_clusters = []
    for members in components:
        # The spec's highest-quality choice is primary. Node ID only makes an
        # exactly tied result deterministic.
        representative = min(
            members,
            key=lambda index: (-nodes[index]["quality"], nodes[index]["node_id"]),
        )
        representative_id = nodes[representative]["node_id"]
        representative_indices.append(representative)
        cluster_ids = sorted(nodes[index]["node_id"] for index in members)
        duplicate_clusters.append(cluster_ids)

        for index in members:
            node = nodes[index]
            node["representative_id"] = representative_id
            node["is_representative"] = index == representative
            node["parent"] = None
            node["children"] = []
            node["adjacent"] = []
            node["coverage"] = 0.0
            if index != representative:
                node["role"] = "duplicate"

    representative_indices.sort(key=lambda index: nodes[index]["node_id"])
    return representative_indices, duplicate_clusters


def _assign_parents(representatives, pairs, contain_frac, contain_area_ratio):
    by_id = {node["node_id"]: node for node in representatives}
    candidates = {node["node_id"]: [] for node in representatives}

    for pair in pairs:
        left = by_id[pair["a"]]
        right = by_id[pair["b"]]
        if (
            pair["c_ab"] >= contain_frac
            and right["area"] >= contain_area_ratio * left["area"]
            and not right["bg_candidate"]
        ):
            candidates[left["node_id"]].append(right)
        if (
            pair["c_ba"] >= contain_frac
            and left["area"] >= contain_area_ratio * right["area"]
            and not left["bg_candidate"]
        ):
            candidates[right["node_id"]].append(left)

    for child in representatives:
        possible_parents = candidates[child["node_id"]]
        if not possible_parents:
            continue
        parent = min(possible_parents, key=lambda node: (node["area"], node["node_id"]))
        child["parent"] = parent["node_id"]
        parent["children"].append(child["node_id"])

    for node in representatives:
        node["children"].sort()


def _mask_distance(left, right):
    """Exact Euclidean distance between closest foreground pixel centers."""
    x0 = min(left["bbox"][0], right["bbox"][0])
    y0 = min(left["bbox"][1], right["bbox"][1])
    x1 = max(left["bbox"][2], right["bbox"][2])
    y1 = max(left["bbox"][3], right["bbox"][3])
    left_crop = left["filled"][y0:y1, x0:x1]
    right_crop = right["filled"][y0:y1, x0:x1]
    distance_to_left = ndimage.distance_transform_edt(~left_crop)
    return float(distance_to_left[right_crop].min())


def _assign_adjacency(representatives, pairs, adjacent_px):
    by_id = {node["node_id"]: node for node in representatives}
    for pair in pairs:
        if pair["inter"] != 0:
            pair["mask_distance"] = 0.0
            continue
        left = by_id[pair["a"]]
        right = by_id[pair["b"]]
        mask_distance = _mask_distance(left, right)
        pair["mask_distance"] = mask_distance
        if mask_distance <= adjacent_px:
            left["adjacent"].append(right["node_id"])
            right["adjacent"].append(left["node_id"])

    for node in representatives:
        node["adjacent"].sort()


def _compute_children_coverage(representatives):
    by_id = {node["node_id"]: node for node in representatives}
    for parent in representatives:
        if not parent["children"]:
            continue
        child_union = np.zeros_like(parent["filled"], dtype=bool)
        for child_id in parent["children"]:
            child_union |= by_id[child_id]["filled"]
        parent["coverage"] = float(child_union.sum() / parent["area"])


def run_graph(nodes, cfg):
    """Build the complete Step-3 relation graph from Step-2 nodes.

    Pixel proximity uses Euclidean distance between pixel centers for both the
    bbox sweep and exact mask adjacency. All overlap and containment metrics use
    the filled masks, as required by sections 4.4 and 7.
    """
    dup_iou, contain_frac, contain_area_ratio, adjacent_px = _graph_config(cfg)
    graph_nodes = _validate_nodes(nodes)
    if not graph_nodes:
        return GraphResult([], [], [], [], [])

    pairs = _compute_pairs(graph_nodes, adjacent_px)
    components = _duplicate_components(graph_nodes, pairs, dup_iou)
    representative_indices, duplicate_clusters = _select_representatives(
        graph_nodes, components
    )
    representatives = [graph_nodes[index] for index in representative_indices]

    # Sweep again after duplicate removal: only the chosen masks continue, and
    # their own bboxes determine all containment and adjacency candidates.
    representative_pairs = _compute_pairs(representatives, adjacent_px)
    _assign_parents(
        representatives,
        representative_pairs,
        contain_frac,
        contain_area_ratio,
    )
    _assign_adjacency(representatives, representative_pairs, adjacent_px)
    _compute_children_coverage(representatives)

    representative_ids = [node["node_id"] for node in representatives]
    graph_nodes.sort(key=lambda node: node["node_id"])
    return GraphResult(
        nodes=graph_nodes,
        pairs=pairs,
        representative_pairs=representative_pairs,
        representative_ids=representative_ids,
        duplicate_clusters=duplicate_clusters,
    )
