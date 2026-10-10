"""Toy panels shared by the Step-4 and Step-6 tests (spec 11).

TEM-like: background 0.8, particle 0.3, cavity 0.6, plus light noise. Masks go
through the real Step-2 post-processing, Step-3 graph and Step-4 resolve, so
features and relations are computed exactly as in the pipeline, without SAM.
"""

import copy

import numpy as np
from skimage.draw import disk

from lnpsam import graph, proposals, resolve
from lnpsam import io as lnp_io

CFG = lnp_io.load_config("config.yaml")
SHAPE = (200, 200)
BG, PARTICLE, CAVITY = 0.8, 0.3, 0.6

# Three isolated solid disks so every panel has confident seeds (8.1).
SEED_CENTERS = [(170, 30), (170, 80), (170, 130)]
SEED_R = 12


def disk_mask(center, radius):
    m = np.zeros(SHAPE, dtype=bool)
    rr, cc = disk(center, radius, shape=SHAPE)
    m[rr, cc] = True
    return m


def build_panel(draw, masks, with_seeds=True, seed=0, cfg=CFG, nm_per_px=None):
    """Draw a toy panel, run Step-2 post-processing, Step 3 and Step 4.

    ``draw`` is a list of ``(mask, intensity)`` painted in order onto the
    background; ``masks`` are the SAM masks. Returns ``(result, ids)`` where
    ``ids[i]`` is the node id of ``masks[i]``.
    """
    gray = np.full(SHAPE, BG, dtype=np.float32)
    all_masks = list(masks)
    if with_seeds:
        for c in SEED_CENTERS:
            seed_mask = disk_mask(c, SEED_R)
            draw = list(draw) + [(seed_mask, PARTICLE)]
            all_masks.append(seed_mask)
    for mask, value in draw:
        gray[mask] = value
    gray += np.random.default_rng(seed).normal(0, 0.02, SHAPE).astype(np.float32)
    gray = np.clip(gray, 0, 1)

    annotations = [
        {
            "segmentation": m,
            "predicted_iou": 0.95,
            "stability_score": 0.95,
            "point_coords": [[0.0, 0.0]],
        }
        for m in all_masks
    ]
    content_mask = np.ones(SHAPE, dtype=bool)
    nodes = proposals.postprocess_proposals(
        annotations,
        gray,
        exclusion=np.zeros(SHAPE, dtype=bool),
        content_mask=content_mask,
        content_box=(0, 0, SHAPE[1], SHAPE[0]),
        crop_offset=(0, 0),
        sam_config=cfg["sam"],
    )
    graph_result = graph.run_graph(nodes, cfg)
    result = resolve.run_resolve(graph_result, gray, content_mask, "TEM", nm_per_px, cfg)

    # 8.6: every node has a role and a rule_id.
    for n in result.nodes:
        assert n["role"] in resolve.ROLES
        assert n["rule_id"]
    return result, list(range(len(masks)))


def by_id(result):
    return {n["node_id"]: n for n in result.nodes}


def roles(result, ids):
    nodes = by_id(result)
    return [nodes[i]["role"] for i in ids]


def cfg_with(**resolve_overrides):
    """CFG with ``resolve:`` keys replaced (dict values update sub-dicts)."""
    cfg = copy.deepcopy(CFG)
    for key, value in resolve_overrides.items():
        if isinstance(value, dict):
            cfg["resolve"][key].update(value)
        else:
            cfg["resolve"][key] = value
    return cfg
