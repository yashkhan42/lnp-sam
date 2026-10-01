"""Step 4: Resolve (spec 8) -- give every representative mask a role and rule_id.

Input is the Step-3 :class:`lnpsam.graph.GraphResult`. The order is:

1. Panel statistics (8.1): seeds, background level ``mu_bg``, particle level
   ``mu_p`` and the normalized intensity ``rho``; reference size and ``d_min``.
2. Per-node values (8.2): ``rho_in``, ``rho_rim``, ``pl``, ring values, ``contact``.
3. Roots are typed with the 8.3 table; children of every ``particle`` or
   ``aggregate`` with the 8.4 group rules and then the per-child rules. Children
   of dropped nodes move up and are typed again (8.3 re-rooting, 8.4).
4. Edge flags (8.5): ``truncated`` and ``touches_exclusion`` propagate from a
   particle to its cavity, core and spots.

Nothing is deleted: every representative leaves with ``role`` and ``rule_id``
(8.6), duplicates keep ``role = duplicate``, and C-UNRESOLVED cases are logged.

Interpretations where the spec is silent (all documented on the functions):
- ``ring_free`` (8.2) does not subtract ``bg_candidate`` masks. They are the
  background, as in 8.1 step 2; ``ring_rho`` already tells background from
  particle material, and subtracting them would hide a missed particle that SAM
  merged into its background mask.
- ``contact`` (8.2) ignores the node's own ancestors and descendants: it measures
  contact with neighbours, not with masks nested inside or around it.
- The 8.4 rules apply to ``aggregate`` parents as well as ``particle`` parents
  (8.3 re-rooting keeps children under aggregates, so they need typing). A child
  of an aggregate typed C-STACKED gets ``aggregate_id`` = that aggregate, and the
  children of a C-SPLIT node inherit its ``aggregate_id``.
- Descendants of a ``cavity``, ``core`` or ``orphan`` are all ``spot``
  (rule ``C-SPOT-NESTED``), counted in ``n_spots`` of the owning particle.

v3 amendments (LNP-SAM_MVP_SPEC_AMENDMENTS.md; each has a config switch that
restores the original rule):
- A ``seed_polarity_check``: seeds must lie on the ``polarity_prior`` side of
  ``mu_bg`` (8.1), so pores and background holes cannot become seeds.
- B ``bg_rules_use: pl``: R-BGPATCH tests ``pl`` (rim included), so a particle
  with a background-level interior but a particle-level rim is not called
  background (8.3). R-GAP keeps ``rho_in``: a gap's outline runs along its
  neighbours' edges, so its rim is particle-like by construction.
- C ``group_members_only``: C-SPLIT and C-AGG count only particle-like children
  at least ``d_min`` across (coverage too), so specks do not turn a single
  particle into an aggregate and a gap enclosed by a merged outline does not
  block its split (8.4). After a split every child, members or not, is retyped
  as a root.
- D ``large_round``: a mask above ``particle.max_area_frac`` that is round and
  solid is one particle (``R-PARTICLE-LARGE``), not an R-LARGE aggregate (8.3).
- E ``cavity.rule: single_hole``: a particle is hollow when it has exactly one
  hole near its centre, of any size from ``d_min``. A hole is a child that is
  background-dark (``rho_in < rho_particle_min``) and darker than the shell by
  ``rho_margin``; overlapping masks of the same hole count once, and hole-like
  children smaller than ``d_min`` do not count. Every other hole-like child is a
  spot (``C-HOLE-SPOT``) (8.4).
- F ``frame_particle``: a ``bg_candidate`` that is round and solid and lies on
  the particle side of the rest of the content (brighter on SEM, darker on TEM)
  is one particle filling the frame, not background. Its flag is cleared and
  the Step-3 graph is rebuilt so it can parent its pore (8.1 step 0).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.segmentation import find_boundaries

from . import graph as graph_mod

# Literal values from the spec text (not tunable thresholds).
_BG_DILATE_PX = 2  # 8.1 step 2: dilate the union of representatives by 2 px
_MIN_D_PX = 3.0  # 8.1 step 5: d_min is never below 3 px
_N_SECTORS = 12  # 8.2: ring_std over 12 equal angular sectors
_SHELL_DILATE_PX = 1  # 8.4: rho_shell subtracts the children dilated by 1 px
_RHO_EPS = 1e-3  # guard against mu_p == mu_bg (division by zero in rho)

_CONTAINER_ROLES = ("particle", "aggregate")
_DROPPED_ROOT_ROLES = ("background", "bg_patch", "speck", "gap", "rejected")
_NESTING_ROLES = ("cavity", "core", "orphan")

ROLES = (
    "particle", "cavity", "core", "spot", "merged", "aggregate", "orphan", "gap",
    "bg_patch", "background", "speck", "rejected", "unresolved", "duplicate",
)


@dataclass
class ResolveResult:
    """Step-4 products for one panel.

    ``nodes`` holds every Step-3 node (duplicates included) as shallow copies
    with ``role``, ``rule_id`` and the 8.2 values added. ``parent``/``children``
    describe the resolved tree; ``graph_parent`` keeps the Step-3 parent.
    ``panel`` holds the 8.1 statistics, panel flags and role counts. ``log``
    lists every C-UNRESOLVED case.
    """

    nodes: list
    panel: dict
    log: list


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _window(bbox, pad, shape):
    """Crop window ``(y0, y1, x0, x1)`` around an ``(x0, y0, x1, y1)`` bbox."""
    x0, y0, x1, y1 = bbox
    H, W = shape
    return max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)


class _BoxIndex:
    """Vectorized bbox-overlap lookup over the representatives."""

    def __init__(self, nodes):
        self.nodes = nodes
        boxes = np.array([n["bbox"] for n in nodes], dtype=np.int64).reshape(-1, 4)
        self.x0, self.y0, self.x1, self.y1 = boxes.T

    def overlapping(self, win):
        y0, y1, x0, x1 = win
        hit = (self.x0 < x1) & (self.x1 > x0) & (self.y0 < y1) & (self.y1 > y0)
        return [self.nodes[i] for i in np.flatnonzero(hit)]


def _union_crop(nodes, win):
    y0, y1, x0, x1 = win
    out = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for n in nodes:
        out |= n["filled"][y0:y1, x0:x1]
    return out


def _is_set(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


# --------------------------------------------------------------------------- #
# 8.1 Panel statistics
# --------------------------------------------------------------------------- #
# Step-3 fields that run_graph sets; dropped before the graph is rebuilt.
_GRAPH_FIELDS = (
    "representative_id", "is_representative", "parent", "children", "adjacent",
    "coverage", "role",
)


def frame_particles(nodes, gray, content_mask, modality, rcfg):
    """v3 F (8.1 step 0): ids of ``bg_candidate`` masks that are one frame-filling particle.

    Step 2 flags ``bg_candidate`` by size and edge contact only, so a close-up
    particle that fills the frame qualifies. It is a particle when it is round
    and solid (the ``large_round`` thresholds) and its own pixels lie on the
    particle side of the content outside it. A real background mask fails one
    test: hole-filled it is a near-rectangle (round, solid) but on the
    background side, or it is irregular.
    """
    if not rcfg.get("frame_particle", False):
        return []
    polarity = int(rcfg["polarity_prior"][modality])
    big = rcfg["large_round"]
    found = []
    for n in nodes:
        if not n["bg_candidate"] or not n.get("is_representative", True):
            continue
        if n["circularity"] < big["circularity"] or n["solidity"] < big["solidity"]:
            continue
        own = np.asarray(n.get("mask", n["filled"]), dtype=bool) & content_mask
        rest = content_mask & ~n["filled"]
        if own.any() and rest.any() and polarity * (gray[own].mean() - gray[rest].mean()) > 0:
            found.append(n["node_id"])
    return found


def _regraph_without_bg_flag(graph_result, node_ids, cfg):
    """Rebuild the Step-3 graph with ``bg_candidate`` cleared on ``node_ids``."""
    nodes = []
    for n in graph_result.nodes:
        node = {k: v for k, v in n.items() if k not in _GRAPH_FIELDS}
        if node["node_id"] in node_ids:
            node["bg_candidate"] = False
        nodes.append(node)
    return graph_mod.run_graph(nodes, cfg)


def _select_seeds(reps, seed_cfg, content_area):
    return [
        n
        for n in reps
        if n["parent"] is None
        and not n["bg_candidate"]
        and not n["touches_border"]
        and n["solidity"] >= seed_cfg["solidity"]
        and n["circularity"] >= seed_cfg["circularity"]
        and n["quality"] >= seed_cfg["quality"]
        and n["eqdiam"] >= seed_cfg["min_d_px"]
        and n["area"] <= seed_cfg["max_area_frac"] * content_area
    ]


def panel_statistics(reps, gray, content_mask, modality, nm_per_px, rcfg):
    """Spec 8.1: seeds, ``mu_bg``, ``mu_p``, ``d_ref`` and ``d_min``.

    ``mu_bg`` is computed before the seeds because v3 A filters seeds by which
    side of the background they lie on; ``mu_bg`` itself never uses seeds.
    """
    content_area = int(content_mask.sum())
    polarity = int(rcfg["polarity_prior"][modality])
    flags = []

    # Background: content not within 2 px of any non-background representative.
    covered = np.zeros(gray.shape, dtype=bool)
    for n in reps:
        if not n["bg_candidate"]:
            covered |= n["filled"]
    if covered.any():
        covered = ndimage.distance_transform_edt(~covered) <= _BG_DILATE_PX
    free = content_mask & ~covered
    content_gray = gray[content_mask]
    if free.sum() >= rcfg["dense_bg_min_frac"] * content_area:
        mu_bg = float(np.median(gray[free]))
        bg_mode = "free"
    else:
        mu_bg = float(np.percentile(content_gray, 10 if polarity > 0 else 90))
        bg_mode = "dense"
        flags.append("DENSE_PANEL")

    def on_particle_side(nodes):
        # v3 A: a seed must be on the polarity_prior side of the background.
        if not rcfg.get("seed_polarity_check", False):
            return nodes
        return [n for n in nodes if polarity * (float(gray[n["filled"]].mean()) - mu_bg) > 0]

    seeds = on_particle_side(_select_seeds(reps, rcfg["seed"], content_area))
    seed_mode = "strict"
    if len(seeds) < rcfg["seed"]["min_count"]:
        relaxed = {**rcfg["seed"], **rcfg["seed_relaxed"]}
        seeds = on_particle_side(_select_seeds(reps, relaxed, content_area))
        seed_mode = "relaxed"
        flags.append("SEEDS_RELAXED")

    if seeds:
        mu_p = float(np.median([float(gray[n["filled"]].mean()) for n in seeds]))
    else:
        mu_p = float(np.percentile(content_gray, 90 if polarity > 0 else 10))
        flags.append("NO_SEEDS")

    if abs(mu_p - mu_bg) < _RHO_EPS:
        mu_p = mu_bg + polarity * _RHO_EPS
        flags.append("DEGENERATE_CONTRAST")
    elif np.sign(mu_p - mu_bg) != polarity:
        # Diagnostic only: seeds are on the unexpected side of the background.
        flags.append("POLARITY_MISMATCH")

    d_ref = float(np.median([n["eqdiam"] for n in seeds])) if seeds else None
    if _is_set(nm_per_px) and nm_per_px > 0:
        d_min = rcfg["min_particle_nm"] / float(nm_per_px)
        d_min_source = "calibrated"
    elif d_ref is not None:
        d_min = rcfg["speck_rel_d"] * d_ref
        d_min_source = "relative"
    else:
        d_min = _MIN_D_PX
        d_min_source = "floor"
        flags.append("NO_D_REF")
    d_min = max(_MIN_D_PX, float(d_min))

    return {
        "modality": modality,
        "polarity": polarity,
        "content_area": content_area,
        "seed_mode": seed_mode,
        "seed_ids": sorted(n["node_id"] for n in seeds),
        "n_seeds": len(seeds),
        "bg_mode": bg_mode,
        "mu_bg": mu_bg,
        "mu_p": mu_p,
        "d_ref": d_ref,
        "d_min": d_min,
        "d_min_source": d_min_source,
        "nm_per_px": float(nm_per_px) if _is_set(nm_per_px) else None,
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# 8.2 Per-node values
# --------------------------------------------------------------------------- #
def _tree_relatives(reps):
    """Graph ancestors and descendants of each representative (8.2 contact)."""
    by_id = {n["node_id"]: n for n in reps}
    relatives = {nid: set() for nid in by_id}
    for nid, node in by_id.items():
        parent = node["parent"]
        while parent is not None:
            relatives[nid].add(parent)
            relatives[parent].add(nid)
            parent = by_id[parent]["parent"]
    return relatives


def node_values(reps, gray, content_mask, stats, rcfg, adjacent_px):
    """Spec 8.2: add ``rho_in``, ``rho_rim``, ``pl``, ring values and ``contact``."""
    mu_bg, mu_p = stats["mu_bg"], stats["mu_p"]

    def rho(pixels):
        return float((pixels.mean() - mu_bg) / (mu_p - mu_bg)) if pixels.size else math.nan

    index = _BoxIndex(reps)
    non_bg = _BoxIndex([n for n in reps if not n["bg_candidate"]])
    windows = {}

    for n in reps:
        half = n["eqdiam"] / 2.0
        ring_w = max(2.0, rcfg["ring_width_frac"] * half)
        pad = int(math.ceil(max(ring_w, adjacent_px))) + 1
        win = _window(n["bbox"], pad, gray.shape)
        windows[n["node_id"]] = win
        y0, y1, x0, x1 = win
        filled = n["filled"][y0:y1, x0:x1]
        g = gray[y0:y1, x0:x1]

        n["rho_in"] = rho(g[filled])
        depth = ndimage.distance_transform_edt(filled)
        rim = filled & (depth <= max(1.0, rcfg["rim_band_frac"] * half))
        n["rho_rim"] = rho(g[rim])
        n["pl"] = float(np.nanmax([n["rho_in"], n["rho_rim"]]))

        # Ring 1..ring_w px outside the mask, inside content_mask.
        dist_out = ndimage.distance_transform_edt(~filled)
        ring = (dist_out >= 1) & (dist_out <= ring_w) & content_mask[y0:y1, x0:x1]
        others = [m for m in non_bg.overlapping(win) if m["node_id"] != n["node_id"]]
        ring_free = ring & ~_union_crop(others, win)
        n["ring_free_frac"] = float(ring_free.sum() / ring.sum()) if ring.any() else 0.0
        n["ring_rho"] = rho(g[ring_free])

        # ring_std: std of rho over 12 angular sectors (empty sectors skipped).
        rr, cc = np.nonzero(ring_free)
        if rr.size:
            cy, cx = n["centroid"]
            angle = np.arctan2(rr + y0 - cy, cc + x0 - cx)
            sector = np.minimum(
                ((angle + math.pi) / (2 * math.pi) * _N_SECTORS).astype(int), _N_SECTORS - 1
            )
            values = g[rr, cc]
            sector_rho = [rho(values[sector == k]) for k in range(_N_SECTORS) if (sector == k).any()]
            n["ring_std"] = float(np.std(sector_rho)) if len(sector_rho) >= 2 else math.nan
        else:
            n["ring_std"] = math.nan

    # contact needs every node's pl, so it is a second pass.
    relatives = _tree_relatives(reps)
    for n in reps:
        win = windows[n["node_id"]]
        y0, y1, x0, x1 = win
        filled = n["filled"][y0:y1, x0:x1]
        boundary = find_boundaries(filled, mode="inner") & filled
        neighbours = [
            m
            for m in index.overlapping(win)
            if m["node_id"] != n["node_id"]
            and m["node_id"] not in relatives[n["node_id"]]
            and m["pl"] >= rcfg["rho_particle_min"]
        ]
        near = _union_crop(neighbours, win)
        if near.any() and boundary.any():
            dist = ndimage.distance_transform_edt(~near)
            n["contact"] = float((dist[boundary] <= adjacent_px).mean())
        else:
            n["contact"] = 0.0


# --------------------------------------------------------------------------- #
# 8.3-8.4 Rules
# --------------------------------------------------------------------------- #
def root_rule(n, stats, rcfg):
    """Spec 8.3 root table; the first matching rule wins. Returns (role, rule_id)."""
    content_area = stats["content_area"]
    orphan, part = rcfg["orphan"], rcfg["particle"]
    if n["bg_candidate"] and n["rho_in"] <= rcfg["rho_bg_max"]:
        return "background", "R-BG"
    if n["eqdiam"] < stats["d_min"]:
        return "speck", "R-SPECK"
    if (
        n["circularity"] >= orphan["circularity"]
        and n["ring_free_frac"] >= orphan["ring_free_min"]
        and n["ring_rho"] >= orphan["ring_rho_min"]
        and n["ring_std"] <= orphan["ring_std_max"]
    ):
        return "orphan", "R-ORPHAN"
    # R-GAP keeps rho_in: a gap's outline borrows its neighbours' edges, so its
    # rim looks particle-like. v3 B applies to R-BGPATCH only.
    if n["rho_in"] <= rcfg["rho_bg_max"] and n["contact"] >= rcfg["gap_contact_min"]:
        return "gap", "R-GAP"
    if n[rcfg.get("bg_rules_use", "rho_in")] <= rcfg["rho_bg_max"]:
        return "bg_patch", "R-BGPATCH"
    if n["pl"] >= rcfg["rho_particle_min"]:
        if n["area"] > part["max_area_frac"] * content_area:
            big = rcfg.get("large_round", {})
            if (
                big.get("enabled", False)
                and n["circularity"] >= big["circularity"]
                and n["solidity"] >= big["solidity"]
            ):
                return "particle", "R-PARTICLE-LARGE"  # v3 D
            return "aggregate", "R-LARGE"
        if n["edge_contrast"] >= part["edge_min"] and n["quality"] >= part["quality_min"]:
            return "particle", "R-PARTICLE"
    return "rejected", "R-REJECT"


class _Resolver:
    def __init__(self, reps, gray, stats, rcfg):
        self.by_id = {n["node_id"]: n for n in reps}
        self.gray = gray
        self.stats = stats
        self.rcfg = rcfg
        self.log = []
        self.kids = {nid: [] for nid in self.by_id}
        for n in reps:
            n["graph_parent"] = n["parent"]
            n["role"] = None
            n["rule_id"] = None
            n["subtype"] = None
            n["aggregate_id"] = None
            n["split_from"] = None
            n["n_spots"] = 0
            n["rule_trace"] = []
            if n["parent"] is not None:
                self.kids[n["parent"]].append(n["node_id"])
        self.queue = deque(sorted(nid for nid, n in self.by_id.items() if n["parent"] is None))

    def rho(self, pixels):
        mu_bg, mu_p = self.stats["mu_bg"], self.stats["mu_p"]
        return float((pixels.mean() - mu_bg) / (mu_p - mu_bg)) if pixels.size else math.nan

    def plike(self, n):
        return n["pl"] >= self.rcfg["rho_particle_min"]

    def set_role(self, n, role, rule_id):
        n["role"] = role
        n["rule_id"] = rule_id
        n["rule_trace"].append(rule_id)

    # ---- roots --------------------------------------------------------------
    def run(self):
        while self.queue:
            n = self.by_id[self.queue.popleft()]
            n["parent"] = None
            role, rule_id = root_rule(n, self.stats, self.rcfg)
            if role == "orphan":
                n["subtype"] = "cavity" if n["rho_in"] < self.rcfg["rho_particle_min"] else "core"
            self.set_role(n, role, rule_id)
            if role in _CONTAINER_ROLES:
                self.type_children(n)
            elif role == "orphan":
                self.nested_spots(n, owner=n)
            elif role in _DROPPED_ROOT_ROLES:
                # Re-rooting: a root has no particle/aggregate ancestor, so its
                # children become roots and are typed with the 8.3 table.
                for cid in self.kids[n["node_id"]]:
                    self.queue.append(cid)
                self.kids[n["node_id"]] = []

    # ---- children -----------------------------------------------------------
    def geometry(self, p, c):
        r = c["eqdiam"] / p["eqdiam"]
        (py, px), (cy, cx) = p["centroid"], c["centroid"]
        delta = math.hypot(cy - py, cx - px) / (p["eqdiam"] / 2.0)
        return r, delta

    def coverage(self, p, children):
        y0, y1, x0, x1 = _window(p["bbox"], 0, p["filled"].shape)
        union = _union_crop(children, (y0, y1, x0, x1)) & p["filled"][y0:y1, x0:x1]
        return float(union.sum() / p["area"])

    def rho_shell(self, p, children):
        pad = _SHELL_DILATE_PX + 1
        win = _window(p["bbox"], pad, p["filled"].shape)
        y0, y1, x0, x1 = win
        inner = _union_crop(children, win)
        if inner.any():
            inner = ndimage.distance_transform_edt(~inner) <= _SHELL_DILATE_PX
        shell = p["filled"][y0:y1, x0:x1] & ~inner
        return self.rho(self.gray[y0:y1, x0:x1][shell])

    def type_children(self, p):
        """Spec 8.4 on a particle or aggregate ``p``: group rules, then each child."""
        pid = p["node_id"]
        children = [self.by_id[c] for c in self.kids[pid]]
        if not children:
            return
        split, agg = self.rcfg["split"], self.rcfg["aggregate"]
        for c in children:
            c["r"], c["delta"] = self.geometry(p, c)

        members = [c for c in children if self.plike(c)]
        if self.rcfg.get("group_members_only", False):
            # v3 C: the group rules see only particle-sized, particle-like
            # children. A gap enclosed by a merged outline, or a speck, neither
            # counts toward C-SPLIT/C-AGG nor blocks them.
            members = [c for c in members if c["eqdiam"] >= self.stats["d_min"]]
            group = members
        else:
            group = children
        cov = self.coverage(p, group) if group else 0.0
        p["coverage"] = cov

        # C-SPLIT: the parent was 2-3 touching particles merged into one mask.
        if (
            2 <= len(group) <= split["max_children"]
            and all(self.plike(c) and c["r"] >= split["r_min"] for c in group)
            and cov >= split["coverage_min"]
        ):
            self.set_role(p, "merged", "C-SPLIT")
            for c in children:
                c["parent"] = None
                c["r"] = c["delta"] = None  # measured against the merged mask
                c["split_from"] = pid
                c["aggregate_id"] = p["aggregate_id"]
                self.queue.append(c["node_id"])
            self.kids[pid] = []
            return

        remaining = children
        # C-AGG: many particle-like children covering enough of the parent.
        if len(members) >= agg["min_children"] and cov >= agg["coverage_min"]:
            self.set_role(p, "aggregate", "C-AGG")
            for c in members:
                self.set_role(c, "particle", "C-AGG")
                c["aggregate_id"] = pid
            for c in members:
                self.type_children(c)
            remaining = [c for c in children if c not in members]

        self.type_each_child(p, remaining)

    def dominant(self, p, pool):
        if not pool:
            return None
        largest = max(pool, key=lambda c: (c["area"], -c["node_id"]))
        others_small = all(
            c["eqdiam"] / p["eqdiam"] <= self.rcfg["spot_r_max"]
            for c in pool
            if c is not largest
        )
        return largest["node_id"] if others_small else None

    def single_hole(self):
        return self.rcfg["cavity"].get("rule", "size") == "single_hole"

    def is_hole(self, c, rho_shell):
        """A hole: background-dark (not particle-like) and clearly darker than the shell.

        The absolute test keeps SEM edge brightening (a solid sphere's centre is
        darker than its bright rim but still particle-bright) from reading as a hole.
        """
        return (
            c["rho_in"] < self.rcfg["rho_particle_min"]
            and c["rho_in"] <= rho_shell - self.rcfg["cavity"]["rho_margin"]
        )

    def hole_cavity(self, children, rho_shell):
        """v3 E: the cavity is the one hole near the centre, else ``None``.

        SAM often outlines one hole with several overlapping masks, so
        overlapping hole masks form one hole; the largest mask represents it.
        Two or more separate holes mean no cavity.
        """
        cav = self.rcfg["cavity"]
        # Speck-sized hole-like children are noise: spots, not a second hole.
        holes = [
            c for c in children
            if self.is_hole(c, rho_shell) and c["eqdiam"] >= self.stats["d_min"]
        ]
        groups = []
        for c in sorted(holes, key=lambda c: -c["area"]):
            hit = [g for g in groups if any((c["filled"] & h["filled"]).any() for h in g)]
            merged = [c] + [h for g in hit for h in g]
            groups = [g for g in groups if g not in hit] + [merged]
        if len(groups) != 1:
            return None
        h = max(groups[0], key=lambda c: (c["area"], -c["node_id"]))
        if h["delta"] <= cav["delta_max"] and h["r"] <= cav["r_max"]:
            return h["node_id"]
        return None

    def child_rule(self, p, c, is_dominant, rho_shell, cavity_id=None):
        """Spec 8.4 per-child table; the first matching rule wins."""
        cav, st = self.rcfg["cavity"], self.rcfg["stacked"]
        r, delta = c["r"], c["delta"]
        if self.single_hole():
            if c["node_id"] == cavity_id:
                return "cavity", "C-CAVITY"
            if self.is_hole(c, rho_shell):
                return "spot", "C-HOLE-SPOT"
        if is_dominant and cav["r_min"] <= r <= cav["r_max"] and delta <= cav["delta_max"]:
            if not self.single_hole() and c["rho_in"] <= rho_shell - cav["rho_margin"]:
                return "cavity", "C-CAVITY"
            if c["rho_in"] >= rho_shell + cav["rho_margin"]:
                return "core", "C-CORE"
        if (
            r >= st["r_min"]
            and delta > st["delta_min"]
            and c["circularity"] >= st["circularity"]
            and c["rho_in"] >= rho_shell - st["rho_tol"]
        ):
            return "particle", "C-STACKED"
        if r < self.rcfg["spot_r_max"]:
            return "spot", "C-SPOT"
        return "unresolved", "C-UNRESOLVED"

    def type_each_child(self, p, remaining):
        pid = p["node_id"]
        if not remaining:
            return
        rho_shell = self.rho_shell(p, [self.by_id[c] for c in self.kids[pid]])
        p["rho_shell"] = rho_shell
        pool = list(remaining)
        pending = list(remaining)
        for c in pending:
            c["r"], c["delta"] = self.geometry(p, c)
        # The cavity is chosen once, from the direct children (v3 E); masks that
        # move up later are never a second cavity.
        cavity_id = self.hole_cavity(pending, rho_shell) if self.single_hole() else None
        while pending:
            dominant_id = self.dominant(p, pool)
            moved = []
            for c in pending:
                c["r"], c["delta"] = self.geometry(p, c)
                role, rule_id = self.child_rule(
                    p, c, c["node_id"] == dominant_id, rho_shell, cavity_id
                )
                self.set_role(c, role, rule_id)
                if role in ("cavity", "core"):
                    self.nested_spots(c, owner=p)
                elif role == "particle":  # C-STACKED
                    if p["role"] == "aggregate":
                        c["aggregate_id"] = pid
                    self.type_children(c)
                else:  # spot or unresolved: dropped; its children move up to p
                    if role == "spot":
                        p["n_spots"] += 1
                    else:
                        self.log.append(
                            f"C-UNRESOLVED node {c['node_id']} under {pid}: "
                            f"r={c['r']:.2f}, delta={c['delta']:.2f}, "
                            f"rho_in={c['rho_in']:.2f}, rho_shell={rho_shell:.2f}"
                        )
                    for gid in self.kids[c["node_id"]]:
                        g = self.by_id[gid]
                        g["parent"] = pid
                        self.kids[pid].append(gid)
                        moved.append(g)
                    self.kids[c["node_id"]] = []
            pool.extend(moved)
            pending = moved

    def nested_spots(self, n, owner):
        """Children (and deeper descendants) of a cavity, core or orphan are spots."""
        stack = list(self.kids[n["node_id"]])
        while stack:
            d = self.by_id[stack.pop()]
            self.set_role(d, "spot", "C-SPOT-NESTED")
            owner["n_spots"] += 1
            stack.extend(self.kids[d["node_id"]])

    # ---- 8.5 ----------------------------------------------------------------
    def edge_flags(self):
        for n in self.by_id.values():
            n["truncated"] = bool(n["role"] == "particle" and n["touches_border"])
            n["touches_exclusion"] = bool(n.get("touches_exclusion", False))
        for p in self.by_id.values():
            if p["role"] != "particle":
                continue
            stack = list(self.kids[p["node_id"]])
            while stack:
                d = self.by_id[stack.pop()]
                if d["role"] not in ("cavity", "core", "spot"):
                    continue
                d["truncated"] = d["truncated"] or p["truncated"]
                d["touches_exclusion"] = d["touches_exclusion"] or p["touches_exclusion"]
                stack.extend(self.kids[d["node_id"]])

    def finish(self):
        for n in self.by_id.values():
            if n["role"] is None:  # defensive: keep, never drop silently (4.4)
                self.set_role(n, "unresolved", "C-UNRESOLVED")
                self.log.append(f"node {n['node_id']} was not reached by any rule")
            if n["role"] != "particle":
                n["aggregate_id"] = None
            n["children"] = sorted(self.kids[n["node_id"]])
        self.edge_flags()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_resolve(graph_result, gray, content_mask, modality, nm_per_px, cfg):
    """Resolve one panel (spec 8). Returns a :class:`ResolveResult`.

    ``gray`` and ``content_mask`` are full-panel arrays (``content_mask`` =
    content box minus the Step-1 exclusion, spec 5.6). ``nm_per_px`` is
    ``None``/NaN when uncalibrated; it only sets ``d_min`` (8.1 step 5).
    """
    rcfg = cfg["resolve"]
    adjacent_px = float(cfg["graph"]["adjacent_px"])
    gray = np.asarray(gray, dtype=np.float32)
    content_mask = np.asarray(content_mask, dtype=bool)
    if modality not in rcfg["polarity_prior"]:
        raise ValueError(f"modality must be one of {list(rcfg['polarity_prior'])}, got {modality!r}")

    # v3 F: a frame-filling particle is not background. bg_candidate masks are
    # never parents (7.3), so the graph is rebuilt to let it parent its pore.
    frame_ids = frame_particles(graph_result.nodes, gray, content_mask, modality, rcfg)
    if frame_ids:
        graph_result = _regraph_without_bg_flag(graph_result, set(frame_ids), cfg)

    nodes = [dict(n) for n in graph_result.nodes]
    for n in nodes:
        n["frame_particle"] = n["node_id"] in frame_ids
        if not n["is_representative"]:
            n["role"] = "duplicate"
            n["rule_id"] = "G-DUP"
    reps = [n for n in nodes if n["is_representative"]]

    stats = panel_statistics(reps, gray, content_mask, modality, nm_per_px, rcfg)
    if frame_ids:
        stats["flags"].append("FRAME_PARTICLE")
    log = []
    if reps:
        node_values(reps, gray, content_mask, stats, rcfg, adjacent_px)
        resolver = _Resolver(reps, gray, stats, rcfg)
        resolver.run()
        resolver.finish()
        log = resolver.log

    counts = {role: 0 for role in ROLES}
    for n in nodes:
        counts[n["role"]] += 1
    stats["role_counts"] = counts
    stats["n_truncated"] = sum(n.get("truncated", False) for n in reps if n["role"] == "particle")
    stats["n_touches_exclusion"] = sum(
        n["touches_exclusion"] for n in reps if n["role"] == "particle"
    )
    return ResolveResult(nodes=nodes, panel=stats, log=log)


_TABLE_COLUMNS = (
    "node_id", "role", "rule_id", "rule_trace", "subtype", "parent", "graph_parent",
    "aggregate_id", "split_from", "n_spots", "truncated", "touches_exclusion",
    "bg_candidate", "frame_particle", "touches_border", "area", "eqdiam", "circularity", "solidity",
    "quality", "edge_contrast", "rho_in", "rho_rim", "pl", "ring_free_frac",
    "ring_rho", "ring_std", "contact", "r", "delta", "rho_shell", "coverage",
)


def roles_table(result):
    """One row per node with its role, rule and the 8.2/8.4 values (for review)."""
    rows = []
    for n in result.nodes:
        row = {col: n.get(col) for col in _TABLE_COLUMNS}
        if n["role"] == "duplicate":
            row["parent"] = n.get("representative_id")
        row["rule_trace"] = ">".join(n.get("rule_trace") or [n["rule_id"]])
        rows.append(row)
    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)
