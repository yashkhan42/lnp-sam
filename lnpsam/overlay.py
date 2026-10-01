"""Overlays.

The full resolved-particle overlay (spec 10.3) is part of Step 6 and is not
implemented yet. :func:`cleanup_overlay` renders Step 1,
:func:`proposal_overlay` draws every surviving Step-2 mask for proposal review,
and :func:`resolve_overlay` draws the Step-4 roles for review.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


def cleanup_overlay(rgb_u8, result):
    """Render a stage-1 overlay: exclusion (red, 30%), content box, calibration.

    ``result`` is a :class:`lnpsam.cleanup.CleanupResult`. Returns a uint8 RGB image
    in full-panel coordinates.
    """
    img = rgb_u8.copy()
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    # Exclusion mask, red at 30% (spec 10.3).
    excl = result.exclusion
    if excl.any():
        red = np.zeros_like(img)
        red[..., 0] = 255
        alpha = 0.30
        img[excl] = (img[excl] * (1 - alpha) + red[excl] * alpha).astype(np.uint8)

    # Content box outline (green).
    x0, y0, x1, y1 = result.content_box
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 0), 1)

    # Calibration text in the top-left corner.
    if result.cal_source == "missing":
        txt = "cal: missing (px)"
    else:
        txt = f"cal: {result.nm_per_px:.4g} nm/px ({result.cal_source})"
    cv2.putText(
        img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA
    )
    return img


def _panel_seed(panel_id):
    """Stable random seed so proposal colors survive cached reruns unchanged."""
    digest = hashlib.sha256(str(panel_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def proposal_overlay(rgb_u8, nodes, panel_id, line_width=1):
    """Draw every surviving Step-2 mask outline in a random RGB color.

    Colors are random but deterministically seeded by ``panel_id`` so visual
    comparisons are stable across cached and uncached runs. The original,
    unfilled node mask is outlined in full-panel coordinates.
    """
    img = np.asarray(rgb_u8).copy()
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"rgb_u8 must have shape (H, W, 3), got {img.shape}")
    if line_width < 1:
        raise ValueError("line_width must be at least 1")

    rng = np.random.default_rng(_panel_seed(panel_id))
    for node in sorted(nodes, key=lambda item: item["node_id"]):
        mask = np.asarray(node["mask"], dtype=np.uint8)
        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"node {node['node_id']} mask shape {mask.shape} does not match panel {img.shape[:2]}"
            )
        if not mask.any():
            continue
        # Keep colors away from black so outlines remain visible on dark TEM/SEM
        # regions. OpenCV writes these values directly into our RGB array.
        color = tuple(int(value) for value in rng.integers(64, 256, size=3))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color, int(line_width), lineType=cv2.LINE_8)
    return img


# Spec 10.3 colours (RGB) and line style. Roles not listed there get neutral
# review colours. Orphans are dashed magenta (a cavity without its particle) so
# they cannot be mistaken for cavities.
_ROLE_STYLE = {
    "particle": ((0, 220, 0), 1, False),
    "cavity": ((255, 0, 255), 1, False),
    "core": ((255, 160, 0), 1, False),
    "gap": ((60, 120, 255), 1, False),
    "speck": ((150, 150, 150), 1, False),
    "orphan": ((255, 0, 255), 1, True),
    "rejected": ((255, 40, 40), 1, True),
    "aggregate": ((255, 255, 255), 3, False),
    "spot": ((255, 255, 0), 1, False),
    "unresolved": ((255, 120, 120), 1, False),
    "merged": ((0, 200, 200), 1, False),
    "bg_patch": ((0, 110, 110), 1, False),
}
_DRAW_ORDER = (
    "bg_patch", "merged", "rejected", "speck", "gap", "unresolved", "aggregate",
    "orphan", "particle", "core", "cavity", "spot",
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _dashed_contours(img, contours, color, width=1, dash=4):
    for contour in contours:
        points = contour[:, 0, :]
        for i, (x, y) in enumerate(points):
            if (i // dash) % 2 == 0:
                cv2.circle(img, (int(x), int(y)), max(0, width // 2), color, -1)


def _legend(width, scale):
    """Colour key strip, wrapped to the image width."""
    items = [(role, *_ROLE_STYLE[role]) for role in reversed(_DRAW_ORDER)]
    font, row_h, swatch = 0.35 * scale, int(14 * scale), int(14 * scale)
    rows, row, x = [], [], 4
    for role, color, line, dashed in items:
        w = swatch + 4 + cv2.getTextSize(role, _FONT, font, 1)[0][0] + 10
        if row and x + w > width:
            rows.append(row)
            row, x = [], 4
        row.append((x, role, color, line, dashed))
        x += w
    rows.append(row)
    strip = np.full((row_h * len(rows) + 4, width, 3), 30, dtype=np.uint8)
    for r, entries in enumerate(rows):
        y = r * row_h + row_h // 2 + 2
        half = max(2, int(2 * scale))  # thick bars stay coloured when the image is scaled down
        for x, role, color, line, dashed in entries:
            step = max(4, int(4 * scale)) if dashed else swatch
            for xx in range(x, x + swatch, 2 * step if dashed else swatch):
                cv2.rectangle(strip, (xx, y - half), (min(xx + step, x + swatch), y + half), color, -1)
            cv2.putText(strip, role, (x + swatch + 4, y + int(4 * scale)), _FONT, font,
                        (230, 230, 230), 1, cv2.LINE_AA)
    return strip


def resolve_overlay(rgb_u8, nodes, exclusion, panel, label_ids=False, scale=1):
    """Step-4 review overlay: filled-mask outlines coloured by role (spec 10.3).

    Particles green (truncated ones marked "T"), cavities magenta, orphans
    dashed magenta, spots yellow, gaps blue, specks gray, rejected red dashed,
    aggregates thick white, exclusion red at 30%; duplicates and ``background``
    are not drawn. A colour key is appended below the image. With
    ``label_ids`` each outline is tagged with its ``node_id`` (the row in
    ``roles.csv``) and the image is upscaled by ``scale`` for legibility.
    """
    img = np.asarray(rgb_u8).copy()
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    excl = np.asarray(exclusion, dtype=bool)
    if excl.any():
        img[excl] = (img[excl] * 0.7 + np.array([255, 0, 0]) * 0.3).astype(np.uint8)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    by_role = {}
    for node in nodes:
        by_role.setdefault(node["role"], []).append(node)
    labels = []
    for role in _DRAW_ORDER:
        color, width, dashed = _ROLE_STYLE[role]
        width = width * scale if role == "aggregate" else max(1, scale // 2)
        for node in by_role.get(role, []):
            mask = np.asarray(node["filled"], dtype=np.uint8)
            if scale != 1:
                mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if dashed:
                _dashed_contours(img, contours, color, width)
            else:
                cv2.drawContours(img, contours, -1, color, width, lineType=cv2.LINE_8)
            if role == "particle" and node.get("truncated"):
                cy, cx = (v * scale for v in node["centroid"])
                cv2.putText(img, "T", (int(cx) - 3, int(cy) + 4), _FONT, 0.35 * scale,
                            color, 1, cv2.LINE_AA)
            if label_ids and contours:
                top = min(contours, key=lambda c: c[:, 0, 1].min())[:, 0, :]
                x, y = top[np.argmin(top[:, 1])]
                labels.append((int(x), int(y), str(node["node_id"]), color))

    # Node-id tags at the top of each outline; colliding tags shift right.
    placed = []
    for x, y, text, color in sorted(labels, key=lambda item: (item[1], item[0])):
        (w, h), _ = cv2.getTextSize(text, _FONT, 0.45, 1)
        y = max(h + 2, y - 3)
        while any(abs(x - px) < max(w, pw) + 3 and abs(y - py) < h + 4 for px, py, pw in placed):
            x += w + 4
        placed.append((x, y, w))
        cv2.rectangle(img, (x - 1, y - h - 2), (x + w + 1, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), _FONT, 0.45, color, 1, cv2.LINE_AA)

    nm = panel.get("nm_per_px")
    cal = f"{nm:.4g} nm/px" if nm else "px"
    counts = panel["role_counts"]
    txt = (f"{cal} P{counts['particle']} A{counts['aggregate']} C{counts['cavity']} "
           f"G{counts['gap']} O{counts['orphan']} R{counts['rejected']}")
    cv2.putText(img, txt, (4, int(12 * scale)), _FONT, 0.35 * scale, (255, 255, 0), 1, cv2.LINE_AA)
    return np.vstack([img, _legend(img.shape[1], scale)])


def overlay(*args, **kwargs):
    """Full particle overlay (spec 10.3) -- NOT YET IMPLEMENTED (stage 6)."""
    raise NotImplementedError(
        "The particle overlay (spec 10.3) is not implemented yet. "
        "Use cleanup_overlay for stage 1."
    )
