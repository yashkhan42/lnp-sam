"""Overlays.

The full resolved-particle overlay (spec 10.3) is part of Step 6 and is not
implemented yet. :func:`cleanup_overlay` renders Step 1, while
:func:`proposal_overlay` draws every surviving Step-2 mask for proposal review.
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


def overlay(*args, **kwargs):
    """Full particle overlay (spec 10.3) -- NOT YET IMPLEMENTED (stage 6)."""
    raise NotImplementedError(
        "The particle overlay (spec 10.3) is not implemented yet. "
        "Use cleanup_overlay for stage 1."
    )
