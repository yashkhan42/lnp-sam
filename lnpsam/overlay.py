"""Overlays.

The full particle overlay (spec 10.3) is part of Step 6 and is not implemented
yet. :func:`cleanup_overlay` renders what Step 1 produced -- exclusion mask in red
at 30%, the content box, and the calibration text -- for verifying stage 1.
"""

from __future__ import annotations

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


def overlay(*args, **kwargs):
    """Full particle overlay (spec 10.3) -- NOT YET IMPLEMENTED (stage 6)."""
    raise NotImplementedError(
        "The particle overlay (spec 10.3) is not implemented yet. "
        "Use cleanup_overlay for stage 1."
    )
