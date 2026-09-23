"""I/O helpers: config, manifest, image loading, and COCO-RLE mask helpers.

Conventions (spec 4.4):
  - Image: ``gray`` is float32 in [0, 1] (0.299R + 0.587G + 0.114B; 16-bit / 65535;
    alpha composited on white). ``rgb_u8`` is uint8 RGB for SAM (gray replicated if needed).
  - Coordinates: arrays ``[row, col]``; points ``(x, y)``; boxes ``(x0, y0, x1, y1)``
    end-exclusive. Everything written to disk is in full-panel coordinates.
  - Masks: COCO RLE via ``pycocotools.mask.encode(np.asfortranarray(m.astype(np.uint8)))``,
    stored as JSON strings. Never store particles as a single label image: masks overlap.
  - Filled mask: ``scipy.ndimage.binary_fill_holes(mask)``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage

# Manifest columns (spec 4.3). ``nm_per_px`` is optional.
MANIFEST_REQUIRED = ("panel_id", "paper_id", "image_path", "modality")
VALID_MODALITIES = ("TEM", "SEM")

# 16-bit single-channel PIL modes.
_MODES_16BIT = ("I", "I;16", "I;16B", "I;16L", "I;16N")


# --------------------------------------------------------------------------- #
# Config / manifest
# --------------------------------------------------------------------------- #
def load_config(path):
    """Load the YAML config (spec 4.5) into a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def paper_id_of(panel_id):
    """``paper_id`` = the part of ``panel_id`` before the first ``__`` (spec 4.3)."""
    return str(panel_id).split("__", 1)[0]


def load_manifest(path):
    """Load and validate ``data/manifest.csv`` (spec 4.3).

    Ensures required columns exist, modality is TEM/SEM, and an ``nm_per_px``
    column is present (NaN where absent).
    """
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} is missing required columns: {missing}")
    bad_modality = sorted(set(df["modality"]) - set(VALID_MODALITIES))
    if bad_modality:
        raise ValueError(
            f"manifest {path} has invalid modality values {bad_modality}; "
            f"expected one of {VALID_MODALITIES}"
        )
    if "nm_per_px" not in df.columns:
        df["nm_per_px"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_image(path):
    """Load an image and return ``(gray, rgb_u8)`` per the spec 4.4 conventions.

    ``gray``  : float32 HxW in [0, 1], luma = 0.299R + 0.587G + 0.114B.
    ``rgb_u8``: uint8 HxWx3 RGB (gray replicated for single-channel sources).

    16-bit sources are scaled by 1/65535; alpha is composited onto white.
    """
    img = Image.open(path)
    mode = img.mode

    if mode in _MODES_16BIT:
        g = np.asarray(img).astype(np.float32) / 65535.0
        g = np.clip(g, 0.0, 1.0)
        rgb_f = np.repeat(g[:, :, None], 3, axis=2)
    elif mode == "RGBA" or mode == "LA" or (mode == "P" and "transparency" in img.info):
        rgba = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
        rgb = rgba[:, :, :3]
        a = rgba[:, :, 3:4]
        rgb_f = rgb * a + (1.0 - a)  # composite on white (background = 1.0)
    else:
        rgb_f = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0

    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    gray = (
        0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    ).astype(np.float32)
    rgb_u8 = np.round(rgb_f * 255.0).astype(np.uint8)
    return gray, rgb_u8


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #
def encode_rle(mask):
    """Encode a 2-D boolean/int mask as a COCO-RLE JSON string (spec 4.4)."""
    m = np.asfortranarray(np.asarray(mask).astype(np.uint8))
    rle = mask_utils.encode(m)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    size = [int(rle["size"][0]), int(rle["size"][1])]
    return json.dumps({"size": size, "counts": counts})


def decode_rle(s):
    """Decode a COCO-RLE JSON string (from :func:`encode_rle`) to a bool mask."""
    d = json.loads(s)
    rle = {
        "size": [int(d["size"][0]), int(d["size"][1])],
        "counts": d["counts"].encode("ascii"),
    }
    return mask_utils.decode(rle).astype(bool)


def fill_holes(mask):
    """Filled mask (spec 4.4): ``scipy.ndimage.binary_fill_holes``."""
    return ndimage.binary_fill_holes(np.asarray(mask, dtype=bool))
