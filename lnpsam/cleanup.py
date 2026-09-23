"""Step 1: Cleanup (spec 5).

Remove figure furniture (margins, text, scale bars, label boxes) and read the
scale, producing the cropped, inpainted image that SAM will see.

Nothing here drops a candidate particle: this step only defines the content box,
the exclusion mask and the calibration. All outputs are in full-panel coordinates
except ``sam_image`` (the content-box crop) and ``crop_offset`` (its origin).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from skimage import measure

# --------------------------------------------------------------------------- #
# Calibration text normalisation / parsing (spec 5.5)
# --------------------------------------------------------------------------- #
# Confusable -> digit map, applied only inside runs that contain a real digit.
_CONFUSABLE_MAP = str.maketrans({"O": "0", "l": "1", "I": "1", ",": "."})
_NUM_RUN_RE = re.compile(r"[0-9OlI.,]+|[^0-9OlI.,]+")
# Scale label regex (case-insensitive), spec 5.5 step 2.
_SCALE_RE = re.compile(r"^(\d+(?:\.\d+)?)(nm|um|µm|μm|jm|pm)$", re.IGNORECASE)
# Units -> nm. OCR usually misreads µm as pm, so pm is treated as 1000 too.
_UNIT_TO_NM = {"nm": 1.0, "um": 1000.0, "µm": 1000.0, "μm": 1000.0, "jm": 1000.0, "pm": 1000.0}

# Lazy per-process easyocr reader (spec 5.2: create the reader once per process).
_READER = None


def get_reader():
    """Return a process-wide ``easyocr.Reader(["en"])`` (created lazily)."""
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"])
    return _READER


@dataclass
class CleanupResult:
    content_box: tuple  # (x0, y0, x1, y1), end-exclusive, full-panel coords
    exclusion: np.ndarray  # full-panel bool HxW: text / bars / label boxes
    content_mask: np.ndarray  # full-panel bool HxW: content_box & ~exclusion
    sam_image: np.ndarray  # cropped, inpainted uint8 RGB (the SAM input)
    crop_offset: tuple  # (x0, y0) origin of sam_image in full-panel coords
    nm_per_px: Optional[float]
    cal_source: str  # "manual" | "auto" | "missing"
    L_px: Optional[float] = None
    scale_value_nm: Optional[float] = None
    scale_text: Optional[str] = None
    ocr_boxes: list = field(default_factory=list)
    bars: list = field(default_factory=list)
    label_boxes: list = field(default_factory=list)
    annotation_boxes: list = field(default_factory=list)
    burned_bars: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 5.1 Content box
# --------------------------------------------------------------------------- #
def _is_margin(line, std_max, white, black):
    m = float(line.mean())
    s = float(line.std())
    return s < std_max and (m > white or m < black)


def content_box(gray, cc):
    """Spec 5.1: scan each side inward (<= ``margin_scan_max_frac`` of the dim);
    a line is margin if ``std < margin_std_max`` and (``mean > margin_white`` or
    ``mean < margin_black``). Returns ``(x0, y0, x1, y1)`` end-exclusive."""
    H, W = gray.shape
    std_max = cc["margin_std_max"]
    white = cc["margin_white"]
    black = cc["margin_black"]
    smax_h = int(cc["margin_scan_max_frac"] * H)
    smax_w = int(cc["margin_scan_max_frac"] * W)

    top = 0
    while top < smax_h and _is_margin(gray[top, :], std_max, white, black):
        top += 1
    bottom = H
    while bottom > H - smax_h and _is_margin(gray[bottom - 1, :], std_max, white, black):
        bottom -= 1
    left = 0
    while left < smax_w and _is_margin(gray[:, left], std_max, white, black):
        left += 1
    right = W
    while right > W - smax_w and _is_margin(gray[:, right - 1], std_max, white, black):
        right -= 1

    # Degenerate guard: never collapse the content box entirely.
    if right <= left or bottom <= top:
        return (0, 0, W, H)
    return (int(left), int(top), int(right), int(bottom))


# --------------------------------------------------------------------------- #
# 5.5 helpers: text normalisation
# --------------------------------------------------------------------------- #
def normalize_scale_text(text):
    """Spec 5.5 step 1: strip spaces; inside digit runs map O->0, l->1, I->1, ,->."."""
    t = text.replace(" ", "")
    out = []
    for run in _NUM_RUN_RE.findall(t):
        if any(ch.isdigit() for ch in run):
            run = run.translate(_CONFUSABLE_MAP)
        out.append(run)
    return "".join(out)


def parse_scale_label(text):
    """Return ``value_nm`` if ``text`` matches the scale regex, else ``None`` (spec 5.5)."""
    norm = normalize_scale_text(text)
    m = _SCALE_RE.match(norm)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value * _UNIT_TO_NM[unit]


# --------------------------------------------------------------------------- #
# 5.2 Text (OCR)
# --------------------------------------------------------------------------- #
def detect_text(gray, cc, reader):
    """Spec 5.2: OCR the upscaled panel; return boxes in full-panel coords.

    Each entry: ``{text, text_norm, conf, box (x0,y0,x1,y1), height, keep}`` where
    ``keep`` marks boxes eligible for text exclusion (conf and height filters).
    """
    H, W = gray.shape
    up = 3 if min(H, W) < 300 else int(cc["ocr_upscale"])
    img_up = cv2.resize(gray, (W * up, H * up), interpolation=cv2.INTER_CUBIC)
    img_u8 = np.clip(np.round(img_up * 255.0), 0, 255).astype(np.uint8)

    results = reader.readtext(img_u8, detail=1, paragraph=False)

    conf_min = cc["ocr_conf_min"]
    min_h = cc["text_h_frac"][0] * H
    max_h = cc["text_h_frac"][1] * H

    boxes = []
    for poly, text, conf in results:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0 = int(max(0, math.floor(min(xs) / up)))
        y0 = int(max(0, math.floor(min(ys) / up)))
        x1 = int(min(W, math.ceil(max(xs) / up)))
        y1 = int(min(H, math.ceil(max(ys) / up)))
        if x1 <= x0 or y1 <= y0:
            continue
        height = y1 - y0
        keep = (conf >= conf_min) and (min_h <= height <= max_h)
        boxes.append(
            {
                "text": text,
                "text_norm": normalize_scale_text(text),
                "conf": float(conf),
                "box": (x0, y0, x1, y1),
                "height": height,
            }
        )
        boxes[-1]["keep"] = keep
    return boxes


def _add_box_dilated(excl, box, d, H, W):
    """Add an axis-aligned box, expanded by ``d`` px on each side, to ``excl``.

    Expanding a filled rectangle by a square structuring element is exactly a
    coordinate expansion, so this matches "dilated by d" for rectangles.
    """
    x0, y0, x1, y1 = box
    d = int(round(d))
    x0d = max(0, x0 - d)
    y0d = max(0, y0 - d)
    x1d = min(W, x1 + d)
    y1d = min(H, y1 + d)
    excl[y0d:y1d, x0d:x1d] = True


# --------------------------------------------------------------------------- #
# 5.3 Scale bar
# --------------------------------------------------------------------------- #
def _longest_run(row_bool):
    """Longest run of True in a 1-D bool array; returns ``(length, start, end)``
    with ``end`` exclusive. ``(0, 0, 0)`` if the row is empty."""
    idx = np.flatnonzero(row_bool)
    if idx.size == 0:
        return 0, 0, 0
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [idx.size - 1]))
    best_len, best_s, best_e = 0, 0, 0
    for s, e in zip(starts, ends):
        length = int(idx[e] - idx[s] + 1)
        if length > best_len:
            best_len, best_s, best_e = length, int(idx[s]), int(idx[e] + 1)
    return best_len, best_s, best_e


def _span_overlap_frac(a, b):
    """Overlap fraction of two [start, end) spans = intersection / min(len)."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    denom = min(a[1] - a[0], b[1] - b[0])
    return inter / denom if denom > 0 else 0.0


def detect_bars(gray, cc):
    """Spec 5.3: detect drawn scale bars in either polarity. Returns bar dicts
    with ``y0,y1,x0,x1`` (full-panel, end-exclusive), ``L_px`` and ``polarity``."""
    H, W = gray.shape
    min_len = cc["bar_min_len_frac"] * W
    max_thick = max(3, cc["bar_max_thick_frac"] * H)
    min_thick = cc.get("bar_min_thick", 0)  # LNP-SAM tuning: drop thin rim streaks
    min_aspect = cc["bar_min_aspect"]
    flat_std_max = cc["bar_flat_std_max"]

    bars = []
    for polarity, binary in (
        ("bright", gray >= cc["bar_bright"]),
        ("dark", gray <= cc["bar_dark"]),
    ):
        # Longest horizontal run per row.
        kept = []  # (row, start, end, length)
        for r in range(H):
            length, s, e = _longest_run(binary[r, :])
            if length >= min_len:
                kept.append((r, s, e, length))
        if not kept:
            continue

        # Group consecutive rows whose runs overlap >= 90%.
        groups = []
        cur = [kept[0]]
        for prev, row in zip(kept, kept[1:]):
            same = row[0] == prev[0] + 1 and _span_overlap_frac(
                (row[1], row[2]), (prev[1], prev[2])
            ) >= 0.90
            if same:
                cur.append(row)
            else:
                groups.append(cur)
                cur = [row]
        groups.append(cur)

        for g in groups:
            rows = [x[0] for x in g]
            spans = [(x[1], x[2]) for x in g]
            lengths = [x[3] for x in g]
            thickness = len(rows)
            longest = max(lengths)
            if thickness > max_thick or thickness < min_thick:
                continue
            if longest / thickness < min_aspect:
                continue
            # Flatness: std of gray along each row's run must be small.
            flat = all(
                float(gray[r, s:e].std()) <= flat_std_max for (r, (s, e)) in zip(rows, spans)
            )
            if not flat:
                continue
            bx0 = int(min(s for s, _ in spans))
            bx1 = int(max(e for _, e in spans))
            # L_px: bar length. 'extent' = full drawn width (bx1-bx0), which keeps
            # end serifs / anti-aliased ends a bracketed bar loses under 'median'
            # (v2). 'median' = original: median run over rows >= 0.8 * longest.
            if cc.get("bar_length_method", "extent") == "median":
                long_lengths = [ln for ln in lengths if ln >= 0.8 * longest]
                L_px = float(np.median(long_lengths))
            else:
                L_px = float(bx1 - bx0)
            x_center = 0.5 * (bx0 + bx1)
            bars.append(
                {
                    "polarity": polarity,
                    "y0": int(min(rows)),
                    "y1": int(max(rows) + 1),
                    "x0": bx0,
                    "x1": bx1,
                    "x_center": float(x_center),
                    "L_px": L_px,
                    "longest": int(longest),
                    "rows": rows,
                    "spans": spans,
                }
            )
    return bars


def _rasterize_bar(bar, excl, H, W, dilate=3):
    """Mark a bar's run pixels in ``excl``, dilated by ``dilate`` px (spec 5.3 step 4)."""
    tmp = np.zeros((H, W), dtype=np.uint8)
    for r, (s, e) in zip(bar["rows"], bar["spans"]):
        tmp[r, s:e] = 1
    if dilate > 0:
        k = 2 * int(dilate) + 1
        tmp = cv2.dilate(tmp, np.ones((k, k), np.uint8))
    excl |= tmp.astype(bool)


# --------------------------------------------------------------------------- #
# 5.4 Label boxes
# --------------------------------------------------------------------------- #
def _box_center(box):
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def detect_label_boxes(gray, cc, ocr_boxes):
    """Spec 5.4: filled bright/dark rectangles that contain an OCR box."""
    H, W = gray.shape
    panel_area = H * W
    fill_min = cc["annot_box_fill_min"]
    area_max = cc["annot_box_area_max_frac"] * panel_area

    centers = [_box_center(b["box"]) for b in ocr_boxes if b["conf"] >= cc["ocr_conf_min"]]

    found = []
    for binary in (gray >= cc["bar_bright"], gray <= cc["bar_dark"]):
        lbl = measure.label(binary, connectivity=2)
        for region in measure.regionprops(lbl):
            minr, minc, maxr, maxc = region.bbox
            bbox_area = (maxr - minr) * (maxc - minc)
            if bbox_area <= 0:
                continue
            if region.area / bbox_area < fill_min:
                continue
            if region.area > area_max:
                continue
            contains = any(minc <= cx < maxc and minr <= cy < maxr for (cx, cy) in centers)
            if not contains:
                continue
            found.append((minc, minr, maxc, maxr))  # (x0, y0, x1, y1)
    return found


# --------------------------------------------------------------------------- #
# A2 (amendment): OCR-anchored annotation boxes
# --------------------------------------------------------------------------- #
def detect_annotation_boxes(gray, cc, ocr_boxes):
    """A2: exclude burned-in label/annotation patches anchored on OCR text.

    §5.4's isolated-rectangle test fails when a label box is 8-connected to
    adjacent particles (e.g. the black "C" box touching dark spheres). Here, for
    each OCR box that sits on a locally pure bright/dark patch, we region-grow the
    patch inside a bounded window and exclude it, so connectivity to particles
    does not matter. Returns a list of ``(x0, y0, x1, y1)`` boxes.
    """
    H, W = gray.shape
    pure_bright = cc.get("annot_ocr_pure_bright", 0.90)
    pure_dark = cc.get("annot_ocr_pure_dark", 0.15)
    pad_frac = cc.get("annot_grow_pad_frac", 0.7)
    area_mult = cc.get("annot_grow_area_mult", 8)
    area_max = cc["annot_box_area_max_frac"] * H * W

    found = []
    for b in ocr_boxes:
        if b["conf"] < cc["ocr_conf_min"]:
            continue
        x0, y0, x1, y1 = b["box"]
        tw, th = x1 - x0, y1 - y0
        if tw <= 0 or th <= 0:
            continue
        med = float(np.median(gray[y0:y1, x0:x1]))
        if med >= pure_bright:
            sel = gray >= pure_bright
        elif med <= pure_dark:
            sel = gray <= pure_dark
        else:
            continue  # text sits on ordinary background, not a label box

        pad = int(max(3, pad_frac * max(tw, th)))
        wx0, wy0 = max(0, x0 - pad), max(0, y0 - pad)
        wx1, wy1 = min(W, x1 + pad), min(H, y1 + pad)
        sub = sel[wy0:wy1, wx0:wx1]
        if not sub.any():
            continue
        lbl = measure.label(sub, connectivity=2)
        # Component with the largest overlap with the (windowed) OCR box.
        tb = (y0 - wy0, x0 - wx0, y1 - wy0, x1 - wx0)
        best_id, best_ov = 0, 0
        for rid in range(1, lbl.max() + 1):
            comp = lbl == rid
            ov = comp[tb[0]:tb[2], tb[1]:tb[3]].sum()
            if ov > best_ov:
                best_ov, best_id = ov, rid
        if best_id == 0:
            continue
        comp = lbl == best_id
        area = int(comp.sum())
        if area > area_mult * tw * th or area > area_max:
            # Grow ran away (grabbed particles); fall back to the padded OCR box.
            found.append((wx0, wy0, wx1, wy1))
            continue
        ys, xs = np.where(comp)
        found.append(
            (wx0 + int(xs.min()), wy0 + int(ys.min()), wx0 + int(xs.max()) + 1, wy0 + int(ys.max()) + 1)
        )
    return found


# --------------------------------------------------------------------------- #
# A6 (amendment): OCR-independent geometric label boxes
# --------------------------------------------------------------------------- #
def detect_geometric_label_boxes(gray, cc):
    """A6: filled pure bright/dark rectangles near an image edge, detected without
    OCR. §5.4 and A2 both need an OCR box to anchor a label box; when the panel
    letter's OCR confidence falls below ``ocr_conf_min`` the box leaks as a flat
    block SAM would segment (e.g. wang2019 panel02's white "c" box). This catches
    it by shape: a solid rectangle (fill ≥ ``annot_box_fill_min``) of bounded area
    within ``burn_bar_edge_frac`` of an edge. Round/irregular particles have low
    bbox fill and are rejected. Returns ``(x0, y0, x1, y1)`` boxes."""
    H, W = gray.shape
    panel = H * W
    fill_min = cc["annot_box_fill_min"]
    area_max = cc["annot_box_area_max_frac"] * panel
    area_min = cc.get("annot_box_area_min_frac", 0.002) * panel
    edge = cc.get("burn_bar_edge_frac", 0.20)
    eh, ew = edge * H, edge * W

    found = []
    for sel in (gray >= cc["bar_bright"], gray <= cc["bar_dark"]):
        lbl = measure.label(sel, connectivity=2)
        for r in measure.regionprops(lbl):
            if not (area_min <= r.area <= area_max):
                continue
            minr, minc, maxr, maxc = r.bbox
            ba = (maxr - minr) * (maxc - minc)
            if ba == 0 or r.area / ba < fill_min:
                continue
            near_edge = minr <= eh or minc <= ew or (H - maxr) <= eh or (W - maxc) <= ew
            if not near_edge:
                continue
            found.append((minc, minr, maxc, maxr))
    return found


# --------------------------------------------------------------------------- #
# A3 (amendment): burned-in bars (pure, flat, edge-adjacent)
# --------------------------------------------------------------------------- #
def detect_burned_in_bars(gray, cc):
    """A3: detect burned-in scale bars/rules -- near-saturated (white or black),
    ultra-flat horizontal bars near an image edge, allowing greater thickness than
    §5.3. Distinguishes drawn furniture from imaged particle rims (mid-grey,
    textured, interior). Returns bar dicts (same shape as :func:`detect_bars`)."""
    H, W = gray.shape
    min_len = cc.get("burn_bar_min_len_frac", 0.03) * W
    max_thick = max(3, cc.get("burn_bar_max_thick_frac", 0.12) * H)
    min_thick = cc.get("burn_bar_min_thick", 3)
    std_max = cc.get("burn_bar_std_max", 0.02)
    edge = cc.get("burn_bar_edge_frac", 0.20)
    edge_h, edge_w = edge * H, edge * W
    contrast_min = cc.get("burn_bar_contrast_min", 0.20)
    pure = {"bright": cc.get("burn_bar_pure_bright", 0.95), "dark": cc.get("burn_bar_pure_dark", 0.05)}

    bars = []
    for polarity, binary in (
        ("bright", gray >= pure["bright"]),
        ("dark", gray <= pure["dark"]),
    ):
        kept = []
        for r in range(H):
            length, s, e = _longest_run(binary[r, :])
            if length >= min_len:
                kept.append((r, s, e, length))
        if not kept:
            continue
        groups = [[kept[0]]]
        for prev, row in zip(kept, kept[1:]):
            if row[0] == prev[0] + 1 and _span_overlap_frac((row[1], row[2]), (prev[1], prev[2])) >= 0.90:
                groups[-1].append(row)
            else:
                groups.append([row])
        for g in groups:
            rows = [x[0] for x in g]
            spans = [(x[1], x[2]) for x in g]
            lengths = [x[3] for x in g]
            thickness = len(rows)
            longest = max(lengths)
            if thickness > max_thick or thickness < min_thick:
                continue
            # Purity: each row's run is near-saturated and ultra-flat.
            if not all(float(gray[r, s:e].std()) <= std_max for r, (s, e) in zip(rows, spans)):
                continue
            y0, y1 = min(rows), max(rows) + 1
            x0 = min(s for s, _ in spans)
            x1 = max(e for _, e in spans)
            # Edge-adjacency: some part of the bar lies within an edge band.
            near_edge = (
                y0 <= edge_h or (H - y1) <= edge_h or x0 <= edge_w or (W - x1) <= edge_w
            )
            if not near_edge:
                continue
            # Contrast: a burned-in bar stands out from the rows just above/below
            # it. A dark particle interior has dark surroundings -> low contrast.
            band = 3
            sur = np.concatenate(
                [gray[max(0, y0 - band):y0, x0:x1].ravel(), gray[y1:min(H, y1 + band), x0:x1].ravel()]
            )
            bar_mean = float(np.mean([gray[r, s:e].mean() for r, (s, e) in zip(rows, spans)]))
            if sur.size and abs(float(sur.mean()) - bar_mean) < contrast_min:
                continue
            long_lengths = [ln for ln in lengths if ln >= 0.8 * longest]
            L_px = float(np.median(long_lengths))
            li = int(np.argmax(lengths))
            xc = 0.5 * (spans[li][0] + spans[li][1])
            bars.append(
                {
                    "polarity": polarity,
                    "y0": int(y0),
                    "y1": int(y1),
                    "x0": int(x0),
                    "x1": int(x1),
                    "x_center": float(xc),
                    "L_px": L_px,
                    "longest": int(longest),
                    "rows": rows,
                    "spans": spans,
                }
            )
    return bars


# --------------------------------------------------------------------------- #
# 5.5 Calibration
# --------------------------------------------------------------------------- #
def calibrate(ocr_boxes, bars, cc, manifest_nm_per_px=None):
    """Spec 5.5: pair a scale label with a bar and derive ``nm_per_px``.

    Priority: manifest override > auto > missing. Returns
    ``(nm_per_px, cal_source, L_px, value_nm, scale_text, cal_bar)`` where
    ``cal_bar`` is the paired bar dict (or ``None``); it is excluded in 5.6.
    """
    lo, hi = cc["nm_per_px_range"]

    # Manual override wins.
    if manifest_nm_per_px is not None and not (
        isinstance(manifest_nm_per_px, float) and math.isnan(manifest_nm_per_px)
    ):
        return float(manifest_nm_per_px), "manual", None, None, None, None

    # Candidate scale labels (OCR text matching the regex).
    labels = []
    for b in ocr_boxes:
        if b["conf"] < cc["ocr_conf_min"]:
            continue
        value_nm = parse_scale_label(b["text"])
        if value_nm is not None:
            labels.append((b, value_nm))

    best = None  # (overlap, nm_per_px, L_px, value_nm, text, bar)
    for b, value_nm in labels:
        lx0, ly0, lx1, ly1 = b["box"]
        lh = ly1 - ly0
        lcx = 0.5 * (lx0 + lx1)

        candidate_bar = None
        best_overlap = -1.0
        for bar in bars:
            # Vertical gap <= 3 label heights.
            gap = max(bar["y0"] - ly1, ly0 - bar["y1"], 0)
            if gap > 3 * lh:
                continue
            # Horizontal centre within 0.75 * L_px.
            if abs(lcx - bar["x_center"]) > 0.75 * bar["L_px"]:
                continue
            overlap = _span_overlap_frac((lx0, lx1), (bar["x0"], bar["x1"]))
            if overlap > best_overlap:
                best_overlap = overlap
                candidate_bar = bar
        if candidate_bar is None:
            continue

        nm_per_px = value_nm / candidate_bar["L_px"]
        if not (lo <= nm_per_px <= hi):
            continue
        if best is None or best_overlap > best[0]:
            best = (best_overlap, nm_per_px, candidate_bar["L_px"], value_nm, b["text"], candidate_bar)

    if best is None:
        return None, "missing", None, None, None, None
    _, nm_per_px, L_px, value_nm, text, bar = best
    return float(nm_per_px), "auto", float(L_px), float(value_nm), text, bar


# --------------------------------------------------------------------------- #
# 5.6 SAM input
# --------------------------------------------------------------------------- #
def build_sam_input(rgb_u8, content_box_xyxy, exclusion, cc):
    """Spec 5.6 + A1: crop to the content box and fill excluded pixels.

    ``fill_method='telea'`` reproduces the original per-channel TELEA inpaint.
    ``fill_method='texture'`` (default) inpaints for a locally correct mean, then
    re-injects Gaussian noise matched to the background's high-frequency amplitude
    so filled regions blend into the noise field instead of forming the smooth
    blobs SAM would otherwise segment. Returns ``(sam_image, crop_offset)``.
    """
    x0, y0, x1, y1 = content_box_xyxy
    crop = rgb_u8[y0:y1, x0:x1, :].copy()
    mask_crop = exclusion[y0:y1, x0:x1]
    if not mask_crop.any():
        return crop, (x0, y0)

    radius = int(cc["inpaint_radius"])
    mask_u8 = mask_crop.astype(np.uint8) * 255
    base = np.empty_like(crop)
    for c in range(3):
        base[:, :, c] = cv2.inpaint(crop[:, :, c], mask_u8, radius, cv2.INPAINT_TELEA)

    if cc.get("fill_method", "texture") == "telea":
        return base, (x0, y0)

    # A1: estimate background-noise sigma (robust) from the un-excluded crop.
    gray_crop = (
        0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    ).astype(np.float32) / 255.0
    valid = ~mask_crop
    sigma = 0.0
    if int(valid.sum()) > 50:
        hf = (gray_crop - cv2.GaussianBlur(gray_crop, (0, 0), 2.0))[valid]
        sigma = 1.4826 * float(np.median(np.abs(hf - np.median(hf))))
    sigma *= cc.get("fill_noise_scale", 1.0)

    out = base.astype(np.float32)
    if sigma > 0:
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, sigma * 255.0, size=mask_crop.shape).astype(np.float32)
        for c in range(3):
            out[:, :, c][mask_crop] += noise[mask_crop]
    return np.clip(out, 0, 255).astype(np.uint8), (x0, y0)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_cleanup(gray, rgb_u8, modality, cfg, reader=None, manifest_nm_per_px=None):
    """Run the full Step-1 cleanup (spec 5) and return a :class:`CleanupResult`.

    ``modality`` is carried for downstream steps; it is unused in cleanup itself.
    ``reader`` is an easyocr reader; if ``None`` a process-wide one is created.
    """
    cc = cfg["cleanup"]
    H, W = gray.shape
    if reader is None:
        reader = get_reader()

    cbox = content_box(gray, cc)

    exclusion = np.zeros((H, W), dtype=bool)

    # 5.2 Text
    ocr_boxes = detect_text(gray, cc, reader)
    for b in ocr_boxes:
        if b["keep"]:
            d = max(2, cc["text_dilate_frac"] * b["height"])
            _add_box_dilated(exclusion, b["box"], d, H, W)

    # 5.3 Scale bars: the calibration detector, plus A3 burned-in furniture bars.
    bars = detect_bars(gray, cc)
    burned_bars = detect_burned_in_bars(gray, cc)

    # 5.5 Calibration (before bar exclusion: it names the bar to exclude).
    nm_per_px, cal_source, L_px, value_nm, scale_text, cal_bar = calibrate(
        ocr_boxes, bars, cc, manifest_nm_per_px
    )

    # A3: only furniture bars enter the exclusion (burned-in + the calibration
    # bar), so imaged particle rims are not masked. 'all' restores §5.3 step 4.
    if cc.get("bar_exclude_scope", "furniture") == "all":
        exclude_bars = list(bars)
    else:
        exclude_bars = list(burned_bars)
        if cal_bar is not None:
            exclude_bars.append(cal_bar)
    for bar in exclude_bars:
        _rasterize_bar(bar, exclusion, H, W, dilate=cc.get("bar_dilate_px", 5))

    # 5.4 Label boxes (original) + A2 OCR-anchored annotation boxes.
    label_boxes = detect_label_boxes(gray, cc, ocr_boxes)
    for lb in label_boxes:
        _add_box_dilated(exclusion, lb, 2, H, W)
    annotation_boxes = detect_annotation_boxes(gray, cc, ocr_boxes)
    for ab in annotation_boxes:
        _add_box_dilated(exclusion, ab, cc.get("annot_dilate_px", 2), H, W)
    # A6: OCR-independent geometric label boxes (catches boxes OCR failed to read).
    if cc.get("geom_label_boxes", True):
        geom_boxes = detect_geometric_label_boxes(gray, cc)
        for gb in geom_boxes:
            _add_box_dilated(exclusion, gb, cc.get("annot_dilate_px", 2), H, W)
    else:
        geom_boxes = []
    annotation_boxes = list(annotation_boxes) + list(geom_boxes)

    # 5.6 SAM input (A1 texture fill)
    sam_image, crop_offset = build_sam_input(rgb_u8, cbox, exclusion, cc)

    content_mask = np.zeros((H, W), dtype=bool)
    x0, y0, x1, y1 = cbox
    content_mask[y0:y1, x0:x1] = True
    content_mask &= ~exclusion

    return CleanupResult(
        content_box=cbox,
        exclusion=exclusion,
        content_mask=content_mask,
        sam_image=sam_image,
        crop_offset=crop_offset,
        nm_per_px=nm_per_px,
        cal_source=cal_source,
        L_px=L_px,
        scale_value_nm=value_nm,
        scale_text=scale_text,
        ocr_boxes=ocr_boxes,
        bars=bars,
        label_boxes=label_boxes,
        annotation_boxes=annotation_boxes,
        burned_bars=burned_bars,
    )
