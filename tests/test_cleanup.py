"""Tests for lnpsam.cleanup (spec 5, 11)."""

import cv2
import numpy as np
import pytest

from lnpsam import cleanup as cl
from lnpsam import io as lnp_io

CFG = lnp_io.load_config("config.yaml")
CC = CFG["cleanup"]


# --------------------------------------------------------------------------- #
# 5.1 content box
# --------------------------------------------------------------------------- #
def test_content_box_white_margins():
    """Spec 11: 10 px white margins -> content_box within 1 px."""
    H = W = 100
    gray = np.full((H, W), 1.0, np.float32)  # white
    gray[10:90, 10:90] = 0.5  # content: mid-gray block
    x0, y0, x1, y1 = cl.content_box(gray, CC)
    assert abs(x0 - 10) <= 1
    assert abs(y0 - 10) <= 1
    assert abs(x1 - 90) <= 1
    assert abs(y1 - 90) <= 1


def test_content_box_no_margin():
    """A panel with no uniform border keeps the full frame."""
    rng = np.random.default_rng(0)
    gray = (0.4 + 0.2 * rng.random((80, 120))).astype(np.float32)
    assert cl.content_box(gray, CC) == (0, 0, 120, 80)


# --------------------------------------------------------------------------- #
# 5.3 scale bar (no OCR)
# --------------------------------------------------------------------------- #
def test_detect_bar_length():
    """A drawn dark 60 px bar is detected with L_px ~ 60."""
    H, W = 200, 300
    gray = np.full((H, W), 0.8, np.float32)
    gray[160:163, 100:160] = 0.0  # 60 px long, 3 px thick
    bars = cl.detect_bars(gray, CC)
    assert len(bars) == 1
    bar = bars[0]
    assert bar["polarity"] == "dark"
    assert abs(bar["L_px"] - 60) <= 1
    assert bar["y0"] == 160 and bar["y1"] == 163


def test_bar_rejects_thick_block():
    """A large solid block (low aspect) is not a bar."""
    H, W = 200, 300
    gray = np.full((H, W), 0.8, np.float32)
    gray[50:120, 100:160] = 0.0  # 60 wide x 70 tall block
    assert cl.detect_bars(gray, CC) == []


def test_bar_min_thick_rejects_thin_streak():
    """A 1 px-thick dark streak (rim-streak-like) is dropped by bar_min_thick=3."""
    H, W = 200, 300
    gray = np.full((H, W), 0.8, np.float32)
    gray[80:81, 100:160] = 0.0  # 60 px long, 1 px thick
    assert CC["bar_min_thick"] == 3
    assert cl.detect_bars(gray, CC) == []
    # A 3 px-thick bar of the same length is still detected.
    gray2 = np.full((H, W), 0.8, np.float32)
    gray2[80:83, 100:160] = 0.0
    assert len(cl.detect_bars(gray2, CC)) == 1


# --------------------------------------------------------------------------- #
# 5.5 text normalisation / scale parsing (no OCR)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("200 nm", 200.0),
        ("2OO nm", 200.0),  # O -> 0
        ("1OO nm", 100.0),
        ("1.5 um", 1500.0),
        ("500 µm", 500000.0),
        ("300pm", 300000.0),  # OCR misreads µm as pm -> 1000
        ("l00 nm", 100.0),  # l -> 1
    ],
)
def test_parse_scale_label(raw, expected):
    assert cl.parse_scale_label(raw) == expected


def test_parse_scale_label_rejects_nonscale():
    assert cl.parse_scale_label("W-SE/OL") is None
    assert cl.parse_scale_label("figure b") is None


# --------------------------------------------------------------------------- #
# 5.5 manual calibration override (no OCR)
# --------------------------------------------------------------------------- #
def test_manual_calibration_override():
    nm, src, L, val, txt, bar, box = cl.calibrate([], [], CC, manifest_nm_per_px=2.5)
    assert src == "manual" and nm == 2.5


def test_missing_calibration():
    nm, src, L, val, txt, bar, box = cl.calibrate([], [], CC, manifest_nm_per_px=float("nan"))
    assert src == "missing" and nm is None


# --------------------------------------------------------------------------- #
# 5.5 calibration is robust to how EasyOCR groups words (no OCR call)
# --------------------------------------------------------------------------- #
def _ocr_box(text, box, conf=0.9):
    x0, y0, x1, y1 = box
    return {
        "text": text,
        "text_norm": cl.normalize_scale_text(text),
        "conf": conf,
        "box": (x0, y0, x1, y1),
        "height": y1 - y0,
        "keep": True,
    }


def _fake_bar(x0, x1, y0, y1, L_px):
    return {
        "polarity": "dark",
        "y0": y0,
        "y1": y1,
        "x0": x0,
        "x1": x1,
        "x_center": 0.5 * (x0 + x1),
        "L_px": float(L_px),
        "longest": x1 - x0,
        "rows": [],
        "spans": [],
    }


def test_calibrate_merges_split_ocr_boxes():
    """'200' and 'nm' as separate side-by-side boxes still calibrate (auto)."""
    boxes = [_ocr_box("200", (100, 180, 130, 200)), _ocr_box("nm", (135, 182, 160, 200))]
    bar = _fake_bar(100, 160, 205, 208, 60)
    nm, src, L, val, txt, cbar, box = cl.calibrate(boxes, [bar], CC)
    assert src == "auto"
    assert abs(nm - 200.0 / 60.0) / (200.0 / 60.0) <= 0.02
    assert val == 200.0
    assert box == (100, 180, 160, 200)  # merged union box (used for exclusion)


def test_calibrate_does_not_merge_distant_or_multiline_boxes():
    bar = _fake_bar(100, 160, 205, 208, 60)
    far = [_ocr_box("200", (100, 180, 130, 200)), _ocr_box("nm", (300, 182, 325, 200))]
    assert cl.calibrate(far, [bar], CC)[1] == "missing"
    multiline = [_ocr_box("200", (100, 180, 130, 200)), _ocr_box("nm", (100, 240, 130, 258))]
    assert cl.calibrate(multiline, [bar], CC)[1] == "missing"


def test_calibrate_prefers_single_box_when_both_valid():
    """A single '200nm' box wins over any merged candidate."""
    boxes = [
        _ocr_box("200nm", (100, 180, 165, 200)),
        _ocr_box("nm", (168, 182, 190, 200)),
    ]
    bar = _fake_bar(100, 165, 205, 208, 65)
    nm, src, L, val, txt, cbar, box = cl.calibrate(boxes, [bar], CC)
    assert src == "auto"
    assert box == (100, 180, 165, 200)  # the single box, not a merge


# --------------------------------------------------------------------------- #
# 5.2 + 5.3 + 5.5 full calibration (needs easyocr)
# --------------------------------------------------------------------------- #
def _reader_or_skip():
    try:
        return cl.get_reader()
    except Exception as e:  # model download / init failure
        pytest.skip(f"easyocr unavailable: {e}")


def test_scale_bar_and_text_calibration():
    """Spec 11: drawn 60 px bar + "200 nm" -> nm_per_px within 2% of 3.33;
    bar and text land inside the exclusion mask."""
    reader = _reader_or_skip()

    H, W = 240, 360
    img = np.full((H, W), 204, np.uint8)  # light-gray background (~0.8)
    # Scale bar: dark, 60 px long, 3 px thick.
    img[196:199, 150:210] = 0
    # Scale text "200 nm" above the bar, roughly centred over it.
    cv2.putText(img, "200 nm", (150, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

    gray = (img.astype(np.float32)) / 255.0
    rgb_u8 = np.repeat(img[:, :, None], 3, axis=2)

    res = cl.run_cleanup(gray, rgb_u8, "TEM", CFG, reader=reader)

    assert res.cal_source == "auto", f"cal failed; ocr={[b['text'] for b in res.ocr_boxes]}"
    assert abs(res.nm_per_px - (200.0 / 60.0)) / (200.0 / 60.0) <= 0.02

    # Bar pixels are inside the exclusion mask.
    assert res.exclusion[196:199, 150:210].all()

    # The scale-label box is inside the exclusion mask. Use res.scale_box (the
    # merged union box) so this holds whether EasyOCR returns "200 nm" as one box
    # or splits it into "200" and "nm".
    assert res.scale_box is not None
    lx0, ly0, lx1, ly1 = res.scale_box
    assert res.exclusion[ly0:ly1, lx0:lx1].mean() > 0.5
