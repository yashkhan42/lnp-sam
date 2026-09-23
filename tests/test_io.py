"""Tests for lnpsam.io (spec 4.4, 11)."""

import numpy as np
from skimage.draw import disk

from lnpsam import io as lnp_io


def _disk_mask(shape, center, radius):
    m = np.zeros(shape, dtype=bool)
    rr, cc = disk(center, radius, shape=shape)
    m[rr, cc] = True
    return m


def test_rle_round_trip_overlapping_masks():
    """Spec 11: RLE round trip with overlapping masks returns identical masks."""
    shape = (60, 60)
    a = _disk_mask(shape, (30, 25), 15)
    b = _disk_mask(shape, (30, 35), 15)
    # The masks must actually overlap for this to be the intended test.
    assert np.logical_and(a, b).any()

    for m in (a, b):
        s = lnp_io.encode_rle(m)
        assert isinstance(s, str)
        back = lnp_io.decode_rle(s)
        assert back.shape == m.shape
        assert np.array_equal(back, m)


def test_rle_preserves_overlap_region():
    """Overlapping masks stay distinct (never merged into one label image)."""
    shape = (40, 40)
    a = _disk_mask(shape, (20, 16), 10)
    b = _disk_mask(shape, (20, 24), 10)
    sa, sb = lnp_io.encode_rle(a), lnp_io.encode_rle(b)
    assert not np.array_equal(lnp_io.decode_rle(sa), lnp_io.decode_rle(sb))


def test_fill_holes():
    """binary_fill_holes fills an interior hole."""
    m = np.zeros((30, 30), dtype=bool)
    rr, cc = disk((15, 15), 10, shape=(30, 30))
    m[rr, cc] = True
    inner_rr, inner_cc = disk((15, 15), 4, shape=(30, 30))
    m[inner_rr, inner_cc] = False  # punch a hole
    filled = lnp_io.fill_holes(m)
    assert filled[15, 15]  # hole is filled
    assert filled.sum() > m.sum()


def test_paper_id_of():
    pid = "girard2024_acssuschemeng_4c00711__p08_x658__panel06"
    assert lnp_io.paper_id_of(pid) == "girard2024_acssuschemeng_4c00711"
