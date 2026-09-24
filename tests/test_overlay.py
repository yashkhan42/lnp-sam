"""Tests for cleanup and proposal overlays."""

import numpy as np

from lnpsam import overlay


def test_proposal_overlay_draws_each_mask_outline_in_stable_random_color():
    image = np.full((30, 40, 3), 128, dtype=np.uint8)
    first = np.zeros((30, 40), dtype=bool)
    second = np.zeros((30, 40), dtype=bool)
    first[4:14, 5:15] = True
    second[16:26, 24:35] = True
    nodes = [
        {"node_id": 8, "mask": second},
        {"node_id": 2, "mask": first},
    ]

    drawn = overlay.proposal_overlay(image, nodes, "golden_panel")
    repeated = overlay.proposal_overlay(image, list(reversed(nodes)), "golden_panel")

    assert np.array_equal(drawn, repeated)
    assert not np.array_equal(drawn[4, 5], image[4, 5])
    assert not np.array_equal(drawn[16, 24], image[16, 24])
    assert not np.array_equal(drawn[4, 5], drawn[16, 24])
    assert np.array_equal(drawn[9, 10], image[9, 10])  # outline only; interior unchanged
    assert np.array_equal(image, np.full_like(image, 128))  # input is not mutated
