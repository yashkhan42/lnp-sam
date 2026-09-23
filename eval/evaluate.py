"""Evaluation against the golden panels (spec 12.3) -- NOT YET IMPLEMENTED.

Metrics: F1@0.5 (Hungarian matching on -IoU), count error, median diameter error,
hollow accuracy, calibration error. Compares LNP-SAM vs the NP-SAM baseline against
ground-truth label images (spec 12.2).
"""

from __future__ import annotations


def main(*args, **kwargs):
    raise NotImplementedError(
        "The evaluator (spec 12.3) is not implemented yet. It belongs to the "
        "evaluation stage, after the pipeline produces particle tables."
    )


if __name__ == "__main__":
    main()
