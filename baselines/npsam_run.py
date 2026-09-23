"""NP-SAM baseline runner (spec 12.3) -- NOT YET IMPLEMENTED (evaluation stage).

Runs in a separate environment with ``npsam==3.2.3`` (its pinned dependencies
conflict with LNP-SAM's, spec 4.1). For each panel::

    img = NPSAMImage(path)
    img.segment(SAM_model="base", PPS=64)   # all other defaults
    # every mask in img.seg.data counts as a predicted particle

Feeds the same evaluator as LNP-SAM (eval/evaluate.py).
"""

from __future__ import annotations


def main(*args, **kwargs):
    raise NotImplementedError(
        "The NP-SAM baseline runner (spec 12.3) is not implemented yet. "
        "It belongs to the evaluation stage and runs in the separate npsam environment."
    )


if __name__ == "__main__":
    main()
