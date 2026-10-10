"""Step 5: Recover orphans (spec 9) -- DEFERRED out of the MVP (spec v3).

Prompting SAM with a box around an orphan cavity can only capture the whole
particle when the cavity is at least ~1/2.2 of the particle's diameter (the
largest ``recover.box_scales``). In-scope hollow particles have small central
openings (12-36 % of the diameter), and the golden panels contain no orphans;
the only near-miss was a gap. The 9.6 fallback for unrecovered orphans runs in
Step 6 (:func:`lnpsam.measure.apply_orphan_fallback`). See
LNP-SAM_MVP_SPEC_AMENDMENTS.md section 12.
"""

from __future__ import annotations


def run_recover(*args, **kwargs):
    raise NotImplementedError(
        "Step 5 (Recover, spec 9) is deferred out of the MVP; unrecovered orphans "
        "are handled by the 9.6 fallback in Step 6 (lnpsam.measure)."
    )
