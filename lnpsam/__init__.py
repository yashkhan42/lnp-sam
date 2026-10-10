"""LNP-SAM: calibrated lignin-nanoparticle sizing from micrograph panels.

See LNP-SAM_MVP_SPEC.md for the full specification. This package implements the
six-step pipeline: cleanup -> proposals -> graph -> resolve -> recover -> measure.
Step 5 (recover) is deferred in the MVP; its orphan fallback runs in measure.
"""

__version__ = "0.1.0"
