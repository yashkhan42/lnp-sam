"""NP-SAM baseline runner (spec 12.3). Runs on the HPC, in the npsam environment.

    python baselines/npsam_run.py --manifest data/manifest.csv --out runs/npsam_base \\
        [--model base] [--pps 64] [--device cuda] [--overwrite]

``npsam==3.2.3`` has its own pinned dependencies (spec 4.1), so this file needs
only numpy at import time and imports npsam inside :func:`segment_panel`. Per
panel: ``NPSAMImage(path).segment(SAM_model=model, PPS=pps)`` with all other
defaults; every mask in ``img.seg.data`` (a ``(N, H, W)`` stack) is a predicted
particle.

Writes, losslessly and with numpy only (``eval/evaluate.py`` reads it):
``<out>/<panel_id>__masks.npz`` (the bit-packed mask stack), ``<out>/<panel_id>__meta.json``
and ``<out>/runs.csv``. Unlike a label image, the stack keeps overlapping masks whole.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np


def pack_masks(stack):
    """``(N, H, W)`` bool masks -> arrays for ``np.savez_compressed``."""
    stack = np.asarray(stack, dtype=bool)
    if stack.ndim == 2:
        stack = stack[None]
    return {"packed": np.packbits(stack, axis=-1), "shape": np.asarray(stack.shape, dtype=np.int64)}


def unpack_masks(path):
    """Read ``<panel_id>__masks.npz`` back into an ``(N, H, W)`` bool stack."""
    with np.load(path) as data:
        n, h, w = (int(v) for v in data["shape"])
        return np.unpackbits(data["packed"], axis=-1, count=w).astype(bool).reshape(n, h, w)


def segment_panel(image_path, model, pps, device=None):
    """Run NP-SAM on one panel; returns the ``(N, H, W)`` mask stack (N may be 0)."""
    import npsam  # only in the npsam environment

    img = npsam.NPSAMImage(image_path)
    kwargs = {"SAM_model": model, "PPS": pps}
    if device:
        kwargs["device"] = device
    try:
        img.segment(**kwargs)
    except ValueError as exc:  # npsam raises this when SAM returns no mask at all
        if "at least one array to stack" not in str(exc):
            raise
        from PIL import Image  # an npsam dependency

        w, h = Image.open(image_path).size
        return np.zeros((0, h, w), dtype=bool)
    masks = np.asarray(img.seg.data)
    if masks.ndim == 2:
        masks = masks[None]
    return masks.astype(bool)


def main(argv=None):
    parser = argparse.ArgumentParser(description="NP-SAM baseline (spec 12.3)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--pps", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    with open(args.manifest) as f:
        panels = list(csv.DictReader(f))
    rows = []
    for row in panels:
        panel_id, image_path = row["panel_id"], row["image_path"]
        meta_path = os.path.join(args.out, f"{panel_id}__meta.json")
        if os.path.exists(meta_path) and not args.overwrite:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("status") == "ok":
                rows.append(meta)
                continue
        start = time.time()
        meta = {"panel_id": panel_id, "model": args.model, "pps": args.pps, "status": "ok",
                "n_masks": 0, "error": ""}
        try:
            stack = segment_panel(image_path, args.model, args.pps, args.device)
            np.savez_compressed(os.path.join(args.out, f"{panel_id}__masks.npz"), **pack_masks(stack))
            meta["n_masks"] = int(stack.shape[0])
        except Exception as exc:  # keep going; the failure is recorded per panel
            meta.update(status="FAILED", error=f"{type(exc).__name__}: {exc}"[:300])
        meta["seconds"] = round(time.time() - start, 1)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        rows.append(meta)
        print(f"  {panel_id}: {meta['status']} masks={meta['n_masks']} {meta['seconds']}s", flush=True)

    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["panel_id", "model", "pps", "status", "n_masks", "seconds", "error"])
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in writer.fieldnames} for r in rows)


if __name__ == "__main__":
    main()
