"""CLI entry point (spec 4.2).

    python -m lnpsam.run --manifest data/manifest.csv --config config.yaml --out runs/<name>

Stage 1 status: this runs Step 1 (Cleanup, spec 5) for every panel in the manifest
and writes the cleanup artifacts. Steps 2-6 are not implemented yet; the run stops
after cleanup with a notice.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from . import cleanup as cleanup_mod
from . import io as lnp_io
from . import overlay as overlay_mod

# Pipeline steps that exist as stubs; presence here documents what is pending.
_PENDING_STEPS = ("proposals", "graph", "resolve", "recover", "measure")


def _write_png(path, rgb_u8):
    # cv2 writes BGR; our arrays are RGB.
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def run_cleanup_stage(manifest_path, config_path, out_dir):
    """Run Step 1 for every manifest row; write per-panel and aggregate outputs."""
    cfg = lnp_io.load_config(config_path)
    manifest = lnp_io.load_manifest(manifest_path)

    cleanup_dir = os.path.join(out_dir, "cleanup")
    os.makedirs(cleanup_dir, exist_ok=True)

    reader = cleanup_mod.get_reader()  # created once per process (spec 5.2)

    rows = []
    for _, row in manifest.iterrows():
        panel_id = row["panel_id"]
        image_path = row["image_path"]
        modality = row["modality"]
        manifest_nm = row.get("nm_per_px", np.nan)

        gray, rgb_u8 = lnp_io.load_image(image_path)
        H, W = gray.shape

        result = cleanup_mod.run_cleanup(
            gray, rgb_u8, modality, cfg, reader=reader, manifest_nm_per_px=manifest_nm
        )

        # Artifacts.
        _write_png(os.path.join(cleanup_dir, f"{panel_id}__saminput.png"), result.sam_image)
        _write_png(
            os.path.join(cleanup_dir, f"{panel_id}__overlay.png"),
            overlay_mod.cleanup_overlay(rgb_u8, result),
        )

        excl_frac = float(result.exclusion.mean())
        record = {
            "panel_id": panel_id,
            "paper_id": row["paper_id"],
            "modality": modality,
            "width": int(W),
            "height": int(H),
            "content_box": list(result.content_box),
            "crop_offset": list(result.crop_offset),
            "nm_per_px": result.nm_per_px,
            "cal_source": result.cal_source,
            "L_px": result.L_px,
            "scale_value_nm": result.scale_value_nm,
            "scale_text": result.scale_text,
            "n_ocr_boxes": len(result.ocr_boxes),
            "n_bars": len(result.bars),
            "n_label_boxes": len(result.label_boxes),
            "n_annotation_boxes": len(result.annotation_boxes),
            "n_burned_bars": len(result.burned_bars),
            "exclusion_frac": excl_frac,
        }
        with open(os.path.join(cleanup_dir, f"{panel_id}.json"), "w") as f:
            json.dump(record, f, indent=2)
        rows.append(record)

        cal = (
            "px"
            if result.cal_source == "missing"
            else f"{result.nm_per_px:.4g} nm/px ({result.cal_source})"
        )
        print(f"  {panel_id}: cal={cal}, excl={excl_frac:.3f}, bars={len(result.bars)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cleanup_dir, "cleanup_all.csv"), index=False)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lnpsam.run", description="LNP-SAM pipeline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[lnpsam] Step 1: Cleanup -> {os.path.join(args.out, 'cleanup')}")
    run_cleanup_stage(args.manifest, args.config, args.out)
    print(
        f"[lnpsam] Cleanup complete. Steps not yet implemented: "
        f"{', '.join(_PENDING_STEPS)} (stages 2-6)."
    )


if __name__ == "__main__":
    main()
