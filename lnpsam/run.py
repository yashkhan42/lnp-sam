"""CLI entry point (spec 4.2).

    python -m lnpsam.run --manifest data/manifest.csv --config config.yaml --out runs/<name>

This runs Step 1 (Cleanup, spec 5) and Step 2 (Proposals, spec 6) for every panel.
Step-2 nodes are cached per panel and SAM is skipped when every consumed input is
unchanged. Steps 3-6 are not implemented yet.
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
from . import proposals as proposals_mod

# Pipeline steps that exist as stubs; presence here documents what is pending.
_PENDING_STEPS = ("graph", "resolve", "recover", "measure")


def _write_png(path, rgb_u8):
    # cv2 writes BGR; our arrays are RGB.
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def run_cleanup_stage(manifest_path, config_path, out_dir, run_step2=False, device="auto"):
    """Run Step 1, optionally followed by cached Step 2, for every panel."""
    cfg = lnp_io.load_config(config_path)
    manifest = lnp_io.load_manifest(manifest_path)

    cleanup_dir = os.path.join(out_dir, "cleanup")
    os.makedirs(cleanup_dir, exist_ok=True)

    if run_step2:
        proposals_dir = os.path.join(out_dir, "proposals")
        os.makedirs(proposals_dir, exist_ok=True)
        resolved_device = proposals_mod.select_device(device)
        checkpoint = cfg["sam"]["checkpoint"]
        print(f"[lnpsam] Step 2 device: {resolved_device}")
    else:
        proposals_dir = None
        resolved_device = None
        checkpoint = None

    reader = cleanup_mod.get_reader()  # created once per process (spec 5.2)

    # The model is loaded on the first cache miss, not at process startup. A
    # completely cached run therefore never allocates the model or reads weights.
    sam_model = None
    mask_generator = None

    def get_mask_generator():
        nonlocal sam_model, mask_generator
        if mask_generator is None:
            print(f"[lnpsam] Loading SAM ViT-B checkpoint: {checkpoint}")
            sam_model = proposals_mod.load_sam_model(checkpoint, resolved_device)
            mask_generator = proposals_mod.make_mask_generator(sam_model, cfg["sam"])
        return mask_generator

    rows = []
    proposal_rows = []
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

        if run_step2:
            nodes, cache_hit, proposal_meta = proposals_mod.run_proposals_cached(
                gray=gray,
                cleanup_result=result,
                cfg=cfg,
                panel_id=panel_id,
                cache_dir=proposals_dir,
                checkpoint=checkpoint,
                device=resolved_device,
                mask_generator_factory=get_mask_generator,
            )
            cache_status = "hit; SAM skipped" if cache_hit else "miss; SAM ran"
            print(f"    proposals={len(nodes)}, cache={cache_status}, device={resolved_device}")
            nodes_path, meta_path = proposals_mod.proposal_cache_paths(
                proposals_dir, panel_id
            )
            proposal_overlay_path = os.path.join(
                proposals_dir, f"{panel_id}__overlay.png"
            )
            _write_png(
                proposal_overlay_path,
                overlay_mod.proposal_overlay(rgb_u8, nodes, panel_id),
            )
            proposal_rows.append(
                {
                    "panel_id": panel_id,
                    "paper_id": row["paper_id"],
                    "device": resolved_device,
                    "cache_hit": cache_hit,
                    "cache_key": proposal_meta["cache_key"],
                    "n_raw_masks": proposal_meta["n_raw_masks"],
                    "n_nodes": proposal_meta["n_nodes"],
                    "nodes_path": str(nodes_path),
                    "meta_path": str(meta_path),
                    "overlay_path": proposal_overlay_path,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cleanup_dir, "cleanup_all.csv"), index=False)
    if run_step2:
        pd.DataFrame(proposal_rows).to_csv(
            os.path.join(proposals_dir, "proposals_all.csv"), index=False
        )
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lnpsam.run", description="LNP-SAM pipeline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Step-2 device: auto (CUDA, then MPS, then CPU), cpu, mps, cuda, or cuda:<index>",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[lnpsam] Step 1: Cleanup -> {os.path.join(args.out, 'cleanup')}")
    print(f"[lnpsam] Step 2: Proposals -> {os.path.join(args.out, 'proposals')}")
    run_cleanup_stage(
        args.manifest,
        args.config,
        args.out,
        run_step2=True,
        device=args.device,
    )
    print(
        f"[lnpsam] Steps 1-2 complete. Steps not yet implemented: "
        f"{', '.join(_PENDING_STEPS)} (stages 3-6)."
    )


if __name__ == "__main__":
    main()
