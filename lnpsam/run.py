"""CLI entry point (spec 4.2).

    python -m lnpsam.run --manifest data/manifest.csv --config config.yaml --out runs/<name>

This runs Step 1 (Cleanup, spec 5), Step 2 (Proposals, spec 6), Step 3
(Graph, spec 7), Step 4 (Resolve, spec 8) and Step 6 (Measure, spec 10) for
every panel. Step 5 (Recover, spec 9) is deferred; its orphan fallback (9.6)
runs inside Step 6. Step-2 nodes are cached per panel and SAM is skipped when
every consumed input is unchanged.

``--proposals-from DIR`` reuses Step-2 nodes produced elsewhere (e.g. on the
HPC): for each panel it loads ``proposals/<panel_id>__nodes.parquet`` found
anywhere under DIR only when its cache key matches the local Step 1 and SAM
settings, and never loads SAM. ``--allow-stale-proposals`` permits reuse of a
single mismatched candidate with a warning. Steps 1, 3, 4 and 6 run locally.

Outputs under ``--out``:

- ``measure/`` -- the final, clean results per panel: ``particles.csv``,
  ``panel.json``, ``particles.tif`` / ``aggregates.tif`` (one mask per page),
  ``overlay.png`` and ``overlay_ids.png`` (particles and aggregates only);
  plus the run-level ``particles_all.csv`` and ``panels_all.csv``.
- ``resolve/`` -- the review trace: every mask with its role and rule
  (``roles.csv``, ``nodes.parquet``), Step-4 statistics (``resolve.json``)
  and the all-roles overlays.
- ``cleanup/`` and ``proposals/`` -- Steps 1-2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

from . import cleanup as cleanup_mod
from . import graph as graph_mod
from . import io as lnp_io
from . import measure as measure_mod
from . import overlay as overlay_mod
from . import proposals as proposals_mod
from . import resolve as resolve_mod

# Steps deferred out of the MVP (spec 3, 9).
_DEFERRED_STEPS = ("recover",)


def _write_png(path, rgb_u8):
    # cv2 writes BGR; our arrays are RGB.
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def _proposal_meta_path(nodes_path):
    return os.fspath(nodes_path).removesuffix("__nodes.parquet") + "__meta.json"


def _read_proposal_meta(nodes_path):
    """Treat missing or unreadable source metadata as unvalidated proposals."""
    try:
        with open(_proposal_meta_path(nodes_path)) as stream:
            metadata = json.load(stream)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError):
        return {}


def _proposal_created_time(metadata):
    """Order matching runs by creation time; unknown times sort oldest."""
    try:
        created = datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return created.timestamp()
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError, OSError):
        return float("-inf")


def _stale_proposal_reason(metadata, expected_digests):
    if not metadata:
        return "no __meta.json"
    if "key_schema" not in metadata:
        return "made by an older LNP-SAM (no key_schema); rerun Step 2"
    if metadata["key_schema"] != proposals_mod._KEY_SCHEMA:
        return f"unsupported key_schema {metadata['key_schema']!r}; rerun Step 2"
    digests = metadata.get("input_digests")
    if not isinstance(digests, dict):
        digests = {}
    different = [name for name, value in expected_digests.items() if digests.get(name) != value]
    if different:
        return "differs in: " + ", ".join(different)
    return "cache_key does not match (input digests match)"


def select_proposal_nodes(
    proposals_from, panel_id, expected_key, expected_digests, allow_stale=False
):
    """Select a matching Step-2 run, or explicitly override one stale candidate.

    Only files inside a ``proposals/`` directory count (Step 6 writes its own
    ``<panel_id>__nodes.parquet`` under ``measure/``).
    """
    pattern = os.path.join(proposals_from, "**", "proposals", f"{panel_id}__nodes.parquet")
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"no proposals/{panel_id}__nodes.parquet under {proposals_from}")
    candidates = [(path, _read_proposal_meta(path)) for path in hits]
    matches = [
        candidate for candidate in candidates
        if candidate[1].get("key_schema") == proposals_mod._KEY_SCHEMA
        and candidate[1].get("cache_key") == expected_key
    ]
    if matches:
        path, metadata = max(matches, key=lambda candidate: _proposal_created_time(candidate[1]))
        if len(matches) > 1:
            print(f"    {panel_id}: selected newest matching proposals from {path}")
        return path, metadata, True
    if allow_stale and len(candidates) == 1:
        path, metadata = candidates[0]
        reason = _stale_proposal_reason(metadata, expected_digests)
        print(f"WARNING: {panel_id}: using stale proposals from {path}: {reason}")
        return path, metadata, False

    lines = [f"Stale proposals for panel {panel_id}:"]
    lines.extend(f"  {path}: {_stale_proposal_reason(meta, expected_digests)}" for path, meta in candidates)
    if allow_stale:
        lines.append("More than one candidate exists; point DIR at a single run before using --allow-stale-proposals.")
    lines.append("Rerun Step 2 for this panel, or pass --allow-stale-proposals to use it anyway.")
    raise proposals_mod.StaleProposalsError("\n".join(lines))


def load_proposal_nodes(path, metadata=None):
    """Step-2 nodes from a nodes.parquet written by Step 2, plus its metadata."""
    frame = pd.read_parquet(path)
    nodes = [proposals_mod._cache_record_to_node(r) for r in frame.to_dict(orient="records")]
    meta = _read_proposal_meta(path) if metadata is None else metadata
    return nodes, meta


def _write_overlays(out_dir, panel_id, draw, rgb_u8, *data):
    """``draw(rgb_u8, *data)``, plus an upscaled copy with every outline tagged by
    its id (the particle_id / node_id of particles.csv, roles.csv, nodes.parquet)."""
    _write_png(os.path.join(out_dir, f"{panel_id}__overlay.png"), draw(rgb_u8, *data))
    _write_png(os.path.join(out_dir, f"{panel_id}__overlay_ids.png"),
               draw(rgb_u8, *data, label_ids=True, scale=3))


def run_cleanup_stage(
    manifest_path,
    config_path,
    out_dir,
    run_step2=False,
    run_step3=False,
    run_step4=False,
    run_step6=False,
    device="auto",
    proposals_from=None,
    allow_stale_proposals=False,
    progress=None,
):
    """Run Step 1, optionally followed by cached Step 2, then Steps 3, 4 and 6.

    With ``proposals_from``, Step 2 validates existing nodes against the local
    Step-1 result (see :func:`select_proposal_nodes`) instead of running SAM.
    ``progress(index, total, panel_id, stage)`` runs at each stage boundary;
    indices are one-based, and exceptions from the callback propagate.
    """
    if allow_stale_proposals and not proposals_from:
        raise ValueError("allow_stale_proposals requires proposals_from")
    if run_step3 and not run_step2:
        raise ValueError("Step 3 requires Step 2 nodes; set run_step2=True")
    if run_step4 and not run_step3:
        raise ValueError("Step 4 requires the Step-3 graph; set run_step3=True")
    if run_step6 and not run_step4:
        raise ValueError("Step 6 requires Step-4 roles; set run_step4=True")
    cfg = lnp_io.load_config(config_path)
    manifest = lnp_io.load_manifest(manifest_path)

    cleanup_dir = os.path.join(out_dir, "cleanup")
    os.makedirs(cleanup_dir, exist_ok=True)

    if run_step2 and proposals_from:
        proposals_dir = os.path.join(out_dir, "proposals")
        os.makedirs(proposals_dir, exist_ok=True)
        resolved_device = None
        checkpoint = None
        print(f"[lnpsam] Step 2: reusing nodes under {proposals_from} (SAM not loaded)")
    elif run_step2:
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
            sam_model = proposals_mod.load_sam_model(
                checkpoint, resolved_device, expected_sha256=cfg["sam"]["checkpoint_sha256"]
            )
            mask_generator = proposals_mod.make_mask_generator(sam_model, cfg["sam"])
        return mask_generator

    resolve_dir = os.path.join(out_dir, "resolve")
    if run_step4:
        os.makedirs(resolve_dir, exist_ok=True)
    measure_dir = os.path.join(out_dir, "measure")
    if run_step6:
        os.makedirs(measure_dir, exist_ok=True)

    rows = []
    proposal_rows = []
    resolve_rows = []
    particle_frames = []
    panel_rows = []
    for image_index, (_, row) in enumerate(manifest.iterrows(), start=1):
        panel_id = row["panel_id"]
        image_path = row["image_path"]
        modality = row["modality"]
        manifest_nm = row.get("nm_per_px", np.nan)

        if progress is not None:
            progress(image_index, len(manifest), panel_id, "cleanup")
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

        if run_step2 and progress is not None:
            progress(image_index, len(manifest), panel_id, "proposals")
        if run_step2 and proposals_from:
            expected_key = proposals_mod.proposal_cache_key(gray, result, cfg["sam"])
            expected_digests = proposals_mod.proposal_input_digests(gray, result, cfg["sam"])
            nodes_path, proposal_meta, proposals_validated = select_proposal_nodes(
                proposals_from, panel_id, expected_key, expected_digests, allow_stale_proposals
            )
            nodes, proposal_meta = load_proposal_nodes(nodes_path, proposal_meta)
            meta_path = _proposal_meta_path(nodes_path)
            cache_hit = None
            proposal_device = proposal_meta.get("device")
            print(f"    proposals={len(nodes)} (reused from {nodes_path}, validated={proposals_validated})")
        elif run_step2:
            nodes, cache_hit, proposal_meta = proposals_mod.run_proposals_cached(
                gray=gray,
                cleanup_result=result,
                cfg=cfg,
                panel_id=panel_id,
                cache_dir=proposals_dir,
                device=resolved_device,
                mask_generator_factory=get_mask_generator,
            )
            cache_status = "hit; SAM skipped" if cache_hit else "miss; SAM ran"
            print(f"    proposals={len(nodes)}, cache={cache_status}, device={resolved_device}")
            nodes_path, meta_path = proposals_mod.proposal_cache_paths(
                proposals_dir, panel_id
            )
            proposals_validated = True
            proposal_device = resolved_device
        if run_step2:
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
                    "device": proposal_device,
                    "cache_hit": cache_hit,
                    "cache_key": proposal_meta.get("cache_key"),
                    "n_raw_masks": proposal_meta.get("n_raw_masks"),
                    "n_nodes": len(nodes),
                    "proposals_validated": proposals_validated,
                    "proposals_source": str(nodes_path),
                    "step2_seconds": (
                        proposal_meta["timing"].get("total_s")
                        if isinstance(proposal_meta.get("timing"), dict) else None
                    ),
                    "nodes_path": str(nodes_path),
                    "meta_path": str(meta_path),
                    "overlay_path": proposal_overlay_path,
                }
            )

            if run_step3:
                if progress is not None:
                    progress(image_index, len(manifest), panel_id, "graph")
                graph_result = graph_mod.run_graph(nodes, cfg)
                duplicate_count = len(nodes) - len(graph_result.representative_ids)
                adjacency_count = sum(
                    len(node["adjacent"])
                    for node in graph_result.nodes
                    if node["is_representative"]
                ) // 2
                parent_count = sum(
                    node["parent"] is not None
                    for node in graph_result.nodes
                    if node["is_representative"]
                )
                print(
                    f"    graph: representatives={len(graph_result.representative_ids)}, "
                    f"duplicates={duplicate_count}, parents={parent_count}, "
                    f"adjacencies={adjacency_count}"
                )

            if run_step4:
                if progress is not None:
                    progress(image_index, len(manifest), panel_id, "resolve")
                resolved = resolve_mod.run_resolve(
                    graph_result, gray, result.content_mask, modality, result.nm_per_px, cfg
                )
                resolve_mod.roles_table(resolved).to_csv(
                    os.path.join(resolve_dir, f"{panel_id}__roles.csv"), index=False
                )
                with open(os.path.join(resolve_dir, f"{panel_id}__resolve.json"), "w") as f:
                    json.dump({**resolved.panel, "log": resolved.log}, f, indent=2)
                _write_overlays(resolve_dir, panel_id, overlay_mod.resolve_overlay, rgb_u8,
                                resolved.nodes, result.exclusion, resolved.panel)
                counts = resolved.panel["role_counts"]
                shown = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
                flags = ",".join(resolved.panel["flags"]) or "-"
                print(f"    resolve: {shown}; flags={flags}")
                for line in resolved.log:
                    print(f"    [unresolved] {line}")
                resolve_rows.append(
                    {
                        "panel_id": panel_id,
                        "modality": modality,
                        **{k: resolved.panel[k] for k in (
                            "nm_per_px", "n_seeds", "seed_mode", "bg_mode", "mu_bg",
                            "mu_p", "d_ref", "d_min", "d_min_source", "n_truncated",
                            "n_touches_exclusion",
                        )},
                        "flags": ";".join(resolved.panel["flags"]),
                        **{f"n_{k}": v for k, v in counts.items()},
                    }
                )

            if run_step6:
                if progress is not None:
                    progress(image_index, len(manifest), panel_id, "measure")
                measured = measure_mod.run_measure(
                    resolved, panel_id, row["paper_id"], result.nm_per_px,
                    result.cal_source, cfg,
                )
                measure_mod.write_outputs(measured, measure_dir, panel_id)
                # The full trace (every node, after the 9.6 fallback) is a review output.
                measure_mod.nodes_table(measured.nodes).to_parquet(
                    os.path.join(resolve_dir, f"{panel_id}__nodes.parquet"), index=False
                )
                particles, aggregates = measure_mod.final_masks(measured)
                _write_overlays(measure_dir, panel_id, overlay_mod.overlay, rgb_u8,
                                particles, aggregates, result.exclusion, measured.panel)
                particle_frames.append(measured.particles)
                panel_rows.append(measure_mod.panel_row(measured.panel))
                stats = measured.panel["stats_all"]
                unit = measured.panel["stats_unit"]
                summary = (
                    f"median d={stats['median']:.1f} {unit} (n={stats['n']})"
                    if stats else "too few particles for stats"
                )
                print(
                    f"    measure: particles={measured.panel['n_particles']}, "
                    f"in_stats={measured.panel['n_in_stats']}, "
                    f"hollow={measured.panel['n_hollow']}, "
                    f"aggregates={measured.panel['n_aggregates']}; {summary}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cleanup_dir, "cleanup_all.csv"), index=False)
    if run_step2:
        pd.DataFrame(proposal_rows).to_csv(
            os.path.join(proposals_dir, "proposals_all.csv"), index=False
        )
    if run_step4:
        pd.DataFrame(resolve_rows).to_csv(
            os.path.join(resolve_dir, "resolve_all.csv"), index=False
        )
    if run_step6:
        particles_all = (
            pd.concat(particle_frames, ignore_index=True)
            if particle_frames else pd.DataFrame(columns=measure_mod.PARTICLE_COLUMNS)
        )
        particles_all.to_csv(os.path.join(out_dir, "particles_all.csv"), index=False)
        pd.DataFrame(panel_rows).to_csv(os.path.join(out_dir, "panels_all.csv"), index=False)
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
    parser.add_argument(
        "--proposals-from",
        default=None,
        help="Reuse Step-2 nodes from proposals/<panel_id>__nodes.parquet under this "
        "directory (e.g. an HPC run) only when their cache key matches local Step 1 "
        "and config, instead of running SAM",
    )
    parser.add_argument(
        "--allow-stale-proposals",
        action="store_true",
        help="Allow one mismatched Step-2 candidate under --proposals-from, with a warning",
    )
    args = parser.parse_args(argv)
    if args.allow_stale_proposals and not args.proposals_from:
        parser.error("--allow-stale-proposals requires --proposals-from")

    os.makedirs(args.out, exist_ok=True)
    print(f"[lnpsam] Step 1: Cleanup -> {os.path.join(args.out, 'cleanup')}")
    print(f"[lnpsam] Step 2: Proposals -> {os.path.join(args.out, 'proposals')}")
    print("[lnpsam] Step 3: Graph")
    print(f"[lnpsam] Step 4: Resolve -> {os.path.join(args.out, 'resolve')}")
    print(f"[lnpsam] Step 6: Measure -> {os.path.join(args.out, 'measure')}")
    try:
        run_cleanup_stage(
            args.manifest,
            args.config,
            args.out,
            run_step2=True,
            run_step3=True,
            run_step4=True,
            run_step6=True,
            device=args.device,
            proposals_from=args.proposals_from,
            allow_stale_proposals=args.allow_stale_proposals,
        )
    except proposals_mod.StaleProposalsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"[lnpsam] Done: {os.path.join(args.out, 'particles_all.csv')}, "
        f"{os.path.join(args.out, 'panels_all.csv')}. Deferred step: "
        f"{', '.join(_DEFERRED_STEPS)} (spec 9; orphan fallback applied in Step 6)."
    )


if __name__ == "__main__":
    main()
