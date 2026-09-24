"""Step 2: permissive SAM proposals and node features (spec 6).

The SAM dependency is imported lazily so the deterministic post-processing can
be tested without a checkpoint or GPU. Masks are generated in the cropped
Step-1 coordinate frame, then expanded to full-panel coordinates before RLEs or
features are produced (spec 4.4).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
from skimage.filters import gaussian, sobel
from skimage.segmentation import find_boundaries

from . import io as lnp_io


_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)

_CACHE_COLUMNS = (
    "node_id",
    "sam_annotation_id",
    "piece_index",
    "predicted_iou",
    "stability_score",
    "point_coords",
    "source",
    "rle",
    "filled_rle",
    "n_holes",
    "excl_overlap",
    "touches_exclusion",
    "area",
    "eqdiam",
    "solidity",
    "circularity",
    "centroid_row",
    "centroid_col",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "touches_border",
    "mean_int",
    "edge_contrast",
    "quality",
    "bg_candidate",
)


def select_device(requested="auto", torch_module=None):
    """Resolve the Step-2 runtime device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. Explicit accelerator
    requests fail instead of silently falling back, which keeps run metadata
    honest about the device that was actually used.
    """
    if torch_module is None:
        import torch as torch_module

    requested = str(requested).lower()
    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError(f"requested device {requested!r}, but CUDA is unavailable")
        return requested
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("requested device 'mps', but Apple MPS is unavailable")
        return requested
    if requested == "cpu":
        return requested
    raise ValueError("device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'")


def load_sam_model(checkpoint, device):
    """Load the spec-mandated SAM ViT-B checkpoint on ``device``."""
    checkpoint = os.path.abspath(os.path.expanduser(os.fspath(checkpoint)))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"SAM ViT-B checkpoint not found: {checkpoint}. "
            "Set sam.checkpoint in config.yaml to the HPC checkpoint path."
        )
    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:  # pragma: no cover - installation is environment-specific
        raise ImportError(
            "Step 2 requires segment-anything; install it as described in spec 4.1"
        ) from exc

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return sam


def make_mask_generator(sam, sam_config, generator_cls=None):
    """Build the automatic mask generator exactly as specified in section 6.1.

    ``generator_cls`` is injectable for unit tests. In production it defaults
    to :class:`segment_anything.SamAutomaticMaskGenerator`.
    """
    if generator_cls is None:
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError as exc:  # pragma: no cover - depends on optional SAM install
            raise ImportError(
                "Step 2 requires segment-anything; install it as described in spec 4.1"
            ) from exc
        generator_cls = SamAutomaticMaskGenerator

    return generator_cls(
        sam,
        points_per_side=sam_config["points_per_side"],
        points_per_batch=sam_config["points_per_batch"],
        pred_iou_thresh=sam_config["pred_iou_thresh"],
        stability_score_thresh=sam_config["stability_score_thresh"],
        box_nms_thresh=sam_config["box_nms_thresh"],
        crop_n_layers=0,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )


def _full_panel_mask(mask_crop, full_shape, crop_offset):
    """Embed a cropped SAM mask into a full-panel boolean mask."""
    mask_crop = np.asarray(mask_crop, dtype=bool)
    if mask_crop.ndim != 2:
        raise ValueError(f"SAM segmentation must be 2-D, got shape {mask_crop.shape}")

    height, width = full_shape
    x0, y0 = (int(crop_offset[0]), int(crop_offset[1]))
    crop_h, crop_w = mask_crop.shape
    if x0 < 0 or y0 < 0 or x0 + crop_w > width or y0 + crop_h > height:
        raise ValueError(
            "SAM segmentation and crop_offset do not fit the full panel: "
            f"crop={mask_crop.shape}, offset={(x0, y0)}, panel={full_shape}"
        )

    full = np.zeros(full_shape, dtype=bool)
    full[y0 : y0 + crop_h, x0 : x0 + crop_w] = mask_crop
    return full


def _offset_points(point_coords, crop_offset):
    """Convert SAM ``(x, y)`` point prompts from crop to full-panel coordinates."""
    x0, y0 = crop_offset
    points = np.asarray(point_coords, dtype=float)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"point_coords must have shape (N, 2), got {points.shape}")
    points = points.copy()
    points[:, 0] += x0
    points[:, 1] += y0
    return points.tolist()


def _split_annotations(annotations, full_shape, crop_offset, min_piece_px):
    """Split SAM masks into 8-connected pieces (spec 6.2.1).

    The largest surviving piece retains the raw SAM annotation index as its
    ``node_id``. Additional pieces receive monotonically increasing integer IDs
    after the range occupied by raw annotations.
    """
    pieces = []
    next_id = len(annotations)

    for annotation_id, ann in enumerate(annotations):
        missing = {
            "segmentation",
            "predicted_iou",
            "stability_score",
            "point_coords",
        } - set(ann)
        if missing:
            raise ValueError(f"SAM annotation {annotation_id} is missing keys: {sorted(missing)}")

        full = _full_panel_mask(ann["segmentation"], full_shape, crop_offset)
        labels, count = ndimage.label(full, structure=_EIGHT_CONNECTED)
        components = []
        for label_id in range(1, count + 1):
            component = labels == label_id
            area = int(component.sum())
            if area >= min_piece_px:
                components.append((area, label_id, component))

        # Area first, then label order for deterministic ties. The first item
        # is the largest component and retains the raw mask's ID.
        components.sort(key=lambda item: (-item[0], item[1]))
        for piece_index, (_, _, component) in enumerate(components):
            if piece_index == 0:
                node_id = annotation_id
            else:
                node_id = next_id
                next_id += 1
            pieces.append(
                {
                    "node_id": int(node_id),
                    "sam_annotation_id": int(annotation_id),
                    "piece_index": int(piece_index),
                    "mask": component,
                    "predicted_iou": float(ann["predicted_iou"]),
                    "stability_score": float(ann["stability_score"]),
                    "point_coords": _offset_points(ann["point_coords"], crop_offset),
                    "source": "sam",
                }
            )
    return pieces


def _hole_count(mask, filled):
    """Count hole components of at least four pixels (spec 6.2.2)."""
    holes = np.logical_and(filled, ~mask)
    labels, count = ndimage.label(holes, structure=_EIGHT_CONNECTED)
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(sizes >= 4))


def _edge_hits(filled, content_box):
    """Return which content edges have mask pixels within one pixel (spec 6.3)."""
    x0, y0, x1, y1 = (int(value) for value in content_box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid content_box: {content_box}")
    return {
        "top": bool(filled[y0 : min(y0 + 2, y1), x0:x1].any()),
        "bottom": bool(filled[max(y0, y1 - 2) : y1, x0:x1].any()),
        "left": bool(filled[y0:y1, x0 : min(x0 + 2, x1)].any()),
        "right": bool(filled[y0:y1, max(x0, x1 - 2) : x1].any()),
    }


def _node_features(node, gray, gradient, gradient_scale, content_box):
    """Compute the section 6.3 features on a node's filled mask."""
    filled = node["filled"]
    area = int(filled.sum())
    rows, cols = np.nonzero(filled)
    if area == 0:
        raise ValueError("cannot compute features for an empty proposal")

    perimeter = float(measure.perimeter(filled))
    if perimeter > 0:
        circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
    else:
        circularity = 1.0

    regions = measure.regionprops(filled.astype(np.uint8))
    solidity = float(regions[0].solidity)
    boundary = find_boundaries(filled, mode="inner")
    boundary_gradient = float(gradient[boundary].mean()) if boundary.any() else 0.0
    edge_contrast = boundary_gradient / gradient_scale if gradient_scale > 0 else 0.0

    edge_hits = _edge_hits(filled, content_box)
    quality = (
        0.5 * node["predicted_iou"]
        + 0.3 * node["stability_score"]
        + 0.2 * min(edge_contrast, 1.0)
    )

    node.update(
        {
            "area": area,
            "eqdiam": float(math.sqrt(4.0 * area / math.pi)),
            "solidity": solidity,
            "circularity": float(circularity),
            "centroid": (float(rows.mean()), float(cols.mean())),
            "bbox": (
                int(cols.min()),
                int(rows.min()),
                int(cols.max()) + 1,
                int(rows.max()) + 1,
            ),
            "touches_border": any(edge_hits.values()),
            "mean_int": float(gray[filled].mean()),
            "edge_contrast": float(edge_contrast),
            "quality": float(quality),
            "_edge_hits": edge_hits,
        }
    )


def postprocess_proposals(
    annotations,
    gray,
    exclusion,
    content_mask,
    content_box,
    crop_offset,
    sam_config,
):
    """Apply all and only the pre-Step-4 removals in spec section 6.2.

    Parameters are full-panel arrays except each SAM annotation's cropped
    ``segmentation``. Returned node masks, RLEs, points, centroids and boxes are
    all in full-panel coordinates. Nodes that touch the border are retained.
    """
    gray = np.asarray(gray, dtype=np.float32)
    exclusion = np.asarray(exclusion, dtype=bool)
    content_mask = np.asarray(content_mask, dtype=bool)
    if gray.ndim != 2:
        raise ValueError(f"gray must be 2-D, got shape {gray.shape}")
    if exclusion.shape != gray.shape or content_mask.shape != gray.shape:
        raise ValueError("gray, exclusion, and content_mask must have identical shapes")
    if not content_mask.any():
        raise ValueError("content_mask must contain at least one pixel")

    min_piece_px = int(sam_config["min_piece_px"])
    excl_drop_frac = float(sam_config["excl_drop_frac"])
    content_area = int(content_mask.sum())

    # G = |sobel(gaussian(gray, 1))|. ``sobel`` returns the gradient magnitude.
    gradient = sobel(gaussian(gray, sigma=1, preserve_range=True))
    gradient_scale = float(np.percentile(gradient[content_mask], 95))

    pieces = _split_annotations(annotations, gray.shape, crop_offset, min_piece_px)
    candidates = []
    for node in pieces:
        mask = node["mask"]
        filled = ndimage.binary_fill_holes(mask)
        filled_area = int(filled.sum())
        if filled_area == 0:
            continue

        excl_overlap = float(np.logical_and(filled, exclusion).sum() / filled_area)
        if excl_overlap >= excl_drop_frac:
            continue

        node.update(
            {
                "filled": filled,
                "rle": lnp_io.encode_rle(mask),
                "filled_rle": lnp_io.encode_rle(filled),
                "n_holes": _hole_count(mask, filled),
                "excl_overlap": excl_overlap,
                "touches_exclusion": excl_overlap > 0.0,
            }
        )
        _node_features(node, gray, gradient, gradient_scale, content_box)
        candidates.append(node)

    # Exact duplicates are identical *filled* masks. Keep the node with the
    # highest section 6.3 quality; stable input order resolves an exact tie.
    unique_by_filled_rle = {}
    for node in candidates:
        key = node["filled_rle"]
        current = unique_by_filled_rle.get(key)
        if current is None or node["quality"] > current["quality"]:
            unique_by_filled_rle[key] = node

    nodes = sorted(unique_by_filled_rle.values(), key=lambda node: node["node_id"])
    for node in nodes:
        edge_count = sum(node.pop("_edge_hits").values())
        node["bg_candidate"] = bool(
            node["area"] >= float(sam_config["bg_area_frac"]) * content_area
            and edge_count >= 2
        )
    return nodes


def _hash_array(digest, name, array):
    """Add an array's name, schema, and exact bytes to a cache digest."""
    array = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(memoryview(array).cast("B"))


def _checkpoint_identity(checkpoint):
    """Return the checkpoint identity used in proposal-cache invalidation."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM ViT-B checkpoint not found: {path}. "
            "Set sam.checkpoint in config.yaml to the HPC checkpoint path."
        )
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def proposal_cache_key(gray, cleanup_result, sam_config, checkpoint, device):
    """Hash every input that can affect Step 2.

    The key covers the source pixels, all Step-1 products consumed by Step 2,
    SAM settings, checkpoint identity, resolved device, installed SAM version,
    and this module's source. Consequently a cache hit means SAM and proposal
    post-processing have no changed inputs.
    """
    checkpoint_identity = _checkpoint_identity(checkpoint)
    try:
        sam_version = importlib.metadata.version("segment-anything")
    except importlib.metadata.PackageNotFoundError:
        sam_version = "missing"

    digest = hashlib.sha256()
    _hash_array(digest, "gray", np.asarray(gray, dtype=np.float32))
    _hash_array(digest, "sam_image", cleanup_result.sam_image)
    _hash_array(digest, "exclusion", np.asarray(cleanup_result.exclusion, dtype=np.uint8))
    _hash_array(digest, "content_mask", np.asarray(cleanup_result.content_mask, dtype=np.uint8))
    metadata = {
        "content_box": list(cleanup_result.content_box),
        "crop_offset": list(cleanup_result.crop_offset),
        "sam_config": sam_config,
        "checkpoint": checkpoint_identity,
        "device": device,
        "segment_anything": sam_version,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest(), checkpoint_identity


def _node_to_cache_record(node):
    """Convert an in-memory node to a stable Parquet row."""
    centroid_row, centroid_col = node["centroid"]
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = node["bbox"]
    return {
        "node_id": int(node["node_id"]),
        "sam_annotation_id": int(node["sam_annotation_id"]),
        "piece_index": int(node["piece_index"]),
        "predicted_iou": float(node["predicted_iou"]),
        "stability_score": float(node["stability_score"]),
        "point_coords": json.dumps(node["point_coords"], separators=(",", ":")),
        "source": node["source"],
        "rle": node["rle"],
        "filled_rle": node["filled_rle"],
        "n_holes": int(node["n_holes"]),
        "excl_overlap": float(node["excl_overlap"]),
        "touches_exclusion": bool(node["touches_exclusion"]),
        "area": int(node["area"]),
        "eqdiam": float(node["eqdiam"]),
        "solidity": float(node["solidity"]),
        "circularity": float(node["circularity"]),
        "centroid_row": float(centroid_row),
        "centroid_col": float(centroid_col),
        "bbox_x0": int(bbox_x0),
        "bbox_y0": int(bbox_y0),
        "bbox_x1": int(bbox_x1),
        "bbox_y1": int(bbox_y1),
        "touches_border": bool(node["touches_border"]),
        "mean_int": float(node["mean_int"]),
        "edge_contrast": float(node["edge_contrast"]),
        "quality": float(node["quality"]),
        "bg_candidate": bool(node["bg_candidate"]),
    }


def _cache_record_to_node(record):
    """Restore an in-memory node, including decoded overlapping masks."""
    rle = record["rle"]
    filled_rle = record["filled_rle"]
    return {
        "node_id": int(record["node_id"]),
        "sam_annotation_id": int(record["sam_annotation_id"]),
        "piece_index": int(record["piece_index"]),
        "mask": lnp_io.decode_rle(rle),
        "filled": lnp_io.decode_rle(filled_rle),
        "predicted_iou": float(record["predicted_iou"]),
        "stability_score": float(record["stability_score"]),
        "point_coords": json.loads(record["point_coords"]),
        "source": str(record["source"]),
        "rle": rle,
        "filled_rle": filled_rle,
        "n_holes": int(record["n_holes"]),
        "excl_overlap": float(record["excl_overlap"]),
        "touches_exclusion": bool(record["touches_exclusion"]),
        "area": int(record["area"]),
        "eqdiam": float(record["eqdiam"]),
        "solidity": float(record["solidity"]),
        "circularity": float(record["circularity"]),
        "centroid": (float(record["centroid_row"]), float(record["centroid_col"])),
        "bbox": (
            int(record["bbox_x0"]),
            int(record["bbox_y0"]),
            int(record["bbox_x1"]),
            int(record["bbox_y1"]),
        ),
        "touches_border": bool(record["touches_border"]),
        "mean_int": float(record["mean_int"]),
        "edge_contrast": float(record["edge_contrast"]),
        "quality": float(record["quality"]),
        "bg_candidate": bool(record["bg_candidate"]),
    }


def proposal_cache_paths(cache_dir, panel_id):
    """Return the per-panel Parquet and metadata paths."""
    cache_dir = Path(cache_dir)
    return (
        cache_dir / f"{panel_id}__nodes.parquet",
        cache_dir / f"{panel_id}__meta.json",
    )


def load_proposal_cache(cache_dir, panel_id, expected_key):
    """Load a valid per-panel Step-2 cache, or return ``None`` on a miss."""
    nodes_path, meta_path = proposal_cache_paths(cache_dir, panel_id)
    if not nodes_path.is_file() or not meta_path.is_file():
        return None
    try:
        with meta_path.open("r") as stream:
            metadata = json.load(stream)
        if metadata.get("cache_key") != expected_key:
            return None
        frame = pd.read_parquet(nodes_path)
        if tuple(frame.columns) != _CACHE_COLUMNS:
            return None
        if len(frame) != int(metadata["n_nodes"]):
            return None
        nodes = [_cache_record_to_node(row) for row in frame.to_dict(orient="records")]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return nodes, metadata


def save_proposal_cache(
    cache_dir,
    panel_id,
    cache_key,
    checkpoint_identity,
    device,
    nodes,
    raw_count,
):
    """Atomically save one panel's Step-2 nodes and cache metadata."""
    nodes_path, meta_path = proposal_cache_paths(cache_dir, panel_id)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [_node_to_cache_record(node) for node in nodes], columns=_CACHE_COLUMNS
    )
    metadata = {
        "panel_id": panel_id,
        "cache_key": cache_key,
        "checkpoint": checkpoint_identity,
        "device": device,
        "n_raw_masks": int(raw_count),
        "n_nodes": int(len(nodes)),
    }

    nodes_tmp = nodes_path.with_suffix(nodes_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    try:
        frame.to_parquet(nodes_tmp, index=False)
        with meta_tmp.open("w") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        os.replace(nodes_tmp, nodes_path)
        os.replace(meta_tmp, meta_path)
    finally:
        for temporary in (nodes_tmp, meta_tmp):
            if temporary.exists():
                temporary.unlink()
    return metadata


def run_proposals_cached(
    gray,
    cleanup_result,
    cfg,
    panel_id,
    cache_dir,
    checkpoint,
    device,
    mask_generator_factory,
):
    """Load unchanged Step-2 output or generate, post-process, and persist it.

    ``mask_generator_factory`` is called only on a cache miss, allowing the CLI
    to avoid loading the checkpoint when every panel is already cached.
    Returns ``(nodes, cache_hit, metadata)``.
    """
    cache_key, checkpoint_identity = proposal_cache_key(
        gray, cleanup_result, cfg["sam"], checkpoint, device
    )
    cached = load_proposal_cache(cache_dir, panel_id, cache_key)
    if cached is not None:
        nodes, metadata = cached
        return nodes, True, metadata

    mask_generator = mask_generator_factory()
    annotations = mask_generator.generate(cleanup_result.sam_image)
    nodes = postprocess_proposals(
        annotations=annotations,
        gray=gray,
        exclusion=cleanup_result.exclusion,
        content_mask=cleanup_result.content_mask,
        content_box=cleanup_result.content_box,
        crop_offset=cleanup_result.crop_offset,
        sam_config=cfg["sam"],
    )
    metadata = save_proposal_cache(
        cache_dir=cache_dir,
        panel_id=panel_id,
        cache_key=cache_key,
        checkpoint_identity=checkpoint_identity,
        device=device,
        nodes=nodes,
        raw_count=len(annotations),
    )
    return nodes, False, metadata


def run_proposals(gray, cleanup_result, cfg, sam=None, mask_generator=None):
    """Run section 6.1 SAM generation and section 6.2-6.3 post-processing.

    Pass either a loaded ViT-B ``sam`` model or an already-created
    ``mask_generator``. The latter supports deterministic tests and callers
    that manage model lifetime themselves.
    """
    if mask_generator is not None and sam is not None:
        raise ValueError("pass either sam or mask_generator, not both")
    if mask_generator is None:
        if sam is None:
            raise ValueError("run_proposals requires a loaded SAM model or mask_generator")
        mask_generator = make_mask_generator(sam, cfg["sam"])

    annotations = mask_generator.generate(cleanup_result.sam_image)
    return postprocess_proposals(
        annotations=annotations,
        gray=gray,
        exclusion=cleanup_result.exclusion,
        content_mask=cleanup_result.content_mask,
        content_box=cleanup_result.content_box,
        crop_offset=cleanup_result.crop_offset,
        sam_config=cfg["sam"],
    )
