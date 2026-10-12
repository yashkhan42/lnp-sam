"""Worker-side run preparation shared by the desktop app and remote server.

Importing this module never imports Qt, torch, SAM, or the science pipeline.
Model verification and pipeline work belong in a background worker.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile

from . import __version__


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
_STAGE_LABELS = {
    "cleanup": "Cleanup",
    "proposals": "SAM proposals",
    "graph": "Build graph",
    "resolve": "Resolve particles",
    "measure": "Measurements",
}


class RunCancelled(Exception):
    """The worker stopped at a cancellation boundary; partial outputs remain."""


def load_app_config(config_path, checkpoint=None) -> dict:
    """Load pipeline settings with an absolute checkpoint path.

    The default config lives at the repository root. Relative checkpoint paths
    in custom configs are relative to that config's directory. A launch-time
    override is relative to the caller's current directory.
    """
    import yaml

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("sam"), dict):
        raise ValueError("The config must contain a sam section.")
    value = checkpoint if checkpoint is not None else config["sam"].get("checkpoint")
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("Set sam.checkpoint in config.yaml to the SAM model file.")
    model_path = Path(value).expanduser()
    if not model_path.is_absolute() and checkpoint is None:
        model_path = config_path.parent / model_path
    config["sam"]["checkpoint"] = str(model_path.resolve())
    return config


def _configured_hash(config) -> str:
    value = config["sam"].get("checkpoint_sha256")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(
            "sam.checkpoint_sha256 must contain exactly 64 hexadecimal characters. "
            "Run shasum -a 256 <model-file> on macOS or sha256sum <model-file> "
            "on Linux and put the hash in config.yaml."
        )
    return value.lower()


def check_local_setup(config_path, checkpoint=None) -> dict:
    """Verify model bytes without importing torch, SAM, or the pipeline."""
    config = load_app_config(config_path, checkpoint)
    expected = _configured_hash(config)
    model_path = Path(config["sam"]["checkpoint"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SAM model file not found. Download it from {SAM_DOWNLOAD_URL} "
            f"to {model_path}."
        )
    digest = hashlib.sha256()
    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"SAM model file doesn't match the configured SHA-256. Download it "
            f"from {SAM_DOWNLOAD_URL} to {model_path}. "
            f"Expected {expected}; found {actual}."
        )
    return {"checkpoint": str(model_path), "checkpoint_sha256": actual}


def app_identity(config_path=DEFAULT_CONFIG, checkpoint=None) -> dict:
    """Portable app/model identity, including when local weights are absent.

    An unset model hash is returned as ``None`` so remote-only usage remains
    possible. Local setup and server startup still require a valid hash.
    """
    config = load_app_config(config_path, checkpoint)
    try:
        checkpoint_sha256 = _configured_hash(config)
    except ValueError:
        checkpoint_sha256 = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=2, check=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"version": __version__, "git_commit": commit,
            "checkpoint_sha256": checkpoint_sha256}


def _validated_inputs(images, modality, nm_per_px):
    paths = [Path(image).expanduser().resolve() for image in images]
    if not paths:
        raise ValueError("Add at least one image before running.")
    for path in paths:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path.name}. Use PNG, JPG or TIF.")
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
    modality = str(modality).upper()
    if modality not in ("TEM", "SEM"):
        raise ValueError("Modality must be TEM or SEM.")
    if nm_per_px is None or nm_per_px == "":
        scale = None
    else:
        try:
            scale = float(nm_per_px)
        except (TypeError, ValueError) as exc:
            raise ValueError("Enter a positive number, or leave blank for automatic scale detection.") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Enter a positive number, or leave blank for automatic scale detection.")
    return paths, modality, scale


def _unique_run_folder(runs_root=None) -> Path:
    root = Path(runs_root).expanduser() if runs_root is not None else Path.home() / "LNP-SAM runs"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = 1
    while True:
        folder = root / (timestamp if suffix == 1 else f"{timestamp}_{suffix}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            suffix += 1


def _panel_stem(stem) -> str:
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_stem).strip("_-")[:100] or "image"
    if sanitized.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        sanitized = f"image_{sanitized}"
    return sanitized


def _prepare_inputs(paths, modality, scale, folder):
    inputs = folder / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    used = set()
    rows = []
    for path in paths:
        base = _panel_stem(path.stem)
        panel_id = base
        suffix = 2
        while panel_id in used:
            panel_id = f"{base}_{suffix}"
            suffix += 1
        used.add(panel_id)
        copied = inputs / f"{panel_id}{path.suffix.lower()}"
        if path != copied:
            shutil.copy2(path, copied)
        rows.append({"panel_id": panel_id, "paper_id": "app", "image_path": str(copied),
                     "modality": modality, "nm_per_px": "" if scale is None else scale})
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("panel_id", "paper_id", "image_path", "modality", "nm_per_px"))
        writer.writeheader()
        writer.writerows(rows)


def prepare_run(images, modality, nm_per_px, runs_root=None) -> Path:
    """Create an isolated batch with unique, safe panel IDs and copied images."""
    paths, modality, scale = _validated_inputs(images, modality, nm_per_px)
    folder = _unique_run_folder(runs_root)
    _prepare_inputs(paths, modality, scale, folder)
    return folder


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RunCancelled("Cancelled")


def _load_pipeline():
    # Called only in a worker, after the MPS fallback environment is configured.
    from . import proposals, run
    return run.run_cleanup_stage, proposals.select_device


def device_label(device) -> str:
    if str(device).startswith("cuda"):
        return "NVIDIA GPU"
    return "Apple GPU" if device == "mps" else "CPU"


def _image_metadata(folder):
    manifest = {}
    manifest_path = folder / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            manifest = {row["panel_id"]: row for row in csv.DictReader(stream)}
    images = []
    for path in sorted((folder / "measure").glob("*__panel.json")):
        with path.open(encoding="utf-8") as stream:
            panel = json.load(stream)
        panel_id = panel.get("panel_id", path.name.removesuffix("__panel.json"))
        record = {"panel_id": panel_id, "cal_source": panel.get("cal_source"),
                  "nm_per_px": panel.get("nm_per_px"), "n_particles": panel.get("n_particles")}
        entry = manifest.get(panel_id, {})
        if entry.get("image_path"):
            record["filename"] = Path(entry["image_path"]).name
        cleanup_path = folder / "cleanup" / f"{panel_id}.json"
        if cleanup_path.is_file():
            with cleanup_path.open(encoding="utf-8") as stream:
                cleanup = json.load(stream)
            record.update(width=int(cleanup["width"]), height=int(cleanup["height"]))
        images.append(record)
    return images


def run_local(
    images, modality, nm_per_px, config_path, checkpoint=None, device="auto",
    runs_root=None, progress=None, status=None, cancel_event=None, run_folder=None,
    where="this computer",
) -> Path:
    """Run the unchanged pipeline in a worker; retry an Apple GPU failure once.

    ``run_folder`` may be a batch prepared by :func:`prepare_run`, or an empty
    server job destination. Cancelling and callback exceptions always propagate.
    """
    import yaml

    callback_failed = False

    def report(message):
        nonlocal callback_failed
        if status is not None:
            try:
                status(message)
            except Exception:
                callback_failed = True
                raise

    _check_cancel(cancel_event)
    report("Checking SAM model…")
    setup = check_local_setup(config_path, checkpoint)
    _check_cancel(cancel_event)
    config = load_app_config(config_path, checkpoint)
    config["sam"]["checkpoint_sha256"] = setup["checkpoint_sha256"]
    if run_folder is None:
        folder = prepare_run(images, modality, nm_per_px, runs_root)
    else:
        folder = Path(run_folder).expanduser().resolve()
        if not (folder / "manifest.csv").is_file():
            paths, modality, scale = _validated_inputs(images, modality, nm_per_px)
            folder.mkdir(parents=True, exist_ok=True)
            _prepare_inputs(paths, modality, scale, folder)
    config_yaml = yaml.safe_dump(config, sort_keys=False)
    saved_config = folder / "config.yaml"
    saved_config.write_text(config_yaml, encoding="utf-8")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    pipeline, select_device = _load_pipeline()
    resolved_device = select_device(device)
    _check_cancel(cancel_event)

    def on_progress(index, total, panel, stage):
        nonlocal callback_failed
        _check_cancel(cancel_event)
        try:
            report(f"{device_label(resolved_device)} · image {index}/{total} · {_STAGE_LABELS.get(stage, stage)}")
            if progress is not None:
                progress(index, total, panel, stage)
        except Exception:
            callback_failed = True
            raise
        _check_cancel(cancel_event)

    def execute():
        report(device_label(resolved_device))
        pipeline(
            folder / "manifest.csv", saved_config, folder,
            run_step2=True, run_step3=True, run_step4=True, run_step6=True,
            device=resolved_device, progress=on_progress,
        )

    fallback = False
    try:
        execute()
    except RunCancelled:
        raise
    except Exception:
        _check_cancel(cancel_event)
        if resolved_device != "mps" or callback_failed:
            raise
        report("Apple GPU failed, retrying on CPU")
        fallback = True
        resolved_device = "cpu"
    if fallback:
        # Exit the exception scope first so the failed GPU model is collectable.
        gc.collect()
        _check_cancel(cancel_event)
        execute()
    _check_cancel(cancel_event)
    identity = app_identity(config_path, checkpoint)
    info = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **identity, "where": where, "device": resolved_device,
        "config": config, "config_yaml": config_yaml,
        "checkpoint_sha256": setup["checkpoint_sha256"], "images": _image_metadata(folder),
    }
    if fallback:
        info["fallback_from"] = "mps"
    (folder / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return folder


def export_results_zip(run_folder, destination) -> Path:
    """Export scientific results only, with paths shared by both compute modes."""
    folder = Path(run_folder).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    required = [folder / name for name in ("particles_all.csv", "panels_all.csv", "run_info.json")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Results are incomplete: {path.name} is missing.")
    measure = folder / "measure"
    if not measure.is_dir():
        raise FileNotFoundError("Results are incomplete: measure/ is missing.")
    files = required + sorted(path for path in measure.rglob("*") if path.is_file())
    if destination in files or destination.is_relative_to(measure):
        raise ValueError("Save the zip to a separate file from the scientific results.")
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError(f"Results contain an unsafe file: {path.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed export leaves an existing user-selected destination intact.
    with tempfile.NamedTemporaryFile(prefix=".lnpsam-export-", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(folder).as_posix())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
