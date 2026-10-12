"""Authenticated, single-worker remote execution for the desktop app.

FastAPI and the scientific runtime are imported only when the server starts.
"""

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import threading
import time
from uuid import uuid4

from .remote import MAX_UPLOAD_BYTES


RETENTION_SECONDS = 24 * 60 * 60
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class _UploadTooLarge(RuntimeError):
    pass


class _RequestGuard:
    """Protect every route and bound multipart bodies, including chunked ones."""

    def __init__(self, app, token, max_upload_bytes):
        self.app = app
        self.token = f"Bearer {token}".encode()
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))

        async def reject(status, detail):
            body = json.dumps({"detail": detail}).encode()
            await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})

        if not secrets.compare_digest(headers.get(b"authorization", b""), self.token):
            return await reject(401, "The server rejected the token.")
        limited = scope["method"] == "POST" and scope["path"] == "/jobs"
        if not limited:
            return await self.app(scope, receive, send)
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await reject(400, "Invalid Content-Length.")
        if content_length > self.max_upload_bytes:
            return await reject(413, "Uploads are limited to 200 MB.")
        total = 0
        too_large = False
        replaced = False

        async def limited_receive():
            nonlocal total, too_large
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_upload_bytes:
                    too_large = True
                    raise _UploadTooLarge("Uploads are limited to 200 MB.")
            return message

        async def limited_send(message):
            nonlocal replaced
            if too_large:
                if message["type"] == "http.response.start" and not replaced:
                    replaced = True
                    await reject(413, "Uploads are limited to 200 MB.")
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except _UploadTooLarge:
            if not replaced:
                await reject(413, "Uploads are limited to 200 MB.")


@dataclass
class _Job:
    job_id: str
    folder: Path
    images: list[Path]
    modality: str
    nm_per_px: float | None
    where: str
    created: float = field(default_factory=time.time)
    state: str = "queued"
    progress: dict | None = None
    error: str | None = None
    result: Path | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    download_started: bool = False
    files_deleted: bool = False


class _Jobs:
    def __init__(self, root, runner, config_path, checkpoint, device, retention_seconds):
        self.root = Path(root)
        self.runner = runner
        self.config_path = config_path
        self.checkpoint = checkpoint
        self.device = device
        self.retention_seconds = retention_seconds
        self.condition = threading.Condition()
        self.jobs = {}
        self.queue = []
        self.stopping = False
        self.thread = None

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._worker, name="lnpsam-jobs", daemon=True)
        self.thread.start()

    def stop(self):
        with self.condition:
            self.stopping = True
            for job in self.jobs.values():
                if job.state in {"queued", "running"}:
                    job.cancel_event.set()
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=5)

    def _delete_files(self, job):
        shutil.rmtree(job.folder, ignore_errors=True)
        job.files_deleted = True

    def purge_expired(self):
        with self.condition:
            now = time.time()
            for job_id, job in list(self.jobs.items()):
                if now - job.created < self.retention_seconds:
                    continue
                if job.state == "running":
                    job.cancel_event.set()
                    continue
                self._delete_files(job)
                self.jobs.pop(job_id)
                if job_id in self.queue:
                    self.queue.remove(job_id)
            # Remove orphan folders left by a previous server process.
            for folder in self.root.iterdir():
                if folder.is_dir() and folder.name not in self.jobs and now - folder.stat().st_mtime >= self.retention_seconds:
                    shutil.rmtree(folder, ignore_errors=True)

    def enqueue(self, job):
        with self.condition:
            self.jobs[job.job_id] = job
            self.queue.append(job.job_id)
            self.condition.notify_all()

    def snapshot(self, job):
        with self.condition:
            position = self.queue.index(job.job_id) + 1 if job.job_id in self.queue else 0
            return {"job_id": job.job_id, "state": job.state, "queue_position": position,
                    "progress": job.progress, "error": job.error,
                    "cancel_requested": job.cancel_event.is_set()}

    def cancel(self, job):
        with self.condition:
            job.cancel_event.set()
            if job.state != "running":
                if job.job_id in self.queue:
                    self.queue.remove(job.job_id)
                job.state = "cancelled"
                self._delete_files(job)
            self.condition.notify_all()

    def finish_download(self, job):
        with self.condition:
            self._delete_files(job)

    def _worker(self):
        from .app_runtime import RunCancelled, export_results_zip

        while True:
            self.purge_expired()
            with self.condition:
                while not self.queue and not self.stopping:
                    self.condition.wait(timeout=min(60, self.retention_seconds))
                    if not self.queue:
                        self.purge_expired()
                if self.stopping:
                    return
                job = self.jobs[self.queue.pop(0)]
                if job.cancel_event.is_set():
                    job.state = "cancelled"
                    self._delete_files(job)
                    continue
                job.state = "running"

            def record_progress(image_index, n_images, panel_id, stage):
                if job.cancel_event.is_set():
                    raise RunCancelled("Cancelled")
                with self.condition:
                    job.progress = {"image_index": image_index, "n_images": n_images,
                                    "panel_id": panel_id, "stage": stage}

            try:
                results = self.runner(
                    job.images, job.modality, job.nm_per_px, self.config_path,
                    checkpoint=self.checkpoint, device=self.device,
                    run_folder=job.folder / "results", progress=record_progress,
                    cancel_event=job.cancel_event, where=job.where,
                )
                if job.cancel_event.is_set():
                    raise RunCancelled("Cancelled")
                archive = export_results_zip(results, job.folder / "results.zip")
                with self.condition:
                    if job.cancel_event.is_set():
                        raise RunCancelled("Cancelled")
                    job.result = archive
                    job.state = "done"
            except Exception as exc:
                with self.condition:
                    if job.cancel_event.is_set() or isinstance(exc, RunCancelled):
                        job.state = "cancelled"
                        self._delete_files(job)
                    else:
                        job.state = "failed"
                        job.error = str(exc) or "The server could not process these images."


def create_app(
    token, *, config_path=None, checkpoint=None, device="auto", jobs_root=None,
    runner=None, identity=None, retention_seconds=RETENTION_SECONDS,
    max_upload_bytes=MAX_UPLOAD_BYTES,
):
    """Create a server; runner/identity injection supports checkpoint-free tests."""
    if not token:
        raise ValueError("A nonempty access token is required.")
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    repository = Path(__file__).resolve().parents[1]
    config_path = Path(config_path) if config_path else repository / "config.yaml"
    jobs_root = Path(jobs_root) if jobs_root else Path.home() / "LNP-SAM server jobs"
    jobs = _Jobs(jobs_root, runner, config_path, checkpoint, device, retention_seconds)

    @asynccontextmanager
    async def lifespan(app):
        # A normal server verifies its own model before accepting any request.
        if runner is None:
            from .app_runtime import app_identity, check_local_setup, run_local

            check_local_setup(config_path, checkpoint)
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            from .proposals import select_device

            jobs.device = select_device(device)
            jobs.runner = run_local
            app.state.identity = {**app_identity(config_path, checkpoint), "device": jobs.device}
        else:
            from . import __version__

            app.state.identity = identity or {"version": __version__, "git_commit": None,
                                               "checkpoint_sha256": None, "device": device}
        jobs.start()
        try:
            yield
        finally:
            jobs.stop()

    app = FastAPI(title="LNP-SAM server", lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(_RequestGuard, token=token, max_upload_bytes=max_upload_bytes)

    def find_job(job_id):
        jobs.purge_expired()
        with jobs.condition:
            job = jobs.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "This job was not found or has expired.")
        return job

    @app.get("/info")
    def info():
        return app.state.identity

    @app.post("/jobs", status_code=202)
    async def submit(request: Request, files: list[UploadFile] = File(...), modality: str = Form(...), nm_per_px: str = Form("")):
        if modality not in {"TEM", "SEM"}:
            raise HTTPException(422, "Choose TEM or SEM.")
        try:
            scale = float(nm_per_px) if nm_per_px.strip() else None
            if scale is not None and (not math.isfinite(scale) or scale <= 0):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "Enter a positive number for nm per pixel, or leave it empty.") from None
        if not files:
            raise HTTPException(422, "Add at least one image before running.")
        job_id = uuid4().hex
        folder = jobs.root / job_id
        uploads = folder / "uploads"
        uploads.mkdir(parents=True)
        paths = []
        # Use the same panel naming as local runs, including punctuation,
        # Unicode and duplicate stems with different file extensions.
        from .app_runtime import _panel_stem

        used_stems = set()
        total = 0
        try:
            for file in files:
                original = Path((file.filename or "").replace("\\", "/")).name
                suffix = Path(original).suffix.lower()
                if suffix not in _ALLOWED_EXTENSIONS:
                    raise HTTPException(422, "Choose PNG, JPG or TIF image files.")
                stem = _panel_stem(Path(original).stem)
                unique_stem = stem
                index = 2
                while unique_stem in used_stems:
                    unique_stem = f"{stem}_{index}"
                    index += 1
                used_stems.add(unique_stem)
                name = f"{unique_stem}{suffix}"
                path = uploads / name
                with path.open("wb") as stream:
                    while chunk := await file.read(1024 * 1024):
                        total += len(chunk)
                        if total > max_upload_bytes:
                            raise HTTPException(413, "Uploads are limited to 200 MB.")
                        stream.write(chunk)
                paths.append(path)
            job = _Job(job_id, folder, paths, modality, scale, str(request.base_url).rstrip("/"))
            jobs.enqueue(job)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            for file in files:
                await file.close()
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        return jobs.snapshot(find_job(job_id))

    @app.delete("/jobs/{job_id}")
    def cancel_job(job_id: str):
        job = find_job(job_id)
        jobs.cancel(job)
        return jobs.snapshot(job)

    @app.get("/jobs/{job_id}/result")
    def result(job_id: str):
        job = find_job(job_id)
        with jobs.condition:
            if job.state != "done":
                raise HTTPException(409, "This job has not completed.")
            if job.files_deleted or job.download_started or job.result is None:
                raise HTTPException(410, "These results have already been downloaded or expired.")
            job.download_started = True
            return FileResponse(job.result, media_type="application/zip", filename="results.zip",
                                background=BackgroundTask(jobs.finish_download, job))

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the authenticated LNP-SAM analysis server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("LNPSAM_SERVER_TOKEN"), help="Access token (or LNPSAM_SERVER_TOKEN).")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--jobs-root", type=Path)
    args = parser.parse_args(argv)
    if not args.token:
        parser.error("Supply --token or set LNPSAM_SERVER_TOKEN.")
    import uvicorn

    app = create_app(args.token, config_path=args.config, checkpoint=args.checkpoint,
                     device=args.device, jobs_root=args.jobs_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
