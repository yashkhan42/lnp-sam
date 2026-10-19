"""HTTP client for the optional LNP-SAM server (no scientific imports)."""

from contextlib import ExitStack
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import threading
import time
from urllib.parse import urlsplit
import zipfile


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024 * 1024
# How long a running job's status checks may keep failing before the app
# gives up. A busy server (e.g. loading models) or a brief network drop
# should not cost the user their run.
POLL_GRACE_SECONDS = 5 * 60


class RemoteError(RuntimeError):
    """A connection or server failure suitable for display in the GUI."""


class _Unreachable(RemoteError):
    """A transient failure: no connection, a timeout, or a gateway error."""


def _cancelled():
    from .app_runtime import RunCancelled

    return RunCancelled("Cancelled")


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise _cancelled()


def _extract_results(archive: Path, destination: Path, cancel_event=None) -> None:
    """Extract only final outputs, with traversal, links and size checks."""
    required = {"particles_all.csv", "panels_all.csv", "run_info.json"}
    _check_cancel(cancel_event)
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 100_000 or sum(item.file_size for item in entries) > MAX_RESULT_BYTES:
            raise RemoteError("The server results are too large to unpack.")
        names = set()
        for item in entries:
            _check_cancel(cancel_event)
            path = PurePosixPath(item.filename)
            if (
                not item.filename
                or not path.parts
                or "\\" in item.filename
                or ":" in item.filename
                or path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(item.external_attr >> 16)
                or (path.parts[0] != "measure" and str(path) not in required)
                or str(path) in names
            ):
                raise RemoteError("The server returned an unsafe results archive.")
            names.add(str(path))
        if not required.issubset(names) or not any(name.startswith("measure/") for name in names):
            raise RemoteError("The server returned an incomplete results archive.")
        # Validation finishes before any files are written.
        for item in entries:
            _check_cancel(cancel_event)
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as member, target.open("wb") as output:
                while chunk := member.read(1024 * 1024):
                    _check_cancel(cancel_event)
                    output.write(chunk)
    _check_cancel(cancel_event)
    try:
        metadata = json.loads((destination / "run_info.json").read_text())
    except (OSError, ValueError) as exc:
        raise RemoteError("The server returned invalid run information.") from exc
    if not isinstance(metadata, dict):
        raise RemoteError("The server returned invalid run information.")
    _check_cancel(cancel_event)


class RemoteClient:
    """Upload, monitor and retrieve one run. Call from a worker thread."""

    def __init__(self, address: str, token: str, *, session=None, poll_interval: float = 1.0,
                 poll_grace: float = POLL_GRACE_SECONDS):
        parsed = urlsplit(address.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise RemoteError("Enter a server address such as http://localhost:8765.")
        if parsed.query or parsed.fragment:
            raise RemoteError("The server address must not contain a query or fragment.")
        if not token:
            raise RemoteError("Enter the server's access token.")
        self.address = address.strip().rstrip("/")
        self._token = token
        self.poll_interval = max(0.01, poll_interval)
        self.poll_grace = max(0.0, poll_grace)
        if session is None:
            import requests

            session = requests.Session()
        self._session = session

    def _request(self, method: str, path: str, **kwargs):
        import requests

        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = self._session.request(
                method, self.address + path, headers=headers, timeout=(5, 60),
                allow_redirects=False, **kwargs,
            )
        except requests.RequestException:
            # Do not expose a requests traceback containing request headers.
            raise _Unreachable("Can't reach the server.") from None
        if response.status_code == 401:
            response.close()
            raise RemoteError("The server rejected the token.")
        if response.status_code in (502, 503, 504):
            response.close()
            raise _Unreachable("Can't reach the server.")
        if response.status_code == 413:
            response.close()
            raise RemoteError("Uploads are limited to 200 MB. Remove some images and try again.")
        if not 200 <= response.status_code < 300:
            status_code = response.status_code
            response.close()
            raise RemoteError(f"The server could not complete the request (HTTP {status_code}).")
        return response

    def _json(self, method: str, path: str, **kwargs) -> dict:
        response = self._request(method, path, **kwargs)
        try:
            result = response.json()
        except ValueError:
            raise RemoteError("The server returned an invalid response.") from None
        finally:
            response.close()
        if not isinstance(result, dict):
            raise RemoteError("The server returned an invalid response.")
        return result

    def info(self) -> dict:
        return self._json("GET", "/info")

    def cancel(self, job_id: str) -> None:
        self._json("DELETE", f"/jobs/{job_id}")

    def run(
        self, images, modality, nm_per_px, run_folder, progress=None,
        status=None, cancel_event=None,
    ) -> Path:
        paths = [Path(item) for item in images]
        if not paths:
            raise RemoteError("Add at least one image before running.")
        if modality not in {"TEM", "SEM"}:
            raise RemoteError("Choose TEM or SEM.")
        if nm_per_px is not None and (not math.isfinite(float(nm_per_px)) or float(nm_per_px) <= 0):
            raise RemoteError("Enter a positive number for nm per pixel, or leave it empty.")
        if sum(item.stat().st_size for item in paths) > MAX_UPLOAD_BYTES:
            raise RemoteError("Uploads are limited to 200 MB. Remove some images and try again.")
        event = cancel_event if cancel_event is not None else threading.Event()
        if event.is_set():
            raise _cancelled()
        folder = Path(run_folder)
        folder.mkdir(parents=True, exist_ok=True)
        job_id = None
        try:
            if status:
                status("Uploading images…")
            with ExitStack() as stack:
                files = [("files", (path.name, stack.enter_context(path.open("rb")), "application/octet-stream")) for path in paths]
                job = self._json(
                    "POST", "/jobs", files=files,
                    data={"modality": modality, "nm_per_px": "" if nm_per_px is None else str(nm_per_px)},
                )
            job_id = job.get("job_id")
            if not isinstance(job_id, str) or not job_id or not all(char.isalnum() or char in "_-" for char in job_id):
                raise RemoteError("The server returned an invalid job identifier.")
            previous_progress = None
            previous_status = None
            unreachable_since = None
            while True:
                if event.is_set():
                    try:
                        self.cancel(job_id)
                    except RemoteError:
                        pass  # Still report a cancellation; the server expires the job.
                    raise _cancelled()
                try:
                    job = self._json("GET", f"/jobs/{job_id}")
                except _Unreachable:
                    # The job keeps running on the server; keep checking for a while.
                    now = time.monotonic()
                    if unreachable_since is None:
                        unreachable_since = now
                    if now - unreachable_since >= self.poll_grace:
                        minutes = max(1, round(self.poll_grace / 60))
                        raise RemoteError(
                            f"Lost contact with the server for {minutes} "
                            f"minute{'s' if minutes != 1 else ''}. Check the connection and run again."
                        ) from None
                    text = "Server not responding · retrying…"
                    if status and text != previous_status:
                        status(text)
                        previous_status = text
                    # Re-send the stage on recovery so the GUI replaces this message.
                    previous_progress = None
                    event.wait(self.poll_interval)
                    continue
                unreachable_since = None
                state = job.get("state")
                latest = job.get("progress")
                if latest and progress:
                    if isinstance(latest, dict):
                        value = tuple(latest.get(key) for key in ("image_index", "n_images", "panel_id", "stage"))
                    elif isinstance(latest, (tuple, list)) and len(latest) == 4:
                        value = tuple(latest)
                    else:
                        raise RemoteError("The server returned invalid progress information.")
                    if value != previous_progress:
                        progress(*value)
                        previous_progress = value
                if state == "cancelled":
                    raise _cancelled()
                if state == "failed":
                    raise RemoteError(str(job.get("error") or "The server could not process these images."))
                if state == "done":
                    break
                if state not in {"queued", "running"}:
                    raise RemoteError("The server returned an invalid job state.")
                position = job.get("queue_position")
                text = (f"Queued · position {position}" if position is not None else "Queued") if state == "queued" else "Running on the server…"
                # A stage callback already provides the more useful status.
                if state == "running" and latest:
                    text = None
                if status and text != previous_status:
                    if text:
                        status(text)
                    previous_status = text
                event.wait(self.poll_interval)
            if event.is_set():
                self.cancel(job_id)
                raise _cancelled()
            if status:
                status("Downloading results…")
            with tempfile.TemporaryDirectory(prefix=".remote-", dir=folder) as temporary:
                workspace = Path(temporary)
                archive = workspace / "results.zip"
                response = self._request("GET", f"/jobs/{job_id}/result", stream=True)
                try:
                    count = 0
                    with archive.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if event.is_set():
                                raise _cancelled()
                            count += len(chunk)
                            if count > MAX_RESULT_BYTES:
                                raise RemoteError("The server results are too large to download.")
                            stream.write(chunk)
                finally:
                    response.close()
                _check_cancel(event)
                extracted = workspace / "extracted"
                extracted.mkdir()
                try:
                    _extract_results(archive, extracted, event)
                except RemoteError:
                    raise
                except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as exc:
                    raise RemoteError("The server returned a damaged results archive.") from exc
                entries = list(extracted.iterdir())
                if any((folder / entry.name).exists() for entry in entries):
                    raise RemoteError("This results folder already contains a run. Choose a new folder.")
                installed = []
                try:
                    for entry in entries:
                        _check_cancel(event)
                        shutil.move(str(entry), folder / entry.name)
                        installed.append(entry.name)
                    _check_cancel(event)
                except BaseException:
                    # Keep a cancelled/failed download from exposing half a run.
                    for name in reversed(installed):
                        shutil.move(str(folder / name), extracted / name)
                    raise
            return folder
        except BaseException as exc:
            if job_id:
                try:
                    self.cancel(job_id)
                except Exception:
                    pass
            import requests

            if isinstance(exc, requests.RequestException):
                raise RemoteError("Can't reach the server.") from None
            raise
