"""Client/server integration and archive safety, without scientific imports."""

import csv
import hashlib
import io
import json
from pathlib import Path
import socket
import stat
import threading
import time
import zipfile

import pytest
import requests
import uvicorn
import yaml

from lnpsam import app_runtime
from lnpsam.remote import RemoteClient, RemoteError, _extract_results
from lnpsam import remote as remote_mod
from lnpsam.server import create_app


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fake SAM")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"sam": {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}))

    def pipeline(manifest, config_path, folder, **kwargs):
        with Path(manifest).open() as stream:
            rows = list(csv.DictReader(stream))
        measure = Path(folder) / "measure"
        measure.mkdir()
        particles = "panel_id,particle_id,d_px,d_nm,in_stats\n"
        panels = "panel_id,n_particles,nm_per_px\n"
        for index, row in enumerate(rows, 1):
            panel_id = row["panel_id"]
            for stage in ("cleanup", "proposals", "graph", "resolve", "measure"):
                kwargs["progress"](index, len(rows), panel_id, stage)
            calibrated = bool(row["nm_per_px"])
            scale = float(row["nm_per_px"]) if calibrated else None
            metadata = {"panel_id": panel_id, "n_particles": 1, "nm_per_px": scale,
                        "cal_source": "manifest" if calibrated else "missing", "stats_unit": "nm" if calibrated else "px"}
            (measure / f"{panel_id}__panel.json").write_text(json.dumps(metadata))
            (measure / f"{panel_id}__overlay.png").write_bytes(Path(row["image_path"]).read_bytes())
            particles += f"{panel_id},1,4.2,{4.2 * scale if scale else ''},True\n"
            panels += f"{panel_id},1,{scale or ''}\n"
        (Path(folder) / "particles_all.csv").write_text(particles)
        (Path(folder) / "panels_all.csv").write_text(panels)

    monkeypatch.setattr(app_runtime, "_load_pipeline", lambda: (pipeline, lambda requested: "cpu"))
    return config


@pytest.fixture
def server_factory(tmp_path):
    running = []

    def start(*, runner, config_path=None):
        app = create_app("secret", runner=runner, config_path=config_path, jobs_root=tmp_path / f"jobs-{len(running)}",
                         identity={"version": "0.1.0", "git_commit": None, "checkpoint_sha256": "a" * 64, "device": "cpu"})
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False, ws="none"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        running.append((server, thread, listener))
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        return f"http://127.0.0.1:{listener.getsockname()[1]}", app

    yield start
    for server, thread, listener in running:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()


def test_live_remote_outputs_match_shared_local_runtime(tmp_path, fake_runtime, server_factory):
    first = tmp_path / "A!!! sample.png"
    second = tmp_path / "A!!! sample.jpg"
    first.write_bytes(b"first fake image")
    second.write_bytes(b"second fake image")
    images = [first, second]
    local = app_runtime.run_local(images, "TEM", 1.2, fake_runtime, device="cpu", runs_root=tmp_path / "local")
    address, app = server_factory(runner=app_runtime.run_local, config_path=fake_runtime)
    client = RemoteClient(address, "secret", poll_interval=0.01)
    assert client.info()["device"] == "cpu"
    statuses = []
    progress = []
    remote = client.run(images, "TEM", 1.2, tmp_path / "remote", status=statuses.append,
                        progress=lambda *value: progress.append(value))
    assert remote == tmp_path / "remote"
    assert "Uploading images…" in statuses and "Downloading results…" in statuses
    assert progress[-1] == (2, 2, "A_sample_2", "measure")
    for name in ["particles_all.csv", "panels_all.csv", *[str(path.relative_to(local)) for path in (local / "measure").iterdir()]]:
        assert (remote / name).read_bytes() == (local / name).read_bytes()
    info = json.loads((remote / "run_info.json").read_text())
    local_info = json.loads((local / "run_info.json").read_text())
    for name in ("version", "git_commit", "device", "checkpoint_sha256", "images", "config"):
        assert info[name] == local_info[name]
    assert info["where"] == address
    assert "secret" not in (remote / "run_info.json").read_text()
    assert all(job.files_deleted for job in app.state.jobs.jobs.values())


def test_wrong_token_and_unreachable_are_plain_language(server_factory, tmp_path):
    address, _ = server_factory(runner=lambda *args, **kwargs: None)
    with pytest.raises(RemoteError, match="The server rejected the token"):
        RemoteClient(address, "wrong").info()

    class Disconnected:
        def request(self, *args, **kwargs):
            raise requests.ConnectionError("Technical details must not be shown")

    with pytest.raises(RemoteError, match="Can't reach the server"):
        RemoteClient(address, "secret", session=Disconnected()).info()


def test_remote_cancellation_reaches_running_server(server_factory, tmp_path):
    entered = threading.Event()
    event = threading.Event()
    release = threading.Event()

    def runner(images, modality, scale, config_path, **kwargs):
        entered.set()
        assert release.wait(5)
        kwargs["progress"](1, 1, "sample", "proposals")
        pytest.fail("Cancelled jobs must stop at a stage boundary")

    address, app = server_factory(runner=runner)
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    errors = []

    def run():
        try:
            RemoteClient(address, "secret", poll_interval=0.01).run([image], "TEM", None, tmp_path / "cancelled", cancel_event=event)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(3)
        event.set()
        thread.join(timeout=3)
        assert len(errors) == 1 and isinstance(errors[0], app_runtime.RunCancelled)
        job = next(iter(app.state.jobs.jobs.values()))
        assert job.cancel_event.is_set()
        release.set()
        deadline = time.monotonic() + 3
        while job.state == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.state == "cancelled" and not job.folder.exists()
    finally:
        event.set()
        release.set()
        thread.join(timeout=3)


@pytest.mark.parametrize("name", ["../escaped.csv", "/tmp/escaped.csv", "measure/../../escaped.csv",
                                  "measure\\escaped.csv", "C:/escaped.csv", "inputs/private.png"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "unsafe")
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(RemoteError, match="unsafe"):
        _extract_results(archive, destination)
    assert not list(destination.iterdir())


def test_archive_rejects_links_and_incomplete_outputs(tmp_path):
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        link = zipfile.ZipInfo("measure/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        stream.writestr(link, "/etc/passwd")
    with pytest.raises(RemoteError, match="unsafe"):
        _extract_results(archive, tmp_path / "extracted")
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("particles_all.csv", "")
    with pytest.raises(RemoteError, match="incomplete"):
        _extract_results(archive, tmp_path / "extracted")


def test_address_validation_does_not_accept_embedded_secrets():
    for address in ("", "lab-gpu:8765", "file:///etc/passwd", "http://user:secret@lab-gpu", "http://lab-gpu?token=secret"):
        with pytest.raises(RemoteError):
            RemoteClient(address, "secret")


def test_interrupted_download_reports_connection_failure_and_removes_temporary_files(tmp_path):
    calls = []

    class Response:
        status_code = 200

        def __init__(self, record=None):
            self.record = record

        def json(self):
            return self.record

        def close(self):
            pass

        def iter_content(self, chunk_size):
            yield b"partial zip"
            raise requests.ConnectionError("Stream disconnected")

    class Session:
        def request(self, method, url, **kwargs):
            calls.append(method)
            if method == "POST":
                return Response({"job_id": "test-job"})
            if method == "DELETE":
                return Response({"state": "cancelled"})
            if url.endswith("/result"):
                return Response()
            return Response({"state": "done"})

    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    destination = tmp_path / "output"
    with pytest.raises(RemoteError, match="Can't reach the server"):
        RemoteClient("http://test-server", "secret", session=Session()).run([image], "TEM", None, destination)
    assert not list(destination.iterdir())
    assert calls[-1] == "DELETE"


@pytest.mark.parametrize("boundary", ["download", "extract", "commit"])
def test_cancellation_during_result_retrieval_leaves_no_partial_results(tmp_path, monkeypatch, boundary):
    event = threading.Event()
    calls = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("particles_all.csv", "panel_id,particle_id\nsample,1\n")
        archive.writestr("panels_all.csv", "panel_id,n_particles\nsample,1\n")
        archive.writestr("run_info.json", "{}")
        archive.writestr("measure/sample__overlay.png", b"x" * (2 * 1024 * 1024))

    class Response:
        status_code = 200

        def __init__(self, record=None):
            self.record = record

        def json(self):
            return self.record

        def close(self):
            pass

        def iter_content(self, chunk_size):
            yield buffer.getvalue()
            if boundary == "download":
                # A cancellation arriving after the last yielded chunk must
                # still prevent extraction and publication of the results.
                event.set()

    class Session:
        def request(self, method, url, **kwargs):
            calls.append(method)
            if method == "POST":
                return Response({"job_id": "test-job"})
            if method == "DELETE":
                return Response({"state": "cancelled"})
            return Response() if url.endswith("/result") else Response({"state": "done"})

    if boundary == "extract":
        original_read = zipfile.ZipExtFile.read

        def read(member, *args, **kwargs):
            chunk = original_read(member, *args, **kwargs)
            if member.name.startswith("measure/"):
                event.set()
            return chunk

        monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    if boundary == "commit":
        original_move = remote_mod.shutil.move
        triggered = False

        def move(source, destination, *args, **kwargs):
            nonlocal triggered
            result = original_move(source, destination, *args, **kwargs)
            if not triggered:
                triggered = True
                event.set()
            return result

        monkeypatch.setattr(remote_mod.shutil, "move", move)
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "manifest.csv").write_text("Prepared input metadata must survive cancellation")
    with pytest.raises(app_runtime.RunCancelled):
        RemoteClient("http://test-server", "secret", session=Session()).run(
            [image], "TEM", None, destination, cancel_event=event)
    assert [path.name for path in destination.iterdir()] == ["manifest.csv"]
    assert calls[-1] == "DELETE"
