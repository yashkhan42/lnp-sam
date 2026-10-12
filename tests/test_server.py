"""The remote server contract without loading torch or a SAM checkpoint."""

import io
import json
from pathlib import Path
import threading
import time
import zipfile

from fastapi.testclient import TestClient
import pytest

from lnpsam.server import create_app


AUTH = {"Authorization": "Bearer test-access-token"}


def write_fake_results(images, modality, nm_per_px, config_path, **kwargs):
    folder = Path(kwargs["run_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    measure = folder / "measure"
    measure.mkdir()
    kwargs["progress"](1, len(images), "sample", "measure")
    (measure / "sample__panel.json").write_text(json.dumps({"panel_id": "sample", "n_particles": 1}))
    (measure / "sample__overlay.png").write_bytes(b"fake overlay")
    (folder / "particles_all.csv").write_text("panel_id,particle_id,d_px,in_stats\nsample,1,4.2,True\n")
    (folder / "panels_all.csv").write_text("panel_id,n_particles\nsample,1\n")
    (folder / "run_info.json").write_text(json.dumps({"where": kwargs["where"], "device": "cpu"}))
    return folder


def submit(client, filename="sample.png", content=b"fake image"):
    response = client.post("/jobs", headers=AUTH, files={"files": (filename, content)}, data={"modality": "TEM", "nm_per_px": "1.2"})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def poll_terminal(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        record = response.json()
        if record["state"] in {"done", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    pytest.fail("Job did not finish")


def test_authentication_covers_every_endpoint(tmp_path):
    app = create_app("test-access-token", runner=write_fake_results, jobs_root=tmp_path)
    with TestClient(app) as client:
        for method, path in [("GET", "/info"), ("POST", "/jobs"), ("GET", "/jobs/missing"),
                             ("GET", "/jobs/missing/result"), ("DELETE", "/jobs/missing"),
                             ("GET", "/docs"), ("GET", "/openapi.json")]:
            assert client.request(method, path).status_code == 401
            assert client.request(method, path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/info", headers=AUTH).json()["version"] == "0.1.0"


def test_submit_progress_download_and_remove_files(tmp_path):
    app = create_app("test-access-token", runner=write_fake_results, jobs_root=tmp_path)
    with TestClient(app) as client:
        job_id = submit(client)
        record = poll_terminal(client, job_id)
        assert record["state"] == "done", record
        assert record["progress"] == {"image_index": 1, "n_images": 1, "panel_id": "sample", "stage": "measure"}
        response = client.get(f"/jobs/{job_id}/result", headers=AUTH)
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert set(archive.namelist()) == {"measure/sample__panel.json", "measure/sample__overlay.png",
                                               "particles_all.csv", "panels_all.csv", "run_info.json"}
            assert "test-access-token" not in archive.read("run_info.json").decode()
        assert not (tmp_path / job_id).exists()
        assert client.get(f"/jobs/{job_id}/result", headers=AUTH).status_code == 410


def test_cancel_queued_and_running_jobs_without_parallel_work(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocking_runner(*args, **kwargs):
        calls.append(args)
        entered.set()
        assert release.wait(5)
        # Cancellation is checked at the next scientific stage boundary.
        kwargs["progress"](1, 1, "sample", "proposals")
        return write_fake_results(*args, **kwargs)

    app = create_app("test-access-token", runner=blocking_runner, jobs_root=tmp_path)
    with TestClient(app) as client:
        try:
            first = submit(client)
            assert entered.wait(2)
            second = submit(client)
            queued = client.get(f"/jobs/{second}", headers=AUTH).json()
            assert (queued["state"], queued["queue_position"]) == ("queued", 1)
            assert client.delete(f"/jobs/{second}", headers=AUTH).json()["state"] == "cancelled"
            assert not (tmp_path / second).exists()
            cancellation = client.delete(f"/jobs/{first}", headers=AUTH).json()
            assert cancellation["state"] == "running" and cancellation["cancel_requested"]
            release.set()
            assert poll_terminal(client, first)["state"] == "cancelled"
            assert not (tmp_path / first).exists()
            assert len(calls) == 1
        finally:
            release.set()


@pytest.mark.parametrize("data", [{"modality": "OTHER"}, {"modality": "TEM", "nm_per_px": "nan"},
                                  {"modality": "TEM", "nm_per_px": "0"}])
def test_invalid_inputs_do_not_queue(tmp_path, data):
    app = create_app("test-access-token", runner=write_fake_results, jobs_root=tmp_path)
    with TestClient(app) as client:
        response = client.post("/jobs", headers=AUTH, files={"files": ("sample.png", b"image")}, data=data)
        assert response.status_code == 422
        assert not app.state.jobs.jobs


def test_upload_limit_applies_before_parsing_and_retention_expires(tmp_path):
    app = create_app("test-access-token", runner=write_fake_results, jobs_root=tmp_path, max_upload_bytes=1024)
    with TestClient(app) as client:
        response = client.post("/jobs", headers=AUTH, files={"files": ("sample.png", b"x" * 2048)}, data={"modality": "TEM"})
        assert response.status_code == 413
        assert not app.state.jobs.jobs
        job_id = submit(client)
        assert poll_terminal(client, job_id)["state"] == "done"
        app.state.jobs.jobs[job_id].created -= 24 * 60 * 60 + 1
        assert client.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404
        assert not (tmp_path / job_id).exists()


def test_upload_limit_covers_chunked_requests_without_content_length(tmp_path):
    app = create_app("test-access-token", runner=write_fake_results, jobs_root=tmp_path, max_upload_bytes=1024)
    with TestClient(app) as client:
        headers = {**AUTH, "Content-Type": "multipart/form-data; boundary=test"}

        def body():
            yield b'--test\r\nContent-Disposition: form-data; name="files"; filename="large.png"\r\n\r\n'
            yield b"x" * 2048
            yield b"\r\n--test--\r\n"

        response = client.post("/jobs", headers=headers, content=body())
        assert response.status_code == 413
        assert not app.state.jobs.jobs


def test_failures_are_reported_and_unready_result_is_conflict(tmp_path):
    def failure(*args, **kwargs):
        raise ValueError("Image sample failed during measurement.")

    app = create_app("test-access-token", runner=failure, jobs_root=tmp_path)
    with TestClient(app) as client:
        job_id = submit(client)
        record = poll_terminal(client, job_id)
        assert record["state"] == "failed"
        assert record["error"] == "Image sample failed during measurement."
        assert client.get(f"/jobs/{job_id}/result", headers=AUTH).status_code == 409
