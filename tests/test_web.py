import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from dubber import gnani, llm, pipeline
from fake_services import FakeServices, tone_wav
from test_dubber import make_settings, reel  # noqa: F401
from web.server import create_app


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(gnani, "BACKOFF_BASE", 0)
    monkeypatch.setattr(llm, "BACKOFF_BASE", 0)
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0)


@pytest.fixture
def services():
    with FakeServices() as fake:
        yield fake


@pytest.fixture
def settings(services, tmp_path):
    return make_settings(services, tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def form(**changes):
    data = {"languages": "ta", "voice_mode": "timbre", "voice": "Shivani", "style": "mixed"}
    data.update(changes)
    return data


def upload(client, reel_path, data=None, reference=None):  # noqa: F811
    files = {"reel": ("my reel.mp4", reel_path.read_bytes(), "video/mp4")}
    if reference:
        files["reference"] = reference
    return client.post("/api/jobs", data=data or form(), files=files)


def wait_for(client, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise AssertionError("The job didn't finish in time.")


def test_page_and_options_load_without_revealing_keys(client):
    assert "VYOMA — AI-Powered Content Localization" in client.get("/").text
    about = client.get("/about")
    assert about.status_code == 200
    assert "One voice." in about.text
    assert "EVON" in about.text
    body = client.get("/api/options")
    assert body.json()["gnani_ready"] is True
    assert body.json()["rewrite_models"] == ["Gemini (gemini-test)"]
    assert len(body.json()["languages"]) == 10
    assert "key-do-not-leak" not in body.text


def test_upload_to_finished_video(client, reel):  # noqa: F811
    response = upload(client, reel)
    assert response.status_code == 200, response.text
    job = wait_for(client, response.json()["id"])
    assert job["status"] == "done", job["error"]
    state = job["languages"]["ta"]
    video = client.get(f"/api/jobs/{job['id']}/files/{state['output']}")
    assert video.status_code == 200
    assert video.headers["content-type"].startswith("video/")
    assert any(item["id"] == job["id"] for item in client.get("/api/jobs").json())
    card = client.get(f"/api/jobs/{job['id']}/scorecard.md")
    assert "Tamil" in card.text


def test_editing_rating_and_montage_through_the_api(client, reel):  # noqa: F811
    job = wait_for(client, upload(client, reel).json()["id"])
    base = f"/api/jobs/{job['id']}"
    assert client.post(f"{base}/languages/ta/lines/0", json={"text": "budget தமிழ்"}).status_code == 200
    job = wait_for(client, job["id"])
    assert job["languages"]["ta"]["lines"][0]["text"] == "budget தமிழ்"
    rated = client.post(f"{base}/languages/ta/rating", json={"rating": 4, "note": "Good"})
    assert rated.json()["languages"]["ta"]["rating"] == 4
    assert "4 of 5" in client.get(f"{base}/scorecard.md").text
    assert client.post(f"{base}/montage", json={"languages": ["ta"], "seconds": 4}).status_code == 200
    job = wait_for(client, job["id"])
    assert job["montage"]["output"] == "montage.mp4"
    assert client.get(f"{base}/files/montage.mp4").status_code == 200


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"languages": ""}, "language"),
        ({"languages": "xx"}, "language"),
        ({"style": "odd"}, "style"),
        ({"voice_mode": "odd"}, "voice"),
        ({"voice": "Nobody"}, "Timbre voice"),
        ({"background": "2"}, "Background"),
        ({"voice_mode": "clone"}, "consent"),
    ],
)
def test_bad_requests_are_refused_clearly(client, reel, changes, message):  # noqa: F811
    response = upload(client, reel, form(**changes))
    assert response.status_code in (400, 422)
    if response.status_code == 400:
        assert message in response.json()["detail"]


def test_wrong_file_type_is_refused(client):
    response = client.post("/api/jobs", data=form(), files={"reel": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_cloning_needs_consent_a_name_and_a_valid_sample(client, reel, settings):  # noqa: F811
    sample = ("me.wav", tone_wav(6.0), "audio/wav")
    clone = {"voice_mode": "clone", "consent": "true", "owner": "Test Student"}
    assert upload(client, reel, form(voice_mode="clone", owner="Test Student"), sample).status_code == 400
    assert upload(client, reel, form(voice_mode="clone", consent="true"), sample).status_code == 400
    assert upload(client, reel, form(**clone)).status_code == 400
    short = upload(client, reel, form(**clone), ("me.wav", tone_wav(3.0), "audio/wav"))
    assert short.status_code == 400 and "5 to 30" in short.json()["detail"]
    assert upload(client, reel, form(**clone), ("me.wav", b"not audio", "audio/wav")).status_code == 400
    # Refused uploads leave nothing behind.
    assert list(settings.runs_dir.glob("*/job.json")) == []


def test_cloned_voice_job_runs_end_to_end(client, reel, services):  # noqa: F811
    response = upload(
        client, reel, form(voice_mode="clone", consent="true", owner="Test Student"), ("me.wav", tone_wav(6.0), "audio/wav")
    )
    assert response.status_code == 200, response.text
    job = wait_for(client, response.json()["id"])
    assert job["status"] == "done", job["error"]
    assert {p["model"] for p in services.state.tts_payloads} == {"vachana-vc-v1"}
    assert job["options"]["owner"] == "Test Student"


def test_phone_recordings_are_converted_for_cloning(client, reel, settings, tmp_path):  # noqa: F811
    import imageio_ffmpeg
    import subprocess

    source = tmp_path / "phone.m4a"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", "-f", "lavfi", "-i", "sine=frequency=200", "-t", "6", str(source)],
        check=True, capture_output=True,
    )
    response = upload(
        client, reel, form(voice_mode="clone", consent="true", owner="Test Student"), ("phone.m4a", source.read_bytes(), "audio/mp4")
    )
    assert response.status_code == 200, response.text
    assert (settings.runs_dir / response.json()["id"] / "reference.wav").exists()
    wait_for(client, response.json()["id"])


def test_other_websites_cant_start_jobs(client, reel):  # noqa: F811
    response = client.post(
        "/api/jobs", data=form(), files={"reel": ("reel.mp4", reel.read_bytes(), "video/mp4")},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_files_cant_escape_the_job_folder_or_expose_secrets(client, reel, settings, tmp_path):  # noqa: F811
    job = wait_for(client, upload(client, reel).json()["id"])
    secret = settings.runs_dir / "secret.mp4"
    secret.write_bytes(b"x")
    base = f"/api/jobs/{job['id']}/files"
    assert client.get(f"{base}/..%2Fsecret.mp4").status_code == 404
    assert client.get(f"{base}/job.json").status_code == 404
    assert client.get(f"{base}/embedding.json").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404


def test_unknown_jobs_and_lines_are_404(client, reel):  # noqa: F811
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    job = wait_for(client, upload(client, reel).json()["id"])
    assert client.post(f"/api/jobs/{job['id']}/languages/ta/lines/99", json={"text": "x"}).status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/languages/ta/lines/0", json={"text": "  "}).status_code == 400


def test_a_job_interrupted_by_closing_the_app_can_resume(services, tmp_path, reel, settings):  # noqa: F811
    job = pipeline.create_job(settings.runs_dir, "interrupted-1", "reel.mp4", {
        "languages": ["ta"], "voice_mode": "timbre", "voice": "Shivani", "style": "mixed", "background": 0.08,
        "check": True, "label": True, "owner": "", "keyterms": [],
    })
    directory = settings.runs_dir / job["id"]
    (directory / "source.mp4").write_bytes(reel.read_bytes())
    job["status"] = "running"
    pipeline.save_job(directory, job)
    with TestClient(create_app(settings)) as test_client:
        assert test_client.get(f"/api/jobs/{job['id']}").json()["status"] == "stopped"
        assert test_client.post(f"/api/jobs/{job['id']}/resume").status_code == 200
        finished = wait_for(test_client, job["id"])
        assert finished["status"] == "done", finished["error"]
