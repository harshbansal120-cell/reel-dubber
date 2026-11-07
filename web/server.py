"""The local web app. It only listens on this computer, and one job runs at a time,
because some Gnani accounts allow only one active request.
"""

import re
import secrets
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from dubber import media
from dubber.config import LANGUAGES, REFERENCE_MAX_S, REFERENCE_MIN_S, TIMBRE_VOICES, load_settings
from dubber.llm import build_providers
from dubber.pipeline import (
    CANCEL_FLAG,
    REFERENCE_FORMATS,
    SOURCE_FORMATS,
    Pipeline,
    create_job,
    load_job,
    save_job,
    scorecard_markdown,
)

HERE = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
JOB_ID = re.compile(r"^[0-9A-Za-z-]{6,40}$")
SERVED_SUFFIXES = SOURCE_FORMATS | {".md"}
BUSY = {"queued", "running"}


class LineEdit(BaseModel):
    text: str


class MontageRequest(BaseModel):
    languages: list[str]
    seconds: float = 6.0


class RatingRequest(BaseModel):
    rating: int | None = None
    note: str = ""


def create_app(settings=None):
    settings = settings or load_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Reel Dubber")
    workers = ThreadPoolExecutor(max_workers=1)

    # Work that was running when the app last closed can be resumed, not left spinning.
    for job_file in settings.runs_dir.glob("*/job.json"):
        job = load_job(job_file.parent)
        if job.get("status") in BUSY:
            job.update(status="stopped", stage="The app closed during this job. Press Resume to carry on.")
            save_job(job_file.parent, job)

    def job_dir(job_id):
        if not JOB_ID.match(job_id) or not (settings.runs_dir / job_id / "job.json").is_file():
            raise HTTPException(404, "That job doesn't exist.")
        return settings.runs_dir / job_id

    def idle_job(job_id):
        directory = job_dir(job_id)
        job = load_job(directory)
        if job["status"] in BUSY:
            raise HTTPException(409, "This job is busy. Wait for it to finish.")
        return directory, job

    def enqueue(directory, job, stage, work):
        job.update(status="queued", stage=stage, error=None)
        save_job(directory, job)
        workers.submit(work)

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        # Stops other websites from spending your credits through this local app.
        if request.method not in ("GET", "HEAD"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Requests from other sites are blocked."}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html", media_type="text/html")

    @app.get("/api/options")
    def options():
        providers = build_providers(settings)
        return {
            "languages": [{"id": k, "name": v["name"], "mix_name": v["mix_name"]} for k, v in LANGUAGES.items()],
            "voices": [{"name": k, "gender": v} for k, v in TIMBRE_VOICES.items()],
            "gnani_ready": bool(settings.gnani_key),
            "rewrite_models": [p.label for p in providers],
        }

    @app.get("/api/jobs")
    def list_jobs():
        jobs = []
        for job_file in settings.runs_dir.glob("*/job.json"):
            job = load_job(job_file.parent)
            jobs.append(
                {
                    "id": job["id"],
                    "created": job["created"],
                    "status": job["status"],
                    "source_name": job["source_name"],
                    "languages": job["options"]["languages"],
                }
            )
        return sorted(jobs, key=lambda j: j["created"], reverse=True)[:20]

    @app.post("/api/jobs")
    async def create(
        reel: UploadFile = File(...),
        reference: UploadFile | None = File(None),
        languages: str = Form(...),
        voice_mode: str = Form("clone"),
        voice: str = Form("Shivani"),
        style: str = Form("mixed"),
        background: float = Form(0.08),
        check: bool = Form(True),
        label: bool = Form(True),
        owner: str = Form(""),
        consent: bool = Form(False),
        keyterms: str = Form(""),
    ):
        chosen = [code.strip() for code in languages.split(",") if code.strip()]
        if not chosen or any(code not in LANGUAGES for code in chosen):
            raise HTTPException(400, "Choose at least one language to dub into.")
        if style not in ("mixed", "pure"):
            raise HTTPException(400, "Choose a dubbing style.")
        if voice_mode not in ("clone", "timbre"):
            raise HTTPException(400, "Choose a voice.")
        if voice_mode == "timbre" and voice not in TIMBRE_VOICES:
            raise HTTPException(400, "That Timbre voice isn't available.")
        if not 0 <= background <= 1:
            raise HTTPException(400, "Background volume must be between 0 and 1.")
        source_suffix = Path(reel.filename or "").suffix.lower()
        if source_suffix not in SOURCE_FORMATS:
            raise HTTPException(400, "Upload a video or audio file, such as MP4, MOV, MP3 or WAV.")
        if voice_mode == "clone":
            if not consent or not owner.strip():
                raise HTTPException(400, "Cloning needs the voice owner's name and their consent.")
            if reference is None or not reference.filename:
                raise HTTPException(400, "Cloning needs a voice sample of 5 to 30 seconds.")

        job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        directory = settings.runs_dir / job_id
        directory.mkdir(parents=True)
        try:
            await _save_upload(reel, directory / f"source{source_suffix}")
            if voice_mode == "clone":
                await _save_reference(reference, directory)
        except HTTPException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

        job = create_job(
            settings.runs_dir,
            job_id,
            Path(reel.filename).name,
            {
                "languages": chosen,
                "voice_mode": voice_mode,
                "voice": voice,
                "style": style,
                "background": background,
                "check": check,
                "label": label,
                "owner": owner.strip(),
                "keyterms": [term.strip() for term in keyterms.split(",") if term.strip()],
            },
        )
        enqueue(directory, job, "Waiting to start", lambda: Pipeline(settings, job_id).run())
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return load_job(job_dir(job_id))

    @app.post("/api/jobs/{job_id}/resume")
    def resume(job_id: str):
        directory, job = idle_job(job_id)
        enqueue(directory, job, "Waiting to start", lambda: Pipeline(settings, job_id).run())
        return load_job(directory)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        directory = job_dir(job_id)
        (directory / CANCEL_FLAG).touch()
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/languages/{lang}/lines/{line_id}")
    def edit_line(job_id: str, lang: str, line_id: int, edit: LineEdit):
        directory, job = idle_job(job_id)
        state = job["languages"].get(lang)
        if not state or not any(item["id"] == line_id for item in state.get("lines", [])):
            raise HTTPException(404, "That line doesn't exist.")
        if not edit.text.strip():
            raise HTTPException(400, "The line can't be empty.")
        enqueue(directory, job, "Waiting to start", lambda: Pipeline(settings, job_id).edit_line(lang, line_id, edit.text))
        return load_job(directory)

    @app.post("/api/jobs/{job_id}/languages/{lang}/rating")
    def rate(job_id: str, lang: str, rating: RatingRequest):
        directory, job = idle_job(job_id)
        state = job["languages"].get(lang)
        if not state:
            raise HTTPException(404, "That language hasn't been dubbed.")
        if rating.rating is not None and not 1 <= rating.rating <= 5:
            raise HTTPException(400, "Ratings run from 1 to 5.")
        state.update(rating=rating.rating, rating_note=rating.note.strip()[:500])
        save_job(directory, job)
        return job

    @app.post("/api/jobs/{job_id}/montage")
    def montage(job_id: str, request: MontageRequest):
        directory, job = idle_job(job_id)
        if not request.languages or any(code not in job["languages"] for code in request.languages):
            raise HTTPException(400, "Choose languages that have been dubbed.")
        seconds = max(2.0, min(request.seconds, 20.0))
        enqueue(directory, job, "Waiting to start", lambda: Pipeline(settings, job_id).set_montage(request.languages, seconds))
        return load_job(directory)

    @app.get("/api/jobs/{job_id}/scorecard.md")
    def scorecard(job_id: str):
        return PlainTextResponse(scorecard_markdown(load_job(job_dir(job_id))), media_type="text/markdown; charset=utf-8")

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str):
        directory = job_dir(job_id).resolve()
        target = (directory / path).resolve()
        if not target.is_relative_to(directory) or target.suffix.lower() not in SERVED_SUFFIXES or not target.is_file():
            raise HTTPException(404, "That file isn't available.")
        return FileResponse(target)

    return app


async def _save_upload(upload, target):
    written = 0
    with open(target, "wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "That file is too large. Keep it under 500 MB.")
            handle.write(chunk)
    if written == 0:
        raise HTTPException(400, f"{upload.filename} is empty.")


async def _save_reference(upload, directory):
    """Check the documented 5-30 seconds before anything is spent on the sample."""
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in SOURCE_FORMATS:
        raise HTTPException(400, "The voice sample must be an audio file, such as WAV, MP3 or M4A.")
    original = directory / f"upload_sample{suffix}"
    await _save_upload(upload, original)
    try:
        info = media.probe(original)
        if not info["has_audio"]:
            raise HTTPException(400, "The voice sample has no sound in it.")
        if not REFERENCE_MIN_S <= info["duration_s"] <= REFERENCE_MAX_S:
            raise HTTPException(
                400, f"The voice sample is {info['duration_s']:.1f} seconds long. It must be 5 to 30 seconds."
            )
        if suffix in REFERENCE_FORMATS:
            original.replace(directory / f"reference{suffix}")
        else:
            media.convert_to_wav(original, directory / "reference.wav")
            original.unlink()
    except media.MediaError as error:
        raise HTTPException(400, "The voice sample couldn't be read as audio.") from error


app = create_app()
