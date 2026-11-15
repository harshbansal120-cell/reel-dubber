"""The local web app. It only listens on this computer, and one job runs at a time,
because some Gnani accounts allow only one active request.
"""

import copy
import json
import os
import re
import secrets
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from dubber import media
from dubber.config import LANGUAGES, REFERENCE_MAX_S, REFERENCE_MIN_S, TIMBRE_VOICES, load_settings
from dubber.gnani import AUDIO_TAGS
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
ASSETS = HERE / "assets"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
JOB_ID = re.compile(r"^[0-9A-Za-z-]{6,40}$")
SERVED_SUFFIXES = SOURCE_FORMATS | {".md", ".srt"}
BUSY = {"queued", "running"}


class LineEdit(BaseModel):
    text: str
    pronunciation: str = ""
    delivery_tag: str = "none"
    speech_speed: float = 1.0


class PerformanceRequest(BaseModel):
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    saves: int | None = None
    posted_on: str = ""


class MontageRequest(BaseModel):
    languages: list[str]
    seconds: float = 6.0


class RatingRequest(BaseModel):
    rating: int | None = None
    note: str = ""


class CreatorKitRequest(BaseModel):
    profiles: list[dict] = Field(default_factory=list)
    personas: list[dict] = Field(default_factory=list)
    pronunciations: list[dict] = Field(default_factory=list)


class AudienceVersionRequest(BaseModel):
    language: str
    persona_id: str


class CommentDraftRequest(BaseModel):
    language: str
    comment: str
    persona_id: str = ""


class CommentRenderRequest(BaseModel):
    script: str


class FinalAudioMixRequest(BaseModel):
    keep_source_audio: bool
    background_percent: int = 8


def create_app(settings=None):
    settings = settings or load_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Reel Dubber")
    workers = ThreadPoolExecutor(max_workers=1)

    def creator_kit_path():
        return settings.runs_dir / "creator-kit.json"

    def load_creator_kit():
        path = creator_kit_path()
        if not path.exists():
            return {"profiles": [], "personas": [], "pronunciations": []}
        try:
            data = json.loads(path.read_text("utf-8"))
            return {key: data.get(key, []) for key in ("profiles", "personas", "pronunciations")}
        except (OSError, ValueError):
            return {"profiles": [], "personas": [], "pronunciations": []}

    def save_creator_kit(data):
        target = creator_kit_path()
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        temp.replace(target)

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

    @app.get("/assets/creator-preview.gif")
    def creator_preview():
        return FileResponse(ASSETS / "Animated Studio Expressions Grid (1).gif", media_type="image/gif")

    @app.get("/api/options")
    def options():
        providers = build_providers(settings)
        return {
            "languages": [{"id": k, "name": v["name"], "mix_name": v["mix_name"]} for k, v in LANGUAGES.items()],
            "voices": [{"name": k, "gender": v} for k, v in TIMBRE_VOICES.items()],
            "gnani_ready": bool(settings.gnani_key),
            "rewrite_models": [p.label for p in providers],
        }

    @app.get("/api/creator-kit")
    def get_creator_kit():
        return load_creator_kit()

    @app.put("/api/creator-kit")
    def put_creator_kit(request: CreatorKitRequest):
        if len(request.profiles) > 20 or len(request.personas) > 20 or len(request.pronunciations) > 200:
            raise HTTPException(400, "The creator kit supports up to 20 profiles, 20 personas and 200 pronunciations.")

        def bounded(value, limit):
            return str(value or "").strip()[:limit]

        profiles = []
        for row in request.profiles:
            name = bounded(row.get("name"), 80)
            if not name:
                continue
            mix = int(row.get("code_mix_percent", 65))
            if not 0 <= mix <= 100:
                raise HTTPException(400, "Profile code-mix values must be between 0 and 100.")
            profiles.append({
                "id": bounded(row.get("id"), 40) or secrets.token_hex(5),
                "name": name,
                "brief": bounded(row.get("brief"), 500),
                "dialect": bounded(row.get("dialect"), 120),
                "code_mix_percent": mix,
                "glossary": [bounded(term, 240) for term in row.get("glossary", []) if bounded(term, 240)][:30],
            })
        personas = []
        for row in request.personas:
            name, brief = bounded(row.get("name"), 80), bounded(row.get("brief"), 500)
            if not name or not brief:
                continue
            mix = int(row.get("code_mix_percent", 65))
            if not 0 <= mix <= 100:
                raise HTTPException(400, "Persona code-mix values must be between 0 and 100.")
            personas.append({
                "id": bounded(row.get("id"), 40) or secrets.token_hex(5),
                "name": name, "brief": brief,
                "dialect": bounded(row.get("dialect"), 120),
                "code_mix_percent": mix,
            })
        pronunciations = []
        seen_terms = set()
        for row in request.pronunciations:
            term, spoken = bounded(row.get("term"), 100), bounded(row.get("spoken"), 120)
            if not term or not spoken or term.casefold() in seen_terms:
                continue
            seen_terms.add(term.casefold())
            pronunciations.append({"term": term, "spoken": spoken})
        data = {"profiles": profiles, "personas": personas, "pronunciations": pronunciations}
        save_creator_kit(data)
        return data

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
                    "variant_label": job.get("variant_label", ""),
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
        code_mix_percent: int = Form(65),
        audio_cleanup: str = Form("auto"),
        distinct_speakers: bool = Form(True),
        background: float = Form(0.08),
        keep_source_audio: bool | None = Form(None),
        check: bool = Form(True),
        label: bool = Form(True),
        owner: str = Form(""),
        consent: bool = Form(False),
        keyterms: str = Form(""),
        creator_profile_id: str = Form(""),
        audience_persona_id: str = Form(""),
        localization_brief: str = Form(""),
        glossary: str = Form(""),
        region_dialect: str = Form(""),
        emotion_analysis: bool = Form(False),
        semantic_check: bool = Form(True),
    ):
        chosen = [code.strip() for code in languages.split(",") if code.strip()]
        if not chosen or any(code not in LANGUAGES for code in chosen):
            raise HTTPException(400, "Choose at least one language to dub into.")
        if style not in ("mixed", "pure"):
            raise HTTPException(400, "Choose a dubbing style.")
        if not 0 <= code_mix_percent <= 100:
            raise HTTPException(400, "Code-mix level must be between 0 and 100.")
        if audio_cleanup not in ("auto", "noisy", "phone"):
            raise HTTPException(400, "Choose a supported audio cleanup profile.")
        if voice_mode not in ("clone", "timbre"):
            raise HTTPException(400, "Choose a voice.")
        if voice_mode == "timbre" and voice not in TIMBRE_VOICES:
            raise HTTPException(400, "That Timbre voice isn't available.")
        if not 0 <= background <= 1:
            raise HTTPException(400, "Background volume must be between 0 and 1.")
        if keep_source_audio is None:
            keep_source_audio = background > 0
        if not keep_source_audio:
            background = 0.0
        source_suffix = Path(reel.filename or "").suffix.lower()
        if source_suffix not in SOURCE_FORMATS:
            raise HTTPException(400, "Upload a video or audio file, such as MP4, MOV, MP3 or WAV.")
        if voice_mode == "clone":
            if not consent or not owner.strip():
                raise HTTPException(400, "Cloning needs the voice owner's name and their consent.")
            if reference is None or not reference.filename:
                raise HTTPException(400, "Cloning needs a voice sample of 5 to 30 seconds.")

        kit = load_creator_kit()
        profile = next((p for p in kit["profiles"] if p["id"] == creator_profile_id), None) if creator_profile_id else None
        persona = next((p for p in kit["personas"] if p["id"] == audience_persona_id), None) if audience_persona_id else None
        if creator_profile_id and not profile:
            raise HTTPException(400, "That creator profile is no longer saved. Refresh the page and choose it again.")
        if audience_persona_id and not persona:
            raise HTTPException(400, "That audience persona is no longer saved. Refresh the page and choose it again.")
        creator_brief = localization_brief.strip()
        creator_dialect = region_dialect.strip()
        if persona:
            creator_brief = (creator_brief + "\nAudience persona " + persona["name"] + ": " + persona["brief"]).strip()[:500]
            creator_dialect = (creator_dialect or persona.get("dialect", ""))[:120]
            code_mix_percent = persona["code_mix_percent"]
        dictionary_terms = [entry["term"] for entry in kit["pronunciations"]]
        all_keyterms = list(dict.fromkeys([term.strip() for term in keyterms.split(",") if term.strip()] + dictionary_terms))

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
                "distinct_speakers": bool(distinct_speakers and voice_mode == "timbre"),
                "style": "mixed" if code_mix_percent >= 40 else "pure",
                "code_mix_percent": code_mix_percent,
                "audio_cleanup": audio_cleanup,
                "keep_source_audio": keep_source_audio,
                "background": background,
                "check": check,
                "label": label,
                "owner": owner.strip(),
                "keyterms": all_keyterms[:200],
                "localization_brief": creator_brief,
                "glossary": [line.strip()[:240] for line in glossary.splitlines() if line.strip()][:30],
                "region_dialect": creator_dialect,
                "creator_profile_id": profile["id"] if profile else "",
                "creator_profile_name": profile["name"] if profile else "",
                "audience_persona": persona,
                "pronunciation_dictionary": kit["pronunciations"],
                "emotion_analysis": emotion_analysis,
                "semantic_check": semantic_check,
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

    @app.post("/api/jobs/{job_id}/languages/{lang}/final-audio")
    def final_audio_mix(job_id: str, lang: str, request: FinalAudioMixRequest):
        directory, job = idle_job(job_id)
        if lang not in job.get("languages", {}) or job["languages"][lang].get("status") != "done":
            raise HTTPException(400, "Choose a language with a finished dub.")
        if not 0 <= request.background_percent <= 40:
            raise HTTPException(400, "Original soundtrack level must be between 0 and 40 percent.")
        gain = request.background_percent / 100 if request.keep_source_audio else 0.0
        enqueue(directory, job, "Rebuilding export with the selected original soundtrack setting",
                lambda: Pipeline(settings, job_id).set_final_audio_mix(lang, request.keep_source_audio, gain))
        return load_job(directory)

    @app.post("/api/jobs/{job_id}/audience-versions")
    def audience_version(job_id: str, request: AudienceVersionRequest):
        parent_dir, parent = idle_job(job_id)
        lang = request.language
        if lang not in parent.get("languages", {}) or parent["languages"][lang].get("status") != "done":
            raise HTTPException(400, "Choose a language with a completed dub.")
        kit = load_creator_kit()
        persona = next((row for row in kit["personas"] if row["id"] == request.persona_id), None)
        if not persona:
            raise HTTPException(400, "Choose a saved audience persona.")
        job_id_new = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        directory = settings.runs_dir / job_id_new
        directory.mkdir(parents=True)
        try:
            for item in parent_dir.iterdir():
                if item.is_file() and (item.stem in {"source", "speech", "source_voice", "embedding", "reference"}):
                    target = directory / item.name
                    try:
                        os.link(item, target)
                    except OSError:
                        shutil.copy2(item, target)
            options = copy.deepcopy(parent["options"])
            options.update(languages=[lang], audience_persona=persona,
                           localization_brief=(str(options.get("localization_brief", "")) + "\nAudience persona " + persona["name"] + ": " + persona["brief"])[:500],
                           region_dialect=persona.get("dialect") or options.get("region_dialect", ""),
                           code_mix_percent=persona["code_mix_percent"],
                           pronunciation_dictionary=kit["pronunciations"])
            child = create_job(settings.runs_dir, job_id_new, parent["source_name"], options)
            child.update(variant_of=job_id, variant_label="Audience: " + persona["name"],
                         source=copy.deepcopy(parent["source"]), source_language=parent.get("source_language"),
                         lines=copy.deepcopy(parent["lines"]), batch=None, emotion_analysis_done=True,
                         warnings=["Reused the source transcript. Audience rewrite and speech generation make additional model requests."])
            save_job(directory, child)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        enqueue(directory, child, "Waiting to create audience version", lambda: Pipeline(settings, job_id_new).run())
        return child

    @app.post("/api/jobs/{job_id}/comment-responses")
    def draft_comment_response(job_id: str, request: CommentDraftRequest):
        directory, job = idle_job(job_id)
        lang = request.language
        comment = request.comment.strip()
        if lang not in job.get("languages", {}) or job["languages"][lang].get("status") != "done":
            raise HTTPException(400, "Choose a language with a completed dub.")
        if not comment or len(comment) > 1200:
            raise HTTPException(400, "Enter a comment under 1,200 characters.")
        providers = build_providers(settings)
        if not providers:
            raise HTTPException(503, "Add a rewrite model to draft a reply.")
        persona = next((row for row in load_creator_kit()["personas"] if row["id"] == request.persona_id), None) if request.persona_id else None
        meta = LANGUAGES[lang]
        transcript = "\n".join(f"[{line.get('start', 0):.1f}s] {line.get('text', '')}" for line in job.get("lines", []))[:12000]
        mix = persona.get("code_mix_percent", job["options"].get("code_mix_percent", 65)) if persona else job["options"].get("code_mix_percent", 65)
        dialect = persona.get("dialect", "") if persona else job["options"].get("region_dialect", "")
        system = (f"Write one concise, natural spoken response in {meta['name']} for a creator replying to an audience comment. "
                  f"Use a code-mix preference of {mix}% and regional style {dialect or 'natural conversational'}. "
                  "Use only facts supported by the video transcript. If it does not answer the question, say so honestly; never invent claims. "
                  "Return JSON only as {\"script\":\"...\"}. Keep it under 600 characters.")
        persona_text = (" Audience style: " + persona["brief"]) if persona else ""
        user = json.dumps({"comment": comment, "video_transcript": transcript, "audience_style": persona_text}, ensure_ascii=False)
        last_error = "No model could draft a response."
        script = ""
        for provider in providers:
            try:
                raw = provider.complete(system, user)
                job["usage"]["llm_requests"] += 1
                match = re.search(r"\{.*\}", raw, re.S)
                parsed = json.loads(match.group(0)) if match else {}
                script = str(parsed.get("script", "")).strip()[:600]
                if script:
                    break
                last_error = "The model returned an empty response."
            except Exception as error:
                last_error = str(error)[:300]
        if not script:
            save_job(directory, job)
            raise HTTPException(502, last_error)
        responses = job.setdefault("comment_responses", [])
        response = {"id": secrets.token_hex(8), "language": lang, "comment": comment, "script": script,
                    "persona_id": persona["id"] if persona else "", "status": "draft"}
        responses.append(response)
        save_job(directory, job)
        return response

    @app.post("/api/jobs/{job_id}/comment-responses/{response_id}/render")
    def render_comment_response(job_id: str, response_id: str, request: CommentRenderRequest):
        directory, job = idle_job(job_id)
        response = next((row for row in job.get("comment_responses", []) if row.get("id") == response_id), None)
        if not response:
            raise HTTPException(404, "That response draft doesn't exist.")
        script = request.script.strip()
        if not script or len(script) > 600:
            raise HTTPException(400, "Keep the reviewed response under 600 characters.")
        enqueue(directory, job, "Waiting to generate response audio", lambda: Pipeline(settings, job_id).render_comment_response(response["language"], response_id, script))
        return load_job(directory)

    @app.post("/api/jobs/{job_id}/resume-direct-transcription")
    def resume_direct_transcription(job_id: str):
        directory, job = idle_job(job_id)
        batch = job.get("batch") or {}
        if not batch.get("job_id") or batch.get("status") != "COMPLETED":
            raise HTTPException(409, "Direct fallback is available after the Prisma batch finishes.")
        if not job.get("source") or float(job["source"].get("duration_s", 0)) > 30:
            raise HTTPException(400, "Direct Prisma transcription is available only for clips up to 30 seconds.")
        if job.get("lines"):
            raise HTTPException(409, "This job already has a transcript.")
        job["direct_transcription"] = True
        enqueue(directory, job, "Waiting to retry with direct transcription", lambda: Pipeline(settings, job_id).run())
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
        if len(edit.text) > 1200 or len(edit.pronunciation) > 240:
            raise HTTPException(400, "Keep the translated line under 1,200 characters and pronunciation guide under 240.")
        if edit.delivery_tag not in AUDIO_TAGS | {"none"}:
            raise HTTPException(400, "Choose a supported voice delivery.")
        if not 0.8 <= edit.speech_speed <= 1.2:
            raise HTTPException(400, "Speaking speed must be between 0.8 and 1.2.")
        enqueue(directory, job, "Waiting to start", lambda: Pipeline(settings, job_id).edit_line(
            lang, line_id, edit.text, edit.pronunciation, edit.delivery_tag, edit.speech_speed
        ))
        return load_job(directory)

    @app.post("/api/jobs/{job_id}/languages/{lang}/lines/{line_id}/voice-direction")
    async def voice_direction(job_id: str, lang: str, line_id: int, audio: UploadFile = File(...)):
        directory, job = idle_job(job_id)
        state = job["languages"].get(lang)
        if not state or state.get("status") != "done" or not any(item["id"] == line_id for item in state.get("lines", [])):
            raise HTTPException(404, "That finished line isn't available.")
        payload = await audio.read(15 * 1024 * 1024 + 1)
        if not payload or len(payload) > 15 * 1024 * 1024:
            raise HTTPException(413, "Keep the voice direction recording under 15 MB.")
        suffix = Path(audio.filename or "voice-direction.webm").suffix.lower()
        if suffix not in SOURCE_FORMATS:
            raise HTTPException(400, "Record a supported audio format.")
        raw = directory / f"voice_direction_{secrets.token_hex(4)}{suffix}"
        normalized = raw.with_name(raw.stem + "_normalized.mp3")
        try:
            raw.write_bytes(payload)
            media.extract_speech(raw, normalized)
            info = media.probe(normalized)
            if not info["has_audio"] or info["duration_s"] > 30:
                raise HTTPException(400, "Record an audio instruction under 30 seconds.")
            speech = normalized.read_bytes()
        except media.MediaError as error:
            raise HTTPException(400, "The voice instruction audio couldn't be read.") from error
        finally:
            raw.unlink(missing_ok=True)
            normalized.unlink(missing_ok=True)
        duration = info["duration_s"]
        enqueue(directory, job, "Listening to your edit with Prisma", lambda: Pipeline(settings, job_id).voice_direct_edit(
            lang, line_id, speech, duration
        ))
        return load_job(directory)

    @app.post("/api/jobs/{job_id}/languages/{lang}/metrics")
    def save_performance(job_id: str, lang: str, metrics: PerformanceRequest):
        directory, job = idle_job(job_id)
        state = job["languages"].get(lang)
        if not state or state.get("status") != "done":
            raise HTTPException(404, "That language doesn't have a finished dub.")
        values = metrics.dict()
        for key in ("views", "likes", "comments", "shares", "saves"):
            if values[key] is not None and values[key] < 0:
                raise HTTPException(400, "Performance counts can't be negative.")
        state["performance"] = values
        save_job(directory, job)
        return job

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
