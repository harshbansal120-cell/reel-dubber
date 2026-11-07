"""The dubbing pipeline. Every paid result is saved as soon as it arrives, so a
stopped or failed run can resume without paying twice.
"""

import hashlib
import json
import logging
import re
import shutil
import time
import wave
from datetime import datetime, timezone

from . import media
from .config import LANGUAGES, MAX_SOURCE_SECONDS
from .gnani import AUDIO_TAGS, CLONE_MODEL, STOCK_MODEL, STT_MODEL, ApiError, GnaniClient, StopRun
from .llm import build_providers
from .metrics import character_error_rate, english_kept, has_digits, spoken_length, wav_info
from .rewrite import RewriteFailed, adapt_lines, shorten_lines, source_english_kept

log = logging.getLogger("dubber")

CANCEL_FLAG = "cancel.flag"
POLL_SECONDS = 10  # The batch docs ask for no faster than this.
BATCH_TIMEOUT_S = 900
BATCH_DONE = {"COMPLETED", "PARTIAL_FAILURE", "FAILED", "START_FAILED", "CANCELLED"}

# Speech rates differ by language and voice, so this is only the first guess.
# The pipeline measures the real rate from the first clips it gets.
DEFAULT_CHARS_PER_SECOND = 13.0
MAX_TEMPO = 1.15  # Above this, speech starts to sound rushed: ask for a shorter line instead.
HARD_MAX_TEMPO = 1.3
MIN_GAP_S = 0.05
LATE_FLAG_S = 0.25
SILENCE_RMS = 50
CER_CHECK = 0.30  # A starting point for this project, not a published benchmark.
MERGE_GAP_S = 0.3
MERGE_MAX_S = 8.0
SLOT_EXTRA_S = 1.0
SOURCE_FORMATS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
REFERENCE_FORMATS = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


class Cancelled(Exception):
    pass


class LanguageFailed(Exception):
    """This language can't be dubbed, but others still might be."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def job_path(job_dir):
    return job_dir / "job.json"


def load_job(job_dir):
    for attempt in range(20):
        try:
            return json.loads(job_path(job_dir).read_text("utf-8"))
        except PermissionError:
            time.sleep(0.05)  # Windows: another thread is replacing the file.
    return json.loads(job_path(job_dir).read_text("utf-8"))


def save_job(job_dir, job):
    job["updated"] = now()
    target = job_path(job_dir)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=1), "utf-8")
    for attempt in range(20):
        try:
            temp.replace(target)
            return
        except PermissionError:
            time.sleep(0.05)
    temp.replace(target)


def new_usage():
    return {
        "embedding_requests": 0,
        "tts_requests": 0,
        "tts_chars": 0,
        "stt_requests": 0,
        "stt_audio_s": 0.0,
        "batch_jobs": 0,
        "llm_requests": 0,
        "retries": 0,
    }


def create_job(runs_dir, job_id, source_name, options):
    job_dir = runs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "created": now(),
        "updated": now(),
        "status": "queued",
        "stage": "Waiting to start",
        "error": None,
        "warnings": [],
        "source_name": source_name,
        "source": None,
        "options": options,
        "batch": None,
        "source_language": None,
        "lines": [],
        "languages": {},
        "usage": new_usage(),
        "montage": None,
    }
    save_job(job_dir, job)
    return job


def merge_segments(segments, total_s):
    """Join fragments of one sentence, then work out how long each line may take.

    Speech detection can split a sentence in two, and translating fragments
    separately reads badly.
    """
    spoken = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        start, end = segment.get("start_time"), segment.get("end_time")
        if text and start is not None and end is not None:
            spoken.append({"start": float(start), "end": float(end), "text": text, "speaker": segment.get("speaker_id")})
    spoken.sort(key=lambda s: s["start"])

    merged = []
    for segment in spoken:
        previous = merged[-1] if merged else None
        if (
            previous
            and previous["speaker"] == segment["speaker"]
            and segment["start"] - previous["end"] <= MERGE_GAP_S
            and segment["end"] - previous["start"] <= MERGE_MAX_S
            and not re.search(r"[.?!।॥]$", previous["text"])
        ):
            previous["end"] = segment["end"]
            previous["text"] += " " + segment["text"]
        else:
            merged.append(dict(segment))

    for index, line in enumerate(merged):
        line["id"] = index
        speaking = line["end"] - line["start"]
        limit = merged[index + 1]["start"] if index + 1 < len(merged) else total_s
        line["slot_s"] = round(max(0.3, min(limit - line["start"], speaking + SLOT_EXTRA_S)), 3)
    return merged


def place_lines(items):
    """Start times for clips. A clip that can't fit pushes the next one later."""
    cursor = 0.0
    for item in items:
        if not item.get("final_s"):
            continue
        item["placed_s"] = round(max(item["start"], cursor), 3)
        item["late_s"] = round(item["placed_s"] - item["start"], 3)
        cursor = item["placed_s"] + item["final_s"] + MIN_GAP_S


def score_language(state):
    """Numbers for the scorecard. Lines with digits are left out of the error rate,
    because Prisma may write a number as words."""
    items = state["lines"]
    voiced = [i for i in items if i.get("clip") and not i.get("silent")]
    checked = [i for i in voiced if (i.get("stt") or {}).get("ok")]
    rated = [i for i in checked if not has_digits(i["text"])]
    source_kept = [source_english_kept(i["source_text"], i["text"]) for i in items if i.get("text")]
    return {
        "lines": len(items),
        "voiced": len(voiced),
        "failed": sum(1 for i in items if not i.get("clip")),
        "mean_cer": round(sum(i["stt"]["cer"] for i in rated) / len(rated), 3) if rated else None,
        "language_matches": sum(1 for i in checked if i["stt"].get("language_matches")),
        "checked": len(checked),
        "heard_english_kept": sum(i["stt"].get("english_kept", 0) for i in checked),
        "heard_english_total": sum(i["stt"].get("english_total", 0) for i in checked),
        "source_english_kept": sum(k for k, _ in source_kept),
        "source_english_total": sum(t for _, t in source_kept),
        "mean_tempo": round(sum(i.get("tempo", 1) for i in voiced) / len(voiced), 3) if voiced else None,
        "max_tempo": max((i.get("tempo", 1) for i in voiced), default=None),
        "rushed": sum(1 for i in voiced if i.get("tempo", 1) > MAX_TEMPO),
        "late": sum(1 for i in voiced if i.get("late_s", 0) > LATE_FLAG_S),
        "shortened": sum(1 for i in items if i.get("shortened")),
        "to_review": sum(1 for i in items if i.get("flags")),
    }


class Pipeline:
    def __init__(self, settings, job_id):
        self.settings = settings
        self.dir = settings.runs_dir / job_id
        self.job = load_job(self.dir)
        self.client = GnaniClient(settings.gnani_base_url, settings.gnani_key, self.job["usage"])
        self.providers = build_providers(settings)
        self.embedding = None
        self.voice_id = None

    # ------------------------------------------------------------ run control

    def save(self):
        save_job(self.dir, self.job)

    def stage(self, message):
        self.job["stage"] = message
        self.save()
        self.check_cancelled()

    def check_cancelled(self):
        # A stop request arrives as a file, so it can't collide with this thread's saves.
        if (self.dir / CANCEL_FLAG).exists():
            raise Cancelled()

    def warn(self, message):
        if message not in self.job["warnings"]:
            self.job["warnings"].append(message)

    def _guarded(self, work):
        """Run a piece of work and always leave the job in a clear state."""
        (self.dir / CANCEL_FLAG).unlink(missing_ok=True)
        self.job.update(status="running", error=None)
        save_job(self.dir, self.job)
        try:
            work()
        except Cancelled:
            self.job.update(status="stopped", stage="Stopped at your request. Press Resume to carry on.")
        except StopRun as error:
            self.job.update(status="failed", stage="Stopped", error=str(error))
        except (ApiError, RewriteFailed, media.MediaError, ValueError, OSError) as error:
            self.job.update(status="failed", stage="Stopped", error=str(error))
        except Exception:
            log.exception("Unexpected failure in job %s", self.job["id"])
            self.job.update(status="failed", stage="Stopped", error="Something unexpected went wrong. See the server log.")
        for state in self.job["languages"].values():
            if state.get("status") == "running":
                state["status"] = "stopped"
        (self.dir / CANCEL_FLAG).unlink(missing_ok=True)
        save_job(self.dir, self.job)
        return self.job

    def run(self):
        return self._guarded(self._run)

    def _run(self):
        self._prepare()
        self._transcribe()
        self._prepare_voice()
        for lang in self.job["options"]["languages"]:
            self._dub_language(lang)
        states = self.job["languages"].values()
        if any(state.get("status") == "done" for state in states):
            self.job.update(status="done", stage="Finished")
        else:
            errors = [state.get("error") for state in states if state.get("error")]
            self.job.update(status="failed", stage="Stopped", error=errors[0] if errors else "No language could be dubbed.")

    # ------------------------------------------------------------ source

    def _source_path(self):
        return self.dir / self.job["source"]["file"]

    def _prepare(self):
        self.stage("Reading your reel")
        file_name = next((p.name for p in self.dir.iterdir() if p.stem == "source" and p.is_file()), None)
        if file_name is None:
            raise ValueError("The uploaded reel is missing.")
        info = media.probe(self.dir / file_name)
        if not info["has_audio"]:
            raise ValueError("This file has no sound to dub.")
        if info["duration_s"] > MAX_SOURCE_SECONDS:
            raise ValueError(
                f"This file is {info['duration_s']:.0f} seconds long. Trim it to {MAX_SOURCE_SECONDS} seconds or less, "
                "to protect your credits."
            )
        self.job["source"] = {"file": file_name, **info}
        if not (self.dir / "speech.mp3").exists():
            media.extract_speech(self.dir / file_name, self.dir / "speech.mp3")
        if not (self.dir / "source_voice.wav").exists():
            media.extract_voice_track(self.dir / file_name, self.dir / "source_voice.wav")
        self.save()

    def _transcribe(self):
        if self.job["lines"]:
            return
        self.stage("Transcribing with Prisma v3.0")
        batch = self.job.get("batch") or {}
        if not batch.get("job_id"):
            config = {"model": STT_MODEL, "language_code": "indic-auto", "format": "codemix", "with_denoise": True}
            keyterms = self.job["options"].get("keyterms") or []
            if keyterms:
                config["keyterms"] = keyterms[:200]
            job_id = self.client.create_batch((self.dir / "speech.mp3").read_bytes(), "speech.mp3", config)
            batch = {"job_id": job_id, "started": False}
            self.job["batch"] = batch
            self.save()
        job_path_api = batch["job_id"]
        if not batch.get("started"):
            self.client.start_batch(job_path_api)
            batch["started"] = True
            self.save()

        deadline = time.monotonic() + BATCH_TIMEOUT_S
        while True:
            status = self.client.batch_status(job_path_api)
            batch["status"] = status.get("status")
            self.stage(f"Transcribing with Prisma v3.0 ({str(batch['status']).replace('_', ' ').lower()})")
            if batch["status"] in BATCH_DONE:
                break
            if time.monotonic() >= deadline:
                raise ValueError("Transcription is taking too long. Press Resume to check on it again.")
            time.sleep(POLL_SECONDS)

        files = self.client.batch_files(job_path_api)
        finished = next((f for f in files if f.get("transcript_url")), None)
        if finished is None:
            reasons = [f.get("error_message") for f in files if f.get("error_message")]
            self.job["batch"] = None  # A failed job can't be restarted, so the next try makes a new one.
            raise ValueError(f"Transcription failed: {reasons[0] if reasons else batch['status']}")
        document = self.client.fetch_transcript(finished["transcript_url"])
        (self.dir / "transcript.json").write_text(json.dumps(document, ensure_ascii=False, indent=1), "utf-8")
        self.client.usage["stt_audio_s"] = round(
            self.client.usage["stt_audio_s"] + float(document.get("duration_seconds") or 0), 2
        )

        lines = merge_segments(document.get("segments") or [], self.job["source"]["duration_s"])
        if not lines:
            raise ValueError("Prisma found no speech in this file.")
        self.job["lines"] = lines
        self.job["source_language"] = document.get("language_code")
        speakers = {line["speaker"] for line in lines if line["speaker"] not in (None, -1)}
        if len(speakers) > 1:
            self.warn(f"Prisma found {len(speakers)} speakers, but this version gives every line the same voice.")
        self.save()

    # ------------------------------------------------------------ voice

    def _prepare_voice(self):
        options = self.job["options"]
        if options["voice_mode"] != "clone":
            self.voice_id = f"{STOCK_MODEL}:{options['voice']}"
            return
        reference = next((p for p in self.dir.iterdir() if p.stem == "reference" and p.is_file()), None)
        if reference is None:
            raise ValueError("The voice sample for cloning is missing.")
        audio = reference.read_bytes()
        digest = sha(audio)
        self.voice_id = f"{CLONE_MODEL}:{digest}"
        path = self.dir / "embedding.json"
        if path.exists():
            cached = json.loads(path.read_text("utf-8"))
            if cached.get("reference_sha256") == digest:
                self.embedding = cached["speaker_embedding"]
                return
        self.stage("Creating your voice embedding")
        self.embedding = self.client.create_embedding(audio, reference.name, REFERENCE_FORMATS[reference.suffix.lower()])
        path.write_text(json.dumps({"reference_sha256": digest, "speaker_embedding": self.embedding}), "utf-8")
        self.save()

    def _tts_language(self, lang):
        meta = LANGUAGES[lang]
        return meta["mix_tts"] if self.job["options"]["style"] == "mixed" else meta["code"]

    def _speak(self, lang, item):
        """Synthesize one line, reusing any clip already paid for with the same text and voice."""
        language = self._tts_language(lang)
        delivery_tag = item.get("delivery_tag") if self.embedding is None else None
        delivery_tag = delivery_tag if delivery_tag in AUDIO_TAGS else None
        speech_text = f"<{delivery_tag}>{item['text']}" if delivery_tag else item["text"]
        key = sha(f"{self.voice_id}|{language}|{speech_text}".encode("utf-8"))[:16]
        base = self.dir / "langs" / lang
        raw = base / "raw" / f"{key}.wav"
        clean = base / "clean" / f"{key}.wav"
        raw.parent.mkdir(parents=True, exist_ok=True)
        clean.parent.mkdir(parents=True, exist_ok=True)
        for stale in ("error", "final", "final_s", "tempo", "placed_s", "late_s"):
            item.pop(stale, None)
        item["flags"] = []
        if not raw.exists():
            if self.embedding is not None:
                audio = self.client.speak_cloned(item["text"], self.embedding)
            else:
                audio = self.client.speak_stock(
                    item["text"], self.job["options"]["voice"], language, delivery_tag
                )
            try:
                wav_info(audio)
            except (wave.Error, EOFError, ZeroDivisionError):
                raise ApiError(200, "The speech service returned something that isn't WAV audio.")
            raw.write_bytes(audio)
            self.save()
        if not clean.exists():
            media.clean_clip(raw, clean)
        duration, rms = wav_info(clean.read_bytes())
        item.update(
            clip=key,
            duration_s=round(duration, 3),
            silent=duration < 0.05 or (rms is not None and rms < SILENCE_RMS),
            chars=spoken_length(item["text"]),
        )

    def _speak_safely(self, lang, item):
        try:
            self._speak(lang, item)
        except ApiError as error:
            item.update(clip=None, duration_s=None, error=f"{error.status}: {error.message}", flags=["failed"])
        self.save()

    # ------------------------------------------------------------ one language

    def _source_language_name(self):
        code = self.job.get("source_language")
        return next((m["name"] for m in LANGUAGES.values() if m["code"] == code), "the original language")

    def _measured_rate(self, state):
        voiced = [i for i in state["lines"] if i.get("clip") and not i.get("silent") and i.get("duration_s")]
        total = sum(i["duration_s"] for i in voiced)
        return sum(i["chars"] for i in voiced) / total if total else DEFAULT_CHARS_PER_SECOND

    def _shorten(self, lang, state, items):
        meta = LANGUAGES[lang]
        expressive = self.job["options"].get("voice_mode") == "timbre"
        self.client.usage["llm_requests"] += 1
        try:
            better, label, notes = shorten_lines(
                self.providers, items, meta, self.job["options"]["style"],
                self._source_language_name(), self._measured_rate(state), MAX_TEMPO, expressive,
            )
        except RewriteFailed as error:
            state["notes"].append(f"Couldn't shorten some lines: {error}")
            return []
        state["notes"].extend(notes)
        changed = []
        for item in items:
            if item["id"] in better:
                replacement = better[item["id"]]
                if expressive:
                    item["text"] = replacement["text"]
                    item["delivery_tag"] = replacement["delivery_tag"]
                else:
                    item["text"] = replacement
                item["shortened"] = True
                changed.append(item)
        self.save()
        return changed

    def _dub_language(self, lang):
        meta = LANGUAGES[lang]
        options = self.job["options"]
        if self.job.get("source_language") == meta["code"]:
            self.warn(f"This reel is already in {meta['name']}, so that dub was skipped.")
            self.job["languages"].pop(lang, None)
            self.save()
            return
        state = self.job["languages"].get(lang)
        if state and state.get("status") == "done" and state.get("voice_id") == self.voice_id:
            return
        if not state or state.get("voice_id") != self.voice_id or not state.get("lines"):
            state = {"status": "running", "error": None, "notes": [], "lines": [], "voice_id": self.voice_id, "rewrite_by": None}
            self.job["languages"][lang] = state
        state.update(status="running", error=None)
        try:
            if not state["lines"]:
                self.stage(f"{meta['name']}: rewriting the lines")
                self.client.usage["llm_requests"] += 1
                expressive = options.get("voice_mode") == "timbre"
                result, label, notes = adapt_lines(
                    self.providers, self.job["lines"], meta, options["style"],
                    self._source_language_name(), DEFAULT_CHARS_PER_SECOND, expressive,
                )
                state["rewrite_by"] = label
                state["notes"].extend(notes)
                state["lines"] = []
                for line in self.job["lines"]:
                    rewritten = result[line["id"]]
                    item = {
                        "id": line["id"],
                        "start": line["start"],
                        "slot_s": line["slot_s"],
                        "source_text": line["text"],
                        "text": rewritten["text"] if expressive else rewritten,
                    }
                    if expressive:
                        item["delivery_tag"] = rewritten["delivery_tag"]
                    state["lines"].append(item)
                self.save()
            self._speak_all(lang, state)
            self._finish_language(lang, state)
        except RewriteFailed as error:
            state.update(status="failed", error=f"Rewrite failed: {error}")
            self.save()
        except LanguageFailed as error:
            state.update(status="failed", error=str(error))
            self.save()
        except media.MediaError as error:
            state.update(status="failed", error=f"Couldn't build the video: {error}")
            self.save()

    def _speak_all(self, lang, state):
        """Speak the longest lines first and measure the rate, so lines likely to run
        too long get shortened before any credits are spent on them."""
        meta = LANGUAGES[lang]
        items = state["lines"]
        pending = sorted((i for i in items if not i.get("clip")), key=lambda i: -spoken_length(i["text"]))
        samples = pending[:2]
        for item in samples:
            self.stage(f"{meta['name']}: speaking line {item['id'] + 1} of {len(items)}")
            self._speak_safely(lang, item)
        if samples and all(item.get("clip") is None for item in samples):
            # One more try, then stop spending on a voice that can't do this language.
            samples = pending[:3]
            if len(samples) == 3:
                self._speak_safely(lang, samples[2])
            if all(item.get("clip") is None for item in samples):
                errors = [item.get("error") for item in samples if item.get("error")]
                raise LanguageFailed(f"The voice couldn't speak {meta['name']}: {errors[0] if errors else 'no audio came back'}")

        rate = self._measured_rate(state)
        remaining = [i for i in pending if not i.get("clip") and not i.get("error")]
        predicted_long = [
            i for i in remaining if spoken_length(i["text"]) / rate > i["slot_s"] * MAX_TEMPO
        ]
        if predicted_long:
            self.stage(f"{meta['name']}: shortening {len(predicted_long)} lines before speaking them")
            self._shorten(lang, state, predicted_long)
        for item in remaining:
            self.stage(f"{meta['name']}: speaking line {item['id'] + 1} of {len(items)}")
            self._speak_safely(lang, item)

        if not state.get("refit_done"):
            too_long = [
                i for i in items
                if i.get("clip") and not i.get("silent") and i["duration_s"] / i["slot_s"] > MAX_TEMPO
            ]
            if too_long:
                self.stage(f"{meta['name']}: shortening {len(too_long)} lines that ran long")
                for item in self._shorten(lang, state, too_long):
                    self.stage(f"{meta['name']}: speaking line {item['id'] + 1} of {len(items)} again")
                    self._speak_safely(lang, item)
            state["refit_done"] = True
            self.save()

    def _finish_language(self, lang, state):
        """Fit each clip to its time, build the voice track and export the video."""
        meta = LANGUAGES[lang]
        items = state["lines"]
        base = self.dir / "langs" / lang
        (base / "final").mkdir(parents=True, exist_ok=True)
        voiced = [i for i in items if i.get("clip") and not i.get("silent")]
        if not voiced:
            errors = [i.get("error") for i in items if i.get("error")]
            state.update(status="failed", error=errors[0] if errors else f"No usable audio came back for {meta['name']}.")
            self.save()
            return

        self.stage(f"{meta['name']}: fitting the timing")
        for item in voiced:
            item["tempo"] = round(min(HARD_MAX_TEMPO, max(1.0, item["duration_s"] / item["slot_s"])), 3)
            final = base / "final" / f"{item['clip']}_{item['tempo']:.3f}.wav"
            if not final.exists():
                media.retime_clip(base / "clean" / f"{item['clip']}.wav", final, item["tempo"])
            item["final"] = final.name
            item["final_s"] = round(wav_info(final.read_bytes())[0], 3)
        place_lines(items)

        voice_path = base / "voice.wav"
        total = self.job["source"]["duration_s"]
        placements = [(base / "final" / i["final"], i["placed_s"]) for i in voiced]
        cut_off = media.compose_track(placements, total, voice_path)
        state["cut_off_s"] = cut_off

        if self.job["options"].get("check", True):
            self._check_clips(lang, state, voiced)
        for item in items:
            flags = []
            if not item.get("clip"):
                flags.append("failed")
            elif item.get("silent"):
                flags.append("silent")
            else:
                stt = item.get("stt") or {}
                if item.get("tempo", 1) > MAX_TEMPO:
                    flags.append("rushed")
                if item.get("late_s", 0) > LATE_FLAG_S:
                    flags.append("late")
                if stt.get("ok") and stt.get("cer", 0) > CER_CHECK and not has_digits(item["text"]):
                    flags.append("check_speech")
                if stt.get("ok") and not stt.get("language_matches"):
                    flags.append("language")
            item["flags"] = flags

        self.stage(f"{meta['name']}: exporting")
        source = self.job["source"]
        output = base / ("dubbed.mp4" if source["has_video"] else "dubbed.m4a")
        temp = output.with_name(f"building{output.suffix}")
        label = [(f"{meta['name']} (AI-dubbed)", 0.0, total)] if self.job["options"].get("label", True) else []
        state["labelled"] = media.mux(
            self._source_path(), voice_path, temp, source["has_video"],
            float(self.job["options"].get("background", 0.08)), label, total,
        )
        temp.replace(output)
        state.update(
            status="done",
            output=str(output.relative_to(self.dir)).replace("\\", "/"),
            output_rev=sha(voice_path.read_bytes())[:8],
            score=score_language(state),
            measured_chars_per_s=round(self._measured_rate(state), 2),
        )
        self.save()

    def _check_clips(self, lang, state, voiced):
        """Transcribe each final clip with Prisma. This shows whether Prisma understood
        it, not whether it sounds good: only a native speaker can say that."""
        meta = LANGUAGES[lang]
        base = self.dir / "langs" / lang
        style = self.job["options"]["style"]
        for item in voiced:
            if (item.get("stt") or {}).get("for") == item["final"]:
                continue
            self.stage(f"{meta['name']}: checking line {item['id'] + 1} with Prisma")
            try:
                body = self.client.transcribe_clip(
                    (base / "final" / item["final"]).read_bytes(), item["final"], meta["code"],
                    "codemix" if style == "mixed" and meta["code"] != "en-IN" else "verbatim", item["final_s"],
                )
            except ApiError as error:
                item["stt"] = {"ok": False, "status": error.status, "error": error.message, "for": item["final"]}
                self.save()
                continue
            transcript = body.get("transcript") or ""
            kept, total = english_kept(item["text"], transcript)
            item["stt"] = {
                "ok": True,
                "for": item["final"],
                "transcript": transcript,
                "detected": body.get("detected_language_code"),
                "language_matches": body.get("detected_language_code") == meta["code"],
                "cer": round(character_error_rate(item["text"], transcript), 3),
                "english_kept": kept,
                "english_total": total,
            }
            self.save()

    # ------------------------------------------------------------ after the run

    def edit_line(self, lang, line_id, text):
        """Change one dubbed line, speak only that line again and rebuild the video."""
        state = self.job["languages"].get(lang)
        item = next((i for i in state["lines"] if i["id"] == line_id), None) if state else None
        if item is None:
            raise ValueError("That line doesn't exist.")

        def work():
            self._prepare_voice()
            item.update(text=text.strip(), edited=True, shortened=False)
            self.stage(f"{LANGUAGES[lang]['name']}: speaking line {line_id + 1} again")
            self._speak_safely(lang, item)
            self._finish_language(lang, state)
            self.job.update(status="done", stage="Finished")

        return self._guarded(work)

    def set_montage(self, languages, seconds):
        """One video that changes language as it plays: the demo's opening hook."""
        return self._guarded(lambda: self._build_montage(languages, seconds))

    def _build_montage(self, languages, seconds):
        self.stage("Building the language montage")
        tracks = [("Original", self.dir / "source_voice.wav")]
        for lang in languages:
            state = self.job["languages"].get(lang)
            if not state or state.get("status") != "done":
                raise ValueError(f"{LANGUAGES[lang]['name']} hasn't been dubbed yet.")
            tracks.append((f"{LANGUAGES[lang]['name']} (AI-dubbed)", self.dir / "langs" / lang / "voice.wav"))
        total = self.job["source"]["duration_s"]
        cuts = [0.0]
        for line in self.job["lines"]:
            if line["start"] - cuts[-1] >= seconds and total - line["start"] >= 1.5:
                cuts.append(line["start"])
        cuts.append(total)
        windows, labels = [], []
        for index in range(len(cuts) - 1):
            name, path = tracks[index % len(tracks)]
            windows.append((path, cuts[index], cuts[index + 1]))
            labels.append((name, cuts[index], cuts[index + 1]))
        voice = self.dir / "montage_voice.wav"
        media.splice_tracks(windows, voice)
        source = self.job["source"]
        output = self.dir / ("montage.mp4" if source["has_video"] else "montage.m4a")
        temp = output.with_name(f"montage_building{output.suffix}")
        media.mux(
            self._source_path(), voice, temp, source["has_video"],
            float(self.job["options"].get("background", 0.08)), labels, total,
        )
        temp.replace(output)
        self.job["montage"] = {
            "output": output.name,
            "rev": sha(voice.read_bytes())[:8],
            "languages": languages,
            "windows": [{"label": t, "start": round(a, 2), "end": round(b, 2)} for t, a, b in labels],
        }
        self.job.update(status="done", stage="Finished")


def scorecard_markdown(job):
    """The scorecard to share, with the same evidence the app shows."""
    options = job["options"]
    voice = "The creator's cloned voice (vachana-vc-v1)" if options["voice_mode"] == "clone" else f"Timbre v3.0, {options['voice']}"
    lines = [
        f"# Dub scorecard: {job['source_name']}",
        "",
        f"- Transcription: Prisma v3.0, batch (`indic-auto`, `codemix`)",
        f"- Voice: {voice}",
        f"- Style: {'code-mixed' if options['style'] == 'mixed' else 'native words preferred'}",
        "",
        "| Language | Rewritten by | Lines voiced | Prisma character errors | Reviewer rating | English words kept | Mean speed-up | Lines to review |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lang, state in job["languages"].items():
        name = LANGUAGES[lang]["name"]
        score = state.get("score")
        if not score:
            lines.append(f"| {name} | {state.get('rewrite_by') or 'none'} | failed | | | | | |")
            continue
        cer = "not checked" if score["mean_cer"] is None else f"{score['mean_cer']:.0%}"
        rating = state.get("rating")
        lines.append(
            f"| {name} | {state.get('rewrite_by')} | {score['voiced']} of {score['lines']} | {cer} | "
            f"{f'{rating} of 5' if rating else 'not rated'} | "
            f"{score['source_english_kept']} of {score['source_english_total']} | "
            f"{score['mean_tempo']} | {score['to_review']} |"
        )
    lines += [
        "",
        "Prisma's character errors show whether Prisma understood the dub, not whether it sounds good. "
        "Reviewer ratings come from native speakers.",
        "",
    ]
    return "\n".join(lines)
