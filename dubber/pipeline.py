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
from .config import LANGUAGES, MAX_SOURCE_SECONDS, TIMBRE_VOICES
from .gnani import AUDIO_TAGS, CLONE_MODEL, STOCK_MODEL, STT_MODEL, ApiError, GnaniClient, StopRun
from .llm import Gemini, LlmError, build_providers
from .metrics import character_error_rate, english_kept, has_digits, spoken_length, wav_info
from .rewrite import RewriteFailed, adapt_lines, apply_voice_instruction, review_translation, shorten_lines, source_english_kept

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
# Keep a single TTS request small enough for the speech service to handle
# reliably. Long Prisma spans can otherwise become multi-minute requests.
MAX_TTS_LINE_CHARS = 220
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


def apply_pronunciation_dictionary(text, entries):
    """Apply spoken forms to TTS text only; readable translated captions stay intact."""
    result = text
    for entry in sorted(entries or [], key=lambda row: len(str(row.get("term", ""))), reverse=True):
        term, spoken = str(entry.get("term", "")).strip(), str(entry.get("spoken", "")).strip()
        if term and spoken:
            result = re.sub(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lambda _: spoken, result, flags=re.IGNORECASE)
    return result


def srt_timestamp(seconds):
    millis = max(0, round(float(seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    whole, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d},{millis:03d}"


def write_subtitles(path, items):
    entries = []
    for item in items:
        if not item.get("final") or item.get("silent") or item.get("placed_s") is None:
            continue
        text = " ".join(item.get("text", "").replace("-->", "→").split())
        if not text:
            continue
        start = item["placed_s"]
        end = start + item.get("final_s", 0)
        entries.append(f"{len(entries) + 1}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}")
    path.write_text("\n\n".join(entries) + ("\n" if entries else ""), "utf-8")


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


def split_text(text, max_chars=MAX_TTS_LINE_CHARS):
    """Split long text on word boundaries, retaining punctuation and all words."""
    words = re.findall(r"\S+", text.strip())
    if not words:
        return []
    chunks, current = [], ""
    for word in words:
        while len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_failed_long_items(items):
    """Break failed cached dub lines into smaller timed requests for Resume."""
    result = []
    for item in items:
        text_chunks = split_text(item.get("text", "")) if not item.get("clip") else []
        if len(text_chunks) < 2:
            result.append(item)
            continue
        source_chunks = split_text(item.get("source_text", ""), max(1, (len(item.get("source_text", "")) // len(text_chunks)) + 1))
        weights = [max(1, spoken_length(chunk)) for chunk in text_chunks]
        total_weight = sum(weights)
        cursor = float(item["start"])
        duration = float(item["slot_s"])
        for index, (chunk, weight) in enumerate(zip(text_chunks, weights)):
            share = weight / total_weight
            slot = duration * share
            next_item = dict(item)
            next_item.update(
                id=len(result), start=round(cursor, 3), slot_s=round(slot, 3),
                source_text=(
                    " ".join(source_chunks[index:]) if index == len(text_chunks) - 1
                    else source_chunks[index] if index < len(source_chunks) else ""
                ),
                text=chunk, clip=None, duration_s=None,
            )
            for key in ("error", "final", "final_s", "tempo", "placed_s", "late_s", "stt", "silent"):
                next_item.pop(key, None)
            result.append(next_item)
            cursor += slot
    for index, item in enumerate(result):
        item["id"] = index
    return result


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
        "semantic_issues": sum(len(i.get("semantic_issues", [])) for i in items),
        "dub_back_recognized": sum(1 for i in voiced if (i.get("stt") or {}).get("dub_back", {}).get("status") == "recognized"),
        "dub_back_review": sum(1 for i in voiced if (i.get("stt") or {}).get("dub_back", {}).get("status") == "review"),
        "pronunciation_flags": sum(1 for i in items if "pronunciation" in i.get("flags", [])),
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
        self.speaker_voices = {}

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

    def render_comment_response(self, lang, response_id, script):
        def work():
            if lang not in LANGUAGES or not script.strip() or len(script) > 600:
                raise ValueError("Choose a language and keep the response under 600 characters.")
            response = next((row for row in self.job.get("comment_responses", []) if row.get("id") == response_id), None)
            if not response:
                raise ValueError("That comment response no longer exists.")
            self.stage("Preparing the response voice")
            self._prepare_voice()
            folder = self.dir / "responses" / response_id
            folder.mkdir(parents=True, exist_ok=True)
            speech_path = folder / "voice.wav"
            spoken = apply_pronunciation_dictionary(script.strip(), self.job["options"].get("pronunciation_dictionary", []))
            if self.embedding is not None:
                audio = self.client.speak_cloned(spoken, self.embedding)
            else:
                audio = self.client.speak_stock(spoken, self.job["options"]["voice"], self._tts_language(lang))
            wav_info(audio)
            speech_path.write_bytes(audio)
            duration = max(0.1, wav_info(audio)[0])
            subtitle_path = folder / f"response-{lang}.srt"
            write_subtitles(subtitle_path, [{"text": script.strip(), "final": "voice.wav", "placed_s": 0.0, "final_s": duration}])
            source = self.job["source"]
            output = folder / ("response.mp4" if source["has_video"] else "response.m4a")
            temp = folder / ("building.mp4" if source["has_video"] else "building.m4a")
            media.mux(self._source_path(), speech_path, temp, source["has_video"],
                      float(self.job["options"].get("background", 0.08)), [], duration)
            temp.replace(output)
            response.update(status="done", script=script.strip(), output=str(output.relative_to(self.dir)).replace("\\", "/"),
                            subtitles=str(subtitle_path.relative_to(self.dir)).replace("\\", "/"), rev=sha(speech_path.read_bytes())[:8])
            self.save()
        previous_status = self.job.get("status", "done")
        result = self._guarded(work)
        if self.job.get("status") in {"failed", "stopped"}:
            response = next((row for row in self.job.get("comment_responses", []) if row.get("id") == response_id), None)
            if response:
                response.update(status="failed", error=self.job.get("error") or self.job.get("stage"))
        self.job.update(status="done" if any(state.get("status") == "done" for state in self.job["languages"].values()) else previous_status,
                        stage="Finished", error=None)
        self.save()
        return result

    def set_final_audio_mix(self, lang, keep_source_audio, background):
        def work():
            state = self.job["languages"].get(lang)
            if not state or state.get("status") != "done":
                raise ValueError("Choose a language with a finished dub.")
            source = self.job["source"]
            output = self.dir / state["output"]
            temp = output.with_name("rebuilding" + output.suffix)
            voice = self.dir / "langs" / lang / "voice.wav"
            if not voice.is_file():
                raise ValueError("The finished dub audio track is missing; resume the job before rebuilding its export.")
            total = float(state.get("timeline_s") or source["duration_s"])
            labels = [(f"{LANGUAGES[lang]['name']} (AI-dubbed)", 0.0, total)] if self.job["options"].get("label", True) else []
            self.stage(f"Rebuilding {LANGUAGES[lang]['name']} export")
            labelled = media.mux(self._source_path(), voice, temp, source["has_video"], background, labels, total)
            temp.replace(output)
            state["keep_source_audio"] = bool(keep_source_audio)
            state["background"] = float(background)
            self.job["montage"] = None
            state["labelled"] = labelled
            state["output_rev"] = sha(voice.read_bytes() + f"|{int(bool(keep_source_audio))}|{background:.3f}".encode("ascii"))[:8]
            self.job.update(status="done", stage="Finished", error=None)
            self.save()
        return self._guarded(work)

    def run(self):
        return self._guarded(self._run)

    def _run(self):
        self._prepare()
        self._transcribe()
        self._analyze_source_delivery()
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
        speech_path = self.dir / "speech.mp3"
        speech_info = None
        if speech_path.is_file() and speech_path.stat().st_size > 0:
            try:
                speech_info = media.probe(speech_path)
            except media.MediaError:
                speech_info = None
        if not speech_info or not speech_info["has_audio"] or speech_info["duration_s"] <= 0:
            cleanup = self.job["options"].get("audio_cleanup", "auto")
            building_speech = self.dir / "speech.building.mp3"
            building_speech.unlink(missing_ok=True)
            media.extract_speech(self.dir / file_name, building_speech, cleanup)
            if not building_speech.is_file() or building_speech.stat().st_size == 0:
                building_speech.unlink(missing_ok=True)
                raise ValueError("Could not prepare an audio track for transcription. Check that the uploaded video contains audible sound, then try again.")
            try:
                prepared_info = media.probe(building_speech)
            except media.MediaError as error:
                building_speech.unlink(missing_ok=True)
                raise ValueError("The audio track prepared for transcription could not be read. Try the upload again.") from error
            if not prepared_info["has_audio"] or prepared_info["duration_s"] <= 0:
                building_speech.unlink(missing_ok=True)
                raise ValueError("Could not prepare a usable audio track for transcription. Try the upload again.")
            building_speech.replace(speech_path)
            self.job["source"]["transcription_cleanup"] = cleanup
            if cleanup == "noisy":
                self.job["warnings"].append(
                    "Strong noise reduction was used for transcription. Review names and quiet speech in the transcript."
                )
            elif cleanup == "phone":
                self.job["warnings"].append(
                    "Phone/VoIP filtering was used for transcription; it can remove speech outside the telephone frequency range."
                )
        if not (self.dir / "source_voice.wav").exists():
            media.extract_voice_track(self.dir / file_name, self.dir / "source_voice.wav")
        self.save()

    def _analyze_source_delivery(self):
        """Optional Gemini audio pass estimates per-line emotion and delivery."""
        options = self.job["options"]
        if not options.get("emotion_analysis") or self.job.get("emotion_analysis_done"):
            return
        if not self.settings.gemini_key:
            self.warn("Source emotion analysis was skipped: add GEMINI_API_KEY and enable it for a future dub.")
            self.job["emotion_analysis_done"] = True
            self.save()
            return
        audio = (self.dir / "speech.mp3").read_bytes()
        if len(audio) > 14_000_000:
            self.warn("Source emotion analysis was skipped because the compressed speech track exceeds Gemini's inline audio limit.")
            self.job["emotion_analysis_done"] = True
            self.save()
            return
        self.stage("Analyzing the source speaker's delivery with Gemini")
        analyzer = Gemini(self.settings.gemini_base_url, self.settings.gemini_key, self.settings.gemini_model)
        self.client.usage["llm_requests"] += 1
        prompt = """Listen to the attached speech audio and estimate delivery per spoken segment. Use the timestamps and transcript lines provided below to align your estimates. Do not infer emotion from words alone. Use only these TTS delivery tags: none, excited, whispers, shouting, laugh, giggle, sigh, slow, pauses. Map sadness/softness to sigh or slow, excitement to excited, obvious laughter to laugh/giggle, and otherwise prefer none. Do not label ordinary emphasis as shouting. Return JSON only: {\"segments\":[{\"start_s\":0.0,\"end_s\":1.0,\"emotion\":\"excited|sarcastic|angry|sad|humorous|calm|neutral\",\"delivery_tag\":\"none\",\"confidence\":0.0}]}. Keep segments aligned to speaker delivery changes; use seconds from the beginning of the audio."""
        prompt += "\nTranscript for alignment: " + json.dumps(
            [{"start_s": line["start"], "end_s": line["end"], "text": line["text"]} for line in self.job["lines"]],
            ensure_ascii=False,
        )
        try:
            raw = analyzer.analyze_audio(audio, "audio/mpeg", prompt)
            text = re.sub(r"```(?:json)?", "", raw).strip()
            analysis = json.loads(text[text.find("{"):text.rfind("}") + 1])
            if not isinstance(analysis, dict):
                raise ValueError("Gemini returned an unexpected audio-analysis shape.")
            segments = analysis.get("segments") or []
            if not isinstance(segments, list):
                raise ValueError("Gemini returned invalid emotion segments.")
        except (LlmError, ValueError, TypeError, KeyError) as error:
            self.warn(f"Source emotion analysis failed; transcript-based delivery suggestions will be used ({error}).")
            self.job["emotion_analysis_done"] = True
            self.save()
            return
        allowed_tags = AUDIO_TAGS | {"none"}
        for line in self.job["lines"]:
            candidates = []
            for segment in segments:
                try:
                    start, end = float(segment["start_s"]), float(segment["end_s"])
                    confidence = float(segment.get("confidence", 0.0))
                except (KeyError, TypeError, ValueError):
                    continue
                overlap = max(0.0, min(line["end"], end) - max(line["start"], start))
                if overlap and confidence >= 0.45:
                    candidates.append((overlap, confidence, segment))
            if candidates:
                _, confidence, segment = max(candidates, key=lambda row: row[0] * row[1])
                tag = segment.get("delivery_tag", "none")
                line["source_emotion"] = str(segment.get("emotion", "neutral"))[:30]
                line["emotion_confidence"] = round(confidence, 2)
                line["emotion_analyzed"] = True
                if tag in allowed_tags:
                    line["delivery_tag"] = tag
        self.job["emotion_analysis_done"] = True
        if options.get("voice_mode") == "clone" and any(line.get("delivery_tag") not in (None, "none") for line in self.job["lines"]):
            self.warn("Source delivery was analyzed, but the cloned-voice endpoint doesn't accept Timbre emotion tags; use a Timbre voice to apply them.")
        self.save()

    def _transcribe(self):
        if self.job["lines"]:
            return
        if self.job.get("direct_transcription"):
            self._transcribe_short_clip_directly()
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
            options = self.job["options"]
            if options.get("voice_mode") == "clone":
                self.warn(f"Prisma found {len(speakers)} speakers. The selected cloned voice will be used for all of them.")
            elif not options.get("distinct_speakers", False):
                self.warn(f"Prisma found {len(speakers)} speakers. Turn on distinct Timbre voices to keep them separate.")
        self.save()

    def _transcribe_short_clip_directly(self):
        """Fallback for short clips when the network cannot reach batch result storage."""
        duration = float(self.job["source"]["duration_s"])
        if duration > 30:
            raise ValueError("Direct transcription is only available for clips up to 30 seconds.")
        self.stage("Transcribing short clip directly with Prisma v3.0")
        document = self.client.transcribe_clip(
            (self.dir / "speech.mp3").read_bytes(), "speech.mp3", "indic-auto", "codemix", duration,
            self.job["options"].get("keyterms") or [],
        )
        transcript = (document.get("transcript") or "").strip()
        if not transcript:
            raise ValueError("Prisma returned no speech for this clip.")
        # REST returns text without batch segment timestamps. Estimate sentence
        # slots and make the approximation visible to the creator.
        sentences = []
        for sentence in re.findall(r"[^.!?।॥]+(?:[.!?।॥]+|$)", transcript):
            sentences.extend(split_text(sentence.strip(), 140))
        if not sentences:
            sentences = [transcript]
        weights = [max(1, len(re.findall(r"\w+", sentence))) for sentence in sentences]
        total_weight = sum(weights)
        cursor = 0.0
        segments = []
        for index, (sentence, weight) in enumerate(zip(sentences, weights)):
            end = duration if index == len(sentences) - 1 else cursor + duration * weight / total_weight
            segments.append({"start_time": round(cursor, 3), "end_time": round(end, 3), "text": sentence, "speaker_id": 0})
            cursor = end
        lines = merge_segments(segments, duration)
        if not lines:
            raise ValueError("Prisma returned no usable speech for this clip.")
        self.job["lines"] = lines
        self.job["source_language"] = document.get("detected_language_code")
        self.job["transcription_mode"] = "direct_short_clip"
        self.warn("Direct transcription was used because batch result storage was unreachable. Sentence timings are estimated; review them in Transcript & review.")
        self.save()

    # ------------------------------------------------------------ voice

    def _prepare_voice(self):
        options = self.job["options"]
        if options["voice_mode"] != "clone":
            speaker_ids = list(dict.fromkeys(
                str(line["speaker"]) for line in self.job.get("lines", [])
                if line.get("speaker") not in (None, -1)
            ))
            self.speaker_voices = {}
            if options.get("distinct_speakers", False) and speaker_ids:
                selected = options["voice"]
                gender = TIMBRE_VOICES[selected]
                voice_pool = [selected] + [
                    name for name, voice_gender in TIMBRE_VOICES.items()
                    if voice_gender == gender and name != selected
                ]
                self.speaker_voices = {
                    speaker: voice_pool[index % len(voice_pool)]
                    for index, speaker in enumerate(speaker_ids)
                }
                if len(speaker_ids) > len(voice_pool):
                    self.warn(
                        f"Prisma found {len(speaker_ids)} speakers, but only {len(voice_pool)} {gender.lower()} Timbre voices are available; some voices are reused."
                    )
            options["speaker_voices"] = self.speaker_voices
            if len(self.speaker_voices) > 1:
                voice_identity = json.dumps(self.speaker_voices, sort_keys=True)
                self.voice_id = f"{STOCK_MODEL}:speakers:{sha(voice_identity.encode('utf-8'))[:16]}"
            else:
                # Keep the old identity so resuming earlier jobs can reuse paid clips.
                self.voice_id = f"{STOCK_MODEL}:{options['voice']}"
            self.save()
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
        spoken_text = item.get("pronunciation", "").strip() or apply_pronunciation_dictionary(
            item["text"], self.job["options"].get("pronunciation_dictionary", [])
        )
        speech_text = f"<{delivery_tag}>{spoken_text}" if delivery_tag else spoken_text
        stock_voice = self.speaker_voices.get(str(item.get("speaker")), self.job["options"]["voice"])
        voice_identity = (
            self.voice_id if self.embedding is not None or len(self.speaker_voices) <= 1
            else f"{self.voice_id}:{stock_voice}"
        )
        key = sha(f"{voice_identity}|{language}|{speech_text}".encode("utf-8"))[:16]
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
                audio = self.client.speak_cloned(spoken_text, self.embedding)
            else:
                audio = self.client.speak_stock(
                    spoken_text, stock_voice, language, delivery_tag
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
            spoken_text=spoken_text,
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
                self.job["options"].get("localization_brief", ""), self.job["options"].get("glossary", []),
                self.job["options"].get("region_dialect", ""),
                self.job["options"].get("code_mix_percent", 65),
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
        if (state and state.get("status") == "done" and state.get("voice_id") == self.voice_id
                and all(item.get("clip") for item in state.get("lines", []))):
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
                    options.get("localization_brief", ""), options.get("glossary", []),
                    options.get("region_dialect", ""),
                    options.get("code_mix_percent", 65),
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
                    if line.get("speaker") not in (None, -1):
                        item["speaker"] = line["speaker"]
                    if expressive:
                        item["delivery_tag"] = line.get("delivery_tag") if line.get("emotion_analyzed") else rewritten["delivery_tag"]
                    if line.get("source_emotion"):
                        item["source_emotion"] = line["source_emotion"]
                    state["lines"].append(item)
                if options.get("semantic_check", True):
                    self.stage(f"{meta['name']}: checking translation meaning")
                    self.client.usage["llm_requests"] += (len(state["lines"]) + 19) // 20
                    try:
                        issues = review_translation(
                            self.providers, state["lines"], meta,
                            options.get("localization_brief", ""), options.get("glossary", []),
                            options.get("region_dialect", ""),
                        )
                        for item in state["lines"]:
                            item["semantic_issues"] = issues.get(item["id"], [])
                    except RewriteFailed as error:
                        state["notes"].append(str(error))
                self.save()
            # Older saved jobs may contain one failed, multi-minute synthesis
            # request. Split that translated text when the user resumes it.
            state["lines"] = split_failed_long_items(state["lines"])
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
            natural_tempo = max(1.0, item["duration_s"] / item["slot_s"])
            requested_speed = min(1.2, max(0.8, float(item.get("speech_speed", 1.0))))
            item["tempo"] = round(min(HARD_MAX_TEMPO, max(0.7, natural_tempo * requested_speed)), 3)
            final = base / "final" / f"{item['clip']}_{item['tempo']:.3f}.wav"
            if not final.exists():
                media.retime_clip(base / "clean" / f"{item['clip']}.wav", final, item["tempo"])
            item["final"] = final.name
            item["final_s"] = round(wav_info(final.read_bytes())[0], 3)
        place_lines(items)
        for item in items:
            item.pop("pause_gap_s", None)
        ordered = sorted((item for item in voiced if item.get("placed_s") is not None), key=lambda item: item["placed_s"])
        for previous, current in zip(ordered, ordered[1:]):
            output_gap = max(0.0, current["placed_s"] - previous["placed_s"] - previous["final_s"])
            source_end = previous["start"] + max(0.0, previous["slot_s"] - 1.0)
            source_gap = max(0.0, current["start"] - source_end)
            if output_gap - source_gap > 1.25:
                current["pause_gap_s"] = round(output_gap - source_gap, 2)

        subtitle_path = base / f"subtitles-{lang}.srt"
        write_subtitles(subtitle_path, items)

        voice_path = base / "voice.wav"
        source_total = self.job["source"]["duration_s"]
        last_speech_end = max((i["placed_s"] + i["final_s"] for i in voiced), default=source_total)
        total = max(source_total, round(last_speech_end, 3))
        state["timeline_s"] = round(total, 3)
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
                if stt.get("ok") and stt.get("cer", 0) > CER_CHECK and not has_digits(item.get("spoken_text") or item["text"]):
                    flags.append("pronunciation" if item.get("pronunciation") else "check_speech")
                if stt.get("ok") and not stt.get("language_matches"):
                    flags.append("language")
                if stt.get("ok") and stt.get("dub_back", {}).get("status") == "review":
                    flags.append("dub_back_review")
            if item.get("semantic_issues"):
                flags.append("meaning")
            if item.get("pause_gap_s", 0) > 1.25:
                flags.append("long_pause")
            item["flags"] = flags

        self.stage(f"{meta['name']}: exporting")
        source = self.job["source"]
        output = base / ("dubbed.mp4" if source["has_video"] else "dubbed.m4a")
        temp = output.with_name(f"building{output.suffix}")
        label = [(f"{meta['name']} (AI-dubbed)", 0.0, total)] if self.job["options"].get("label", True) else []
        source_audio = bool(self.job["options"].get("keep_source_audio", float(self.job["options"].get("background", 0.08)) > 0))
        source_level = float(self.job["options"].get("background", 0.08)) if source_audio else 0.0
        state["labelled"] = media.mux(
            self._source_path(), voice_path, temp, source["has_video"],
            source_level, label, total,
        )
        temp.replace(output)
        state.update(
            status="done",
            output=str(output.relative_to(self.dir)).replace("\\", "/"),
            subtitles=str(subtitle_path.relative_to(self.dir)).replace("\\", "/"),
            output_rev=sha(voice_path.read_bytes())[:8],
            keep_source_audio=source_audio,
            background=source_level,
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
            checked_text = item.get("spoken_text") or item["text"]
            kept, total = english_kept(checked_text, transcript)
            cer = round(character_error_rate(checked_text, transcript), 3)
            language_matches = body.get("detected_language_code") == meta["code"]
            reasons = []
            if cer > CER_CHECK:
                reasons.append("Prisma heard wording that differs from the expected speech.")
            if not language_matches:
                reasons.append("Prisma detected a different language.")
            item["stt"] = {
                "ok": True,
                "for": item["final"],
                "transcript": transcript,
                "detected": body.get("detected_language_code"),
                "language_matches": language_matches,
                "cer": cer,
                "dub_back": {
                    "status": "recognized" if cer <= CER_CHECK and language_matches else "review",
                    "text_similarity_pct": round(max(0.0, 1.0 - cer) * 100),
                    "expected": checked_text,
                    "heard": transcript,
                    "reasons": reasons,
                    "manual_number_check": has_digits(checked_text),
                },
                "english_kept": kept,
                "english_total": total,
            }
            self.save()

    # ------------------------------------------------------------ after the run

    def edit_line(self, lang, line_id, text, pronunciation="", delivery_tag="none", speech_speed=1.0):
        """Change one dubbed line, speak only that line again and rebuild the video."""
        state = self.job["languages"].get(lang)
        item = next((i for i in state["lines"] if i["id"] == line_id), None) if state else None
        if item is None:
            raise ValueError("That line doesn't exist.")
        return self._guarded(lambda: self._apply_line_edit(lang, line_id, state, item, text, pronunciation, delivery_tag, speech_speed))

    def voice_direct_edit(self, lang, line_id, audio, duration_s):
        """Use Prisma to hear a creator instruction, then revise and regenerate one line."""
        state = self.job["languages"].get(lang)
        item = next((i for i in state["lines"] if i["id"] == line_id), None) if state else None
        if item is None:
            raise ValueError("That line doesn't exist.")
        if duration_s > 30:
            raise ValueError("Keep a voice direction under 30 seconds.")

        def work():
            self.stage("Listening to your edit with Prisma")
            heard = self.client.transcribe_clip(audio, "voice-direction.mp3", "indic-auto", "codemix", duration_s)
            instruction = (heard.get("transcript") or "").strip()
            if not instruction:
                raise ValueError("Prisma couldn't hear an edit instruction. Try again in a quieter place.")
            self.stage("Applying your direction to the selected line")
            self.client.usage["llm_requests"] += 1
            edit = apply_voice_instruction(
                self.providers, instruction, item, LANGUAGES[lang], self.job["options"]["style"],
                self.job["options"].get("code_mix_percent", 65), self.job["options"].get("glossary", []),
            )
            item["voice_director_instruction"] = instruction[:500]
            self._apply_line_edit(
                lang, line_id, state, item, edit["text"], edit["pronunciation"],
                edit["delivery_tag"], edit["speech_speed"],
            )

        return self._guarded(work)

    def _apply_line_edit(self, lang, line_id, state, item, text, pronunciation, delivery_tag, speech_speed):
        self._prepare_voice()
        safe_tag = delivery_tag if delivery_tag in AUDIO_TAGS | {"none"} else item.get("delivery_tag", "none")
        try:
            safe_speed = min(1.2, max(0.8, float(speech_speed)))
        except (TypeError, ValueError):
            safe_speed = float(item.get("speech_speed", 1.0))
        item.update(
            text=text.strip(), pronunciation=str(pronunciation or "").strip()[:240], delivery_tag=safe_tag,
            speech_speed=safe_speed, edited=True, shortened=False,
        )
        item.pop("clip", None)
        item.pop("stt", None)
        item.pop("final", None)
        if self.job["options"].get("semantic_check", True):
            try:
                self.client.usage["llm_requests"] += 1
                issues = review_translation(
                    self.providers, [item], LANGUAGES[lang],
                    self.job["options"].get("localization_brief", ""),
                    self.job["options"].get("glossary", []),
                    self.job["options"].get("region_dialect", ""),
                )
                item["semantic_issues"] = issues.get(line_id, [])
            except RewriteFailed as error:
                state.setdefault("notes", []).append(str(error))
        self.stage(f"{LANGUAGES[lang]['name']}: speaking line {line_id + 1} again")
        self._speak_safely(lang, item)
        self._finish_language(lang, state)
        self.job.update(status="done", stage="Finished")

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
        "| Language | Rewritten by | Lines voiced | DubBack recognized | Prisma character errors | Meaning flags | Pronunciation flags | Reviewer rating | English words kept | Mean speed-up | Lines to review |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lang, state in job["languages"].items():
        name = LANGUAGES[lang]["name"]
        score = state.get("score")
        if not score:
            lines.append(f"| {name} | {state.get('rewrite_by') or 'none'} | failed | | | | | | | | |")
            continue
        cer = "not checked" if score["mean_cer"] is None else f"{score['mean_cer']:.0%}"
        rating = state.get("rating")
        lines.append(
            f"| {name} | {state.get('rewrite_by')} | {score['voiced']} of {score['lines']} | "
            f"{(str(score['dub_back_recognized']) + ' of ' + str(score.get('checked', 0))) if 'dub_back_recognized' in score else 'not available'} | {cer} | "
            f"{score.get('semantic_issues', 0)} | {score.get('pronunciation_flags', 0)} | "
            f"{f'{rating} of 5' if rating else 'not rated'} | "
            f"{score['source_english_kept']} of {score['source_english_total']} | "
            f"{score['mean_tempo']} | {score['to_review']} |"
        )
    lines += [
        "",
        "DubBack compares each generated clip with Prisma's transcript; a match is recognition evidence, not proof of semantic or artistic quality. Listen closely to names and numbers. "
        "Meaning flags are model suggestions for human review. Pronunciation flags compare Prisma's hearing with the requested speech spelling. Reviewer ratings come from native speakers.",
        "",
    ]
    return "\n".join(lines)
