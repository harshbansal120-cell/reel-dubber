"""Client for Gnani's speech APIs, written from docs.gnani.ai."""

import json
import threading
import time

import requests

CLONE_MODEL = "vachana-vc-v1"
STOCK_MODEL = "timbre-v3.0"
STT_MODEL = "prisma-v3.0"
AUDIO_TAGS = {"excited", "whispers", "shouting", "laugh", "giggle", "sigh", "slow", "pauses"}
AUDIO_CONFIG = {
    "sample_rate": 24000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}

# The docs ask for retries on these, with backoff. 429 has no Retry-After header.
RETRYABLE = {429, 500, 503, 504}
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.0
REQUEST_TIMEOUT = 120

# Some accounts are limited to one active request at a time.
_one_at_a_time = threading.Lock()


class StopRun(Exception):
    """Carrying on would fail the same way: bad key or no credits."""


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def error_message(response):
    """Pull the message out of the several error shapes the API uses."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return str(body)[:300]
    for field in ("error", "detail"):
        value = body.get(field)
        if isinstance(value, dict) and value.get("message"):
            return str(value["message"])[:300]
        if isinstance(value, list):
            return str(value)[:300]
    return str(body.get("message") or body)[:300]


def find_embedding(body):
    """The reference nests the embedding under data; accept the flatter forms too."""
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    for candidate in (data.get("voice_clone_embedding"), body.get("voice_clone_embedding"), data, body):
        if isinstance(candidate, dict) and candidate.get("embedding"):
            return {name: candidate.get(name) for name in ("embedding", "shape", "dtype")}
    raise ApiError(200, "The embeddings response had no embedding in it.")


class GnaniClient:
    def __init__(self, base_url, key, usage):
        self.base_url = base_url
        self.key = key
        self.usage = usage
        self.session = requests.Session()

    def _request(self, method, path, **kwargs):
        if not self.key:
            raise StopRun("No Gnani API key is set. Add GNANI_API_KEY to the .env file, then restart.")
        delay = BACKOFF_BASE
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last_try = attempt == MAX_ATTEMPTS
            try:
                with _one_at_a_time:
                    response = self.session.request(
                        method,
                        self.base_url + path,
                        headers={"X-API-Key-ID": self.key},
                        timeout=REQUEST_TIMEOUT,
                        **kwargs,
                    )
            except (requests.ConnectionError, requests.Timeout) as error:
                if last_try:
                    raise ApiError(0, f"Could not reach the Gnani API ({type(error).__name__}).")
            else:
                if response.status_code == 401:
                    raise StopRun("Gnani rejected the API key (401). Check GNANI_API_KEY in the .env file.")
                if response.status_code == 402:
                    raise StopRun("You are out of Gnani credits (402). Finished work is saved.")
                if response.ok:
                    return response
                if response.status_code not in RETRYABLE or last_try:
                    raise ApiError(response.status_code, error_message(response).replace(self.key, "***"))
            self.usage["retries"] += 1
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def create_embedding(self, audio, filename, mime):
        response = self._request(
            "POST", "/api/v1/tts/voice-clone/embeddings", files={"audio_file": (filename, audio, mime)}
        )
        self.usage["embedding_requests"] += 1
        return find_embedding(response.json())

    def _tts(self, payload):
        response = self._request("POST", "/api/v1/tts/inference", json=payload)
        self.usage["tts_requests"] += 1
        self.usage["tts_chars"] += len(payload["text"])
        return response.content

    def speak_cloned(self, text, embedding):
        return self._tts(
            {"text": text, "model": CLONE_MODEL, "audio_config": AUDIO_CONFIG, "speaker_embedding": embedding}
        )

    def speak_stock(self, text, voice, language, delivery_tag=None):
        if delivery_tag in AUDIO_TAGS:
            text = f"<{delivery_tag}>{text}"
        return self._tts(
            {
                "text": text,
                "voice": voice,
                "model": STOCK_MODEL,
                "language": language,
                "speed": 1.0,
                "audio_config": AUDIO_CONFIG,
            }
        )

    def transcribe_clip(self, audio, filename, language_code, output_format, seconds, keyterms=None):
        mime = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
        data = {"model": STT_MODEL, "language_code": language_code, "format": output_format}
        if keyterms:
            data["keyterms"] = json.dumps(keyterms[:200], ensure_ascii=False)
        response = self._request(
            "POST",
            "/stt/v3",
            files={"audio_file": (filename, audio, mime)},
            data=data,
        )
        self.usage["stt_requests"] += 1
        self.usage["stt_audio_s"] = round(self.usage["stt_audio_s"] + seconds, 2)
        return response.json()

    def create_batch(self, audio, filename, config):
        response = self._request(
            "POST",
            "/stt/v3/batch/jobs",
            files={"config": (None, json.dumps(config), "application/json"), "files": (filename, audio)},
        )
        self.usage["batch_jobs"] += 1
        return response.json()["job_id"]

    def start_batch(self, job_id):
        self._request("POST", f"/stt/v3/batch/jobs/{job_id}/start")

    def batch_status(self, job_id):
        return self._request("GET", f"/stt/v3/batch/jobs/{job_id}").json()

    def batch_files(self, job_id):
        return self._request("GET", f"/stt/v3/batch/jobs/{job_id}/files").json().get("data") or []

    @staticmethod
    def fetch_transcript(url):
        """The transcript link is a pre-signed storage URL, so it must not get the API key."""
        # Do not include the URL in errors: it contains a temporary AWS
        # signature. Retry transient object-store timeouts without resubmitting
        # the already completed (and billable) batch job.
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=(20, 90))
                response.raise_for_status()
                return response.json()
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise ApiError(
                        0,
                        "Prisma finished transcription, but the transcript could not be downloaded from Gnani's result storage. "
                        "Check that outbound HTTPS to gnaniasraudios.s3.amazonaws.com is allowed, then press Resume. "
                        "Resume reuses the completed batch instead of submitting transcription again.",
                    ) from None
                time.sleep(attempt + 1)
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else 0
                if status in (401, 403, 404):
                    raise ApiError(
                        status,
                        "The transcript download link expired or was rejected. Press Resume to request a fresh result link; "
                        "the completed transcription batch will be reused.",
                    ) from None
                raise ApiError(status, "Gnani returned an error while downloading the completed transcript.") from None
            except ValueError:
                raise ApiError(0, "Gnani's transcript download was not valid JSON. Press Resume to fetch it again.") from None
