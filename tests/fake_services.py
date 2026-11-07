"""Local stand-ins for Gnani and the rewrite models, built from their docs.

They prove this app's logic, not how the real APIs or voices behave.
"""

import email.parser
import email.policy
import io
import json
import math
import re
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

KEY = "test-key-do-not-leak"
GEMINI_KEY = "test-gemini-key-do-not-leak"
EVON_KEY = "test-evon-key-do-not-leak"
EMBEDDING = {"embedding": "ZmFrZS1lbWJlZGRpbmc=", "shape": [1, 768], "dtype": "torch.bfloat16"}
SAMPLE_CHAR = {"hi-IN": "क", "ta-IN": "த", "bn-IN": "ব", "te-IN": "త", "kn-IN": "ಕ", "ml-IN": "ക", "mr-IN": "क", "gu-IN": "ક", "pa-IN": "ਕ", "en-IN": "a"}
DEFAULT_SEGMENTS = [
    {"segment_id": 0, "start_time": 0.4, "end_time": 3.0, "text": "Friends, आज का video", "speaker_id": 0, "language_detected": "hi-IN"},
    {"segment_id": 1, "start_time": 3.1, "end_time": 5.5, "text": "थोड़ा different है।", "speaker_id": 0, "language_detected": "hi-IN"},
    {"segment_id": 2, "start_time": 6.5, "end_time": 9.5, "text": "पूरा budget एक week में plan किया।", "speaker_id": 0, "language_detected": "hi-IN"},
    {"segment_id": 3, "start_time": 10.0, "end_time": 13.0, "text": "आप भी try करो।", "speaker_id": 0, "language_detected": "hi-IN"},
    {"segment_id": 4, "start_time": 13.1, "end_time": 13.3, "text": "", "speaker_id": -1, "language_detected": None},
]


def tone_wav(seconds, frequency=220.0, silent=False):
    rate = 24000
    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = 0 if silent else int(9000 * math.sin(2 * math.pi * frequency * i / rate))
        frames += value.to_bytes(2, "little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


class State:
    def __init__(self):
        self.calls = []
        self.tts_texts = []
        self.tts_payloads = []
        self.chars_per_second = 13.0
        self.segments = list(DEFAULT_SEGMENTS)
        self.language_code = "hi-IN"
        self.batch_config = None
        self.batch_config_type = None
        self.batch_polls = 0
        self.batch_fail = None
        self.transcript_headers = None
        self.fail_tts_if = lambda text: False
        self.silent_tts_if = lambda text: False
        self.rate_limit_once = False
        self.credits_after = None
        self.embedding_error = None
        self.stt_override = None
        self.llm_stretch = 1.0  # Makes first answers too long, to trigger shortening.
        self.llm_shorten_requests = 0
        self.gemini_wrong_language_times = 0
        self.evon_wrong_language_times = 0
        self.gemini_error = None
        self.llm_calls = {"gemini": 0, "evon": 0}
        self.llm_prompts = []


def fake_rewrite(state, user_text, model_name):
    """Answer a rewrite request the way a model would, in the requested script."""
    payload = json.loads(user_text[: user_text.rfind("}") + 1] if "problem" not in user_text else user_text.split("\n\nYour last answer")[0])
    state.llm_prompts.append(payload)
    char = SAMPLE_CHAR[payload["language_code"]]
    shortening = "task" in payload
    if shortening:
        state.llm_shorten_requests += 1
    lines = []
    for line in payload["lines"]:
        budget = line["max_chars"]
        size = int(budget if shortening else budget * state.llm_stretch)
        filler = max(size - len("budget"), 2)
        lines.append({"id": line["id"], "text": "budget " + char * filler})
    wrong_field = f"{model_name}_wrong_language_times"
    if getattr(state, wrong_field) > 0:
        setattr(state, wrong_field, getattr(state, wrong_field) - 1)
        lines = [{"id": line["id"], "text": "This answer came back in English only"} for line in payload["lines"]]
    return json.dumps({"lines": lines}, ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    @property
    def state(self):
        return self.server.state

    def respond(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def multipart(self, body):
        header = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
        message = email.parser.BytesParser(policy=email.policy.default).parsebytes(header + body)
        fields = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            fields[name] = (part.get_filename(), part.get_content_type(), part.get_payload(decode=True))
        return fields

    def authorised(self):
        if self.headers.get("X-API-Key-ID") == KEY:
            return True
        self.respond(401, {"detail": {"error_code": "INVALID_API_KEY", "message": "Unauthorized", "status_code": 401}})
        return False

    def do_GET(self):
        path = urlparse(self.path).path
        self.state.calls.append(("GET", path))
        if path == "/results/transcript.json":
            self.state.transcript_headers = dict(self.headers)
            spoken = [s["text"] for s in self.state.segments if s["text"]]
            self.respond(200, {
                "language_code": self.state.language_code,
                "duration_seconds": "14.0",
                "model": "prisma-v3.0",
                "segments": self.state.segments,
                "full_transcript": " ".join(spoken),
            })
            return
        if not self.authorised():
            return
        if path.endswith("/files"):
            if self.state.batch_fail:
                self.respond(200, {"data": [{"status": "FAILED", "transcript_url": None, "error_message": self.state.batch_fail}]})
            else:
                host = self.headers["Host"]
                self.respond(200, {"data": [{"status": "COMPLETED", "transcript_url": f"http://{host}/results/transcript.json"}]})
        elif path.startswith("/stt/v3/batch/jobs/"):
            self.state.batch_polls += 1
            if self.state.batch_fail:
                self.respond(200, {"status": "FAILED"})
            else:
                self.respond(200, {"status": "IN_PROGRESS" if self.state.batch_polls < 2 else "COMPLETED"})
        else:
            self.respond(404, {"message": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        raw = self.body()
        self.state.calls.append(("POST", path))
        if path.endswith(":generateContent"):
            self.gemini(json.loads(raw))
        elif path == "/v1/chat/completions":
            self.evon(json.loads(raw))
        elif not self.authorised():
            return
        elif path == "/api/v1/tts/voice-clone/embeddings":
            if self.state.embedding_error:
                self.respond(self.state.embedding_error, {"success": False, "message": "We are facing technical difficulties.", "status_code": self.state.embedding_error})
            elif "audio_file" not in self.multipart(raw):
                self.respond(422, {"detail": [{"msg": "Field required"}]})
            else:
                self.respond(200, {"success": True, "data": {"voice_clone_embedding": EMBEDDING}})
        elif path == "/api/v1/tts/inference":
            self.tts(json.loads(raw))
        elif path == "/stt/v3":
            self.stt(self.multipart(raw))
        elif path == "/stt/v3/batch/jobs":
            fields = self.multipart(raw)
            self.state.batch_config_type = fields["config"][1]
            self.state.batch_config = json.loads(fields["config"][2])
            self.respond(201, {"job_id": "job-1", "status": "CREATED"})
        elif path == "/stt/v3/batch/jobs/job-1/start":
            self.respond(202, {"job_id": "job-1", "status": "STARTING"})
        else:
            self.respond(404, {"message": "not found"})

    def gemini(self, request):
        state = self.state
        state.llm_calls["gemini"] += 1
        if self.headers.get("x-goog-api-key") != GEMINI_KEY:
            self.respond(403, {"error": {"message": "bad key"}})
            return
        if state.gemini_error:
            self.respond(state.gemini_error, {"error": {"message": "unavailable"}})
            return
        user = request["contents"][0]["parts"][0]["text"]
        text = fake_rewrite(state, user, "gemini")
        self.respond(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})

    def evon(self, request):
        state = self.state
        state.llm_calls["evon"] += 1
        if request.get("chat_template_kwargs") != {"enable_thinking": False}:
            self.respond(400, {"message": "expected reasoning off"})
            return
        text = fake_rewrite(state, request["messages"][1]["content"], "evon")
        self.respond(200, {"choices": [{"message": {"content": "<think>hidden</think>```json\n" + text + "\n```"}}]})

    def tts(self, request):
        state = self.state
        text = request.get("text")
        if not text or not request.get("audio_config"):
            self.respond(400, {"success": False, "message": "Invalid request: Field required", "status_code": 400})
            return
        model = request.get("model")
        if model == "vachana-vc-v1":
            if request.get("speaker_embedding") != EMBEDDING:
                self.respond(500, {"success": False, "message": "We are facing technical difficulties.", "status_code": 500})
                return
        elif model == "timbre-v3.0":
            if not request.get("voice") or not request.get("language"):
                self.respond(400, {"success": False, "message": "Invalid request: voice and language", "status_code": 400})
                return
        else:
            self.respond(400, {"success": False, "message": f"Model '{model}' is not supported.", "status_code": 400})
            return
        if state.fail_tts_if(text):
            self.respond(400, {"success": False, "message": "This voice can't speak that language.", "status_code": 400})
            return
        if state.rate_limit_once:
            state.rate_limit_once = False
            self.respond(429, {"detail": {"error_code": "RATE_LIMITED", "message": "Only 1 active request is allowed at a time.", "status_code": 429}})
            return
        if state.credits_after is not None and len(state.tts_texts) >= state.credits_after:
            self.respond(402, {"detail": {"message": "Insufficient credit", "status_code": 402}})
            return
        state.tts_texts.append(text)
        state.tts_payloads.append({k: v for k, v in request.items() if k != "speaker_embedding"})
        chars = len(re.sub(r"\s", "", text))
        audio = tone_wav(max(0.3, chars / state.chars_per_second), 180 + len(state.tts_texts) * 7, silent=state.silent_tts_if(text))
        self.respond(200, audio, "audio/wav")

    def stt(self, fields):
        if "audio_file" not in fields or "language_code" not in fields:
            self.respond(422, {"detail": [{"msg": "Field required"}]})
            return
        form = {name: fields[name][2].decode() for name in ("model", "language_code", "format") if name in fields}
        if form.get("model") != "prisma-v3.0":
            self.respond(400, {"success": False, "error": {"type": "INVALID_MODEL", "message": "Unexpected form."}})
            return
        code = form["language_code"]
        transcript = self.state.stt_override if self.state.stt_override is not None else "budget " + SAMPLE_CHAR[code] * 20
        self.respond(200, {"success": True, "request_id": "r1", "transcript": transcript, "model": "prisma-v3.0",
                           "detected_language_code": code, "processing_time": 0.5, "end_to_end_latency": 0.6})


class FakeServices:
    def __enter__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.state = State()
        self.state = self.server.state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
