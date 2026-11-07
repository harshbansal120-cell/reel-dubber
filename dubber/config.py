"""Settings and the language table."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Codes are the ones Prisma and Timbre share. `mix_tts` is the Timbre language
# value for code-mixed text: the docs give `hi-en` for Hinglish and recommend
# `auto` otherwise, except Marathi, which `auto` would read as Hindi.
LANGUAGES = {
    "hi": {"name": "Hindi", "mix_name": "Hinglish", "code": "hi-IN", "script": "Devanagari", "range": (0x0900, 0x097F), "mix_tts": "hi-en"},
    "ta": {"name": "Tamil", "mix_name": "Tanglish", "code": "ta-IN", "script": "Tamil", "range": (0x0B80, 0x0BFF), "mix_tts": "auto"},
    "bn": {"name": "Bengali", "mix_name": "Benglish", "code": "bn-IN", "script": "Bengali", "range": (0x0980, 0x09FF), "mix_tts": "auto"},
    "te": {"name": "Telugu", "mix_name": "Tenglish", "code": "te-IN", "script": "Telugu", "range": (0x0C00, 0x0C7F), "mix_tts": "auto"},
    "kn": {"name": "Kannada", "mix_name": "Kanglish", "code": "kn-IN", "script": "Kannada", "range": (0x0C80, 0x0CFF), "mix_tts": "auto"},
    "ml": {"name": "Malayalam", "mix_name": "Manglish", "code": "ml-IN", "script": "Malayalam", "range": (0x0D00, 0x0D7F), "mix_tts": "auto"},
    "mr": {"name": "Marathi", "mix_name": "Minglish", "code": "mr-IN", "script": "Devanagari", "range": (0x0900, 0x097F), "mix_tts": "mr-IN"},
    "gu": {"name": "Gujarati", "mix_name": "Gujlish", "code": "gu-IN", "script": "Gujarati", "range": (0x0A80, 0x0AFF), "mix_tts": "auto"},
    "pa": {"name": "Punjabi", "mix_name": "Punglish", "code": "pa-IN", "script": "Gurmukhi", "range": (0x0A00, 0x0A7F), "mix_tts": "auto"},
    "en": {"name": "English", "mix_name": "Indian English", "code": "en-IN", "script": "Latin", "range": (0x0041, 0x024F), "mix_tts": "en-IN"},
}

TIMBRE_VOICES = {
    "Avinash": "Male", "Chetana": "Female", "Devansh": "Male", "Harshad": "Male", "Kalpana": "Female",
    "Lakshmi": "Female", "Prakash": "Male", "Riyaa": "Female", "Shivani": "Female", "Shubham": "Male",
    "Tarika": "Female", "Vasudha": "Female", "Yashraj": "Male",
}

MAX_SOURCE_SECONDS = 180
REFERENCE_MIN_S = 5
REFERENCE_MAX_S = 30


def read_env_file(path):
    values = {}
    if path.is_file():
        for line in path.read_text("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.partition("=")
                values[name.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class Settings:
    gnani_key: str
    gnani_base_url: str
    llm_provider: str
    gemini_key: str
    gemini_model: str
    gemini_base_url: str
    evon_base_url: str
    evon_key: str
    evon_model: str
    runs_dir: Path


def load_settings(env_file=None):
    """Real environment variables win over the .env file."""
    file_values = read_env_file(env_file or ROOT / ".env")

    def get(name, default=""):
        return os.environ.get(name) or file_values.get(name) or default

    return Settings(
        gnani_key=get("GNANI_API_KEY"),
        gnani_base_url=get("GNANI_BASE_URL", "https://api.vachana.ai").rstrip("/"),
        llm_provider=get("LLM_PROVIDER").lower(),
        gemini_key=get("GEMINI_API_KEY"),
        gemini_model=get("GEMINI_MODEL", "gemini-3.8-flash"),
        gemini_base_url=get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/"),
        evon_base_url=get("EVON_BASE_URL").rstrip("/"),
        evon_key=get("EVON_API_KEY"),
        evon_model=get("EVON_MODEL", "gnani/gnani-evon-v3.3-30B-A3B"),
        runs_dir=Path(get("DUBBER_RUNS_DIR", str(ROOT / "runs"))),
    )
