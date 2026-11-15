"""The language model that rewrites lines: Evon if you host it, Gemini otherwise.

Neither Prisma nor Timbre translates, and Evon isn't on Gnani's API platform, so
this step goes through whichever model is configured. Each result records which
model wrote it, so the app never hides that.
"""

import base64
import re
import time

import requests

MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0
TIMEOUT = 180


class LlmError(Exception):
    pass


def _post(url, headers, payload, label):
    delay = BACKOFF_BASE
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == MAX_ATTEMPTS:
                raise LlmError(f"Could not reach {label} ({type(error).__name__}).")
        else:
            if response.ok:
                return response.json()
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                raise LlmError(f"{label} returned HTTP {response.status_code}.")
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


class Gemini:
    def __init__(self, base_url, key, model):
        self.base_url, self.key, self.model = base_url, key, model
        self.label = f"Gemini ({model})"

    def complete(self, system, user):
        body = _post(
            f"{self.base_url}/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.key},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.4,
                    "maxOutputTokens": 8192,
                },
            },
            "Gemini",
        )
        candidates = body.get("candidates") or []
        if not candidates:
            raise LlmError("Gemini returned no answer.")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text:
            raise LlmError(f"Gemini returned an empty answer ({candidates[0].get('finishReason')}).")
        return text

    def analyze_audio(self, audio, mime, prompt):
        """Ask Gemini to inspect the original delivery, not just its transcript."""
        body = _post(
            f"{self.base_url}/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.key},
            {
                "contents": [{"role": "user", "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime, "data": base64.b64encode(audio).decode("ascii")}},
                ]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.1,
                    "maxOutputTokens": 4096,
                },
            },
            "Gemini audio analysis",
        )
        candidates = body.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise LlmError("Gemini returned no audio analysis.")
        return text


class Evon:
    """An OpenAI-compatible Evon endpoint, such as vLLM on a rented GPU or Gnani's hosted beta."""

    def __init__(self, base_url, key, model):
        self.base_url, self.key, self.model = base_url, key, model
        self.label = f"Evon ({model.rsplit('/', 1)[-1]})"

    def complete(self, system, user):
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        body = _post(
            f"{self.base_url}/v1/chat/completions",
            headers,
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                # The model card's suggested settings for direct answers without reasoning.
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "Evon",
        )
        choices = body.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if not text:
            raise LlmError("Evon returned an empty answer.")
        return text


def build_providers(settings):
    """Models to try, in order. Evon comes first when it's configured."""
    available = {}
    if settings.evon_base_url:
        available["evon"] = Evon(settings.evon_base_url, settings.evon_key, settings.evon_model)
    if settings.gemini_key:
        available["gemini"] = Gemini(settings.gemini_base_url, settings.gemini_key, settings.gemini_model)
    if settings.llm_provider in available:
        return [available[settings.llm_provider]]
    return list(available.values())
