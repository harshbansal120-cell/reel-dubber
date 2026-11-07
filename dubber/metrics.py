"""Text and audio measurements used for checks and the scorecard."""

import io
import math
import re
import sys
import unicodedata
import wave
from array import array
from collections import Counter


def normalize(text):
    """Lowercase NFC text with spaces, punctuation and joiners removed.

    Spacing around English words varies in code-mixed text ("budget-ஐ" vs
    "budget ஐ"), so it shouldn't count as an error.
    """
    text = unicodedata.normalize("NFC", text).lower()
    return "".join(c for c in text if not c.isspace() and unicodedata.category(c)[0] not in "PSC")


def spoken_length(text):
    return len(normalize(text))


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def character_error_rate(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return min(1.0, edit_distance(reference, hypothesis) / len(reference))


def english_words(text):
    return re.findall(r"[a-z]+", unicodedata.normalize("NFC", text).lower())


def english_kept(reference, hypothesis):
    """How many of the reference's English words appear in the hypothesis in Latin script."""
    wanted = Counter(english_words(reference))
    found = Counter(english_words(hypothesis))
    return sum((wanted & found).values()), sum(wanted.values())


def has_digits(text):
    return any(c.isdigit() for c in text)


def script_share(text, code_range):
    """Share of a text's letters and vowel signs that fall in the given script."""
    low, high = code_range
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    return sum(low <= ord(c) <= high for c in letters) / len(letters)


def wav_parts(data):
    """A WAV's format and audio. Streamed WAV headers can carry wrong lengths, such as
    0 or 0xFFFFFFFF, so a length that doesn't fit the file is ignored."""
    with wave.open(io.BytesIO(data)) as handle:
        rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
    position = 12
    while position + 8 <= len(data):
        chunk, size = data[position : position + 4], int.from_bytes(data[position + 4 : position + 8], "little")
        if chunk == b"data":
            start = position + 8
            fits = 0 < size <= len(data) - start
            return rate, channels, width, data[start : start + size] if fits else data[start:]
        position += 8 + size + (size & 1)
    return rate, channels, width, b""


def wav_info(data):
    """Duration and loudness of a WAV, measured from the audio itself."""
    rate, channels, width, pcm = wav_parts(data)
    duration = len(pcm) / (rate * channels * width)
    rms = None
    if width == 2:
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
    return duration, rms
