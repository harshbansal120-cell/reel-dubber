"""Audio and video work, using the ffmpeg bundled with imageio-ffmpeg."""

import re
import shutil
import subprocess
import wave
from array import array
from pathlib import Path

import imageio_ffmpeg

from .metrics import wav_parts

RATE = 24000
WIDTH = 2
FADE_S = 0.015
TRIM_WINDOW = RATE // 50  # 20 ms
TRIM_PEAK = 184  # About -45 dB.
FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


class MediaError(Exception):
    pass


def _ffmpeg(*args, timeout=900):
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise MediaError(result.stderr.strip()[-600:] or "ffmpeg failed")


def probe(path):
    """Read duration and streams from ffmpeg's own report on the file."""
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    report = result.stderr
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", report)
    if not match:
        raise MediaError("This file doesn't look like audio or video that can be read.")
    hours, minutes, seconds = match.groups()
    return {
        "duration_s": round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3),
        "has_video": bool(re.search(r"Stream #.*Video:", report)),
        "has_audio": bool(re.search(r"Stream #.*Audio:", report)),
    }


def extract_speech(source, target, cleanup="auto"):
    """Prepare a compact ASR copy; the original track is kept for mixing/playback.

    Auto applies gentle level control and removes inaudible rumble/hiss. The
    stronger presets are opt-in because aggressive denoising can erase speech.
    All filters preserve timestamps.
    """
    filters = {
        # dynaudnorm's Gaussian filter size (g) must be odd.
        "auto": "highpass=f=70,lowpass=f=7800,dynaudnorm=f=150:g=13:p=0.9",
        "noisy": "highpass=f=90,lowpass=f=7600,afftdn=nr=10:nf=-40:tn=1,dynaudnorm=f=150:g=13:p=0.9",
        "phone": "highpass=f=140,lowpass=f=3900,afftdn=nr=8:nf=-42:tn=1,dynaudnorm=f=150:g=13:p=0.9",
    }
    if cleanup not in filters:
        cleanup = "auto"
    _ffmpeg("-i", source, "-vn", "-af", filters[cleanup], "-ac", 1, "-ar", 16000,
            "-c:a", "libmp3lame", "-b:a", "64k", target)


def extract_voice_track(source, target):
    _ffmpeg("-i", source, "-vn", "-ac", 1, "-ar", RATE, "-c:a", "pcm_s16le", target)


def convert_to_wav(source, target):
    """For voice samples from phone recorders, which the cloning API doesn't list."""
    _ffmpeg("-i", source, "-vn", "-ac", 1, "-ar", 44100, "-c:a", "pcm_s16le", target)


def _is_track_format(path):
    rate, channels, width, _ = wav_parts(Path(path).read_bytes())
    return rate == RATE and channels == 1 and width == WIDTH


def trim_silence(pcm):
    """Remove quiet audio from both ends of 16-bit PCM."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    loud = [i for i in range(0, len(samples), TRIM_WINDOW) if max(map(abs, samples[i : i + TRIM_WINDOW]), default=0) >= TRIM_PEAK]
    if not loud:
        return b""
    begin = max(0, loud[0] - TRIM_WINDOW)
    end = min(len(samples), loud[-1] + 2 * TRIM_WINDOW)
    return samples[begin:end].tobytes()


def clean_clip(source, target):
    """Trim silence from both ends, so a clip's length is its real speaking time.

    Starting ffmpeg can take over a second on Windows, and this runs for every
    line, so it's only used for audio that isn't in the expected format already.
    """
    if not _is_track_format(source):
        temp = Path(target).with_name(Path(target).stem + "_converted.wav")
        _ffmpeg("-i", source, "-ac", 1, "-ar", RATE, "-c:a", "pcm_s16le", temp)
        write_pcm(target, trim_silence(read_pcm(temp)))
        temp.unlink()
        return
    write_pcm(target, trim_silence(read_pcm(source)))


def retime_clip(source, target, tempo):
    if abs(tempo - 1.0) < 0.005:
        shutil.copyfile(source, target)
        return
    _ffmpeg("-i", source, "-af", f"atempo={tempo:.3f}", "-ac", 1, "-ar", RATE, "-c:a", "pcm_s16le", target)


def read_pcm(path):
    rate, channels, width, pcm = wav_parts(Path(path).read_bytes())
    if rate != RATE or channels != 1 or width != WIDTH:
        raise MediaError(f"{Path(path).name} isn't in the expected audio format.")
    return pcm


def write_pcm(path, pcm):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(WIDTH)
        handle.setframerate(RATE)
        handle.writeframes(pcm)


def _sample_offset(seconds):
    return int(round(seconds * RATE)) * WIDTH


def compose_track(placements, total_s, target):
    """Lay (path, start_s) clips onto silence. Returns seconds cut off at the end."""
    track = bytearray(_sample_offset(total_s))
    overflow = 0.0
    for path, start_s in placements:
        pcm = read_pcm(path)
        begin = _sample_offset(start_s)
        room = max(0, len(track) - begin)
        if len(pcm) > room:
            overflow += (len(pcm) - room) / (RATE * WIDTH)
            pcm = pcm[: room - room % WIDTH]
        track[begin : begin + len(pcm)] = pcm
    write_pcm(target, bytes(track))
    return round(overflow, 3)


def splice_tracks(windows, target):
    """Join (path, start_s, end_s) windows, with short fades so the joins don't click."""
    output = bytearray()
    fade = int(FADE_S * RATE)
    for path, start_s, end_s in windows:
        pcm = read_pcm(path)[_sample_offset(start_s) : _sample_offset(end_s)]
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        count = min(fade, len(samples) // 2)
        for i in range(count):
            gain = i / max(count, 1)
            samples[i] = int(samples[i] * gain)
            samples[-1 - i] = int(samples[-1 - i] * gain)
        output += samples.tobytes()
    write_pcm(target, bytes(output))


def _font():
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate.replace(":", "\\:")
    return None


def _labels_filter(labels):
    """A chain of drawtext filters for (text, start_s, end_s) labels."""
    font = _font()
    if font is None:
        return None
    parts = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    for text, start_s, end_s in labels:
        safe = re.sub(r"[^A-Za-z0-9 ()-]", "", text)
        parts.append(
            f"drawtext=fontfile='{font}':text='{safe}':fontcolor=white:fontsize=h/30:"
            f"x=(w-text_w)/2:y=h*0.05:box=1:boxcolor=black@0.55:boxborderw=14:"
            f"enable='between(t,{start_s:.3f},{end_s:.3f})'"
        )
    return ",".join(parts)


def mux(source, voice, target, has_video, background, labels, duration_s):
    """Combine the source picture with the new voice over a quiet copy of the original sound.

    Returns True if the on-screen labels were drawn. Without a usable font or
    filter, the export still completes, unlabelled.
    """
    if background > 0:
        audio = (
            f"[0:a]volume={background:.3f}[bed];[1:a]aresample=48000[voice];"
            # The bundled FFmpeg build predates amix's normalize option.
            "[bed][voice]amix=inputs=2:duration=longest[a]"
        )
    else:
        audio = "[1:a]aresample=48000[a]"
    base = ["-i", source, "-i", voice]
    finish = ["-map", "[a]", "-c:a", "aac", "-b:a", "160k", "-t", f"{duration_s:.3f}", "-movflags", "+faststart", target]
    if not has_video:
        _ffmpeg(*base, "-filter_complex", audio, *finish)
        return False
    picture = _labels_filter(labels) if labels else None
    labels_drawn = picture is not None
    source_duration = probe(source)["duration_s"]
    tail_s = max(0.0, duration_s - source_duration)
    if tail_s > 0:
        if picture is None:
            picture = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        picture += f",tpad=stop_mode=clone:stop_duration={tail_s:.3f}"
    if picture:
        try:
            _ffmpeg(
                *base,
                "-filter_complex", f"{audio};[0:v]{picture}[v]",
                "-map", "[v]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", 20, "-pix_fmt", "yuv420p",
                *finish,
            )
            return labels_drawn
        except MediaError:
            pass
    if tail_s > 0:
        try:
            _ffmpeg(
                *base,
                "-filter_complex", f"{audio};[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop_duration={tail_s:.3f}[v]",
                "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", 20,
                "-pix_fmt", "yuv420p", *finish,
            )
            return False
        except MediaError:
            pass
    _ffmpeg(*base, "-filter_complex", audio, "-map", "0:v", "-c:v", "copy", *finish)
    return False
