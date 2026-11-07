import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from dubber import gnani, llm, media, metrics, pipeline, rewrite
from dubber.config import LANGUAGES, Settings, read_env_file
from fake_services import EVON_KEY, GEMINI_KEY, KEY, FakeServices, tone_wav


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(gnani, "BACKOFF_BASE", 0)
    monkeypatch.setattr(llm, "BACKOFF_BASE", 0)
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0)


@pytest.fixture
def services():
    with FakeServices() as fake:
        yield fake


@pytest.fixture(scope="session")
def reel(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "reel.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", "-f", "lavfi", "-i", "testsrc=size=240x426:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=44100", "-t", "14", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "ultrafast", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def sample_audio(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "talk.wav"
    path.write_bytes(tone_wav(14.0))
    return path


def make_settings(services, tmp_path, evon=False, gemini=True, key=KEY):
    return Settings(
        gnani_key=key,
        gnani_base_url=services.url,
        llm_provider="",
        gemini_key=GEMINI_KEY if gemini else "",
        gemini_model="gemini-test",
        gemini_base_url=services.url,
        evon_base_url=services.url if evon else "",
        evon_key=EVON_KEY if evon else "",
        evon_model="gnani/gnani-evon-v3.3-30B-A3B",
        runs_dir=tmp_path / "runs",
    )


def options(**changes):
    base = {
        "languages": ["ta", "bn"],
        "voice_mode": "timbre",
        "voice": "Shivani",
        "style": "mixed",
        "background": 0.08,
        "check": True,
        "label": True,
        "owner": "",
        "keyterms": [],
    }
    base.update(changes)
    return base


def start_job(settings, source, reference=None, **changes):
    job_id = "job-test-1"
    job = pipeline.create_job(settings.runs_dir, job_id, source.name, options(**changes))
    directory = settings.runs_dir / job_id
    (directory / f"source{source.suffix}").write_bytes(source.read_bytes())
    if reference:
        (directory / "reference.wav").write_bytes(reference)
    return job_id, directory


def run(settings, source, **changes):
    job_id, directory = start_job(settings, source, **changes)
    return pipeline.Pipeline(settings, job_id).run(), directory


# ------------------------------------------------------------------ text


def test_normalize_ignores_spacing_punctuation_and_joiners():
    assert metrics.normalize("Budget-ஐயும், week।") == metrics.normalize("budget ஐயும் WEEK")
    assert metrics.normalize("क‍ख") == "कख"


def test_character_error_rate_is_capped():
    assert metrics.character_error_rate("abcd", "abxd") == 0.25
    assert metrics.character_error_rate("ab", "abcdefgh") == 1


def test_script_share_spots_the_wrong_language():
    tamil = LANGUAGES["ta"]["range"]
    assert metrics.script_share("வணக்கம் நண்பர்களே", tamil) == 1
    assert metrics.script_share("Hello friends", tamil) == 0
    assert metrics.script_share("नमस्ते दोस्तों", tamil) == 0
    assert 0 < metrics.script_share("இன்னைக்கு video கொஞ்சம் different", tamil) < 1


def test_wav_info_trusts_the_audio_over_a_wrong_header():
    audio = bytearray(tone_wav(2.0))
    start = audio.index(b"data")
    audio[start + 4 : start + 8] = (0xFFFFFFFF).to_bytes(4, "little")
    duration, rms = metrics.wav_info(bytes(audio))
    assert duration == pytest.approx(2.0)
    assert rms > 1000


@pytest.mark.parametrize("wrong_length", [0, 0xFFFFFFFF])
def test_clips_with_wrong_length_headers_keep_their_audio(tmp_path, wrong_length):
    audio = bytearray(tone_wav(2.0))
    start = audio.index(b"data")
    audio[start + 4 : start + 8] = wrong_length.to_bytes(4, "little")
    source = tmp_path / "streamed.wav"
    source.write_bytes(bytes(audio))
    target = tmp_path / "clean.wav"
    media.clean_clip(source, target)
    assert metrics.wav_info(target.read_bytes())[0] == pytest.approx(2.0, abs=0.05)


def test_env_file_parsing(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\nA=1\nB='two'\n\nC = \"three\"\n", "utf-8")
    assert read_env_file(path) == {"A": "1", "B": "two", "C": "three"}


# -------------------------------------------------------------- line timing


def test_fragments_merge_until_a_sentence_ends():
    segments = [
        {"start_time": 0.0, "end_time": 1.0, "text": "one", "speaker_id": 0},
        {"start_time": 1.1, "end_time": 2.0, "text": "two.", "speaker_id": 0},
        {"start_time": 2.1, "end_time": 3.0, "text": "three", "speaker_id": 0},
        {"start_time": 3.1, "end_time": 4.0, "text": "four", "speaker_id": 1},
        {"start_time": 4.1, "end_time": 4.2, "text": "", "speaker_id": -1},
    ]
    lines = pipeline.merge_segments(segments, 5.0)
    assert [line["text"] for line in lines] == ["one two.", "three", "four"]
    assert [line["id"] for line in lines] == [0, 1, 2]


def test_a_line_may_use_the_pause_after_it_but_not_all_of_it():
    segments = [
        {"start_time": 0.0, "end_time": 2.0, "text": "a.", "speaker_id": 0},
        {"start_time": 20.0, "end_time": 22.0, "text": "b.", "speaker_id": 0},
        {"start_time": 22.5, "end_time": 24.0, "text": "c.", "speaker_id": 0},
    ]
    lines = pipeline.merge_segments(segments, 25.0)
    assert lines[0]["slot_s"] == 3.0
    assert lines[1]["slot_s"] == 2.5


def test_long_merges_are_capped():
    segments = [{"start_time": i * 3.0, "end_time": i * 3.0 + 2.9, "text": f"w{i}", "speaker_id": 0} for i in range(6)]
    lines = pipeline.merge_segments(segments, 20.0)
    assert all(line["end"] - line["start"] <= pipeline.MERGE_MAX_S for line in lines)


def test_a_long_clip_pushes_the_next_one_later():
    items = [
        {"start": 0.0, "final_s": 3.0},
        {"start": 2.0, "final_s": 1.0},
        {"start": 10.0, "final_s": 1.0},
        {"start": 11.0},  # A line with no clip takes no time.
    ]
    pipeline.place_lines(items)
    assert items[1]["placed_s"] == pytest.approx(3.05)
    assert items[1]["late_s"] == pytest.approx(1.05)
    assert items[2]["late_s"] == 0
    assert "placed_s" not in items[3]


# ------------------------------------------------------------------ rewrite


def test_parse_lines_handles_fences_and_surrounding_text():
    assert rewrite.parse_lines('Here you go:\n```json\n{"lines": [{"id": 0, "text": " hi "}]}\n```') == {0: "hi"}
    with pytest.raises(ValueError):
        rewrite.parse_lines("no json here")


def test_problems_catch_missing_lines_and_wrong_language():
    tamil = LANGUAGES["ta"]
    assert "missing" in rewrite.problems_with({0: "வணக்கம்"}, [0, 1], tamil, "mixed")[0]
    assert "Tamil" in rewrite.problems_with({0: "Hello everyone"}, [0], tamil, "mixed")[0]
    assert rewrite.problems_with({0: "வணக்கம் friends"}, [0], tamil, "mixed") == []
    assert rewrite.problems_with({0: "Hello everyone"}, [0], LANGUAGES["en"], "mixed") == []


def test_style_changes_the_instruction():
    mixed = rewrite.system_prompt(LANGUAGES["ta"], "mixed", "Hindi")
    pure = rewrite.system_prompt(LANGUAGES["ta"], "pure", "Hindi")
    assert "Tanglish" in mixed and "Latin script" in mixed
    assert "Tanglish" not in pure


# ------------------------------------------------------------ full pipeline


def test_full_dub_makes_a_video_for_each_language(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job, directory = run(settings, reel)
    assert job["status"] == "done", job["error"]
    assert job["source_language"] == "hi-IN"
    assert len(job["lines"]) == 3  # The first two fragments join into one sentence.
    for lang in ("ta", "bn"):
        state = job["languages"][lang]
        assert state["status"] == "done"
        assert state["rewrite_by"] == "Gemini (gemini-test)"
        info = media.probe(directory / state["output"])
        assert info["has_video"] and info["has_audio"]
        assert info["duration_s"] == pytest.approx(14.0, abs=0.3)
        assert state["score"]["voiced"] == 3
        assert state["score"]["mean_cer"] is not None
        assert state["labelled"] is True
    assert services.state.batch_config["model"] == "prisma-v3.0"
    assert services.state.batch_config["format"] == "codemix"
    assert services.state.batch_config_type == "application/json"


def test_transcript_link_never_gets_the_api_key(services, reel, tmp_path):
    run(make_settings(services, tmp_path), reel)
    assert "X-API-Key-ID" not in services.state.transcript_headers


def test_voice_settings_reach_timbre(services, reel, tmp_path):
    run(make_settings(services, tmp_path), reel, languages=["ta", "mr"], voice="Riyaa")
    sent = {(p["model"], p["voice"], p["language"]) for p in services.state.tts_payloads}
    assert sent == {("timbre-v3.0", "Riyaa", "auto"), ("timbre-v3.0", "Riyaa", "mr-IN")}


def test_native_style_sends_the_language_code(services, reel, tmp_path):
    run(make_settings(services, tmp_path), reel, languages=["ta"], style="pure")
    assert {p["language"] for p in services.state.tts_payloads} == {"ta-IN"}


def test_cloned_voice_uses_the_embedding_and_keeps_it_private(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job_id, directory = start_job(settings, reel, reference=tone_wav(6.0), voice_mode="clone", owner="Test Student", languages=["ta"])
    job = pipeline.Pipeline(settings, job_id).run()
    assert job["status"] == "done", job["error"]
    assert {p["model"] for p in services.state.tts_payloads} == {"vachana-vc-v1"}
    saved = (directory / "job.json").read_text("utf-8")
    assert "ZmFrZS1lbWJlZGRpbmc=" not in saved
    assert KEY not in saved and GEMINI_KEY not in saved


def test_skips_a_language_the_reel_is_already_in(services, reel, tmp_path):
    job, _ = run(make_settings(services, tmp_path), reel, languages=["hi", "ta"])
    assert "hi" not in job["languages"]
    assert any("already in Hindi" in warning for warning in job["warnings"])
    assert job["languages"]["ta"]["status"] == "done"


def test_audio_only_input_makes_audio_output(services, sample_audio, tmp_path):
    job, directory = run(make_settings(services, tmp_path), sample_audio, languages=["ta"])
    assert job["status"] == "done", job["error"]
    assert job["languages"]["ta"]["output"].endswith(".m4a")
    assert media.probe(directory / job["languages"]["ta"]["output"])["has_video"] is False


def test_too_long_a_reel_spends_nothing(services, sample_audio, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_SOURCE_SECONDS", 5)
    job, _ = run(make_settings(services, tmp_path), sample_audio)
    assert job["status"] == "failed"
    assert "Trim it" in job["error"]
    assert services.state.calls == []


def test_no_speech_is_a_clear_failure(services, reel, tmp_path):
    services.state.segments = [{"start_time": 0, "end_time": 1, "text": "", "speaker_id": -1}]
    job, _ = run(make_settings(services, tmp_path), reel)
    assert job["status"] == "failed"
    assert "no speech" in job["error"]
    assert services.state.tts_texts == []


def test_failed_transcription_explains_itself_and_can_retry(services, reel, tmp_path):
    services.state.batch_fail = "Empty transcript after 3 retries"
    settings = make_settings(services, tmp_path)
    job, _ = run(settings, reel)
    assert job["status"] == "failed"
    assert "Empty transcript" in job["error"]
    services.state.batch_fail = None
    job = pipeline.Pipeline(settings, job["id"]).run()
    assert job["status"] == "done", job["error"]


# ------------------------------------------------------------- saving credits


def test_lines_predicted_too_long_are_shortened_before_being_spoken(services, reel, tmp_path):
    services.state.llm_stretch = 1.8
    services.state.chars_per_second = 13.0
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    state = job["languages"]["ta"]
    assert services.state.llm_shorten_requests >= 1
    assert state["score"]["shortened"] >= 1
    # Only two lines are sampled at full length, so at most one extra speech request is the third.
    assert len(services.state.tts_texts) <= len(job["lines"]) + 2
    assert all(item["tempo"] <= pipeline.HARD_MAX_TEMPO for item in state["lines"])


def test_speech_slower_than_expected_is_fixed_after_measuring(services, reel, tmp_path):
    services.state.chars_per_second = 7.0
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    state = job["languages"]["ta"]
    assert state["measured_chars_per_s"] == pytest.approx(7.0, abs=0.7)
    assert state["score"]["shortened"] >= 1
    assert state["status"] == "done"


def test_running_again_spends_nothing(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job, _ = run(settings, reel)
    before = list(services.state.calls)
    again = pipeline.Pipeline(settings, job["id"]).run()
    assert again["status"] == "done"
    assert services.state.calls == before


def test_out_of_credits_stops_and_resume_only_pays_for_the_rest(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    services.state.credits_after = 2
    job, directory = run(settings, reel)
    assert job["status"] == "failed"
    assert "out of Gnani credits" in job["error"]
    assert job["languages"]["ta"]["status"] == "stopped"
    assert len(services.state.tts_texts) == 2
    services.state.credits_after = None
    job = pipeline.Pipeline(settings, job["id"]).run()
    assert job["status"] == "done", job["error"]
    assert services.state.tts_texts.count(services.state.tts_texts[0]) == 1
    assert len(services.state.tts_texts) == 6  # 3 lines in each language, no repeats.
    assert services.state.calls.count(("POST", "/stt/v3/batch/jobs")) == 1


def test_stop_request_stops_between_steps_and_can_resume(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job_id, directory = start_job(settings, reel, languages=["ta"])
    work = pipeline.Pipeline(settings, job_id)
    original = work._speak_safely
    calls = []

    def speak_then_stop(lang, item):
        original(lang, item)
        calls.append(item["id"])
        (directory / pipeline.CANCEL_FLAG).touch()

    work._speak_safely = speak_then_stop
    job = work.run()
    assert job["status"] == "stopped"
    assert len(calls) == 1
    assert not (directory / pipeline.CANCEL_FLAG).exists()
    job = pipeline.Pipeline(settings, job_id).run()
    assert job["status"] == "done", job["error"]
    assert len(services.state.tts_texts) == 3


def test_a_rate_limit_is_retried(services, reel, tmp_path):
    services.state.rate_limit_once = True
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    assert job["status"] == "done", job["error"]
    assert job["usage"]["retries"] == 1


def test_bad_key_stops_the_job_and_stays_secret(services, reel, tmp_path):
    settings = make_settings(services, tmp_path, key="wrong-key-do-not-leak")
    job, directory = run(settings, reel)
    assert job["status"] == "failed"
    assert "rejected the API key" in job["error"]
    assert "wrong-key-do-not-leak" not in (directory / "job.json").read_text("utf-8")


def test_missing_key_says_what_to_do(services, reel, tmp_path):
    job, _ = run(make_settings(services, tmp_path, key=""), reel)
    assert job["status"] == "failed"
    assert "GNANI_API_KEY" in job["error"]


# -------------------------------------------------------------- language failures


def test_a_voice_that_cant_speak_a_language_only_fails_that_language(services, reel, tmp_path):
    services.state.fail_tts_if = lambda text: "த" in text
    job, _ = run(make_settings(services, tmp_path), reel)
    assert job["status"] == "done"
    assert job["languages"]["ta"]["status"] == "failed"
    assert "couldn't speak Tamil" in job["languages"]["ta"]["error"]
    assert job["languages"]["bn"]["status"] == "done"


def test_all_languages_failing_fails_the_job(services, reel, tmp_path):
    services.state.fail_tts_if = lambda text: True
    job, _ = run(make_settings(services, tmp_path), reel)
    assert job["status"] == "failed"
    assert "couldn't speak" in job["error"]
    # It gives up after 3 lines in each language, rather than paying for every line.
    assert services.state.calls.count(("POST", "/api/v1/tts/inference")) == 6


def test_one_failed_line_doesnt_lose_the_rest(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    failed_once = []

    def fail_shortest(text):
        if not failed_once and len(services.state.tts_texts) == 2:
            failed_once.append(text)
            return True
        return False

    services.state.fail_tts_if = fail_shortest
    job, _ = run(settings, reel, languages=["ta"])
    state = job["languages"]["ta"]
    assert state["status"] == "done"
    assert state["score"]["failed"] == 1
    assert state["score"]["voiced"] == 2
    assert any("failed" in item["flags"] for item in state["lines"])


def test_silent_audio_is_flagged_and_left_out(services, reel, tmp_path):
    spoken = []

    def silent_first(text):
        spoken.append(text)
        return len(spoken) == 1

    services.state.silent_tts_if = silent_first
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    state = job["languages"]["ta"]
    assert sum("silent" in item["flags"] for item in state["lines"]) == 1
    assert state["score"]["voiced"] == 2


def test_prisma_disagreement_is_flagged(services, reel, tmp_path):
    services.state.stt_override = "completely different words"
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    assert any("check_speech" in item["flags"] for item in job["languages"]["ta"]["lines"])
    assert job["languages"]["ta"]["score"]["mean_cer"] > pipeline.CER_CHECK
    assert job["languages"]["ta"]["score"]["to_review"] == 3


def test_turning_checks_off_saves_those_calls(services, reel, tmp_path):
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"], check=False)
    assert job["status"] == "done", job["error"]
    assert ("POST", "/stt/v3") not in services.state.calls
    assert job["languages"]["ta"]["score"]["mean_cer"] is None


# ------------------------------------------------------------ rewrite models


def test_no_rewrite_model_is_a_clear_failure(services, reel, tmp_path):
    job, _ = run(make_settings(services, tmp_path, gemini=False), reel, languages=["ta"])
    assert job["status"] == "failed"
    assert "GEMINI_API_KEY" in job["error"]
    assert services.state.tts_texts == []


def test_evon_is_preferred_when_configured(services, reel, tmp_path):
    job, _ = run(make_settings(services, tmp_path, evon=True), reel, languages=["ta"])
    assert job["languages"]["ta"]["rewrite_by"] == "Evon (gnani-evon-v3.3-30B-A3B)"
    assert services.state.llm_calls["gemini"] == 0


def test_wrong_language_gets_one_correction(services, reel, tmp_path):
    services.state.gemini_wrong_language_times = 1
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    state = job["languages"]["ta"]
    assert state["status"] == "done"
    assert services.state.llm_calls["gemini"] == 2
    assert any("Tamil" in note for note in state["notes"])


def test_evon_drifting_to_english_falls_back_to_gemini_and_says_so(services, reel, tmp_path):
    services.state.evon_wrong_language_times = 2
    job, _ = run(make_settings(services, tmp_path, evon=True), reel, languages=["ta"])
    state = job["languages"]["ta"]
    assert state["rewrite_by"] == "Gemini (gemini-test)"
    assert any("Evon" in note for note in state["notes"])


def test_every_model_failing_only_fails_that_language(services, reel, tmp_path):
    services.state.gemini_wrong_language_times = 2
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta", "bn"])
    assert job["languages"]["ta"]["status"] == "failed"
    assert "Rewrite failed" in job["languages"]["ta"]["error"]
    assert job["languages"]["bn"]["status"] == "done"
    assert job["status"] == "done"


def test_model_outage_is_a_clear_failure(services, reel, tmp_path):
    services.state.gemini_error = 503
    job, _ = run(make_settings(services, tmp_path), reel, languages=["ta"])
    assert job["status"] == "failed"
    assert "HTTP 503" in job["error"]


# ------------------------------------------------------------ after the run


def test_editing_a_line_only_pays_for_that_line(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job, directory = run(settings, reel, languages=["ta"])
    state = job["languages"]["ta"]
    before = {"tts": len(services.state.tts_texts), "stt": services.state.calls.count(("POST", "/stt/v3")), "rev": state["output_rev"]}
    job = pipeline.Pipeline(settings, job["id"]).edit_line("ta", 1, "budget தமிழ் வரி")
    state = job["languages"]["ta"]
    assert job["status"] == "done", job["error"]
    assert len(services.state.tts_texts) == before["tts"] + 1
    assert services.state.tts_texts[-1] == "budget தமிழ் வரி"
    assert services.state.calls.count(("POST", "/stt/v3")) == before["stt"] + 1
    assert state["lines"][1]["edited"] is True
    assert state["output_rev"] != before["rev"]


def test_montage_switches_language_at_line_starts(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job, directory = run(settings, reel)
    job = pipeline.Pipeline(settings, job["id"]).set_montage(["ta", "bn"], 4.0)
    assert job["status"] == "done", job["error"]
    montage = job["montage"]
    assert montage["windows"][0]["label"] == "Original"
    assert montage["windows"][1]["label"] == "Tamil (AI-dubbed)"
    starts = {line["start"] for line in job["lines"]} | {0.0}
    assert all(window["start"] in starts for window in montage["windows"])
    assert media.probe(directory / montage["output"])["duration_s"] == pytest.approx(14.0, abs=0.3)


def test_montage_needs_finished_languages(services, reel, tmp_path):
    settings = make_settings(services, tmp_path)
    job, _ = run(settings, reel, languages=["ta"])
    job = pipeline.Pipeline(settings, job["id"]).set_montage(["bn"], 4.0)
    assert job["status"] == "failed"
    assert "Bengali" in job["error"]


def test_scorecard_names_every_model_and_shows_failures(services, reel, tmp_path):
    services.state.fail_tts_if = lambda text: "ব" in text
    job, _ = run(make_settings(services, tmp_path), reel)
    card = pipeline.scorecard_markdown(job)
    assert "Prisma v3.0" in card
    assert "Timbre v3.0, Shivani" in card
    assert "Gemini (gemini-test)" in card
    assert "| Bengali |" in card and "failed" in card
    assert "not whether it sounds good" in card


def test_splicing_keeps_the_total_length(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    media.write_pcm(a, tone_wav(4.0)[44:])
    media.write_pcm(b, tone_wav(4.0, 440)[44:])
    out = tmp_path / "out.wav"
    media.splice_tracks([(a, 0.0, 1.5), (b, 1.5, 4.0)], out)
    assert metrics.wav_info(out.read_bytes())[0] == pytest.approx(4.0, abs=0.01)


def test_compose_reports_what_gets_cut_off(tmp_path):
    clip = tmp_path / "clip.wav"
    media.write_pcm(clip, tone_wav(2.0)[44:])
    out = tmp_path / "out.wav"
    assert media.compose_track([(clip, 0.0), (clip, 2.5)], 3.0, out) == pytest.approx(1.5, abs=0.01)
    assert metrics.wav_info(out.read_bytes())[0] == pytest.approx(3.0, abs=0.01)


def test_trim_silence_keeps_the_speech_and_a_little_margin():
    pcm = tone_wav(1.0)[44:]
    padded = bytes(24000 * 2) + pcm + bytes(24000 * 2)  # 1 s of silence on each side.
    trimmed = media.trim_silence(padded)
    assert 1.0 <= len(trimmed) / (24000 * 2) <= 1.1
    assert media.trim_silence(bytes(24000 * 2)) == b""


def test_clips_in_other_formats_are_converted(tmp_path):
    source = tmp_path / "odd.wav"
    import wave

    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(tone_wav(1.0)[44:])
    target = tmp_path / "clean.wav"
    media.clean_clip(source, target)
    assert media.read_pcm(target)
