# VYOMA

## व्योम

**AI-POWERED CONTENT LOCALIZATION**

### One voice. Limitless reach.

Powered by sovereign Indian AI capabilities. VYOMA localizes short videos for Indian language audiences while keeping everyday code-mixing natural. A Hinglish clip can become Tanglish or Benglish, not textbook Tamil or Bengali.

Built for Gnani AI's Great Indian AI Internship Challenge. The current configured stack uses Gnani for transcription and speech, and Google Gemini for rewriting and semantic review.

## What each part does

| Step | Technology |
|---|---|
| Transcribe, with timings and speaker labels | **Gnani Prisma v3.0**, batch (`indic-auto`, `codemix`) |
| Rewrite the lines to fit the time and sound natural | **Google Gemini** in the current setup; an Evon v3.3 provider is implemented but not currently configured |
| Speak in the creator's voice | **Gnani voice cloning** (`vachana-vc-v1`) |
| Speak expressively in a stock voice, as the alternative | **Gnani Timbre v3.0**, with supported inline audio tags |
| Check that each dubbed line can be understood | **Gnani Prisma v3.0**, REST |
| Fit timing, mix audio, export video | ffmpeg |

The app records which model rewrote each language and shows the provider in the result. Current runs use Gemini for rewriting and semantic review. Prisma and Timbre are Gnani services; Gemini also handles the optional source-emotion analysis. Do not describe a run as using Evon unless its configured Evon endpoint actually handled that run.

## Run it

You need Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your `GNANI_API_KEY` and `GEMINI_API_KEY` for the current setup. Then:

```powershell
python run_app.py
```

Open http://127.0.0.1:8000. The app only listens on your own computer.

### Rewrite provider status

Gemini is the active rewrite provider in the current setup. The code also includes an Evon adapter for an OpenAI-compatible Evon endpoint, but that does not mean Evon is being used: it must be available and selected in `.env` first. To use Evon later, configure `EVON_BASE_URL`, optional `EVON_API_KEY`, and `EVON_MODEL`, then set `LLM_PROVIDER=evon` and restart the app. To keep using Gemini explicitly, set `LLM_PROVIDER=gemini`. Each completed dub records its actual rewrite provider in the job result.

## Using it

1. Add a reel of up to 3 minutes, choose languages and pick a voice. Try a 15-second clip in one language first, because Gnani doesn't publish credit rates. Compare your dashboard before and after.
2. For a cloned voice, add 5 to 30 seconds of clean speech. Clone only your own voice or a teammate's, with their consent.
   In fine-tuning, choose **Auto** for normal recordings, **Noisy room** for background noise, or **Phone / VoIP** for narrow-band calls. Cleanup is applied only to the transcription copy. Strong cleanup can soften quiet words, so review the transcript.
3. Watch each result. Any line can be corrected and spoken again, which only pays for that line.
   After dubbing, adjust the **Final soundtrack** control to keep the original ambience/music under the dub at an adjustable level or mute the original track, then rebuild the export without repeating transcription or speech generation.
4. Build the montage: one video that changes language as it plays.
5. Ask a native speaker to rate each language, then download the scorecard.
6. Download the synchronized `.srt` captions beside each language video. In transcript review, preview any line, adjust its delivery and speed, or enter a phonetic speech spelling while keeping the caption text unchanged.
7. Record views, likes, comments, shares and saves for each published version. The workspace compares engagement rates from the numbers you enter.
8. Set the CodeMix level from mostly native wording to a stronger English mix. It guides the rewrite model; it is not an exact word-count target.
9. In **Transcript & review**, choose a line and record a short Voice Director instruction. Prisma transcribes it, the configured language model edits only that selected line, and VYOMA regenerates that line.
10. DubBack checks each generated line with Prisma, shows the expected and recognized text, and flags lines for listening review. It cannot verify meaning or naturalness on its own.

### Reusable creator tools

- **Creator style profiles** save a creator's localization brief, dialect, code-mix preference and glossary for later dubs.
- **Audience personas** describe who a version is for. Choose one for a first dub, or create a second audience version from a completed dub. The second version reuses the transcript and source audio, but performs a new rewrite and TTS generation (additional model and voice requests).
- **Pronunciation dictionary** entries use `term -> spoken form`. They affect generated speech, while captions retain their readable translated spelling. Entries also become Prisma key-term hints on new uploads.
- **Comment-to-reel** drafts a short response using the selected language and the source transcript as context. Review or edit the text before approving speech generation. The resulting response clip uses the original video and one TTS request; it does not publish to a social platform.

These creator tools and saved preferences are stored locally in `runs/creator-kit.json`. Persona responses should still be reviewed for factual accuracy and cultural fit before publishing.

### Localize for your audience

Before starting a dub, select a regional dialect or describe the audience and tone (for example, a young Chennai tech audience and playful Tanglish). Add one glossary preference per line to keep brand names or phrases consistent. For language-specific wording, label the preference with the target language, such as `Hindi: ...; Tamil: ...`. These preferences guide the initial rewrite, semantic review, and timing-driven shortening.

The optional semantic review asks the configured language model to flag clear meaning changes, omissions, or ignored terms; it never silently rewrites the result. The separate optional emotion analysis sends the compressed speech track to Gemini and uses its per-line delivery tags with Timbre voices. It is off by default. These checks use model requests in addition to speech generation and Prisma checks.

Stopped or failed jobs can resume. Finished clips are never paid for twice.

## How the timing works

Dubbed speech rarely takes the same time as the original, and stretching audio too far sounds rushed.

1. The rewrite gets a character budget for each line, based on the time the speaker took.
2. The two longest lines are spoken first, to measure the real speech rate for that language and voice.
3. Lines likely to run long are shortened **before** credits are spent on them.
4. Lines that still run long get one more rewrite.
5. Remaining small overruns are sped up by as much as 15%, or 30% at most. These lines are flagged "Rushed".
6. A line that still won't fit pushes the next one later, and is flagged "Starts late".
7. If the final line runs past the original video, the export holds the last frame until the speech ends instead of cutting off the audio.

Rewrites also preserve the speaker's intent and adapt clear idioms, jokes, and cultural references without adding new claims. With a Timbre stock voice, the rewrite can suggest an expressive delivery tag such as excited, whispering, or laughing. These suggestions come from transcript wording and punctuation; the app does not analyze vocal emotion in the source recording. Gnani documents these tags for Timbre v3.0, not for the cloned voice endpoint.

## What the scorecard shows

- **Prisma character errors:** each final clip is transcribed and compared with the requested speech spelling. This shows whether Prisma understood it, not whether it sounds good.
- **Meaning and pronunciation flags:** model review suggestions and mismatches between speech spelling and Prisma's hearing. Review them yourself; these checks can make mistakes.
- **English words kept:** how many of the reel's English words survived the rewrite.
- **Mean speed-up and lines to review:** how hard the timing had to work.
- **Native speaker rating:** the only real measure of quality. Enter it yourself after a native speaker listens.

## Known limits

- **No lip sync.** The mouth won't match the new words.
- **Voice Director scope.** Spoken directions apply to one selected line at a time. Each direction uses a Prisma request, a rewrite-model call and a new speech generation for that line.
- **DubBack limits.** Prisma recognition is a useful pronunciation signal, not a semantic or native-speaker quality guarantee. Listen closely to names and numbers.
- **Emotion on cloned voices.** Gemini can analyze source delivery, but the current cloned-voice endpoint does not accept Timbre's delivery tags. Use a Timbre voice to apply those tags.
- **Social performance data.** Views and engagement metrics are entered manually; VYOMA does not connect to Instagram, TikTok, or YouTube accounts.
- **Original soundtrack control.** The final mix can retain the source soundtrack under the dub at 8% by default, or mute the entire source track. This can be changed after dubbing and rebuilding the export does not repeat transcription or speech generation. VYOMA does not yet separate background noise, music, and original speech into independent tracks. Noise cleanup improves the transcription copy only; it does not denoise the final source soundtrack.
- **Speaker voices.** Timbre mode can assign distinct same-gender stock voices to detected speakers, reusing voices if there are more speakers than available matching voices. Voice cloning still uses one cloned voice for all speakers because the form accepts one voice sample.
- **Cloning across languages is unproven.** Gnani's docs don't say whether a cloned voice can speak other languages. Try one language first, and use a Timbre voice if it fails.
- **Local only.** It runs on one computer, one job at a time.

## Your data

`.env` and `runs/` are ignored by git. `runs/` holds your reels, voice samples, voice embeddings and dubbed videos, so don't publish it. For the challenge, use your own script with no real phone numbers, IDs or recorded calls. On Gemini's free tier, Google's reviewers may read what you send, so don't include personal information.

## Tests

```powershell
pip install pytest httpx
python -m pytest tests
```

The tests use local fakes of the Gnani and model APIs, built from their docs, so they need no keys or credits. They check this app's logic, not how the real APIs or voices behave.
