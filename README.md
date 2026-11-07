# Reel Dubber

Dubs a short video into Indian languages in the creator's own voice, keeping the language mix natural. A Hinglish reel becomes Tanglish or Benglish, not textbook Tamil or Bengali. Built for Gnani AI's Great Indian AI Internship Challenge.

## What each part does

| Step | Technology |
|---|---|
| Transcribe, with timings and speaker labels | **Gnani Prisma v3.0**, batch (`indic-auto`, `codemix`) |
| Rewrite the lines to fit the time and sound natural | **Gnani Evon v3.3** if you host it, otherwise **Google Gemini** |
| Speak in the creator's voice | **Gnani voice cloning** (`vachana-vc-v1`) |
| Speak expressively in a stock voice, as the alternative | **Gnani Timbre v3.0**, with supported inline audio tags |
| Check that each dubbed line can be understood | **Gnani Prisma v3.0**, REST |
| Fit timing, mix audio, export video | ffmpeg |

The app records which model rewrote each language and shows it, so say the same in your submission.

## Run it

You need Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your `GNANI_API_KEY`, plus `GEMINI_API_KEY` or `EVON_BASE_URL`. Then:

```powershell
python run_app.py
```

Open http://127.0.0.1:8000. The app only listens on your own computer.

## Using it

1. Add a reel of up to 3 minutes, choose languages and pick a voice. Try a 15-second clip in one language first, because Gnani doesn't publish credit rates. Compare your dashboard before and after.
2. For a cloned voice, add 5 to 30 seconds of clean speech. Clone only your own voice or a teammate's, with their consent.
3. Watch each result. Any line can be corrected and spoken again, which only pays for that line.
4. Build the montage: one video that changes language as it plays.
5. Ask a native speaker to rate each language, then download the scorecard.

Stopped or failed jobs can resume. Finished clips are never paid for twice.

## How the timing works

Dubbed speech rarely takes the same time as the original, and stretching audio too far sounds rushed.

1. The rewrite gets a character budget for each line, based on the time the speaker took.
2. The two longest lines are spoken first, to measure the real speech rate for that language and voice.
3. Lines likely to run long are shortened **before** credits are spent on them.
4. Lines that still run long get one more rewrite.
5. Remaining small overruns are sped up by as much as 15%, or 30% at most. These lines are flagged "Rushed".
6. A line that still won't fit pushes the next one later, and is flagged "Starts late".

Rewrites also preserve the speaker's intent and adapt clear idioms, jokes, and cultural references without adding new claims. With a Timbre stock voice, the rewrite can suggest an expressive delivery tag such as excited, whispering, or laughing. These suggestions come from transcript wording and punctuation; the app does not analyze vocal emotion in the source recording. Gnani documents these tags for Timbre v3.0, not for the cloned voice endpoint.

## What the scorecard shows

- **Prisma character errors:** each final clip is transcribed and compared with the intended line. This shows whether Prisma understood it, not whether it sounds good.
- **English words kept:** how many of the reel's English words survived the rewrite.
- **Mean speed-up and lines to review:** how hard the timing had to work.
- **Native speaker rating:** the only real measure of quality. Enter it yourself after a native speaker listens.

## Known limits

- **No lip sync.** The mouth won't match the new words.
- **The original voice stays faintly audible.** Nothing removes the voice from the original sound, so it plays quietly under the dub, at 8% by default. Setting it to 0 also removes any music.
- **One voice per dub.** Prisma labels speakers, but every line gets the same voice.
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
