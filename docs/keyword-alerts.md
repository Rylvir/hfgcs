# Keyword alerts (ntfy + audio)

Push a phone alert when a decode mentions watched callsigns or procedure words,
**with a short audio clip attached**. This replaces the hilltop-only
`hfgcs-skyking-ntfy` watcher, which posted Skyking text with no audio and no
confidence floor (so Whisper prompt-hallucinations on truncated noise files
paged the phone).

## What it matches

Configured in `~/.config/hfgcs/keyword-alerts.toml` (see
`config/keyword-alerts.example.toml`):

- Skyking / Skymaster / Skybird / Nightwatch (incl. NW1–NW4)
- Gordo ##, bomber callsigns DOOM / ROGUE / SKULL (context-required), exercise
  BAFFY / PANGA / PYREX, B-2 ZOLAR

`doom` / `rogue` / `skull` only match with **callsign context** (digits right
after the word, or procedure words like “this is …”, “radio check” nearby), so
ordinary English in a transcript does not alert.

## Filters (no more audio-less / phantom Skykings)

1. **Audio required.** If the capture WAV is missing, nothing is posted.
2. **Confidence / word-probability floor.** Broadcasts with confidence `0` and
   low ASR word probability are dropped.
3. **Prompt-hallucination guard.** Truncated files whose transcript is only the
   “Skyking, skyking, do not answer” preamble (often missing codeword/time/auth)
   are not alerted — the failure mode of decoding 0.3.x on long noise clips.
4. **Simulcast dedupe.** The same matched text on several channels within
   `dedupe_s` (default 120 s) becomes one alert listing the channels; the
   highest-confidence clip is attached.

## ntfy attachment

Uses ntfy’s upload API (`PUT` body + `Filename` / `Message` headers). Public
**ntfy.sh** allows about **2 MiB** per file (attachments expire in ~3 h). The
clip is the matched broadcast ± `pad_s` (default 12 s), encoded MP3 via `lame`
or `ffmpeg`/`imageio-ffmpeg`. If the clip is still too large or encoding fails,
the alert still goes out with a review-page / filename pointer and no attach.

## Run

```bash
# Replay existing decodes (no push):
python -m hfgcs.keyword_alerts --dry-run \
  --decodes ~/hfgcs-app/decodes.jsonl --recordings ~/hfgcs-app/recordings

# Live watch (systemd: hfgcs-keyword-ntfy.service):
export NTFY_SERVER=https://ntfy.sh NTFY_TOPIC=…   # from ntfy-*.env
python -m hfgcs.keyword_alerts --watch
```

On deploy, copy the example TOML, keep using the existing `ntfy-skyking.env`
topic (or a new `ntfy-keyword.env`), stop `hfgcs-skyking-ntfy`, enable
`hfgcs-keyword-ntfy`. Changing `hfgcs/keyword_alerts.py` or the unit file
restarts only that alert unit — not the recorders or the decode worker.
