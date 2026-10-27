# Decoding & Transcription design

Status: draft v0.1, 2026-10-26 (PDT). Owner: Decoding & Transcription. Changes land as PRs through the
Engineering Lead. This doc covers (1) ASR, (2) the EAM/Skyking grammar parser and its JSON schema,
(3) the labelled evaluation set, and (4) ALE decoding.

Code that exists today (prototype, stdlib only, branch `decoding-design`):

| Path | What |
|---|---|
| `decoding/tokens.py` | ASR word tokens, NATO/ICAO + HF variants (niner, tree, fife) to characters, spoken numbers |
| `decoding/eam_parser.py` | Grammar parser: `parse_transcript`, `parse_eam`, `parse_skyking`, segmentation, radio checks (test counts) |
| `decoding/output.py`, `decoding/run.py` | Per-recording decode document and its writer, `python -m decoding.run` (see "Decode output") |
| `decoding/worker.py`, `decoding/voice_gate.py` | Decode service: watches the catalog, voice gate, per-clip time cap, load-decodes (see "Decode service") |
| `decoding/align.py` | Read 1 / read 2 alignment and per-character consensus confidence |
| `decoding/rebroadcast.py` | Links airings of one message across time, frequency and receiver |
| `decoding/metrics.py` | CER, field exact-match, detection precision/recall |
| `decoding/asr.py` | faster-whisper front end (lazy import), prompt and hotword builders |
| `decoding/ingest_missileer.py` | Samples the missileer.watch CC0 feed into eval-label candidates |
| `tests/test_eam_parser.py`, `tests/test_ingest_missileer.py`, `tests/fixtures/missileer_archive_sample.json` | Unit tests and a 4-row fixture |

## Interface to the recording pipeline

Input is the Phase 0 catalog (`hfgcs_catalog.jsonl`, `"schema": 2`, branch `phase0-dsp-fixes`). Each line
describes one recording: `start_utc` (first sample of activity), `file_start_utc` (start of the pre-roll),
`end_utc`, `frequency_khz`, `mode`, `receiver_id`, `file` (`recordings/HFGCS_<f>kHz_<YYYYmmddTHHMMSSZ>_<receiver>.wav`,
PCM_16 mono at 8 kHz), `sha256`, `audio_rate`, `duration_sec`, `preroll_sec`, `snr_db_peak`/`snr_db_mean`,
`noise_floor_db` and `truncated`.

The decoder is a separate worker. It never touches the capture process.

1. It tails the catalog. Each entry is keyed by `sha256`, so re-runs are idempotent.
2. It writes one sidecar per recording, `<wav>.decode.json`, and appends the identical document as one line to `decodes.jsonl` (`python -m decoding.run`; format in "Decode output" below and `docs/decode.schema.json`). The Database & Analytics lead loads both with `hfgcs_db.py load-decodes`.
3. Absolute times are computed as `file_start_utc + word.start`. This matches missileer's `voice_onset` basis.
4. Entries with `truncated: true` are parsed, but the parser is expected to flag them as `partial_start`/`single_read`.

## 1. Speech-to-text for HF USB voice

**Pipeline:** WAV (8 kHz) or Opus, then ffmpeg preprocessing, then silero-vad (inside faster-whisper), then
faster-whisper with word timestamps, then word list `[{word, probability, start, end}]`, then the parser.

### Preprocessing
- **Bandpass to 300–3000 Hz** (`highpass=f=300,lowpass=f=3000`). This removes the hum and low-frequency fading energy below 300 Hz, and the hiss and adjacent-channel splatter above 3 kHz. HFGCS audio sits in the 150–2750 Hz slice of a 3 kHz USB channel.
- **Resample to 16 kHz mono** (Whisper's native rate). Upsampling 8 kHz audio adds no information, but it does no harm.
- **Level:** use a slow dynamic normaliser (`dynaudnorm=f=250:g=15`) instead of per-chunk peak normalisation. HF QSB fades over seconds, and a fast AGC lifts noise between characters. Option: `noisereduce` (spectral gating, MIT) behind a flag. Spectral gating can smear phonetic consonants, so it stays **off by default until the eval set shows a CER gain**. The same rule applies to RNNoise (`arnndn`).
- Keep the original file. The preprocessed 16 kHz WAV is temporary.

### VAD (silero-vad via faster-whisper): **off by default since the eval below**
On the eval set, silero dropped whole character groups of weak HF speech even at threshold 0.35 (one gold clip came
back as just "Message, Alpha, Charlie."). Turning it off was the largest single CER gain (gold 0.82 → 0.55), so
`AsrConfig.vad_filter` now defaults to `False` and the no-speech / hallucination guards do the gating. The tuned
parameters below are kept for `vad_filter=True` (cheaper on long, mostly-silent streams; about 2× faster):
faster-whisper's `VadOptions` defaults (v1.2.1) are tuned for clean speech. HF voice needs these changes:
- `threshold` **0.35** (default 0.5). Weak characters under noise score below 0.5 and get dropped.
- `min_silence_duration_ms` **1500**. Operators pause between groups and before "I say again". Splitting there breaks the context.
- `speech_pad_ms` 400 and `min_speech_duration_ms` 200, to keep single short words ("Two") inside a segment.
- Squelch tails, carrier drops and heterodynes give false VAD triggers. These are handled by the no-speech/hallucination guards below, not by VAD.
- Tune `threshold` on the dev split over the values {0.25, 0.35, 0.5}. The metric is body CER plus the rate of characters dropped.

### Model and compute
| Setting | CPU box (no GPU, 8 cores here) | GPU |
|---|---|---|
| Model | `small.en`, **float32** (default since decoding 0.3; int8 is not deterministic, see below) | `large-v3` or `distil-large-v3`, float16 |
| Approx. speed | small.en int8 runs near real time on 8 cores (to be measured on our clips) | far faster than real time |

Route low-confidence messages to a second pass with a larger model (two-pass escalation; the idea comes from shnikees/netcontroller, which is GPL, so we take the idea only). English-only `.en` models avoid language-detection errors on noise.

### Decoding settings
- `language="en"`, `beam_size=5`, `word_timestamps=True`.
- `condition_on_previous_text=True` (changed after the eval below). It can degenerate into repetition loops on long
  reads, so the parser strips n-gram loops first.
- `temperature=[0, 0.2, 0.4]` (a short fallback ladder). `no_speech_threshold=0.6`, `hallucination_silence_threshold=2.0` (this skips long silences where Whisper invents text).
- **`initial_prompt`**: a realistic EAM opening in the exact spelling we want back (Whisper imitates the style of its prompt). It includes "All stations … this is <callsign of the day>", phonetic characters in Title Case, "Niner", "X-ray", "stand by", "Message follows", "I say again" and "Skyking". Keep it short, because the prompt shares the 448-token context. See `decoding/asr.build_prompt`.
- **`hotwords`**: the NATO alphabet, "Skyking", "Mainsail", the Priyom station names and the **day's NCS callsign**. The NCS callsign changes daily at 00:00 UTC (per Priyom). Once a reliable callsign has been heard, feed it back for the rest of that UTC day. See `decoding/asr.build_hotwords`.

### Constraining output to the phonetic alphabet
Whisper has no grammar decoding. We constrain the output in three layers:
1. A prompt and hotwords that bias towards phonetic words.
2. Tolerant normalisation in the parser: `alfa`/`alpha`, `juliet(t)`, `niner`/`nine`, `tree`, `fife`, "x ray" to X, numerals, and tokens like "4K".
3. *Later:* a lexicon-constrained rescoring pass, which takes N-best hypotheses or token logits, keeps only in-vocabulary words inside message bodies and picks the best path. Alternatives are `suppress_tokens` with a whitelist, or a forced-alignment pass with whisperX/wav2vec2 (CTC) over a closed vocabulary.

Measure each layer's effect on the eval set before adopting it.

### Confidence signals
- `word.probability` from faster-whisper is the per-character ASR probability. It is passed through to `chars[].read1/read2` observations.
- **Read agreement** is the stronger signal (see §2): two independent readings plus the preamble (heard three times) give up to five observations of the first six characters.
- Segment `avg_logprob` and `no_speech_prob` are kept for gating.

### Reproducibility and long recordings (decoding 0.3, 2026-10-26)
**Root cause of the dropout.** CTranslate2's int8 CPU path is not deterministic on our CPUs. The encoder output for
the same 30 s of audio changed from call to call, with a max abs diff of about 34. That held with 1 thread, a fixed
seed, temperature 0, `MKL_CBWR=AUTO,STRICT` and `CT2_FORCE_CPU_ISA`. Whisper's no-speech and hallucination-silence
skips then fired on different windows each run. On weak HF speech a skip can jump over a whole read, or the rest of
the file. That is why the same clip gave 4 words on one run and 86 on another, why the 374 s two-EAM sample once gave
31 words and one EAM, and why the 0.542/0.486 gold CER from the int8 cache was not reproducible. A fresh int8 run on the
box scored 0.803 gold first-pass, 0.855 with vote3, and 0.568 silver first-pass. int16 and float32 are bit-identical
from run to run, and float32 is also identical at 1, 2, 4 and 8 threads, so `cpu_threads` stays out of the cache key.
float32 costs some speed: 120 s of audio at 4 threads took 19.6–20.0 s, against 9.5–18.2 s for int8.

**Defaults now (`AsrConfig`):**
- `compute_type="float32"`.
- `seed=0`: `ctranslate2.set_random_seed` runs before every decode, so the 0.2/0.4 temperature fallback is
  repeatable.
- **Windowing.** Audio longer than 1.33 × `window_s` (240 s) is decoded as independent windows of about 240 s. Each
  cut goes at the quietest 0.5 s within ±20 s of its nominal point.
- **Loop guard.** A segment is a loop when it has more than 3 words/s, or when one 1–3-word unit makes up 60 % of its
  12 or more words. Repeats are first cut to one occurrence ("Six Six Six …" becomes "Six"), and the segment is kept
  if that fixes it. Otherwise it is a loop, as is a window that goes over 3 words/s overall.
- **Wall-time cap.** One budget per decode: `max_rtf` (2) × the clip duration, for everything. A window's first decode
  may use 0.6 of its share (1.2 × its length). Fallbacks and repairs run only while budget is left, and skipped ones
  are reported. `guard.wall_s` holds the total.
- **Fallback.** After a loop or a timeout, the segments before the problem are kept. The rest of the window is
  decoded once more with `fallback` (VAD on, no conditioning, temperature 0).
- **Coverage guard.** A word-free stretch over 12 s that is not digital silence is decoded again on its own (fresh
  context), and only the words inside the gap are merged in. A stretch a fallback has already retried is not decoded
  again.
- **Reporting.** Each decode reports `asr.guard`, which holds the windows, fallbacks, repairs and coverage.
- **Cache key.** `AsrConfig.key()` hashes every field except `cpu_threads`.

The wall-time cap is the one guard that depends on the machine. Under heavy CPU contention it can fire where a loop
check would have, so a decode that hits it may differ between machines. Its `guard.fallbacks[].reason` is "timeout".
Healthy EAM coverage (word-span seconds / duration) is about 0.2–0.5.

**Worker use.** `python -m decoding.run <catalog.jsonl>` skips recordings whose sidecar already has the same sha256
and `decoder_version`. It runs at nice +10 by default (`--nice`) and appends to `decodes.jsonl` under a file lock.
It logs one line per recording to stderr: duration, wall time, RTF, words, coverage, windows, fallbacks and repairs.
`--max-rtf` and `--limit` are available. Exit codes: 0 means everything was decoded or already up to date. 1 means at
least one recording was missing or failed (the rest were still processed). 2 means the input list could not be read.

### Chunking long recordings
Recordings are often 2–4 minutes (missileer clips run 79–117 s plus pre-roll). faster-whisper already windows audio in 30 s chunks after VAD. We keep the word timestamps absolute, and we keep `chunk_length` at the default of 30. For multi-hour streams (YouTube or a NEET INTEL archive), split on VAD silences of 3 s or more into pieces of at most 10 minutes, with 5 s of overlap. Deduplicate words in the overlap by timestamp. whisperX's batched VAD-cut pipeline (wav2vec2 alignment) is an option once we have a GPU, to get tighter word times.

### Fine-tuning (later, Phase 3+)
Once there are 300 or more ear-verified messages (more than 2 hours of voice), fine-tune `small.en`/`distil` with LoRA on HFGCS phraseology. This follows the precedent of the ATCO2 airband fine-tunes that skywatch's eval script compares (`jlvdoorn/whisper-base.en-atco2-asr`). We can augment by synthesising EAM readings (TTS plus HF channel simulation: fading, noise, offset). Always evaluate on a held-out *receiver/date* split.

### First off-air smoke test (2026-10-26, CPU, `small.en` int8, settings above)
Two missileer CC0 clips with listener-corrected copy (the `/workspace/hfgcs-evaldata` sample; not committed):

| Clip | Length | ASR time (8 CPU cores) | Our body CER | missileer machine copy CER |
|---|---|---|---|---|
| eam-8c26a8bf (clip starts mid-message) | 96 s | 20 s | 0.72 | 0.91 |
| eam-7130de7e (35-char message) | 207 s | 23 s | 0.71 | 0.63 |

ASR speed is fine: about 5–10× faster than real time. **ASR accuracy is the bottleneck.** small.en dropped or mangled most phonetic words (it wrote "Avak", "V80" and "K-Back"), misheard "this complete message" as "this sweet message" and "follows" as "followed", and fell into a hallucination loop ("New York City" ×60) after the sign-off, even with `hallucination_silence_threshold`. The parser still found the structure: on eam-7130de7e it recovered the preamble as H7JAA (gold QH7JAP: Q dropped, P misheard), and it read both declared lengths (35 and 32) and both reads.

That run led to these parser changes, each with a regression test:
- accept "followed"
- accept letters spelled with hyphens ("K-R-A-B-X")
- strip n-gram repetition loops
- vote on the preamble by whole reading before voting per position
- end a body at any "message" token
- detect a partial start from the trailer alone

PyAV and faster-whisper 1.2.1 are out of step on this box (`metadata_errors` TypeError). `asr.py` therefore decodes with ffmpeg and passes a numpy array.

The follow-up experiments are in the next subsection.

### CER iteration on the eval set (2026-10-26, CPU, `small.en` int8 only)
**Command** (repeatable; ASR output is cached per config under `eval/cache/<AsrConfig.key()>/`, so rescoring after a
parser change is instant with `--cached-only`):
```
python -m decoding.evaluate --subset gold   --config baseline --config best [--cached-only] --out results/gold.json
python -m decoding.evaluate --subset silver --config baseline --config best --out results/silver.json
```
- **gold** = the 14 listener-corrected clips (8 EAMs, 6 radio checks = negatives).
- **silver** = 30 EAMs from the uncorrected feed with a full body (no `.`) and both reads. Silver labels are
  missileer's own machine copy, so silver CER measures agreement with their pipeline, not truth. It is a check
  against overfitting the 8 gold EAMs, not an accuracy figure.
- Body CER = masked Levenshtein / reference length over all EAMs (micro); `.` in a label matches anything.
  "Preamble EM" / "length EM" are exact matches of the first 6 characters / the declared length (count in brackets).
  "False EAM" = an EAM emitted on a gold negative. RTF = ASR seconds / audio seconds on 8 cores.
- Configs are overrides on the v0.1 settings (`evaluate.CONFIGS`, `V01_BASE`). Every row below was rescored with
  the final parser. Rows not listed were not run, because VAD-off dominated.

**Gold (14 clips; missileer's own machine copy scores 0.418 on the same 8 EAMs):**

| config | what changes | body CER (micro) | CER median | preamble EM | length EM | detected | false EAM | RTF |
|---|---|---|---|---|---|---|---|---|
| baseline | v0.1: silero VAD (t=0.35), EAM prompt + hotwords, bp_norm | 0.823 | 0.867 | 0.000 | 0.333 (3) | 0.750 | 0/6 | 0.069 |
| best | **chosen defaults**: VAD off, cond_prev on, + snap | 0.542 | 0.562 | 0.125 | 0.333 (3) | 1.000 | 0/6 | 0.196 |
| no_vad | VAD off | 0.554 | 0.600 | 0.250 | 0.333 (3) | 0.875 | 0/6 | 0.176 |
| no_vad_snap | VAD off + snap | 0.518 | 0.600 | 0.250 | 0.333 (3) | 1.000 | 0/6 | 0.176 |
| nv_cond_prev | VAD off, condition_on_previous_text | 0.534 | 0.562 | 0.125 | 0.333 (3) | 1.000 | 0/6 | 0.196 |
| nv_cond_prev_snap | = best | 0.542 | 0.562 | 0.125 | 0.333 (3) | 1.000 | 0/6 | 0.196 |
| no_vad_noskip | VAD off, all skip/no-speech guards off | 0.566 | 0.733 | 0.125 | 0.333 (3) | 1.000 | 0/6 | 0.121 |
| nv_no_prompt | VAD off, no prompt | 0.819 | 1.000 | 0.125 | 0.667 (3) | 0.750 | 0/6 | 0.190 |
| nv_phonetic | VAD off, phonetic-only prompt | 0.747 | 0.906 | 0.125 | 1.000 (3) | 1.000 | 0/6 | 0.357 |
| nv_no_hot | VAD off, no hotwords | 0.679 | 0.700 | 0.125 | 1.000 (3) | 1.000 | 0/6 | 0.117 |
| nv_pre_none | VAD off, no preprocessing | 0.735 | 0.933 | 0.125 | 0.667 (3) | 0.750 | 0/6 | 0.194 |
| nv_pre_bp | VAD off, bandpass only | 0.719 | 0.900 | 0.125 | 0.333 (3) | 0.875 | 0/6 | 0.154 |
| nv_pre_denoise | VAD off, + afftdn denoise | 0.627 | 0.600 | 0.125 | 0.667 (3) | 0.875 | 0/6 | 0.173 |
| nv_pre_denoise_snap | VAD off, denoise + snap | 0.631 | 0.633 | 0.250 | 0.667 (3) | 0.875 | 0/6 | 0.173 |
| nv_pre_wide | VAD off, 100-3800 Hz | 0.647 | 0.833 | 0.125 | 0.667 (3) | 0.750 | 0/6 | 0.148 |
| nv_no_hst | VAD off, no hallucination_silence_threshold | 0.839 | 1.000 | 0.125 | 0.333 (3) | 0.500 | 0/6 | 0.077 |
| vocab_novad | VAD off, suppress_tokens whitelist | 0.631 | 0.733 | 0.125 | 0.000 (3) | 0.750 | 0/6 | 0.157 |
| chunk8 | own VAD windows ≤8 s | 0.912 | 0.967 | 0.000 | 0.333 (3) | 0.750 | 1/6 | 0.145 |
| no_prompt | no prompt | 0.867 | 0.967 | 0.000 | 0.333 (3) | 0.625 | 2/6 | 0.051 |
| no_prompt_no_hot | no prompt, no hotwords | 0.831 | 0.857 | 0.000 | 0.667 (3) | 0.750 | 1/6 | 0.042 |
| phonetic_prompt | phonetic prompt | 0.884 | 1.000 | 0.000 | 0.667 (3) | 0.625 | 0/6 | 0.057 |
| phonetic_prompt_no_hot | phonetic prompt, no hotwords | 0.896 | 0.933 | 0.000 | 0.667 (3) | 0.625 | 1/6 | 0.035 |

**Silver (30 EAMs):**

| config | body CER (micro) | CER median | preamble EM | length EM | detected | RTF |
|---|---|---|---|---|---|---|
| baseline | 0.693 | 0.667 | 0.200 | 0.000 (6) | 0.867 | 0.034 |
| best | 0.387 | 0.229 | 0.500 | 0.667 (6) | 1.000 | 0.132 |
| no_vad | 0.602 | 0.346 | 0.433 | 0.667 (6) | 0.967 | 0.090 |
| no_vad_snap | 0.591 | 0.306 | 0.500 | 0.667 (6) | 0.967 | 0.090 |
| nv_cond_prev | 0.416 | 0.257 | 0.467 | 0.667 (6) | 1.000 | 0.132 |

**Production default = `pipeline.DecodeConfig()` = eval config `default` (alias `best_vote`):** the `best` single
pass below, plus the gated 3-way-vote second pass (next subsection), with preamble fusion off. Gold body CER is
**0.486** and silver **0.364**. CPU is 1.35× (silver) to 1.77× (gold) the single pass, and the second-pass decodes
share the first pass's loaded model, so they add no extra RAM. `python -m decoding.evaluate` with no `--config`
scores this default. Usage: `Decoder().decode(path, day_callsigns)` returns
`{"records", "second_pass": {"fired", "passes"}, "duration_s"}`.

**Chosen first pass** (`AsrConfig()` = the `best` row): VAD off; `condition_on_previous_text=True`; EAM prompt plus
hotwords; `bp_norm` preprocessing; `hallucination_silence_threshold=2.0`; then `snap.snap_words` on the word list
before the parser. On gold, `best` (0.542) and `no_vad_snap` (0.518) are within noise (about 6 characters of 249).
On silver, `best` is far ahead (0.387 vs 0.591, median 0.229 vs 0.306), so pooled over both it wins (about 0.42 vs
0.58). Caveat: silver labels are missileer's machine copy, and their pipeline may also decode with previous-text
conditioning, which would flatter cond-prev on silver. Re-check once there are more gold labels. Peak RSS of one
evaluate process is about 1.7 GB. `vad_t10_pad` was stopped at 10/14 gold clips (4 EAMs, 0.842) and is not listed.

**What helped (body CER, baseline → best: gold 0.823 → 0.542, best gold row 0.518 `no_vad_snap`; silver 0.693 → 0.387):**
1. **VAD off** (0.823 → 0.699 with the old parser). Silero cut weak characters and whole groups.
2. **Parser robustness** on real ASR output (0.699 → 0.554 on the same no-VAD transcripts), each with a regression
   test: "follow"/"followed"/"Classic. Follow." accepted as *message follows*, an inferred "message follows" when it
   is missing, "I'd / iced it / said again" joined to *say again*, preamble repeats stripped from the start of read 1,
   concatenated reads split by self-alignment, and an unstructured fallback for long character runs.
3. **`condition_on_previous_text=True` once VAD is off** (silver 0.602 → 0.416; gold 0.554 → 0.534). Context
   between 30 s windows keeps the phonetic style going, and the repetition loops it used to cause are now removed
   by `drop_repetition_loops`.
4. **Phonetic snapping** (`snap.py`; gold 0.554 → 0.518 and silver 0.602 → 0.591 on no-VAD; silver 0.416 → 0.387 on
   cond-prev, but gold 0.534 → 0.542 there; preamble EM up on silver).
   Near-miss words ("Elpha", "Fokstrot") are snapped to the NATO word by edit distance plus a sound key, with a
   protected list of real English words ("ray", "brake", "for", "to") that must not become letters.
5. Keeping the EAM prompt plus hotwords (no prompt: 0.819; phonetic prompt: 0.747) and the `bp_norm` chain (no
   preprocessing 0.735; bandpass only 0.719; denoise 0.627; wide 0.647).

**What did not help:** chunked decoding with our own VAD windows (0.91), `noskip` (all skip guards off: 0.566 without VAD), dropping `hallucination_silence_threshold` (0.839, loops return), the closed-vocabulary
`suppress_tokens` whitelist (0.631: it forces garbage words into letters), and `afftdn` denoise (0.627).

**Second pass: 3-way character vote (`vote.py`, config `best_vote` = `default`; on by default since 2026-10-26,
turn it off with `DecodeConfig(second_pass=False)`).**
- *Trigger* (`vote.needs_second_pass`): the first-pass EAM's `confidence_detail.reads_agree_ratio < 0.3`, which includes
  every single-read EAM. Clips with no EAM body (radio checks, noise) never trigger. Structural flags were tried as
  the trigger, but `single_read`/`length_mismatch` are on about 85 % of current first passes, so they select nothing.
- *Second pass*: decode again with `no_vad_snap` and `baseline` (VAD on, cheap). Each body is aligned to the
  medoid of the three (ties go to pass 1) with `align.py` and voted per column. A column is dropped when most passes
  have a gap there, and insertions relative to the medoid are ignored. The record keeps `body_first_pass` and gets
  the flag `second_pass_vote`.

| subset | single pass (`best`) | `best_vote` (gated) | vote on every EAM | fired | RTF single → gated (cost) |
|---|---|---|---|---|---|
| gold (8 EAMs, 14 clips) | 0.542 | **0.486** | 0.486 | 5 of 8 EAMs, 0 of 6 negatives | 0.196 → 0.347 (1.77×) |
| silver (30 EAMs) | 0.387 | **0.364** | 0.354 | 10 of 30 | 0.132 → 0.178 (1.35×) |

The earlier gold-only lead, a third pass of `no_vad_noskip` instead of `baseline`, does not hold up on silver: always-vote gives gold 0.474 but silver **0.482** (gated: 0.522 / 0.461), and `no_vad_noskip` alone scores 0.724 on silver. So the second pass uses `baseline`. Gated voting gets all of the always-vote gain on gold and about 70 % of it on silver, at about a third of the extra
CPU. The cost is higher on gold because its triggered clips are the long ones. Preamble and length exact-match are
unchanged, because the vote only rewrites `body`. The CPU numbers are measured ASR seconds from the cache, with all
passes on 8 threads.

**Preamble fusion (`preamble.py`, configs `best_pre_vote` / `best_pre_patch`; not on by default).**
- The preamble is voted over every hearing: up to three pre-reads plus the first six characters of read 1 and of
  read 2 (read starts are skipped on `mid_message_start`). Pre-reads longer than six are cut to their last six,
  because junk such as a misheard "for" lands in front, just before "stand by". Shorter pre-reads only vote. The
  pivot is the medoid of the six-character hearings.
- `patch` additionally replaces the best-aligned 4–8-character start of the body with the fused preamble when it is
  at most 2 edits away.

| subset | preamble EM before → after (vote) | body CER `best` → `best_pre_patch` | body CER `best_vote` → `best_vote_pre_patch` |
|---|---|---|---|
| gold (8) | 0.125 → 0.125 | 0.542 → 0.550 | 0.486 → 0.494 |
| silver (30) | 0.500 → 0.533 | 0.387 → 0.379 | 0.364 → 0.357 |

This is neutral within noise: one more silver preamble right, and ±2 characters of body. It is not enabled by default.
The reason gold does not move is systematic: on 5 of 8 gold EAMs **every** hearing has lost the same leading
character or two (QHUDS2 → "HDS2Z", ACC6GX → "CCGX", QHJY7J → "HJET7J"; the Q/V of the QH…/VQ… prefixes is
clipped at each onset). Voting cannot recover a character that no hearing contains. A second decode with the
fused preamble in the prompt was not run: it needs a per-clip re-decode (about 2× CPU on the triggered clips), and
it would at most restore one dropped onset character. **Preamble-prefix prior, tried and not adopted (2026-10-26).** First-pair counts were learned from the 148 silver
training preambles only (silver EAMs not in the 30-clip eval subset; gold excluded): 56 ×59, QH ×21, AC ×11,
WE ×10, VQ ×6. Rule: when the preamble is shorter than 6, or the body is shorter than expected, and the first heard
character x is far more often the *second* character of a pair (c, x) than a first character
(P = n(cx) / (n(cx) + n(x first) + n(other ·x)) ≥ 0.6, n(cx) ≥ 5), prepend c. At 0.6 the rule restores H→QH,
6→56 and C→AC; Q is not restored, because Q usually starts a preamble. Results:
- gold body CER 0.542 → 0.538 (`best`) and 0.486 → 0.482 (default), one character;
- silver: no change at all (0.387 / 0.364);
- preamble exact-match unchanged everywhere (gold 0.125, silver 0.500), because the gold preambles that lose
  their Q also have a second error (QHUDS2 → "HDS2Z").

The rule did not help on silver, so per the decision rule it is not committed as code.

**Mid-message start detection (`eam_parser.mid_message_start_kind`).** A headless EAM (`partial_start`) whose
transcript has **neither a preamble hearing nor "message follows"** (an inferred one counts as present), yet does have
a body, gets the flag `mid_message_start` and a field `mid_message_start` with one of three values:
- `read2_only`: fewer than 3 characters before "I say again".
- `in_read1`: a read-1 tail, then "I say again", or a self-split of concatenated reads.
- `body_only`: no "I say again", so the clip started inside read 1 or read 2 and which one is unknown.

On a mid-message start, preamble fusion ignores the read starts. A missing word can also be an ASR drop, so this is a
flag, not proof. As a reference we use missileer's anchors: no `standby` and no `message_follows` anchor means a
mid-message start.

| subset (EAMs scored) | `partial_start` | `mid_message_start` | per kind | agrees with missileer anchors |
|---|---|---|---|---|
| gold (8) | 4 | 1 (eam-8c26a8bf) | in_read1 1 | 1/1 true positive, 0 false positives, 0 missed |
| silver (30) | 7 | 1 (eam-a3b2b5e0) | in_read1 1 | 1/1 true positive, 0 false positives, 0 missed |

The silver scoring subset requires both reads, so it under-represents mid-message clips. missileer's own `partial`
flag is set on 7 of those 30 and on none of the 8 gold EAMs, so it means something else (an incomplete copy), not a
late start. Body CER and exact-match are unchanged, because detection does not alter the body.

**Remaining failure modes:**
- Mangled phonetic words that are too far for safe snapping: "Dory", "Big Door", "Koopalink", "Whitty", "Aloha".
- Clips that start mid-message (no header or preamble), so only one partial read is available. They are now flagged
  (`mid_message_start`) but not recovered; simulcast or rebroadcast copies are the fix.
- Hallucination loops after the sign-off or in long carrier gaps. They are removed as loops but sometimes leave junk
  characters at a body edge.
- Low preamble exact match on gold (0.25). One wrong character in 6 fails the match.
- Unknown daily callsigns, which the hotwords cannot know until one has been heard.
- One gold clip (eam-ed75798a) holds two messages, and CER above 1 there is a labelling and segmentation mismatch.
- Gold has only 8 EAMs, so differences under about 0.03 are noise. Silver confirms the ordering of the top rows.

Fine-tuning plan (CPU LoRA, augmentation, silver-label risk, gold-label power analysis): `docs/finetune.md`.

Next: `medium.en` or `distil-large-v3` as a second pass for low-confidence bodies (needs more RAM or a GPU),
the 3-way vote, and a CTC forced-alignment rescoring pass over the phonetic lexicon.

### Reuse from brisi-jt/skywatch (verified: MIT, `LICENSE` © 2026 James Towns, last push 2026-10-21)
Our repo is MIT too, so copying is allowed with the notice kept. **As of this commit nothing is copied.** The patterns
below are already implemented in our own code, and `verbalize.py` is ICAO-specific. `THIRD_PARTY.md` records
skywatch as a design reference. Any future copy keeps skywatch's copyright header in the file and adds a line there.
- **Reusable with attribution** (MIT notice kept in any copied file):
  - The `ASREngine` Protocol plus a `TranscriptionResult` dataclass with provenance (engine, model, language, segments with `avg_word_prob`). We use the same contract, so whisper.cpp can be swapped in later.
  - The lazy-loading `FasterWhisperEngine`, which imports faster-whisper on first use so CI runs without the ASR extra. `decoding/asr.py` follows this pattern.
  - Callsign biasing: `verbalize.py` turns callsigns into spoken forms and passes them as both `hotwords` and `initial_prompt` ("Aircraft callsigns heard nearby: …"). Its note that Whisper emits plain-English digit words rather than "niner/tree" matches our tolerant normaliser.
  - The `scripts/eval_asr.py` A/B harness (two models over a YAML eval set) and its `content/eval_set.yaml` label layout.
- **Not applicable / not present:** skywatch sets **no VAD parameters**. Squelch comes from rtl-airband, and `transcribe()` uses faster-whisper defaults (no beam, VAD or temperature settings). It also has no hallucination filter. Our VAD tuning is our own.
- Not to copy: the vendored `airlines.dat` / `plane_alert_db.csv` data, which have their own upstream licenses. We don't need them anyway.

## 2. EAM / Skyking grammar parser

### Procedure (source: Priyom, https://priyom.org/military-stations/united-states/hfgcs, read 2026-10-26)
EAM: "usually 30 characters long".
1. "All stations, all stations, this is <callsign>, <callsign>."
2. Optional: "For <callsign>", repeated twice.
3. "The first six characters of the message, followed by 'Stand by,' repeated altogether thrice."
4. "Message follows." If the message is not 30 characters long: "Message of <number> characters follows."
5. The message in full. 6. "I say again." 7. The message in full.
8. Optional: "This complete message of <number> characters."
9. "This is <callsign>, out." or "This is <callsign>, more to follow, stand by."

Skyking:
1. "Skyking, skyking, do not answer."
2. The codeword, twice.
3. "Time" followed by the minutes of the hour as two digits, each spelled separately.
4. "Authentication" followed by two letters.
5. "I say again," followed by steps 1–4.
6. "This is <callsign>, out."

Other Priyom notes:
- Test count: "This is <callsign> with a test count; testing, 1 2 3 4 5, 5 4 3 2 1; this is <callsign>, out."
- Operators sometimes say "disregard" after a message.
- Mainsail means any ground station. The NCS callsign changes daily at 00:00 UTC.

Community notes (scout digest, unverified here): Skyking has been dormant for years, and "SKY CHIEF" was once misheard as "SKYKING". So the Skyking detector requires "do not answer" or a time/authentication structure before it raises alerts.

### Grammar (tolerant FSM over normalised tokens)
```
recording   := noise* (broadcast noise*)*
broadcast   := eam | skyking | radio_check | other
radio_check := THIS IS callsign [WITH A] TEST COUNT digits… [THIS IS callsign OUT]   # HFGCS test count
eam         := ALL STATIONS{1,2} THIS IS callsign{1,2}
               (FOR addressee){0,2}
               (char{6} STANDBY){1,3}                      # preamble x3
               MESSAGE [OF number CHARACTER(S)] FOLLOW(S)
               char+                                       # read 1
               [I SAY AGAIN char+]                         # read 2
               [THIS COMPLETE MESSAGE OF number CHARACTERS]
               [THIS IS callsign (OUT | MORE TO FOLLOW STANDBY)] [DISREGARD]
skyking     := (SKYKING{1,2} [DO NOT ANSWER] codeword{1,2} TIME digit digit AUTHENTICATION letter letter)
               [I SAY AGAIN <same>] THIS IS callsign OUT
char        := NATO word | digit word | numeral | single letter
```
Segmentation: a new broadcast starts at "all stations" or "skyking" only after the previous one has closed ("out" or "more to follow"), so the Skyking repeat after "I say again" is not split off. A radio check ("this is X … test count" within 8 words) after a closed broadcast also starts one. Record types are `eam`, `skyking`, `radio_check` and `other` (nothing recognisable), the names `labels.jsonl` uses; until parser schema 0.2 they were `test_count` and `unknown`. Every deviation is a **flag**, not an exception. These include a missing header (`partial_start`), a clip that began inside the message itself (`mid_message_start`, see below), `single_read`, `missing_preamble`, `preamble_reads_disagree`, `reads_disagree`, `length_mismatch`, `declared_length_conflict`, `preamble_body_mismatch`, `non_character_words_in_body`, `closing_callsign_mismatch`, `invalid_time`, and others.

### Per-character confidence
Read 1 and read 2 are aligned with Levenshtein (`align.py`). Each column then gets:
- **Agree:** confidence = 1 − (1−p₁)(1−p₂), where p is ASR word probability.
- **Substitution:** keep the more probable character; confidence = p·0.5.
- **Gap** (present in one read only): confidence = p·0.4.
- **Single read only:** p·0.6.
- **Preamble (first 6 positions):** a matching preamble character lifts confidence by 1−(1−c)(1−0.5). When the reads disagree, the preamble breaks the tie.

The constants are placeholders, to be calibrated on the eval set with a reliability diagram. `body_masked` replaces characters below 0.5 with `.` (NEET INTEL convention). Planned: fuse more airings (simulcast receptions and rebroadcasts) with the same per-position vote, as missileer's `preamble_vote` does.

### Rebroadcast / simulcast linking (`rebroadcast.py`)
Airings are linked with union-find when both hold:
- `similarity.text_similarity(body_a, body_b) ≥ 0.8` (that is, body CER ≤ 0.2). `text_similarity` implements the
  `text_similarity(a, b) -> float | None` contract in `docs/database.md` on `db/schema-v1`: `.`/`?` wildcards,
  `None` when too few known characters, a value in [0, 1], and agreement with the DB's built-in wildcard Hamming on
  equal-length bodies. A `None` result never links.
- They are within 24 h of each other.

Within a group, each airing gets a relation:
- `first` for the earliest.
- `simulcast` when it is within 30 s of the first (same emission heard on another frequency or receiver).
- `rebroadcast` otherwise.

The group id is `eam-<first UTC>-<first 6 chars>`. NEET INTEL reports that about 1 in 3 EAMs is rebroadcast, mostly within 2 h. Linking by preamble alone fails, because messages that share a preamble exist (the scout digest's 56HETY ×9). That is why we compare full bodies.

### Parser record schema (v0.2; one record per broadcast)
```jsonc
{
  "type": "eam" | "skyking" | "radio_check" | "other",
  "schema_version": "0.2",
  // common to all types (v0.2)
  "callsign": "MAINSAIL" | null,          // identifying station: closing "this is X", else the opening one
  "callsigns_all": [{"callsign": "MAINSAIL", "role": "opening" | "closing" | "addressee" | "station"}],
  "confidence": 0.97,                     // 0-1, see below
  "confidence_detail": {...},             // the per-type structure (v0.1's "confidence" for EAMs)
  // EAM
  "opening_callsign": "MAINSAIL", "opening_callsign_confidence": 0.93, "addressee": "RED DOG" | null,
  "preamble": "QKR4TZ", "preamble_reads": ["QKR4TZ", "QKR4TZ", "QKR4TZ"],
  "declared_length": 30 | null, "length_source": "announced" | "default", "expected_length": 30,
  "read1": "…", "read2": "…", "body": "…", "body_masked": "QKR4TZ7H.B2…", "body_length": 30,
  "chars": [{"char": "Q", "confidence": 0.998, "read1": "Q", "read2": "Q", "agree": true, "preamble": "Q"}],
  "confidence_detail": {"mean": 0.97, "min": 0.41, "low_count": 1, "reads_agree_ratio": 0.967},
  "timing": {"read1": {"start_sec": 74.1, "end_sec": 140.2}, "read2": {"start_sec": 145.0, "end_sec": 211.7}},
  "closing_callsign": "MAINSAIL", "out": true, "more_to_follow": false, "disregard": false,
  // Skyking: each field = {"value", "read1", "read2", "agree", "confidence"}; also closing_callsign
  "codeword": {...}, "time": {...}, "authentication": {...},
  // other: "text"
  "flags": ["reads_disagree"],
  "segment": {"token_start": 0, "token_end": 212, "start_sec": 31.0, "end_sec": 215.3,
              "word_start": 0, "word_end": 230}   // inclusive input (ASR) word indices
}
```
`confidence` per type: EAM = mean per-character body confidence (`confidence_detail.mean`); Skyking = the minimum
of the codeword, time and authentication confidences (`confidence_detail` = those three); `radio_check`/`other` =
mean ASR word probability of the broadcast's words (`confidence_detail = {mean_word_prob, n_words}`; words
rewritten by `snap_words` count at their halved probability). The parser keeps each token's source word range
(`Token.src`), so `segment.word_start/word_end` map a broadcast back to the ASR words even when tokens were split
("Alpha,Bravo"), joined ("X ray", "I say again") or dropped (repetition loops).

## Decode output (contract v1 with Database & Analytics, 2026-10-26)

`python -m decoding.run <catalog.jsonl | audio …>` (or `decoding.run.decode_recording`) decodes each recording
with the default `Decoder` and writes `<audio>.decode.json` next to it, plus the **identical** document as one
compact line of `decodes.jsonl` (beside the catalog or the first audio file; `--jsonl`, `--no-jsonl`). A sidecar
with the same sha256 and `decoder_version` is skipped (`--force` re-decodes). `--asr-cache DIR` reuses
`decoding.evaluate`'s cache layout. JSON Schema: `docs/decode.schema.json`. Sample:
`tests/fixtures/decode_sample_two_broadcasts.decode.json` (a real decode of two concatenated gold clips).

| Field | Meaning |
|---|---|
| `schema_version` | document schema, `"1.0"` |
| `decoder_version` | `decoding/<decoding.__version__>;parser=<parser schema>;asr=<AsrConfig.key()>;second_pass=vote3@0.3\|off` (also mirrored in `asr.decoder_version`) |
| `recording` | `path` and `sha256` of the file decoded, then every Phase 0 schema-2 catalog field verbatim (`start_utc`, `file_start_utc`, `end_utc`, `frequency_khz`, `mode`, `receiver_id`, `file`, `audio_rate`, `duration_sec`, `preroll_sec`, `snr_db_peak`, `snr_db_mean`, `noise_floor_db`, `truncated`, `notes`). A catalog sha256 that differs from the file is kept as `catalog_sha256`. Bare audio gets `file` = `path`. |
| `duration_s` | decoded audio length (s) |
| `decoded_at` | ISO 8601 UTC, whole seconds, trailing `Z` |
| `asr` | `model`, `compute_type`, `config_key`, `day_callsigns`, `duration_s`, `asr_seconds`, `text`; `words` = first-pass words `{word, start, end, probability[, snapped_to]}`; `segments` = `{start, end, text, avg_logprob, no_speech_prob, word_span}`; `first_pass` = `{config_key, model, compute_type, asr_seconds}`; `second_pass` = `{mode, enabled, threshold, configs[{config_key, overrides, snap}], fired, passes, asr_seconds, second_pass_asr_seconds}` |
| `broadcasts[]` | parser records with these fields first: |
| `broadcast_id` | `<sha256[:12]>@<start_s rounded to 0.1>`, e.g. `068ccac7a46b@0.7`. If two broadcasts of one recording would collide, `#2`, `#3` is appended (never seen). |
| `type` | `eam` \| `skyking` \| `radio_check` \| `other` |
| `start_s`, `end_s` | seconds from the start of the audio file (first/last word of the broadcast); always numbers (0.0 for an empty transcript) |
| `asr_word_span` | inclusive `[i, j]` into `asr.words`; null only when there are no words |
| `transcript` | that broadcast's own ASR words, `" ".join(asr.words[i..j].word)` |
| `callsign`, `callsigns_all` | see the parser schema above |
| `confidence`, `confidence_detail` | 0-1 float, and the per-type structure |

All times are file-relative; absolute time = `recording.file_start_utc + start_s`. When `recording.truncated` is
true the last broadcast also gets the flag `truncated`. Timings and word spans always refer to the **first** ASR
pass: the second-pass vote only replaces `body` (keeping `body_first_pass`), so spans stay valid.

## Decode service (`python -m decoding.worker`, unit `hfgcs-decode`)

The long-running worker that turns new captures into rows on the review page. Install steps for hilltop:
docs/deploy.md "Decode service".

| Mode | What it does |
|---|---|
| (default) | watch loop: every `--poll` s (30), process captures whose `end_utc` is after the worker's first start |
| `--once` | one pass of the watch loop, then exit |
| `--backfill` | every capture in the catalog, older ones included, oldest first; repeats until nothing is left; exits (`hfgcs-decode-backfill.service`) |
| `--check` / `--prefetch` / `--status` | readiness check (exit 78 if not ready) / download the model / per-capture state |

Per capture, one at a time (a lock file in the state dir also serialises a backfill running beside the unit):

1. **Ingest**: when the catalog grew, `hfgcs_db.py ingest-catalog` (idempotent), so every capture is on the
   review page even before or without a decode.
2. **Voice gate** (`decoding/voice_gate.py`): ffmpeg `bp_norm` to 16 kHz, then silero-vad (bundled with
   faster-whisper) at threshold 0.3; under 1.0 s of speech, or digital silence, the clip is recorded as
   `no_voice` and not decoded. Deliberately permissive (silero misses weak HF speech, see "VAD" above); the ASR
   still runs with VAD off. `--no-gate`, `--gate-threshold`, `--min-speech`.
3. **Decode** in a child process (`decode_recording`: `small.en`, parser, vote3 second pass when the reads
   disagree) writing the sidecar and a `decodes.jsonl` line, as `python -m decoding.run` does. The child runs
   in its own process group under a **wall-time cap** of `--timeout-factor` (10) x the clip length, clamped to
   `--timeout-min` (120 s, covers model load) .. `--timeout-max` (7200 s: decoding 0.3 bounds each ASR pass at `max_rtf` 2x the clip, and the vote adds two passes, so a 900 s clip can legitimately take ~6x = 5400 s plus model load); over the cap the group is SIGTERMed,
   then SIGKILLed, and the clip is recorded as `timeout`. Failures and timeouts are tried `--max-attempts` (2)
   times in all; `--retry-failed` tries them again.
4. **Load**: `hfgcs_db.py --db $HFGCS_DB load-decodes <sidecar>`. load-decodes keeps rows a person edited or
   checked (the new machine copy is stored beside them, unapplied). A failed load is retried on the next pass
   without decoding again.

Idempotency: `~/.local/state/hfgcs/decode-state.json` keyed by the catalog sha256 holds each clip's verdict and
the `decoder_version` (and the gate settings for `no_voice`). A finished clip is skipped; a new decoder version
(package version, ASR config, parser schema, second-pass mode) makes every clip due again. Underneath, the
sidecar check in `decode_recording` and load-decodes' `(decoder_version, record_sha256)` dedup make a repeat a
no-op. `decode-history.jsonl` beside it has one line per clip: status, wall seconds, clip length, broadcasts.

The worker is stdlib only and never imports faster-whisper itself. The child needs the `decode` extra
(`pip install -e '.[decode]'`: faster-whisper plus imageio-ffmpeg, whose static ffmpeg is linked into
`~/.cache/hfgcs/bin` when the system has no `ffmpeg`) and the model in the Hugging Face cache
(`python -m decoding.worker --prefetch`; `small.en` is ~460 MB). Without them the worker exits 78 with the install
steps, and the unit does not restart-loop. Settings come from flags or `~/.config/hfgcs/decode.env`
(`deploy/env/decode.env.example`): `HFGCS_DB`, `HFGCS_CATALOG`, `HFGCS_DECODE_MODEL`, `HFGCS_DECODE_THREADS`
(3 on hilltop), `HFGCS_DECODE_TIMEOUT_*`, `HFGCS_DECODE_POLL_S`.

Cost (measured 2026-10-26 on the box, 3 threads at nice 19 on a heavily loaded 8-vCPU Xeon): the 207 s gold
clip took 307 s wall but only ~120 CPU-seconds (first pass only; the vote did not fire), i.e. about 0.6
CPU-seconds per second of audio per pass. A 96 s gold clip where the vote fired (three passes) took 192 CPU-seconds and peaked at 1.6 GB RSS. The gate rejected 30 s of pink noise and a 20 s tone in about 1 s
each. On hilltop (i5-6400, 4 cores, other load, `Nice=19`/`CPUWeight=10`) expect roughly real time when about
one core is idle, and up to ~3x real time for a clip where the vote fires (three passes).

### Worker integration with decoding 0.3.x (note for the Engineering Lead, 2026-10-26)
`decoding.worker` already decodes through `decode_recording`, so it gets 0.3.x's float32, windowing, loop guard and
budget without code changes. The version bump (`decoding/0.3.1;…;asr=437b8c03f291;…`) makes every clip due once
more. To run the decoder by hand, or from another scheduler:

    nice -n 19 python -m decoding.run <catalog.jsonl> --max-rtf 2 --cpu-threads 3 [--jsonl decodes.jsonl] [--limit N]

- **Exit codes.** 0 means every recording was decoded or already up to date. 1 means at least one was missing or failed;
  the others were still processed and the traceback is in the log. 2 means the input list could not be read. Each
  recording gets one INFO line on stderr: duration, wall time, RTF, words, coverage, windows, fallbacks, repairs and
  whether the second pass fired. The summary `decoded N, up to date N, failed N` goes to stdout.
- **Idempotency.** A recording whose `<audio>.decode.json` already has the same sha256 and `decoder_version` is
  skipped, so re-running over a growing catalog is cheap. `--force` decodes again. Lines are appended to
  `decodes.jsonl` under an flock, so concurrent runs are safe. `decoding.run` adds nice +10 on its own (`--nice`).
  `nice -n 19` or the unit's `Nice=19`/`CPUWeight=10` is still the right setting on hilltop.
- **Threads.** Use 3 on hilltop (4 cores; the recorders and OpenWebRX come first). float32 output does not depend on
  the thread count, so any setting gives the same decode.
- **Time.** float32 small.en first pass runs at about 0.35 RTF at 8 threads on an idle box, and 0.5–0.6 at 4 threads
  under load. Expect about real time at 3 threads on hilltop. On bad audio, Whisper's temperature retries push a pass
  to the cap: `--max-rtf 2` bounds each ASR pass, including its fallbacks and repairs, at 2× the clip length. When the
  vote fires there are three passes, so the worst case is about 6× plus model load. That is still inside the worker's
  own 10× process kill (`--timeout-factor`). A clip that hits the inner cap is decoded anyway, with
  `asr.guard.fallbacks[].reason = "timeout"`. It is not a failure.
- **Memory.** Peak RSS was 3.3 GB for the 374 s two-EAM sample at 8 threads with the vote firing (float32 weights
  are about 1 GB, 4× int8's). Budget for about 3–3.5 GB for the decode child on hilltop.
- **Feeding the DB.** The sidecars (`<audio>.decode.json`) or the `decodes.jsonl` lines are contract-v1 documents,
  and they go straight into `scripts/hfgcs_db.py --db $HFGCS_DB load-decodes <files>` from #11. The worker does this
  per clip. load-decodes keeps human-edited rows and dedups on `(decoder_version, record_sha256)`. 0.3 adds an
  optional `asr.guard` object, which is in `docs/decode.schema.json`. The loader ignores it.

### Known issues (decoding 0.3.1, shipped as-is so hilltop has decodes)
- **eam-b858e080 (43 s gold radio check) now gives 0 words** (25 with the old int8 cache). The first segment and its
  VAD fallback are both judged loops and dropped. The fix is in progress.
- **eam-8c26a8bf (96 s gold EAM) is typed Skyking, with body CER 1.0** (0.56 with the old int8 cache). float32 hears
  "Skyking" in the opening, and the first pass skips 39–73 s; the coverage repair gets those words back.
- **The full gold/silver re-measure is pending.** The 5-clip sanity check gave identical output across runs and these
  body CERs against the int8 cache: eam-356f1a01 0.17 (0.27), eam-f27e2d58 0.57 (0.67), eam-8c26a8bf 1.00 (0.56).
  Earlier partial numbers: fresh int8 gold first pass 0.803 (0.855 with the vote), silver first pass 0.568; float32
  gold first pass 0.683 before the budget and loop-salvage changes. The 0.542 / 0.486 gold figures came from int8
  cached output and cannot be reproduced.
- **RTF can reach 2× real time per pass on bad audio** (wall-time cap), and 1.3–2 was seen on 3 of 5 gold clips
  under load.

## 3. Labelled evaluation set

### Sources and license check
| Source | What | License (checked 2026-10-26) | Use |
|---|---|---|---|
| **missileer.watch** archive feed (`https://data.missileer.watch/public/archive.json`) | 238 rows (2026-09-22 → 2026-10-26), Opus clips, anchors, per-character times, 14 listener corrections | **CC0-1.0**: the feed's `license` field plus the site footer: "Our recordings and decodes are released under CC0 1.0 (public domain dedication); attribution appreciated". Covers recordings and decodes | **Seed set.** Ingest with attribution |
| Our own Signal Ops recordings | Multi-receiver, multi-channel | Ours | Main set once capture is reliable |
| NEET INTEL YouTube 24/7 archive and `shortwaves.csv` | Hand-transcribed copy plus stream audio | No license on the repos; YouTube ToS applies | Reference copy only for cross-checking, unless permission is obtained. Don't redistribute audio |
| YouTube / archive.org EAM uploads, Reddit r/shortwave clips | Ad-hoc clips | Per upload: archive.org items often carry CC licenses. YouTube's default Standard License does not allow redistribution | Only items with an explicit CC/PD license, recorded in the manifest. Otherwise ask the uploader |
| eam.watch, warwatchintel | Community logs (text) | No license stated | Coverage and counts cross-check only. No scraping without permission |

**License check process:** each source gets a manifest row with `license`, `license_evidence_url`, `checked_at`, `checked_by` and `attribution`. The ingest code refuses a feed whose license field doesn't match the expected value (`ingest_missileer.py` exits 2 if the license is not `CC0-1.0`). Audio without a verified redistributable license may be used locally for evaluation, but it is never committed. Labels and audio go in a separate data location (`evalset/`, git-ignored, or an LFS/data repo). Only small fixtures are committed.

### missileer.watch feed: what we verified (vs. the scout digest)
- Confirmed: CC0-1.0 license, 238 rows, first row 2026-09-22 01:05 UTC, Opus clips (`clips.valarsystems.com`, ogg/opus, 48 kHz mono, about 21 kb/s), per-character timing (`char_times` = [read-1 times, read-2 times], one entry per character, null where it was not matched), spoken `anchors` (standby, message_follows, i_say_again, completes, out), `receptions[]` with `coreception`, 14 `corrected` rows carrying `machine_text`, and a self-describing `_schema`.
- **Caveats the digest didn't state:**
  - Only 153 rows are `grade: message`. 72 are `unconfirmed`, 10 are radio checks, 2 auth and 1 traffic.
  - 116 rows are `partial`. Only 61 rows have a fully readable `text` (no `?`).
  - All rows are on **11175 kHz**, from receivers `desktop-kiwi-kph`, `-iceland` and `-iceland+kph`. Only 4 messages have a true co-reception.
  - `voice_onset` timing exists only since 2026-10-12 (78 rows). Older rows use `clip_start`.
  - Of the 14 corrections, several are radio checks, and some have text longer than `char_count`. One fixes a rotated machine copy.
  - Most importantly, **uncorrected `text` is missileer's machine ASR, not ground truth.** Only the 14 corrected rows were checked by a listener.
  - The feed covers missileer's own receivers only (eam.watch rows are excluded).

### Gold / silver decision (2026-10-26)
- **gold** = the 14 rows with a listener correction (`corrected: true`): 8 EAMs and 6 radio checks. The radio checks serve as negatives for detection. These labels were checked by missileer's listeners, not yet by us, and the doc treats them as gold by team decision.
- **silver** = the other 224 rows. Their `text` is missileer's machine copy. We use it for structure and detection checks, and as a second CER number that measures *agreement with another ASR*. It is never the headline metric.
- The `silver` CER subset is restricted to complete, fully readable EAMs whose audio has an `i_say_again` anchor (30 clips). `evaluate.load_subset("silver")` applies this filter.

### Eval set as built: `/workspace/hfgcs-data/eval/` (outside the repo, never committed)
Built 2026-10-26 09:50 PDT (16:50 UTC) with:

    python -m decoding.ingest_missileer --build /workspace/hfgcs-data/eval --audio all --max-mb 300

The build refuses to run unless the feed's `license` is `CC0-1.0`. Gold audio is always fetched; silver audio is fetched only up to the byte budget. It writes:

| Path | Content |
|---|---|
| `feed/archive_<generated_at>.json` | Unmodified feed snapshot (feed generated 2026-10-26 16:44 UTC) |
| `labels.jsonl` | 238 label records (`ingest_missileer.to_label`) |
| `manifest.jsonl`, `manifest.csv` | One row per clip: `clip_id, source_id, source_url, feed_url, license, attribution, retrieved_at, status, type, split, sha256, bytes, path` |
| `audio/gold/*.opus` | 14 clips, 6.0 MB |
| `audio/silver/*.opus` | 220 clips, 126 MB (4 silver rows have no clip in the feed) |
| `cache/<asr-config-key>/<id>.json` | Cached ASR outputs written by `decoding.evaluate` |
| `results/*.json` | Per-clip scores for each evaluation run |

**Totals:** 238 rows (gold 14, silver 224) and 234 audio files, 132 MB, well under the 300 MB budget. Split by id hash: 163 dev / 75 test.
All files are Opus in Ogg, 48 kHz mono, about 21 kb/s. License CC0-1.0, "attribution appreciated: missileer.watch".

### Ingestion procedure
1. `--build` (above) for the full set, or `--sample N` for a quick look. Ids are stable (`mw:<missileer id>`) and `labels.jsonl` is rewritten idempotently. Audio that has already been downloaded is not fetched again (each file is written to a `.part` file and then renamed).
2. Each clip gets a deterministic dev/test split by id hash (30% test). Before the set is frozen, re-split by **date and rebroadcast group**, so that airings of one message don't straddle splits.
3. **Ear-check (next):** confirm or correct gold. Add `read1`/`read2`, callsign and addressee, which the feed doesn't have. Promote silver rows to gold as they are checked.
4. Keep `reference_machine` (missileer's copy and `char_times`) as a **competitor baseline**: `evaluate` prints missileer's machine CER on the same subset.
5. Re-ingest weekly, keyed by id. Respect `hidden` (collapsed duplicates). Retracted rows disappear from the feed; the manifest keeps them, flagged.

Field mapping:
- `id→source_id`
- `text→labels.body` (`?`→`.`)
- `preamble→labels.preamble`
- `announced_length→labels.declared_length`
- `grade→type` (message/unconfirmed/fragment→`eam`, radio-check→`radio_check`, …)
- `heard_at/heard_at_basis/clip_start_at/voice_start_s→timing`
- `station→receiver`
- `receptions[].coreception→coreceptions`
- `corrected/correction_source/machine_text→label_status + reference_machine`
- `anchors`/`char_times`/`position_notes→reference_machine`

The feed has no callsign or addressee.

### Label format (`labels.jsonl`, one object per broadcast)
- `id`, `source`, `source_id`, `license`, `attribution`, `audio_url`, `audio_path`, `audio_sha256`, `audio_bytes`
- `heard_at_utc`, `frequency_khz`, `receiver`, `type`
- `label_status` (`silver` | `gold`)
- `labels{body, preamble, declared_length, callsign, addressee, read1, read2}`
- `reference_machine{…}`, `anchors`, `has_both_reads`, `flags`, `split`

Ear-checked labels will add `labelled_by`, `labelled_at`, `ear_notes`, and for Skyking `labels.{codeword, time, authentication}`.
Negative examples (`type: radio_check | traffic | noise | phone_patch`) are included for detection evaluation.

### Metrics (`decoding/metrics.py`, `decoding/evaluate.py`; report per model/config, per receiver, per SNR bin)
- **Type accuracy** (`type_acc`, `evaluate.predicted_type`): the clip's parsed type (EAM with a body, else Skyking, else `radio_check`, else `other`) against the label type; label types without a parser type (`auth`, `traffic`) count as `other`.
- **Body CER** (`evaluate.masked_edit_distance`): edit distance against the gold body divided by its length. A `.` in the label (unheard) matches any character at zero cost. We report micro CER (total edits divided by total reference characters) and the median per clip. Reported for the consensus body, for each read separately, and for missileer's machine copy as a baseline.
- **Field exact-match:** callsign, addressee, preamble, declared length, and the Skyking codeword/time/authentication.
- **Callsign accuracy**: exact match after normalisation, and a top-1 match against the day's callsign list.
- **Detection precision/recall:** broadcast type over all recordings, including negatives. Skyking false positives are tracked separately.
- **Calibration:** a reliability diagram plus ECE of per-character `confidence`, and the share of wrong characters caught in `body_masked`.
- **Rebroadcast linking:** pairwise precision/recall against gold groups.

### Size targets
| Milestone | Gold EAMs | Other | Purpose |
|---|---|---|---|
| v0 (week 1) | 30, from missileer corrected rows plus complete rows, ear-checked | 10 negatives | Pipeline smoke test, first CER number |
| v1 (weeks 3–4) | 150 (at least 3 receivers, both day and night channels) | 50 | Model and VAD choice, calibration |
| v2 (week 8+) | 400+ (enough to train) | 150 | Fine-tuning, with a held-out test of 100 |

Skyking is rare to absent, so its evaluation uses synthetic or TTS readings, flagged `synthetic: true`.

## 4. ALE decoder

All three projects target MIL-STD-188-141 **2G** ALE, which is what HFGCS uses. Priyom: "2G ALE Automatic Message Display (AMD)", with soundings every 3 h, and SIPRNet links "established using 2G ALE". **None of them implements 3G ALE (141B Appendix C) or 4G/WBALE**, so 3G is not a deciding factor.

| | gr-ale (gat3way) | PC-ALE 2.0 (Alex-Pennington) | LinuxALE (DigitalHERMES mirror / SourceForge) |
|---|---|---|---|
| License | GPL-3.0-or-later (headers); README says "based on LinuxALE" | **MIT** (LICENSE file) | GPL-2.0 (LICENSE file) |
| Last commit | 2016-07-14 | **2025-12-23** | 2021-03-09 (mirror) |
| Language / build | C++ GNU Radio OOT module, SWIG (GR 3.7 era; GR ≥3.9 dropped SWIG, so it needs porting) | C++17, CMake, 91 unit tests claimed | C, Makefile, OSS `/dev/dsp` or a raw file (`-f`), 8 kHz |
| Input | float USB audio at 8 kHz, multi-channel example | Library; README lists Audio I/O and SDR integration as "Needed / In progress", so we write the audio front end | Sound card or recorded 16-bit 8 kHz file |
| Features | Sounding and call establishment only (README: "incomplete support") | 141B App A 2G, AQC-ALE, FS-1052 ARQ, LQA, AMD message types in its protocol layer | 2G words/calls, AMD; "limited demodulation capabilities" per its README |
| Headless Linux | Needs a GNU Radio runtime | Yes (library and CLI that we wrap) | Yes, but OSS audio is legacy; file and stdin paths work |

Verified on 2026-10-26 by cloning each repo and checking LICENSE files and headers, the GitHub API `pushed_at`, and the READMEs. We did not build any of them, because the box lacks `make`/`cmake`. Building them is step 1 of the ALE PR.

**Pick: PC-ALE as the primary decoder, with LinuxALE as the reference oracle.**
- **License fit:** MIT is compatible with any license the owner chooses (see `docs/LICENSE_PROPOSAL.md` on the phase0 branch). The GPL decoders could only run as separate processes.
- It is the only maintained project, it has tests, and it covers AMD (the main HFGCS ALE traffic) and LQA.
- **Gap:** no audio I/O yet. We write a small C++ CLI, `pcale-decode`, that reads 8 kHz s16 mono PCM from stdin and prints JSON lines (`{t, freq_khz, word_type, from, to, amd_text, ber}`). It then plugs into the Phase 0 catalog exactly like the voice decoder: an `ale` channel-group recording becomes `pcale-decode`, and the result is written as a `<wav>.ale.json` sidecar.
- **Risk:** it is unproven on off-air signals. Until it matches LinuxALE on the same recordings (run as a separate GPL process, never linked), its output is labelled `experimental`. gr-ale is not recommended: it is unmaintained, and its GR 3.7 / SWIG base doesn't build on current GNU Radio. Its multi-channel GRC example is still a useful reference.
- Other digital signals on HFGCS channels (MIL-STD-188-110A on the SIPRNet frequencies, the 2026-10-14 FSK burst) go to the classifier. It logs spectrogram, exact offset, baud estimate and tone count. Decoding 110A is out of scope for now. milwave-rs has no LICENSE file, so it is watch-only.

## Phased plan
- **P1 (now): parser and eval seeding.** Land `decoding/` (parser, align, metrics, rebroadcast), the tests and this doc. Ingest a v0 sample from missileer and start ear-checking.
- **P2: ASR worker.** `decoding/asr.py` plus a worker that tails the catalog and writes sidecars. Optional dependency `[asr]` = faster-whisper. Eval CLI `python -m decoding.evaluate labels.jsonl` (CER and field metrics, markdown report). Tune VAD, prompt and model on dev.
- **P3: Confidence calibration and fusion.** Multi-airing fusion, second-pass escalation, the review UI hook (low-confidence characters go to a human queue), NEET INTEL CSV export with `.` masks.
- **P4: ALE and the classifier.** The `pcale-decode` wrapper plus a LinuxALE oracle, ALE sidecars, and a voice/ALE/data/carrier/noise classifier (spectral features first, small CNN on spectrograms later).

## Open questions
1. Repo license (blocks copying skywatch code verbatim; ideas are fine). Do we accept GPL helper processes (LinuxALE oracle)?
2. Where do eval audio and labels live (git-ignored `evalset/`, Git LFS, or a separate data repo)? Who ear-checks?
3. Should the decoder run on the capture host (CPU only) or a separate box with a GPU?
4. Should we contact missileer.watch to say we use the feed and to ask for a callsign field or an id-based delta feed?
5. Should `docs/schemas/*.json` be owned by Decoding or by Database & Analytics?
