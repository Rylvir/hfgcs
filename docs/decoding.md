# Decoding & Transcription design

Status: draft v0.1, 2026-10-26 (PDT). Owner: Decoding & Transcription. Changes land as PRs through the
Engineering Lead. This doc covers (1) ASR, (2) the EAM/Skyking grammar parser and its JSON schema,
(3) the labelled evaluation set, and (4) ALE decoding.

Code that exists today (prototype, stdlib only, branch `decoding-design`):

| Path | What |
|---|---|
| `decoding/tokens.py` | ASR word tokens, NATO/ICAO + HF variants (niner, tree, fife) to characters, spoken numbers |
| `decoding/eam_parser.py` | Grammar parser: `parse_transcript`, `parse_eam`, `parse_skyking`, segmentation, test counts |
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
2. It writes one sidecar per recording, `<wav>.decode.json`, containing `{recording: <catalog entry>, asr: {...}, broadcasts: [<parser records>]}`. It also appends a line to `decodes.jsonl`. The Database & Analytics lead ingests both into `transcripts` / `messages` / `message_airings`.
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

### VAD (silero-vad via faster-whisper `vad_filter=True`)
faster-whisper's `VadOptions` defaults (v1.2.1) are tuned for clean speech. HF voice needs these changes:
- `threshold` **0.35** (default 0.5). Weak characters under noise score below 0.5 and get dropped.
- `min_silence_duration_ms` **1500**. Operators pause between groups and before "I say again". Splitting there breaks the context.
- `speech_pad_ms` 400 and `min_speech_duration_ms` 200, to keep single short words ("Two") inside a segment.
- Squelch tails, carrier drops and heterodynes give false VAD triggers. These are handled by the no-speech/hallucination guards below, not by VAD.
- Tune `threshold` on the dev split over the values {0.25, 0.35, 0.5}. The metric is body CER plus the rate of characters dropped.

### Model and compute
| Setting | CPU box (no GPU, 8 cores here) | GPU |
|---|---|---|
| Model | `small.en`, int8 (default); `medium.en`, int8 for second-pass review | `large-v3` or `distil-large-v3`, float16 |
| Approx. speed | small.en int8 runs near real time on 8 cores (to be measured on our clips) | far faster than real time |

Route low-confidence messages to a second pass with a larger model (two-pass escalation; the idea comes from shnikees/netcontroller, which is GPL, so we take the idea only). English-only `.en` models avoid language-detection errors on noise.

### Decoding settings
- `language="en"`, `beam_size=5`, `word_timestamps=True`.
- `condition_on_previous_text=False`. With it on, a 30-character read degenerates into repetition loops.
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

Next experiments, in order:
1. `medium.en` and `distil-large-v3`. This needs more than the roughly 2 GB of RAM currently free on this box, or a GPU.
2. VAD threshold sweep.
3. Prompt without punctuation.
4. A CTC forced-alignment / closed-vocabulary rescoring pass over the phonetic lexicon.

### Reuse from brisi-jt/skywatch (verified: MIT, `LICENSE` © 2026 James Towns, last push 2026-10-21)
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
broadcast   := eam | skyking | test_count
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
Segmentation: a new broadcast starts at "all stations" or "skyking" only after the previous one has closed ("out" or "more to follow"), so the Skyking repeat after "I say again" is not split off. Every deviation is a **flag**, not an exception. These include a missing header (`partial_start`), `single_read`, `missing_preamble`, `preamble_reads_disagree`, `reads_disagree`, `length_mismatch`, `declared_length_conflict`, `preamble_body_mismatch`, `non_character_words_in_body`, `closing_callsign_mismatch`, `invalid_time`, and others.

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
- Body CER ≤ 0.2 between them.
- They are within 24 h of each other.

Within a group, each airing gets a relation:
- `first` for the earliest.
- `simulcast` when it is within 30 s of the first (same emission heard on another frequency or receiver).
- `rebroadcast` otherwise.

The group id is `eam-<first UTC>-<first 6 chars>`. NEET INTEL reports that about 1 in 3 EAMs is rebroadcast, mostly within 2 h. Linking by preamble alone fails, because messages that share a preamble exist (the scout digest's 56HETY ×9). That is why we compare full bodies.

### Output JSON schema (v0.1; one record per broadcast)
```jsonc
{
  "type": "eam" | "skyking" | "test_count" | "unknown",
  "schema_version": "0.1",
  // EAM
  "callsign": "MAINSAIL", "callsign_confidence": 0.93, "addressee": "RED DOG" | null,
  "preamble": "QKR4TZ", "preamble_reads": ["QKR4TZ", "QKR4TZ", "QKR4TZ"],
  "declared_length": 30 | null, "length_source": "announced" | "default", "expected_length": 30,
  "read1": "…", "read2": "…", "body": "…", "body_masked": "QKR4TZ7H.B2…", "body_length": 30,
  "chars": [{"char": "Q", "confidence": 0.998, "read1": "Q", "read2": "Q", "agree": true, "preamble": "Q"}],
  "confidence": {"mean": 0.97, "min": 0.41, "low_count": 1, "reads_agree_ratio": 0.967},
  "timing": {"read1": {"start_sec": 74.1, "end_sec": 140.2}, "read2": {"start_sec": 145.0, "end_sec": 211.7}},
  "closing_callsign": "MAINSAIL", "out": true, "more_to_follow": false, "disregard": false,
  // Skyking: each field = {"value", "read1", "read2", "agree", "confidence"}
  "codeword": {...}, "time": {...}, "authentication": {...},
  "flags": ["reads_disagree"],
  "segment": {"token_start": 0, "token_end": 212, "start_sec": 31.0, "end_sec": 215.3}
}
```
The sidecar wraps these records with the recording entry and ASR provenance (`model`, `compute_type`, prompt hash, VAD parameters). We will publish a formal JSON Schema file (`docs/schemas/broadcast.schema.json`) once v0.1 stabilises.

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
