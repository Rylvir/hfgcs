# Fine-tuning `small.en` for HFGCS voice on CPU: plan (not started)

Status: plan only. No training has been run. The figures below come from the eval in `docs/decoding.md` §1
(2026-10-26). Time estimates are back-of-envelope and must be checked with a 20-step benchmark (step 1 of the
schedule) before any long run.

## 1. Why, and what "better" has to mean
- The current best (`AsrConfig()` + snap) reaches gold body CER **0.542**, with 0.486 using the gated vote.
  missileer's machine copy scores **0.418**. The errors are mainly mangled phonetic words ("Dory", "Koopalink"),
  dropped onset letters (the Q/V of QH…/VQ… preambles), and hallucination loops. These are acoustic and
  vocabulary errors that prompting cannot fix, but a fine-tune can.
- **Target:** gold body CER at or below 0.40 (beating missileer), with no loss in detection (1.0) and no false EAMs
  on negatives. A gain counts only when it is significant on held-out gold (§5) and silver does not get worse.

## 2. Data
| set | size now | labels | use |
|---|---|---|---|
| silver EAMs (missileer machine copy, CC0) | 217 EAM clips, 220 clips in total, **11.5 h** audio (avg 188 s) | 47 fully readable; 170 contain `.` (unheard) | training (filtered and masked, §3) |
| gold (listener-corrected) | 14 clips (8 EAMs, 6 radio checks) | ear-verified | **test only**, never trained on |
| synthetic | unlimited | exact | training: TTS EAM readings through the HF channel simulator (§2.2) |

### 2.1 Training windows
- Whisper trains on windows of 30 s or less. The procedure is cut into windows at word or character boundaries
  using missileer `char_times` (per character, per reading) and `anchors` (standby / i_say_again / out).
  - Where `char_times` are null, our own first-pass word timestamps are used, but only when our transcript and the
    label agree on the anchor words.
- Expect about 6 windows per clip, so roughly 1,300 windows from silver.
- Targets are written in the same style the parser expects: Title-Case NATO words, "Niner", "X-ray", "stand by",
  "Message follows", "I say again".

### 2.2 Augmentation (offline, cached; each recipe is seeded and logged in the manifest)
| effect | implementation | range |
|---|---|---|
| HF noise | additive band-limited pink/white noise plus impulsive static (lightning crashes) | SNR 0–20 dB |
| Fading (QSB) | Watterson-style two-path Rayleigh multiplier | 0.1–2 Hz fade rate, 0.5–2 ms delay spread |
| SSB mistuning / frequency offset | single-sideband frequency shift (Hilbert transform) | ±50 to ±300 Hz, so voices sound "Donald Duck" |
| QRM | an adjacent-channel voice, RTTY or carrier mixed in | −15 to −3 dB relative |
| Band-limiting | 300–2400 Hz up to 150–3000 Hz bandpass; 8 kHz resample; Opus 16–24 kbit/s round trip | random per sample |
| AGC / level | dynaudnorm on or off, ±12 dB gain, clipping | |

- Each silver window gets 4 augmented variants plus the clean original, giving about 6,500 samples.
- Synthetic: about 2,000 windows (a few TTS voices × random 30-character bodies, preambles and callsigns), passed
  through the same chain. Synthetic data teaches the closed vocabulary with **exact** labels, which offsets the
  silver-label risk (§3).

## 3. Risk: learning missileer's machine errors from silver
- Silver labels are missileer's own ASR output. On the 8 gold EAMs that copy has CER 0.418 against the corrected
  text. Those rows were corrected *because* they were wrong, so 0.418 overstates the typical error, but silver is
  certainly not ground truth. Training on it directly would teach us their mistakes, and their pipeline possibly
  shares our model family's biases.
- Mitigations, in order:
  1. **Mask, don't guess:** characters that are `.` in the label get the loss ignored (`ignore_index` on their
     target tokens). Unheard characters are never invented as targets.
  2. **Agreement filter:** keep a body character as a target only where missileer's two readings agree. Disagreeing
     positions are masked like `.`. Procedure words (anchors) are kept, because they are low-risk.
  3. **Cross-check filter:** drop a window when our first pass and the label disagree by more than CER 0.6. At that
     point either the label or the audio is unusable. This filter is biased towards easy audio, so it is used only
     for the first epoch.
  4. **Mix in synthetic data** with exact labels (about 25–30 % of samples) and every gold-free ear-verified label we
     get.
  5. **Watch for imitation:** track agreement with silver labels *and* gold CER separately. Silver agreement rising
     while gold CER stays flat is the signature of learning their errors.
- **Selection bias:** silver over-represents messages missileer could already decode (only 47 of 217 are fully
  readable), so a fine-tune may help most where it is least needed. The synthetic set and low-SNR augmentation
  cover the hard tail.

## 4. LoRA vs. full fine-tune on this box (8 cores, 16 GB total, about 3–7 GB free)
`small.en` has 244 M parameters, about 0.97 GB in fp32. CPU training is fp32 (bf16 autocast only if the CPU
supports AVX-512-BF16).

| option | trainable | peak RAM (estimate) | time per 30 s sample (fwd+bwd, 8 threads, estimate) | full run (estimate) | verdict |
|---|---|---|---|---|---|
| **A. Decoder-only LoRA over cached encoder outputs** (r=16 on q/k/v/out of the decoder self- and cross-attention) | about 2 M | about 2–2.5 GB | encoder pass once per sample (about 1.5–3 s); training step about 0.3–0.6 s | caching about 6,500 × 2 s ≈ **3.5 h** once; then about 45 min per epoch, **3 epochs ≈ 2.5 h** | **start here** |
| B. Encoder + decoder LoRA (r=8–16), augmentation on the fly | about 3–5 M | about 3–4 GB (gradient checkpointing, batch 1 × grad-accum 16) | about 5–8 s | about 6,500 × 6.5 s ≈ **12 h per epoch**, 2 epochs ≈ 24 h | if A plateaus: fixes acoustic errors such as onsets and QSB that A cannot |
| C. Full fine-tune with AdamW | 244 M | weights + grads + 2 Adam moments ≈ 4 GB, plus activations 1.5–2 GB ≈ **5.5–6+ GB** | about 7–10 s | 24 h or more per epoch | **not on this box:** at or over the free-RAM ceiling and swap-prone; it also overfits about 10 h of data more easily. Use a GPU (hours, not days) |

- Encoder caching for A: each encoder output is 1500 × 768 values in fp16, about 2.3 MB, so 6,500 samples need about
  15 GB of disk. The alternative is to keep two augmented variants per window (about 7 GB).
- Deployment: merge LoRA into the weights and convert with `ct2-transformers-converter` to CTranslate2 int8. Our
  inference path (`faster-whisper`) stays unchanged, with the same RTF of about 0.13–0.2.
- Stack: `torch` CPU wheel, `transformers`, `peft`, `datasets` (all Apache-2.0/BSD), in their own venv
  (about 1.5 GB of disk). Training runs at `nice` priority with `OMP_NUM_THREADS=6`, so capture and other agents keep
  2 cores. A checkpoint is saved every 200 steps so an interruption loses little.

## 5. Held-out protocol
1. **Gold is test only.** No gold clip, and no rebroadcast or simulcast copy of a gold message, enters training.
   Copies are found with `similarity.text_similarity ≥ 0.8` within 48 h, which is the same rule the DB uses.
2. **Split by UTC day and receiver, not by clip.** Whole days (and, for a second check, a whole receiver such as
   `desktop-kiwi-iceland+kph`) are held out, so the model cannot learn a day's callsign or a receiver's noise
   profile. A content group spanning days goes entirely to one side.
3. **Validation for early stopping and hyperparameters:** a silver validation slice of held-out days, plus a
   small ear-verified *validation* gold set kept separate from the gold *test* set (§6). The test set is scored once
   per finished model.
4. **Metrics:** gold body CER (micro, plus per-clip paired), detection, false EAMs on the radio-check negatives,
   preamble exact-match, and RTF. Report each against `best` on the same clips with a paired bootstrap 95 % CI and
   a Wilcoxon signed-rank test.
5. **Regression guard:** also score held-out silver against labels. A model that improves gold but loses more than
   0.05 on silver, or vice versa, is not adopted without listening checks.

## 6. How many gold labels (power analysis from the current per-clip CER)
Per-EAM body CER for `best` (`eval/results/gold_final.json`, `silver_final.json`):
- gold: mean 0.549, SD 0.31 (0.25 with CER capped at 1), n = 8.
- silver: SD 0.35 capped.
- **Paired** differences between two decent configs on the same clips: SD about 0.18–0.23 on gold
  (`best` vs `no_vad_snap` 0.18, vs `no_vad` 0.23). Silver's paired SD (0.57) is inflated by multi-message clips
  with CER above 1.

A paired comparison (base vs fine-tuned on the same EAMs), two-sided α = 0.05, power 0.8:
n = ((1.96 + 0.84) · σ_d / δ)².

| true CER gain δ | σ_d = 0.20 | σ_d = 0.25 | σ_d = 0.30 |
|---|---|---|---|
| 0.10 (for example 0.54 → 0.44) | **32** | 50 | 71 |
| 0.05 | **126** | 197 | 283 |
| 0.03 | 349 | 546 | 785 |

- **To measure** a 10-point gain we need about **35–50 gold EAMs** in the test set. A 5-point gain needs about
  **130–200**. The current 8 detect only gains of about 0.2 or more (n = 8 → minimum detectable δ ≈ 2.8 × 0.2 / √8
  ≈ 0.20).
- Characters within a message are correlated, so the unit is the message, not the character. The CER-capped SD is
  used for planning, and the bootstrap handles the heavy tail.
- **To validate** (early stopping and hyperparameters), a separate **30–50** gold EAMs, so the test set is not tuned on.
- **To train on gold:** at least **300** ear-verified EAMs (about 15 h of audio), as earlier in §1 "Fine-tuning".
  Below that, gold is worth more as test and validation than as training data. Gold supplements silver plus
  synthetic data; it does not replace them.
- **Labelling cost:** about 5 min of listening per 3-minute clip, so 130 test + 40 validation EAMs ≈ **14 h** of
  listener time. Priority order: clips where our first pass and missileer disagree most; the gated second pass's
  low-agreement clips; then a random sample for an unbiased CER.

## 7. Schedule (each step is cache-backed and resumable)
1. Benchmark: 20 training steps of A and B on 10 windows, to measure s/step and peak RSS. Replace the estimates in
   §4.
2. Build windows, masks and augmentation into `/workspace/hfgcs-data/ft/` with a manifest (source, recipe, seed,
   sha256). Never commit audio.
3. Option A, 3 epochs. Evaluate on validation each epoch, then once on gold test.
4. If A gains less than 0.05: option B overnight, with the same protocol.
5. Convert the best checkpoint to CTranslate2 int8. Add it as an `AsrConfig(model=<path>)` row in the decoding.md
   table.

Prerequisites before step 3: at least 35 gold test EAMs and 30 validation EAMs (§6). Without them, a result
cannot be told apart from noise.
