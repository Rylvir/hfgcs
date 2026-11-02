# Skyking hallucination experiment (2026-11-02)

**Branch:** `decoding-skyking-exp` (from `origin/main` @ `df008ac`).  
**Scope:** experiment and report only — no merge, push, or deploy. Box only, `nice 19`.

## What is on main vs. the earlier anti-hallucination work

`git log origin/main -- decoding/` shows only through **Decoding 0.3.3 (#19)**. These local commits from `/workspace/hfgcs-decoding-fixes` are **not** on main:

| SHA | Summary |
|---|---|
| e48b25e | counts as radio checks after a Skyking head; split before `all stations` |
| 13a3fc6 | suppress codeword-less Skykings / empty EAMs; no fragment callsigns |
| 8a2b912 | drop no-voice prompt echoes; per-chunk wall-time; no repair of whole noise windows |
| a90c95c (=0.3.7) | suppress Skykings with neither time nor auth, or codeword > 3 words |
| 0e04cd0, 784b9c6 | loop salvage / EAM-vs-Skyking message words (also not on main; PR #19 stopped at 0.3.3) |

Live keyword alerts are **#27** (`hfgcs.keyword_alerts`). The old `hfgcs-skyking-ntfy` was stopped 2026-11-01 23:50 PT.

## Ground truth used

| Clip | Label | Notes |
|---|---|---|
| 04 15016 2026-10-29T1535Z | **POS** Bravo Force | User: real (likely Global Thunder). |
| 05 11175 2026-10-30T1200Z | **POS?** Hellstorm | User: probably real; ambiguous. See findings. |
| 06 15016 2026-10-27T0037Z | **NEG** Monogram | Daily callsign standing by; decoder prefixed Skyking. |
| 4724 2026-10-31T0235Z | **NEG** | Old alerter post. |
| 11175 2026-11-02T0218Z | **NEG** West Coast test count | Old alerter post. |
| 15016 2026-11-02T0643Z | **NEG** | Old alerter post (WAV: `…T060354Z…`). |
| 4724 2026-11-02T0722Z | **NEG** | Old alerter post; **WAV not on disk** (no 4724 capture that morning after 01:07Z). |
| memclip 4724 2026-10-26T223059Z | **NEG** | 900 s noise / PR #19 regression clip. |
| 01, 02, 03 | Unjudged | Reported only; not scored. |

**Fifth false alert:** the old unit’s journal has **exactly four** `posted skyking` lines (one duplicated). There is no fifth in `hfgcs-skyking-ntfy` since 2026-10-26. Monogram + memclip are extra negatives for this study, not journal posts.

Hilltop live sidecars for Bravo/Hellstorm still say `decoding/0.2.0` + int8 key `5bfe156c6443` (decoded before 0.3.3).

## Variants (ASR)

| ID | Changes vs live 0.3.3 defaults |
|---|---|
| **v0_live** | unchanged (`prompt=eam` includes “Skyking, Skyking, do not answer.”, hotwords include Skyking, `condition_on_previous_text=True`) |
| **v1** | `prompt=phonetic`, `hotwords=False`, `condition_on_previous_text=False` |
| **v2** | `prompt=None`, `hotwords=False`, `condition_on_previous_text=False` |
| **v3** | live prompt, but `hotwords=False` |
| **v4** | `prompt=phonetic`, `hotwords=False`, cond still on |

Post-filters on every variant (pure Python, `decoding/hallucination.py`):

- **raw** — parser only  
- **sup** — suppress weak Skykings (no/long codeword, missing time+auth)  
- **seg_sup** — drop high `no_speech_prob` + low `avg_logprob` segments, then suppress  

Clips were ~40–260 s windows cut from the full WAVs (paths under `/workspace/skyking-exp/`).

## Before / after table (raw Skyking emissions)

“Phantom” = `type==skyking` on a **NEG** clip. RTF = wall / audio on the box at 4 threads, nice 19.

| Variant | Phantoms (raw) | Phantoms after `seg_sup` | Bravo Force text kept? | Hellstorm text kept? | Median RTF |
|---|---|---|---|---|---|
| **v0_live** (current seeding) | **4 / 5** | **0 / 5** | yes (as EAM) | yes | **1.02** |
| v1 phonetic + no cond + no hot | 0 / 5 | 0 / 5 | yes (EAM) | yes | 0.18 |
| **v2 no prompt + no cond + no hot** | **0 / 5** | 0 / 5 | yes (EAM) | yes | **0.12** |
| v3 live prompt, no hotwords | 0 / 5 | 0 / 5 | yes (EAM) | yes | 0.18 |
| v4 phonetic, cond on, no hot | 0 / 5 | 0 / 5 | yes (EAM) | yes | 0.20 |

Per-clip raw Skyking under **v0_live**: neg_4724_1031, neg_westcoast, neg_15016_0643, neg_memclip (all `cw=None conf=0.0`); pos_hellstorm also got a conf-0 Skyking (“Skyking driving…”); pos_bravo did **not** emit `type=skyking` on 0.3.3 float32 (message body only).  
Under **v2**, every NEG is `other`/`eam` with no Skyking; Bravo still contains “Bravo Force…”; Hellstorm is “stand by for traffic”.

### Hellstorm (user: probably real)

Live int8 sidecar: `skyking @ 501.78` codeword **DRIVING**, transcript `Skyking driving. This is Hellstorm. Out.` — and an earlier EAM already has `This is Hellstorm. Hellstorm. Stand by for traffic.`  
On every no-hotword / no-prompt variant the clip is **routine traffic**, not a Skyking. Recommendation: treat as **NEG / routine** unless the user confirms otherwise.

### Bravo Force (user: real)

Live int8: `skyking @ 45.94` with a **long** “codeword” (whole body) and missing time/auth — would fail the proposed gate.  
0.3.3 float32 (all variants): message kept as **EAM** with “Bravo Force” / phonetic groups intact; **not** `type=skyking`. A Skyking-only alert would miss it; a content keyword (or exercise list) would still catch it.

## Production defaults (decoding 0.3.4)

Shipped after Monitoring approval. `AsrConfig()` / `DecodeConfig()` / `decoding.worker` now use:

```text
prompt = None
condition_on_previous_text = False
hotwords = True   # NATO + Mainsail + day callsigns; Skyking is omitted when prompt is None
```

`Decoder.decode` always runs `filter_records(suppress=True)` before writing the document.
`hfgcs.keyword_alerts` applies the same suppress for older sidecars, runs `skyking_alert_ok`,
skips routine traffic, and **never ntfy-alerts on `type==eam`** (log only until re-enabled).

Tradeoff vs the old eam-prompt seed (experiment table above): ~0.13 worse gold CER on a
6-EAM slice, ~3.5× faster, no conf-0 Skyking phantoms on the false-alert set.

## Proposed alert rule (fits #27)

Replace the lone `skyking` word match with the gate in `decoding.hallucination.skyking_alert_ok`, already wired into `hfgcs.keyword_alerts.find_matches` on this branch:

1. `type == "skyking"` (after suppress)  
2. Short codeword present (≤ 3 words) **or** transcript has non-preamble content when the document has no `codeword` field  
3. Not missing both time and authentication when those flags exist  
4. `confidence ≥ 0.25` **or** vote agreement **or** cross-channel corroboration (`text_similarity` within ~2 min on another frequency)  
5. Not routine traffic (`standing by for traffic` / radio check / test count / …)  

Unconfirmed (low conf, single channel): **log only**, do not ntfy.

### Routine-traffic rule

`is_routine_traffic`: callsign + “standing by for traffic” (and ASR garbles like “get in by the traffic”), radio check, test count, how copy — **log, never alert**. Covers Monogram and (likely) Hellstorm.

## Gold CER (6 gold EAMs, first pass + snap, box, 4 threads, nice 19)

Fresh uncached decodes under each ASR seeding. Same clips:
`eam-86d80b13`, `eam-f27e2d58`, `eam-7ca8cd24`, `eam-356f1a01`, `eam-3d5248c9`, `eam-ed75798a`.

| Config | body CER (micro) | CER median | detected | type_acc | RTF |
|---|---|---|---|---|---|
| **v0 live** (eam prompt + Skyking hotwords + cond) | **0.516** | 0.667 | 0.83 | 0.83 | **0.75** |
| phonetic_only + hotwords (no Skyking) + no cond | 0.626 | 0.567 | 1.00 | 1.00 | 0.63 |
| **noprompt + no hotwords + no cond** | 0.648 | 0.567 | 1.00 | 1.00 | **0.21** |

Reference: earlier fresh **int8** full gold first-pass was ~0.803 (14 clips). This 6-EAM float32 live slice is better on CER than that int8 run, but **noprompt costs ~0.13 CER** on this slice while cutting RTF ~3.5× vs live and wiping phantoms on the alert set.

Artifacts: `/workspace/skyking-exp/gold/{live,noprompt,phonetic_only}_summary.json`.

## Files on this branch

- `decoding/hallucination.py` — suppress, segment filter, alert gate, routine detector  
- `decoding/asr.py` — `phonetic_only` prompt; hotwords omit Skyking for `phonetic_only` / `None`  
- `hfgcs/keyword_alerts.py` — Skyking keyword goes through `skyking_alert_ok`; routine skipped  
- `tests/test_hallucination.py`  
- Artifacts (not committed): `/workspace/skyking-exp/{clips,results,gold,wavs}`

## Open questions for the user

1. Confirm Hellstorm: real Skyking or standing-by (we lean standing-by)?  
2. Should “Bravo Force” / other exercise names be a separate keyword while Skyking stays strict?  
3. Prefer `phonetic_only` (keeps alphabet bias) or full `prompt=None`?  
4. OK to land suppress + alert gate without waiting for the full 0.3.7 ASR wall-time/voice-gate stack?

## Deploy plan (hilltop)

Restart **only** the decode worker and the keyword alerter. Do **not** restart recorders,
OpenWebRX, or the review app.

1. Deploy this SHA to `~/hfgcs-app` as usual (`hfgcs_deploy` / docs/deploy.md). Confirm the
   unit mapper schedules **`hfgcs-decode`** and **`hfgcs-keyword-ntfy`** only (paths under
   `decoding/` and `hfgcs/keyword_alerts.py`).
2. `systemctl --user restart hfgcs-decode hfgcs-keyword-ntfy`
3. Check: `journalctl --user -u hfgcs-decode -u hfgcs-keyword-ntfy -n 50` — decoder_version
   starts with `decoding/0.3.4;` and the asr key differs from 0.3.3.
4. Spot-check: a known noise/false-alert clip should not ntfy; a real Skyking with a short
   codeword still can.

**Rollback:** redeploy the previous SHA (0.3.3 / df008ac-era decode) and
`systemctl --user restart hfgcs-decode hfgcs-keyword-ntfy`. Recorders are untouched either way.
A new `decoder_version` re-queues clips once; rolling back re-queues again for that version.

