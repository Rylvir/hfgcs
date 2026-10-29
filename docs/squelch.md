# Squelch: v1, v2 and shadow mode

The recorder keeps a capture while a channel carries a signal: the squelch decides when a recording
starts and stops. There are two squelches. **v1** is the original. **v2** (`hfgcs.dsp.SquelchV2`) fixes
the failures found in the 2026-10-29 audit below. Shadow mode runs both on the same stream: one records,
the other only logs what it would have recorded, so they can be compared on real traffic before
switching.

Defaults: `squelch = "v1"` records and `squelch_shadow = "v2"` logs. Deploying this changes no
recording.

## What was wrong with v1

1. **KiwiSDR channels measured the wrong thing.** kiwirecorder runs with the Kiwi's AGC on (kiwiclient
   v1.11 default: threshold -100 dB, 1 s decay), and v1 measured the audio after it.
   - When the band noise is above the AGC threshold (likely on 4724 kHz at night), speech and noise come
     out equally loud. v1 then sees about 0 dB SNR whatever the signal strength.
   - Every connection also starts about 8 dB quiet while the AGC settles. v1 took the first block as its
     noise floor, so it opened on plain noise and stayed open. Kiwi sessions rotate about every 23 min.
     This is the 4724 chain of 900 s noise captures.
2. **Words with pauses never opened it.** Opening needed 0.8 s *continuously* above 8 dB, and each pause
   between phonetic words reset the count. A message whose continuous intro ("All stations, this is …")
   was lost was never recorded.
3. **The floor was easy to poison.** It was set from the first block. About 0.4 s of digital silence
   (a dropout) dragged it down about 13 dB, enough to open on noise. An S-meter value of `Infinity`
   (Python's JSON parser accepts it) made it NaN for good. That one is fixed in v1 as well.
4. **Static crashes held it open.** Every crash restarted the hang, so after a message it recorded static
   up to the 900 s cap.
5. **The floor froze while open.** After the band noise rose, or under a carrier, v1 recorded to the cap
   and then reopened on the same noise.
6. **The floor's time constant was per block.** It was not set in seconds, so it differed by source.

Drafts #18 and #24 address 5 by raising the floor after 30 s open. In simulation (below) that
fix had three problems:

- On a Kiwi channel whose AGC hides speech, it turns the noise recordings, which happen to contain the
  messages, into missed messages.
- With hiss in a strong message's pauses, it raises the floor and cuts the message.
- #24's close on a 15 s median SNR needs a message to be more than half voiced. A word-by-word reading is
  less, so it cut normal EAMs to about 15 %.

## v2

| Setting (config) | Default | What it does |
|---|---|---|
| `warmup_seconds` | 3 | No decisions until this much valid audio; the first floor is its median. |
| (fixed) | -150 dB | Levels at or below this (digital silence) and non-finite ones never move the floor; while open they count toward the hang. |
| `open_window_seconds` | 3 | Opens once `min_activity_seconds` (0.8 s) above `open_snr_db` fall inside this window, so words with pauses open it. |
| `noise_floor_tau_seconds` | 4 | While closed, the floor follows the band with this time constant (seconds, whatever the block size). |
| `stuck_seconds`, `stuck_flat_db` | 20, 2 dB | A capture whose level has been flat (standard deviation under 2 dB) for 20 s is a carrier or a noise bed, not speech. It closes (`close_reason` "stuck") and the floor is re-learned from those levels. |
| `long_hang_seconds`, `long_hang_after_seconds` | 8, 10 | Once a capture has run 10 s, the hang is 8 s, so a long pause or fade inside a message does not split it. |
| `open_snr_db`, `close_snr_db`, `hang_seconds`, `min_activity_seconds` | 8, 5, 4, 0.8 | As in v1. |

**Levels.** v2 uses the receiver's own level where the source sends one, else a level robust to static:

- **KiwiSDR:** the Kiwi's S-meter (RSSI), measured before its AGC. kiwirecorder runs with `--progress`,
  which prints one `Block: …, RSSI: -97.3` line per audio block on stderr (`[kiwisdr] rssi = true`, the
  default).
  - Each block carries the median of the last 5 readings.
  - Readings outside -170..+30 dBm are ignored.
  - With no new reading for about 2 s the block has no level, and a warning is logged once per session.
- **OpenWebRX:** its S-meter, as before.
- **Local SDR and file replay:** the median of the slot's 10 ms powers over the last 100 ms, so a static
  crash shorter than about 45 ms does not count as signal.

v1 keeps its old levels, so shadow mode compares the old system with the new one.

## Shadow mode

- `squelch` names the squelch that records, `squelch_shadow` the one that only logs (`""` turns it off).
- The shadow file defaults to `<catalog stem>.squelch-shadow.jsonl` beside the catalog (or set
  `shadow_file`). It gets one JSON line per capture the shadow squelch would have made: `algorithm`,
  `live_algorithm`, `level` (`reported` / `slot_median` / `slot`), `frequency_khz`, `receiver_id`,
  `start_utc` (activity start), `end_utc`, `duration_sec`, `close_reason` (`hang`, `stuck`, `cap`, or
  `flush` when the stream ended or was retuned), `snr_db_peak`, `snr_db_mean` and `noise_floor_db`.
  No audio is kept.
- Compare it with what was recorded:

  ```bash
  python -m hfgcs.shadow --catalog data/hfgcs_catalog.jsonl --since 2026-10-30 --list
  ```

  It reports, per receiver and channel, the captures and minutes each squelch made. It also lists
  **live only** captures (recorded now, but v2 would have skipped them: noise, or a miss?) and **shadow
  only** ones (v2 would have recorded them and v1 did not). The decoded EAMs and
  `hfgcs_db.py coverage` against missileer.watch show which side was right. So does listening, for
  live-only captures.
- To switch, set `squelch = "v2"` (and `squelch_shadow = "v1"` to keep comparing) and restart the
  recorders.
- **Limitation:** when one radio scans several channels, a shadow capture is cut at each retune
  (`flush`), because only the live squelch keeps the dwell going. The deployed recorders are single
  channel.

## Simulation (synthetic signals)

Eight seeds per case; 480 s streams with one 150 s EAM (3 s intro, then phonetic words with pauses).
The Kiwi path uses a model of its AGC (the open-source design it is based on, default threshold);
these are not real recordings. Mean share of the message captured / seconds of noise recorded:

| Case | v1, Kiwi audio (today) | #24, Kiwi audio | v2, Kiwi RSSI |
|---|---|---|---|
| Quiet band | 100 % / 318 s | 16 % / 35 s | 100 % / 0 s |
| Noisy band (AGC hides speech) | 100 % / 318 s | 0 % / 35 s | 100 % / 0 s |
| Intro missed | 88 % / 278 s | 0 % / 31 s | 100 % / 0 s |
| Hiss in a strong message's pauses | 100 % / 318 s | 46 % / 35 s | 100 % / 0 s |
| Noise rises 12 dB | 100 % / 318 s | 0 % / 71 s | 100 % / 22 s |
| Dropouts | 100 % / 317 s | 78 % / 195 s | 100 % / 0 s |
| Static crashes | 87 % / 185 s | 55 % / 48 s | 100 % / 3 s |
| Slow reading (0.6-1.2 s gaps) | 100 % / 317 s | 15 % / 35 s | 100 % / 0 s |
| Weak (9 dB), deep fading | 100 % / 318 s | 0 % / 35 s | 94 % / 0 s |

v1 "captures" messages only because it is stuck open recording everything. Without RSSI, v2 on the
Kiwi audio level gets about 80 % in the noisy-band cases: the level fix matters as much as the squelch.

## Considered and not used

- **A speech-rhythm ("syllabic") detector** on the audio. It separated speech from noise, carriers and
  FSK, but static crashes (and the AGC pumping they cause) looked like speech. Worth revisiting with
  real recordings.
- **An ML voice detector (Silero) in the recorder.** It misses weak HF speech (docs/decoding.md) and is
  heavy. It stays at the decode stage, where it already skips clips with no voice.
- **Turning the Kiwi's AGC off** (`kiwirecorder -g`). It would make the audio level usable, but it needs a
  gain per receiver and risks clipping. RSSI is cleaner.
