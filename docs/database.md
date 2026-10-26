# HFGCS message database: design proposal

Status: **proposal v1** from Database & Analytics, 2026-10-26, with the chief of staff's design decisions applied (§11). It is committed locally on branch `db/schema-v1` and has not been pushed. The Engineering Lead decides what lands.
Files: `schema.sql`, `scripts/hfgcs_db.py` (stdlib-only CLI), `datasette/metadata.json`, `tests/test_hfgcs_db.py` and `tests/fixtures/*` (all fake data).

## 1. What the DB has to ingest today
In the repo (commit `55fe420`), `hfgcs_scanner.py` writes `recordings/HFGCS_<kHz>kHz_<YYYYmmdd_HHMMSS>.wav` (8 kHz mono). For each file it appends one line to `hfgcs_catalog.jsonl`:
`{timestamp, frequency_khz, file, duration_sec, noise_floor_db, notes}`.
- `timestamp` is taken *after* the recording, so it marks the **end** of the capture. On ingest, start = timestamp − duration.
- The scanner records no SNR, receiver ID, transcript or parsed fields yet. The schema has columns ready for them.
- The repo has no DB, ASR or parser yet. The repo has 0 issues and 0 PRs, so there is no prior DB discussion to follow.
- Other work in progress lives in other branches: `hfgcs/` (Phase 0) and `decoding/` (parser, ASR, `ingest_missileer.py` for ASR labels). **Rebroadcast linking is owned by this DB tool.** Decoding contributes only an optional text-similarity function (§3.1).

## 2. Schema overview (`schema.sql`, SQLite ≥ 3.34; tested on 3.46.1)
| Table | Purpose |
|---|---|
| `receivers` | Every audio or log source: local HF+, OpenWebRX, KiwiSDR, livestream, manual. Fields: code, kind, URL, lat/lon, grid, antenna. |
| `captures` | One audio segment. Fields: receiver, `frequency_khz`, `start_utc`/`end_utc`, duration, `snr_db`, noise floor, `audio_path`/`audio_url`, sha256, `voice_start_s` (+ machine value and method), classification, external id, raw JSON. |
| `messages` | One parsed message per reception, linked by FK to `captures`. NULL capture for imported logs. Fields: `heard_utc` + `heard_at_basis`, `msg_type` (FK → `message_types`), `broadcast_type`, callsign, prefix, preamble, `message_text`, `length` (logged), `text_length` (generated), `announced_length`, addressee, `group_no`, NEET `qualifier`/`specification`/`e6_tail`, Skyking codeword/time/auth, `confidence` + per-char JSON, missileer reception fields, review status, `cluster_id`, origin/provenance, `import_key`. |
| `message_types` | Type taxonomy (see §6). |
| `message_links` | Pairwise same-content links: `simulcast` (same emission on another receiver or frequency), `rebroadcast`, `partial`, `manual`. Each link stores `delta_sec`, similarity, method and `confirmed`. |
| `message_clusters` | One row per distinct content. Fields: consensus text, first/last heard, `n_messages`, `n_airings` (simulcast copies collapsed). |
| `prefix_groups` | Prefix → G1..G5 with `valid_from`/`valid_to`, because prefixes get reused across years. |
| `messages_fts` | External-content FTS5 table with the **trigram** tokenizer, so substrings of the 30+ character strings match (`MATCH 'QWFO'`, minimum 3 characters). Insert, update and delete triggers keep it in sync; the tests run `integrity-check`. |
| `settings` | Tunables, such as the dedup thresholds (§3). |
| `external_sources`, `import_runs` | Provenance, licence and attribution for each source; one log row per import, with ETag and the feed's `generated_at`. |
| `external_daily_counts` | Independent daily EAM counts, **entered by hand from a CSV**, for coverage checks. |

**Views:**
- `v_messages`: joined rows, plus yymmdd/hour/group resolution
- `v_daily_counts`: `eams` (distinct messages, the default), `eam_broadcasts`, `eam_receptions`, skykings, non-30 EAMs, prefixes
- `v_hourly_heatmap`: weekday × UTC hour (`eams` distinct, plus receptions)
- `v_length_by_month`
- `v_prefix_activity`
- `v_rebroadcasts`
- `v_receiver_coverage`
- `v_review_queue`
- `v_neet_export`: the exact NEET column order and format
- `v_our_daily_eams`: `eams` (distinct), `eam_broadcasts` and `eam_receptions` per day
- `v_coverage_comparison`

**Conventions:**
- Timestamps are UTC ISO text (`YYYY-MM-DDTHH:MM:SS[.sss]Z`), enforced by CHECK constraints.
- Frequencies are dial frequencies in kHz.
- Transcription glyphs are kept as typed: `.` unreadable, `?` unknown, `_` no body, `%` no-traffic sentinel.
- FKs only enforce with `PRAGMA foreign_keys=ON`, which must be set on every connection. The CLI sets it.
- There is no migration tool yet. `CREATE … IF NOT EXISTS` will not add columns to an existing DB. Once real data exists, add numbered migrations keyed on `schema_version`.

## 3. Dedup / rebroadcast algorithm (`hfgcs_db.py link`)
1. **Candidates:** EAM and Skyking rows that are not rejected and have a real body (not `_` or `%`).
2. **Normalize:** `text_norm` = upper-case with spaces, tabs and hyphens removed. `.` and `?` are wildcards.
3. **Blocking:** compare only rows of the same normalized length, within the rebroadcast window. Prefixes must agree when both are known.
4. **Similarity:** `decoding.similarity.text_similarity` if it can be imported (§3.1); otherwise the built-in `builtin:wildcard_hamming:v1`.
   - The built-in computes matches ÷ positions where both characters are known.
   - It needs at least `max(6, min_known × length)` known positions, or the pair is not compared.
   - A link is made when similarity ≥ `threshold`.
5. **Kind:** Δt ≤ `simulcast_sec` is a `simulcast` (the same emission on another receiver or frequency). A later copy within `window_hours` is a `rebroadcast`.
6. **Clusters:** union-find over all links, then per-position majority vote for `canonical_text`. `n_airings` counts simulcast-connected components.
7. **Reruns:** every similarity is computed before anything is written, so a failing similarity function leaves the DB unchanged. Only our own unconfirmed links (`builtin:*`, `decoding:*`) are rebuilt. Human-confirmed links and source-asserted links (`external:missileer`) are kept.

**Settings.** Precedence is CLI flag > `settings` table > built-in default.

| Key | Default | CLI flag |
|---|---|---|
| `link.simulcast_sec` | 120 | `--simulcast-sec` |
| `link.window_hours` | 48 | `--window-hours` |
| `link.threshold` | 0.90 | `--threshold` |
| `link.min_known` | 0.5 | `--min-known` |
| `link.similarity` | `auto` | `--similarity auto\|builtin\|decoding` |

- View settings with `hfgcs_db.py settings`; change one with `hfgcs_db.py settings link.threshold 0.92`. They are also visible in Datasette (`link_settings`).
- **These defaults are provisional.** They will be tuned against missileer.watch data (co-receptions, rebroadcast links and listener corrections are ground truth) once Decoding has an evaluation set. Each change will be recorded here with its precision and recall.

### 3.1 Interface expected from Decoding
```python
# decoding/similarity.py
def text_similarity(a: str, b: str) -> float | None: ...
```
- **Inputs:** two normalized copies (upper-case; no spaces or hyphens). `.` means unreadable and `?` means unknown; both are wildcards. Lengths may differ, although v1 only passes equal lengths.
- **Returns:** a `float` in **[0.0, 1.0]**, where 1.0 means the same content. Return `None` when the pair cannot be compared (for example, too few known characters).
  - The DB tool applies the threshold itself.
  - A value outside [0, 1] raises an error, and the run aborts without writing anything.
- **Contract:** pure and deterministic, with no I/O or global state, and fast (it is called on the order of 10⁴–10⁵ times per run). Importing it must not pull in heavy dependencies such as the ASR stack.
- **Selection:** with `--similarity auto` (the default), the tool uses this function when it is importable from the repo root and otherwise falls back to the built-in. `decoding` makes the function mandatory; `builtin` never imports it. Links record which method made them: `decoding:text_similarity` or `builtin:wildcard_hamming:v1`.

**Validation on NEET INTEL's 1,913-row 2023 sample** (local clone; not committed):
- These figures come from the built-in at the default settings.
- 1,904 EAMs fell into 1,241 distinct contents. 336 contents were heard more than once (27 %, close to NEET's "about 1 in 3"), averaging 2.9 airings each.
- 96 % of chains span 2 h or less; 97.9 % of rebroadcast links are within 2 h.
- Of 147 rows with NEET's `&` qualifier, the algorithm placed 142 as later airings. That supports reading `&` as "repeat".

**Not handled in v1:**
- Truncated copies of different lengths (planned: a `partial` link via prefix alignment).
- Re-reads within one transmission ("I say again"). The parser should handle those before rows reach the DB.

## 4. NEET INTEL CSV mapping (import/export, exact format)
NEET INTEL's columns are `DAY,DATE,UTC,CALLSIGN,PR,MESSAGE,Q,LN,SPECIFICATION,E6,NOTES,SOURCE,SET`. The source is their `EAM_SOFTWARE_SUITE/shortwaves.csv` plus the readme. The files are UTF-8, LF line endings, minimal quoting, and sorted by date and time.

| NEET | DB | Notes |
|---|---|---|
| DAY | derived from `heard_utc` | Checked on import (warning on mismatch); recomputed on export |
| DATE `YYYY.MM.DD` + UTC `HH:MM` | `heard_utc` (`…:00Z`), `heard_at_basis='log_minute'` | Minute precision only |
| CALLSIGN | `callsign` | Daily rotating callsign, e.g. `NEW DEAL` |
| PR | `prefix` | `??`, `_` and `%` are kept as typed |
| MESSAGE | `message_text`; `preamble` = first 6 characters | `.` `?` `_` `%` glyphs kept |
| Q | `qualifier` (raw glyph) | Inferred from the data only; the readme calls it "idiosyncratic". `>` appears with FOR rows, `+` with non-30 `[N CHAR]`, `&` with a repeat, `\` with disregard or callsign-only, `*` with uncertain copy, `%` with no-traffic days. |
| LN | `length` | Equals `len(MESSAGE)` in every sample row |
| SPECIFICATION | `specification` (raw); `FOR X` → `addressee`; `DISREGARDED` → `broadcast_type='disregard'` | For pipeline rows (raw is NULL), the export derives `FOR …` / `DISREGARDED` / `[N CHAR]` |
| E6 | `e6_tail` (raw, e.g. `AE0410 (FOR)`, `?`) | |
| NOTES / SOURCE / SET | `notes` / `source` / `set_label` | On export, SOURCE falls back to the receiver code |

Row classification on import: a `%` row → `msg_type='NIL'`; a `_` row → `OTHER`/`callsign_only`; callsign `SKYKING` → `SKYKING`; everything else → `EAM`.
`import_key` is a sha1 of the raw row plus an occurrence counter, so re-imports are idempotent.

**Round-trip results:**
- The full NEET sample is **byte-identical** on round trip, including from a mixed DB with `--origin neet_import`. A re-import adds 0 rows.
- The fake fixture (`tests/fixtures/neet_format_fake.csv`, covering quoting, `%`, `_`, `.`, FOR and DISREGARDED) is also byte-identical.

**Commands:**
- `scripts/hfgcs_db.py --db hfgcs.db import-neet file.csv`
- `scripts/hfgcs_db.py --db hfgcs.db export-neet out.csv [--query "2310&C30"] [--origin pipeline]`

Their repos and data have no licence. Keep their CSVs out of our repo, and import their data only with permission.

## 5. missileer.watch feed (reception fields + importer)
**Licence, verified 2026-10-26 ~09:45 PDT with a read-only GET:**
- `archive.json` and `status.json` both carry `"license": "CC0-1.0"` and `"attribution": "attribution appreciated: missileer.watch"`.
- The site footer says "Our recordings and decodes are released under CC0 1.0 (public domain dedication); attribution appreciated: missileer.watch" and links to the CC0 deed.
- robots.txt disallows only `/account` and `/admin`.
- No public code repo was found, so the feed's design is used as ideas and the data is ingested.

**Feed shape (schema v1):**
- `messages[]` holds `id, heard_at, heard_at_basis (voice_onset|clip_start), clip_start_at, preamble, preamble_vote, text, machine_text, corrected, correction_source, station, grade (message|unconfirmed|fragment|radio-check|auth|traffic|8888), group_count, char_count, partial, malformed, frequency_khz, announced_length(_machine), ear_queue(_reason), position_notes, anchors[{kind,t}], char_times[[1st reading],[2nd reading]], receptions[]`.
- Each reception holds its own `heard_at, station, frequency_khz, first, coreception, clip_url, clip_duration_s, voice_start_s, anchors, char_times`.
- `hidden[]` lists collapsed stubs.

**Mapping (`import-missileer`):**
- Each reception → one `messages` row. If it has a clip, it also gets a `captures` row: `audio_url`, `start_utc=clip_start_at`, duration, `voice_start_s`.
- Receivers are named `missileer:<station>`, e.g. `missileer:desktop-kiwi-kph`.
- `coreception=true` → `simulcast` link to the first reception; a later non-co-reception → `rebroadcast` link. Both use method `external:missileer`.
- Grade → `msg_type`: message/unconfirmed/fragment → EAM, radio-check → RADIOCHECK, auth → AUTH, traffic → TRAFFIC, 8888 → EAM_8888.
- Corrected rows → `review_status='reviewed'`. Unconfirmed or ear-queue rows → `needs_review`.
- Rows are upserted on `import_key` so later corrections flow in. Hidden stubs are skipped.
- Fetching uses `If-None-Match` with the stored ETag (the server returns 304 when nothing changed). The importer refuses to run if the feed's licence is not CC0-1.0.

**Tested:**
- On the live archive (238 entries and 245 receptions from 2026-09-22 to 2026-10-26 UTC; 236 captures; 7 reception links), run 4 times: idempotent, and the 4th run got `not_modified`.
- On a fake fixture as well.

**Caveats:**
- `clip_start_at` is the message's *first* recording start, and several clips can share it. Our natural-key unique index therefore applies only to our own captures, and external captures are keyed by clip URL.
- One station value is `desktop-kiwi-iceland+kph` (a combined copy), which is imported as its own receiver.

**Plan:**
- Poll hourly at most (the feed sends `cache-control: max-age=60`), using the ETag.
- Credit "missileer.watch (CC0)" in the UI and exports.
- The Decoding team's `decoding/ingest_missileer.py` (ASR labels) can read from the DB instead of the raw feed.

## 6. Message type taxonomy (`message_types`)
The type names were informed by eam.watch's public categories (`ALLSTATIONS, SKYKING, RADIOCHECK, SKYMASTER, SKYBIRD, DISREGARDED, BACKEND, OTHER`, seen in the legacy repo's schema, which has no licence, so no code was reused) and by missileer's grades.

| code | eam.watch | missileer | counts as EAM |
|---|---|---|---|
| EAM | ALLSTATIONS | message | ✓ |
| EAM_8888 | – | 8888 | ✓ |
| SKYKING | SKYKING | – | |
| SKYMASTER | SKYMASTER | – | |
| SKYBIRD | SKYBIRD | – | |
| RADIOCHECK | RADIOCHECK | radio-check | |
| AUTH | – | auth | |
| TRAFFIC | – | traffic | |
| BACKEND | BACKEND | – | |
| OTHER | OTHER | – | |
| NIL | – | – | |

- DISREGARDED is modelled as `broadcast_type='disregard'` rather than a type, so a disregarded EAM still counts.
- Not verified: eam.watch's exact meaning of BACKEND and SKYBIRD (they have no public definitions), and whether its live site still uses this list.
- eam.watch validation ideas to reuse in our own parser: ALLSTATIONS bodies use only `A–Z2–7` (base32-like), and Skyking follows the `<CODEWORD> TIME nn AUTH xx` pattern.

## 7. Daily counts and coverage comparison
**Counting rule (decision 1).** Daily reports show both counts:
- `eams` (the default) = **distinct messages**: the number of different contents heard on that UTC day, with rebroadcasts and simulcast copies collapsed. A content heard again on a later day counts once on each day it was heard.
- `eam_broadcasts` = per broadcast: simulcast copies collapsed, rebroadcasts counted.
- `eam_receptions` = every row.

**Coverage.** `v_coverage_comparison` is **per distinct message**: `coverage_pct = our_eams / external_count` and `diff = our_eams − external_count`. `our_broadcasts` is shown for information only. Tested with fake counts.

**warwatchintel: manual CSV only, no scraping (decision 4).**
- warwatchintel.com shows no data licence. Its robots.txt sets `Content-Signal: search=yes, ai-train=no, use=reference` and blocks AI crawlers.
- The DB tool has no fetcher for it, and `external_daily_counts.method` is constrained to `'manual'`.
- Process:
  1. A person reads the "EAMs per day" tiles on the public page and types them into a CSV: `source,date_utc,metric,count,day_boundary,method,collected_by,url,notes`, with `source=warwatchintel`, `metric=eams`, `method=manual`.
  2. Load it with `hfgcs_db.py import-daily-counts counts.csv`. Rows are upserted, so corrections overwrite.
  3. Suggested cadence is weekly, since the page shows the last 14 days.
- Asking the operator for a data export or permission remains an open item (§10). Any operator-supplied file would still go through the same manual CSV path.
- Still unknown: the site's day boundary (UTC is assumed, as the page says times are zulu) and whether its count collapses repeats. The page mentions "repeat detection", which fits our distinct-message basis, but this is unconfirmed.
- Reference point from this review (a single page view, 2026-10-26 ~09:50 PDT): the tiles read Oct 22 = 17, Oct 23 = 31, Oct 24 = 25 UTC. The scout digest's "31 on Oct 22, 25 on Oct 23" looks shifted by one day. missileer's archive has 2 distinct EAMs on Oct 22 UTC. None of these numbers are stored.

## 8. Datasette
```
pip install datasette          # tested with 0.65.5
datasette serve hfgcs.db -m datasette/metadata.json   # the file must be named hfgcs.db (metadata keys on it)
```
- The `messages` table is configured for FTS (`_search=`, raw mode), with facets on type, prefix, review status, origin and length.
- **Canned queries (all tested against a demo DB):**
  - `neet_search`: date, len, grp, prefix, callsign, receiver, review, text
  - `neet_days`: day-level, so lengths and groups listed must **all** occur, like the NEET galleries
  - `daily_counts`: distinct `eams` plus `eam_broadcasts`
  - `link_settings`
  - `hour_heatmap`
  - `length_by_month`
  - `rebroadcasts`
  - `rebroadcast_delays`
  - `prefix_activity`
  - `coverage_vs_external`: per distinct message
  - `receiver_coverage`
  - `review_queue`
  - `position_heatmap`: character × position for one length, the basis for NEET-style structure heatmaps
- Serve read-only (`--immutable` on a snapshot) for anything public.
- Datasette 1.0 moves canned queries to `datasette.yaml`; port the file when we upgrade.

**NEET-style query syntax** (CLI `query`, and the same filters in `neet_search`). Tokens are joined with `&`. They follow NEET's gallery help, with our extensions:

| Token | Meaning |
|---|---|
| `YYMMDD` | Date prefix: `2601` = Jan 2026, `--06` = every June, `----01` = every 1st |
| `C30`, `C32,27` | Length 30; length 32 or 27 |
| `G1`, `G23` | Group 1; group 2 or 3 |
| `R` | Not yet reviewed |
| `S<receiver>` / `S-<receiver>` | Receiver, or exclude receiver. NEET's `S#` means stream; ours means receiver |
| `PR=CZ` | Prefix |
| `CS=NEW DEAL` | Callsign (LIKE) |
| `F=11175` | Frequency kHz |
| `T=SKYKING` | Message type |
| `E6` | Row has an E-6B tail |
| `"QWFO"` | Full-text search |

A trailing `x` (NEET's week-collapse) is accepted and ignored. In the CLI, `C`/`G` are message-level (either value); use `--days` or `neet_days` for NEET's day-level semantics.

## 9. Proposed analytics
- **Structure analysis:** `position_heatmap` per length and group, then a Python step for chi-square/z-scores per position, pairwise-equality suppression, and G4 block segmentation, following NEET's published concepts.
- **Heatmaps:** weekday × hour, frequency × hour (once we capture several channels), and receiver × hour for coverage.
- **Calendar/timeline:** daily counts by group and length, and a rebroadcast timeline built from `message_links` (delay distribution, E-6B-flagged airings).
- **Coverage:** daily `coverage_pct` against external counts; per-receiver uptime from `captures`.
- **Quality:** review queue, ASR character error rate against corrected copies (missileer `machine_text` vs `text`), and length-mismatch rate (`announced_length` vs `text_length`).
- **Periodic reports:**
  - daily digest (counts, new prefixes, non-30 lengths, Skyking)
  - weekly coverage report
  - monthly structure/length report, exported as NEET CSV and Markdown

## 10. Open questions
1. **Licences and permissions:** may we ask NEET INTEL to import their logs and share our CSVs back, and warwatchintel for daily counts (still manual CSV either way)? Which licence do we choose for our own data (CC0, like missileer)?
2. **Group (G1–G5) assignment:** who maintains `prefix_groups`, and from what source? Prefixes are reused across years.
3. **Audio storage and retention:** local FLAC/Opus vs. object storage, public URLs, and how long to keep noise-only captures.
4. **Timestamps:** the scanner should log capture start and voice onset (Engineering). Until it does, the start times we derive from the end time are approximate.
5. **Multi-day relays:** should E-6B relays more than 48 h apart link? This will be decided when the thresholds are tuned.
6. **SQLite vs. Postgres:** SQLite + Datasette is fine at the current scale. Revisit if several sites write concurrently.

## 11. Decisions applied (chief of staff, 2026-10-26)
1. Daily reports show both counts. Distinct messages is the default (`eams`) and per-broadcast is its own column (`eam_broadcasts`). Coverage against external counts is per distinct message.
2. The dedup thresholds (120 s / 48 h / 0.90) are configurable defaults: CLI flags plus the `settings` table. They will be tuned on missileer data once Decoding has an evaluation set.
3. Rebroadcast linking lives in the DB tool. Decoding exposes only `text_similarity` (§3.1), which the tool uses when it is importable and otherwise falls back to its built-in.
4. No scraping of warwatchintel. Counts come only from hand-entered CSVs (`method='manual'` is enforced).
