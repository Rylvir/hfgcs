# Review app (private) and project page (public)

Two separate web pages, two separate processes, two separate systemd units:

| | Private review app | Public project page | Datasette (optional) |
|---|---|---|---|
| Code | `hfgcs/review_app.py` + `hfgcs/review_db.py` | `hfgcs/static_site.py` + `public_site/` | `datasette/metadata.json` |
| Port | **8090** | **8091** | 8092 |
| Bind on hilltop | `192.168.2.214` (LAN) | `192.168.2.214` | `192.168.2.214` |
| Code default bind | `127.0.0.1` | `127.0.0.1` | – |
| Who reaches it | LAN `192.168.2.0/24` only | mountain's Cloudflare tunnel → `hfgcs.scanplacer.com` | LAN only |
| Data | DB read/write, recordings read | `public_site/` files only | DB read-only |
| Unit | `deploy/systemd/hfgcs-review.service` | `deploy/systemd/hfgcs-public.service` | `deploy/systemd/hfgcs-datasette.service` |

Both servers are Python standard library only (`http.server`, `sqlite3`). The base install already needs
no web framework, and main's dependencies (numpy, scipy, soundfile) contain none. Both apps are small (one
list page, one form per message, a few static files), and a framework would add a dependency to audit for
no real gain. Neither page uses JavaScript.

**The tunnel must only ever point at 8091.** The review app and Datasette have no public mode.

## 1. Private review app

### What it does
- Lists recordings (`captures`) newest first, grouped by UTC day. Each one has an HTML5 audio player beside
  every decoded message for that capture. The player starts at the recorder's pre-roll offset
  (`voice_start_s`), so playback begins at the voice onset.
- Shows the parsed fields: type, callsign, prefix, declared length, addressee, group (from `prefix_groups`
  unless overridden), confidence, and character count.
- Edits the transcript, message text, callsign, prefix, length, addressee and type. It also sets a
  **checked** flag ("transcript and fields are correct") and keeps free-text notes.
  - The first time the message text is changed, the machine copy moves to `machine_text`, `corrected`
    is set to 1, and `correction_source` is set to `review-ui` (the missileer.watch convention).
  - A checked message gets `review_status = reviewed`. Unchecking it sets `needs_review`.
  - Every changed value is appended to `edit_history`, which the history link shows.
  - Saves use optimistic locking. A form rendered before someone else's save is refused with 409 rather
    than overwriting their edit.
- If a recording has no decoded message, "Add a message to transcribe" creates an empty `manual` message
  for it.
- Filters: UTC date, frequency, "unchecked only" (recordings with no message or with any unchecked
  message). Results come 25 per page.
- Next to each day there is a plain link to NEET INTEL's Daily Timecard gallery for that day (see §1.4).
  It is a link only; the app never fetches anything from NEET INTEL.
- Gold-label export: `GET /export/gold.jsonl`, or the CLI (see §1.3).

### 1.1 Database: migrations 0002 and 0003
`migrations/0002_review.sql` adds:
- `reviews(message_id PK→messages, checked 0/1, notes, reviewer, reviewed_utc, created_utc, updated_utc)`.
- `edit_history(id, edited_utc, editor, editor_source, change_id, table_name, message_id, field,
  old_value, new_value)`.
  - It is **append-only**: triggers abort any `UPDATE` or `DELETE`.
  - One form save produces one row per changed field, all sharing a `change_id`.
  - `editor_source` is `cf_access`, `form`, `default` or `cli`.
  - There is deliberately no foreign key, so history survives if a message is deleted or re-imported.
- `v_gold_labels`: checked reviews joined to the message, capture and receiver.
- Migration 0003 adds `captures.snr_peak_db` (the page shows mean and peak SNR) and `message_decodes`. When
  `load-decodes` brings a newer machine copy for a message a person already edited or checked, the page shows
  it under the form as "newer machine decode …, not applied over your edits".

Apply it with `scripts/hfgcs_db.py --db "$HFGCS_DB" migrate`. `init` also applies pending migrations, and
`migrate --status` shows the current version. The runner is `hfgcs/db_migrations.py`: each
`migrations/NNNN_*.sql` file runs once, in one transaction, and is recorded in `schema_version`. The app
refuses to start on a DB missing migration 0002 or 0003 unless it is started with `--migrate`.

### 1.2 Run locally
```bash
python3 scripts/hfgcs_db.py --db /tmp/hfgcs.db init          # schema v1 + migrations 0002, 0003
python3 scripts/hfgcs_db.py --db /tmp/hfgcs.db ingest-catalog hfgcs_catalog.jsonl --recordings-root .
python -m hfgcs.review_app --db /tmp/hfgcs.db serve --recordings ./recordings   # http://127.0.0.1:8090/
```

| Setting | Env var | Flag | Default |
|---|---|---|---|
| DB file | `HFGCS_DB` | `--db` | `~/hfgcs-app/data/hfgcs.db` |
| Recordings dir (only place audio is served from) | `HFGCS_RECORDINGS` | `--recordings` | `~/hfgcs-app/recordings` |
| Bind address | `HFGCS_REVIEW_HOST` | `--host` | `127.0.0.1` |
| Port | `HFGCS_REVIEW_PORT` | `--port` | `8090` |
| Allow `0.0.0.0` / `::` | – | `--allow-all-interfaces` | refused |
| Extra accepted `Host` headers | `HFGCS_REVIEW_ALLOWED_HOSTS` (comma list) | – | none |
| Editor name if the form field is empty | `HFGCS_REVIEW_EDITOR` | – | (none: name required) |
| Trust `Cf-Access-Authenticated-User-Email` | `HFGCS_REVIEW_TRUST_CF_ACCESS=1` | `--trust-cf-access` | off |
| CSRF signing key | `HFGCS_REVIEW_SECRET` | – | random per start |
| Apply migrations at start | – | `--migrate` | off |

`scripts/hfgcs_db.py` also reads `HFGCS_DB` as its `--db` default.

### 1.3 Gold labels for Decoding
```bash
python -m hfgcs.review_app export-gold --out labels_review.jsonl     # uses $HFGCS_DB and $HFGCS_RECORDINGS
curl -s http://192.168.2.214:8090/export/gold.jsonl > labels_review.jsonl   # same content, from the LAN
```
There is one JSON object per **checked** message, ordered by `heard_utc`. The shape follows Decoding's
`labels.jsonl` (`decoding/ingest_missileer.py`), so `python -m decoding.evaluate --labels
labels_review.jsonl --subset gold` can read it:

| Key | Meaning |
|---|---|
| `id` | `hfgcs:msg<message id>` (stable) |
| `source`, `source_id` | `"hfgcs-review"`, `msg<message id>` (Decoding's cache key) |
| `export_schema` | `1` |
| `label_status` | always `"gold"` |
| `type` | Decoding's vocabulary: `eam`, `radio_check`, `auth`, `traffic`, `skyking`, `other` |
| `msg_type` | our `message_types.code` (e.g. `EAM`, `EAM_8888`) |
| `labels.body` | normalized text (upper-case, no spaces, tabs or hyphens) for `EAM` / `EAM_8888`, else null |
| `labels.preamble` | `preamble` column, else the first 6 characters of an EAM body |
| `labels.declared_length` | `length` (LN), else `announced_length` |
| `labels.callsign`, `labels.addressee`, `labels.prefix` | as reviewed |
| `labels.read1`, `labels.read2` | null (not captured by the review form) |
| `transcript` | reviewed raw transcript |
| `reference_machine.text` | the machine copy (`machine_text` when corrected, else the current text) |
| `reference_machine.confidence` | decoder confidence |
| `corrected` | whether the reviewer changed the message text |
| `audio_path` | absolute path, set only if the file resolves inside the recordings dir; else null |
| `audio_file`, `audio_sha256` | as stored in `captures` |
| `heard_at_utc`, `clip_start_at_utc`, `voice_start_s` | message time, capture (file) start, voice offset in the file |
| `frequency_khz`, `receiver`, `capture_id`, `message_id` | provenance |
| `reviewer`, `reviewed_utc`, `notes` | review metadata |
| `split` | `dev`/`test`, using the same SHA-256 rule as `decoding/ingest_missileer.split_for` (on `source_id`) |

### 1.4 NEET INTEL daily timecard link
The pattern is `https://neetintel.github.io/dtcp_gallery.html?YYMMDD`, for example `…?261025` for
2026-10-25 UTC. It was derived from the local clone of the public `neetintel.github.io` repo (commit
`ecf782c`), file `dtcp_gallery.html`:
- Each timecard card is labelled with its day as `YYMMDD`, e.g. `card-date">230623` through `261025`.
- The page seeds its search from the query string. In the source, the comment reads
  "Seed the search from the URL: ?c32,,27 or ?250111 …", and the code reads
  `location.search` into the search box and runs the search. The site help says
  "every search is also a link".
- Cards also name an archive.org file (`HFGCS+TIMECARD+YYMMDD.mp4` in item
  `hfgcs-daily-timecard-project`). We do not link that directly, because not every day has a card.

### 1.5 Security model
- **Bind:** the code default is `127.0.0.1`. `0.0.0.0` / `::` / empty are refused unless
  `--allow-all-interfaces` is passed. On hilltop it binds `192.168.2.214` only, as defence in depth beside
  ufw.
- **Host header allow-list:** `<bind address>:<port>` plus `HFGCS_REVIEW_ALLOWED_HOSTS`. Anything else gets
  400, which defeats DNS rebinding.
- **CSRF**, on every POST:
  1. The request must be same-origin. `Origin` must match `Host`; if `Origin` is absent, `Referer` must;
     `Sec-Fetch-Site`, if sent, must be `same-origin`/`none`. A request with neither header is refused.
  2. A signed double-submit token is required: the hidden form field must equal the `hfgcs_csrf` cookie
     (`HttpOnly; SameSite=Strict`), and its HMAC-SHA256 signature must verify with the server key.
  3. Only `application/x-www-form-urlencoded` bodies of at most 64 KB are accepted.
- **Audio:** served only by capture id (`/audio/<id>`), never by a path from the URL. The stored path is
  resolved (symlinks and `..` included) and must be an existing regular file of an audio type inside
  `HFGCS_RECORDINGS`. Range requests are supported for seeking.
- **Headers:** `Content-Security-Policy: default-src 'none'; style-src 'self'; media-src 'self'; …;
  frame-ancestors 'none'`, plus `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: same-origin`,
  `Cross-Origin-Resource-Policy: same-origin`, `Cache-Control: no-store` on pages, and
  `<meta name=robots content=noindex>`.
- **Editor identity:** the name typed in the form, remembered in a cookie. It is not authentication: anyone
  on the LAN who can reach 8090 can edit, and the history records what they typed. `Cf-Access-
  Authenticated-User-Email` is used **only** with `HFGCS_REVIEW_TRUST_CF_ACCESS=1`. Enable that only if the
  app ever sits behind Cloudflare Access with no other path to it, because on the LAN anyone can forge the
  header.
- SQL is parameterized throughout, and all output is HTML-escaped.

## 2. Public project page
`public_site/` holds plain HTML/CSS: `index.html`, `style.css`, `404.html`, `favicon.svg` and `robots.txt`.
The page explains the project and links to <https://waterfall.scanplacer.com> and the GitHub repo. It shows
**no recordings, transcripts, logs or location**.

`hfgcs/static_site.py` serves that directory and nothing else:
- GET/HEAD only.
- No directory listing: a directory without `index.html` returns 404.
- No dotfiles.
- Only `.html .css .svg .png .ico .txt .webmanifest`.
- Symlinks or `..` escaping the directory return 404.
- Strict CSP.

It never imports the DB or review modules.

```bash
python -m hfgcs.static_site                      # 127.0.0.1:8091, serves ./public_site
HFGCS_PUBLIC_HOST=192.168.2.214 python -m hfgcs.static_site
```
| Env var | Flag | Default |
|---|---|---|
| `HFGCS_PUBLIC_DIR` | `--dir` | `<repo>/public_site` |
| `HFGCS_PUBLIC_HOST` | `--host` | `127.0.0.1` (`0.0.0.0` needs `--allow-all-interfaces`) |
| `HFGCS_PUBLIC_PORT` | `--port` | `8091` |

`tests/test_public_site.py` fails if `public_site/` contains any of the following:
- audio files, or file types outside the allowed list
- hidden files
- SQLite/RIFF/Ogg/FLAC magic bytes
- references to the DB, Datasette, the review app or its routes, ports 8090/8092, private IPs, hilltop,
  `HFGCS_DB`, or `<audio>`/`<video>` tags
- coordinates: decimal pairs, degree signs, DMS, `lat=`/`lon=`, `geo:` URIs, map links, Maidenhead locators

Add new public assets as SVG or HTML. If a PNG is ever needed, strip its metadata first; the test does not
parse image metadata.

## 3. Deploy on hilltop (user `scribe`, `~/hfgcs-app`)
Assumed layout. This is not on main yet and needs the Engineering Lead's confirmation:
```
~/hfgcs-app/                 git checkout of this repo
  .venv/                     python3.12 -m venv .venv && .venv/bin/pip install -e .   (default interpreter, §3.1)
  recordings/                written by the 8992 recorder (record_dir)
  hfgcs_catalog.jsonl        the recorder's catalog (catalog_file)
  data/hfgcs.db              HFGCS_DB
  public_site/               served on 8091
~/.config/hfgcs/review.env   from deploy/env/review.env.example (chmod 600)
~/.config/hfgcs/public.env   from deploy/env/public.env.example
```
```bash
mkdir -p ~/hfgcs-app/data ~/.config/hfgcs ~/.config/systemd/user
cd ~/hfgcs-app
.venv/bin/python scripts/hfgcs_db.py --db data/hfgcs.db init
sqlite3 data/hfgcs.db 'PRAGMA journal_mode=WAL;'   # recommended: ingest + review writes don't block reads
.venv/bin/python scripts/hfgcs_db.py --db data/hfgcs.db ingest-catalog hfgcs_catalog.jsonl --recordings-root .
.venv/bin/python scripts/hfgcs_db.py --db data/hfgcs.db load-decodes recordings/   # Decoding's sidecars; links after
cp deploy/env/review.env.example ~/.config/hfgcs/review.env && chmod 600 ~/.config/hfgcs/review.env
cp deploy/env/public.env.example ~/.config/hfgcs/public.env
cp deploy/systemd/hfgcs-review.service deploy/systemd/hfgcs-public.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hfgcs-review.service hfgcs-public.service
sudo loginctl enable-linger scribe        # start user units at boot without a login
journalctl --user -u hfgcs-review -f
```
- Optional Datasette: `pip install datasette` into the interpreter's environment, then copy and enable
  `deploy/systemd/hfgcs-datasette.service` (LAN only, 8092, read-only). Nothing depends on it.
- The recorder's catalog is already handled. `ingest-catalog` reads the schema-2 lines written by
  `hfgcs/capture.py` (`start_utc`/`file_start_utc`/`end_utc`, `receiver_id`, `sha256`,
  `snr_db_mean`/`snr_db_peak`, `noise_floor_db`, `preroll_sec`, `file`), as well as the legacy scanner
  format (see `docs/database.md` §1). Re-running it is idempotent, so a timer or cron job can call it.
- The units restart on failure every 10 s. If the user manager starts before `192.168.2.214` is assigned,
  the first bind fails and the next retry succeeds.
- Existing DBs: `scripts/hfgcs_db.py --db data/hfgcs.db migrate`. Migration 0003 adds `captures.snr_peak_db`
  (back-filled from the stored catalog JSON), `messages.decoder_version`/`decoded_utc` and `message_decodes`.
  The review app needs migrations up to 0003.
- Transcripts: `load-decodes` (docs/database.md §12; Decoding's output contract v1) is idempotent. Run it after `ingest-catalog`, from the
  same timer, and it never overwrites a row a person has edited or checked.

### 3.1 Choosing the Python interpreter
The units do not hardcode the venv. Each `ExecStart` runs `deploy/bin/hfgcs-python`, a small `sh` wrapper
that `exec`s `$HFGCS_PYTHON`, falling back to `$HOME/hfgcs-app/.venv/bin/python` (the Python 3.12 venv on
hilltop). Each unit also sets `Environment=HFGCS_PYTHON=%h/hfgcs-app/.venv/bin/python` *before* its
`EnvironmentFile=`, so there are two ways to override it:

1. **Env file (simplest):** add `HFGCS_PYTHON=/path/to/python` to `~/.config/hfgcs/review.env` /
   `public.env` (`datasette.env` for the optional Datasette unit), then `systemctl --user restart <unit>`.
   Values in `EnvironmentFile=` override `Environment=`.
2. **Drop-in:** `systemctl --user edit hfgcs-review.service` and add
   ```ini
   [Service]
   Environment=HFGCS_PYTHON=/opt/python3.12/bin/python3
   ```
   This works too as long as the env file does not set `HFGCS_PYTHON` (the env file still wins). To replace
   the command entirely, a drop-in can clear and redefine it: `ExecStart=` followed by
   `ExecStart=/path/to/python -m hfgcs.review_app serve`.

If the interpreter is missing, the wrapper exits 127 with the path it tried, which shows up in
`journalctl --user -u <unit>`. The optional Datasette unit runs `hfgcs-python -m datasette`, so it uses the
same interpreter, which must have `datasette` installed.

### 3.2 Firewall (suggested; adjust to the existing ufw rules)
```bash
sudo ufw allow from 192.168.2.0/24 to any port 8090 proto tcp   # review app: LAN only
sudo ufw allow from 192.168.2.0/24 to any port 8092 proto tcp   # Datasette, only if enabled
sudo ufw allow from <mountain's LAN IP> to any port 8091 proto tcp   # public page: tunnel host only
```

### 3.3 Tunnel (on mountain; for reference only)
The ingress rule for `hfgcs.scanplacer.com` must be `service: http://192.168.2.214:8091`. No hostname may
map to `:8090` or `:8092`.
