# Deploying on hilltop

**Use `deploy/deploy.sh`, not `systemctl --user restart hfgcs-...`.**

`hfgcs-8992` is OpenWebRX's only client most of the time. When it disconnects, OpenWebRX stops the SDR
(it SIGTERMs the soapy connector), and when the recorder reconnects it starts it again. Each manual restart
of `hfgcs-8992` on 2026-10-26 (15:09, 15:26, 15:49, 16:17 and 16:20 PT) left a 2–3 s hole in the 8992 audio,
and anyone else listening on the waterfall lost it too. A KiwiSDR restart gives up a volunteer's slot and
opens a new session. A restart in the middle of a transmission truncates that recording. So a deploy
restarts a recorder **only if its own code changed**, **only while it is idle**, **never inside a known event
window**, and **one unit at a time**.

## Usage

Run it on hilltop as `scribe` (the systemd `--user` units' owner). Never use sudo; the script refuses root.

```bash
cd ~/hfgcs-app
deploy/deploy.sh --dry-run                 # fetch, then print the plan; changes nothing
deploy/deploy.sh                           # fast-forward to origin/main and restart what changed
deploy/deploy.sh --ref origin/some-branch  # another target (must be a fast-forward)
deploy/deploy.sh --force hfgcs-review      # also restart a unit nothing changed for (repeatable; 'all')
deploy/deploy.sh --quiet-window 2026-10-26T23:45Z/2026-10-27T07:00Z   # one-off window from the CLI
deploy/deploy.sh --help                    # every option
```

Options worth knowing: `--min-idle 30` (seconds a recorder must have been idle), `--idle-timeout 600` (then
skip it), `--stream-timeout 60` (wait for audio after each recorder restart), `--no-fetch`, `--no-migrate`,
`--no-install-units`, `--db PATH`, `--quiet-windows FILE`, `--since SHA`.

Exit status: `0` all done; `3` done, but some restart was deferred or skipped (it is remembered; run the
script again later); `1` a reinstall, migration or restart failed; `2` refused (root, local changes in the
checkout, not a fast-forward).

## What it does

1. Records the deployed SHA, refuses a dirty checkout, `git fetch`es and fast-forwards to `--ref`
   (default `origin/main`). It never merges, rebases or resets. Then it re-runs the *new* copy of the
   script for the remaining steps (`--since <old sha>`).
2. Reinstalls (`.venv/bin/python -m pip install -e '.[openwebrx]'`, plus `decode` once `hfgcs-decode` is
   installed) only if `pyproject.toml` changed. If the core `dependencies` or `requires-python` changed, every
   unit counts as changed; if only an optional extra changed, only the units that use it do (`openwebrx`:
   `hfgcs-8992`; `decode`/`asr`: `hfgcs-decode`; `dev`: none; an unknown extra: every unit).
3. If the review DB (`$HFGCS_DB`, else `HFGCS_DB` in `~/.config/hfgcs/review.env`, else `data/hfgcs.db`)
   has pending `migrations/NNNN_*.sql`, it backs it up to `data/backups/hfgcs-<UTC>-<old sha>.db`
   (SQLite online backup) and applies them. A failure stops the deploy before any restart.
4. Works out the affected units from the changed paths:

   | Changed path | Restarts |
   |---|---|
   | `hfgcs/sources/openwebrx.py` | `hfgcs-8992` (every `hfgcs-<kHz>` unit) |
   | `hfgcs/sources/kiwisdr.py` | every `hfgcs-kiwi-*` |
   | `hfgcs/dsp.py`, `hfgcs/capture.py`, `hfgcs/config.py`, `hfgcs/runstate.py`, `hfgcs/shadow.py`, `hfgcs/sources/base.py`, `hfgcs/sources/__init__.py`, `hfgcs_scanner.py` | all recorders |
   | `hfgcs/review_app.py`, `hfgcs/review_db.py`, `hfgcs/db_migrations.py`, `migrations/*` | `hfgcs-review` |
   | `hfgcs/static_site.py`, `public_site/*` | `hfgcs-public` |
   | `decoding/*` | `hfgcs-decode` only (the recorders never import `decoding/`) |
   | `deploy/bin/hfgcs-python` | `hfgcs-review`, `hfgcs-public`, `hfgcs-decode` |
   | `hfgcs/__init__.py`, core dependency changes in `pyproject.toml` | everything |
   | an optional extra in `pyproject.toml` | the units that use it (step 2) |
   | `deploy/systemd/<unit>.service` | that unit: the installed copy in `~/.config/systemd/user/` is replaced (old one kept as `.bak-<UTC>`), then `daemon-reload` and a restart. Units that are not installed are never installed or started. `hfgcs-decode-backfill` (one-shot) is installed but never started. |
   | anything else: docs, tests, `scripts/` (incl. `hfgcs_db.py`, which the decode worker runs as a fresh subprocess per load), report tools such as `hfgcs/shadow_score.py`, `deploy/env/*`, the deploy tool itself | nothing |

   Units are found with `systemctl --user list-unit-files 'hfgcs-*'` and classified by name. Restarts left
   over from an earlier run (deferred or skipped) are added; they live in
   `~/.local/state/hfgcs/pending-restarts.json`.
5. Restarts, strictly one at a time, in this order: `hfgcs-review`, `hfgcs-public`, `hfgcs-decode`, the
   `hfgcs-kiwi-*` units by name, and `hfgcs-8992` **last**. Stopped units are left stopped. `hfgcs-decode` is
   not a recorder: it restarts without an idle wait and inside quiet windows (its in-flight clip is decoded
   again on the next start).
   - **Recorders wait for idle.** Each recorder writes a state file (below). The script restarts it only
     once it has been `idle` for `--min-idle` seconds (default 30), polling every 2 s for up to
     `--idle-timeout` (default 10 min). A recorder that is still capturing is skipped and remembered.
     `--force` never interrupts a capture.
   - **Quiet windows.** Inside a window (see below), recorder restarts are deferred and remembered, unless
     the unit is named with `--force`. The window is checked again while waiting for idle.
   - **After each recorder restart** it waits until the unit is `active` and the new process has
     received audio (its state file), or 60 s (`--stream-timeout`), before touching the next unit.
6. Prints a summary (restarted / deferred / skipped / failed) and appends the run to
   `~/.local/state/hfgcs/deploy-history.jsonl`; `~/.local/state/hfgcs/deployed-sha` holds the deployed commit.

## Quiet windows

`~/.config/hfgcs/quiet-windows.toml` (copy `deploy/quiet-windows.example.toml`), all times UTC:

```toml
windows = ["2026-10-26T23:45Z/2026-10-27T07:00Z",
           { span = "2026-11-02T00:00Z/2026-11-02T06:00Z", note = "exercise" }]
daily = ["23:45-01:00"]   # every day, may wrap midnight
```

Times without a zone are rejected. `--quiet-window SPEC` adds windows from the command line.

## Recorder state file and connection log

Each recorder (`hfgcs_scanner.py` running in an `hfgcs-*` systemd unit, or with `--state-name`/
`$HFGCS_STATE_NAME`) keeps `$XDG_RUNTIME_DIR/hfgcs/<unit>.state` up to date, replacing it atomically:

```json
{"schema": 1, "name": "hfgcs-8992", "pid": 12345, "state": "idle", "since_utc": "2026-10-26T23:10:04.120Z",
 "receiver": "owrx-hilltop", "channel_khz": 8992.0, "last_audio_utc": "2026-10-26T23:12:30.004Z", ...}
```

`state` is `starting`, `idle`, `capturing` (a recording is pending or open) or `stopped`. The deploy
script trusts the file only if its `pid` is the unit's current `MainPID`. `last_audio_utc` is refreshed at
most every 5 s.

Connect, disconnect (with the websocket close code and reason), reconnect attempts, failed connects and
"audio resumed after an N s gap" are logged at INFO (disconnects at WARNING) with UTC timestamps, e.g.

```
openwebrx DISCONNECT at 2026-10-26T23:09:12.481Z: lost ws://localhost:8073/ws/: connection closed (close code 1006)
openwebrx AUDIO_RESUMED at 2026-10-26T23:09:15.322Z: audio from ws://localhost:8073/ws/ resumed after a 2.84 s gap
```

and appended as JSON lines to `logs/<unit>-connection.log` (e.g. `~/hfgcs-app/logs/hfgcs-8992-connection.log`,
`--events-log` to change it). The KiwiSDR source logs the same events per host.

## First deploy of this script

Recorders started before this change write no state file, so the script cannot tell whether they are
capturing and skips them ("no state file"). Once:

```bash
cd ~/hfgcs-app && git fetch origin && git merge --ff-only origin/main
deploy/deploy.sh --since e6ca4dd --dry-run             # e6ca4dd = the commit deployed before
tail -n 5 logs/hfgcs-8992.log                          # no recording in progress? outside any event?
deploy/deploy.sh --since e6ca4dd --force hfgcs-8992 --force hfgcs-kiwi-11175 ...   # each recorder
```

`--force` restarts a recorder without a state file without the idle check, so pick a quiet moment. From
then on every recorder writes its state file and plain `deploy/deploy.sh` is enough.

## Decode service (`hfgcs-decode`)

`hfgcs-decode.service` runs `python -m decoding.worker`: it watches `hfgcs_catalog.jsonl`, voice-gates each new
capture, decodes it with `small.en` plus the vote pass in a child process under a wall-time cap, and loads the
result with `hfgcs_db.py load-decodes`, so it shows on the review page. Details: docs/decoding.md "Decode
service". It runs at `Nice=19`, `CPUWeight=10`, `IOSchedulingClass=idle`, `MemoryHigh=3G`/`MemoryMax=4G`, and
it never touches a recorder, OpenWebRX or a KiwiSDR.

### Adding it on hilltop (no recorder restart)

As `scribe`, never sudo. None of these steps restarts a recorder, so they are fine inside a quiet window;
the decode work itself is low priority. First get the code (normally `deploy/deploy.sh` once this is merged:
the only units it could touch are `hfgcs-decode*`, which are not installed yet, so it restarts nothing and
reinstalls `.[openwebrx]` because pyproject.toml changed; check with `--dry-run` first).

```bash
cd ~/hfgcs-app
deploy/deploy.sh --dry-run                     # expect: plan: no unit needs a restart
deploy/deploy.sh

# 1. The decode extra: faster-whisper (CTranslate2, onnxruntime for silero-vad, tokenizers,
#    huggingface_hub, PyAV) and imageio-ffmpeg (a static ffmpeg; hilltop has no ffmpeg binary). ~350 MB.
.venv/bin/python -m pip install -e '.[openwebrx,decode]'

# 2. The model, once (small.en, ~460 MB into ~/.cache/huggingface). The unit runs with HF_HUB_OFFLINE=1.
.venv/bin/python -m decoding.worker --prefetch
.venv/bin/python -m decoding.worker --check     # "ready: model small.en, ffmpeg ..., decoder_version ..."

# 3. Settings (no secrets): DB path, catalog, model, threads, caps.
install -m 600 deploy/env/decode.env.example ~/.config/hfgcs/decode.env

# 4. The units. Copying new unit files and daemon-reload restart nothing that is running.
cp deploy/systemd/hfgcs-decode.service deploy/systemd/hfgcs-decode-backfill.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hfgcs-decode
journalctl --user -fu hfgcs-decode               # "first start: watching captures that end after ..."

# 5. The captures recorded before step 4 (one-shot; shares the one-clip-at-a-time lock with hfgcs-decode).
systemctl --user start --no-block hfgcs-decode-backfill
journalctl --user -fu hfgcs-decode-backfill
.venv/bin/python -m decoding.worker --status    # per capture: decoded / no_voice / timeout / failed / new
```

Exit status 78 means a dependency or the model is missing; the unit then stays stopped (no restart loop) and
the journal says what to install. To stop decoding: `systemctl --user stop hfgcs-decode` (the clip in progress
is decoded again on the next start). State: `~/.local/state/hfgcs/decode-state.json` and
`decode-history.jsonl` (one line per clip with wall time and result).
