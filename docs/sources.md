# Audio sources

`hfgcs_scanner.py` can record from four kinds of receiver. Pick one with
`source = "..."` in the TOML config or `--source` on the command line:

| Source | Delivers | Use it for | Extra dependency |
|---|---|---|---|
| `openwebrx` | USB audio from the server's own demodulator | **The Airspy HF+ at waterfall.scanplacer.com** (OpenWebRX owns it) | `pip install -e ".[openwebrx]"` (websockets) |
| `kiwisdr` | USB audio from one KiwiSDR slot | Channels the Airspy can't cover | kiwiclient (`git clone https://github.com/jks-prv/kiwiclient`), run as a separate program |
| `soapy` | IQ from a local SoapySDR device | A radio nothing else is using | `python3-soapysdr` from the distro |
| `file` | WAV audio or IQ replay | Tests, offline reprocessing | none |

The default is still `soapy`, so existing setups behave as before.
**Never run the `soapy` source against the Airspy while OpenWebRX is running**:
it either fails to open the device or knocks OpenWebRX off it.

Every source implements `hfgcs.sources.base.AudioSource`: `open()` / `close()`,
`tune(dial_khz)`, `can_tune()` / `refusal()`, `read(timeout)` returning a
`Chunk` (samples, rate, optional receiver-reported level, gap marker),
`sample_rate`, `kind` (`"iq"` or `"audio"`) and `receiver_id` (written to the
catalog and file name). `ChannelCapture` runs the Weaver USB demodulator on
IQ sources and only resamples/band-limits audio sources; the squelch, pre-roll,
hang time, AGC and catalog are shared.

When the scanner has a single usable channel it stays on it (no retune traffic);
with several it dwells on each in turn. Channels a source refuses are listed
on stderr and skipped; if none are left the scanner exits with status 2.

## OpenWebRX (`--source openwebrx`)

```toml
source = "openwebrx"
channels = [11175]          # must be inside the profile the server is on now

[openwebrx]
url = "https://waterfall.scanplacer.com"   # or http://127.0.0.1:8073 on the host itself
receiver_id = "owrx-scanplacer"            # default: owrx-<host>
low_cut_hz = 300
high_cut_hz = 3000
squelch = "smeter"          # server S-meter (the audio is AGC'd); or "audio"
allow_profile_switch = false
```

```bash
.venv/bin/python hfgcs_scanner.py --source openwebrx --channel 11175
```

How it works: it connects to `<url>/ws/`, does the
`SERVER DE CLIENT` / `CLIENT DE SERVER` handshake, sends
`connectionproperties` (12 kHz audio), waits for the `config` message
(`center_freq`, `samp_rate`, profile), then starts **its own** USB demodulator
with `dspcontrol` (`mod: usb`, `low_cut`/`high_cut`, `offset_freq` = dial −
center, squelch off). Audio frames (type 2) are decoded from OpenWebRX's
sync-framed IMA ADPCM or raw int16 and resampled to 8 kHz. The server's
`smeter` messages drive the squelch, because OpenWebRX AGCs the audio.

### Etiquette safeguards (the server is shared)

- **No profile switching by default.** `selectprofile` and center-frequency
  changes retune the radio for every listener. The source only moves its own
  demodulator inside the profile that is active when it connects.
- **Refuses channels outside the active profile** with a clear `TuneRefused`
  error naming the profile and its span (the usable span is 90% of
  `samp_rate`, `span_margin`). Setting `allow_profile_switch = true` (or
  `--allow-profile-switch`) allows a `selectprofile` to a profile named
  `HFGCS <kHz>` (or `[openwebrx] profile = "sdr|profile"`), no more than once
  every 11 s. Only do this with the owner's OK.
- **Never sends `setfrequency`** (there is a guard in the code).
- **If someone else moves the radio** so our channel is no longer inside it,
  the source stops recording (drops audio, closes any open recording) and
  waits. It does not move the radio back. If the channel is still inside the
  new window it just re-sends its own offset.
- **Polite rate limits:** ≥ 1 s between our `dspcontrol` retunes, ≥ 10 s
  between connection attempts, reconnects with exponential backoff and jitter
  (5 s doubling to 5 min, give up after 20 consecutive failures). A server
  `backoff` message is honoured.
- Identifies as `client=hfgcs` in the handshake and with a
  `User-Agent: hfgcs/0.1 (+https://github.com/Rylvir/hfgcs)` header; no
  websocket pings (OpenWebRX does not answer them).
- One websocket per scanner; no waterfall is requested beyond what the server
  sends anyway.

### Waiting for the right profile (long-running / service mode)

With `--wait-for-profile` (or `[openwebrx] wait_for_profile = true`) a channel
outside the active profile is not an error. The scanner stays connected,
leaves its demodulator off and idles. When the owner (or anyone) selects a
profile that covers the channel, it starts recording by itself. If the profile
later moves away, it closes the open recording and idles again. It never
retunes the radio back. State changes are logged once each:

    WARNING WAITING: 8992 kHz is not in the active profile airspyhf|hfgcs-15016 (...); idling ...
    WARNING RECORDING: 8992 kHz is covered by the active profile airspyhf|hfgcs-8992 (...)

In this mode reconnects keep going (backoff capped at 5 min) instead of giving
up after `max_reconnects`, so a service survives an OpenWebRX restart.

Connection events (connect, disconnect with the close code, reconnect attempts,
and the length of the audio gap when audio resumes) are logged with UTC
timestamps and appended to `logs/<unit>-connection.log` as JSON lines; see
docs/deploy.md. Restart `hfgcs-8992` only through `deploy/deploy.sh`: OpenWebRX
stops the SDR whenever its only client disconnects.

Defaults: **on** when `--duration` is not given (service mode), **off** for
one-shot runs with `--duration`, `--probe` or `--select-profile`. A one-shot run
outside the active profile still exits with code 2. `--no-wait-for-profile`
forces it off.

### One-shot profile selection (`--select-profile NAME`)

For a deliberate, owner-approved switch of the shared radio:

```bash
# the key is read from the environment and never printed or logged
set -a; . ~/.config/hfgcs/hfgcs.env; set +a
python hfgcs_scanner.py --source openwebrx --url http://localhost:8073 \
    --select-profile "HFGCS 8992 kHz" [--unlock-key-env OPENWEBRX_MAGIC_KEY]
```

`NAME` is a profile id (`sdr|profile`), its full name, or a name suffix such as
`HFGCS 8992 kHz` (it must match exactly one profile). The command sends **one**
`selectprofile` (with the OpenWebRX+ unlock "magic key" as `params.key`), waits for the
server's new `config`, prints the profile before and after, and exits (0 = done
or already active, 1 = key variable not set, 3 = refused or not confirmed; a
locked-profile refusal is reported, not retried). Nothing else changes:
`allow_profile_switch` stays false, and the recorder itself never needs the key.

This source was first run against a live OpenWebRX+ 1.2.126 server on
2026-10-26.

## KiwiSDR (`--source kiwisdr`)

```toml
source = "kiwisdr"
channels = [8992]

[kiwisdr]
enabled = true                      # set by hand: installing is not authorising
ident = "N0CALL hfgcs"              # shown in the sysop's user list; must identify you
hosts = ["kiwisdr.example.org:8073", "backup.example.net:8074"]
channel_khz = 8992
kiwirecorder = "/opt/kiwiclient/kiwirecorder.py"   # or $KIWIRECORDER
blocklist_file = "kiwi_blocklist.txt"              # "host" or "host:port" per line
```

The source runs `kiwirecorder.py --nc --nc-wav -m usb -f <dial> -L 300 -H 3000
-u <ident> --tlimit 1320` as a subprocess and reads the WAV stream from its
stdout. kiwiclient is **not vendored or imported** (see THIRD_PARTY.md).

**One session per kiwirecorder process.** When a Kiwi ends a running session,
kiwirecorder (checked against kiwiclient v1.11, commit `4eb733e`, `kiwi/worker.py`)
logs `<host>:<port> server closed the connection cleanly. Reconnecting after 5 seconds`
(or `unexpectedly`) at INFO and reconnects in the same process every 5 s with no
limit; `--connect-retries` / `--busy-retries` do not cover that path, and the only
switch that skips it (`--no-api`) also stops sending the ident and frequency. So
the source runs kiwirecorder with `--log info`, reads its stderr continuously,
and on the first in-process reconnect or redirect message (`server closed the
connection`, `Reconnecting after N seconds`, `Redirecting to`, `sleeping and
reconnecting`) stops it (SIGINT, SIGTERM after 2 s, before the 5 s reconnect).
The session counts as failed for that host: a server-ended session is treated as
being kicked (host retired 60 min) and the usual backoff applies. The `--tlimit`
end and our own 22-min cap still release the slot normally, followed by the
≥ 60 s rest.

`hfgcs_scanner.py` exits 0 when stopped with SIGINT (systemd `KillSignal=SIGINT`),
even if the unit's log pipe (`tee`) was stopped by the same signal first; real
errors still exit non-zero.

### Etiquette rules implemented

Adapted from `ETIQUETTE.md` in
[Chipensaw/HFGCS-Monitor-KiwiSDR](https://github.com/Chipensaw/HFGCS-Monitor-KiwiSDR)
(Apache-2.0) and the Signal Ops coverage survey (2026-10-26):

| Rule | How |
|---|---|
| Nothing is contacted until you opt in | `enabled = true` must be set; the default is off |
| Honest identity | `ident` is required (placeholders such as `hfgcs` are refused) and sent as `ident_user`; our `/status` probe sends a `User-Agent` naming this repo. kiwirecorder's own websocket uses kiwiclient's defaults (it has no User-Agent option) |
| One slot per receiver operator | Process-wide registry keyed by host name and resolved IP; a second channel on the same host/IP is refused |
| One frequency per receiver | `tune()` to a different channel is refused |
| Session cap ~22 min | The slot is released after `session_max_s = 1320` (also passed as `--tlimit`), then we rest ≥ 60 s (with jitter) and rotate |
| Skip small, full, offline receivers | One `/status` GET before connecting: needs `users_max ≥ 4`, a free slot, `offline=no` |
| Proxied endpoints excluded | `*.proxy.kiwisdr.com` skipped unless `allow_proxy = true` |
| Blocklist always honoured | `blocklist` / `blocklist_file` |
| A refusal is an answer | `too_busy`, `ip_limit`, redirect, bad password or being kicked: abandoned at once (`--busy-retries 1`, no retry loop), host retired 60 min |
| No reconnect hammering | Any other failure retires that receiver 30 min; exponential backoff 30 s → 10 min with jitter |
| One session per process | kiwirecorder's own in-process reconnect (after the server ends a session) or redirect is caught on stderr and the process stopped at once; it counts as a failed session for that host |
| No waterfall | Audio only (`--nc`), so no second socket or extra bandwidth |

Not implemented yet (open items): make-before-break rotation (we do
break-before-make, which never holds two slots), a propagation gate that
releases the slot when the band is closed, and automatic receiver selection.

## SoapySDR (`--source soapy`, default)

```toml
source = "soapy"
soapy_args = "driver=airspyhf"
receiver_id = "airspyhf-local"
tuner_agc = false
```

Unchanged from Phase 0: IQ at `iq_rate`, hardware retune per channel with a
settling delay. SoapySDR is imported only when this source is opened.

## File replay (`--source file`)

```bash
# demodulated audio (any rate; mono or first channel)
hfgcs_scanner.py --source file --file rec_11175.wav --channel 11175
# IQ: 2-channel WAV (I=left, Q=right) or raw cf32 / cs16 / cu8
hfgcs_scanner.py --source file --file wide.cf32 --sample-rate 192000 --center-khz 11100 --channel 11175
```

`[file]` keys: `path`, `format` (`auto`, `wav`, `iq-wav`, `cf32`, `cs16`,
`cu8`), `kind` (`auto`/`audio`/`iq`), `sample_rate`, `center_khz` (IQ),
`dial_khz` (audio: refuse other channels), `realtime` (pace at real time),
`loop`, `block_seconds`, `receiver_id` (default `file-<name>`). An IQ file with
`center_khz` can be replayed for any channel inside its span; the source mixes
that channel to baseband. Replay ends at end of file (unless `loop`).
