# HFGCS scanner.
#
# Reads from any receiver backend in hfgcs.sources (local SoapySDR, an
# OpenWebRX server's audio websocket, a KiwiSDR through kiwirecorder, or a
# WAV/IQ file), opens a voice-slot squelch on activity, records USB audio
# with pre-roll until the signal drops for the hang time, and appends one
# JSON line per recording to the catalog.
#
# Optional dependencies (SoapySDR, websockets, kiwiclient) are only loaded
# for the source that needs them, so the DSP and tests run without them.

import argparse
import dataclasses
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

from hfgcs.capture import ChannelCapture, save_recording
from hfgcs.config import SOURCE_NAMES, Channel, Config, load_config
from hfgcs.dsp import power_db
from hfgcs.sources import AudioSource, SourceError, TuneRefused, make_source

READ_TIMEOUT = 0.5


def utc_now():
    return datetime.now(timezone.utc)


def _say(msg: str) -> None:
    """print() that tolerates a closed stdout.

    Under systemd (``KillSignal=SIGINT``) every process in the unit gets the
    signal, including a ``tee`` the unit pipes our output through, so stdout may
    already be a broken pipe when we report a requested stop.
    """
    try:
        print(msg, flush=True)
    except OSError:
        pass


def _settle_stdio() -> None:
    """Flush stdout/stderr; if the pipe is gone, point them at /dev/null.

    Otherwise Python's final flush fails and the exit status becomes 120 even
    though the shutdown was clean.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stream.fileno())
                os.close(devnull)
            except (OSError, ValueError):
                pass


class HFGCSScanner:
    """Drives one source: tune, feed ChannelCapture, save recordings."""

    def __init__(self, cfg: Config, source: AudioSource | None = None):
        self.cfg = cfg
        self.source = source if source is not None else make_source(cfg)
        self.captures: dict[float, ChannelCapture] = {}
        self.channels: list[float] = []
        self.saved: list[dict] = []
        self.refused: dict[float, str] = {}
        self.eof = False
        self.interrupted = False

    # -- setup ------------------------------------------------------------
    def start(self) -> list[float]:
        """Open the source and work out which channels it may tune."""
        self.source.open()
        self.channels, self.refused = [], {}
        for freq in self.cfg.channels_khz:
            reason = self.source.refusal(freq) if hasattr(self.source, "refusal") else None
            if reason is None and not self.source.can_tune(freq):
                reason = "source cannot tune it"
            if reason is None and self.channels and getattr(self.source, "single_channel", False):
                reason = f"{self.source.name} records one channel at a time ({self.channels[0]:g} kHz)"
            if reason is None:
                self.channels.append(freq)
            else:
                self.refused[freq] = reason
                print(f"  skip {freq:g} kHz: {reason}", file=sys.stderr)
        if not self.channels:
            raise TuneRefused(f"{self.source.name}: none of the configured channels can be tuned")
        return self.channels

    def capture_for(self, freq: float, rate: int) -> ChannelCapture:
        cap = self.captures.get(freq)
        # A new receiver (e.g. a KiwiSDR host rotation) gets a new capture so the
        # catalog's receiver_id follows the host that actually made the audio.
        if cap is None or cap.input_rate != rate or cap.receiver_id != self.source.receiver_id:
            cap = ChannelCapture(
                self.cfg,
                freq,
                utc_now(),
                input_kind=self.source.kind,
                input_rate=rate,
                receiver_id=self.source.receiver_id,
                squelch_level=self.source.squelch_level,
            )
            self.captures[freq] = cap
        return cap

    # -- running ----------------------------------------------------------
    def _save(self, rec):
        entry = save_recording(rec, self.cfg.record_dir, self.cfg.catalog_file)
        self.saved.append(entry)
        print(
            f"  -> {entry['file']} ({entry['duration_sec']:.1f}s, "
            f"SNR peak {entry['snr_db_peak']} dB, start {entry['start_utc']}, rx {entry['receiver_id']})"
        )

    def dwell(self, freq: float, seconds: float | None) -> float:
        """Record ``freq`` for ``seconds`` of stream time (None: until the stream ends).

        A recording in progress always runs to its end. Returns the stream
        seconds consumed. Sets ``self.eof`` when the source is exhausted.
        """
        self.source.tune(freq)
        cap = self.capture_for(freq, self.source.sample_rate)
        cap.retune(utc_now())
        elapsed = 0.0
        while seconds is None or elapsed < seconds or cap.active:
            chunk = self.source.read(READ_TIMEOUT)
            if chunk is None:
                self.eof = True
                break
            if chunk.gap or chunk.rate != cap.input_rate:
                for rec in cap.flush():
                    self._save(rec)
                cap = self.capture_for(freq, chunk.rate)
                cap.retune(chunk.t0 or utc_now())
            if len(chunk.samples) == 0:
                continue
            elapsed += len(chunk.samples) / chunk.rate
            for rec in cap.feed(chunk.samples, chunk.level_db):
                self._save(rec)
        for rec in cap.flush():
            self._save(rec)
        return elapsed

    def run(self, max_seconds: float | None = None) -> None:
        print(f"HFGCS scanner: source {self.source.name}, receiver {self.source.receiver_id}")
        print(f"Channels {self.channels} kHz")
        total = 0.0
        try:
            while not self.eof and (max_seconds is None or total < max_seconds):
                for freq in self.channels:
                    print(f"[{utc_now().isoformat()}] {freq:.1f} kHz")
                    if len(self.channels) == 1:
                        # One channel: stay on it (no retune traffic to shared receivers).
                        remaining = None if max_seconds is None else max_seconds - total
                        total += self.dwell(freq, remaining)
                    else:
                        total += self.dwell(freq, self.cfg.dwell_seconds)
                    if self.eof or (max_seconds is not None and total >= max_seconds):
                        break
        except KeyboardInterrupt:
            self.interrupted = True
            _say("\nStopping...")

    def probe(self, seconds: float = 0.8) -> dict[float, float]:
        """Measure the voice slot on each usable channel."""
        levels = {}
        for freq in self.channels:
            self.source.tune(freq)
            cap = self.capture_for(freq, self.source.sample_rate)
            cap.retune(utc_now())
            powers, elapsed = [], 0.0
            while elapsed < seconds:
                chunk = self.source.read(READ_TIMEOUT)
                if chunk is None:
                    break
                if not len(chunk.samples):
                    continue
                elapsed += len(chunk.samples) / chunk.rate
                if self.source.squelch_level == "reported" and chunk.level_db is not None:
                    powers.append(chunk.level_db)
                else:
                    _, p = cap.demod.process(chunk.samples)
                    if not np.isnan(p):
                        powers.append(power_db(p))
            levels[freq] = float(np.median(powers)) if powers else float("nan")
            print(f"  {freq:8.1f} kHz  voice-slot level {levels[freq]:6.1f} dB")
        return levels

    def close(self):
        self.source.close()


def apply_cli(cfg: Config, args) -> Config:
    """Fold command-line overrides into the config."""
    changes: dict = {}
    if args.source:
        changes["source"] = args.source
    if args.channel:
        changes["channels"] = tuple(Channel(float(k)) for k in args.channel)
    if args.receiver_id:
        changes["receiver_id"] = args.receiver_id
    owrx = {}
    if args.url:
        owrx["url"] = args.url
    if args.allow_profile_switch:
        owrx["allow_profile_switch"] = True
    wait = getattr(args, "wait_for_profile", None)
    long_running = (
        hasattr(args, "duration")
        and args.duration is None
        and not getattr(args, "probe", False)
        and not getattr(args, "select_profile", None)
    )
    if wait is None and long_running:
        wait = True  # long-running/service mode: idle until a covering profile is active
    if wait is not None and wait != cfg.openwebrx.wait_for_profile:
        owrx["wait_for_profile"] = bool(wait)
    source = args.source or cfg.source
    # --receiver-id names the receiver in the catalog whatever the backend
    # (the per-source settings otherwise win over the top-level value).
    if args.receiver_id and source == "openwebrx":
        owrx["receiver_id"] = args.receiver_id
    if owrx:
        changes["openwebrx"] = dataclasses.replace(cfg.openwebrx, **owrx)
    kiwi = {}
    if args.kiwi_host:
        kiwi["hosts"] = tuple(args.kiwi_host)
    if args.kiwi_ident:
        kiwi["ident"] = args.kiwi_ident
    if args.receiver_id and source == "kiwisdr":
        kiwi["receiver_id"] = args.receiver_id
    if kiwi:
        changes["kiwisdr"] = dataclasses.replace(cfg.kiwisdr, **kiwi)
    fil = {}
    if args.file:
        fil["path"] = args.file
    if args.file_format:
        fil["format"] = args.file_format
    if args.sample_rate:
        fil["sample_rate"] = args.sample_rate
    if args.center_khz:
        fil["center_khz"] = args.center_khz
    if args.realtime:
        fil["realtime"] = True
    if args.receiver_id and source == "file":
        fil["receiver_id"] = args.receiver_id
    if fil:
        changes["file"] = dataclasses.replace(cfg.file, **fil)
    return dataclasses.replace(cfg, **changes) if changes else cfg


def build_parser():
    p = argparse.ArgumentParser(description="HFGCS scanner / recorder")
    p.add_argument("--config", help="TOML file overriding hfgcs.config defaults")
    p.add_argument(
        "--source", choices=SOURCE_NAMES, help="receiver backend (default: config 'source', i.e. soapy)"
    )
    p.add_argument("--channel", type=float, action="append", help="dial kHz to use (repeatable)")
    p.add_argument("--receiver-id", help="receiver ID written to the catalog")
    p.add_argument("--duration", type=float, help="stop after this many seconds of audio")
    p.add_argument("--probe", action="store_true", help="measure each channel and exit")
    p.add_argument("--probe-seconds", type=float, default=0.8)
    g = p.add_argument_group("openwebrx")
    g.add_argument("--url", help="OpenWebRX base URL")
    g.add_argument(
        "--allow-profile-switch",
        action="store_true",
        help="allow selecting another SDR profile on the shared OpenWebRX (off by default)",
    )
    g.add_argument(
        "--wait-for-profile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="if the channel is not in the active profile, stay connected and idle until the owner "
        "selects a covering profile, and stop/idle again if it moves away (never retunes). "
        "Default: on when --duration is not given, off for one-shot runs (which exit 2 instead)",
    )
    g.add_argument(
        "--select-profile",
        metavar="NAME",
        help="one-shot: select this OpenWebRX profile (id, name or name suffix such as "
        "'HFGCS 8992 kHz') exactly once using the unlock key from the environment, then exit",
    )
    g.add_argument(
        "--unlock-key-env",
        default="OPENWEBRX_MAGIC_KEY",
        metavar="VAR",
        help="environment variable holding the OpenWebRX+ unlock (magic) key for --select-profile "
        "(default: %(default)s). The key is never printed or logged",
    )
    g = p.add_argument_group("kiwisdr")
    g.add_argument("--kiwi-host", action="append", help="host:port (repeatable, tried in order)")
    g.add_argument("--kiwi-ident", help="name shown in the Kiwi user list")
    g = p.add_argument_group("file")
    g.add_argument("--file", help="WAV or raw IQ file to replay")
    g.add_argument("--file-format", choices=["auto", "wav", "iq-wav", "cf32", "cs16", "cu8"])
    g.add_argument("--sample-rate", type=int, help="raw IQ sample rate")
    g.add_argument("--center-khz", type=float, help="IQ recording centre frequency")
    g.add_argument("--realtime", action="store_true", help="pace replay at real time")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def select_profile(cfg: Config, name: str, key_env: str) -> int:
    """``--select-profile``: one deliberate profile switch, then exit."""
    from hfgcs.sources.openwebrx import OpenWebRXSource

    key = os.environ.get(key_env, "")
    if not key:
        print(
            f"Refusing --select-profile: environment variable {key_env} is not set "
            "(profile switching on this server needs the unlock key)",
            file=sys.stderr,
        )
        return 1
    src = OpenWebRXSource(cfg.openwebrx)
    try:
        src.open()
        before = src.active_profile_id()
        switched = src.select_profile_once(name, key)
        after = src.active_profile_id()
        done = "yes" if switched else "no (already active)"
        print(f"profile before: {before}; after: {after}; switched: {done}")
        print(f"active window: {src.window.label}")
        return 0
    except SourceError as exc:
        print(f"--select-profile failed: {exc}", file=sys.stderr)
        return 3
    finally:
        del key
        src.close()


def main(argv=None):
    """Exit status: 0 on a normal end or a requested stop (SIGINT), non-zero on errors."""
    try:
        code = _main(argv)
    except KeyboardInterrupt:
        # SIGINT outside the capture loop (starting up or closing the source) is
        # still a requested stop, not a failure: systemd should not log "Failed".
        code = 0
    _settle_stdio()
    return code


def _main(argv=None):
    args = build_parser().parse_args(argv)
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    if args.select_profile:
        try:
            cfg = apply_cli(load_config(args.config), args)
        except ValueError as exc:
            print(f"Bad configuration: {exc}", file=sys.stderr)
            return 1
        return select_profile(cfg, args.select_profile, args.unlock_key_env)
    scanner = None
    try:
        cfg = apply_cli(load_config(args.config), args)
        scanner = HFGCSScanner(cfg)
        scanner.start()
    except (SourceError, ValueError) as exc:
        print(f"Could not start the {args.source or 'configured'} source: {exc}", file=sys.stderr)
        if (args.source or "soapy") == "soapy":
            print("If OpenWebRX is using the radio, use --source openwebrx instead.", file=sys.stderr)
        if scanner is not None:
            try:
                scanner.close()
            except Exception:
                pass
        return 2 if isinstance(exc, TuneRefused) else 1
    try:
        if args.probe:
            scanner.probe(args.probe_seconds)
        else:
            scanner.run(args.duration)
    except SourceError as exc:
        print(f"Source failed: {exc}", file=sys.stderr)
        return 1
    finally:
        scanner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
