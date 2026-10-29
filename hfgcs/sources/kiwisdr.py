"""KiwiSDR source, through jks-prv/kiwiclient's ``kiwirecorder.py``.

kiwiclient has no top-level license file (GitHub reports none) and parts of
it are GPL-2.0-or-later, so it is **not vendored or imported**: we run
``kiwirecorder.py --nc --nc-wav`` as a separate program and read the WAV
stream it writes to stdout. Install it yourself (``git clone
https://github.com/jks-prv/kiwiclient``) and point ``[kiwisdr] kiwirecorder``
or ``$KIWIRECORDER`` at ``kiwirecorder.py``.

A Kiwi slot is a volunteer's receiver. The etiquette rules (docs/sources.md,
adapted from Chipensaw/HFGCS-Monitor-KiwiSDR's ETIQUETTE.md and the Signal
Ops coverage survey) are enforced here, not left to flags:

* nothing is contacted until ``enabled = true`` and a real ``ident`` are set;
* one slot per receiver *operator* (host name and resolved IP), process-wide;
* one frequency per receiver (``tune`` to another channel is refused);
* sessions capped at ~22 min, then the slot is released and we rest/rotate;
* receivers with fewer than 4 slots, full, offline, proxied
  (``proxy.kiwisdr.com``) or blocklisted are skipped;
* ``too_busy`` / ``ip_limit`` / redirect / bad password / kicked: abandon at
  once, no retry, host retired for an hour; other failures retire it 30 min;
* exponential backoff with jitter between attempts;
* one session per kiwirecorder process: kiwirecorder reconnects by itself after
  the server ends a session (see ``SESSION_END_PATTERNS``), so we run it at
  ``--log info``, watch its stderr, and stop it on the first such message;
  the session counts as failed toward the host's retirement and backoff;
* every connection identifies itself (``ident_user`` via ``-u``; our HTTP
  ``/status`` probe sends a User-Agent).

The audio comes out of the Kiwi's AGC, so its level says little about signal
strength. With ``[kiwisdr] rssi = true`` (the default) kiwirecorder also runs with
``--progress``, which prints the Kiwi's S-meter (RSSI, measured before the AGC) for
every audio block on stderr; each block we hand on carries the median of the last
few readings as ``level_db`` for the v2 squelch (docs/squelch.md).
"""

from __future__ import annotations

import codecs
import logging
import os
import queue
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import KiwiSettings
from .base import BaseSource, Chunk, SourceError

log = logging.getLogger(__name__)

# Generic values that identify nobody. A project-style name such as "HFGCS-Monitor"
# is accepted as-is; no callsign suffix is required.
PLACEHOLDER_IDENTS = {"", "hfgcs", "changeme", "your-callsign", "anonymous", "user"}

# kiwirecorder / kiwiclient messages that mean "the sysop said no".
REFUSAL_PATTERNS = [
    (re.compile(r"too.?busy|client slots taken", re.I), "too_busy"),
    (re.compile(r"no multiple connections|ip.?limit", re.I), "ip_limit"),
    (re.compile(r"redirect", re.I), "redirect"),
    (re.compile(r"bad password|badp", re.I), "bad_password"),
    (re.compile(r"inactivity|time.?limit|kicked|reconnecting after|server closed the connection", re.I),
     "kicked"),
    (re.compile(r"server is down|KiwiDownError", re.I), "down"),
]

# kiwirecorder messages that mean the process is about to open another session by
# itself. Checked against jks-prv/kiwiclient v1.11 (commit 4eb733e), kiwi/worker.py:
#   KiwiServerTerminatedConnection (the Kiwi ended a running session), logged at INFO:
#     "<host>:<port> server closed the connection cleanly. Reconnecting after 5 seconds"
#     "<host>:<port> server closed the connection unexpectedly. Reconnecting after 5 seconds"
#   then it waits 5 s and reconnects in the same process with no limit
#   (--connect-retries / --busy-retries do not apply to this path);
#   KiwiTooBusyError (only if --busy-retries were > 1), WARNING:
#     "<host>:<port> Too busy now. Reconnecting after <n> seconds"
#   KiwiRedirectError, WARNING (then it connects to the other Kiwi):
#     "<host>:<port> Too busy now. Redirecting to <host>:<port>"
#   connect failure (only retried if --connect-retries were > 1), WARNING:
#     "Failed to connect, sleeping and reconnecting error='...'"
# kiwirecorder has no supported switch that disables the reconnect: --no-api does
# skip it, but it is a test option that also stops sending the ident, frequency and
# mode, so it cannot be used.
SESSION_END_PATTERNS = [
    re.compile(r"server closed the connection", re.I),
    re.compile(r"reconnecting after \d+ seconds", re.I),
    re.compile(r"redirecting to", re.I),
    re.compile(r"sleeping and reconnecting", re.I),
]
KIWIRECORDER_LOG_LEVEL = "info"  # SESSION_END_PATTERNS' first line is logged at INFO
# kiwirecorder --progress (kiwiclient v1.11, KiwiNetcat): "\rBlock: %08x, RSSI: %6.1f %s " per block.
RSSI_LINE = re.compile(r"^Block:\s*[0-9a-fA-F]+,\s*RSSI:\s*(-?\d+(?:\.\d+)?)")
RSSI_RANGE_DBM = (-170.0, 30.0)  # what a Kiwi S-meter can report; anything else is ignored
RSSI_MEDIAN_OF = 5  # readings per level (~0.2 s): one static crash does not move it
RSSI_STALE_BLOCKS = 24  # blocks (~2 s) without a new reading: no level until one arrives
MAX_STDERR_LINE = 4096  # progress lines end in "\r" only; never let one grow without bound
STOP_GRACE_S = 2.0  # SIGINT grace before SIGTERM; shorter than kiwirecorder's 5 s reconnect wait

# Process-wide bookkeeping shared by every KiwiSource instance.
_LOCK = threading.Lock()
_ACTIVE: dict[str, str] = {}  # operator key (host name or IP) -> "host:port" using it
_RETIRED: dict[str, float] = {}  # "host:port" or "host" -> monotonic time it may be used again


def reset_etiquette_state() -> None:
    """Forget active slots and retirements (tests only)."""
    with _LOCK:
        _ACTIVE.clear()
        _RETIRED.clear()


@dataclass(frozen=True)
class KiwiHost:
    host: str
    port: int = 8073

    @classmethod
    def parse(cls, text: str) -> KiwiHost:
        text = text.strip()
        for prefix in ("http://", "https://"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        text = text.rstrip("/")
        if ":" in text:
            host, port = text.rsplit(":", 1)
            return cls(host.lower(), int(port))
        return cls(text.lower())

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


def fetch_status(host: KiwiHost, user_agent: str, timeout: float) -> dict[str, str]:
    """Read the Kiwi's public ``/status`` page (one small HTTP GET)."""
    req = urllib.request.Request(f"http://{host}/status", headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read(65536).decode("utf-8", "replace")
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def resolve(host: str) -> str | None:
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


# Kiwi audio runs at 12 kHz (20.25 kHz on some 3-channel builds); kiwirecorder writes the
# measured rate as an integer, e.g. 11999 or 12001, which no small rational resampler handles.
KIWI_NOMINAL_RATES = (12000, 20250)


def nominal_rate(rate: int, tolerance: float = 0.005) -> int:
    """Snap a measured Kiwi rate within ``tolerance`` of a nominal one to it."""
    for nominal in KIWI_NOMINAL_RATES:
        if abs(rate - nominal) <= tolerance * nominal:
            return nominal
    return rate


def parse_wav_header(read) -> tuple[int, int]:
    """Read a (streaming) WAV header with ``read(n)``; return (rate, channels)."""
    riff = read(12)
    if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
        raise SourceError("kiwirecorder did not start a WAV stream")
    rate = channels = 0
    while True:
        head = read(8)
        if len(head) < 8:
            raise SourceError("kiwirecorder WAV stream ended in the header")
        cid, size = head[:4], struct.unpack("<I", head[4:])[0]
        if cid == b"data":
            if not rate:
                raise SourceError("WAV stream has no fmt chunk")
            return rate, channels
        body = read(size + (size & 1)) if cid == b"fmt " else read(size)
        if cid == b"fmt ":
            _, channels, rate = struct.unpack("<HHI", body[:8])


class KiwiSession:
    """One running kiwirecorder process feeding a queue of audio blocks.

    The process carries exactly one Kiwi session: stderr is read continuously (so
    the pipe never fills) and the first message saying kiwirecorder is about to
    reconnect by itself stops the process (SIGINT, then SIGTERM after
    ``stop_grace_s``) before it can open a second session.
    """

    def __init__(self, proc, block_samples: int = 1024, stop_grace_s: float | None = None):
        self.proc = proc
        self.q: queue.Queue = queue.Queue(maxsize=512)
        self.rate = 0
        self.channels = 1
        self.reason: str | None = None
        self.session_end: str | None = None  # kiwirecorder line that ended the session
        self.stderr_tail: list[str] = []
        self.rssi: deque[float] = deque(maxlen=64)  # recent --progress RSSI readings (dBm)
        self.rssi_count = 0  # readings so far (to tell a stale level from a fresh one)
        self._rssi_lock = threading.Lock()  # the stderr thread appends, read() copies
        self.header_ready = threading.Event()
        self.ended = threading.Event()
        self.stderr_done = threading.Event()
        self.stopping = False  # set once we stop the process (ours or after session_end)
        self.stop_grace_s = STOP_GRACE_S if stop_grace_s is None else stop_grace_s
        self._stop_lock = threading.Lock()
        self.block_bytes = 2 * block_samples
        threading.Thread(target=self._stdout, daemon=True).start()
        if proc.stderr is not None:
            threading.Thread(target=self._stderr, daemon=True).start()
        else:
            self.stderr_done.set()

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = self.proc.stdout.read(n - len(buf))
            if not part:
                break
            buf += part
        return buf

    def _stdout(self) -> None:
        try:
            rate, self.channels = parse_wav_header(self._read_exact)
            self.rate = nominal_rate(rate)
            self.header_ready.set()
            frame = 2 * self.channels
            while True:
                data = self._read_exact(self.block_bytes * self.channels)
                data = data[: len(data) // frame * frame]
                if data:
                    pcm = np.frombuffer(data, "<i2").reshape(-1, self.channels)[:, 0]
                    self.q.put(pcm.astype(np.float32) / 32768.0)
                if len(data) < self.block_bytes * self.channels:
                    break
        except SourceError as exc:
            self.reason = self.reason or f"stream: {exc}"
        finally:
            self.ended.set()
            self.header_ready.set()

    def _stderr(self) -> None:
        # Keep reading to EOF even after we decide to stop, so the child never blocks on a
        # full stderr pipe. Lines end in "\n" or, for --progress, in "\r" only.
        stream = self.proc.stderr
        read = getattr(stream, "read1", None)
        decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
        buf = ""
        try:
            chunks = iter(lambda: read(4096), b"") if read is not None else iter(stream)
            for data in chunks:
                buf += decode(data) if isinstance(data, bytes) else data
                *lines, buf = re.split(r"[\r\n]", buf)
                for line in lines:
                    self._stderr_piece(line)
                if len(buf) > MAX_STDERR_LINE:
                    self._stderr_piece(buf)
                    buf = ""
            self._stderr_piece(buf)
        except (OSError, ValueError):
            pass
        finally:
            self.stderr_done.set()

    def _stderr_piece(self, line: str) -> None:
        line = line.strip()[:MAX_STDERR_LINE]
        if not line:
            return
        m = RSSI_LINE.match(line)
        if m:
            value = float(m.group(1))
            if RSSI_RANGE_DBM[0] <= value <= RSSI_RANGE_DBM[1]:
                with self._rssi_lock:
                    self.rssi.append(value)
                    self.rssi_count += 1
            # A progress line ends in a space, not a newline: the next log line (e.g. the
            # server ending the session) is glued onto it and must still be seen.
            line = line[m.end():].strip()
            if not line:
                return
        self._stderr_line(line)

    def recent_rssi(self, n: int) -> tuple[int, list[float]]:
        """(readings so far, the last ``n`` readings)."""
        with self._rssi_lock:
            return self.rssi_count, list(self.rssi)[-n:]

    def _stderr_line(self, line: str) -> None:
        self.stderr_tail = (self.stderr_tail + [line])[-20:]
        if self.stopping:
            return  # anything printed while we shut it down is not the server's doing
        for pattern, reason in REFUSAL_PATTERNS:
            if pattern.search(line) and self.reason is None:
                self.reason = reason
        if self.session_end is None and any(p.search(line) for p in SESSION_END_PATTERNS):
            self.session_end = line
            self.reason = self.reason or "kiwirecorder about to reconnect in-process"
            log.warning("kiwisdr: kiwirecorder session ended (%s); stopping it before it reconnects", line)
            self.stopping = True
            threading.Thread(target=self.stop, daemon=True).start()

    def wait_stderr(self, timeout: float = 2.0) -> None:
        """Let the stderr reader catch up after the process ends (for ``reason``)."""
        self.stderr_done.wait(timeout)

    def stop(self) -> None:
        """SIGINT, then SIGTERM after ``stop_grace_s``, then SIGKILL; safe to call twice."""
        with self._stop_lock:
            self.stopping = True
            if self.proc.poll() is not None:
                return
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(self.stop_grace_s)
                return
            except subprocess.TimeoutExpired:
                pass
            except (OSError, ValueError):
                pass
            try:
                self.proc.terminate()
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            except (OSError, ValueError):
                pass


class KiwiSource(BaseSource):
    """USB audio from one KiwiSDR slot at a time, rotating through ``hosts``."""

    name = "kiwisdr"
    kind = "audio"
    squelch_level = "slot"
    single_channel = True

    def __init__(
        self,
        settings: KiwiSettings,
        *,
        popen=None,
        status=None,
        resolver=None,
        clock=None,
        sleep=None,
    ):
        self.settings = settings
        super().__init__(settings.receiver_id)
        self.hosts = [KiwiHost.parse(h) for h in settings.hosts]
        self._popen = popen or subprocess.Popen
        self._status = status or fetch_status
        self._resolve = resolver or resolve
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.session: KiwiSession | None = None
        self.current: KiwiHost | None = None
        self._keys: list[str] = []
        self._next_host = 0
        self._failures = 0
        self._not_before = 0.0
        self._session_start = 0.0
        self._rate = 12_000
        self._gap = False
        self._last_audio = 0.0  # clock time of the last block with any non-zero sample
        self.log: list[str] = []  # human-readable decisions (also logged)
        self._down_since: float | None = None  # a session ended; cleared when audio resumes
        self._last_audio_at: float | None = None  # clock time of the last block received
        self._blocked = self._load_blocklist()
        self._rssi_seen = 0  # session.rssi_count at the last block handed on
        self._rssi_stale = 0  # blocks handed on since a new RSSI reading arrived
        self._rssi_blocks = 0  # blocks handed on in this session
        self._rssi_warned = False

    @property
    def level_reported(self) -> bool:
        """Blocks carry the Kiwi's RSSI (--progress) as ``level_db``."""
        return bool(self.settings.rssi)

    def _rssi_level(self, session: KiwiSession) -> float | None:
        """Median of the latest RSSI readings, or None if none arrived lately."""
        if not self.settings.rssi:
            return None
        self._rssi_blocks += 1
        count, recent = session.recent_rssi(RSSI_MEDIAN_OF)
        if count != self._rssi_seen:
            self._rssi_seen, self._rssi_stale = count, 0
        else:
            self._rssi_stale += 1
        if recent and self._rssi_stale <= RSSI_STALE_BLOCKS:
            return float(np.median(recent))
        if not self._rssi_warned and self._rssi_blocks > 2 * RSSI_STALE_BLOCKS:
            self._rssi_warned = True
            log.warning("kiwisdr: no RSSI from kiwirecorder --progress for %s; the v2 squelch has "
                        "no level until readings arrive", self.current)
        return None

    # -- policy ------------------------------------------------------------------
    def _note(self, msg: str) -> None:
        self.log.append(msg)
        log.info("kiwisdr: %s", msg)

    def _load_blocklist(self) -> set[str]:
        entries = {e.strip().lower() for e in self.settings.blocklist if e.strip()}
        if self.settings.blocklist_file and os.path.exists(self.settings.blocklist_file):
            with open(self.settings.blocklist_file) as fh:
                for line in fh:
                    entry = line.split("#", 1)[0].split()
                    if entry:
                        entries.add(entry[0].lower())
        return entries

    def check_authorised(self) -> None:
        s = self.settings
        if not s.enabled:
            raise SourceError(
                "KiwiSDR source is disabled: set [kiwisdr] enabled = true by hand once you have "
                "read docs/sources.md (installing is not authorising contact with volunteers' receivers)"
            )
        if s.ident.strip().lower() in PLACEHOLDER_IDENTS:
            raise SourceError(
                "[kiwisdr] ident must identify you (callsign or name); it is shown to the sysop"
            )
        if not self.hosts:
            raise SourceError("[kiwisdr] hosts is empty")

    def skip_reason(self, host: KiwiHost) -> str | None:
        """Why this receiver must not be used right now (without contacting it)."""
        if host.host in self._blocked or str(host) in self._blocked:
            return "blocklisted"
        if host.host.endswith("proxy.kiwisdr.com") and not self.settings.allow_proxy:
            return "proxied endpoint (proxy.kiwisdr.com bandwidth is donated)"
        now = self._clock()
        with _LOCK:
            for key in (str(host), host.host):
                if _RETIRED.get(key, 0) > now:
                    return f"retired for {(_RETIRED[key] - now) / 60:.0f} more min"
            if host.host in _ACTIVE:
                return f"already using a slot on this operator ({_ACTIVE[host.host]})"
        return None

    def _retire(self, host: KiwiHost, seconds: float, whole_host: bool) -> None:
        with _LOCK:
            _RETIRED[host.host if whole_host else str(host)] = self._clock() + seconds

    def _claim(self, host: KiwiHost) -> bool:
        keys = [host.host]
        ip = self._resolve(host.host)
        if ip:
            keys.append(ip)
        with _LOCK:
            for key in keys:
                if key in _ACTIVE:
                    return False
            for key in keys:
                _ACTIVE[key] = str(host)
        self._keys = keys
        return True

    def _release(self) -> None:
        with _LOCK:
            for key in self._keys:
                _ACTIVE.pop(key, None)
        self._keys = []

    def status_ok(self, host: KiwiHost) -> str | None:
        """Read /status; return a reason to skip, or None."""
        if not self.settings.check_status:
            return None
        try:
            st = self._status(host, self.settings.user_agent, self.settings.status_timeout_s)
        except Exception as exc:
            return f"/status unreachable ({exc})"
        try:
            users, users_max = int(st.get("users", "0")), int(st.get("users_max", "0"))
        except ValueError:
            users, users_max = 0, 0
        if st.get("offline", "no") != "no" or st.get("status", "active") != "active":
            return "offline"
        if users_max < self.settings.min_users_max:
            return f"only {users_max} slots (< {self.settings.min_users_max})"
        if users >= users_max:
            return f"full ({users}/{users_max})"
        return None

    # -- tuning ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def receiver_id(self) -> str:
        if self._receiver_id:
            return self._receiver_id
        host = self.current or (self.hosts[0] if self.hosts else None)
        return f"kiwi-{host.host}-{host.port}" if host else "kiwisdr"

    def refusal(self, dial_khz: float) -> str | None:
        fixed = self.settings.channel_khz or self.dial_khz
        if fixed and abs(dial_khz - fixed) > 0.01:
            return f"one frequency per KiwiSDR slot (this source records {fixed:g} kHz)"
        return None

    def tune(self, dial_khz: float) -> None:
        same = self.dial_khz is not None and abs(dial_khz - self.dial_khz) < 0.01
        super().tune(dial_khz)
        if not same and self.session is not None:
            self._end_session("retune")

    # -- sessions ------------------------------------------------------------------
    def kiwirecorder_path(self) -> str:
        path = self.settings.kiwirecorder or os.environ.get("KIWIRECORDER") or shutil.which("kiwirecorder.py")
        if not path:
            raise SourceError(
                "kiwirecorder.py not found: clone https://github.com/jks-prv/kiwiclient and set "
                "[kiwisdr] kiwirecorder or $KIWIRECORDER"
            )
        return path

    def build_command(self, host: KiwiHost, dial_khz: float) -> list[str]:
        s = self.settings
        cmd = [
            s.python or sys.executable, "-u", self.kiwirecorder_path(),
            "-s", host.host, "-p", str(host.port),
            "-f", f"{dial_khz:g}", "-m", "usb",
            "-L", f"{s.low_cut_hz:g}", "-H", f"{s.high_cut_hz:g}",
            "-u", s.ident,
            "--nc", "--nc-wav",
            "--tlimit", str(int(s.session_max_s)),
            "--connect-retries", "1", "--connect-timeout", "10",
            "--busy-retries", "1", "-k", "30",
            # INFO shows kiwirecorder's in-process reconnect line (SESSION_END_PATTERNS).
            "--log", KIWIRECORDER_LOG_LEVEL,
        ]
        if s.rssi:
            cmd.append("--progress")  # per-block RSSI on stderr (does not touch the audio stream)
        if s.password:
            cmd += ["--pw", s.password]
        if s.tlimit_password:
            cmd += ["--tlimit-pw", s.tlimit_password]
        return cmd

    def open(self) -> None:
        self.check_authorised()
        super().open()
        if self.settings.channel_khz and self.dial_khz is None:
            self.dial_khz = float(self.settings.channel_khz)

    def close(self) -> None:
        self._end_session("close", rest=False)
        super().close()

    def _start_session(self) -> bool:
        """Try the hosts in order; True once one is streaming."""
        tried = 0
        while tried < len(self.hosts):
            host = self.hosts[self._next_host % len(self.hosts)]
            self._next_host += 1
            tried += 1
            reason = self.skip_reason(host)
            if reason is None:
                reason = self.status_ok(host)
                if reason is not None:
                    self._retire(host, self.settings.retire_after_failure_s, whole_host=False)
            if reason is not None:
                self._note(f"skip {host}: {reason}")
                continue
            if not self._claim(host):
                self._note(f"skip {host}: same operator already in use")
                continue
            self.current = host
            cmd = self.build_command(host, self.dial_khz)
            self._note(f"connect {host} {self.dial_khz:g} kHz USB as {self.settings.ident!r}")
            try:
                proc = self._popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except OSError as exc:
                self._release()
                raise SourceError(f"could not run kiwirecorder: {exc}") from exc
            session = KiwiSession(proc)
            try:
                session.header_ready.wait(30)
            except BaseException:
                # e.g. SIGINT while connecting: do not leave the process holding the slot
                session.stop()
                self._release()
                raise
            if session.rate and not session.session_end:
                self.session = session
                self._rssi_seen = self._rssi_stale = self._rssi_blocks = 0
                self._rssi_warned = False
                self._rate = session.rate
                self._session_start = self._last_audio = self._clock()
                self._gap = True
                self.connection_event(
                    "connect", f"streaming from {host} at {self._rate} Hz", host=str(host),
                    rate=self._rate,
                )
                return True
            self._session_failed(host, session)
            self.connection_event(
                "connect_failed", f"{host}: {session.reason or 'connect failed'}", logging.WARNING,
                host=str(host), error=session.reason or "connect failed",
            )
        return False

    def _session_failed(self, host: KiwiHost, session: KiwiSession) -> None:
        session.stop()
        session.wait_stderr()
        self._release()
        reason = session.reason or "connect failed"
        if session.session_end:
            self._note(
                f"{host}: kiwirecorder: {session.session_end!r}; process stopped (one session per process)"
            )
        refusal = reason in ("too_busy", "ip_limit", "redirect", "bad_password", "kicked")
        seconds = self.settings.retire_after_refusal_s if refusal else self.settings.retire_after_failure_s
        self._retire(host, seconds, whole_host=refusal)
        self._failures += 1
        self._note(f"{host}: {reason}; abandoned, retired {seconds / 60:.0f} min (no retry)")

    def _lost(self, host, why: str) -> None:
        """A running session ended (not by us closing the source)."""
        if self._down_since is None:
            self._down_since = self._clock()
        self.connection_event("disconnect", f"{host}: {why}", logging.WARNING, host=str(host), reason=why)

    def _end_session(self, why: str, rest: bool = True) -> None:
        session, self.session = self.session, None
        if session is None:
            return
        session.stop()
        self._release()
        self._note(f"release {self.current}: {why}")
        if rest:
            self._lost(self.current, why)
        else:
            self.connection_event("disconnect", f"{self.current}: {why}", host=str(self.current), reason=why)
        if rest:
            rest_s = self.settings.rest_between_sessions_s * random.uniform(1.0, 1.3)
            self._not_before = self._clock() + rest_s
        self._gap = True

    def _backoff(self) -> None:
        s = self.settings
        delay = min(s.backoff_max_s, s.backoff_initial_s * 2 ** max(0, self._failures - 1))
        self._not_before = self._clock() + delay * random.uniform(0.8, 1.2)
        self._note(f"no receiver available; next attempt in {delay:.0f} s")

    def read(self, timeout: float | None = None) -> Chunk | None:
        if not self.is_open:
            raise SourceError("KiwiSDR source is not open")
        if self.dial_khz is None:
            raise SourceError("tune() the KiwiSDR source first")
        timeout = 1.0 if timeout is None else timeout
        if self.session is None:
            wait = self._not_before - self._clock()
            if wait > 0:
                self._sleep(min(wait, timeout))
                return Chunk(np.zeros(0, np.float32), self._rate)
            if not self._start_session():
                self._backoff()
                return Chunk(np.zeros(0, np.float32), self._rate)
        session = self.session
        if self._clock() - self._session_start >= self.settings.session_max_s:
            self._end_session(f"session cap {self.settings.session_max_s / 60:.0f} min reached")
            return Chunk(np.zeros(0, np.float32), self._rate)
        try:
            audio = session.q.get(timeout=timeout)
        except queue.Empty:
            if not session.ended.is_set() and self._stalled(session):
                return Chunk(np.zeros(0, np.float32), self._rate)
            if session.ended.is_set():
                host = self.current
                self.session = None
                ran = self._clock() - self._session_start
                session.wait_stderr()  # a session-end line may still be in the pipe
                self._lost(host, session.reason or session.session_end or f"stream ended after {ran:.0f} s")
                if session.reason or session.session_end or ran < 60:
                    self._session_failed(host, session)
                    self._backoff()
                else:
                    session.stop()
                    self._release()
                    self._note(f"{host}: stream ended after {ran:.0f} s")
                    self._not_before = self._clock() + self.settings.rest_between_sessions_s
                self._gap = True
            return Chunk(np.zeros(0, np.float32), self._rate)
        if np.any(audio):
            self._last_audio = self._clock()
        elif self._stalled(session):
            return Chunk(np.zeros(0, np.float32), self._rate)
        self._failures = 0
        now = self._clock()
        if self._down_since is not None and len(audio):
            since = self._last_audio_at if self._last_audio_at is not None else self._down_since
            gap_s = round(now - since, 3)
            self.connection_event(
                "audio_resumed", f"audio from {self.current} resumed after a {gap_s:.1f} s gap",
                host=str(self.current), gap_s=gap_s, down_s=round(now - self._down_since, 3),
            )
            self._down_since = None
        self._last_audio_at = now
        gap, self._gap = self._gap, False
        return Chunk(audio, self._rate, gap=gap, level_db=self._rssi_level(session))

    def _stalled(self, session: KiwiSession) -> bool:
        """Watchdog: drop a connected session that gives no (or only all-zero) samples."""
        limit = self.settings.stall_timeout_s
        if not limit or self._clock() - self._last_audio < limit:
            return False
        host = self.current
        self.session = None
        session.reason = session.reason or f"no audio for {limit:.0f} s (stalled or silent receiver)"
        self._lost(host, session.reason)
        self._session_failed(host, session)
        self._backoff()
        self._gap = True
        return True
