"""Run-state and connection-event files written by the recorders.

The deploy script (``scripts/hfgcs_deploy.py``) must never restart a recorder
while it is capturing, and it waits for a restarted KiwiSDR unit to stream
before it touches the next one. The recorder tells it both through a small
JSON **state file**, replaced atomically (write to a temp file, then
``os.replace``) so a reader never sees half a file:

``$XDG_RUNTIME_DIR/hfgcs/<name>.state``::

    {"schema": 1, "name": "hfgcs-8992", "pid": 1234,
     "state": "idle" | "capturing" | "starting" | "stopped",
     "since_utc": "2026-10-26T23:10:04.120Z",   # when ``state`` began
     "receiver": "owrx-hilltop", "channel_khz": 8992.0,
     "started_utc": "...",                       # process start
     "last_audio_utc": "..." | null,              # last block with samples (<= 5 s stale)
     "updated_utc": "...",
     "last_connection_event": {...} | null}

``<name>`` is the systemd unit (``hfgcs-8992``, ``hfgcs-kiwi-11175``) found in
``/proc/self/cgroup``, or ``$HFGCS_STATE_NAME`` / ``--state-name``.

Connection events (connect, disconnect, reconnect attempts, audio resumed
after a gap) are appended as JSON lines to ``logs/<name>-connection.log`` so
audio gaps can be traced later. Stdlib only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = 1
STATES = ("starting", "idle", "capturing", "stopped")
HEARTBEAT_S = 5.0  # last_audio_utc / updated_utc are refreshed at most this often


def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without a zone: {text!r}")
    return dt.astimezone(timezone.utc)


def runtime_dir() -> Path:
    """``$HFGCS_RUNTIME_DIR``, else ``$XDG_RUNTIME_DIR/hfgcs``, else ``/run/user/<uid>/hfgcs``."""
    if os.environ.get("HFGCS_RUNTIME_DIR"):
        return Path(os.environ["HFGCS_RUNTIME_DIR"])
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base:
        guess = Path(f"/run/user/{os.getuid()}")
        base = str(guess) if guess.is_dir() else os.path.join(tempfile.gettempdir(), f"hfgcs-{os.getuid()}")
    return Path(base) / "hfgcs"


def state_path(name: str, directory: Path | None = None) -> Path:
    return (directory or runtime_dir()) / f"{safe_name(name)}.state"


def safe_name(name: str) -> str:
    name = name[: -len(".service")] if name.endswith(".service") else name
    return re.sub(r"[^A-Za-z0-9._@-]+", "-", name).strip("-") or "hfgcs"


def systemd_unit_name(cgroup_file: str = "/proc/self/cgroup") -> str | None:
    """The ``*.service`` this process runs in (systemd puts it in the cgroup path)."""
    try:
        text = Path(cgroup_file).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        path = line.rsplit(":", 1)[-1]
        for part in reversed(path.split("/")):
            if part.endswith(".service") and not part.startswith("user@"):
                return part[: -len(".service")]
    return None


def default_name(source: str = "", channels: list[float] | tuple[float, ...] = ()) -> str:
    """``$HFGCS_STATE_NAME``, else the systemd unit, else ``<source>-<channel>``."""
    if os.environ.get("HFGCS_STATE_NAME"):
        return safe_name(os.environ["HFGCS_STATE_NAME"])
    unit = systemd_unit_name()
    if unit:
        return safe_name(unit)
    chan = f"-{channels[0]:g}" if len(channels) == 1 else ""
    return safe_name(f"hfgcs-{source or 'scanner'}{chan}")


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_state(path: Path) -> dict | None:
    """The state file as a dict, or None if missing or unreadable."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class RunState:
    """Writes the recorder's state file. Every write is atomic; failures are logged, never raised."""

    def __init__(self, name: str, path: Path | None = None, *, clock=None, now=None, pid: int | None = None):
        self.name = safe_name(name)
        self.path = Path(path) if path else state_path(self.name)
        self._clock = clock or time.monotonic
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        started = utc_iso(self._now())
        self.data: dict = {
            "schema": SCHEMA,
            "name": self.name,
            "pid": os.getpid() if pid is None else pid,
            "state": "starting",
            "since_utc": started,
            "started_utc": started,
            "receiver": "",
            "channel_khz": None,
            "last_audio_utc": None,
            "updated_utc": started,
            "last_connection_event": None,
        }
        self._last_write = float("-inf")
        self._warned = False
        self.write()

    @property
    def state(self) -> str:
        return self.data["state"]

    def write(self) -> None:
        with self._lock:
            self.data["updated_utc"] = utc_iso(self._now())
            try:
                write_json_atomic(self.path, self.data)
            except OSError as exc:
                if not self._warned:
                    log.warning("could not write state file %s: %s", self.path, exc)
                    self._warned = True
            self._last_write = self._clock()

    def set(self, state: str, *, receiver: str | None = None, channel_khz: float | None = None) -> None:
        """Record a state change (written at once); same state: only receiver/channel updates."""
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, not {state!r}")
        changed = False
        if state != self.data["state"]:
            self.data["state"] = state
            self.data["since_utc"] = utc_iso(self._now())
            changed = True
        if receiver is not None and receiver != self.data["receiver"]:
            self.data["receiver"] = receiver
            changed = True
        if channel_khz is not None and channel_khz != self.data["channel_khz"]:
            self.data["channel_khz"] = float(channel_khz)
            changed = True
        if changed:
            self.write()

    def audio(self) -> None:
        """Audio arrived: refresh ``last_audio_utc`` (written at most every HEARTBEAT_S)."""
        first = self.data["last_audio_utc"] is None
        self.data["last_audio_utc"] = utc_iso(self._now())
        if first or self._clock() - self._last_write >= HEARTBEAT_S:
            self.write()

    def connection_event(self, event: dict) -> None:
        self.data["last_connection_event"] = event
        self.write()


class ConnectionEvents:
    """Logs connection events and appends them as JSON lines (and to the state file)."""

    def __init__(self, path: Path | None, *, name: str = "", state: RunState | None = None, now=None):
        self.path = Path(path) if path else None
        self.name = name
        self.state = state
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._warned = False

    def emit(self, event: str, **fields) -> dict:
        rec = {"utc": utc_iso(self._now()), "event": event}
        if self.name:
            rec["name"] = self.name
        rec.update({k: v for k, v in fields.items() if v is not None})
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
            except OSError as exc:
                if not self._warned:
                    log.warning("could not append to %s: %s", self.path, exc)
                    self._warned = True
        if self.state is not None:
            self.state.connection_event(rec)
        return rec


def describe_close(exc: BaseException | None) -> dict:
    """Close code and reason from a websockets ConnectionClosed (any version), else the error text."""
    if exc is None:
        return {}
    out: dict = {"error": f"{type(exc).__name__}: {exc}".strip()}
    if hasattr(exc, "rcvd"):  # websockets >= 10: the close frame we received, None if there was none
        frame = exc.rcvd
        code = 1006 if frame is None else getattr(frame, "code", None)  # 1006: abnormal closure
        reason = None if frame is None else getattr(frame, "reason", None)
    else:
        code, reason = getattr(exc, "code", None), getattr(exc, "reason", None)
    if isinstance(code, int):
        out["close_code"] = int(code)
    if reason:
        out["close_reason"] = str(reason)
    return out
