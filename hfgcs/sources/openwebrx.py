"""OpenWebRX / OpenWebRX+ audio websocket source.

Records the demodulated USB audio that an OpenWebRX server already produces,
so the radio stays owned by OpenWebRX and is shared politely with every other
listener.

Protocol notes (written from reading the OpenWebRX/csdr sources for protocol
understanding only; no OpenWebRX code is copied, see THIRD_PARTY.md):

* Connect to ``<base>/ws/``. The client sends
  ``SERVER DE CLIENT client=<name> type=receiver`` and the server answers
  ``CLIENT DE SERVER server=openwebrx version=...``.
* The client sends ``{"type": "connectionproperties", "params":
  {"output_rate": 12000, "hd_output_rate": 48000}}``.
* The server sends JSON text messages: ``config`` (``center_freq``,
  ``samp_rate``, ``sdr_id``, ``profile_id``, ``audio_compression``, ...;
  later messages may be partial updates), ``profiles``, ``smeter`` (linear
  channel power), ``backoff`` / ``sdr_error`` and others.
* ``{"type": "dspcontrol", "params": {"mod": "usb", "low_cut": ..,
  "high_cut": .., "offset_freq": <Hz from center>, "squelch_level": -150}}``
  then ``{"type": "dspcontrol", "action": "start"}`` starts this client's own
  demodulator. It only moves *our* demodulator inside the current window.
* ``selectprofile`` (``{"type": "selectprofile", "params": {"profile":
  "<sdr_id>|<profile_id>", "key": <unlock key, optional>}}``) and
  ``setfrequency`` retune the shared hardware for every listener. This module
  never sends ``setfrequency``. It sends ``selectprofile`` only when
  ``allow_profile_switch`` is set, or once for an explicit one-shot
  :meth:`OpenWebRXSource.select_profile_once`.
* OpenWebRX+ can lock profiles ("key_locked"). A locked profile is only
  selected when ``params.key`` equals the server's magic key; otherwise the
  server answers with ``{"type": "log_message", "value": "This profile is
  locked, keeping current profile."}`` and re-sends the current ``config``.
  Switching profiles faster than about every 10 s raises a "robot score" and
  can get the client banned, hence ``min_profile_interval_s``.
* The ``profiles`` message lists ``{"id": "<sdr_id>|<profile_id>", "name":
  "<SDR name> <profile name>"}``, e.g. ``"Airspy HF+ HFGCS 8992 kHz"``.
* Binary frames start with a type byte: 1 = waterfall FFT, 2 = audio,
  3 = secondary FFT, 4 = HD audio. Audio is int16 little-endian, or (when
  ``audio_compression == "adpcm"``) IMA ADPCM with a resync header every
  1000-odd bytes: ``b"SYNC"``, int16 step index, int16 predictor, then
  codec bytes (low nibble first). Codec state carries over between frames.
"""

from __future__ import annotations

import inspect
import json
import logging
import random
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

from ..config import OpenWebRXSettings
from ..runstate import describe_close
from .base import BaseSource, Chunk, SourceError, TuneRefused, db

log = logging.getLogger(__name__)

# Standard IMA ADPCM tables (public domain algorithm, Intel/DVI 1992).
IMA_INDEX = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]
IMA_STEP = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]

SYNC_WORD = b"SYNC"
SYNC_DATA_BYTES = 1001  # codec bytes between two sync headers

MSG_FFT, MSG_AUDIO, MSG_FFT2, MSG_HD_AUDIO = 1, 2, 3, 4


def ws_url(http_url: str) -> str:
    """``https://host[/]`` -> ``wss://host/ws/`` (also accepts ws:// URLs)."""
    url = http_url.rstrip("/")
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):] + "/ws/"
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):] + "/ws/"
    if url.startswith("wss://") or url.startswith("ws://"):
        return url + "/" if url.endswith("/ws") else url.rstrip("/") + "/ws/"
    raise ValueError(f"Unsupported OpenWebRX URL: {http_url}")


class ImaAdpcmCodec:
    """IMA ADPCM nibble codec (state: predictor and step index)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, index: int = 0, predictor: int = 0) -> None:
        self.index = min(88, max(0, int(index)))
        self.predictor = int(predictor)

    def decode_nibble(self, nibble: int) -> int:
        step = IMA_STEP[self.index]
        diff = step >> 3
        if nibble & 1:
            diff += step >> 2
        if nibble & 2:
            diff += step >> 1
        if nibble & 4:
            diff += step
        if nibble & 8:
            diff = -diff
        self.predictor = min(32767, max(-32768, self.predictor + diff))
        self.index = min(88, max(0, self.index + IMA_INDEX[nibble]))
        return self.predictor

    def encode_sample(self, sample: int) -> int:
        step = IMA_STEP[self.index]
        diff = int(sample) - self.predictor
        code = 0
        if diff < 0:
            code = 8
            diff = -diff
        if diff >= step:
            code |= 4
            diff -= step
        step >>= 1
        if diff >= step:
            code |= 2
            diff -= step
        step >>= 1
        if diff >= step:
            code |= 1
        self.decode_nibble(code)
        return code


class SyncAdpcmDecoder:
    """Streaming decoder for OpenWebRX's sync-framed ADPCM audio.

    Feed it the payload of each audio frame; state (codec and position in
    the sync framing) carries over, and bytes before the first sync header
    are skipped, so it can join a stream at any point.
    """

    def __init__(self) -> None:
        self.codec = ImaAdpcmCodec()
        self.phase = "search"
        self.matched = 0
        self.header = bytearray()
        self.remaining = 0

    def decode(self, data: bytes) -> np.ndarray:
        out = np.empty(2 * len(data), np.int16)
        n = 0
        codec = self.codec
        for byte in data:
            if self.phase == "search":
                if byte == SYNC_WORD[self.matched]:
                    self.matched += 1
                    if self.matched == len(SYNC_WORD):
                        self.phase, self.matched, self.header = "header", 0, bytearray()
                else:
                    self.matched = 1 if byte == SYNC_WORD[0] else 0
            elif self.phase == "header":
                self.header.append(byte)
                if len(self.header) == 4:
                    index, predictor = np.frombuffer(bytes(self.header), "<i2")
                    codec.reset(int(index), int(predictor))
                    self.phase, self.remaining = "data", SYNC_DATA_BYTES
            else:
                out[n] = codec.decode_nibble(byte & 0x0F)
                out[n + 1] = codec.decode_nibble(byte >> 4)
                n += 2
                self.remaining -= 1
                if self.remaining == 0:
                    self.phase = "search"
        return out[:n]


def encode_sync_adpcm(samples: np.ndarray, codec: ImaAdpcmCodec | None = None, counter: int = 0):
    """Encode int16 samples the way an OpenWebRX server frames them.

    Returns ``(bytes, counter)``; pass ``counter`` back in to continue the
    stream. Used by the tests' mock server and handy for debugging.
    """
    codec = codec or ImaAdpcmCodec()
    samples = np.asarray(samples, dtype=np.int16)
    if len(samples) % 2:
        samples = np.append(samples, np.int16(0))
    out = bytearray()
    for i in range(0, len(samples), 2):
        if counter <= 0:
            out += SYNC_WORD + np.array([codec.index, codec.predictor], "<i2").tobytes()
            counter = SYNC_DATA_BYTES
        lo = codec.encode_sample(int(samples[i]))
        hi = codec.encode_sample(int(samples[i + 1]))
        out.append(lo | (hi << 4))
        counter -= 1
    return bytes(out), counter


@dataclass
class ProfileWindow:
    """The span the shared SDR is currently tuned to."""

    center_hz: float
    samp_rate: float
    sdr_id: str = ""
    profile_id: str = ""

    @property
    def label(self) -> str:
        pid = f"{self.sdr_id}|{self.profile_id}" if self.sdr_id or self.profile_id else "?"
        return (
            f"profile {pid} (center {self.center_hz / 1e3:g} kHz, "
            f"span ±{self.samp_rate / 2e3:g} kHz)"
        )

    def covers(self, dial_hz: float, low_hz: float, high_hz: float, margin: float) -> bool:
        half = margin * self.samp_rate / 2.0
        return (dial_hz + low_hz) >= self.center_hz - half and (dial_hz + high_hz) <= self.center_hz + half


def _default_connect(url: str, settings: OpenWebRXSettings):
    try:
        from websockets.sync.client import connect
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SourceError("the OpenWebRX source needs 'websockets' (pip install .[openwebrx])") from exc
    parsed = urlparse(settings.url if "://" in settings.url else "http://" + settings.url)
    scheme = "https" if parsed.scheme in ("https", "wss") else "http"
    kwargs = {
        "origin": f"{scheme}://{parsed.netloc}",
        "max_size": 8_000_000,
        "open_timeout": settings.handshake_timeout_s,
        "user_agent_header": settings.user_agent,
        # OpenWebRX does not answer pings; a keepalive would kill the session.
        "ping_interval": None,
        # We manage the connection's lifetime ourselves (websockets >= 16).
        "legacy": True,
    }
    accepted = inspect.signature(connect).parameters
    return connect(url, **{k: v for k, v in kwargs.items() if k in accepted})


class OpenWebRXSource(BaseSource):
    """USB audio from an OpenWebRX demodulator, inside the active profile only."""

    name = "openwebrx"
    kind = "audio"

    def __init__(self, settings: OpenWebRXSettings | None = None, *, connect=None, clock=None, sleep=None):
        self.settings = settings or OpenWebRXSettings()
        url = self.settings.url
        host = urlparse(url if "://" in url else "//" + url).hostname
        super().__init__(self.settings.receiver_id or f"owrx-{host or 'server'}")
        if self.settings.squelch not in ("smeter", "audio"):
            raise ValueError("openwebrx.squelch must be 'smeter' or 'audio'")
        self.squelch_level = "reported" if self.settings.squelch == "smeter" else "slot"
        self.level_reported = self.squelch_level == "reported"
        self.url = ws_url(self.settings.url)
        self._connect = connect or _default_connect
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.ws = None
        self.server_version = ""
        self.config: dict = {}
        self.profiles: list[dict] = []
        self.window: ProfileWindow | None = None
        self.last_level_db: float | None = None
        self.off_channel_reason: str | None = None
        self.sent: list[dict] = []  # every JSON command we sent (for logs and tests)
        self.decoder = SyncAdpcmDecoder()
        self.failures = 0
        self._gap = False
        self._last_connect = float("-inf")
        self._last_tune = float("-inf")
        self._last_profile = float("-inf")
        self._dsp_started = False
        # "recording" or "waiting": logged once per change, not per poll.
        self.state: str | None = None
        self.log_messages: list[str] = []  # server log_message texts (e.g. "profile is locked")
        # Connection bookkeeping for the event log (monotonic clock times).
        self.connects = 0
        self._last_audio_at: float | None = None
        self._down_since: float | None = None  # set by a disconnect, cleared when audio resumes
        self._reconnect_attempt = 0

    # -- connection ---------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return int(self.settings.output_rate)

    def open(self) -> None:
        self._connect_once()
        super().open()

    def close(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            self.connection_event(
                "disconnect", f"closing {self.url} (client shutdown)", url=self.url, reason="client shutdown"
            )
            try:
                ws.close()
            except Exception:  # pragma: no cover - best effort
                pass
        super().close()

    def _polite_wait(self, last: float, interval: float) -> None:
        wait = last + interval - self._clock()
        if wait > 0:
            self._sleep(wait)

    def _connect_once(self) -> None:
        self._polite_wait(self._last_connect, self.settings.min_connect_interval_s)
        self._last_connect = self._clock()
        self.decoder = SyncAdpcmDecoder()
        self.config, self.window = {}, None
        self._dsp_started = False
        try:
            self._connect_once_inner()
        except SourceError as exc:
            self.connection_event(
                "connect_failed", f"{exc}", logging.WARNING, url=self.url, error=str(exc),
                attempt=self._reconnect_attempt or None,
            )
            raise
        self.connects += 1
        down = None if self._down_since is None else round(self._clock() - self._down_since, 3)
        n = self._reconnect_attempt
        again = "" if down is None else f"; reconnected after {down:.1f} s (attempt {n})"
        self.connection_event(
            "connect",
            f"connected to {self.url} ({self.server_version}), {self.window.label}{again}",
            url=self.url, server=self.server_version, window=self.window.label,
            reconnect=self.connects > 1, attempt=self._reconnect_attempt or None, down_s=down,
        )

    def _connect_once_inner(self) -> None:
        try:
            self.ws = self._connect(self.url, self.settings)
            self.ws.send("SERVER DE CLIENT client=hfgcs type=receiver")
            self._send({
                "type": "connectionproperties",
                "params": {"output_rate": self.sample_rate, "hd_output_rate": 48000},
            })
            deadline = self._clock() + self.settings.handshake_timeout_s
            while self._clock() < deadline and (not self.server_version or self.window is None):
                msg = self.ws.recv(timeout=max(0.05, deadline - self._clock()))
                self._handle(msg)
        except TimeoutError as exc:
            self._drop()
            raise SourceError(
                f"{self.url}: no handshake/config within {self.settings.handshake_timeout_s}s"
            ) from exc
        except SourceError:
            self._drop()
            raise
        except Exception as exc:
            self._drop()
            raise SourceError(f"{self.url}: {exc}") from exc
        if not self.server_version:
            self._drop()
            raise SourceError(f"{self.url}: OpenWebRX did not acknowledge the websocket")
        if self.window is None:
            self._drop()
            raise SourceError(f"{self.url}: server sent no center_freq/samp_rate")

    def _drop(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # pragma: no cover
                pass

    def _send(self, obj: dict, *, one_shot_select: bool = False) -> None:
        if obj.get("type") == "setfrequency":  # pragma: no cover - guard
            raise AssertionError("hfgcs never changes the OpenWebRX center frequency")
        if obj.get("type") == "selectprofile" and not (self.settings.allow_profile_switch or one_shot_select):
            raise AssertionError("selectprofile needs allow_profile_switch")  # pragma: no cover
        # Keep a copy for logs/tests with any unlock key redacted.
        params = obj.get("params")
        if isinstance(params, dict) and "key" in params:
            self.sent.append({**obj, "params": {**params, "key": "<redacted>"}})
        else:
            self.sent.append(obj)
        self.ws.send(json.dumps(obj))

    def _disconnected(self, why: str, exc: BaseException | None = None) -> None:
        """Record a lost connection (once per outage) before reconnecting."""
        if self._down_since is None:
            self._down_since = self._clock()
        info = describe_close(exc)
        code = f" (close code {info['close_code']}" if "close_code" in info else ""
        if code:
            code += f", reason {info['close_reason']!r})" if info.get("close_reason") else ")"
        self.connection_event(
            "disconnect", f"lost {self.url}: {why}{code}", logging.WARNING,
            url=self.url, reason=why, **info,
        )

    def _reconnect(self) -> None:
        """Reconnect with exponential backoff and jitter, then retune."""
        self._drop()
        self._gap = True
        if self._down_since is None:
            self._down_since = self._clock()
        self._reconnect_attempt = 0
        while True:
            self.failures += 1
            if self.failures > self.settings.max_reconnects and not self.settings.wait_for_profile:
                raise SourceError(f"{self.url}: giving up after {self.settings.max_reconnects} reconnects")
            delay = min(
                self.settings.backoff_max_s,
                self.settings.backoff_initial_s * 2 ** (self.failures - 1),
            ) * random.uniform(0.8, 1.2)
            self._reconnect_attempt += 1
            self.connection_event(
                "reconnect_attempt",
                f"reconnecting to {self.url} in {delay:.1f} s (attempt {self._reconnect_attempt})",
                url=self.url, attempt=self._reconnect_attempt, delay_s=round(delay, 3),
            )
            self._sleep(delay)
            try:
                self._connect_once()
            except SourceError:
                continue  # logged as connect_failed
            if self.dial_khz is not None:
                self._dsp_started = False
                outside = self._outside_reason(self.dial_khz)
                if outside is None:
                    self._apply_tune(self.dial_khz)
                    self._set_state("recording", self.dial_khz)
                elif self.settings.wait_for_profile:
                    self._idle(outside)
                else:
                    log.warning("after reconnect, %s kHz is not receivable: %s", self.dial_khz, outside)
                    self.off_channel_reason = outside
            return

    # -- messages ------------------------------------------------------------
    def _handle(self, msg) -> np.ndarray | None:
        if isinstance(msg, str):
            self._handle_text(msg)
            return None
        if not msg:
            return None
        kind, payload = msg[0], bytes(msg[1:])
        if kind != MSG_AUDIO:
            return None
        if self.config.get("audio_compression") == "adpcm":
            pcm = self.decoder.decode(payload)
        else:
            pcm = np.frombuffer(payload[: len(payload) // 2 * 2], "<i2")
        return pcm.astype(np.float32) / 32768.0

    def _handle_text(self, msg: str) -> None:
        if msg.startswith("CLIENT DE SERVER"):
            self.server_version = msg
            return
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return
        kind, value = data.get("type"), data.get("value")
        if kind == "config" and isinstance(value, dict):
            self.config.update(value)
            self._update_window()
        elif kind == "profiles" and isinstance(value, list):
            self.profiles = value
        elif kind == "smeter" and value is not None:
            try:
                self.last_level_db = db(float(value))
            except (TypeError, ValueError):
                pass
        elif kind == "backoff":
            raise SourceError(f"server asked us to back off: {data.get('reason') or value}")
        elif kind in ("sdr_error", "demodulator_error"):
            log.warning("OpenWebRX %s: %s", kind, value)
        elif kind == "log_message" and value is not None:
            self.log_messages.append(str(value))
            log.info("OpenWebRX log: %s", value)

    def _update_window(self) -> None:
        cfg = self.config
        if not cfg.get("center_freq") or not cfg.get("samp_rate"):
            return
        new = ProfileWindow(
            float(cfg["center_freq"]),
            float(cfg["samp_rate"]),
            str(cfg.get("sdr_id") or ""),
            str(cfg.get("profile_id") or ""),
        )
        old, self.window = self.window, new
        if old is not None and (old.center_hz, old.samp_rate, old.profile_id, old.sdr_id) != (
            new.center_hz, new.samp_rate, new.profile_id, new.sdr_id
        ):
            log.warning("OpenWebRX window changed to %s (profile switched)", new.label)
            if self.dial_khz is not None and self.ws is not None and not self._selecting:
                self._follow_window()

    _idle_gap_sent = False
    _selecting = False  # True while select_profile_once waits for the new config

    def _follow_window(self) -> None:
        """The shared radio moved. Retune inside it, or stop recording; never move it back."""
        reason = self._outside_reason(self.dial_khz)
        self._gap = True
        if reason is None:
            self._apply_tune(self.dial_khz)
            self._set_state("recording", self.dial_khz)
        elif self.settings.wait_for_profile:
            self._idle(reason)
        else:
            if self.off_channel_reason is None:
                log.warning("%s kHz is no longer receivable: %s; audio dropped", self.dial_khz, reason)
            self.off_channel_reason = reason

    def _set_state(self, state: str, dial_khz: float, reason: str = "") -> None:
        if state == self.state:
            return
        self.state = state
        label = self.window.label if self.window else "?"
        if state == "waiting":
            log.warning(
                "WAITING: %g kHz is not in the active %s; idling until a covering profile is selected",
                dial_khz, label,
            )
        else:
            log.warning("RECORDING: %g kHz is covered by the active %s", dial_khz, label)

    def _idle(self, reason: str) -> None:
        """Stay connected without our audio; resume when a covering profile appears."""
        self.off_channel_reason = reason
        self._gap = True
        self._set_state("waiting", self.dial_khz, reason)

    # -- tuning --------------------------------------------------------------
    def _outside_reason(self, dial_khz: float) -> str | None:
        """Why the active window does not cover ``dial_khz`` (None if it does)."""
        if self.window is None:
            return "not connected"
        s = self.settings
        if self.window.covers(dial_khz * 1e3, s.low_cut_hz, s.high_cut_hz, s.span_margin):
            return None
        return f"outside the active {self.window.label}"

    def refusal(self, dial_khz: float, switching: bool | None = None) -> str | None:
        if self.window is None:
            return "not connected (open the source first)"
        s = self.settings
        if self._outside_reason(dial_khz) is None:
            return None
        if switching is None:
            switching = s.allow_profile_switch
        if switching or s.wait_for_profile:
            return None  # switch (allowed) or wait for the owner to select a covering profile
        return (
            f"outside the active {self.window.label}. The OpenWebRX radio is shared, so hfgcs "
            "does not switch profiles or move the center frequency; ask the owner to select a "
            "profile that covers it, run with --wait-for-profile, or set "
            "openwebrx.allow_profile_switch = true"
        )

    def tune(self, dial_khz: float) -> None:
        super().tune(dial_khz)  # raises TuneRefused outside the window (unless allowed/waiting)
        dial_khz = float(dial_khz)
        outside = self._outside_reason(dial_khz)
        if outside is not None and not self.settings.allow_profile_switch:
            self._idle(outside)  # wait_for_profile: never retune the shared radio
            return
        self._apply_tune(dial_khz)
        self._set_state("recording", dial_khz)

    def _apply_tune(self, dial_khz: float) -> None:
        s = self.settings
        dial_hz = dial_khz * 1e3
        if not self.window.covers(dial_hz, s.low_cut_hz, s.high_cut_hz, s.span_margin):
            self._switch_profile(dial_khz)
        self._polite_wait(self._last_tune, s.min_tune_interval_s)
        self._last_tune = self._clock()
        self._send({
            "type": "dspcontrol",
            "params": {
                "mod": "usb",
                "low_cut": s.low_cut_hz,
                "high_cut": s.high_cut_hz,
                "offset_freq": int(round(dial_hz - self.window.center_hz)),
                "squelch_level": -150,
            },
        })
        if not self._dsp_started:
            self._send({"type": "dspcontrol", "action": "start"})
            self._dsp_started = True
        self.off_channel_reason = None

    def _switch_profile(self, dial_khz: float) -> None:
        """Only reached with allow_profile_switch=True."""
        s = self.settings
        profile = s.profile or next(
            (
                p.get("id")
                for p in self.profiles
                if str(p.get("name", "")).endswith((f"HFGCS {dial_khz:g}", f"HFGCS {dial_khz:g} kHz"))
            ),
            None,
        )
        if not profile:
            raise TuneRefused(f"no OpenWebRX profile known for {dial_khz:g} kHz (set openwebrx.profile)")
        self._polite_wait(self._last_profile, s.min_profile_interval_s)
        self._last_profile = self._clock()
        log.warning("switching the shared OpenWebRX radio to profile %s (allow_profile_switch)", profile)
        self._send({"type": "selectprofile", "params": {"profile": profile}})
        old = self.window
        deadline = self._clock() + s.handshake_timeout_s
        while self._clock() < deadline:
            try:
                self._handle(self.ws.recv(timeout=max(0.05, deadline - self._clock())))
            except TimeoutError:
                break
            moved = self.window is not old
            if moved and self.window.covers(dial_khz * 1e3, s.low_cut_hz, s.high_cut_hz, 1.0):
                return
        raise TuneRefused(f"profile {profile} did not bring {dial_khz:g} kHz into the window")

    # -- one-shot profile selection ----------------------------------------------
    def find_profile(self, name: str) -> dict | None:
        """A profile by id ("sdr|profile"), exact name, or name suffix ("HFGCS 8992 kHz")."""
        for match in (
            lambda p: p.get("id") == name,
            lambda p: p.get("name") == name,
            lambda p: str(p.get("name", "")).endswith(" " + name),
        ):
            hits = [p for p in self.profiles if match(p)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise SourceError(f"profile name {name!r} is ambiguous: {[p.get('id') for p in hits]}")
        return None

    def active_profile_id(self) -> str:
        w = self.window
        return f"{w.sdr_id}|{w.profile_id}" if w else ""

    def select_profile_once(self, name: str, key: str | None) -> bool:
        """Select profile ``name`` exactly once (a deliberate, owner-approved retune).

        Sends at most one ``selectprofile``. Returns False if it was already
        active (nothing sent), True once the server confirms the switch.
        Raises SourceError if the server refuses (locked, wrong key) or does
        not confirm. ``key`` is sent to the server only, never logged.
        """
        if self.ws is None or self.window is None:
            raise SourceError("not connected")
        if not self.profiles:  # profiles usually arrive with the handshake
            deadline = self._clock() + self.settings.handshake_timeout_s
            while not self.profiles and self._clock() < deadline:
                try:
                    self._handle(self.ws.recv(timeout=max(0.05, deadline - self._clock())))
                except TimeoutError:
                    break
        profile = self.find_profile(name)
        if profile is None:
            known = ", ".join(str(p.get("name")) for p in self.profiles) or "none"
            raise SourceError(f"no OpenWebRX profile matches {name!r} (known: {known})")
        pid = str(profile["id"])
        if pid == self.active_profile_id():
            log.warning("profile %s (%s) is already active; nothing to do", pid, profile.get("name"))
            return False
        params = {"profile": pid}
        if key:
            params["key"] = key
        log.warning("selecting OpenWebRX profile %s (%s) once (one-shot)", pid, profile.get("name"))
        self._selecting = True
        n_logs = len(self.log_messages)
        try:
            self._send({"type": "selectprofile", "params": params}, one_shot_select=True)
            deadline = self._clock() + max(self.settings.handshake_timeout_s, 15.0)
            while self._clock() < deadline:
                try:
                    self._handle(self.ws.recv(timeout=max(0.05, deadline - self._clock())))
                except TimeoutError:
                    break
                if self.active_profile_id() == pid:
                    log.warning("OpenWebRX confirmed profile %s: %s", pid, self.window.label)
                    return True
                refused = [m for m in self.log_messages[n_logs:] if "locked" in m.lower()]
                if refused:
                    raise SourceError(f"server refused profile {pid}: {refused[0]}")
        finally:
            self._selecting = False
        raise SourceError(f"server did not confirm profile {pid} (still {self.active_profile_id()})")

    # -- streaming -------------------------------------------------------------
    def read(self, timeout: float | None = None) -> Chunk | None:
        if self.ws is None:
            self._reconnect()
        deadline = self._clock() + (1.0 if timeout is None else timeout)
        while True:
            if self.off_channel_reason is not None and self._gap:
                # Tell the scanner now, so it closes a recording in progress.
                self._gap = False
                self._idle_gap_sent = True
                return Chunk(np.zeros(0, np.float32), self.sample_rate, level_db=None, gap=True)
            remaining = deadline - self._clock()
            if remaining <= 0:
                return Chunk(np.zeros(0, np.float32), self.sample_rate, level_db=self.last_level_db)
            try:
                msg = self.ws.recv(timeout=remaining)
            except TimeoutError:
                continue
            except SourceError:
                raise
            except Exception as exc:  # connection closed or broken
                self._disconnected("connection closed" if "closed" in type(exc).__name__.lower()
                                   else "stream error", exc)
                self._reconnect()
                continue
            try:
                audio = self._handle(msg)
            except SourceError as exc:  # server asked us to back off
                self._disconnected(str(exc))
                self.failures = max(self.failures, 3)
                self._reconnect()
                continue
            if audio is None or len(audio) == 0 or self.off_channel_reason is not None:
                continue
            self.failures = 0
            now = self._clock()
            if self._down_since is not None:
                since = self._last_audio_at if self._last_audio_at is not None else self._down_since
                gap_s = round(now - since, 3)
                self.connection_event(
                    "audio_resumed", f"audio from {self.url} resumed after a {gap_s:.2f} s gap",
                    url=self.url, gap_s=gap_s, down_s=round(now - self._down_since, 3),
                )
                self._down_since = None
            self._last_audio_at = now
            gap, self._gap = self._gap or self._idle_gap_sent, False
            self._idle_gap_sent = False
            return Chunk(audio, self.sample_rate, level_db=self.last_level_db, gap=gap)
