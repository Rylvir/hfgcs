"""The audio source interface every receiver backend implements.

A source delivers blocks of samples for one tuned HFGCS channel:

* ``kind == "iq"``: complex baseband centred on the dial frequency (the USB
  voice slot is +0..+3 kHz). :class:`hfgcs.capture.ChannelCapture` runs the
  Weaver USB demodulator on it. Used by SoapySDR and IQ file replay.
* ``kind == "audio"``: real, already-demodulated USB audio. The capture
  pipeline only resamples, band-limits and squelches it. Used by OpenWebRX,
  KiwiSDR and WAV replay.

Sources are context managers: ``open()`` acquires the receiver (a USB device,
a websocket, a KiwiSDR slot), ``close()`` must always release it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import numpy as np

SourceKind = Literal["iq", "audio"]


class SourceError(RuntimeError):
    """A receiver could not be opened or stopped delivering samples."""


class TuneRefused(SourceError):
    """The source will not tune to this channel (etiquette or hardware limits)."""


@dataclass
class Chunk:
    """One block of samples from a source."""

    samples: np.ndarray  # complex64 IQ or float32 audio in [-1, 1]
    rate: int
    t0: datetime | None = None  # UTC time of the first sample, if known
    # Channel power the receiver itself reports for this block (e.g. the
    # OpenWebRX S-meter), in dB. Used for squelch when the audio is AGC'd.
    level_db: float | None = None
    # True when the stream was interrupted before this block (reconnect,
    # receiver moved off our channel): the scanner closes any open recording.
    gap: bool = False


@runtime_checkable
class AudioSource(Protocol):
    """What the scanner needs from any receiver backend."""

    name: str
    kind: SourceKind
    #: How the squelch should measure a block: "slot" = power of the
    #: demodulated voice slot, "reported" = :attr:`Chunk.level_db`.
    squelch_level: Literal["slot", "reported"]

    @property
    def sample_rate(self) -> int: ...

    @property
    def receiver_id(self) -> str: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def can_tune(self, dial_khz: float) -> bool: ...

    def tune(self, dial_khz: float) -> None: ...

    def read(self, timeout: float | None = None) -> Chunk | None:
        """Next block; an empty block if nothing arrived in time; ``None`` at end of stream."""
        ...


class BaseSource:
    """Shared plumbing: context manager, tune bookkeeping, refusal messages."""

    name = "base"
    kind: SourceKind = "audio"
    squelch_level: Literal["slot", "reported"] = "slot"
    #: True if the source can only ever deliver one channel (e.g. one Kiwi slot).
    single_channel = False
    #: Optional :class:`hfgcs.runstate.ConnectionEvents` the scanner attaches; connection
    #: events are logged either way and also appended there when it is set.
    events = None

    def __init__(self, receiver_id: str = ""):
        self._receiver_id = receiver_id
        self.dial_khz: float | None = None
        self.is_open = False

    @property
    def sample_rate(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def receiver_id(self) -> str:
        return self._receiver_id or self.name

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def refusal(self, dial_khz: float) -> str | None:
        """Why ``dial_khz`` cannot be tuned, or ``None`` if it can."""
        return None

    def can_tune(self, dial_khz: float) -> bool:
        return self.refusal(dial_khz) is None

    def tune(self, dial_khz: float) -> None:
        reason = self.refusal(dial_khz)
        if reason is not None:
            raise TuneRefused(f"{self.name}: refusing {dial_khz:g} kHz: {reason}")
        self.dial_khz = float(dial_khz)

    def read(self, timeout: float | None = None) -> Chunk | None:  # pragma: no cover
        raise NotImplementedError

    def connection_event(self, event: str, message: str, level: int = logging.INFO, **fields) -> dict:
        """Log one connection event with its UTC time and append it to ``self.events``.

        ``event`` is one of connect, connect_failed, disconnect, reconnect_attempt,
        audio_resumed. The log line reads e.g.
        ``openwebrx DISCONNECT at 2026-10-26T23:10:04.120Z: ... (close code 1006)``.
        """
        from ..runstate import utc_iso

        fields = {"source": self.name, "receiver": self.receiver_id, **fields}
        if self.events is not None:
            rec = self.events.emit(event, **fields)
        else:
            rec = {"utc": utc_iso(), "event": event, **{k: v for k, v in fields.items() if v is not None}}
        logging.getLogger(f"hfgcs.sources.{self.name}").log(
            level, "%s %s at %s: %s", self.name, event.upper(), rec["utc"], message
        )
        return rec

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def db(power: float) -> float:
    return 10.0 * math.log10(max(power, 1e-20))
