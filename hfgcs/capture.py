"""Hardware-independent capture logic: demod -> AGC -> squelch -> recordings.

``ChannelCapture`` takes raw IQ blocks for one channel and returns finished
:class:`Recording` objects. Any audio source (SoapySDR, KiwiSDR, OpenWebRX,
file replay) can drive it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .dsp import Agc, AudioFrontEnd, Squelch, UsbDemodulator, power_db

log = logging.getLogger("hfgcs.squelch")
# One line a few times a minute, plus a line on each open, close and floor raise.
SQUELCH_LOG_EVERY_S = 20.0


@dataclass
class Recording:
    frequency_khz: float
    receiver_id: str
    audio_rate: int
    audio: np.ndarray
    start_utc: datetime  # first sample of the detected activity
    file_start_utc: datetime  # first sample in the file (start minus pre-roll)
    end_utc: datetime
    preroll_seconds: float
    snr_db_peak: float
    snr_db_mean: float
    noise_floor_db: float
    truncated: bool = False
    continued_from: datetime | None = None  # previous file's activity start, when a cap split one signal

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / self.audio_rate


class ChannelCapture:
    """Squelch and record one channel from any source.

    ``input_kind`` is ``"iq"`` (complex baseband at ``input_rate``, USB
    demodulated here) or ``"audio"`` (USB audio already demodulated by the
    receiver, e.g. OpenWebRX or KiwiSDR). ``squelch_level="reported"`` makes
    the squelch use the receiver's own level (``feed(..., level_db=...)``),
    for sources whose audio is already AGC'd.
    """

    def __init__(
        self,
        cfg: Config,
        frequency_khz: float,
        start_utc: datetime | None = None,
        *,
        input_kind: str = "iq",
        input_rate: int | None = None,
        receiver_id: str | None = None,
        squelch_level: str = "slot",
    ):
        self.cfg = cfg
        self.frequency_khz = frequency_khz
        self.rate = cfg.audio_rate
        self.receiver_id = receiver_id or cfg.receiver_id
        if squelch_level not in ("slot", "reported"):
            raise ValueError(f"squelch_level must be 'slot' or 'reported', not {squelch_level!r}")
        self.squelch_level = squelch_level
        if input_kind == "iq":
            self.demod = UsbDemodulator(
                input_rate or cfg.iq_rate, cfg.audio_rate, cfg.usb_low_hz, cfg.usb_high_hz
            )
        elif input_kind == "audio":
            self.demod = AudioFrontEnd(
                input_rate or cfg.audio_rate, cfg.audio_rate, max(cfg.usb_low_hz, 300.0), cfg.usb_high_hz
            )
        else:
            raise ValueError(f"input_kind must be 'iq' or 'audio', not {input_kind!r}")
        self.input_kind = input_kind
        self.input_rate = input_rate or (cfg.iq_rate if input_kind == "iq" else cfg.audio_rate)
        self.agc = Agc(
            cfg.audio_rate,
            cfg.agc_target,
            cfg.agc_attack_seconds,
            cfg.agc_decay_seconds,
            cfg.agc_max_gain_db,
        )
        # The noise floor survives retunes; DSP state does not.
        self.squelch = Squelch(
            cfg.open_snr_db,
            cfg.close_snr_db,
            cfg.min_activity_seconds,
            cfg.hang_seconds,
            cfg.noise_floor_alpha,
            cfg.noise_floor_window_seconds,
            cfg.quiet_median_seconds,
        )
        self.ring_len = int(round((cfg.preroll_seconds + cfg.min_activity_seconds) * self.rate))
        self._link_from: datetime | None = None
        self._link_until = 0
        self._trace_t = 0.0
        self._traced_raises = 0
        self.retune(start_utc)

    # -- stream control -------------------------------------------------
    def retune(self, start_utc: datetime | None = None) -> None:
        """Start a fresh stream segment (after tuning). Keeps the noise floor."""
        self.demod.reset()
        self.agc.reset()
        self.squelch.release()
        self.squelch.snr_history.clear()
        self._link_from = None
        self._trace_t = self.squelch.stream_s
        self.t0 = start_utc or datetime.now(timezone.utc)
        self.samples = 0  # audio samples produced in this segment
        self.ring = np.zeros(0, np.float32)
        self._rec: list[np.ndarray] | None = None
        self._rec_meta: dict = {}

    @property
    def active(self) -> bool:
        """True while a transmission is pending or being recorded."""
        return self._rec is not None or self.squelch.state == "pending"

    def _time(self, sample_index: int) -> datetime:
        return self.t0 + timedelta(seconds=sample_index / self.rate)

    # -- processing -----------------------------------------------------
    def feed(self, samples: np.ndarray, level_db: float | None = None) -> list[Recording]:
        """Process one block (IQ or audio, per ``input_kind``)."""
        audio, slot_power = self.demod.process(samples)
        if len(audio) == 0:
            return []
        audio = self.agc.process(audio)
        block_s = len(audio) / self.rate
        self.samples += len(audio)
        if self.squelch_level == "reported":
            level = float("nan") if level_db is None else float(level_db)
        else:
            level = power_db(slot_power)
        event = self.squelch.update(level, block_s)
        done: list[Recording] = []

        if self._rec is None:
            self.ring = np.concatenate([self.ring, audio])[-self.ring_len :]
            if event == "open":
                # Activity began min_activity seconds ago; everything before
                # that in the ring is pre-roll.
                activity_samples = int(round(self.cfg.min_activity_seconds * self.rate))
                activity_start = max(self.samples - activity_samples, self.samples - len(self.ring))
                file_start = self.samples - len(self.ring)
                continued = None
                if self._link_from is not None and self.samples <= self._link_until:
                    continued = self._link_from
                self._link_from = None
                self._rec = [self.ring.copy()]
                self._rec_meta = {
                    "start": activity_start,
                    "file_start": file_start,
                    "preroll": (activity_start - file_start) / self.rate,
                    "noise": self.squelch.noise_floor_db,
                    "continued_from": continued,
                }
                self.ring = np.zeros(0, np.float32)
        else:
            self._rec.append(audio)
            recorded = (self.samples - self._rec_meta["file_start"]) / self.rate
            if event == "close":
                done.append(self._finish())
            elif recorded >= self.cfg.max_record_seconds:
                done.append(self._cap())
                event = "cap"
        self._trace(event)
        return done

    def _cap(self) -> Recording:
        """The length cap. A still-strong signal continues in the next file.

        Anything else is a stuck squelch: raise the floor and do not reopen
        on the same energy.
        """
        median = self.squelch.median_snr
        real = median is not None and median >= self.cfg.open_snr_db
        rec = self._finish(truncated=True)
        if real:
            self._link_from = rec.start_utc
            grace = self.cfg.min_activity_seconds + 2.0
            self._link_until = self.samples + int(round(grace * self.rate))
            self.squelch.force_close()
        else:
            self._link_from = None
            self.squelch.disarm()
        return rec

    def flush(self) -> list[Recording]:
        """Close any open recording (e.g. before retuning or on shutdown)."""
        if self._rec is None:
            return []
        rec = self._finish(truncated=True)
        self.squelch.force_close()
        return [rec]

    def _trace(self, event: str | None) -> None:
        """Log opens, closes, floor raises, and a line every ``SQUELCH_LOG_EVERY_S``."""
        sq = self.squelch
        reasons = []
        if event in ("open", "close", "cap"):
            reasons.append(event)
        if sq.floor_raises != self._traced_raises:
            reasons.append("floor")
            self._traced_raises = sq.floor_raises
        if sq.stream_s - self._trace_t >= SQUELCH_LOG_EVERY_S:
            reasons.append("tick")
            self._trace_t = sq.stream_s
        if not reasons:
            return
        log.info(
            "squelch %s %g kHz state=%s level=%s floor=%s snr=%s median=%s raises=%d",
            "+".join(reasons),
            self.frequency_khz,
            sq.state,
            _db(sq.last_level_db),
            _db(sq.noise_floor_db),
            _db(sq.last_snr),
            _db(sq.median_snr),
            sq.floor_raises,
        )

    def _finish(self, truncated: bool = False) -> Recording:
        meta = self._rec_meta
        audio = np.concatenate(self._rec).astype(np.float32)
        snrs = self.squelch.snr_history or [float("nan")]
        rec = Recording(
            frequency_khz=self.frequency_khz,
            receiver_id=self.receiver_id,
            audio_rate=self.rate,
            audio=audio,
            start_utc=self._time(meta["start"]),
            file_start_utc=self._time(meta["file_start"]),
            end_utc=self._time(self.samples),
            preroll_seconds=meta["preroll"],
            snr_db_peak=float(np.max(snrs)),
            snr_db_mean=float(np.mean(snrs)),
            noise_floor_db=float(meta["noise"]),
            truncated=truncated,
            continued_from=meta.get("continued_from"),
        )
        self._rec = None
        self._rec_meta = {}
        self.squelch.snr_history.clear()
        return rec


def _db(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.1f}"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def save_recording(rec: Recording, record_dir: Path, catalog_file: Path) -> dict:
    """Write the WAV and append one JSON line to the catalog. Returns the entry."""
    import soundfile as sf

    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    ts = rec.start_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rx = re.sub(r"[^A-Za-z0-9._-]+", "-", rec.receiver_id).strip("-") or "rx"
    path = record_dir / f"HFGCS_{rec.frequency_khz:.0f}kHz_{ts}_{rx}.wav"
    sf.write(path, rec.audio, rec.audio_rate, subtype="PCM_16")
    entry = {
        "schema": 2,
        "start_utc": _iso(rec.start_utc),
        "file_start_utc": _iso(rec.file_start_utc),
        "end_utc": _iso(rec.end_utc),
        "frequency_khz": rec.frequency_khz,
        "mode": "USB",
        "receiver_id": rec.receiver_id,
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "audio_rate": rec.audio_rate,
        "duration_sec": round(rec.duration_sec, 3),
        "preroll_sec": round(rec.preroll_seconds, 3),
        "snr_db_peak": round(rec.snr_db_peak, 1),
        "snr_db_mean": round(rec.snr_db_mean, 1),
        "noise_floor_db": round(rec.noise_floor_db, 1),
        "truncated": rec.truncated,
        "continued_from": _iso(rec.continued_from) if rec.continued_from else None,
        "notes": "USB voice candidate - review for EAM",
    }
    with open(catalog_file, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
