"""Hardware-independent capture logic: demod -> AGC -> squelch -> recordings.

``ChannelCapture`` takes raw IQ blocks for one channel and returns finished
:class:`Recording` objects. Any audio source (SoapySDR, KiwiSDR, OpenWebRX,
file replay) can drive it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .dsp import FRAME_SECONDS, Agc, AudioFrontEnd, Squelch, SquelchV2, UsbDemodulator, power_db

log = logging.getLogger(__name__)

# The v2 slot level is the median of the 10 ms slot powers over this span (across block
# boundaries), so a static crash of up to about half of it does not count as signal.
MEDIAN_LEVEL_SECONDS = 0.1


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

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / self.audio_rate


class ChannelCapture:
    """Squelch and record one channel from any source.

    ``input_kind`` is ``"iq"`` (complex baseband at ``input_rate``, USB
    demodulated here) or ``"audio"`` (USB audio already demodulated by the
    receiver, e.g. OpenWebRX or KiwiSDR). ``squelch_level="reported"`` makes
    the v1 squelch use the receiver's own level (``feed(..., level_db=...)``),
    for sources whose audio is already AGC'd. ``level_reported`` says the source
    sends such a level with its blocks (KiwiSDR RSSI, OpenWebRX S-meter); the v2
    squelch uses it when there is one, else the median of the slot's 10 ms frame
    powers.

    ``cfg.squelch`` picks the squelch that records; ``cfg.squelch_shadow`` runs the
    other one on the same blocks without recording. Its decisions collect in
    :attr:`shadow_events` (dicts, one per shadow capture) for the caller to write.
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
        level_reported: bool = False,
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
        self.level_reported = level_reported or squelch_level == "reported"
        # The noise floor survives retunes; DSP state does not.
        self.squelch, self.level_mode = self._make_squelch(cfg.squelch)
        self.shadow = self.shadow_level_mode = None
        if cfg.squelch_shadow and cfg.squelch_shadow != cfg.squelch:
            self.shadow, self.shadow_level_mode = self._make_squelch(cfg.squelch_shadow)
        self.shadow_events: list[dict] = []
        self._shadow_open: dict | None = None
        self._slices: deque[float] = deque(maxlen=max(1, int(round(MEDIAN_LEVEL_SECONDS / FRAME_SECONDS))))
        # v2 may count activity from up to open_window_seconds before it opens
        lead = cfg.open_window_seconds if cfg.squelch == "v2" else cfg.min_activity_seconds
        self.ring_len = int(round((cfg.preroll_seconds + lead) * self.rate))
        self.retune(start_utc)

    def _make_squelch(self, name: str):
        """(squelch, level mode): "slot" = mean slot power, "reported" = the
        receiver's level, "slot_median" = median of the 10 ms slot powers."""
        cfg = self.cfg
        if name == "v1":
            sq = Squelch(cfg.open_snr_db, cfg.close_snr_db, cfg.min_activity_seconds, cfg.hang_seconds,
                         cfg.noise_floor_alpha)
            return sq, self.squelch_level
        sq = SquelchV2(
            cfg.open_snr_db, cfg.close_snr_db, cfg.min_activity_seconds, cfg.open_window_seconds,
            cfg.hang_seconds, cfg.long_hang_seconds, cfg.long_hang_after_seconds, cfg.noise_floor_tau_seconds,
            cfg.warmup_seconds, cfg.stuck_seconds, cfg.stuck_flat_db,
        )
        return sq, ("reported" if self.level_reported else "slot_median")

    # -- stream control -------------------------------------------------
    def retune(self, start_utc: datetime | None = None) -> None:
        """Start a fresh stream segment (after tuning). Keeps the noise floor."""
        self.demod.reset()
        self.agc.reset()
        self._slices.clear()
        self.squelch.force_close()
        self.squelch.snr_history.clear()
        if self.shadow is not None:
            self._shadow_safe(self._shadow_flush)
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
    def _level(self, mode: str, slot_power: float, level_db: float | None) -> float:
        if mode == "reported":
            return float("nan") if level_db is None else float(level_db)
        if mode == "slot_median":
            return power_db(float(np.median(self._slices))) if self._slices else float("nan")
        return power_db(slot_power)

    def feed(self, samples: np.ndarray, level_db: float | None = None) -> list[Recording]:
        """Process one block (IQ or audio, per ``input_kind``)."""
        audio, slot_power = self.demod.process(samples)
        if len(audio) == 0:
            return []
        audio = self.agc.process(audio)
        block_s = len(audio) / self.rate
        self.samples += len(audio)
        self._slices.extend(self.demod.frame_power)
        event = self.squelch.update(self._level(self.level_mode, slot_power, level_db), block_s)
        if self.shadow is not None:
            self._shadow_safe(self._feed_shadow, self._level(self.shadow_level_mode, slot_power, level_db),
                              block_s)
        done: list[Recording] = []

        if self._rec is None:
            self.ring = np.concatenate([self.ring, audio])[-self.ring_len :]
            if event == "open":
                # Activity began activity_s seconds ago; everything before
                # that in the ring is pre-roll.
                activity_samples = int(round(self.squelch.activity_s * self.rate))
                activity_start = max(self.samples - activity_samples, self.samples - len(self.ring))
                # The ring may hold more than pre-roll + activity (v2 keeps open_window_seconds).
                preroll_samples = int(round(self.cfg.preroll_seconds * self.rate))
                file_start = max(self.samples - len(self.ring), activity_start - preroll_samples)
                self._rec = [self.ring[len(self.ring) - (self.samples - file_start) :].copy()]
                self._rec_meta = {
                    "start": activity_start,
                    "file_start": file_start,
                    "preroll": (activity_start - file_start) / self.rate,
                    "noise": self.squelch.noise_floor_db,
                }
                self.ring = np.zeros(0, np.float32)
        else:
            self._rec.append(audio)
            recorded = (self.samples - self._rec_meta["file_start"]) / self.rate
            if event == "close":
                done.append(self._finish())
            elif recorded >= self.cfg.max_record_seconds:
                done.append(self._finish(truncated=True))
                self.squelch.force_close()
        return done

    def flush(self) -> list[Recording]:
        """Close any open recording (e.g. before retuning or on shutdown)."""
        if self.shadow is not None:
            self._shadow_safe(self._shadow_flush)
        if self._rec is None:
            return []
        rec = self._finish(truncated=True)
        self.squelch.force_close()
        return [rec]

    # -- shadow squelch ---------------------------------------------------
    def _shadow_safe(self, fn, *args) -> None:
        """The shadow squelch only observes: a fault in it never stops the recording."""
        try:
            fn(*args)
        except Exception:
            log.exception("shadow squelch failed on %g kHz; shadow mode is off for this channel",
                          self.frequency_khz)
            self.shadow = self.shadow_level_mode = self._shadow_open = None

    def _shadow_flush(self) -> None:
        if self._shadow_open is not None:
            self._shadow_close("flush")
        self.shadow.force_close()
        self.shadow.snr_history.clear()

    def _feed_shadow(self, level: float, block_s: float) -> None:
        sq = self.shadow
        event = sq.update(level, block_s)
        if event == "open":
            start = max(0, self.samples - int(round(sq.activity_s * self.rate)))
            self._shadow_open = {"start": start, "noise": sq.noise_floor_db}
        elif self._shadow_open is not None:
            if event == "close":
                self._shadow_close(sq.close_reason or "close")
            elif (self.samples - self._shadow_open["start"]) / self.rate >= self.cfg.max_record_seconds:
                self._shadow_close("cap")
                sq.force_close()

    def _shadow_close(self, reason: str) -> None:
        """One shadow capture: what the shadow squelch would have recorded."""
        sq, meta = self.shadow, self._shadow_open
        snrs = [x for x in sq.snr_history if math.isfinite(x)]
        self.shadow_events.append({
            "schema": 1,
            "algorithm": self.cfg.squelch_shadow,
            "live_algorithm": self.cfg.squelch,
            "level": self.shadow_level_mode,
            "frequency_khz": self.frequency_khz,
            "receiver_id": self.receiver_id,
            "start_utc": _iso(self._time(meta["start"])),
            "end_utc": _iso(self._time(self.samples)),
            "duration_sec": round((self.samples - meta["start"]) / self.rate, 3),
            "close_reason": reason,
            "snr_db_peak": round(max(snrs), 1) if snrs else None,
            "snr_db_mean": round(float(np.mean(snrs)), 1) if snrs else None,
            "noise_floor_db": None if meta["noise"] is None else round(float(meta["noise"]), 1),
        })
        sq.snr_history.clear()
        self._shadow_open = None

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
        )
        self._rec = None
        self._rec_meta = {}
        self.squelch.snr_history.clear()
        return rec


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
        "notes": "USB voice candidate - review for EAM",
    }
    with open(catalog_file, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
