"""Hardware-independent capture logic: demod -> AGC -> squelch -> recordings.

``ChannelCapture`` takes raw IQ blocks for one channel and returns finished
:class:`Recording` objects. Any audio source (SoapySDR, KiwiSDR, OpenWebRX,
file replay) can drive it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .dsp import Agc, Squelch, UsbDemodulator, power_db


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
    def __init__(self, cfg: Config, frequency_khz: float, start_utc: datetime | None = None):
        self.cfg = cfg
        self.frequency_khz = frequency_khz
        self.rate = cfg.audio_rate
        self.demod = UsbDemodulator(cfg.iq_rate, cfg.audio_rate, cfg.usb_low_hz, cfg.usb_high_hz)
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
        )
        self.ring_len = int(round((cfg.preroll_seconds + cfg.min_activity_seconds) * self.rate))
        self.retune(start_utc)

    # -- stream control -------------------------------------------------
    def retune(self, start_utc: datetime | None = None) -> None:
        """Start a fresh stream segment (after tuning). Keeps the noise floor."""
        self.demod.reset()
        self.agc.reset()
        self.squelch.force_close()
        self.squelch.snr_history.clear()
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
    def feed(self, iq: np.ndarray) -> list[Recording]:
        audio, slot_power = self.demod.process(iq)
        if len(audio) == 0:
            return []
        audio = self.agc.process(audio)
        block_s = len(audio) / self.rate
        self.samples += len(audio)
        event = self.squelch.update(power_db(slot_power), block_s)
        done: list[Recording] = []

        if self._rec is None:
            self.ring = np.concatenate([self.ring, audio])[-self.ring_len :]
            if event == "open":
                # Activity began min_activity seconds ago; everything before
                # that in the ring is pre-roll.
                activity_samples = int(round(self.cfg.min_activity_seconds * self.rate))
                activity_start = max(self.samples - activity_samples, self.samples - len(self.ring))
                file_start = self.samples - len(self.ring)
                self._rec = [self.ring.copy()]
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
        if self._rec is None:
            return []
        rec = self._finish(truncated=True)
        self.squelch.force_close()
        return [rec]

    def _finish(self, truncated: bool = False) -> Recording:
        meta = self._rec_meta
        audio = np.concatenate(self._rec).astype(np.float32)
        snrs = self.squelch.snr_history or [float("nan")]
        rec = Recording(
            frequency_khz=self.frequency_khz,
            receiver_id=self.cfg.receiver_id,
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
    path = record_dir / f"HFGCS_{rec.frequency_khz:.0f}kHz_{ts}_{rec.receiver_id}.wav"
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
