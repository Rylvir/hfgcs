"""Streaming SSB DSP: USB demodulator, AGC and squelch.

Every block keeps its state between calls (mixer phase, filter memory,
decimator phase, AGC envelope), so a stream cut into arbitrary chunks gives
the same output as one long call. Nothing here touches hardware.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal


class Nco:
    """Numerically controlled oscillator with a phase that persists across blocks."""

    def __init__(self, freq_hz: float, sample_rate: float):
        self.step = 2.0 * math.pi * freq_hz / sample_rate
        self.phase = 0.0

    def mix(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        ph = self.phase + self.step * np.arange(n)
        self.phase = math.fmod(self.phase + self.step * n, 2.0 * math.pi)
        return x * np.exp(1j * ph)

    def reset(self) -> None:
        self.phase = 0.0


class UsbDemodulator:
    """Weaver-style USB demodulator with a stateful decimator.

    1. Mix the IQ down by the centre of the audio passband, so the wanted
       USB slot (dial + low .. dial + high) sits symmetrically around 0 Hz
       and the opposite (LSB) side sits entirely outside it.
    2. Low-pass the *complex* signal at half the bandwidth (elliptic,
       stateful SOS). This is the sideband filter.
    3. Decimate to the audio rate (anti-aliased by step 2).
    4. Mix back up by the passband centre and take the real part.

    ``process`` returns the audio plus the mean power of the complex voice
    slot, which is what the squelch should look at.
    """

    def __init__(
        self,
        iq_rate: int,
        audio_rate: int,
        low_hz: float = 0.0,
        high_hz: float = 3000.0,
        stop_db: float = 80.0,
    ):
        if iq_rate % audio_rate:
            raise ValueError("iq_rate must be an integer multiple of audio_rate")
        if not 0 <= low_hz < high_hz < audio_rate / 2:
            raise ValueError("passband must fit below audio Nyquist")
        self.iq_rate = iq_rate
        self.audio_rate = audio_rate
        self.decim = iq_rate // audio_rate
        self.center_hz = (low_hz + high_hz) / 2.0
        half_bw = (high_hz - low_hz) / 2.0
        self.sos = signal.ellip(10, 0.2, stop_db, half_bw, btype="low", fs=iq_rate, output="sos")
        self.down = Nco(-self.center_hz, iq_rate)
        self.up = Nco(self.center_hz, audio_rate)
        self.reset()

    def reset(self) -> None:
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.complex128)
        self.skip = 0  # samples to drop at the start of the next block
        self.down.reset()
        self.up.reset()

    def process(self, iq: np.ndarray) -> tuple[np.ndarray, float]:
        iq = np.asarray(iq, dtype=np.complex128)
        if len(iq) == 0:
            return np.zeros(0, np.float32), float("nan")
        bb = self.down.mix(iq)
        bb, self.zi = signal.sosfilt(self.sos, bb, zi=self.zi)
        dec = bb[self.skip :: self.decim]
        self.skip = (self.skip - len(iq)) % self.decim
        if len(dec) == 0:
            return np.zeros(0, np.float32), float("nan")
        power = float(np.mean(np.abs(dec) ** 2))
        audio = np.real(self.up.mix(dec))
        return audio.astype(np.float32), power


def power_db(power: float) -> float:
    return 10.0 * math.log10(power + 1e-20)


class Agc:
    """Single peak-tracking AGC with separate attack and decay times."""

    def __init__(
        self,
        sample_rate: int,
        target: float = 0.3,
        attack_s: float = 0.01,
        decay_s: float = 0.5,
        max_gain_db: float = 60.0,
    ):
        self.target = target
        self.attack = 1.0 - math.exp(-1.0 / (attack_s * sample_rate))
        self.decay = 1.0 - math.exp(-1.0 / (decay_s * sample_rate))
        self.min_env = target / (10 ** (max_gain_db / 20.0))
        self.reset()

    def reset(self) -> None:
        self.env = self.min_env

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), np.float32)
        env, att, dec, floor, tgt = self.env, self.attack, self.decay, self.min_env, self.target
        for i, v in enumerate(np.asarray(x, dtype=np.float64)):
            a = abs(v)
            env += (att if a > env else dec) * (a - env)
            if env < floor:
                env = floor
            out[i] = v * tgt / env
        self.env = env
        np.clip(out, -1.0, 1.0, out=out)
        return out


class Squelch:
    """Voice-slot squelch with hysteresis, hang time and a frozen noise floor.

    Time is counted from the length of the blocks fed in, not the wall clock,
    so the behaviour is deterministic and testable.
    States: ``closed`` -> ``pending`` (above threshold, not yet long enough)
    -> ``open`` -> back to ``closed`` after ``hang_s`` below the close level.
    The noise floor only adapts while ``closed``.
    """

    def __init__(
        self,
        open_snr_db: float = 8.0,
        close_snr_db: float = 5.0,
        min_activity_s: float = 0.8,
        hang_s: float = 4.0,
        alpha: float = 0.05,
    ):
        self.open_snr_db = open_snr_db
        self.close_snr_db = close_snr_db
        self.min_activity_s = min_activity_s
        self.hang_s = hang_s
        self.alpha = alpha
        self.noise_floor_db: float | None = None
        self.state = "closed"
        self.pending_s = 0.0
        self.below_s = 0.0
        self.snr_history: list[float] = []

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def update(self, level_db: float, block_s: float) -> str | None:
        """Feed one block level. Returns "open", "close" or None."""
        if math.isnan(level_db):
            return None
        if self.noise_floor_db is None:
            self.noise_floor_db = level_db
        snr = level_db - self.noise_floor_db

        if self.state == "open":
            self.snr_history.append(snr)
            self.below_s = 0.0 if snr >= self.close_snr_db else self.below_s + block_s
            if self.below_s >= self.hang_s:
                self.state = "closed"
                self.below_s = 0.0
                return "close"
            return None

        if snr >= self.open_snr_db:
            self.state = "pending"
            self.pending_s += block_s
            self.snr_history.append(snr)
            if self.pending_s >= self.min_activity_s - 1e-9:
                self.state = "open"
                self.pending_s = 0.0
                self.below_s = 0.0
                return "open"
            return None

        # Below threshold while closed or pending: drop the pending burst and
        # let the noise floor follow the band.
        self.state = "closed"
        self.pending_s = 0.0
        self.snr_history.clear()
        self.noise_floor_db += self.alpha * (level_db - self.noise_floor_db)
        return None

    def force_close(self) -> None:
        self.state = "closed"
        self.pending_s = 0.0
        self.below_s = 0.0
