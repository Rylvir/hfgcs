"""Streaming SSB DSP: USB demodulator, AGC and squelch (v1 and v2, see docs/squelch.md).

Every block keeps its state between calls (mixer phase, filter memory,
decimator phase, AGC envelope), so a stream cut into arbitrary chunks gives
the same output as one long call. Nothing here touches hardware.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy import signal

FRAME_SECONDS = 0.01  # analysis frame for the median block level (v2 squelch)


def frame_powers(x: np.ndarray, frame: int) -> np.ndarray:
    """Mean power of consecutive ~``frame``-sample slices of ``x`` (real or complex).

    The block is split into ``round(len / frame)`` nearly equal slices, so a short
    tail is never a frame of its own."""
    x = np.asarray(x)
    if len(x) == 0:
        return np.zeros(0)
    p = np.abs(x) ** 2 if np.iscomplexobj(x) else np.asarray(x, dtype=np.float64) ** 2
    k = max(1, int(round(len(p) / frame)))
    return np.array([part.mean() for part in np.array_split(p, k)])


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
        self.frame = max(1, int(round(FRAME_SECONDS * audio_rate)))
        self.frame_power = np.zeros(0)  # 10 ms slot powers of the last block (median level)
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
        self.frame_power = frame_powers(dec, self.frame)
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
        self.close_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    @property
    def activity_s(self) -> float:
        """How long before the "open" event the activity began."""
        return self.min_activity_s

    def update(self, level_db: float, block_s: float) -> str | None:
        """Feed one block level. Returns "open", "close" or None."""
        if not math.isfinite(level_db):
            # NaN = no level for this block; an infinite one (e.g. an S-meter value
            # "Infinity") would otherwise make the floor NaN for good.
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
                self.close_reason = "hang"
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
                self.close_reason = None
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


class SquelchV2:
    """The audited squelch (docs/squelch.md); same interface as :class:`Squelch`.

    Each difference from v1 answers a failure seen on our channels:

    * **Warm-up.** No decisions until ``warmup_s`` of valid levels; the first floor is
      their median. v1 took the first block, and a KiwiSDR stream starts about 8 dB
      quiet while its AGC settles, so v1 opened on plain noise and stayed open.
    * **Dropouts.** Non-finite levels and levels at or below ``silence_db`` (digital
      silence) carry no information: they never move the floor (0.4 s of zeros dragged
      v1's floor down about 13 dB) but count toward the hang while open.
    * **Opening.** Opens once ``min_activity_s`` of blocks above ``open_snr_db`` fall
      inside the last ``open_window_s``. v1 needed them back to back, so a message read
      word by word with pauses never opened it when its continuous intro was missed.
    * **Floor tracking.** While closed the floor follows the band with a time constant
      in seconds (``floor_tau_s``), whatever the block size.
    * **Stuck captures.** A capture whose level has been flat (standard deviation below
      ``stuck_flat_db``) for ``stuck_s`` is a carrier or a noise bed, not speech, which
      moves several dB between syllables and pauses: it closes and the floor is
      re-learned from those levels. v1 stayed open to the length cap and reopened on
      the same noise.
    * **Long hang.** Once a capture has run ``long_hang_after_s`` the hang is
      ``long_hang_s``, so a long pause or a fade inside a message does not split it.

    Feed it the receiver's own level where there is one (KiwiSDR RSSI, OpenWebRX
    S-meter), else the median of the slot's recent 10 ms powers
    (:data:`hfgcs.capture.MEDIAN_LEVEL_SECONDS`), so a short static crash does not
    count as signal.
    """

    def __init__(
        self,
        open_snr_db: float = 8.0,
        close_snr_db: float = 5.0,
        min_activity_s: float = 0.8,
        open_window_s: float = 3.0,
        hang_s: float = 4.0,
        long_hang_s: float = 8.0,
        long_hang_after_s: float = 10.0,
        floor_tau_s: float = 4.0,
        warmup_s: float = 3.0,
        stuck_s: float = 20.0,
        stuck_flat_db: float = 2.0,
        silence_db: float = -150.0,
    ):
        if open_window_s < min_activity_s:
            raise ValueError("open_window_s must be at least min_activity_s")
        if floor_tau_s <= 0:
            raise ValueError("floor_tau_s must be > 0")
        self.open_snr_db = open_snr_db
        self.close_snr_db = close_snr_db
        self.min_activity_s = min_activity_s
        self.open_window_s = open_window_s
        self.hang_s = hang_s
        self.long_hang_s = max(long_hang_s, hang_s)
        self.long_hang_after_s = long_hang_after_s
        self.floor_tau_s = floor_tau_s
        self.warmup_s = warmup_s
        self.stuck_s = stuck_s
        self.stuck_flat_db = stuck_flat_db
        self.silence_db = silence_db
        self.noise_floor_db: float | None = None
        self.state = "closed"
        self.below_s = 0.0
        self.snr_history: list[float] = []
        self.close_reason: str | None = None
        self._t = 0.0  # stream time: the sum of all block lengths fed
        self._warm: list[float] = []
        self._warm_s = 0.0
        self._above: deque[tuple[float, float]] = deque()  # (block end, length) above open_snr_db
        self._recent: deque[tuple[float, float]] = deque()  # (block end, level) of valid blocks
        self._open_t = 0.0
        self._activity_s = min_activity_s

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    @property
    def activity_s(self) -> float:
        """How long before the last "open" event the activity began."""
        return self._activity_s

    def update(self, level_db: float, block_s: float) -> str | None:
        """Feed one block level. Returns "open", "close" or None."""
        self._t += block_s
        valid = math.isfinite(level_db) and level_db > self.silence_db
        recent = self._recent
        if valid:
            recent.append((self._t, level_db))
        while recent and recent[0][0] <= self._t - self.stuck_s:
            recent.popleft()

        if self.noise_floor_db is None:  # warm-up: learn, decide nothing
            if valid:
                self._warm.append(level_db)
                self._warm_s += block_s
                if self._warm_s >= self.warmup_s - 1e-9:
                    self.noise_floor_db = float(np.median(self._warm))
                    self._warm = []
            return None

        if self.state == "open":
            open_for = self._t - self._open_t
            snr = level_db - self.noise_floor_db if valid else None
            if snr is not None:
                self.snr_history.append(snr)
            loud = snr is not None and snr >= self.close_snr_db
            self.below_s = 0.0 if loud else self.below_s + block_s
            if open_for >= self.stuck_s and self._flat():
                self.noise_floor_db = float(np.median([lv for _, lv in recent]))
                return self._close("stuck")
            hang = self.long_hang_s if open_for >= self.long_hang_after_s else self.hang_s
            if self.below_s >= hang - 1e-9:
                return self._close("hang")
            return None

        if not valid:
            return None  # a dropout neither opens the squelch nor teaches the floor
        snr = level_db - self.noise_floor_db
        above = self._above
        if snr >= self.open_snr_db:
            above.append((self._t, block_s))
            self.snr_history.append(snr)
        while above and above[0][0] <= self._t - self.open_window_s:
            above.popleft()
        if above and sum(b for _, b in above) >= self.min_activity_s - 1e-9:
            self._activity_s = self._t - (above[0][0] - above[0][1])
            self._open_t = self._t - self._activity_s
            self.state = "open"
            self.below_s = 0.0
            self.close_reason = None
            above.clear()
            return "open"
        if snr < self.open_snr_db:  # follow the band
            weight = 1.0 - math.exp(-block_s / self.floor_tau_s)
            self.noise_floor_db += weight * (level_db - self.noise_floor_db)
        self.state = "pending" if above else "closed"
        if not above:
            self.snr_history.clear()
        return None

    def force_close(self) -> None:
        self.state = "closed"
        self.below_s = 0.0
        self._above.clear()

    def _flat(self) -> bool:
        if not self.stuck_flat_db or len(self._recent) < 10:
            return False
        return float(np.std([lv for _, lv in self._recent])) < self.stuck_flat_db

    def _close(self, reason: str) -> str:
        self.state = "closed"
        self.below_s = 0.0
        self.close_reason = reason
        self._above.clear()
        return "close"


class Resampler:
    """Streaming rational resampler (polyphase FIR) with state across blocks.

    Output sample ``n`` sits at input time ``n * in_rate / out_rate``; blocks
    of any size give the same output as one long call.
    """

    def __init__(self, in_rate: int, out_rate: int, taps_per_phase: int = 32):
        g = math.gcd(int(in_rate), int(out_rate))
        self.up = int(out_rate) // g
        self.down = int(in_rate) // g
        self.in_rate, self.out_rate = int(in_rate), int(out_rate)
        if self.up > 512:
            raise ValueError(f"unsupported resampling ratio {in_rate} -> {out_rate}")
        self.taps = taps_per_phase
        if self.up == self.down == 1:
            self.poly = None
        else:
            n = self.up * taps_per_phase
            cutoff = 0.9 / max(self.up, self.down)  # of the upsampled Nyquist
            h = signal.firwin(n, cutoff, window=("kaiser", 8.0)) * self.up
            # poly[p, j] = h[p + j*up], reversed so a dot product with
            # x[q-T+1 .. q] gives sum_j h[p + j*up] * x[q - j].
            self.poly = h.reshape(taps_per_phase, self.up).T[:, ::-1].copy()
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.taps - 1, np.float64)
        self.n_in = 0  # input samples consumed so far
        self.u = 0  # upsampled index of the next output sample

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.poly is None:
            return x.astype(np.float32)
        if len(x) == 0:
            return np.zeros(0, np.float32)
        buf = np.concatenate([self.hist, x])
        end_u = (self.n_in + len(x)) * self.up  # first upsampled index not yet available
        us = np.arange(self.u, end_u, self.down, dtype=np.int64)
        out = np.zeros(0, np.float32)
        if len(us):
            q = us // self.up
            p = us % self.up
            idx = q - (self.n_in - len(self.hist))  # index of x[q] inside buf
            win = np.lib.stride_tricks.sliding_window_view(buf, self.taps)
            out = np.einsum("ij,ij->i", win[idx - self.taps + 1], self.poly[p]).astype(np.float32)
            self.u = int(us[-1]) + self.down
        self.n_in += len(x)
        self.hist = buf[-(self.taps - 1) :]
        return out


class AudioFrontEnd:
    """Front end for sources that already deliver demodulated USB audio.

    Resamples to the pipeline audio rate and band-limits to the voice slot
    (stateful filters), returning the same ``(audio, slot_power)`` pair as
    :meth:`UsbDemodulator.process` so :class:`ChannelCapture` can use either.
    """

    def __init__(self, in_rate: int, audio_rate: int, low_hz: float = 300.0, high_hz: float = 3000.0):
        self.resampler = Resampler(in_rate, audio_rate)
        low = max(low_hz, 100.0)
        high = min(high_hz, 0.45 * audio_rate)
        self.sos = signal.butter(4, [low, high], btype="bandpass", fs=audio_rate, output="sos")
        self.frame = max(1, int(round(FRAME_SECONDS * audio_rate)))
        self.frame_power = np.zeros(0)  # 10 ms slot powers of the last block (median level)
        self.reset()

    def reset(self) -> None:
        self.resampler.reset()
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        y = self.resampler.process(np.asarray(audio, dtype=np.float64).real)
        if len(y) == 0:
            return np.zeros(0, np.float32), float("nan")
        y, self.zi = signal.sosfilt(self.sos, y, zi=self.zi)
        self.frame_power = frame_powers(y, self.frame)
        return y.astype(np.float32), float(np.mean(y.astype(np.float64) ** 2))
