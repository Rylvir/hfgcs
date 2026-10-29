"""Streaming SSB DSP: USB demodulator, AGC and squelch.

Every block keeps its state between calls (mixer phase, filter memory,
decimator phase, AGC envelope), so a stream cut into arbitrary chunks gives
the same output as one long call. Nothing here touches hardware.
"""

from __future__ import annotations

import math
from collections import deque

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
    """Voice-slot squelch with hysteresis, hang time and a mostly frozen noise floor.

    Time is counted from the length of the blocks fed in, not the wall clock,
    so the behaviour is deterministic and testable.
    States: ``closed`` -> ``pending`` (above threshold, not yet long enough)
    -> ``open`` -> back to ``closed`` after ``hang_s`` below the close level,
    or sooner when the median SNR over ``quiet_median_s`` is below the close
    level (a click resets the hang; it does not keep a quiet channel open).
    The noise floor follows the band (EMA) only while ``closed``.

    Stale-floor recovery (``floor_window_s``): voice has pauses, so over any
    ``floor_window_s`` of real traffic the quietest block sits near the noise.
    If the squelch has been open that long and even its quietest block in the
    window is more than ``close_snr_db`` above the floor, the floor is stale
    (learned too low, e.g. from a KiwiSDR stream's first blocks while its
    server-side AGC settles): it is raised to that minimum, so the squelch
    then closes after ``hang_s`` instead of recording noise until
    ``max_record_seconds``. The floor is only ever raised this way, never
    lowered. ``None`` or 0 disables it.

    ``disarm`` is the length-cap path for a channel that is open but not a
    real signal: raise the floor the same way and hold closed until the level
    falls below the close threshold, or rises ``open_snr_db`` above the level
    that was capped. A real signal still running at the cap is left armed so
    the next file can continue it.
    """

    def __init__(
        self,
        open_snr_db: float = 8.0,
        close_snr_db: float = 5.0,
        min_activity_s: float = 0.8,
        hang_s: float = 4.0,
        alpha: float = 0.05,
        floor_window_s: float | None = None,
        quiet_median_s: float | None = None,
    ):
        self.open_snr_db = open_snr_db
        self.close_snr_db = close_snr_db
        self.min_activity_s = min_activity_s
        self.hang_s = hang_s
        self.alpha = alpha
        self.floor_window_s = floor_window_s or None
        self.quiet_median_s = quiet_median_s or None
        self.noise_floor_db: float | None = None
        self.floor_raises = 0  # times the floor was raised while open or at the cap
        self._t = 0.0  # stream time (sum of block lengths)
        self._open_since: float | None = None
        self._mins: deque[tuple[float, float]] = deque()  # (block end time, level), increasing levels
        self._snr_win: deque[tuple[float, float]] = deque()  # (time, snr) while open
        self.median_snr: float | None = None
        self.last_level_db: float | None = None
        self.last_snr: float | None = None
        self._rearm = False
        self._rearm_level: float | None = None
        self.state = "closed"
        self.pending_s = 0.0
        self.below_s = 0.0
        self.snr_history: list[float] = []

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    @property
    def stream_s(self) -> float:
        return self._t

    def update(self, level_db: float, block_s: float) -> str | None:
        """Feed one block level. Returns "open", "close" or None."""
        if math.isnan(level_db):
            return None
        self._t += block_s
        if self.noise_floor_db is None:
            self.noise_floor_db = level_db
        self.last_level_db = level_db
        if self.floor_window_s:
            self._track_min(level_db)
        snr = level_db - self.noise_floor_db
        self.last_snr = snr

        if self._rearm:
            louder = (
                self._rearm_level is not None
                and level_db >= self._rearm_level + self.open_snr_db
            )
            if snr < self.close_snr_db or louder:
                self._rearm = False
                self._rearm_level = None
            else:
                return None

        if self.state == "open":
            if self._stale_floor():
                self.noise_floor_db = self._mins[0][1]
                self.floor_raises += 1
                snr = level_db - self.noise_floor_db
                self.last_snr = snr
            self.snr_history.append(snr)
            self._track_median(snr)
            self.below_s = 0.0 if snr >= self.close_snr_db else self.below_s + block_s
            quiet_median = self.median_snr is not None and self.median_snr < self.close_snr_db
            if self.below_s >= self.hang_s or quiet_median:
                self._to_closed()
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
                self._open_since = self._t - self.min_activity_s
                self._snr_win.clear()
                self.median_snr = None
                self._track_median(snr)
                return "open"
            return None

        # Below threshold while closed or pending: drop the pending burst and
        # let the noise floor follow the band.
        self._to_closed()
        self.pending_s = 0.0
        self.snr_history.clear()
        self.noise_floor_db += self.alpha * (level_db - self.noise_floor_db)
        self.last_snr = level_db - self.noise_floor_db
        return None

    def force_close(self) -> None:
        """Close without raising the floor or holding the channel shut."""
        self._to_closed()
        self.pending_s = 0.0

    def release(self) -> None:
        """New stream segment (a retune). Keeps the floor, drops any hold."""
        self.force_close()
        self._rearm = False
        self._rearm_level = None

    def disarm(self) -> None:
        """Length cap on a stuck channel: raise the floor and do not reopen on it."""
        if self.noise_floor_db is not None:
            raised = None
            if self._mins and self._mins[0][1] > self.noise_floor_db:
                raised = self._mins[0][1]
            elif self.last_level_db is not None:
                candidate = self.last_level_db - self.close_snr_db
                if candidate > self.noise_floor_db:
                    raised = candidate
            if raised is not None:
                self.noise_floor_db = raised
                self.floor_raises += 1
        self._rearm_level = self.last_level_db
        self.force_close()
        self._rearm = True

    def _to_closed(self) -> None:
        self.state = "closed"
        self.below_s = 0.0
        self._open_since = None
        self._snr_win.clear()
        self.median_snr = None

    def _track_min(self, level_db: float) -> None:
        """Sliding-window minimum of the block levels (monotonic deque)."""
        q = self._mins
        while q and q[-1][1] >= level_db:
            q.pop()
        q.append((self._t, level_db))
        cutoff = self._t - self.floor_window_s
        while q and q[0][0] <= cutoff:
            q.popleft()

    def _track_median(self, snr: float) -> None:
        if not self.quiet_median_s or self._open_since is None:
            self.median_snr = None
            return
        q = self._snr_win
        q.append((self._t, snr))
        cutoff = self._t - self.quiet_median_s
        while q and q[0][0] <= cutoff:
            q.popleft()
        if self._t - self._open_since < self.quiet_median_s - 1e-9 or len(q) < 2:
            self.median_snr = None
            return
        self.median_snr = float(np.median([s for _, s in q]))

    def _stale_floor(self) -> bool:
        if not self.floor_window_s or self._open_since is None or not self._mins:
            return False
        if self._t - self._open_since < self.floor_window_s:
            return False  # the window must hold only audio from this open period
        return self._mins[0][1] - self.noise_floor_db > self.close_snr_db


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
        self.reset()

    def reset(self) -> None:
        self.resampler.reset()
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        y = self.resampler.process(np.asarray(audio, dtype=np.float64).real)
        if len(y) == 0:
            return np.zeros(0, np.float32), float("nan")
        y, self.zi = signal.sosfilt(self.sos, y, zi=self.zi)
        return y.astype(np.float32), float(np.mean(y.astype(np.float64) ** 2))
