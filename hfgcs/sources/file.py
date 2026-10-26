"""File replay: WAV audio or IQ recordings, for tests and offline reprocessing.

* Audio WAV (mono, or the first channel of a multi-channel file that is not
  IQ): replayed as already-demodulated USB audio.
* IQ: a 2-channel WAV (I = left, Q = right; SDR# / SDR++ / kiwirecorder style)
  or a raw file of complex samples (``cf32``, ``cs16`` interleaved int16,
  ``cu8`` RTL-style offset bytes) with ``sample_rate`` and ``center_khz``.
  ``tune()`` shifts the wanted channel to baseband with a stateful mixer, so
  one wideband recording can be replayed for every channel inside it.

``realtime = true`` paces reads to the recording's rate.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..config import FileSettings
from ..dsp import Nco
from .base import BaseSource, Chunk, SourceError

RAW_FORMATS = {"cf32": (np.float32, 8), "cs16": (np.int16, 4), "cu8": (np.uint8, 2)}


class FileSource(BaseSource):
    name = "file"

    def __init__(self, settings: FileSettings, *, clock=None, sleep=None):
        if not settings.path:
            raise SourceError("file source needs a path (--file or [file] path)")
        self.settings = settings
        self.path = Path(settings.path)
        super().__init__(settings.receiver_id or f"file-{self.path.stem}")
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.format = self._detect_format()
        self._rate = 0
        self._channels = 1
        self._probe()
        self.kind = self._detect_kind()
        self.single_channel = self.kind == "audio"
        self.squelch_level = "slot"
        self.nco: Nco | None = None
        self._fh = None
        self._raw = None
        self._pos = 0
        self._t_start = 0.0
        self._sent = 0

    # -- format detection ------------------------------------------------------
    def _detect_format(self) -> str:
        fmt = self.settings.format
        if fmt != "auto":
            if fmt not in ("wav", "iq-wav", *RAW_FORMATS):
                raise SourceError(f"unknown file format {fmt!r}")
            return fmt
        suffix = self.path.suffix.lower().lstrip(".")
        if suffix in RAW_FORMATS:
            return suffix
        if suffix in ("raw", "iq", "bin"):
            raise SourceError(f"{self.path}: set [file] format (cf32, cs16 or cu8) for raw IQ")
        return "wav"

    def _probe(self) -> None:
        if not self.path.exists():
            raise SourceError(f"{self.path}: no such file")
        if self.format in RAW_FORMATS:
            if not self.settings.sample_rate:
                raise SourceError(f"{self.path}: raw IQ needs [file] sample_rate")
            self._rate = int(self.settings.sample_rate)
            self._channels = 2
            return
        import soundfile as sf

        info = sf.info(str(self.path))
        self._rate, self._channels = int(info.samplerate), int(info.channels)

    def _detect_kind(self) -> str:
        kind = self.settings.kind
        if kind == "auto":
            iq = self.format in RAW_FORMATS or self.format == "iq-wav" or self._channels == 2
            kind = "iq" if iq else "audio"
        if kind not in ("iq", "audio"):
            raise SourceError(f"[file] kind must be auto, iq or audio, not {kind!r}")
        if kind == "iq" and self._channels < 2:
            raise SourceError(f"{self.path}: IQ replay needs a 2-channel WAV or a raw IQ format")
        return kind

    @property
    def sample_rate(self) -> int:
        return self._rate

    # -- lifecycle ---------------------------------------------------------------
    def open(self) -> None:
        if self.format in RAW_FORMATS:
            dtype, _ = RAW_FORMATS[self.format]
            self._raw = np.memmap(self.path, dtype=dtype, mode="r")
        else:
            import soundfile as sf

            self._fh = sf.SoundFile(str(self.path))
        self._pos = 0
        self._t_start = self._clock()
        self._sent = 0
        super().open()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh, self._raw = None, None
        super().close()

    # -- tuning --------------------------------------------------------------------
    def refusal(self, dial_khz: float) -> str | None:
        s = self.settings
        if self.kind == "audio":
            if s.dial_khz and abs(dial_khz - s.dial_khz) > 0.01:
                return f"this audio recording was made on {s.dial_khz:g} kHz"
            return None
        if not s.center_khz:
            return None  # unknown centre: treat the file as already at the dial frequency
        offset_hz = (dial_khz - s.center_khz) * 1e3
        half = 0.45 * self._rate
        if offset_hz < -half or offset_hz + 3000.0 > half:
            return f"outside the recording ({s.center_khz:g} kHz ±{half / 1e3:g} kHz)"
        return None

    def tune(self, dial_khz: float) -> None:
        super().tune(dial_khz)
        if self.kind == "iq" and self.settings.center_khz:
            offset_hz = (dial_khz - self.settings.center_khz) * 1e3
            self.nco = Nco(-offset_hz, self._rate) if offset_hz else None

    # -- reading -----------------------------------------------------------------------
    def _read_block(self, n: int) -> np.ndarray:
        if self._raw is not None:
            dtype, _ = RAW_FORMATS[self.format]
            a = np.asarray(self._raw[2 * self._pos : 2 * (self._pos + n)], dtype=np.float32)
            a = a[: len(a) // 2 * 2]
            if dtype is np.int16:
                a = a / 32768.0
            elif dtype is np.uint8:
                a = (a - 127.5) / 127.5
            self._pos += len(a) // 2
            return (a[0::2] + 1j * a[1::2]).astype(np.complex64)
        data = self._fh.read(n, dtype="float32", always_2d=True)
        self._pos += len(data)
        if self.kind == "iq":
            return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
        return data[:, 0].astype(np.float32)

    def _rewind(self) -> None:
        self._pos = 0
        if self._fh is not None:
            self._fh.seek(0)

    def read(self, timeout: float | None = None) -> Chunk | None:
        if not self.is_open:
            raise SourceError("file source is not open")
        n = max(1, int(math.ceil(self.settings.block_seconds * self._rate)))
        block = self._read_block(n)
        gap = False
        if len(block) == 0 and self.settings.loop:
            self._rewind()
            block, gap = self._read_block(n), True
        if len(block) == 0:
            return None
        if self.nco is not None:
            block = self.nco.mix(block).astype(np.complex64)
        if self.settings.realtime:
            due = self._t_start + (self._sent + len(block)) / self._rate
            wait = due - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._sent += len(block)
        return Chunk(block, self._rate, gap=gap)
