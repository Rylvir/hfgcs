"""Local SoapySDR receiver (Airspy HF+ by default), the original capture path.

SoapySDR comes from the system (``sudo apt install python3-soapysdr``) and is
imported lazily. Only use this when nothing else owns the radio: while
OpenWebRX is running it holds the Airspy, so use the ``openwebrx`` source.
"""

from __future__ import annotations

import time

import numpy as np

from .base import BaseSource, Chunk, SourceError

SETTLE_SECONDS = 0.12  # discard samples right after a retune
BLOCK_SAMPLES = 8192


def load_soapy():
    try:
        import SoapySDR
    except ImportError as exc:  # pragma: no cover - needs the hardware stack
        raise SourceError("SoapySDR Python bindings not found (sudo apt install python3-soapysdr)") from exc
    return SoapySDR


class SoapySource(BaseSource):
    name = "soapy"
    kind = "iq"
    squelch_level = "slot"

    def __init__(self, cfg, soapy_module=None):
        super().__init__(cfg.receiver_id)
        self.cfg = cfg
        self._soapy = soapy_module
        self.sdr = None
        self.stream = None

    @property
    def sample_rate(self) -> int:
        return int(self.cfg.iq_rate)

    def open(self) -> None:
        S = self._soapy or load_soapy()
        self._soapy = S
        try:
            self.sdr = S.Device(self.cfg.soapy_args)
        except Exception as exc:
            raise SourceError(
                f"could not open SoapySDR device {self.cfg.soapy_args!r}: {exc}. "
                "If OpenWebRX is using the radio, use --source openwebrx instead."
            ) from exc
        self.sdr.setSampleRate(S.SOAPY_SDR_RX, 0, self.cfg.iq_rate)
        # Tuner AGC off by default: it makes dB squelch thresholds meaningless.
        try:
            self.sdr.setGainMode(S.SOAPY_SDR_RX, 0, self.cfg.tuner_agc)
        except Exception:
            pass
        self.stream = self.sdr.setupStream(S.SOAPY_SDR_RX, S.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.stream)
        super().open()
        end = time.time() + 2.0  # the HF+ returns timeouts until the first buffer
        while time.time() < end and not len(self._read_iq()):
            pass

    def close(self) -> None:
        if self.sdr is not None and self.stream is not None:
            for call in (self.sdr.deactivateStream, self.sdr.closeStream):
                try:
                    call(self.stream)
                except Exception:
                    pass
        self.sdr = self.stream = None
        super().close()

    def device_info(self) -> dict:
        try:
            return dict(self.sdr.getHardwareInfo())
        except Exception:
            return {}

    def _read_iq(self, nsamps: int = BLOCK_SAMPLES) -> np.ndarray:
        buff = np.zeros(nsamps, np.complex64)
        sr = self.sdr.readStream(self.stream, [buff], nsamps)
        if sr.ret > 0:
            return buff[: sr.ret]
        return np.zeros(0, np.complex64)

    def tune(self, dial_khz: float) -> None:
        """Tune the hardware, then drop the samples produced while the PLL settles."""
        super().tune(dial_khz)
        self.sdr.setFrequency(self._soapy.SOAPY_SDR_RX, 0, dial_khz * 1000)
        end = time.time() + SETTLE_SECONDS
        while time.time() < end:
            self._read_iq()

    def read(self, timeout: float | None = None) -> Chunk | None:
        return Chunk(self._read_iq(), self.sample_rate)
