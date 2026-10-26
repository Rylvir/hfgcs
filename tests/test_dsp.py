import numpy as np
import pytest
from conftest import AUDIO_RATE, IQ_RATE, chunks, tone

from hfgcs.dsp import Agc, UsbDemodulator

SETTLE = 1000  # audio samples to skip while the filter fills


def demod(iq, sizes=None):
    d = UsbDemodulator(IQ_RATE, AUDIO_RATE)
    if sizes is None:
        return d.process(iq)[0]
    out, i = [], 0
    for n in sizes:
        out.append(d.process(iq[i : i + n])[0])
        i += n
    out.append(d.process(iq[i:])[0])
    return np.concatenate(out)


def peak_hz(x, rate=AUDIO_RATE, pad=8):
    x = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x, n=pad * len(x)))
    return np.argmax(spec) * rate / (pad * len(x))


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-15)


@pytest.mark.parametrize("f", [300.0, 500.0, 1000.0, 1700.0, 2500.0, 2800.0])
def test_usb_tone_frequency(f):
    out = demod(tone(f, 1.0))[SETTLE:]
    assert abs(peak_hz(out) - f) < 1.0
    assert abs(rms_db(out) - rms_db(np.ones(1) / np.sqrt(2))) < 0.5  # passband gain ~ 0 dB


@pytest.mark.parametrize("f", [-300.0, -500.0, -1000.0, -2500.0])
def test_lsb_rejection(f):
    ref = rms_db(demod(tone(-f, 1.0))[SETTLE:])
    lsb = rms_db(demod(tone(f, 1.0))[SETTLE:])
    assert ref - lsb > 40.0


def test_out_of_slot_rejected():
    out = demod(tone(20_000.0, 1.0))[SETTLE:]
    assert rms_db(out) < -60


def test_chunked_equals_one_shot():
    rng = np.random.default_rng(7)
    iq = tone(1234.0, 1.0) + 0.3 * tone(2222.0, 1.0, phase=1.0)
    sizes = rng.integers(1, 9000, size=40)
    sizes = sizes[np.cumsum(sizes) < len(iq)]
    np.testing.assert_allclose(demod(iq, sizes), demod(iq), atol=1e-9)


def test_no_clicks_at_chunk_boundaries():
    f = 1000.0
    d = UsbDemodulator(IQ_RATE, AUDIO_RATE)
    out = np.concatenate([d.process(c)[0] for c in chunks(tone(f, 2.0), 8192)])[SETTLE:]
    # Fit and remove the ideal sinusoid: residual (incl. any boundary
    # discontinuity) must be far below the tone.
    n = np.arange(len(out))
    basis = np.column_stack([np.cos(2 * np.pi * f * n / AUDIO_RATE), np.sin(2 * np.pi * f * n / AUDIO_RATE)])
    coef, *_ = np.linalg.lstsq(basis, out, rcond=None)
    resid = out - basis @ coef
    assert rms_db(resid) - rms_db(out) < -50
    assert np.max(np.abs(resid)) < 0.01


def test_agc_chunked_equals_one_shot_and_levels():
    rng = np.random.default_rng(3)
    t = np.arange(AUDIO_RATE * 3) / AUDIO_RATE
    x = (0.01 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
    x[AUDIO_RATE:] *= 20  # level step
    one = Agc(AUDIO_RATE).process(x)
    agc = Agc(AUDIO_RATE)
    parts, i = [], 0
    for n in rng.integers(1, 1000, size=200):
        parts.append(agc.process(x[i : i + n]))
        i += n
    parts.append(agc.process(x[i:]))
    np.testing.assert_allclose(np.concatenate(parts), one, atol=1e-6)
    # Settles to target peak on both levels (no per-chunk normalisation).
    for seg in (one[AUDIO_RATE // 2 : AUDIO_RATE], one[2 * AUDIO_RATE :]):
        assert 0.2 < np.max(np.abs(seg)) < 0.45
