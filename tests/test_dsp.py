import numpy as np
import pytest
from conftest import AUDIO_RATE, IQ_RATE, chunks, tone

from hfgcs.dsp import Agc, Squelch, UsbDemodulator

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


def _feed(sq, level, seconds, block=0.25):
    events = []
    left = seconds
    while left > 1e-9:
        dt = min(block, left)
        events.append(sq.update(level, dt))
        left -= dt
    return events


def test_squelch_stale_floor_raise_only_after_a_full_window_open():
    sq = Squelch(min_activity_s=0.4, hang_s=2.0, floor_window_s=10.0, quiet_median_s=None)
    _feed(sq, -60.0, 1.0)
    assert sq.noise_floor_db == pytest.approx(-60.0)
    _feed(sq, -48.0, 9.5)  # +12 dB, open, but the window is not full yet
    assert sq.is_open and sq.floor_raises == 0 and sq.noise_floor_db == pytest.approx(-60.0)
    events = _feed(sq, -48.0, 8.0)
    assert sq.floor_raises == 1
    assert sq.noise_floor_db == pytest.approx(-48.0)
    assert "close" in events


def test_squelch_floor_window_disabled_by_default():
    sq = Squelch()
    _feed(sq, -60.0, 1.0)
    _feed(sq, -48.0, 20.0)  # flat +12 dB, long enough to open and stay open
    assert sq.is_open
    assert sq.floor_raises == 0
    assert sq.noise_floor_db == pytest.approx(-60.0)


def test_squelch_quiet_median_closes_when_clicks_keep_resetting_hang():
    """0.5 s spikes every 1.5 s never finish a 2 s hang, but the median is the quiet bed."""
    sq = Squelch(min_activity_s=0.4, hang_s=2.0, floor_window_s=None, quiet_median_s=5.0)
    _feed(sq, -60.0, 1.0)
    closed_at = None
    t = 0.0
    while t < 12.0 and closed_at is None:
        phase = t % 1.5
        level = -48.0 if phase < 0.5 else -57.0  # snr +12, then +3
        if sq.update(level, 0.1) == "close":
            closed_at = t
        t += 0.1
    assert closed_at is not None and 4.0 < closed_at < 9.0
    assert sq.floor_raises == 0


def test_squelch_keyed_speech_does_not_median_close():
    """0.8 s on / 0.4 s off: the median is the signal, and the gaps are shorter than hang."""
    sq = Squelch(min_activity_s=0.4, hang_s=2.0, floor_window_s=30.0, quiet_median_s=5.0)
    _feed(sq, -60.0, 1.0)
    t = 0.0
    while t < 20.0:
        level = -48.0 if (t % 1.2) < 0.8 else -57.0
        assert sq.update(level, 0.1) != "close"
        t += 0.1
    assert sq.is_open and sq.floor_raises == 0
    assert sq.median_snr is not None and sq.median_snr >= sq.close_snr_db


def test_disarm_holds_the_stuck_level_until_a_louder_signal():
    sq = Squelch(min_activity_s=0.4, hang_s=2.0, floor_window_s=None, quiet_median_s=None)
    _feed(sq, -60.0, 1.0)
    _feed(sq, -48.0, 1.0)
    assert sq.is_open
    sq.disarm()
    assert not sq.is_open and sq._rearm and sq.floor_raises == 1
    assert sq.noise_floor_db == pytest.approx(-53.0)  # stuck level minus close_snr_db
    _feed(sq, -48.0, 3.0)
    assert not sq.is_open and sq._rearm
    events = _feed(sq, -40.0, 1.0)  # >= stuck level + open_snr_db
    assert "open" in events and not sq._rearm


def test_dip_below_close_clears_disarm_so_a_real_rise_records():
    sq = Squelch(min_activity_s=0.4, hang_s=2.0, floor_window_s=None, quiet_median_s=None)
    _feed(sq, -60.0, 1.0)
    _feed(sq, -48.0, 1.0)
    sq.disarm()
    # Just under the close threshold relative to the raised floor. Clears the hold.
    assert sq.update(-58.2, 0.2) is None
    assert not sq._rearm
    events = _feed(sq, -44.0, 1.0)  # snr >= open against the raised floor
    assert "open" in events


def test_release_drops_a_disarm_hold():
    sq = Squelch()
    _feed(sq, -60.0, 0.5)
    _feed(sq, -48.0, 1.0)
    sq.disarm()
    assert sq._rearm
    sq.release()
    assert not sq._rearm and not sq.is_open
