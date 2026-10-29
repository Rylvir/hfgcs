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


# -- v2 squelch (docs/squelch.md) -----------------------------------------------------------
BLOCK = 0.085  # a KiwiSDR block (1024 samples at 12 kHz)


def feed_v(sq, level, seconds, block=BLOCK):
    return [sq.update(level, block) for _ in range(int(round(seconds / block)))]


def word_levels(seconds, on=-48.0, off=-60.0, word=0.5, gap=0.4, block=BLOCK):
    """Block levels of a message read word by word (no phrase longer than 0.8 s)."""
    return [on if (i * block) % (word + gap) < word else off for i in range(int(round(seconds / block)))]


def test_frame_powers_split_a_block_into_10ms_slices():
    from hfgcs.dsp import frame_powers

    p = frame_powers(np.r_[np.ones(80), 3 * np.ones(80), np.ones(80)], 80)
    assert p.tolist() == [1.0, 9.0, 1.0]
    assert len(frame_powers(np.ones(245), 80)) == 3          # a short tail joins the slices
    assert len(frame_powers(np.ones(30), 80)) == 1 and len(frame_powers(np.zeros(0), 80)) == 0
    assert frame_powers(np.full(160, 2j), 80).tolist() == [4.0, 4.0]


def test_v1_ignores_non_finite_levels():
    from hfgcs.dsp import Squelch

    sq = Squelch(8.0, 5.0, 0.8, 4.0, 0.02)
    sq.update(float("inf"), 0.1)            # an S-meter "Infinity" made the floor NaN for good
    assert sq.noise_floor_db is None
    for _ in range(30):
        sq.update(-60.0, 0.1)
    assert "open" in [sq.update(-40.0, 0.1) for _ in range(20)]


def test_v2_warmup_floor_is_a_median_and_dropouts_teach_nothing():
    from hfgcs.dsp import SquelchV2

    sq = SquelchV2()
    events = feed_v(sq, -200.0, 0.5) + feed_v(sq, -72.0, 1.0) + feed_v(sq, -60.0, 3.0)
    assert set(events) == {None}            # no decisions while warming up
    assert -60.5 < sq.noise_floor_db < -59.5   # not the AGC-settling -72 (v1 would take the -200)
    events = feed_v(sq, -200.0, 0.5) + feed_v(sq, float("inf"), 0.3) + feed_v(sq, float("nan"), 0.3)
    events += feed_v(sq, -60.0, 10.0)
    assert "open" not in events and -60.5 < sq.noise_floor_db < -59.5


def test_v2_opens_on_word_by_word_speech_that_v1_never_opens_on():
    from hfgcs.dsp import Squelch, SquelchV2

    v1, v2 = Squelch(8.0, 5.0, 0.8, 4.0, 0.02), SquelchV2()
    for sq in (v1, v2):
        feed_v(sq, -60.0, 4.0)
    levels = word_levels(30.0)
    assert "open" not in [v1.update(lv, BLOCK) for lv in levels]
    events = [v2.update(lv, BLOCK) for lv in levels]
    assert events.count("open") == 1 and "close" not in events
    assert 0.8 <= v2.activity_s <= 2.0


def test_v2_closes_a_flat_noise_bed_and_relearns_the_floor():
    from hfgcs.dsp import SquelchV2

    sq = SquelchV2()
    feed_v(sq, -60.0, 4.0)
    events = feed_v(sq, -48.0, 30.0)        # the band noise rises 12 dB and stays
    assert events.count("open") == 1 and events.count("close") == 1 and sq.close_reason == "stuck"
    open_s = (events.index("close") - events.index("open")) * BLOCK
    assert 18.5 < open_s < 21.0
    assert abs(sq.noise_floor_db + 48.0) < 0.5
    assert "open" not in feed_v(sq, -48.0, 30.0)    # and does not reopen on the same bed


def test_v2_keeps_a_long_message_with_hiss_in_its_pauses():
    """The level moves with the speech, so it is not 'stuck', and pauses 8 dB over the band
    noise (transmitter keyed) are still above the close level: the 150 s message stays whole."""
    from hfgcs.dsp import SquelchV2

    sq = SquelchV2()
    feed_v(sq, -60.0, 4.0)
    levels = [-35.0] * int(3 / BLOCK) + word_levels(87.0, on=-35.0, off=-52.0)
    levels += [-52.0] * int(5 / BLOCK) + word_levels(60.0, on=-35.0, off=-52.0)
    events = [sq.update(lv, BLOCK) for lv in levels]
    assert events.count("open") == 1 and "close" not in events
    after = feed_v(sq, -60.0, 10.0)         # transmitter off
    assert after.count("close") == 1 and sq.close_reason == "hang"
    assert 7.5 < after.index("close") * BLOCK < 8.5   # the long hang, after a long capture


def test_v2_long_hang_only_after_a_long_capture():
    from hfgcs.dsp import SquelchV2

    sq = SquelchV2()
    feed_v(sq, -60.0, 4.0)
    early = feed_v(sq, -45.0, 3.0) + feed_v(sq, -60.0, 5.0)      # 5 s fade 3 s in: closes (4 s hang)
    assert early.count("open") == 1 and early.count("close") == 1
    late = feed_v(sq, -45.0, 15.0) + feed_v(sq, -60.0, 6.0) + feed_v(sq, -45.0, 2.0)
    assert late.count("open") == 1 and "close" not in late       # 6 s fade 15 s in: bridged (8 s)


def test_v2_settings_are_checked():
    from hfgcs.dsp import SquelchV2

    with pytest.raises(ValueError):
        SquelchV2(min_activity_s=1.0, open_window_s=0.5)
    with pytest.raises(ValueError):
        SquelchV2(floor_tau_s=0.0)
