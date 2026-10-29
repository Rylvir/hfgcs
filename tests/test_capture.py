import json
import logging
from datetime import timedelta

import numpy as np
import pytest
from conftest import T0, chunks, noise, tone

from hfgcs.capture import ChannelCapture, save_recording
from hfgcs.config import Config

NOISE_AMP = 0.01
SIG_AMP = 0.01  # about 18 dB SNR in the 3 kHz slot


def run(cfg, segments, seed=0):
    """segments: list of (seconds, has_signal[, freq_offset])."""
    cap = ChannelCapture(cfg, 11175.0, start_utc=T0)
    recs, seed_i = [], seed
    for seg in segments:
        secs, on = seg[0], seg[1]
        f = seg[2] if len(seg) > 2 else 1000.0
        x = noise(secs, NOISE_AMP, seed=seed_i)
        seed_i += 1
        if on:
            x = x + tone(f, secs, SIG_AMP)
        for c in chunks(x):
            recs.extend(cap.feed(c))
    return cap, recs


def cfg(**kw):
    base = dict(hang_seconds=2.0, preroll_seconds=1.5, min_activity_seconds=0.8)
    base.update(kw)
    return Config(**base)


def test_open_hang_close_single_recording():
    c = cfg()
    # 3 s noise, 5 s signal, 1 s gap (< hang), 4 s signal, 5 s noise
    cap, recs = run(c, [(3, False), (5, True), (1, False), (4, True), (5, False)])
    assert len(recs) == 1
    r = recs[0]
    assert abs((r.start_utc - T0).total_seconds() - 3.0) < 0.1
    # pre-roll + 5 + 1 + 4 + hang
    assert abs(r.duration_sec - (c.preroll_seconds + 10.0 + c.hang_seconds)) < 0.2
    assert r.snr_db_peak > 12
    assert r.receiver_id == c.receiver_id
    assert not r.truncated
    assert not cap.active


def test_gap_longer_than_hang_splits():
    _, recs = run(cfg(), [(3, False), (3, True), (4, False), (3, True), (4, False)])
    assert len(recs) == 2


def test_long_transmission_not_truncated():
    _, recs = run(cfg(), [(2, False), (30, True), (4, False)])
    assert len(recs) == 1
    assert recs[0].duration_sec > 30


def test_short_burst_ignored():
    _, recs = run(cfg(), [(3, False), (0.3, True), (4, False)])
    assert recs == []


@pytest.mark.parametrize("preroll", [0.8, 1.5, 3.0])
def test_preroll_length(preroll):
    c = cfg(preroll_seconds=preroll)
    _, recs = run(c, [(5, False), (3, True), (4, False)])
    assert len(recs) == 1
    r = recs[0]
    assert abs(r.preroll_seconds - preroll) < 0.05
    assert abs((r.start_utc - r.file_start_utc).total_seconds() - preroll) < 0.05
    # Pre-roll audio is noise; signal starts right after it.
    n = int(preroll * c.audio_rate)
    assert np.sqrt(np.mean(r.audio[n + 800 : n + 4000] ** 2)) > 0.1


def test_onset_audio_not_dropped():
    """Samples during the min-activity window must be in the file (bug: dropped)."""
    c = cfg()
    _, recs = run(c, [(3, False), (3, True), (4, False)])
    r = recs[0]
    # signal (onset at 3 s) occupies from file offset preroll onward
    assert r.duration_sec >= c.preroll_seconds + 3.0 + c.hang_seconds - 0.1


def test_noise_floor_frozen_during_traffic():
    c = cfg()
    cap = ChannelCapture(c, 8992.0, start_utc=T0)
    for ch in chunks(noise(3, NOISE_AMP, seed=1)):
        cap.feed(ch)
    floor = cap.squelch.noise_floor_db
    x = noise(10, NOISE_AMP, seed=2) + tone(1000, 10, SIG_AMP)
    for ch in chunks(x):
        cap.feed(ch)
        if cap.squelch.is_open:
            assert cap.squelch.noise_floor_db == floor


def test_squelch_uses_voice_slot_not_full_band():
    # A strong carrier 40 kHz away must not open the squelch.
    _, recs = run(cfg(), [(3, False), (5, True, 40_000.0), (4, False)])
    assert recs == []


def test_max_record_cap():
    _, recs = run(cfg(max_record_seconds=5.0), [(2, False), (8, True)])
    assert recs and recs[0].truncated
    assert recs[0].duration_sec <= 5.1


def test_save_recording_catalog(tmp_path):
    c = cfg(receiver_id="test-rx")
    _, recs = run(c, [(3, False), (3, True), (4, False)])
    entry = save_recording(recs[0], tmp_path / "rec", tmp_path / "cat.jsonl")
    line = json.loads((tmp_path / "cat.jsonl").read_text().strip())
    assert line == entry
    assert line["receiver_id"] == "test-rx"
    assert line["start_utc"].endswith("Z")
    expected = (T0 + timedelta(seconds=3)).isoformat()[:19]
    assert line["start_utc"][:19] == expected
    assert line["snr_db_peak"] > 10
    assert line["frequency_khz"] == 11175.0
    assert line["continued_from"] is None
    assert (tmp_path / "rec").glob("*.wav")
    import soundfile as sf

    audio, rate = sf.read(line["file"])
    assert rate == 8000 and abs(len(audio) / rate - line["duration_sec"]) < 0.01


# Audio-rate front end: the squelch sees slot power, and a minute of audio stays cheap.
# A tone at 0.08 on noise of 0.01 is about +16 dB in the slot. Noise at 4x amplitude
# is about +12 dB and has no pauses, so the stale-floor window treats it as not voice.


def audio_cap(c, khz=8992.0):
    return ChannelCapture(c, khz, start_utc=T0, input_kind="audio", input_rate=8000)


def feed_audio(cap, seconds, amp, tone_amp=0.0, seed=0):
    rate = 8000
    n = int(round(seconds * rate))
    rng = np.random.default_rng(seed)
    x = (amp * rng.standard_normal(n)).astype(np.float32)
    if tone_amp:
        t = np.arange(n) / rate
        x = x + (tone_amp * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    recs = []
    for i in range(0, n, 400):
        recs.extend(cap.feed(x[i : i + 400]))
    return recs


def reported_cap(c, khz=11175.0):
    return ChannelCapture(
        c, khz, start_utc=T0, input_kind="audio", input_rate=8000, squelch_level="reported"
    )


def feed_level(cap, seconds, level_db, block=400):
    n = int(round(seconds * 8000))
    recs = []
    for i in range(0, n, block):
        recs.extend(cap.feed(np.zeros(min(block, n - i), np.float32), level_db=level_db))
    return recs


def test_noise_bed_raises_stale_floor_and_closes():
    c = cfg()  # floor window 30 s, hang 2 s
    cap = audio_cap(c)
    feed_audio(cap, 3.0, 0.01, seed=1)
    recs = feed_audio(cap, 50.0, 0.04, seed=2)
    assert cap.squelch.floor_raises >= 1
    assert len(recs) == 1 and not recs[0].truncated
    # pre-roll + one floor window + hang, plus a second of slack
    assert recs[0].duration_sec < c.preroll_seconds + c.noise_floor_window_seconds + c.hang_seconds + 1.0
    assert not cap.squelch.is_open


def test_floor_window_disabled_lets_a_loud_bed_continue_past_the_cap():
    c = cfg(noise_floor_window_seconds=0, max_record_seconds=60.0)
    cap = audio_cap(c)
    feed_audio(cap, 2.0, 0.01, seed=1)
    recs = feed_audio(cap, 70.0, 0.04, seed=3)
    assert recs and recs[0].truncated and recs[0].duration_sec >= 59.0
    assert cap.squelch.is_open and not cap.squelch._rearm  # median is still a real signal
    recs += feed_audio(cap, 4.0, 0.01, seed=4)
    assert len(recs) == 2
    assert recs[1].continued_from == recs[0].start_utc


def test_floor_learned_in_silence_recovers_and_still_hears_voice():
    c = cfg(noise_floor_window_seconds=20.0, quiet_median_seconds=0)
    cap = audio_cap(c)
    feed_audio(cap, 0.2, 1e-4, seed=1)
    recs = feed_audio(cap, 40.0, 0.01, seed=4)
    assert cap.squelch.floor_raises >= 1
    assert len(recs) == 1 and not cap.squelch.is_open
    assert recs[0].duration_sec < c.preroll_seconds + 20.0 + c.hang_seconds + 1.0
    recs += feed_audio(cap, 3.0, 0.01, tone_amp=0.08, seed=5)
    recs += feed_audio(cap, 3.0, 0.01, seed=6)
    assert len(recs) == 2 and not recs[1].truncated


def test_keyed_speech_does_not_raise_the_floor():
    c = cfg()
    cap = audio_cap(c)
    feed_audio(cap, 3.0, 0.01, seed=1)
    recs = []
    t = 0.0
    seed = 10
    while t < 90.0 - 1e-9:
        dur = min(0.1, 90.0 - t)
        if (t % 1.2) < 0.8:
            recs += feed_audio(cap, dur, 0.01, tone_amp=0.08, seed=seed)
        else:
            recs += feed_audio(cap, dur, 0.01, seed=seed)
        seed += 1
        t += dur
    assert recs == [] and cap.squelch.is_open and cap.squelch.floor_raises == 0
    recs += feed_audio(cap, 4.0, 0.01, seed=99)
    assert len(recs) == 1 and recs[0].duration_sec > 90.0 and not recs[0].truncated
    assert cap.squelch.floor_raises == 0


def test_cap_on_a_real_signal_continues(tmp_path):
    c = cfg(max_record_seconds=6.0, noise_floor_window_seconds=0, quiet_median_seconds=2.0)
    cap = reported_cap(c)
    recs = feed_level(cap, 2.0, -60.0)
    recs += feed_level(cap, 8.0, -40.0)  # snr +20, long enough for one cap and a continuation
    recs += feed_level(cap, 4.0, -60.0)
    assert len(recs) == 2
    assert recs[0].truncated and recs[0].continued_from is None
    assert recs[1].continued_from == recs[0].start_utc
    assert not cap.squelch._rearm
    entry = save_recording(recs[1], tmp_path / "rec", tmp_path / "cat.jsonl")
    continued = recs[0].start_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    assert entry["continued_from"] == continued
    assert entry["truncated"] is False


def test_cap_on_a_stuck_channel_does_not_reopen_until_louder():
    c = cfg(
        max_record_seconds=6.0,
        noise_floor_window_seconds=0,
        quiet_median_seconds=2.0,
        min_activity_seconds=0.5,
    )
    cap = reported_cap(c)
    recs = feed_level(cap, 1.0, -60.0)
    recs += feed_level(cap, 0.6, -48.0)  # snr +12, long enough to open
    recs += feed_level(cap, 8.0, -53.5)  # snr +6.5: above close, below open
    assert len(recs) == 1 and recs[0].truncated and recs[0].continued_from is None
    assert cap.squelch.floor_raises >= 1 and cap.squelch._rearm and not cap.squelch.is_open
    assert feed_level(cap, 3.0, -53.5) == []
    assert not cap.squelch.is_open
    recs += feed_level(cap, 2.0, -45.0)  # >= the disarmed level + open_snr_db
    recs += feed_level(cap, 3.0, -60.0)
    assert len(recs) == 2 and recs[1].continued_from is None and not recs[1].truncated


def test_squelch_logs_open_close_cap_and_a_tick(caplog):
    c = cfg(max_record_seconds=4.0, noise_floor_window_seconds=0, quiet_median_seconds=2.0)
    cap = reported_cap(c, khz=8992.0)
    with caplog.at_level(logging.INFO, logger="hfgcs.squelch"):
        feed_level(cap, 1.0, -60.0)
        feed_level(cap, 8.0, -40.0)  # opens, then the 4 s cap continues it
        feed_level(cap, 16.0, -60.0)  # closes the continuation; stream time passes 20 s
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "open" in text and "close" in text and "cap" in text and "tick" in text
    assert "8992" in text and "state=" in text and "floor=" in text and "median=" in text
