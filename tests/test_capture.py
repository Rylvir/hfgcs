import json
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
    assert (tmp_path / "rec").glob("*.wav")
    import soundfile as sf

    audio, rate = sf.read(line["file"])
    assert rate == 8000 and abs(len(audio) / rate - line["duration_sec"]) < 0.01


# -- v2 squelch and shadow mode (docs/squelch.md) -------------------------------------------
def keyed(seconds, word=0.5, gap=0.4, amp=SIG_AMP, rate=192_000):
    """A tone keyed word by word: no 'phrase' longer than ``word`` seconds."""
    t = np.arange(int(round(seconds * rate))) / rate
    return tone(1000, seconds, amp) * ((t % (word + gap)) < word)


def with_crashes(x, every_s=0.5, crash_s=0.005, db=25.0, seed=3, rate=192_000):
    """Static crashes: short wideband bursts every_s apart, db above the noise."""
    rng = np.random.default_rng(seed)
    x = x.copy()
    n = int(crash_s * rate)
    for i in range(int(0.2 * rate), len(x) - n, int(every_s * rate)):
        burst = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x[i : i + n] += burst * NOISE_AMP * 10 ** (db / 20) / np.sqrt(2)
    return x


def feed_all(cap, x):
    recs = []
    for c in chunks(x):
        recs.extend(cap.feed(c))
    return recs


def test_static_crashes_hold_v1_open_but_not_v2():
    """After a transmission, crashes every 0.5 s keep resetting v1's hang; v2's median level ignores them."""
    x = np.r_[noise(4, NOISE_AMP, seed=1), noise(5, NOISE_AMP, seed=2) + tone(1000, 5, SIG_AMP),
              with_crashes(noise(40, NOISE_AMP, seed=4))]
    for name, stays_open in (("v1", True), ("v2", False)):
        cap = ChannelCapture(cfg(squelch=name, squelch_shadow=""), 4724.0, start_utc=T0)
        recs = feed_all(cap, x)
        if stays_open:
            assert recs == [] and cap.active
            recs = cap.flush()
            assert recs[0].truncated and recs[0].duration_sec > 40
        else:
            assert len(recs) == 1 and not recs[0].truncated and recs[0].duration_sec < 12
            assert recs[0].start_utc - T0 < timedelta(seconds=4.3)


def test_v2_live_records_a_word_by_word_message_with_its_preroll():
    c = cfg(squelch="v2", squelch_shadow="")
    cap = ChannelCapture(c, 11175.0, start_utc=T0)
    recs = feed_all(cap, np.r_[noise(5, NOISE_AMP, seed=1), noise(20, NOISE_AMP, seed=2) + keyed(20),
                               noise(12, NOISE_AMP, seed=3)])
    assert len(recs) == 1
    r = recs[0]
    # activity starts at the first word (5 s); the file starts pre-roll before it
    assert abs((r.start_utc - T0).total_seconds() - 5.0) < 0.3
    assert abs(r.preroll_seconds - c.preroll_seconds) < 0.1
    # 20 s of words, then the long hang (the capture ran past long_hang_after_seconds)
    assert 20 + c.long_hang_seconds - 1 < r.duration_sec < 20 + c.preroll_seconds + c.long_hang_seconds + 1


def test_shadow_logs_what_v2_would_record_while_v1_records():
    c = cfg(squelch="v1", squelch_shadow="v2")
    cap = ChannelCapture(c, 11175.0, start_utc=T0, receiver_id="rx-test")
    x = np.r_[noise(5, NOISE_AMP, seed=1), noise(20, NOISE_AMP, seed=2) + keyed(20),
              noise(12, NOISE_AMP, seed=3)]
    assert feed_all(cap, x) == []           # v1 never opens on words with pauses
    assert len(cap.shadow_events) == 1
    e = cap.shadow_events[0]
    assert e["algorithm"] == "v2" and e["live_algorithm"] == "v1" and e["level"] == "slot_median"
    assert e["close_reason"] == "hang" and e["receiver_id"] == "rx-test" and e["frequency_khz"] == 11175.0
    assert e["start_utc"].startswith("2026-10-26T12:00:0") and 27 < e["duration_sec"] < 29.5
    assert e["snr_db_peak"] > 10 and e["noise_floor_db"] is not None
    # an open shadow capture is closed (and logged) by a flush, e.g. a stream gap
    feed_all(cap, noise(5, NOISE_AMP, seed=5) + tone(1000, 5, SIG_AMP))
    cap.flush()
    assert cap.shadow_events[-1]["close_reason"] == "flush"


def test_reported_level_drives_v2_when_the_source_sends_one():
    c = cfg(squelch="v1", squelch_shadow="v2")
    cap = ChannelCapture(c, 11175.0, start_utc=T0, input_kind="audio", input_rate=8000, level_reported=True)
    assert cap.level_mode == "slot" and cap.shadow_level_mode == "reported"
    rng = np.random.default_rng(0)
    block = 0.085
    for i in range(int(40 / block)):
        level = -100.0 if 8 <= i * block < 20 else -115.0     # RSSI only; the audio stays quiet
        cap.feed(0.001 * rng.standard_normal(int(block * 8000)).astype(np.float32), level_db=level)
    assert len(cap.shadow_events) == 1 and cap.shadow_events[0]["level"] == "reported"
    assert 19 < cap.shadow_events[0]["duration_sec"] < 21      # 12 s of signal + the 8 s long hang


def test_a_fault_in_the_shadow_squelch_never_stops_the_recording():
    c = cfg(squelch="v1", squelch_shadow="v2")
    cap = ChannelCapture(c, 11175.0, start_utc=T0)

    def broken(level, block_s):
        raise ZeroDivisionError("bug in the shadow")

    cap.shadow.update = broken
    signal = noise(5, NOISE_AMP, seed=2) + tone(1000, 5, SIG_AMP)
    recs = feed_all(cap, np.r_[noise(3, NOISE_AMP, seed=1), signal, noise(4, NOISE_AMP, seed=3)])
    assert len(recs) == 1 and cap.shadow is None          # recorded; the shadow turned itself off
    assert cap.flush() == [] and cap.shadow_events == []
