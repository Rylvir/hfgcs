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
