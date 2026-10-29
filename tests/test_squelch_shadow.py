"""Shadow squelch end to end: the scanner writes it, hfgcs.shadow compares it (docs/squelch.md)."""

import contextlib
import io
import json
import types

import numpy as np
import pytest
import soundfile as sf

import hfgcs_scanner
from hfgcs import shadow
from hfgcs.config import Config, FileSettings, load_config
from hfgcs.sources.file import FileSource

RATE = 12_000


def keyed_wav(path, seconds_before=5.0, words_s=20.0, after_s=12.0):
    """USB audio: noise, then a 1 kHz tone keyed word by word (0.5 s on / 0.4 s off), then noise."""
    rng = np.random.default_rng(7)
    n = int((seconds_before + words_s + after_s) * RATE)
    x = 0.01 * rng.standard_normal(n)
    t = np.arange(int(words_s * RATE)) / RATE
    i = int(seconds_before * RATE)
    x[i : i + len(t)] += 0.1 * np.sin(2 * np.pi * 1000 * t) * ((t % 0.9) < 0.5)
    sf.write(path, x.astype(np.float32), RATE)


def run_scanner(tmp_path, wav, **cfg_kw):
    c = Config(hang_seconds=2.0, preroll_seconds=1.5, record_dir=tmp_path / "rec",
               catalog_file=tmp_path / "catalog.jsonl", channels=(), **cfg_kw)
    c = hfgcs_scanner.apply_cli(c, types.SimpleNamespace(
        source="file", channel=[11175.0], receiver_id=None, url=None, allow_profile_switch=False,
        kiwi_host=None, kiwi_ident=None, file=None, file_format=None, sample_rate=None,
        center_khz=None, realtime=False,
    ))
    sc = hfgcs_scanner.HFGCSScanner(c, FileSource(FileSettings(path=str(wav), dial_khz=11175)))
    assert sc.start() == [11175.0]
    sc.run()
    sc.close()
    return sc


def test_scanner_writes_shadow_decisions_beside_the_catalog(tmp_path):
    wav = tmp_path / "words.wav"
    keyed_wav(wav)
    sc = run_scanner(tmp_path, wav)                     # defaults: v1 records, v2 in shadow
    assert sc.saved == []                               # v1 never opens on words with pauses
    path = tmp_path / "catalog.squelch-shadow.jsonl"
    assert sc.cfg.shadow_path == path
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 1
    e = events[0]
    assert (e["algorithm"], e["live_algorithm"], e["close_reason"]) == ("v2", "v1", "hang")
    assert e["receiver_id"] == "file-words" and e["frequency_khz"] == 11175.0


def test_v2_live_records_what_the_shadow_saw(tmp_path):
    wav = tmp_path / "words.wav"
    keyed_wav(wav)
    sc = run_scanner(tmp_path, wav, squelch="v2", squelch_shadow="")
    assert len(sc.saved) == 1 and sc.saved[0]["duration_sec"] > 20
    assert not (tmp_path / "catalog.squelch-shadow.jsonl").exists()


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_shadow_report_counts_captures_only_one_squelch_made(tmp_path):
    catalog, shadow_file = tmp_path / "cat.jsonl", tmp_path / "cat.squelch-shadow.jsonl"
    rx = dict(receiver_id="kiwi-a", frequency_khz=4724.0)
    _write(catalog, [
        {**rx, "file_start_utc": "2026-11-01T10:00:00.000Z", "end_utc": "2026-11-01T10:15:00.000Z",
         "truncated": True, "snr_db_peak": 9.0},                                    # stuck noise
        {**rx, "file_start_utc": "2026-11-01T11:00:00.000Z", "end_utc": "2026-11-01T11:03:00.000Z",
         "truncated": False, "snr_db_peak": 20.0},                                  # a message both saw
    ])
    _write(shadow_file, [
        {**rx, "algorithm": "v2", "live_algorithm": "v1", "start_utc": "2026-11-01T11:00:01.000Z",
         "end_utc": "2026-11-01T11:03:05.000Z", "close_reason": "hang", "snr_db_peak": 20.0},
        {**rx, "algorithm": "v2", "live_algorithm": "v1", "start_utc": "2026-11-01T12:00:00.000Z",
         "end_utc": "2026-11-01T12:02:00.000Z", "close_reason": "hang", "snr_db_peak": 14.0},  # v1 missed
    ])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert shadow.main(["--catalog", str(catalog), "--list"]) == 0
    text = out.getvalue()
    assert "v2 shadow vs v1 live" in text
    row = next(line for line in text.splitlines() if line.startswith("kiwi-a"))
    # the 10:00 live capture ends before the shadow file starts, so only 11:00 is compared
    assert row.split()[2:8] == ["1", "3.0", "2", "5.1", "0", "1"]
    assert "shadow only: 2026-11-01T12:00:00.000Z" in text
    with contextlib.redirect_stdout(io.StringIO()):
        assert shadow.main(["--catalog", str(catalog), "--since", "2026-11-01"]) == 0
        assert shadow.main(["--catalog", str(tmp_path / "missing.jsonl")]) == 2


def test_squelch_config_defaults_and_checks(tmp_path):
    c = Config(catalog_file=tmp_path / "data" / "hfgcs_catalog.jsonl")
    assert (c.squelch, c.squelch_shadow) == ("v1", "v2")
    assert c.shadow_path == tmp_path / "data" / "hfgcs_catalog.squelch-shadow.jsonl"
    assert Config(shadow_file=str(tmp_path / "s.jsonl")).shadow_path == tmp_path / "s.jsonl"
    assert Config(shadow_file="").shadow_file is None
    for bad in (dict(squelch="v3"), dict(squelch_shadow="v9"), dict(open_window_seconds=0.5),
                dict(stuck_seconds=1.0), dict(long_hang_seconds=1.0), dict(noise_floor_tau_seconds=0),
                dict(noise_floor_alpha=2.0), dict(warmup_seconds=-1)):
        with pytest.raises(ValueError):
            Config(**bad)
    toml = tmp_path / "c.toml"
    toml.write_text('squelch = "v2"\nsquelch_shadow = "v1"\nopen_window_seconds = 2.5\n'
                    'shadow_file = "x.jsonl"\n\n[kiwisdr]\nrssi = false\n')
    c = load_config(toml)
    assert (c.squelch, c.squelch_shadow, c.open_window_seconds) == ("v2", "v1", 2.5)
    assert str(c.shadow_path) == "x.jsonl" and c.kiwisdr.rssi is False


def test_an_unwritable_shadow_file_is_logged_not_fatal(tmp_path, caplog):
    sc = hfgcs_scanner.HFGCSScanner(Config(record_dir=tmp_path, catalog_file=tmp_path / "c.jsonl",
                                           shadow_file=tmp_path), source=types.SimpleNamespace())
    cap = types.SimpleNamespace(shadow_events=[{"algorithm": "v2"}])
    sc._save_shadow(cap)                                   # tmp_path is a directory: open() fails
    assert cap.shadow_events == [] and "could not write 1 shadow squelch event" in caplog.text
