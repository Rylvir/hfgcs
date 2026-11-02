"""decoding.worker (the hfgcs-decode service): capture discovery, idempotency, the voice gate,
the per-clip wall-time cap, and load-decodes into a real review DB. The ASR and the gate are
fakes (tests/fake_decode_impl.py), so no model or ffmpeg is needed."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import sys
import time
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from decoding import voice_gate
from decoding import worker as w

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
T0 = datetime(2026, 10, 26, 22, 0, 0, tzinfo=timezone.utc)
FAKE_ASR = "fake_decode_impl:transcribe_eam"
FAKE_GATE = "fake_decode_impl:gate_by_name"


def z(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def write_wav(path, seconds=3.0, rate=8000, seed=0):
    rng = np.random.default_rng(seed)
    pcm = (rng.standard_normal(int(seconds * rate)) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())


def add_capture(app, name, minutes, seconds=3.0, freq=8992.0):
    """Write recordings/<name>.wav and append its schema-2 catalog line (paths as on hilltop)."""
    rec = app / "recordings"
    rec.mkdir(exist_ok=True)
    f = rec / f"{name}.wav"
    write_wav(f, seconds, seed=abs(minutes))
    start = T0 + timedelta(minutes=minutes)
    entry = {"schema": 2, "start_utc": z(start + timedelta(seconds=1)), "file_start_utc": z(start),
             "end_utc": z(start + timedelta(seconds=seconds)), "frequency_khz": freq, "mode": "USB",
             "receiver_id": "owrx-hilltop", "file": f"recordings/{name}.wav",
             "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "audio_rate": 8000,
             "duration_sec": seconds, "preroll_sec": 1.0, "snr_db_peak": 15.0, "snr_db_mean": 8.0,
             "noise_floor_db": -60.0, "truncated": False, "notes": "USB voice candidate - review for EAM"}
    with open(app / "hfgcs_catalog.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


@pytest.fixture(scope="module")
def dbtool():
    spec = importlib.util.spec_from_file_location("hfgcs_db_worker_tests", ROOT / "scripts" / "hfgcs_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def app(tmp_path, monkeypatch, dbtool):
    a = tmp_path / "hfgcs-app"
    a.mkdir()
    db = a / "data" / "hfgcs.db"
    db.parent.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        assert dbtool.main(["--db", str(db), "init"]) == 0
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("FAKE_DECODE_LOG", str(tmp_path / "fake.log"))
    monkeypatch.setenv("FAKE_DECODE_PIDFILE", str(tmp_path / "slow.pid"))
    return a


def cfg_for(app, **kw):
    args = dict(catalog=app / "hfgcs_catalog.jsonl", db=app / "data" / "hfgcs.db", state_dir=app / "state",
                transcribe_impl=FAKE_ASR, gate_impl=FAKE_GATE, poll_s=0.1)
    args.update(kw)
    return w.WorkerConfig(**args)


def fake_log(app):
    p = app.parent / "fake.log"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def messages(app):
    con = sqlite3.connect(app / "data" / "hfgcs.db")
    try:
        return con.execute("SELECT m.msg_type, c.audio_path FROM messages m "
                           "JOIN captures c ON c.id = m.capture_id ORDER BY m.id").fetchall()
    finally:
        con.close()


# -- discovery ---------------------------------------------------------------------------------
def test_read_catalog_oldest_first_and_tolerant(app):
    add_capture(app, "late", 30)
    add_capture(app, "early", 5)
    with open(app / "hfgcs_catalog.jsonl", "a") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"schema": 2, "note": "no file"}) + "\n")
        fh.write('{"half-written": ')                       # recorder mid-append: ignored silently
    caps = w.read_catalog(app / "hfgcs_catalog.jsonl")
    assert [c.name for c in caps] == ["early.wav", "late.wav"]
    assert caps[0].audio == (app / "recordings" / "early.wav").resolve()
    assert caps[0].duration_s == 3.0 and caps[0].key == caps[0].entry["sha256"]


def test_watch_takes_only_new_captures_backfill_takes_all(app):
    add_capture(app, "old", 0)
    add_capture(app, "new", 60)
    cfg = cfg_for(app)
    st = w.State(cfg.state_file)
    st.set_watch_from(z(T0 + timedelta(minutes=30)))
    caps = w.read_catalog(cfg.catalog)
    assert [c.name for c, _ in w.pending(caps, st, cfg, backfill=False)] == ["new.wav"]
    assert [c.name for c, _ in w.pending(caps, st, cfg, backfill=True)] == ["old.wav", "new.wav"]


def test_timeout_for_clamps():
    cfg = w.WorkerConfig(timeout_factor=10, timeout_min_s=120, timeout_max_s=1800)
    assert cfg.timeout_for(3.0) == 120
    assert cfg.timeout_for(60.0) == 600
    assert cfg.timeout_for(900.0) == 1800
    assert cfg.timeout_for(None) == 1800
    assert w.WorkerConfig().timeout_for(900.0) == 7200   # 10x, at most 2 h (3 passes x 2x x 900 s)
    assert w.WorkerConfig().timeout_for(186.0) == 1860


# -- end to end: gate -> decode -> load-decodes, idempotent ----------------------------------------
def test_backfill_decodes_loads_and_is_idempotent(app):
    add_capture(app, "HFGCS_8992kHz_a", 0)
    add_capture(app, "HFGCS_8992kHz_silent", 1)
    add_capture(app, "HFGCS_4724kHz_b", 2, freq=4724.0)
    worker = w.Worker(cfg_for(app))
    assert worker.backfill() == 0
    st = w.State(worker.cfg.state_file).load()
    by_file = {Path(r["file"]).name: r for r in st.clips.values()}
    assert by_file["HFGCS_8992kHz_a.wav"]["status"] == "decoded" and by_file["HFGCS_8992kHz_a.wav"]["loaded"]
    assert by_file["HFGCS_8992kHz_silent.wav"]["status"] == "no_voice"       # skipped, not an error
    assert not (app / "recordings" / "HFGCS_8992kHz_silent.wav.decode.json").exists()
    side = app / "recordings" / "HFGCS_8992kHz_a.wav.decode.json"
    doc = json.loads(side.read_text())
    assert doc["schema_version"] and doc["broadcasts"][0]["type"] == "eam"
    assert len((app / "decodes.jsonl").read_text().splitlines()) == 2
    msgs = messages(app)
    assert sorted(p for _, p in msgs) == ["recordings/HFGCS_4724kHz_b.wav", "recordings/HFGCS_8992kHz_a.wav"]
    # captures are ingested too, so the silent one is on the review page (without a decode)
    con = sqlite3.connect(app / "data" / "hfgcs.db")
    assert con.execute("SELECT count(*) FROM captures").fetchone()[0] == 3
    con.close()
    asr_calls = [x for x in fake_log(app) if x["kind"] == "asr"]
    assert len(asr_calls) == 2

    # re-run: nothing decoded, nothing duplicated
    assert w.Worker(cfg_for(app)).backfill() == 0
    assert len([x for x in fake_log(app) if x["kind"] == "asr"]) == 2
    assert messages(app) == msgs
    assert len((app / "decodes.jsonl").read_text().splitlines()) == 2


def test_watch_once_skips_backlog_then_picks_up_new_capture(app, monkeypatch):
    add_capture(app, "backlog", -600)                # ended long before the worker first started
    cfg = cfg_for(app)
    worker = w.Worker(cfg)
    worker.init_watch_from()
    assert worker.run_pass() == {}
    add_capture(app, "fresh", 24 * 60 * 365)       # ends after watch_from
    counts = worker.run_pass()
    assert counts == {"decoded": 1}
    assert [Path(p).name for _, p in messages(app)] == ["fresh.wav"]


def test_human_edits_survive_a_redecode(app):
    add_capture(app, "edited", 0)
    assert w.Worker(cfg_for(app)).backfill() == 0
    con = sqlite3.connect(app / "data" / "hfgcs.db")
    con.execute("UPDATE messages SET message_text = 'HUMANFIX', corrected = 1")
    con.commit()
    con.close()
    # a new decoder version makes the clip due again; load-decodes must keep the human copy
    cfg = cfg_for(app, second_pass=False)
    assert w.next_action(next(iter(w.State(cfg.state_file).load().clips.values())), cfg,
                         cfg.decoder_version(), cfg.gate_version()) == "decode"
    assert w.Worker(cfg).backfill() == 0
    con = sqlite3.connect(app / "data" / "hfgcs.db")
    assert con.execute("SELECT message_text FROM messages").fetchall() == [("HUMANFIX",)]
    con.close()


# -- failures ---------------------------------------------------------------------------------------
def test_timeout_kills_the_child_and_is_retried_up_to_max_attempts(app):
    add_capture(app, "slow", 0)
    cfg = cfg_for(app, timeout_factor=0.1, timeout_min_s=2.0, timeout_max_s=2.0, max_attempts=2)
    worker = w.Worker(cfg)
    t = time.monotonic()
    assert worker.run_pass(backfill=True) == {"timeout": 1}
    assert time.monotonic() - t < 20
    pid = int((app.parent / "slow.pid").read_text())
    with pytest.raises(ProcessLookupError):        # the child (and its group) is gone
        os.kill(pid, 0)
    rec = next(iter(w.State(cfg.state_file).load().clips.values()))
    assert rec["status"] == "timeout" and rec["attempts"] == 1 and "2 s cap" in rec["error"]
    assert worker.run_pass(backfill=True) == {"timeout": 1}
    assert worker.run_pass(backfill=True) == {}    # given up after 2 attempts
    assert messages(app) == []
    cfg2 = cfg_for(app, timeout_min_s=2.0, timeout_max_s=2.0, retry_failed=True)
    assert w.next_action(rec | {"attempts": 2}, cfg2, cfg2.decoder_version(), cfg2.gate_version()) == "decode"


def test_failed_decode_is_recorded_and_others_continue(app):
    add_capture(app, "boom", 0)
    add_capture(app, "fine", 1)
    worker = w.Worker(cfg_for(app))
    assert worker.run_pass(backfill=True) == {"failed": 1, "decoded": 1}
    recs = {Path(r["file"]).name: r for r in w.State(worker.cfg.state_file).load().clips.values()}
    assert "fake ASR exploded" in recs["boom.wav"]["error"]
    assert recs["fine.wav"]["loaded"]


def test_load_is_called_with_the_sidecar_and_a_failed_load_is_retried_without_redecoding(app):
    add_capture(app, "clip", 0)
    calls = []
    results = iter([(False, "database is locked"), (True, "loaded decodes: inserted 1")])

    def loader(cfg, sidecar):
        calls.append(Path(sidecar).name)
        return next(results)

    worker = w.Worker(cfg_for(app), loader=loader, ingester=lambda cfg: (True, "ok"))
    assert worker.run_pass(backfill=True) == {"decoded_not_loaded": 1}
    assert worker.run_pass(backfill=True) == {"loaded": 1}
    assert calls == ["clip.wav.decode.json", "clip.wav.decode.json"]
    assert len([x for x in fake_log(app) if x["kind"] == "asr"]) == 1


def test_missing_dependency_is_a_config_error_not_a_crash_loop(app, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    cfg = cfg_for(app, transcribe_impl=None)
    problems = w.check_environment(cfg)
    assert any("faster-whisper is not installed" in p for p in problems)
    rc = w.main(["--check", "--catalog", str(cfg.catalog), "--state-dir", str(cfg.state_dir)])
    out = capsys.readouterr().out
    assert rc == w.EXIT_CONFIG == 78
    assert "pip install -e '.[openwebrx,decode]'" in out and "--prefetch" in out

    class Runner:
        def interrupt(self):
            pass

        def run(self, c, timeout, jsonl):
            return w.ChildResult("config_error", 0.1, {"error": "model 'small.en' is not in the cache"})

    add_capture(app, "clip", 0)
    with pytest.raises(w.ConfigError):
        w.Worker(cfg_for(app), runner=Runner()).run_pass(backfill=True)


def test_interrupted_child_is_not_counted_as_an_attempt(app):
    add_capture(app, "clip", 0)

    class Runner:
        def interrupt(self):
            pass

        def run(self, c, timeout, jsonl):
            return w.ChildResult("interrupted", 1.0)

    worker = w.Worker(cfg_for(app), runner=Runner())
    assert worker.run_pass(backfill=True) == {"interrupted": 1}
    assert w.State(worker.cfg.state_file).load().clips == {}


def test_status_lists_every_capture(app, capsys):
    add_capture(app, "one", 0)
    w.Worker(cfg_for(app)).backfill()
    add_capture(app, "two", 1)
    assert w.main(["--status", "--catalog", str(app / "hfgcs_catalog.jsonl"),
                   "--state-dir", str(app / "state")]) == 0
    out = capsys.readouterr().out
    assert "one.wav: decoded" in out and "two.wav: new (next: decode)" in out


# -- voice gate -------------------------------------------------------------------------------------
def test_voice_gate_decisions():
    cfg = voice_gate.GateConfig(min_speech_s=1.0)
    noise = np.full(16000 * 5, 0.05, dtype=np.float32)
    silent = np.zeros(16000 * 5, dtype=np.float32)
    load = {"noise": noise, "silent": silent}
    g = voice_gate.check_voice(Path("noise"), cfg, loader=lambda a, c: load[a.name],
                               detector=lambda pcm, c: [(0.5, 1.0)])
    assert g["voice"] is False and g["speech_s"] == 0.5 and "only 0.5 s" in g["reason"]
    g = voice_gate.check_voice(Path("noise"), cfg, loader=lambda a, c: load[a.name],
                               detector=lambda pcm, c: [(0.0, 1.0), (2.0, 3.5)])
    assert g["voice"] is True and g["speech_s"] == 2.5 and g["regions"] == 2

    def never(pcm, c):
        raise AssertionError("silero must not run on digital silence")

    g = voice_gate.check_voice(Path("silent"), cfg, loader=lambda a, c: load[a.name], detector=never)
    assert g["voice"] is False and g["reason"] == "digital silence"
    assert voice_gate.check_voice(Path("x"), voice_gate.GateConfig(enabled=False))["voice"] is True
    assert cfg.version() != voice_gate.GateConfig(threshold=0.5).version()


def test_unit_files_are_low_priority_and_use_the_decode_env():
    unit = (ROOT / "deploy" / "systemd" / "hfgcs-decode.service").read_text()
    for line in ("Nice=19", "CPUWeight=10", "IOSchedulingClass=idle", "Restart=on-failure",
                 "RestartPreventExitStatus=78", "EnvironmentFile=%h/.config/hfgcs/decode.env",
                 "ExecStart=%h/hfgcs-app/deploy/bin/hfgcs-python -m decoding.worker", "MemoryMax=",
                 "KillMode=mixed"):
        assert line in unit
    oneshot = (ROOT / "deploy" / "systemd" / "hfgcs-decode-backfill.service").read_text()
    assert "Type=oneshot" in oneshot and "--backfill" in oneshot and "\n[Install]" not in oneshot
    assert "Nice=19" in oneshot and "CPUWeight=10" in oneshot
    env = (ROOT / "deploy" / "env" / "decode.env.example").read_text()
    assert "HFGCS_DB=" in env and "HFGCS_DECODE_MODEL=small.en" in env


def test_decoder_version_bump_makes_clips_due_again(app, monkeypatch):
    """A decoding release (``decoding.__version__``) re-decodes and re-loads every decoded clip once;
    no_voice verdicts stay (the gate did not change); the watch loop only takes captures after
    watch_from, so the backlog needs one --backfill (hfgcs-decode-backfill.service)."""
    import decoding

    add_capture(app, "HFGCS_8992kHz_old", 0)
    add_capture(app, "HFGCS_8992kHz_silent", 1)
    add_capture(app, "HFGCS_4724kHz_new", 60, freq=4724.0)
    cfg = cfg_for(app)
    assert w.Worker(cfg).backfill() == 0
    old_version = cfg.decoder_version()
    st = w.State(cfg.state_file).load()
    st.set_watch_from(z(T0 + timedelta(minutes=30)))
    st.save()
    caps = w.read_catalog(cfg.catalog)
    assert w.pending(caps, w.State(cfg.state_file).load(), cfg, backfill=True) == []
    asr_before = len([x for x in fake_log(app) if x["kind"] == "asr"])

    bumped = decoding.__version__ + "+bump"
    monkeypatch.setattr(decoding, "__version__", bumped)
    monkeypatch.setenv("FAKE_DECODING_VERSION", bumped)
    assert cfg.decoder_version() != old_version and f"decoding/{bumped};" in cfg.decoder_version()
    st = w.State(cfg.state_file).load()
    due = {c.name: a for c, a in w.pending(caps, st, cfg, backfill=True)}
    assert due == {"HFGCS_8992kHz_old.wav": "decode", "HFGCS_4724kHz_new.wav": "decode"}
    assert [c.name for c, _ in w.pending(caps, st, cfg, backfill=False)] == ["HFGCS_4724kHz_new.wav"]

    assert w.Worker(cfg).backfill() == 0
    assert len([x for x in fake_log(app) if x["kind"] == "asr"]) == asr_before + 2
    st = w.State(cfg.state_file).load()
    by_file = {Path(r["file"]).name: r for r in st.clips.values()}
    for name in ("HFGCS_8992kHz_old.wav", "HFGCS_4724kHz_new.wav"):
        assert by_file[name]["decoder_version"] == cfg.decoder_version() and by_file[name]["loaded"]
        doc = json.loads((app / "recordings" / (name + ".decode.json")).read_text())
        assert doc["decoder_version"] == cfg.decoder_version()
    assert by_file["HFGCS_8992kHz_silent.wav"]["status"] == "no_voice"
    assert w.pending(caps, st, cfg, backfill=True) == []


def test_worker_uses_the_decoding_default_compute_type():
    """The worker sets only model and threads; compute type (float32 since 0.3) comes from AsrConfig."""
    from decoding.asr import AsrConfig

    asr = w.WorkerConfig(model="small.en", cpu_threads=3).decode_config().asr
    assert asr.compute_type == AsrConfig().compute_type == "float32"
    assert (asr.model, asr.cpu_threads) == ("small.en", 3)
