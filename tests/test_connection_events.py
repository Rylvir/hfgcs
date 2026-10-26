"""Connection event logging (OpenWebRX, KiwiSDR) and the recorder state file."""

import json
import logging
import re

import numpy as np
import pytest

from hfgcs.runstate import (
    ConnectionEvents,
    RunState,
    default_name,
    describe_close,
    read_state,
    systemd_unit_name,
)

UTC_RE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z"


def events(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


# -- state file ------------------------------------------------------------------------------------
def test_run_state_transitions_are_written_atomically(tmp_path):
    path = tmp_path / "rt" / "hfgcs-8992.state"
    st = RunState("hfgcs-8992.service", path, pid=4321)
    data = read_state(path)
    assert data["state"] == "starting" and data["pid"] == 4321 and data["name"] == "hfgcs-8992"
    st.set("idle", receiver="owrx-hilltop", channel_khz=8992)
    st.set("capturing")
    data = read_state(path)
    assert data["state"] == "capturing" and data["receiver"] == "owrx-hilltop"
    assert data["channel_khz"] == 8992.0
    assert re.fullmatch(UTC_RE, data["since_utc"])
    st.audio()
    assert read_state(path)["last_audio_utc"] is not None
    assert [p.name for p in path.parent.iterdir()] == ["hfgcs-8992.state"]  # no temp files left behind
    with pytest.raises(ValueError):
        st.set("busy")


def test_state_name_from_systemd_cgroup(tmp_path, monkeypatch):
    cg = tmp_path / "cgroup"
    cg.write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/hfgcs-kiwi-4724.service\n")
    assert systemd_unit_name(str(cg)) == "hfgcs-kiwi-4724"
    cg.write_text("0::/user.slice/user-1000.slice/session-3.scope\n")
    assert systemd_unit_name(str(cg)) is None
    monkeypatch.setenv("HFGCS_STATE_NAME", "hfgcs-8992")
    assert default_name("openwebrx", [8992.0]) == "hfgcs-8992"


def test_describe_close_codes():
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    from websockets.frames import Close

    info = describe_close(ConnectionClosedOK(Close(1001, "going away"), None))
    assert info["close_code"] == 1001 and info["close_reason"] == "going away"
    assert info["error"].startswith("ConnectionClosedOK")
    assert describe_close(ConnectionClosedError(None, None))["close_code"] == 1006
    assert describe_close(OSError("reset")) == {"error": "OSError: reset"}


def test_scanner_writes_capturing_then_idle(tmp_path):
    import hfgcs_scanner
    from hfgcs.config import Channel, Config
    from hfgcs.sources.base import BaseSource, Chunk

    rate = 8000
    rng = np.random.default_rng(3)
    t = np.arange(int(3 * rate)) / rate
    audio = np.concatenate([
        rng.standard_normal(2 * rate) * 0.001,
        0.3 * np.sin(2 * np.pi * 1000 * t),
        rng.standard_normal(4 * rate) * 0.001,
    ]).astype(np.float32)

    class Src(BaseSource):
        name, kind = "fake", "audio"

        def __init__(self):
            super().__init__("rx-fake")
            self.pos = 0

        @property
        def sample_rate(self):
            return rate

        def read(self, timeout=None):
            if self.pos >= len(audio):
                return None
            blk = audio[self.pos:self.pos + 800]
            self.pos += 800
            return Chunk(blk, rate)

    seen = []

    class Spy(RunState):
        def set(self, state, **kw):
            super().set(state, **kw)
            if not seen or seen[-1] != self.state:
                seen.append(self.state)

    st = Spy("hfgcs-test", tmp_path / "s.state", pid=1)
    cfg = Config(channels=(Channel(8992.0),), audio_rate=8000, hang_seconds=1.0, preroll_seconds=1.0,
                 min_activity_seconds=0.5, record_dir=tmp_path / "rec", catalog_file=tmp_path / "c.jsonl")
    sc = hfgcs_scanner.HFGCSScanner(cfg, Src(), state=st)
    sc.start()
    sc.run()
    sc.close()
    assert len(sc.saved) == 1
    assert seen == ["idle", "capturing", "idle", "stopped"]
    data = read_state(tmp_path / "s.state")
    assert data["receiver"] == "rx-fake" and data["last_audio_utc"]


# -- OpenWebRX ---------------------------------------------------------------------------------------
def test_openwebrx_logs_disconnect_reconnect_and_gap(tmp_path, caplog):
    pytest.importorskip("websockets")
    from owrx_mock import MockOpenWebRX, tone_pcm
    from test_openwebrx import collect, settings

    from hfgcs.sources.openwebrx import OpenWebRXSource

    log = tmp_path / "logs" / "hfgcs-8992-connection.log"
    st = RunState("hfgcs-8992", tmp_path / "st.state", pid=1)
    audio = lambda c: [(tone_pcm(1000, 1.0), 1e-3)]  # noqa: E731
    with MockOpenWebRX(audio=audio, events={(0, 10): ("close", None)}) as srv:
        src = OpenWebRXSource(settings(srv.url), sleep=lambda s: None)
        src.events = ConnectionEvents(log, name="hfgcs-8992", state=st)
        with caplog.at_level(logging.INFO), src:
            src.tune(11175)
            collect(src, 0.8)
    ev = events(log)
    kinds = [e["event"] for e in ev]
    assert kinds[:5] == ["connect", "disconnect", "reconnect_attempt", "connect", "audio_resumed"]
    assert kinds[-1] == "disconnect" and ev[-1]["reason"] == "client shutdown"
    lost = ev[1]
    assert lost["close_code"] == 1000 and lost["reason"] == "connection closed" and lost["url"].startswith("ws://")
    assert ev[2]["attempt"] == 1 and ev[3]["reconnect"] is True and ev[3]["down_s"] >= 0
    assert ev[4]["gap_s"] >= ev[4]["down_s"] >= 0
    assert all(re.fullmatch(UTC_RE, e["utc"]) and e["name"] == "hfgcs-8992" for e in ev)
    text = caplog.text
    assert re.search(rf"openwebrx DISCONNECT at {UTC_RE}: lost ws://.*\(close code 1000", text)
    assert re.search(rf"openwebrx RECONNECT_ATTEMPT at {UTC_RE}: reconnecting to .* \(attempt 1\)", text)
    assert re.search(rf"openwebrx AUDIO_RESUMED at {UTC_RE}: audio from .* resumed after a [\d.]+ s gap",
                     text)
    assert any(r.levelno == logging.INFO and "CONNECT at" in r.getMessage() for r in caplog.records)
    assert read_state(tmp_path / "st.state")["last_connection_event"]["event"] == "disconnect"


def test_openwebrx_logs_failed_connects():
    from hfgcs.config import OpenWebRXSettings
    from hfgcs.sources import SourceError
    from hfgcs.sources.openwebrx import OpenWebRXSource

    seen = []

    class Sink:
        def emit(self, event, **f):
            seen.append((event, f))
            return {"utc": "2026-10-26T00:00:00.000Z", "event": event, **f}

    def dead(url, s):
        raise OSError("connection refused")

    s = OpenWebRXSettings(url="http://127.0.0.1:9", max_reconnects=2, min_connect_interval_s=0,
                          backoff_initial_s=0.01)
    src = OpenWebRXSource(s, connect=dead, sleep=lambda s: None)
    src.events = Sink()
    with pytest.raises(SourceError):
        src.open()
    with pytest.raises(SourceError, match="giving up"):
        src.read(0.1)
    kinds = [k for k, _ in seen]
    assert kinds == ["connect_failed", "reconnect_attempt", "connect_failed", "reconnect_attempt",
                     "connect_failed"]
    assert "connection refused" in seen[0][1]["error"]


# -- KiwiSDR -----------------------------------------------------------------------------------------
def test_kiwi_logs_disconnect_and_gap(tmp_path, monkeypatch):
    from test_kiwisdr import Clock, make

    from hfgcs.sources.kiwisdr import reset_etiquette_state

    reset_etiquette_state()
    monkeypatch.setenv("FAKE_KIWI_LOG", str(tmp_path / "argv.jsonl"))
    monkeypatch.setenv("FAKE_KIWI_SECONDS", "1.0")
    log = tmp_path / "kiwi.log"
    clock = Clock()
    src = make(["zero.example:8073", "good.example:8073"], clock=clock, stall_timeout_s=20)
    src.events = ConnectionEvents(log, name="hfgcs-kiwi-11175")
    try:
        with src:
            src.tune(11175)
            for _ in range(200):
                c = src.read(0.2)
                if len(c.samples) and np.any(c.samples):
                    break
                clock.t += 5
    finally:
        reset_etiquette_state()
    ev = events(log)
    kinds = [e["event"] for e in ev]
    assert kinds[:4] == ["connect", "disconnect", "connect", "audio_resumed"]
    assert ev[0]["host"] == "zero.example:8073" and "no audio for 20 s" in ev[1]["reason"]
    assert ev[2]["host"] == "good.example:8073" and ev[3]["gap_s"] >= 20
    assert kinds[-1] == "disconnect" and ev[-1]["reason"] == "close"


def test_scanner_cli_state_file_only_for_named_runs(tmp_path, monkeypatch):
    import hfgcs_scanner

    monkeypatch.delenv("HFGCS_STATE_NAME", raising=False)
    monkeypatch.setattr(hfgcs_scanner, "systemd_unit_name", lambda: None)
    args = hfgcs_scanner.build_parser().parse_args([])
    assert hfgcs_scanner.make_run_state(args) == (None, None)  # manual runs write nothing
    monkeypatch.setattr(hfgcs_scanner, "systemd_unit_name", lambda: "actions.runner.x")
    assert hfgcs_scanner.make_run_state(args) == (None, None)  # only hfgcs-* units
    monkeypatch.setattr(hfgcs_scanner, "systemd_unit_name", lambda: "hfgcs-8992")
    monkeypatch.setenv("HFGCS_RUNTIME_DIR", str(tmp_path / "rt"))
    monkeypatch.chdir(tmp_path)
    state, ev = hfgcs_scanner.make_run_state(args)
    assert state.path == tmp_path / "rt" / "hfgcs-8992.state"
    assert read_state(state.path)["state"] == "starting"
    assert str(ev.path) == "logs/hfgcs-8992-connection.log" and ev.state is state
    args = hfgcs_scanner.build_parser().parse_args(["--no-state-file"])
    assert hfgcs_scanner.make_run_state(args) == (None, None)
