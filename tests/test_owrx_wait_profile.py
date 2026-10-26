"""Wait-idle-for-profile and the one-shot --select-profile (local mock server only)."""

import logging
import threading
import types

import numpy as np
import pytest
from owrx_mock import MockOpenWebRX, tone_pcm

from hfgcs.config import OpenWebRXSettings
from hfgcs.sources import SourceError, TuneRefused
from hfgcs.sources.openwebrx import OpenWebRXSource

pytest.importorskip("websockets")

KEY = "test-magic-key-not-a-secret"
P8992, P15016 = "airspyhf|hfgcs-8992", "airspyhf|hfgcs-15016"


def settings(url, **kw):
    base = dict(
        url=url,
        min_tune_interval_s=0.0,
        min_profile_interval_s=0.0,
        min_connect_interval_s=0.0,
        backoff_initial_s=0.01,
        backoff_max_s=0.02,
        handshake_timeout_s=5.0,
    )
    base.update(kw)
    return OpenWebRXSettings(**base)


def read_audio(src, seconds, max_reads=400):
    got, chunks = 0, []
    for _ in range(max_reads):
        c = src.read(0.1)
        chunks.append(c)
        got += len(c.samples)
        if got >= seconds * src.sample_rate:
            break
    return got, chunks


def types_sent(srv):
    return [m["type"] for m in srv.all_sent()]


def test_waits_idle_then_records_when_owner_selects_covering_profile(caplog):
    caplog.set_level(logging.WARNING)
    with MockOpenWebRX(profile=P15016, audio=lambda c: [(tone_pcm(1000, 2.0), 1e-3)]) as srv:
        with OpenWebRXSource(settings(srv.url, wait_for_profile=True)) as src:
            assert src.can_tune(8992)  # waiting is allowed, nothing is refused
            src.tune(8992)
            assert src.state == "waiting"
            for _ in range(5):
                assert len(src.read(0.05).samples) == 0
            assert "dspcontrol" not in types_sent(srv)  # idle: our demodulator never started
            srv.switch_profile(P8992)  # the owner parks the radio on 8992
            got, chunks = read_audio(src, 0.5)
            assert got >= 0.5 * src.sample_rate
            assert src.state == "recording"
            assert any(c.gap for c in chunks)
        sent = srv.all_sent()
        dsp = [m for m in sent if m["type"] == "dspcontrol" and "params" in m]
        assert dsp and dsp[0]["params"]["offset_freq"] == 0
        assert "selectprofile" not in types_sent(srv) and "setfrequency" not in types_sent(srv)
    assert caplog.text.count("WAITING: 8992 kHz") == 1  # once per state change, not per poll
    assert caplog.text.count("RECORDING: 8992 kHz") == 1


def test_profile_moves_away_mid_run_stops_and_idles_never_retunes(caplog):
    caplog.set_level(logging.WARNING)
    events = {(0, 20): ("config", P15016), (0, 60): ("config", P8992)}
    with MockOpenWebRX(profile=P8992, audio=lambda c: [(tone_pcm(1000, 3.0), 1e-3)], events=events) as srv:
        with OpenWebRXSource(settings(srv.url, wait_for_profile=True)) as src:
            src.tune(8992)
            states, gaps, got = set(), 0, 0
            for _ in range(600):
                c = src.read(0.1)
                states.add(src.state)
                gaps += c.gap
                got += len(c.samples)
                if got >= 60 * 240:
                    break
            assert states == {"recording", "waiting"}
            assert src.state == "recording"
            assert gaps >= 2  # one when it went idle (closes the recording), one on resume
        assert "selectprofile" not in types_sent(srv)
    assert caplog.text.count("WAITING: 8992 kHz") == 1
    assert caplog.text.count("RECORDING: 8992 kHz") == 2


def test_without_wait_the_channel_is_still_refused():
    with MockOpenWebRX(profile=P15016) as srv:
        with OpenWebRXSource(settings(srv.url)) as src:
            assert not src.can_tune(8992)
            with pytest.raises(TuneRefused, match="--wait-for-profile"):
                src.tune(8992)


def test_wait_mode_never_gives_up_reconnecting():
    class Stop(Exception):
        pass

    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 8:
            raise Stop

    def dead(url, s):
        raise OSError("connection refused")

    src = OpenWebRXSource(
        settings("http://127.0.0.1:9", max_reconnects=2, wait_for_profile=True), connect=dead, sleep=sleep
    )
    with pytest.raises(Stop):
        src.read(0.1)
    assert len(sleeps) == 8  # kept going well past max_reconnects (systemd-friendly)


def test_scanner_wait_mode_records_after_profile_appears(tmp_path):
    import hfgcs_scanner
    from hfgcs.config import Channel, Config

    rng = np.random.default_rng(2)

    def quiet(seconds):
        return (rng.standard_normal(int(seconds * 12_000)) * 3000).astype(np.int16)

    def script(conn):
        return [(quiet(2), 1e-9), (tone_pcm(1000, 2.0, amp=0.1), 1e-6), (quiet(3), 1e-9)]

    with MockOpenWebRX(profile=P15016, audio=script) as srv:
        c = Config(
            channels=(Channel(8992.0),),
            hang_seconds=1.0,
            preroll_seconds=1.0,
            min_activity_seconds=0.5,
            record_dir=tmp_path / "rec",
            catalog_file=tmp_path / "cat.jsonl",
            source="openwebrx",
            openwebrx=settings(srv.url, wait_for_profile=True, receiver_id="owrx-hilltop"),
        )
        sc = hfgcs_scanner.HFGCSScanner(c)
        assert sc.start() == [8992.0]
        threading.Timer(0.5, srv.switch_profile, args=(P8992,)).start()
        sc.run(max_seconds=6.5)
        sc.close()
    assert len(sc.saved) == 1
    assert sc.saved[0]["receiver_id"] == "owrx-hilltop"
    assert sc.saved[0]["frequency_khz"] == 8992.0
    assert srv.selects == []


def _args(**kw):
    base = dict(
        source="openwebrx", channel=None, receiver_id=None, url=None, allow_profile_switch=False,
        kiwi_host=None, kiwi_ident=None, file=None, file_format=None, sample_rate=None,
        center_khz=None, realtime=False, duration=None, probe=False, select_profile=None,
        wait_for_profile=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_receiver_id_flag_reaches_the_openwebrx_source():
    import hfgcs_scanner
    from hfgcs.config import Config

    cfg = hfgcs_scanner.apply_cli(Config(), _args(receiver_id="owrx-hilltop"))
    assert cfg.openwebrx.receiver_id == "owrx-hilltop"
    assert OpenWebRXSource(cfg.openwebrx).receiver_id == "owrx-hilltop"


def test_wait_for_profile_defaults():
    import hfgcs_scanner
    from hfgcs.config import Config

    cfg = Config()
    assert hfgcs_scanner.apply_cli(cfg, _args()).openwebrx.wait_for_profile is True  # service mode
    assert hfgcs_scanner.apply_cli(cfg, _args(duration=300)).openwebrx.wait_for_profile is False
    assert hfgcs_scanner.apply_cli(cfg, _args(duration=300, wait_for_profile=True)).openwebrx.wait_for_profile
    assert not hfgcs_scanner.apply_cli(cfg, _args(wait_for_profile=False)).openwebrx.wait_for_profile
    assert not hfgcs_scanner.apply_cli(cfg, _args(probe=True)).openwebrx.wait_for_profile
    assert not hfgcs_scanner.apply_cli(cfg, _args()).openwebrx.allow_profile_switch


def test_one_shot_run_outside_profile_still_exits_2(tmp_path):
    import hfgcs_scanner

    with MockOpenWebRX(profile=P15016) as srv:
        rc = hfgcs_scanner.main(
            ["--source", "openwebrx", "--url", srv.url, "--channel", "8992", "--duration", "5"]
        )
    assert rc == 2


# -- --select-profile -------------------------------------------------------------------------
def test_select_profile_once_with_key(monkeypatch, capsys, caplog):
    import hfgcs_scanner

    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENWEBRX_MAGIC_KEY", KEY)
    with MockOpenWebRX(profile=P15016, magic_key=KEY, locked={P8992, P15016}) as srv:
        rc = hfgcs_scanner.main(
            ["--source", "openwebrx", "--url", srv.url, "--select-profile", "HFGCS 8992 kHz", "-v"]
        )
        assert rc == 0
        assert srv.selects == [P8992]
        sel = [m for m in srv.all_sent() if m["type"] == "selectprofile"]
        assert len(sel) == 1 and sel[0]["params"] == {"profile": P8992, "key": KEY}
        assert "dspcontrol" not in types_sent(srv)
    out = capsys.readouterr()
    assert "switched: yes" in out.out and P8992 in out.out
    assert KEY not in out.out + out.err + caplog.text  # never printed or logged


def test_select_profile_custom_env_name_and_already_active(monkeypatch, capsys):
    import hfgcs_scanner

    monkeypatch.setenv("OTHER_KEY_VAR", KEY)
    with MockOpenWebRX(profile=P8992, magic_key=KEY, locked={P8992}) as srv:
        rc = hfgcs_scanner.main([
            "--source", "openwebrx", "--url", srv.url, "--select-profile", "HFGCS 8992 kHz",
            "--unlock-key-env", "OTHER_KEY_VAR",
        ])
        assert rc == 0
        assert "selectprofile" not in types_sent(srv)  # already active: nothing sent
    assert "already active" in capsys.readouterr().out


def test_select_profile_refuses_without_key(monkeypatch, capsys):
    import hfgcs_scanner

    monkeypatch.delenv("OPENWEBRX_MAGIC_KEY", raising=False)
    with MockOpenWebRX(profile=P15016) as srv:
        rc = hfgcs_scanner.main(["--source", "openwebrx", "--url", srv.url, "--select-profile", P8992])
        assert rc == 1
        assert srv.connections == 0
    assert "OPENWEBRX_MAGIC_KEY is not set" in capsys.readouterr().err


def test_select_profile_wrong_key_is_reported_and_not_retried(monkeypatch, capsys):
    import hfgcs_scanner

    monkeypatch.setenv("OPENWEBRX_MAGIC_KEY", "wrong")
    with MockOpenWebRX(profile=P15016, magic_key=KEY, locked={P8992}) as srv:
        rc = hfgcs_scanner.main(["--source", "openwebrx", "--url", srv.url, "--select-profile", P8992])
        assert rc == 3
        assert srv.selects == []
        assert types_sent(srv).count("selectprofile") == 1
    err = capsys.readouterr().err
    assert "locked" in err and "wrong" not in err


def test_select_profile_unknown_or_ambiguous(monkeypatch):
    with MockOpenWebRX(profile=P15016) as srv:
        with OpenWebRXSource(settings(srv.url)) as src:
            with pytest.raises(SourceError, match="no OpenWebRX profile matches"):
                src.select_profile_once("HFGCS 9999 kHz", KEY)
            # exact name wins over a suffix match; ids work too
            assert src.find_profile("HFGCS 15016")["id"] == "airspy|hfgcs-15016"
            assert src.find_profile("HFGCS 15016 kHz")["id"] == P15016
            src.profiles = src.profiles + [{"id": "other|x", "name": "Other HFGCS 15016 kHz"}]
            with pytest.raises(SourceError, match="ambiguous"):
                src.select_profile_once("HFGCS 15016 kHz", KEY)
        assert "selectprofile" not in types_sent(srv)


def test_sent_log_redacts_key():
    with MockOpenWebRX(profile=P15016, magic_key=KEY, locked={P8992}) as srv:
        with OpenWebRXSource(settings(srv.url)) as src:
            assert src.select_profile_once(P8992, KEY) is True
            assert src.window.center_hz == 8_992_000
            assert KEY not in repr(src.sent)
