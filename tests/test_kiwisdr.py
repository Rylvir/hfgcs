"""KiwiSDR source with a fake kiwirecorder: etiquette rules and streaming."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hfgcs.config import KiwiSettings
from hfgcs.sources import SourceError, TuneRefused, kiwisdr
from hfgcs.sources.kiwisdr import KiwiSource, nominal_rate, parse_wav_header, reset_etiquette_state

FAKE = str(Path(__file__).with_name("fake_kiwirecorder.py"))
GOOD_STATUS = {"status": "active", "offline": "no", "users": "1", "users_max": "8"}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    reset_etiquette_state()
    monkeypatch.setenv("FAKE_KIWI_LOG", str(tmp_path / "argv.jsonl"))
    monkeypatch.setenv("FAKE_KIWI_SECONDS", "1.0")
    yield
    reset_etiquette_state()


def calls(tmp_path):
    p = tmp_path / "argv.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def make(hosts, status=None, resolver=None, clock=None, popen=None, **kw):
    base = dict(
        hosts=tuple(hosts),
        channel_khz=11175,
        enabled=True,
        ident="N0CALL-hfgcs-test",
        kiwirecorder=FAKE,
        python=sys.executable,
        backoff_initial_s=30,
        rest_between_sessions_s=60,
    )
    base.update(kw)
    st = status or (lambda host, ua, timeout: dict(GOOD_STATUS))
    clock = clock or Clock()
    return KiwiSource(
        KiwiSettings(**base),
        status=st,
        resolver=resolver or (lambda h: None),
        clock=clock,
        sleep=clock.sleep,
        popen=popen,
    )


def read_audio(src, seconds, reads=400):
    out, chunks = [], []
    for _ in range(reads):
        c = src.read(0.5)
        chunks.append(c)
        if len(c.samples):
            out.append(c.samples)
        if sum(map(len, out)) >= seconds * 12000:
            break
    return (np.concatenate(out) if out else np.zeros(0, np.float32)), chunks


def test_must_be_enabled_with_real_ident():
    with pytest.raises(SourceError, match="enabled = true"):
        make(["a.example:8073"], enabled=False).open()
    with pytest.raises(SourceError, match="ident"):
        make(["a.example:8073"], ident="hfgcs").open()


def test_streams_usb_audio_and_identifies(tmp_path):
    seen_ua = []
    def status(host, ua, timeout):
        seen_ua.append(ua)
        return dict(GOOD_STATUS)

    with make(["kiwi.example:8074"], status=status) as src:
        src.tune(11175)
        audio, chunks = read_audio(src, 0.9)
        assert src.sample_rate == 12000
        assert chunks[0].gap or any(c.gap for c in chunks)
        assert src.receiver_id == "kiwi-kiwi.example-8074"
    spec = np.abs(np.fft.rfft(audio[:9000] * np.hanning(9000)))
    assert abs(np.argmax(spec) * 12000 / 9000 - 1000) < 3
    argv = calls(tmp_path)[0]
    for flag in ("--nc", "--nc-wav"):
        assert flag in argv
    assert argv[argv.index("-u") + 1] == "N0CALL-hfgcs-test"
    assert argv[argv.index("-m") + 1] == "usb"
    assert argv[argv.index("-f") + 1] == "11175"
    assert argv[argv.index("--tlimit") + 1] == "1320"
    assert "--wf" not in argv
    assert seen_ua and "hfgcs" in seen_ua[0]


def test_one_frequency_per_receiver():
    src = make(["kiwi.example:8073"])
    src.open()
    with pytest.raises(TuneRefused, match="one frequency"):
        src.tune(8992)
    src.close()


def test_one_slot_per_operator_across_sources(tmp_path):
    a = make(["kiwi.example:8073"])
    b = make(["kiwi.example:8074"], channel_khz=8992)
    with a:
        a.tune(11175)
        read_audio(a, 0.1)
        with b:
            b.tune(8992)
            c = b.read(0.1)
            assert len(c.samples) == 0
            assert any("already using a slot" in m for m in b.log)
    assert len(calls(tmp_path)) == 1


def test_same_ip_counts_as_same_operator(tmp_path):
    def ip(host):
        return "192.0.2.7"

    a = make(["one.example:8073"], resolver=ip)
    b = make(["two.example:8073"], resolver=ip, channel_khz=8992)
    with a:
        a.tune(11175)
        read_audio(a, 0.1)
        with b:
            b.tune(8992)
            b.read(0.1)
            assert any("same operator" in m for m in b.log)
    assert len(calls(tmp_path)) == 1


def test_too_busy_abandons_without_retry_and_rotates(tmp_path):
    clock = Clock()
    src = make(["busy.example:8073", "good.example:8073"], clock=clock)
    with src:
        src.tune(11175)
        audio, _ = read_audio(src, 0.2)
        assert len(audio)
        assert src.current.host == "good.example"
        assert any("too_busy" in m and "retired 60 min" in m for m in src.log)
        assert src.skip_reason(src.hosts[0]).startswith("retired")
    hosts = [a[a.index("-s") + 1] for a in calls(tmp_path)]
    assert hosts == ["busy.example", "good.example"]  # busy host tried exactly once


def test_status_filters_and_blocklist(tmp_path):
    statuses = {
        "small.example": {**GOOD_STATUS, "users_max": "2"},
        "full.example": {**GOOD_STATUS, "users": "8"},
        "off.example": {**GOOD_STATUS, "offline": "yes"},
    }
    def status(host, ua, timeout):
        return statuses.get(host.host, GOOD_STATUS)

    src = make(
        ["small.example", "full.example", "off.example", "n0x.proxy.kiwisdr.com", "blocked.example:8073"],
        status=status,
        blocklist=("blocked.example",),
    )
    with src:
        src.tune(11175)
        c = src.read(0.1)
        assert len(c.samples) == 0
    text = "\n".join(src.log)
    assert "only 2 slots" in text and "full (8/8)" in text and "offline" in text
    assert "proxied endpoint" in text and "blocklisted" in text
    assert "next attempt in 30 s" in text
    assert calls(tmp_path) == []


def test_session_cap_releases_slot_and_rests(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KIWI_SECONDS", "5")
    clock = Clock()
    src = make(["kiwi.example:8073"], clock=clock, session_max_s=60)
    with src:
        src.tune(11175)
        read_audio(src, 0.1)
        clock.t += 61
        src.read(0.1)
        assert src.session is None
        assert any("session cap" in m for m in src.log)
        assert src.skip_reason(src.hosts[0]) is None  # slot released
        before = clock.t
        c = src.read(10)  # resting: no reconnect yet
        assert len(c.samples) == 0 and clock.t - before == pytest.approx(10)
        assert len(calls(tmp_path)) == 1


def test_connect_failure_backs_off_exponentially(tmp_path):
    clock = Clock()
    src = make(["dead.example:8073"], clock=clock, retire_after_failure_s=0, backoff_initial_s=30)
    with src:
        src.tune(11175)
        waits = []
        for _ in range(3):
            src.read(0.1)  # attempt -> fails -> backoff scheduled
            waits.append(src._not_before - clock.t)
            clock.t = src._not_before
        for w, n in zip(waits, [30, 60, 120], strict=True):
            assert 0.8 * n <= w <= 1.2 * n


def test_parse_wav_header_rejects_garbage():
    import io

    with pytest.raises(SourceError):
        parse_wav_header(io.BytesIO(b"not a wav file at all").read)


def test_scanner_uses_only_one_channel_on_a_kiwi(tmp_path):
    import hfgcs_scanner
    from hfgcs.config import Config

    src = make(["kiwi.example:8073"], channel_khz=0)
    sc = hfgcs_scanner.HFGCSScanner(Config(record_dir=tmp_path), src)
    assert sc.start() == [4724.0]
    assert "one channel at a time" in sc.refused[11175.0]
    sc.close()


@pytest.mark.parametrize("ident", ["HFGCS-Monitor", "hfgcs-monitor", "HFGCS-MONITOR", " HFGCS-Monitor "])
def test_accepts_hfgcs_monitor_ident(ident):
    make(["a.example:8073"], ident=ident).check_authorised()


@pytest.mark.parametrize("ident", ["", "   ", "changeme", "CHANGEME", "hfgcs", "your-callsign"])
def test_rejects_placeholder_idents(ident):
    with pytest.raises(SourceError, match="ident"):
        make(["a.example:8073"], ident=ident).check_authorised()


@pytest.mark.parametrize(
    "measured,nominal",
    [(11999, 12000), (12001, 12000), (12000, 12000), (20249, 20250), (48000, 48000)],
)
def test_nominal_rate_snaps_measured_kiwi_rates(measured, nominal):
    assert nominal_rate(measured) == nominal


def test_measured_rate_in_wav_header_is_usable(tmp_path, monkeypatch):
    # Real kiwirecorder writes the measured rate (e.g. 11999) into the --nc-wav header.
    from hfgcs.dsp import AudioFrontEnd

    monkeypatch.setenv("FAKE_KIWI_RATE", "11999")
    with make(["kiwi.example:8073"]) as src:
        src.tune(11175)
        audio, _ = read_audio(src, 0.5)
        assert len(audio)
        assert src.sample_rate == 12000
        AudioFrontEnd(src.sample_rate, 8000)  # 11999 -> 8000 used to raise ValueError


@pytest.mark.parametrize("bad", ["stall", "zero"])
def test_watchdog_drops_silent_session_and_rotates(tmp_path, bad):
    clock = Clock()
    src = make([f"{bad}.example:8073", "good.example:8073"], clock=clock, stall_timeout_s=20)
    with src:
        src.tune(11175)
        got = []
        for _ in range(200):
            c = src.read(0.2)
            if len(c.samples) and np.any(c.samples):
                got.append(c)
                break
            clock.t += 5
        assert got and src.current.host == "good.example"
        assert any(f"{bad}.example:8073: no audio for 20 s" in m for m in src.log)
        assert src.skip_reason(src.hosts[0]).startswith("retired")
    hosts = [a[a.index("-s") + 1] for a in calls(tmp_path)]
    assert hosts[:2] == [f"{bad}.example", "good.example"]


def test_scanner_capture_receiver_id_follows_kiwi_host(tmp_path):
    import hfgcs_scanner
    from hfgcs.config import Config

    src = make(["a.example:8073", "b.example:8074"])
    sc = hfgcs_scanner.HFGCSScanner(Config(record_dir=tmp_path), src)
    src.current = src.hosts[0]
    assert sc.capture_for(11175.0, 12000).receiver_id == "kiwi-a.example-8073"
    src.current = src.hosts[1]
    assert sc.capture_for(11175.0, 12000).receiver_id == "kiwi-b.example-8074"


# -- one session per kiwirecorder process -------------------------------------------


class RecordingPopen:
    """subprocess.Popen that keeps every process it starts."""

    def __init__(self):
        self.procs = []

    def __call__(self, *a, **kw):
        proc = subprocess.Popen(*a, **kw)
        self.procs.append(proc)
        return proc


def session_hosts(tmp_path):
    return [a[a.index("-s") + 1] for a in calls(tmp_path) if a[0] != "RECONNECT"]


def reconnects(tmp_path):
    return [a for a in calls(tmp_path) if a[0] == "RECONNECT"]


def read_until_host(src, clock, host, reads=200):
    for _ in range(reads):
        c = src.read(0.2)
        if len(c.samples) and src.current is not None and src.current.host == host:
            return c
        clock.t += 1
    raise AssertionError(f"never streamed from {host}: {src.log}")


def test_kiwirecorder_runs_at_a_log_level_that_shows_reconnects(tmp_path):
    with make(["kiwi.example:8073"]) as src:
        src.tune(11175)
        read_audio(src, 0.1)
    argv = calls(tmp_path)[0]
    assert argv[argv.index("--log") + 1] == "info"
    assert argv[argv.index("--connect-retries") + 1] == "1"
    assert argv[argv.index("--busy-retries") + 1] == "1"


@pytest.mark.parametrize(
    "line",
    [
        "kphsdr.com:8073 server closed the connection cleanly. Reconnecting after 5 seconds",
        "kphsdr.com:8073 server closed the connection unexpectedly. Reconnecting after 5 seconds",
        "kphsdr.com:8073 Too busy now. Reconnecting after 15 seconds",
        "kphsdr.com:8073 Too busy now. Redirecting to other.example:8073",
        "2026-10-26 15:38:01,123 pid 4242 Failed to connect, sleeping and reconnecting error='timed out'",
    ],
)
def test_session_end_patterns_match_kiwiclient_v1_11_lines(line):
    assert any(p.search(line) for p in kiwisdr.SESSION_END_PATTERNS)


@pytest.mark.parametrize(
    "line",
    [
        "2026-10-26 15:38:01,123 pid 4242 started netcat recorder 0",
        "2026-10-26 15:38:01,123 pid 4242 Kiwi server version: 1.826",
        "2026-10-26 15:38:01,123 pid 4242 Sample rate drift correction: advertised=12000.00 Hz, "
        "actual=11999.10 Hz, drift=0.010 sec, correction_factor=0.999925",
    ],
)
def test_ordinary_info_lines_do_not_end_the_session(line):
    assert not any(p.search(line) for p in kiwisdr.SESSION_END_PATTERNS)


@pytest.mark.parametrize("bad", ["kick", "stubborn"])
def test_server_ended_session_stops_process_before_it_reconnects(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("FAKE_KIWI_RECONNECT_S", "1.5")
    monkeypatch.setattr(kiwisdr, "STOP_GRACE_S", 0.3)
    clock, popen = Clock(), RecordingPopen()
    src = make([f"{bad}.example:8073", "good.example:8073"], clock=clock, popen=popen)
    with src:
        src.tune(11175)
        read_until_host(src, clock, "good.example")
        first = popen.procs[0]
        assert first.poll() is not None  # the kicked process is gone
        if bad == "stubborn":
            assert first.returncode == -signal.SIGTERM  # ignored SIGINT, terminated after the grace
        text = "\n".join(src.log)
        assert f"{bad}.example:8073: kicked; abandoned, retired 60 min" in text
        assert "server closed the connection cleanly. Reconnecting after 5 seconds" in text
        assert "one session per process" in text
        assert "next attempt in 30 s" in text  # counted toward backoff
        assert src.skip_reason(src.hosts[0]).startswith("retired")  # set aside
    time.sleep(1.8)  # past the fake's reconnect delay
    assert reconnects(tmp_path) == []  # it never opened a second session
    assert session_hosts(tmp_path) == [f"{bad}.example", "good.example"]  # one attempt on the bad host


def test_session_end_line_counts_even_if_process_exits_after_a_long_session(tmp_path):
    # kiwirecorder prints the line and exits by itself after > 60 s of audio: still a failed
    # session (host retired, backoff), not a normal end with only a rest.
    clock, popen = Clock(), RecordingPopen()
    src = make(["kickexit.example:8073", "good.example:8073"], clock=clock, popen=popen)
    with src:
        src.tune(11175)
        read_audio(src, 0.5)
        clock.t += 120
        read_until_host(src, clock, "good.example")
        assert any("kickexit.example:8073: kicked; abandoned, retired 60 min" in m for m in src.log)
        assert src.skip_reason(src.hosts[0]).startswith("retired")
    assert session_hosts(tmp_path) == ["kickexit.example", "good.example"]


def test_redirect_stops_process_instead_of_following_it(tmp_path, monkeypatch):
    monkeypatch.setattr(kiwisdr, "STOP_GRACE_S", 0.3)
    clock, popen = Clock(), RecordingPopen()
    src = make(["redirect.example:8073", "good.example:8073"], clock=clock, popen=popen)
    with src:
        src.tune(11175)
        read_until_host(src, clock, "good.example")
        assert popen.procs[0].poll() is not None
        assert any("redirect.example:8073: too_busy; abandoned, retired 60 min" in m for m in src.log)
    assert session_hosts(tmp_path) == ["redirect.example", "good.example"]


def test_natural_end_after_tlimit_rests_without_retiring(tmp_path):
    # kiwirecorder's own --tlimit end: it exits quietly; the slot is released, we rest >= 60 s.
    clock = Clock()
    src = make(["kiwi.example:8073"], clock=clock)
    with src:
        src.tune(11175)
        read_audio(src, 0.5)
        clock.t += 1300  # just under our own 1320 s cap, which would otherwise fire first
        for _ in range(50):
            src.read(0.2)
            if src.session is None:
                break
        assert src.session is None
        assert any("stream ended after" in m for m in src.log)
        assert src.skip_reason(src.hosts[0]) is None  # not retired
        assert src._not_before - clock.t >= 59
        assert src._failures == 0


def test_stderr_is_drained_while_streaming(tmp_path):
    # A chatty kiwirecorder must not block on a full stderr pipe.
    script = tmp_path / "chatty.py"
    script.write_text(
        "import sys, struct\n"
        "sys.stderr.write('x' * 200 + '\\n' * 1)\n"
        "for _ in range(2000): sys.stderr.write('info line ' + 'y' * 100 + '\\n')\n"
        "sys.stderr.flush()\n"
        "out = sys.stdout.buffer\n"
        "fmt = struct.pack('<HHIIHH', 1, 1, 12000, 24000, 2, 16)\n"
        "out.write(b'RIFF' + struct.pack('<I', 0x7FFFFFF0) + b'WAVE')\n"
        "out.write(b'fmt ' + struct.pack('<I', len(fmt)) + fmt)\n"
        "out.write(b'data' + struct.pack('<I', 0x7FFFFFF0 - 36))\n"
        "out.write(b'\\x10\\x00' * 12000)\n"
        "out.flush()\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    session = kiwisdr.KiwiSession(proc)
    assert session.header_ready.wait(10) and session.rate == 12000
    assert session.ended.wait(10)
    session.wait_stderr(5)
    assert session.stderr_done.is_set() and session.session_end is None
    session.stop()
    assert proc.wait(5) == 0


def _scanner_config(tmp_path, **kiwi):
    lines = [
        'source = "kiwisdr"',
        f'record_dir = "{tmp_path / "rec"}"',
        "[kiwisdr]",
        "enabled = true",
        'ident = "HFGCS-Monitor"',
        "channel_khz = 11175",
        'hosts = ["stall.example:8073"]',
        "check_status = false",
        "stall_timeout_s = 0",
        f'kiwirecorder = "{FAKE}"',
        f'python = "{sys.executable}"',
    ] + [f"{k} = {v}" for k, v in kiwi.items()]
    path = tmp_path / "kiwi.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_scanner_exits_0_on_sigint_even_with_a_dead_log_pipe(tmp_path):
    # systemd KillSignal=SIGINT also stops the unit's tee, so stdout is a broken pipe by the
    # time we print "Stopping...". That used to exit 1 (logged as Failed).
    cfg = _scanner_config(tmp_path)
    scanner = Path(__file__).resolve().parent.parent / "hfgcs_scanner.py"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(scanner), "--config", str(cfg), "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 20
        seen = b""
        while b"connect stall.example" not in seen and time.monotonic() < deadline:
            seen += proc.stdout.readline()
        assert b"connect stall.example" in seen, seen
        time.sleep(0.3)
        proc.stdout.close()  # the log pipe dies first
        proc.send_signal(signal.SIGINT)
        assert proc.wait(20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_scanner_real_errors_stay_nonzero(tmp_path):
    import hfgcs_scanner

    cfg = _scanner_config(tmp_path, enabled="false")
    assert hfgcs_scanner.main(["--config", str(cfg)]) == 1


def test_scanner_sigint_outside_capture_loop_is_clean(monkeypatch):
    import hfgcs_scanner

    def boom(argv=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(hfgcs_scanner, "_main", boom)
    assert hfgcs_scanner.main([]) == 0


# -- RSSI (kiwirecorder --progress) for the v2 squelch -------------------------------------
def test_rssi_from_progress_lines_rides_on_each_block(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KIWI_RSSI", "-97.5")
    with make(["kiwi.example:8073"]) as src:
        assert src.level_reported
        src.tune(11175)
        _, chunks = read_audio(src, 0.9)
    assert "--progress" in calls(tmp_path)[0]
    levels = [c.level_db for c in chunks if len(c.samples)]
    assert levels and all(lv is None or lv == -97.5 for lv in levels)
    assert levels[-1] == -97.5            # the reader thread has caught up by the end


def test_rssi_off_runs_without_progress(tmp_path):
    with make(["kiwi.example:8073"], rssi=False) as src:
        assert not src.level_reported
        src.tune(11175)
        _, chunks = read_audio(src, 0.5)
    assert "--progress" not in calls(tmp_path)[0]
    assert all(c.level_db is None for c in chunks)


def test_rssi_out_of_range_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KIWI_RSSI", "999.0")
    with make(["kiwi.example:8073"]) as src:
        src.tune(11175)
        _, chunks = read_audio(src, 0.5)
    assert all(c.level_db is None for c in chunks)


def test_stderr_progress_lines_are_parsed_and_bounded(tmp_path):
    """CR-only progress lines, a huge line with no newline, and a session-end line after them."""
    script = tmp_path / "progress.py"
    script.write_text(
        "import sys, struct\n"
        "for i in range(300): sys.stderr.write('\\rBlock: %08x, RSSI: %6.1f ' % (i, -100.0 - i % 3))\n"
        "sys.stderr.write('z' * 100000)\n"
        "sys.stderr.write('\\nkiwi.example:8073 server closed the connection cleanly.'\n"
        "                 ' Reconnecting after 5 seconds\\n')\n"
        "sys.stderr.flush()\n"
        "out = sys.stdout.buffer\n"
        "fmt = struct.pack('<HHIIHH', 1, 1, 12000, 24000, 2, 16)\n"
        "out.write(b'RIFF' + struct.pack('<I', 0x7FFFFFF0) + b'WAVE')\n"
        "out.write(b'fmt ' + struct.pack('<I', len(fmt)) + fmt)\n"
        "out.write(b'data' + struct.pack('<I', 0x7FFFFFF0 - 36))\n"
        "out.flush()\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    session = kiwisdr.KiwiSession(proc)
    session.wait_stderr(10)
    assert session.stderr_done.is_set()
    assert session.rssi_count == 300 and set(session.rssi) <= {-100.0, -101.0, -102.0}
    assert all(len(line) <= kiwisdr.MAX_STDERR_LINE for line in session.stderr_tail)
    assert session.session_end is not None and "server closed" in session.session_end
    session.stop()
