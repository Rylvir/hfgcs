"""KiwiSDR source with a fake kiwirecorder: etiquette rules and streaming."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hfgcs.config import KiwiSettings
from hfgcs.sources import SourceError, TuneRefused
from hfgcs.sources.kiwisdr import KiwiSource, parse_wav_header, reset_etiquette_state

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


def make(hosts, status=None, resolver=None, clock=None, **kw):
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
