"""OpenWebRX audio source: protocol, ADPCM, etiquette. Uses a local mock server only."""

import json

import numpy as np
import pytest
from owrx_mock import MockOpenWebRX, tone_pcm

from hfgcs.config import OpenWebRXSettings
from hfgcs.sources import SourceError, TuneRefused
from hfgcs.sources.openwebrx import (
    ImaAdpcmCodec,
    OpenWebRXSource,
    ProfileWindow,
    SyncAdpcmDecoder,
    encode_sync_adpcm,
    ws_url,
)

pytest.importorskip("websockets")


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


def collect(src, seconds, max_reads=2000):
    """Read until ``seconds`` of audio arrived (or the reads run out)."""
    out, chunks = [], []
    n = 0
    for _ in range(max_reads):
        c = src.read(0.2)
        chunks.append(c)
        if len(c.samples):
            out.append(c.samples)
            n += len(c.samples)
        if n >= seconds * src.sample_rate:
            break
    return (np.concatenate(out) if out else np.zeros(0, np.float32)), chunks


def peak_hz(x, rate):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * rate / len(x)


# -- ADPCM ------------------------------------------------------------------------------------
def test_adpcm_reference_vector():
    # SYNC, index 0, predictor 0, then 0x77: nibble 7 at step 7 -> +11, index 8 (step 16) -> +30
    data = b"SYNC" + np.array([0, 0], "<i2").tobytes() + bytes([0x77])
    assert SyncAdpcmDecoder().decode(data).tolist() == [11, 41]


def test_adpcm_roundtrip_any_framing():
    pcm = tone_pcm(1000, 1.0)
    enc, _ = encode_sync_adpcm(pcm)
    assert enc.count(b"SYNC") >= len(pcm) // 2 // 1001
    whole = SyncAdpcmDecoder().decode(enc)
    dec = SyncAdpcmDecoder()
    rng = np.random.default_rng(3)
    cuts = np.sort(rng.choice(np.arange(1, len(enc)), 40, replace=False))
    parts = np.split(np.frombuffer(enc, np.uint8), cuts)  # cuts land inside SYNC headers too
    pieces = np.concatenate([dec.decode(p.tobytes()) for p in parts])
    np.testing.assert_array_equal(pieces, whole)
    err = whole[: len(pcm)].astype(float) - pcm
    snr = 10 * np.log10(np.mean(pcm.astype(float) ** 2) / np.mean(err**2))
    assert snr > 20


def test_adpcm_joins_mid_stream():
    pcm = tone_pcm(700, 0.5)
    enc, _ = encode_sync_adpcm(pcm)
    first_sync = enc.index(b"SYNC", 10)  # the second header
    dec = SyncAdpcmDecoder().decode(enc[first_sync - 7 :])  # garbage before it is skipped
    ref = SyncAdpcmDecoder().decode(enc)
    np.testing.assert_array_equal(dec, ref[-len(dec) :])


def test_codec_matches_standard_ima_steps():
    c = ImaAdpcmCodec()
    assert [c.decode_nibble(n) for n in (0, 8, 4)] == [0, 0, 7]


def test_ws_url():
    assert ws_url("https://waterfall.scanplacer.com") == "wss://waterfall.scanplacer.com/ws/"
    assert ws_url("http://127.0.0.1:8073/") == "ws://127.0.0.1:8073/ws/"
    assert ws_url("ws://h:1/ws/") == "ws://h:1/ws/"


def test_profile_window_bounds():
    w = ProfileWindow(11_175_000, 768_000)
    assert w.covers(11_175_000, 300, 3000, 0.9)
    assert w.covers(11_175_000 + 340_000, 300, 3000, 0.9)
    assert not w.covers(11_175_000 + 380_000, 300, 3000, 0.9)
    assert not w.covers(13_200_000, 300, 3000, 1.0)


# -- protocol against the mock server ------------------------------------------------------------
@pytest.mark.parametrize("compression", ["adpcm", "none"])
def test_handshake_tune_and_audio(compression):
    with MockOpenWebRX(compression=compression) as srv:
        src = OpenWebRXSource(settings(srv.url))
        with src:
            assert src.server_version.startswith("CLIENT DE SERVER")
            assert src.window.center_hz == 11_175_000
            src.tune(11175)
            audio, chunks = collect(src, 1.0)
        assert src.sample_rate == 12_000
        assert len(audio) >= 12_000
        assert abs(peak_hz(audio[:12_000], 12_000) - 1000) < 3
        assert chunks[-1].level_db == pytest.approx(-30.0)
        sent = srv.all_sent()
        kinds = [m["type"] for m in sent]
        assert kinds[0] == "connectionproperties"
        assert sent[0]["params"]["output_rate"] == 12_000
        dsp = [m for m in sent if m["type"] == "dspcontrol" and "params" in m][0]["params"]
        expected = {"mod": "usb", "low_cut": 300.0, "high_cut": 3000.0}
        expected.update(offset_freq=0, squelch_level=-150)
        assert dsp == expected
        assert "selectprofile" not in kinds and "setfrequency" not in kinds
        assert srv.received[0][0] == "SERVER DE CLIENT client=hfgcs type=receiver"
        assert "hfgcs" in srv.headers[0].get("User-Agent", srv.headers[0].get("user-agent", ""))


def test_offset_within_wide_profile():
    with MockOpenWebRX(profile="airspy|wide-11000") as srv:
        with OpenWebRXSource(settings(srv.url)) as src:
            src.tune(11175)
            collect(src, 0.1)
        dsp = [m for m in srv.all_sent() if m["type"] == "dspcontrol" and "params" in m]
        assert dsp[0]["params"]["offset_freq"] == 175_000


def test_refuses_channel_outside_active_profile():
    with MockOpenWebRX() as srv:
        with OpenWebRXSource(settings(srv.url)) as src:
            assert src.can_tune(11175)
            assert not src.can_tune(13200)
            with pytest.raises(TuneRefused, match="outside the active profile airspy\\|hfgcs-11175"):
                src.tune(13200)
            assert "allow_profile_switch" in src.refusal(13200)
        kinds = [m["type"] for m in srv.all_sent()]
        assert "selectprofile" not in kinds
        assert "dspcontrol" not in kinds  # nothing was started for the refused channel


def test_profile_switch_only_when_allowed():
    with MockOpenWebRX() as srv:
        with OpenWebRXSource(settings(srv.url, allow_profile_switch=True)) as src:
            src.tune(13200)
            assert src.window.center_hz == 13_200_000
            audio, _ = collect(src, 0.2)
        sent = srv.all_sent()
        assert {"type": "selectprofile", "params": {"profile": "airspy|hfgcs-13200"}} in sent
        dsp = [m for m in sent if m["type"] == "dspcontrol" and "params" in m]
        assert dsp[-1]["params"]["offset_freq"] == 0
        assert "setfrequency" not in [m["type"] for m in sent]


def test_someone_else_moves_the_radio_we_stop_and_do_not_move_it_back(caplog):
    def script(conn):
        return [(tone_pcm(1000, 3.0), 1e-3)]

    events = {(0, 20): ("config", "airspy|hfgcs-15016"), (0, 60): ("config", "airspy|hfgcs-11175")}
    with MockOpenWebRX(audio=script, events=events) as srv:
        with OpenWebRXSource(settings(srv.url)) as src:
            src.tune(11175)
            got = 0
            saw_gap = False
            for _ in range(400):
                c = src.read(0.2)
                got += len(c.samples)
                saw_gap |= c.gap and got > 20 * 240
                if got >= 50 * 240:
                    break
            assert saw_gap
            assert src.off_channel_reason is None
        assert "11175.0 kHz is no longer receivable" in caplog.text
        sent = srv.all_sent()
        assert "selectprofile" not in [m["type"] for m in sent]
        # frames 20..59 were for the other profile and were dropped
        assert got < (len(tone_pcm(1000, 3.0)) - 40 * 240) + 240


def test_reconnects_after_drop_with_gap_marker():
    def script(conn):
        return [(tone_pcm(1000, 1.0), 1e-3)]

    with MockOpenWebRX(audio=script, events={(0, 10): ("close", None)}) as srv:
        sleeps = []
        src = OpenWebRXSource(settings(srv.url), sleep=sleeps.append)
        with src:
            src.tune(11175)
            audio, chunks = collect(src, 0.8)
        assert srv.connections == 2
        assert any(c.gap for c in chunks)
        assert sleeps and all(0.008 <= s <= 0.024 for s in sleeps)
        # the retune after reconnecting stays inside the profile
        assert sum(1 for m in srv.all_sent() if m["type"] == "connectionproperties") == 2


def test_backoff_grows_and_gives_up():
    class Dead:
        def __call__(self, url, s):
            raise OSError("connection refused")

    sleeps = []
    src = OpenWebRXSource(
        settings("http://127.0.0.1:9", backoff_initial_s=5, backoff_max_s=40, max_reconnects=5),
        connect=Dead(),
        sleep=sleeps.append,
    )
    with pytest.raises(SourceError):
        src.open()
    with pytest.raises(SourceError, match="giving up after 5"):
        src.read(0.1)
    nominal = [5, 10, 20, 40, 40]
    assert len(sleeps) == 5
    for s, n in zip(sleeps, nominal, strict=True):
        assert 0.8 * n <= s <= 1.2 * n


def test_polite_tune_rate_limit():
    with MockOpenWebRX(profile="airspy|wide-11000") as srv:
        sleeps = []
        src = OpenWebRXSource(settings(srv.url, min_tune_interval_s=1.0), sleep=sleeps.append)
        with src:
            src.tune(11175)
            src.tune(11226)
        assert len(sleeps) == 1 and 0.5 < sleeps[0] <= 1.0


def test_backoff_message_is_honoured():
    class Conn:
        def __init__(self):
            backoff = json.dumps({"type": "backoff", "reason": "full"})
            self.msgs = ["CLIENT DE SERVER server=openwebrx", backoff]

        def send(self, m):
            pass

        def recv(self, timeout=None):
            return self.msgs.pop(0)

        def close(self):
            pass

    src = OpenWebRXSource(settings("http://x"), connect=lambda u, s: Conn())
    with pytest.raises(SourceError, match="back off"):
        src.open()


def test_scanner_records_from_openwebrx_with_smeter_squelch(tmp_path):
    import soundfile as sf

    import hfgcs_scanner
    from hfgcs.config import Channel, Config

    rng = np.random.default_rng(1)

    def quiet(seconds):
        return (rng.standard_normal(int(seconds * 12_000)) * 3000).astype(np.int16)  # AGC'd noise

    # The audio level is flat (server AGC); only the S-meter shows the signal.
    def script(conn):
        return [(quiet(3), 1e-9), (tone_pcm(1000, 3.0, amp=0.1), 1e-6), (quiet(3), 1e-9)]

    with MockOpenWebRX(audio=script) as srv:
        c = Config(
            channels=(Channel(11175.0),),
            hang_seconds=1.0,
            preroll_seconds=1.0,
            min_activity_seconds=0.5,
            record_dir=tmp_path / "rec",
            catalog_file=tmp_path / "cat.jsonl",
            source="openwebrx",
            openwebrx=settings(srv.url),
        )
        sc = hfgcs_scanner.HFGCSScanner(c)
        assert sc.start() == [11175.0]
        sc.run(max_seconds=8.9)
        sc.close()
    assert len(sc.saved) == 1
    e = sc.saved[0]
    assert e["receiver_id"] == "owrx-127.0.0.1"
    assert e["snr_db_peak"] > 20
    audio, rate = sf.read(e["file"])
    seg = audio[int(1.5 * rate) : int(2.5 * rate)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    assert abs(np.argmax(spec) * rate / len(seg) - 1000) < 3


def test_scanner_skips_channels_outside_profile_and_fails_if_none(tmp_path):
    import hfgcs_scanner
    from hfgcs.config import Channel, Config

    with MockOpenWebRX() as srv:
        c = Config(source="openwebrx", openwebrx=settings(srv.url), record_dir=tmp_path)
        sc = hfgcs_scanner.HFGCSScanner(c)
        assert sc.start() == [11175.0]
        assert sorted(sc.refused) == [4724.0, 6739.0, 8992.0, 13200.0, 15016.0]
        sc.close()
        c = Config(source="openwebrx", openwebrx=settings(srv.url), channels=(Channel(4724.0),))
        sc = hfgcs_scanner.HFGCSScanner(c)
        with pytest.raises(TuneRefused, match="none of the configured channels"):
            sc.start()
        sc.close()
    assert "selectprofile" not in [m["type"] for m in srv.all_sent()]
