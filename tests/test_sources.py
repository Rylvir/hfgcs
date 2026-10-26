"""Source interface, file replay through the capture pipeline, SoapySDR wrapper, config/CLI."""

import json
import types

import numpy as np
import pytest
import soundfile as sf
from conftest import noise, tone

import hfgcs_scanner
from hfgcs.config import Config, FileSettings, load_config
from hfgcs.dsp import AudioFrontEnd, Resampler
from hfgcs.sources import AudioSource, SourceError, TuneRefused, make_source
from hfgcs.sources.file import FileSource
from hfgcs.sources.soapy import SoapySource


def peak_hz(x, rate):
    x = np.asarray(x, float)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * rate / len(x)


def cfg(tmp_path, **kw):
    base = dict(
        hang_seconds=1.5,
        preroll_seconds=1.0,
        min_activity_seconds=0.5,
        record_dir=tmp_path / "rec",
        catalog_file=tmp_path / "catalog.jsonl",
    )
    base.update(kw)
    return Config(**base)


# -- DSP helpers ---------------------------------------------------------------------------------
@pytest.mark.parametrize("rates", [(12_000, 8_000), (48_000, 8_000), (11_025, 8_000), (8_000, 8_000)])
def test_resampler_streaming_equals_one_shot_and_keeps_pitch(rates):
    fin, fout = rates
    t = np.arange(fin) / fin
    x = np.sin(2 * np.pi * 1234 * t)
    one = Resampler(fin, fout).process(x)
    r = Resampler(fin, fout)
    sizes = np.random.default_rng(0).integers(1, 700, 200)
    parts, i = [], 0
    for n in sizes:
        parts.append(r.process(x[i : i + n]))
        i += n
    parts.append(r.process(x[i:]))
    np.testing.assert_allclose(np.concatenate(parts), one, atol=1e-5)
    assert abs(len(one) - fout) <= 1
    assert abs(peak_hz(one[200:], fout) - 1234) < 3


def test_audio_front_end_measures_slot_power():
    fe = AudioFrontEnd(12_000, 8_000)
    t = np.arange(12_000) / 12_000
    y, p = fe.process(0.5 * np.sin(2 * np.pi * 1000 * t))
    assert len(y) == 8000
    assert p == pytest.approx(0.125, rel=0.1)


# -- interface -------------------------------------------------------------------------------------
def test_sources_implement_protocol(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(800, np.float32), 8000)
    c = Config()
    srcs = [
        make_source(c, "soapy"),
        make_source(c, "openwebrx"),
        make_source(c, "kiwisdr"),
        FileSource(FileSettings(path=str(path))),
    ]
    for s in srcs:
        assert isinstance(s, AudioSource)
        assert s.kind in ("iq", "audio")
    assert [s.kind for s in srcs] == ["iq", "audio", "audio", "audio"]
    assert srcs[1].receiver_id == "owrx-waterfall.scanplacer.com"
    assert srcs[1].settings.allow_profile_switch is False
    with pytest.raises(ValueError):
        make_source(c, "websdr")


# -- file replay end to end ------------------------------------------------------------------------
def burst(rate, f_hz, kind):
    """3 s noise, 3 s tone, 4 s noise."""
    rng = np.random.default_rng(5)
    def n(seconds):
        return int(seconds * rate)

    if kind == "audio":
        x = 0.01 * rng.standard_normal(n(10))
        t = np.arange(n(3)) / rate
        x[n(3) : n(6)] += 0.2 * np.sin(2 * np.pi * f_hz * t)
        return x.astype(np.float32)
    x = noise(10, 0.01, seed=7, rate=rate)
    x[n(3) : n(6)] += tone(f_hz, 3, 0.05, rate=rate)
    return x.astype(np.complex64)


def run_file(tmp_path, settings, channel, **cfg_kw):
    c = cfg(tmp_path, channels=(), **cfg_kw)
    c = hfgcs_scanner.apply_cli(c, types.SimpleNamespace(
        source="file", channel=[channel], receiver_id=None, url=None, allow_profile_switch=False,
        kiwi_host=None, kiwi_ident=None, file=None, file_format=None, sample_rate=None,
        center_khz=None, realtime=False,
    ))
    sc = hfgcs_scanner.HFGCSScanner(c, FileSource(settings))
    assert sc.start() == [channel]
    sc.run()
    sc.close()
    return sc


def check_recording(sc, f_expected):
    assert len(sc.saved) == 1
    entry = sc.saved[0]
    audio, rate = sf.read(entry["file"])
    assert rate == 8000
    start = int((1.0 + 0.5) * rate)  # skip pre-roll and the first half second
    assert abs(peak_hz(audio[start : start + 8000], rate) - f_expected) < 3
    assert 3.0 < entry["duration_sec"] < 8.0
    lines = [json.loads(x) for x in open(sc.cfg.catalog_file)]
    assert lines[0]["receiver_id"] == entry["receiver_id"]
    return entry


def test_wav_audio_replay_through_scanner(tmp_path):
    path = tmp_path / "owrx_11175.wav"
    sf.write(path, burst(12_000, 1000, "audio"), 12_000)
    sc = run_file(tmp_path, FileSettings(path=str(path), dial_khz=11175), 11175.0)
    entry = check_recording(sc, 1000)
    assert entry["receiver_id"] == "file-owrx_11175"
    assert entry["frequency_khz"] == 11175.0


def test_iq_cf32_replay_tunes_inside_recording(tmp_path):
    rate = 96_000
    # USB voice at 11175 kHz + 1 kHz, recorded with the radio centred at 11160 kHz
    x = burst(rate, 15_000 + 1000, "iq")
    path = tmp_path / "wide.cf32"
    x.astype(np.complex64).tofile(path)
    settings = FileSettings(path=str(path), sample_rate=rate, center_khz=11160)
    src = FileSource(settings)
    assert src.kind == "iq"
    assert not src.can_tune(11300)
    with pytest.raises(TuneRefused):
        src.tune(11300)
    sc = run_file(tmp_path, settings, 11175.0)
    check_recording(sc, 1000)


def test_iq_wav_replay(tmp_path):
    rate = 48_000
    x = burst(rate, 1500, "iq")
    path = tmp_path / "iq.wav"
    sf.write(path, np.stack([x.real, x.imag], axis=1).astype(np.float32), rate, subtype="FLOAT")
    sc = run_file(tmp_path, FileSettings(path=str(path)), 8992.0)
    check_recording(sc, 1500)


def test_realtime_pacing_and_loop(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(8000, np.float32), 8000)
    clock = types.SimpleNamespace(t=0.0)
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock.t += s

    src = FileSource(
        FileSettings(path=str(path), realtime=True, loop=True, block_seconds=0.25),
        clock=lambda: clock.t,
        sleep=sleep,
    )
    with src:
        chunks = [src.read() for _ in range(6)]
    assert sum(sleeps) == pytest.approx(1.5)
    assert [c.gap for c in chunks] == [False] * 4 + [True, False]


def test_file_errors(tmp_path):
    with pytest.raises(SourceError, match="no such file"):
        FileSource(FileSettings(path=str(tmp_path / "missing.wav")))
    raw = tmp_path / "x.raw"
    raw.write_bytes(b"\0" * 16)
    with pytest.raises(SourceError, match="format"):
        FileSource(FileSettings(path=str(raw)))
    with pytest.raises(SourceError, match="sample_rate"):
        FileSource(FileSettings(path=str(raw), format="cs16"))


# -- SoapySDR stays available as the "soapy" source --------------------------------------------------
class FakeSoapy:
    SOAPY_SDR_RX, SOAPY_SDR_CF32 = 0, "CF32"

    def __init__(self):
        self.calls = []
        fake = self

        class Device:
            def __init__(self, args):
                fake.calls.append(("open", args))

            def setSampleRate(self, d, c, r):
                fake.calls.append(("rate", r))

            def setGainMode(self, d, c, agc):
                fake.calls.append(("agc", agc))

            def setupStream(self, d, fmt):
                return "stream"

            def activateStream(self, s):
                pass

            def deactivateStream(self, s):
                fake.calls.append(("deactivate", s))

            def closeStream(self, s):
                fake.calls.append(("close", s))

            def setFrequency(self, d, c, hz):
                fake.calls.append(("freq", hz))

            def readStream(self, s, bufs, n):
                bufs[0][:] = 0.1
                return types.SimpleNamespace(ret=n)

        self.Device = Device


def test_soapy_source_with_fake_driver():
    fake = FakeSoapy()
    c = Config()
    src = SoapySource(c, soapy_module=fake)
    with src:
        src.tune(11175)
        chunk = src.read()
    assert ("open", "driver=airspyhf") in fake.calls
    assert ("agc", False) in fake.calls
    assert ("freq", 11_175_000) in fake.calls
    assert ("close", "stream") in fake.calls
    assert len(chunk.samples) == 8192 and chunk.rate == c.iq_rate
    assert src.receiver_id == "airspyhf-local"


# -- config and CLI ----------------------------------------------------------------------------------
def test_toml_source_tables(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        'source = "openwebrx"\nchannels = [11175]\n'
        '[openwebrx]\nurl = "http://127.0.0.1:8073"\nlow_cut_hz = 300\n'
        '[kiwisdr]\nhosts = ["a.example:8073"]\nident = "N0CALL"\n'
    )
    c = load_config(p)
    assert c.source == "openwebrx"
    assert c.openwebrx.url == "http://127.0.0.1:8073"
    assert c.openwebrx.allow_profile_switch is False
    assert c.kiwisdr.hosts == ("a.example:8073",)
    p.write_text('[openwebrx]\nallow_center_freq_change = true\n')
    with pytest.raises(ValueError, match="Unknown \\[openwebrx\\] keys"):
        load_config(p)
    p.write_text('source = "websdr"\n')
    with pytest.raises(ValueError, match="source must be"):
        load_config(p)


def test_cli_flags_build_sources():
    args = hfgcs_scanner.build_parser().parse_args(
        ["--source", "openwebrx", "--url", "http://127.0.0.1:8073", "--channel", "11175"]
    )
    c = hfgcs_scanner.apply_cli(Config(), args)
    assert c.source == "openwebrx" and c.channels_khz == [11175.0]
    assert make_source(c).url == "ws://127.0.0.1:8073/ws/"
    args = hfgcs_scanner.build_parser().parse_args(["--source", "kiwisdr", "--kiwi-host", "k.example:8074"])
    assert hfgcs_scanner.apply_cli(Config(), args).kiwisdr.hosts == ("k.example:8074",)


def test_cli_kiwi_disabled_by_default(capsys):
    argv = ["--source", "kiwisdr", "--kiwi-host", "k.example:8074", "--kiwi-ident", "N0CALL"]
    rc = hfgcs_scanner.main(argv)
    assert rc == 1
    assert "enabled = true" in capsys.readouterr().err


def test_cli_file_replay(tmp_path):
    path = tmp_path / "rec.wav"
    sf.write(path, burst(8000, 800, "audio"), 8000)
    conf = tmp_path / "c.toml"
    conf.write_text(
        f'record_dir = "{tmp_path / "out"}"\ncatalog_file = "{tmp_path / "cat.jsonl"}"\n'
        "hang_seconds = 1.5\npreroll_seconds = 1.0\nmin_activity_seconds = 0.5\n"
    )
    argv = ["--config", str(conf), "--source", "file", "--file", str(path), "--channel", "4724"]
    rc = hfgcs_scanner.main(argv)
    assert rc == 0
    lines = (tmp_path / "cat.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["frequency_khz"] == 4724.0
