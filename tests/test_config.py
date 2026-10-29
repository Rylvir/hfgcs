import pytest

from hfgcs.config import HFGCS_CHANNELS, Config, load_config

SIX = [4724.0, 6739.0, 8992.0, 11175.0, 13200.0, 15016.0]


def test_six_channels():
    assert [c.dial_khz for c in HFGCS_CHANNELS] == SIX
    assert Config().channels_khz == SIX


def test_tools_use_shared_channels():
    import hfgcs_scanner  # must import without SoapySDR installed
    import hfgcs_waterfall

    assert [hz / 1000 for _, hz in hfgcs_waterfall.HFGCS_PROFILES] == SIX
    assert [name for name, _ in hfgcs_waterfall.HFGCS_PROFILES][0] == "HFGCS 4724"
    assert hfgcs_scanner.load_config().channels_khz == SIX


def test_preroll_minimum():
    with pytest.raises(ValueError):
        Config(preroll_seconds=0.5)


def test_load_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('receiver_id = "kiwi-test"\nhang_seconds = 6.5\nchannels = [8992, 11175]\n')
    cfg = load_config(p)
    assert cfg.receiver_id == "kiwi-test"
    assert cfg.hang_seconds == 6.5
    assert cfg.channels_khz == [8992.0, 11175.0]


def test_load_toml_rejects_unknown(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("bogus = 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_frequency_groups_distinct_and_attributed():
    from hfgcs.config import ALE_STATION_ADDRESSES, FREQUENCY_GROUPS, PRIYOM_HFGCS

    assert set(FREQUENCY_GROUPS) == {"voice", "ale", "siprnet", "exercise"}
    seen = {}
    for name, chans in FREQUENCY_GROUPS.items():
        khz = [c.dial_khz for c in chans]
        assert len(khz) == len(set(khz)), f"duplicate in {name}"
        for c in chans:
            assert c.group == name
            assert c.mode and c.source
            assert c.dial_khz not in seen, f"{c.dial_khz} in {seen.get(c.dial_khz)} and {name}"
            seen[c.dial_khz] = name
    ale = [c.dial_khz for c in FREQUENCY_GROUPS["ale"]]
    assert len(ale) == 10 and ale[0] == 3137 and ale[-1] == 23337
    assert all(c.source == PRIYOM_HFGCS for c in FREQUENCY_GROUPS["ale"] + FREQUENCY_GROUPS["siprnet"])
    assert len(FREQUENCY_GROUPS["siprnet"]) == 8
    assert len(ALE_STATION_ADDRESSES) == 13


def test_scanner_defaults_to_voice_only():
    cfg = Config()
    assert all(c.group == "voice" for c in cfg.channels)
    assert cfg.channels_khz == SIX


def test_load_toml_channel_groups(tmp_path):
    from hfgcs.config import channels_for

    p = tmp_path / "c.toml"
    p.write_text('channel_groups = ["voice", "ale"]\n')
    cfg = load_config(p)
    assert cfg.channels == channels_for("voice", "ale")
    assert len(cfg.channels) == 16
    with pytest.raises(KeyError):
        channels_for("bogus")


def test_squelch_window_defaults_and_validation():
    cfg = Config()
    assert cfg.noise_floor_window_seconds == 30.0
    assert cfg.quiet_median_seconds == 15.0
    Config(noise_floor_window_seconds=0, quiet_median_seconds=0)
    Config(noise_floor_window_seconds=4, quiet_median_seconds=4, hang_seconds=4)
    with pytest.raises(ValueError):
        Config(noise_floor_window_seconds=-1)
    with pytest.raises(ValueError):
        Config(noise_floor_window_seconds=2, hang_seconds=4)
    with pytest.raises(ValueError):
        Config(quiet_median_seconds=-0.1)
    with pytest.raises(ValueError):
        Config(quiet_median_seconds=2, hang_seconds=4)
