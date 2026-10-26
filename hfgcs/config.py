"""Shared configuration for every HFGCS tool.

All channel lists and tuning constants live here so the scanner, the
OpenWebRX survey and future receivers agree. Values can be overridden from
a TOML file (see ``config.example.toml``) with :func:`load_config`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:
        _toml = None


PRIYOM_HFGCS = (
    "Priyom.org HFGCS page, https://priyom.org/military-stations/united-states/hfgcs (checked 2026-10-26)"
)
SKYWAVE_QUICKTUNE = "skywavelinux.com HFGCS quick-tune list, https://skywavelinux.com/hfgcs-quick-tune-list.html"
COMMUNITY_LOGS = "community logs (NEET INTEL, eam.watch); not on Priyom's voice list"


@dataclass(frozen=True)
class Channel:
    """One HFGCS frequency. ``dial_khz`` is the USB suppressed carrier."""

    dial_khz: float
    note: str = ""
    mode: str = "USB"
    group: str = "voice"
    source: str = ""

    @property
    def dial_hz(self) -> float:
        return self.dial_khz * 1000.0

    @property
    def profile_name(self) -> str:
        """Profile label expected on the OpenWebRX receiver."""
        return f"HFGCS {self.dial_khz:g}"


def _group(group: str, mode: str, source: str, khz: list[float]) -> tuple[Channel, ...]:
    return tuple(Channel(float(k), mode=mode, group=group, source=source) for k in khz)


# The six HFGCS USB voice channels. These are the scanner's default.
HFGCS_CHANNELS: tuple[Channel, ...] = (
    Channel(4724.0, "night primary", source=PRIYOM_HFGCS),
    Channel(6739.0, "secondary", source=COMMUNITY_LOGS),
    Channel(8992.0, "primary", source=PRIYOM_HFGCS),
    Channel(11175.0, "day primary, most active", source=PRIYOM_HFGCS),
    Channel(13200.0, "secondary", source=COMMUNITY_LOGS),
    Channel(15016.0, "day secondary", source=PRIYOM_HFGCS),
)

# 2G ALE (MIL-STD-188-141) network: AMD text, soundings every ~3 h, some voice.
HFGCS_ALE_CHANNELS = _group(
    "ale", "ALE-2G", PRIYOM_HFGCS,
    [3137, 4721, 5708, 6721, 9025, 11226, 13215, 15043, 18003, 23337],
)

# SIPRNet HF gateway: links set up with 2G ALE, data in MIL-STD-188-110A.
HFGCS_SIPRNET_CHANNELS = _group(
    "siprnet", "ALE-2G/188-110A", PRIYOM_HFGCS,
    [3113, 5702, 6715, 8968, 11181, 15091, 17976, 27870],
)

# Frequencies reported in use during exercises (not a Priyom list).
HFGCS_EXERCISE_CHANNELS = _group(
    "exercise", "USB", SKYWAVE_QUICKTUNE,
    [6697, 8776, 11244, 12082, 16309],
)

FREQUENCY_GROUPS: dict[str, tuple[Channel, ...]] = {
    "voice": HFGCS_CHANNELS,
    "ale": HFGCS_ALE_CHANNELS,
    "siprnet": HFGCS_SIPRNET_CHANNELS,
    "exercise": HFGCS_EXERCISE_CHANNELS,
}

# Ground-station ALE self-addresses (SIPRNet uses the same with suffix "SPR").
# Source: PRIYOM_HFGCS.
ALE_STATION_ADDRESSES: dict[str, str] = {
    "ADW": "Andrews",
    "AED": "Elmendorf",
    "CRO": "Croughton",
    "GUA": "Andersen",
    "HAW": "Ascension",
    "HIK": "Hickam",
    "ICZ": "Sigonella",
    "JDG": "Diego Garcia",
    "JNR": "Puerto Rico",
    "JTY": "Yokota",
    "MCC": "West Coast",
    "OFF": "Offutt",
    "PLA": "Lajes",
}


def channels_for(*groups: str) -> tuple[Channel, ...]:
    """Channels from the named groups, e.g. ``channels_for("voice", "ale")``."""
    out: list[Channel] = []
    for g in groups:
        if g not in FREQUENCY_GROUPS:
            raise KeyError(f"unknown frequency group {g!r}; known: {sorted(FREQUENCY_GROUPS)}")
        out.extend(FREQUENCY_GROUPS[g])
    return tuple(out)


@dataclass
class Config:
    # The scanner tunes these. Default: the six voice channels only.
    channels: tuple[Channel, ...] = HFGCS_CHANNELS
    receiver_id: str = "airspyhf-local"
    soapy_args: str = "driver=airspyhf"
    tuner_agc: bool = False  # tuner AGC makes dBFS squelch levels meaningless

    # Sample rates
    iq_rate: int = 192_000
    audio_rate: int = 8_000

    # USB passband (audio Hz above the dial frequency)
    usb_low_hz: float = 0.0
    usb_high_hz: float = 3_000.0

    # Scan and capture
    dwell_seconds: float = 2.0
    hang_seconds: float = 4.0  # keep recording this long after the signal drops
    preroll_seconds: float = 1.5  # audio kept before activity is confirmed
    min_activity_seconds: float = 0.8
    max_record_seconds: float = 900.0  # safety cap for a stuck squelch

    # Squelch (measured on the 3 kHz voice slot)
    open_snr_db: float = 8.0
    close_snr_db: float = 5.0  # hysteresis
    noise_floor_alpha: float = 0.02  # EMA weight per block, only while closed

    # AGC on the demodulated audio
    agc_target: float = 0.3
    agc_attack_seconds: float = 0.01
    agc_decay_seconds: float = 0.5
    agc_max_gain_db: float = 60.0

    record_dir: Path = Path("recordings")
    catalog_file: Path = Path("hfgcs_catalog.jsonl")

    @property
    def channels_khz(self) -> list[float]:
        return [c.dial_khz for c in self.channels]

    def __post_init__(self) -> None:
        if self.iq_rate % self.audio_rate:
            raise ValueError("iq_rate must be an integer multiple of audio_rate")
        if self.preroll_seconds < 0.8:
            raise ValueError("preroll_seconds must be at least 0.8 s")
        if self.close_snr_db > self.open_snr_db:
            raise ValueError("close_snr_db must not exceed open_snr_db")
        self.record_dir = Path(self.record_dir)
        self.catalog_file = Path(self.catalog_file)


def load_config(path: str | Path | None = None) -> Config:
    """Return the default config, overridden by a TOML file if given."""
    if path is None:
        return Config()
    if _toml is None:
        raise RuntimeError("Reading TOML on Python 3.10 needs the 'tomli' package")
    with open(path, "rb") as fh:
        data = _toml.load(fh)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - known - {"channel_groups"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if "channel_groups" in data:
        if "channels" in data:
            raise ValueError("use either 'channels' or 'channel_groups', not both")
        data["channels"] = channels_for(*data.pop("channel_groups"))
    elif "channels" in data:
        data["channels"] = tuple(
            Channel(float(c["dial_khz"]), c.get("note", "")) if isinstance(c, dict) else Channel(float(c))
            for c in data["channels"]
        )
    return Config(**data)


DEFAULT = Config()
