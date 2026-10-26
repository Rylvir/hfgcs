"""Pluggable receivers for the HFGCS scanner.

``make_source(cfg)`` builds the backend named by ``cfg.source``:

* ``soapy``: local SoapySDR device (IQ), the original path;
* ``openwebrx``: demodulated USB audio from an OpenWebRX server's websocket;
* ``kiwisdr``: USB audio from a KiwiSDR via kiwiclient's kiwirecorder.py;
* ``file``: WAV audio or IQ file replay.

Heavy or optional dependencies (SoapySDR, websockets, kiwiclient) are only
imported when that source is used.
"""

from __future__ import annotations

from ..config import SOURCE_NAMES, Config
from .base import AudioSource, BaseSource, Chunk, SourceError, TuneRefused

__all__ = [
    "SOURCE_NAMES",
    "AudioSource",
    "BaseSource",
    "Chunk",
    "SourceError",
    "TuneRefused",
    "make_source",
]


def make_source(cfg: Config, name: str | None = None) -> AudioSource:
    name = name or cfg.source
    if name == "soapy":
        from .soapy import SoapySource

        return SoapySource(cfg)
    if name == "openwebrx":
        from .openwebrx import OpenWebRXSource

        return OpenWebRXSource(cfg.openwebrx)
    if name == "kiwisdr":
        from .kiwisdr import KiwiSource

        return KiwiSource(cfg.kiwisdr)
    if name == "file":
        from .file import FileSource

        return FileSource(cfg.file)
    raise ValueError(f"unknown source {name!r}; choose from {SOURCE_NAMES}")
