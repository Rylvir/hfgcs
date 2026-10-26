"""Audio clip -> broadcast records, with the default decoding policy.

Default (``DecodeConfig()``), chosen on the eval set (docs/decoding.md §1):
  1. first pass: ``AsrConfig()`` (VAD off, previous-text conditioning, EAM
     prompt + hotwords, bp_norm), then ``snap_words``, then the parser;
  2. second pass, only when the first-pass EAM has
     ``reads_agree_ratio < 0.3`` (``vote.needs_second_pass``): decode again
     with the two ``vote.SECOND_PASS_PASSES`` and replace the body with the
     3-way character vote (flag ``second_pass_vote``);
  3. preamble fusion stays off (neutral on the eval).
The ASR stack is imported lazily, so this module imports without
faster-whisper installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

from .asr import AsrConfig, Transcriber
from .eam_parser import parse_transcript
from .preamble import fuse_preamble
from .snap import snap_words
from .vote import AGREE_THRESHOLD, SECOND_PASS_PASSES, apply_vote, needs_second_pass


@dataclass
class DecodeConfig:
    asr: AsrConfig = field(default_factory=AsrConfig)
    snap: bool = True
    second_pass: bool = True
    second_pass_threshold: float = AGREE_THRESHOLD
    second_pass_passes: tuple = SECOND_PASS_PASSES
    preamble_fusion: bool = False


def main_eam(records: Sequence[dict]) -> Optional[dict]:
    """The EAM record the second pass works on: the one with the longest body."""
    eams = [r for r in records if r.get("type") == "eam" and r.get("body")]
    return max(eams, key=lambda r: len(r["body"])) if eams else None


def _parse(words: list, snap: bool) -> list[dict]:
    return parse_transcript(snap_words(words) if snap else words)


class Decoder:
    """``transcribe`` may be injected (tests, caching); it is called as
    ``transcribe(asr_config, audio, day_callsigns) -> {"words": [...], ...}``."""

    def __init__(self, cfg: Optional[DecodeConfig] = None, transcribe: Optional[Callable] = None):
        self.cfg = cfg or DecodeConfig()
        self._transcribers: dict[str, Transcriber] = {}
        self._transcribe = transcribe or self._asr

    def _asr(self, cfg: AsrConfig, audio: Path, day_callsigns: Sequence[str]) -> dict:
        key = cfg.key()
        if key not in self._transcribers:
            first = next(iter(self._transcribers.values()), None)
            self._transcribers[key] = Transcriber(cfg, share_with=first)
        return self._transcribers[key].transcribe(Path(audio), day_callsigns)

    def decode(self, audio: Path, day_callsigns: Sequence[str] = ()) -> dict:
        """Returns ``{records, second_pass, duration_s, asr, decoder_version}``;
        ``asr`` is the raw first-pass transcriber output (the words the
        records' ``segment.word_start/word_end`` index)."""
        c = self.cfg
        first = self._transcribe(c.asr, audio, day_callsigns)
        records = _parse(first["words"], c.snap)
        info = second_pass_info(c)
        info.update(fired=False, passes=1, asr_seconds=first.get("asr_seconds"))
        rec = main_eam(records)
        if c.second_pass and needs_second_pass(rec, c.second_pass_threshold):
            others = []
            extra_s = []
            for overrides, snap in c.second_pass_passes:
                out = self._transcribe(replace(c.asr, **overrides), audio, day_callsigns)
                extra_s.append(out.get("asr_seconds"))
                others.append(main_eam(_parse(out["words"], snap)))
            voted = apply_vote(rec, others)
            records = [voted if r is rec else r for r in records]
            info.update(fired=True, passes=1 + len(others), second_pass_asr_seconds=extra_s)
        if c.preamble_fusion:
            records = [fuse_preamble(r) for r in records]
        return {
            "records": records,
            "second_pass": info,
            "duration_s": first.get("duration_s"),
            "asr": {**first, "config_key": c.asr.key(), "day_callsigns": list(day_callsigns)},
            "decoder_version": decoder_version(c),
        }


def second_pass_tag(cfg: DecodeConfig) -> str:
    """``vote3@0.3`` (n-way character vote at that agreement threshold) or ``off``."""
    if not cfg.second_pass:
        return "off"
    return f"vote{1 + len(cfg.second_pass_passes)}@{cfg.second_pass_threshold:g}"


def second_pass_info(cfg: DecodeConfig) -> dict:
    """Static description of the second pass (what would run, not whether it did)."""
    return {
        "mode": second_pass_tag(cfg),
        "enabled": cfg.second_pass,
        "threshold": cfg.second_pass_threshold,
        "configs": [
            {"config_key": replace(cfg.asr, **o).key(), "overrides": dict(o), "snap": snap}
            for o, snap in cfg.second_pass_passes
        ]
        if cfg.second_pass
        else [],
    }


def decoder_version(cfg: DecodeConfig) -> str:
    """``decoding/<pkg>;parser=<schema>;asr=<AsrConfig.key()>;second_pass=<vote3@0.3|off>``."""
    from . import __version__
    from .eam_parser import SCHEMA_VERSION

    return (
        f"decoding/{__version__};parser={SCHEMA_VERSION};asr={cfg.asr.key()};"
        f"second_pass={second_pass_tag(cfg)}"
    )
