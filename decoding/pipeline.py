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
        c = self.cfg
        first = self._transcribe(c.asr, audio, day_callsigns)
        records = _parse(first["words"], c.snap)
        info = {"fired": False, "passes": 1, "asr_seconds": first.get("asr_seconds")}
        rec = main_eam(records)
        if c.second_pass and needs_second_pass(rec, c.second_pass_threshold):
            others = []
            for overrides, snap in c.second_pass_passes:
                out = self._transcribe(replace(c.asr, **overrides), audio, day_callsigns)
                others.append(main_eam(_parse(out["words"], snap)))
            voted = apply_vote(rec, others)
            records = [voted if r is rec else r for r in records]
            info.update(fired=True, passes=1 + len(others))
        if c.preamble_fusion:
            records = [fuse_preamble(r) for r in records]
        return {"records": records, "second_pass": info, "duration_s": first.get("duration_s")}
