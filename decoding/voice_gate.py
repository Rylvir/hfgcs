"""Voice gate for the decode service: is there any speech in this recording?

The recorder's squelch opens on any signal above the noise floor, so some captures hold
only carrier, tones, data bursts or static. Running Whisper on those costs minutes of CPU
on hilltop and invites hallucinated text. Before decoding, the worker (``decoding.worker``)
asks silero-vad, which ships inside faster-whisper (no extra download), how much speech
the clip holds. A clip below ``min_speech_s`` is recorded as skipped (``no_voice``), not as
an error, and is not decoded.

The gate is deliberately permissive (threshold 0.3, one second of speech is enough):
silero misses weak HF speech (docs/decoding.md, "VAD"), and a missed EAM costs far more
than a wasted decode. The ASR itself still runs with VAD off.

Heavy imports (numpy, faster_whisper) happen on first use.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .asr import AsrConfig, _read_wav16k, preprocess

SILENT_RMS = 1e-4  # about -80 dBFS: digital silence / dead air


@dataclass
class GateConfig:
    enabled: bool = True
    threshold: float = 0.3  # silero speech probability (faster-whisper default 0.5)
    min_speech_s: float = 1.0  # less detected speech than this = no voice
    preprocess: str = "bp_norm"  # same ffmpeg chain the ASR sees (asr.PREPROCESS)
    min_silence_duration_ms: int = 1500
    speech_pad_ms: int = 400
    min_speech_duration_ms: int = 200

    def version(self) -> str:
        """Changes when a gate setting changes, so no_voice verdicts are re-checked."""
        if not self.enabled:
            return "off"
        return (f"silero@{self.threshold:g};min={self.min_speech_s:g}s;pre={self.preprocess};"
                f"sil={self.min_silence_duration_ms};pad={self.speech_pad_ms};"
                f"minsp={self.min_speech_duration_ms}")


def load_pcm16k(audio: Path, chain: str):
    """The clip as 16 kHz mono float32, filtered like the ASR input (needs ffmpeg)."""
    with tempfile.TemporaryDirectory() as td:
        return _read_wav16k(preprocess(Path(audio), Path(td) / "gate.wav", AsrConfig(preprocess=chain)))


def silero_regions(pcm, cfg: GateConfig) -> list[tuple[float, float]]:
    """Speech regions (seconds) from faster-whisper's bundled silero-vad."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps  # lazy: optional dependency

    opts = VadOptions(
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_duration_ms,
        min_silence_duration_ms=cfg.min_silence_duration_ms,
        speech_pad_ms=cfg.speech_pad_ms,
    )
    return [(t["start"] / 16000, t["end"] / 16000) for t in get_speech_timestamps(pcm, opts)]


def check_voice(
    audio: Path,
    cfg: Optional[GateConfig] = None,
    *,
    loader: Callable = load_pcm16k,
    detector: Callable = silero_regions,
) -> dict:
    """``{"voice": bool, "speech_s", "regions", "duration_s", "method", "reason"}``.

    ``loader(audio, chain) -> float array`` and ``detector(pcm, cfg) -> [(start, end), ...]``
    are injectable for tests."""
    cfg = cfg or GateConfig()
    if not cfg.enabled:
        return {"voice": True, "method": "off", "reason": "gate disabled"}
    pcm = loader(Path(audio), cfg.preprocess)
    duration = len(pcm) / 16000
    rms = float((pcm.astype("float64") ** 2).mean() ** 0.5) if len(pcm) else 0.0
    base = {"duration_s": round(duration, 2), "method": "silero", "gate_version": cfg.version()}
    if rms < SILENT_RMS:
        return {**base, "voice": False, "speech_s": 0.0, "regions": 0, "rms": rms,
                "reason": "digital silence"}
    regions = detector(pcm, cfg)
    speech = round(sum(max(0.0, e - s) for s, e in regions), 2)
    voice = speech >= cfg.min_speech_s
    reason = (f"{speech:.1f} s of speech" if voice
              else f"only {speech:.1f} s of speech (< {cfg.min_speech_s:g} s)")
    return {**base, "voice": voice, "speech_s": speech, "regions": len(regions), "reason": reason}


__all__ = ["GateConfig", "check_voice", "load_pcm16k", "silero_regions"]
