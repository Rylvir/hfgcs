"""faster-whisper front end for HF USB voice (prototype; see docs/decoding.md section 1).

The heavy import happens on first use, so the parser and tests run without
the ASR extra installed (pattern borrowed from brisi-jt/skywatch, MIT).
Output is a list of faster-whisper style word dicts that
``decoding.parse_transcript`` accepts directly.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

NATO_WORDS = (
    "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike "
    "November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray "
    "Yankee Zulu"
).split()
# Priyom HFGCS page: ground station callsigns.
STATIONS = (
    "Mainsail Andersen Andrews Ascension Croughton Diego Garcia Elmendorf Hickam "
    "Lajes Offutt Puerto Rico Sigonella West Coast Yokota"
).split()
PROCEDURE = [
    "Skyking",
    "all stations",
    "stand by",
    "message follows",
    "I say again",
    "this is",
    "out",
    "time",
    "authentication",
    "do not answer",
    "characters",
]


def build_prompt(day_callsigns: Sequence[str] = ()) -> str:
    """initial_prompt: a realistic EAM opening in the exact spelling we want
    back (Whisper copies the style of its prompt). Keep it short: the prompt
    shares the 448-token context with the output."""
    cs = ", ".join(day_callsigns) or "Mainsail"
    return (
        f"All stations, all stations, this is {cs}. Alpha Bravo Seven Two Zulu Kilo, stand by. "
        "Message follows. Alpha Bravo Seven Two Zulu Kilo X-ray Niner Echo Four. "
        "I say again. Skyking, Skyking, do not answer."
    )


def build_hotwords(day_callsigns: Sequence[str] = ()) -> str:
    return " ".join([*day_callsigns, "Skyking", "Mainsail", *NATO_WORDS])


@dataclass
class AsrConfig:
    model: str = "small.en"  # CPU default; "large-v3" / "distil-large-v3" on GPU
    device: str = "auto"
    compute_type: str = "int8"  # "float16" on GPU
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: dict = field(
        default_factory=lambda: {
            "threshold": 0.35,  # HF speech under noise scores low; 0.5 drops weak chars
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 1500,  # operator pauses between characters/groups
            "speech_pad_ms": 400,
        }
    )
    condition_on_previous_text: bool = False  # stops repetition loops on long reads
    temperature: tuple = (0.0, 0.2, 0.4)
    no_speech_threshold: float = 0.6
    hallucination_silence_threshold: Optional[float] = 2.0
    bandpass: tuple = (300, 3000)


def preprocess(src: Path, dst: Path, cfg: AsrConfig) -> Path:
    """Bandpass + resample to 16 kHz mono PCM with ffmpeg (Opus/WAV/FLAC in)."""
    lo, hi = cfg.bandpass
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-af",
            f"highpass=f={lo},lowpass=f={hi},dynaudnorm=f=250:g=15",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
    )
    return dst


def _read_wav16k(path: Path):
    """Load the preprocessed 16 kHz PCM_16 WAV as float32 (bypasses PyAV,
    whose version skew with faster-whisper broke decode_audio on this box)."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


class Transcriber:
    def __init__(self, cfg: Optional[AsrConfig] = None):
        self.cfg = cfg or AsrConfig()
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: optional dependency

            self._model = WhisperModel(
                self.cfg.model, device=self.cfg.device, compute_type=self.cfg.compute_type
            )
        return self._model

    def transcribe(self, audio: Path, day_callsigns: Sequence[str] = (), offset_s: float = 0.0) -> dict:
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            wav = preprocess(Path(audio), Path(td) / "in.wav", c)
            segs, info = self._load().transcribe(
                _read_wav16k(wav),
                language="en",
                beam_size=c.beam_size,
                word_timestamps=True,
                vad_filter=c.vad_filter,
                vad_parameters=c.vad_parameters,
                initial_prompt=build_prompt(day_callsigns),
                hotwords=build_hotwords(day_callsigns),
                condition_on_previous_text=c.condition_on_previous_text,
                temperature=list(c.temperature),
                no_speech_threshold=c.no_speech_threshold,
                hallucination_silence_threshold=c.hallucination_silence_threshold,
            )
            segs = list(segs)
        words = [
            {
                "word": w.word.strip(),
                "probability": round(w.probability, 4),
                "start": round(w.start + offset_s, 2),
                "end": round(w.end + offset_s, 2),
            }
            for s in segs
            for w in (s.words or [])
        ]
        return {
            "model": c.model,
            "compute_type": c.compute_type,
            "duration_s": info.duration,
            "text": " ".join(s.text.strip() for s in segs),
            "words": words,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "text": s.text,
                }
                for s in segs
            ],
        }
