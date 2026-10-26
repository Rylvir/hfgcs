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


PROCEDURE_WORDS = (
    "all stations this is for stand by standby message messages follows follow of characters character "
    "I say again out more to complete this completes test count testing with do not answer time "
    "authentication Skyking Sky king disregard break Mainsail Andrews Offutt"
).split()
DIGIT_WORDS_SPOKEN = "zero one two three four five six seven eight nine niner".split()


def allowed_words() -> list[str]:
    return [*NATO_WORDS, "Xray", "Alfa", "Juliett", *DIGIT_WORDS_SPOKEN, *PROCEDURE_WORDS]


def constrained_suppress_tokens(tokenizer, eot: int) -> list[int]:
    """Token ids (below ``eot``) to suppress so the decoder can only spell
    allowed words, digits and basic punctuation. ``tokenizer`` is a HF
    ``tokenizers.Tokenizer`` (faster-whisper's ``model.hf_tokenizer``)."""
    allowed: set[int] = set()
    forms = []
    for w in allowed_words():
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            forms += [v, " " + v]
    forms += [str(d) for d in range(10)] + [" " + str(d) for d in range(10)]
    forms += [",", ".", " ,", "-", "?", " -", "'"]
    for f in forms:
        allowed.update(tokenizer.encode(f, add_special_tokens=False).ids)
    return [-1] + [i for i in range(eot) if i not in allowed]


def build_hotwords(day_callsigns: Sequence[str] = ()) -> str:
    return " ".join([*day_callsigns, "Skyking", "Mainsail", *NATO_WORDS])


PREPROCESS = {
    # name -> ffmpeg -af chain (always followed by mono 16 kHz resample)
    "none": None,
    "bp": "highpass=f=300,lowpass=f=3000",
    "bp_norm": "highpass=f=300,lowpass=f=3000,dynaudnorm=f=250:g=15",
    "bp_denoise_norm": "highpass=f=300,lowpass=f=3000,afftdn=nf=-25,dynaudnorm=f=250:g=15",
    "wide_norm": "highpass=f=150,lowpass=f=3400,dynaudnorm=f=250:g=15",
    "bp_loudnorm": "highpass=f=300,lowpass=f=3000,loudnorm=I=-20:TP=-2",
}


@dataclass
class AsrConfig:
    model: str = "small.en"  # CPU default; "large-v3" / "distil-large-v3" on GPU
    device: str = "cpu"
    compute_type: str = "int8"  # "float16" on GPU
    cpu_threads: int = 8
    beam_size: int = 5
    # Silero VAD off by default: on weak HF speech it drops whole character
    # groups (eval: gold body CER 0.82 -> 0.55 with it off; see docs/decoding.md).
    vad_filter: bool = False
    vad_parameters: dict = field(
        default_factory=lambda: {
            "threshold": 0.35,  # HF speech under noise scores low; 0.5 drops weak chars
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 1500,  # operator pauses between characters/groups
            "speech_pad_ms": 400,
        }
    )
    # On since the eval (silver 0.60 -> 0.42 with VAD off); the loops it causes on long reads are removed
    # by tokens.drop_repetition_loops before parsing.
    condition_on_previous_text: bool = True
    temperature: tuple = (0.0, 0.2, 0.4)
    compression_ratio_threshold: Optional[float] = 2.4
    log_prob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    hallucination_silence_threshold: Optional[float] = 2.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    prompt: Optional[str] = "eam"  # "eam" | "phonetic" | None | literal text
    hotwords: bool = True
    preprocess: str = "bp_norm"
    # Chunked mode: run silero VAD ourselves, merge speech into windows of at
    # most chunk_s seconds, and decode each window independently. None = let
    # faster-whisper window the whole file (30 s windows).
    chunk_s: Optional[float] = None
    chunk_gap_s: float = 0.6  # merge speech regions closer than this
    chunk_pad_s: float = 0.3
    chunk_prompt: bool = False  # pass the prompt to every chunk
    # Closed-vocabulary decoding: suppress every text token that cannot spell
    # a NATO word, a digit or a procedure word (see allowed_words()).
    vocab_constrain: bool = False

    def key(self) -> str:
        """Stable hash of everything that changes the ASR output (cache key)."""
        import hashlib
        import json
        from dataclasses import asdict

        d = asdict(self)
        d.pop("cpu_threads")
        if d.get("chunk_s") is None:  # keep cache keys of pre-chunking configs stable
            for k in ("chunk_s", "chunk_gap_s", "chunk_pad_s", "chunk_prompt"):
                d.pop(k, None)
        if not d.get("vocab_constrain"):
            d.pop("vocab_constrain", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


PHONETIC_PROMPT = (
    "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November "
    "Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu "
    "Zero One Two Three Four Five Six Seven Eight Niner. Stand by. Message follows. I say again."
)


def resolve_prompt(cfg: AsrConfig, day_callsigns: Sequence[str] = ()) -> Optional[str]:
    if cfg.prompt is None:
        return None
    if cfg.prompt == "eam":
        return build_prompt(day_callsigns)
    if cfg.prompt == "phonetic":
        return PHONETIC_PROMPT
    return cfg.prompt


def preprocess(src: Path, dst: Path, cfg: AsrConfig) -> Path:
    """Filter + resample to 16 kHz mono PCM with ffmpeg (Opus/WAV/FLAC in)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    chain = PREPROCESS[cfg.preprocess]
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src)]
    if chain:
        cmd += ["-af", chain]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
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
                self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.cpu_threads,
            )
        return self._model

    def _suppress(self) -> Optional[list[int]]:
        if not self.cfg.vocab_constrain:
            return None
        if getattr(self, "_suppress_ids", None) is None:
            m = self._load()
            eot = m.hf_tokenizer.token_to_id("<|endoftext|>")
            self._suppress_ids = constrained_suppress_tokens(m.hf_tokenizer, eot)
        return self._suppress_ids

    def _decode_kwargs(self, day_callsigns, prompt: bool = True) -> dict:
        c = self.cfg
        extra = {"suppress_tokens": self._suppress()} if c.vocab_constrain else {}
        return dict(
            **extra,
            language="en",
            beam_size=c.beam_size,
            word_timestamps=True,
            initial_prompt=resolve_prompt(c, day_callsigns) if prompt else None,
            hotwords=build_hotwords(day_callsigns) if c.hotwords else None,
            compression_ratio_threshold=c.compression_ratio_threshold,
            log_prob_threshold=c.log_prob_threshold,
            repetition_penalty=c.repetition_penalty,
            no_repeat_ngram_size=c.no_repeat_ngram_size,
            condition_on_previous_text=c.condition_on_previous_text,
            temperature=list(c.temperature),
            no_speech_threshold=c.no_speech_threshold,
            hallucination_silence_threshold=c.hallucination_silence_threshold,
        )

    def transcribe(self, audio: Path, day_callsigns: Sequence[str] = (), offset_s: float = 0.0) -> dict:
        c = self.cfg
        with tempfile.TemporaryDirectory() as td:
            pcm = _read_wav16k(preprocess(Path(audio), Path(td) / "in.wav", c))
        duration = len(pcm) / 16000
        model = self._load()
        segs_out = []  # (offset, segment)
        if c.chunk_s:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            speech = get_speech_timestamps(pcm, VadOptions(**c.vad_parameters)) if c.vad_filter else []
            regions = [(t["start"] / 16000, t["end"] / 16000) for t in speech] or [(0.0, duration)]
            for start, end in merge_regions(regions, c.chunk_s, c.chunk_gap_s, c.chunk_pad_s, duration):
                piece = pcm[int(start * 16000) : int(end * 16000)]
                segs, _ = model.transcribe(
                    piece, vad_filter=False, **self._decode_kwargs(day_callsigns, prompt=c.chunk_prompt)
                )
                segs_out += [(start, s) for s in segs]
        else:
            segs, _ = model.transcribe(
                pcm,
                vad_filter=c.vad_filter,
                vad_parameters=c.vad_parameters,
                **self._decode_kwargs(day_callsigns),
            )
            segs_out = [(0.0, s) for s in segs]
        words = [
            {
                "word": w.word.strip(),
                "probability": round(w.probability, 4),
                "start": round(w.start + off + offset_s, 2),
                "end": round(w.end + off + offset_s, 2),
            }
            for off, s in segs_out
            for w in (s.words or [])
        ]
        return {
            "model": c.model,
            "compute_type": c.compute_type,
            "duration_s": duration,
            "text": " ".join(s.text.strip() for _, s in segs_out),
            "words": words,
            "segments": [
                {
                    "start": s.start + off,
                    "end": s.end + off,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "text": s.text,
                }
                for off, s in segs_out
            ],
        }


def merge_regions(
    regions: Sequence[tuple[float, float]], max_len: float, gap: float, pad: float, duration: float
) -> list[tuple[float, float]]:
    """Merge VAD speech regions (seconds) into decode windows: join regions
    closer than ``gap``, never let a window exceed ``max_len`` (long regions
    are split evenly), pad each window by ``pad`` within [0, duration]."""
    merged: list[list[float]] = []
    for s, e in sorted(regions):
        if merged and s - merged[-1][1] <= gap and e - merged[-1][0] <= max_len:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[tuple[float, float]] = []
    for s, e in merged:
        n = max(1, int(-(-(e - s) // max_len)))
        step = (e - s) / n
        for i in range(n):
            a, b = s + i * step, s + (i + 1) * step
            out.append((round(max(0.0, a - pad), 3), round(min(duration, b + pad), 3)))
    return out
