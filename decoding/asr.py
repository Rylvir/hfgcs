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
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

from .coverage import (
    active_fraction,
    collapse_repeats,
    frame_rms,
    segment_is_loop,
    window_bounds,
    word_coverage,
    word_gaps,
)

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
    # float32 on CPU: CTranslate2's int8 path is not deterministic on our CPUs (the
    # encoder output changes from call to call, even single-threaded), so int8
    # decodes of one clip differed run to run and its 30 s skip decisions cascaded
    # into dropped minutes (docs/decoding.md "Reproducibility"). float32 is
    # bit-identical across runs at the same speed. "float16" on GPU.
    compute_type: str = "float32"
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
    # ---- robustness for long / 24/7 recordings (non-chunked mode; see coverage.py)
    seed: int = 0  # ctranslate2.set_random_seed before every decode (temperature fallback samples)
    # Decode audio longer than 1.33 x window_s in independent windows of about
    # window_s, cut at the quietest point within window_search_s of each nominal
    # cut, so one derailed Whisper step can cost at most one window. None = never.
    window_s: Optional[float] = 240.0
    window_search_s: float = 20.0
    # Wall-time cap per window = max_rtf x window duration; a window that runs over,
    # or loops (> max_words_per_s, or one bigram over half its words), is decoded
    # again with ``fallback`` overrides (VAD on, no conditioning, no sampling).
    max_rtf: Optional[float] = 2.0
    max_words_per_s: float = 3.0
    fallback: dict = field(
        default_factory=lambda: {
            "vad_filter": True,
            "condition_on_previous_text": False,
            "temperature": (0.0,),
        }
    )
    # Coverage guard: a word-free stretch longer than gap_repair_s that is not
    # digital silence is decoded again on its own (fresh context, same settings);
    # words found inside it are merged in. None = off.
    gap_repair_s: Optional[float] = 12.0

    def key(self) -> str:
        """Hash of every field that can change the ASR output (cache key and the
        ``asr=`` part of ``decoder_version``). Only ``cpu_threads`` is left out:
        float32 output does not depend on it (bit-identical at 1/2/4/8 threads)."""
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


def _same_model(a: AsrConfig, b: AsrConfig) -> bool:
    fields = ("model", "device", "compute_type", "cpu_threads")
    return all(getattr(a, f) == getattr(b, f) for f in fields)


# A single segment counts as a loop above SEGMENT_WPS_FACTOR x max_words_per_s (6 words/s by
# default): fast test counts ("15, 1, 2, 3, 2, 3, 1") run ~4 words/s by Whisper's timestamps.
SEGMENT_WPS_FACTOR = 2.0
PRIMARY_SHARE = 0.6  # of max_rtf: the first decode of a window may take 0.6 x 2 = 1.2 x its duration


class Transcriber:
    def __init__(self, cfg: Optional[AsrConfig] = None, share_with: Optional["Transcriber"] = None):
        """``share_with``: reuse another Transcriber's loaded WhisperModel when
        model, device and compute type match (second-pass decodes then cost no
        extra RAM)."""
        self.cfg = cfg or AsrConfig()
        self._model = None
        self._share = share_with

    def _load(self):
        o = self._share
        if self._model is None and o is not None and _same_model(o.cfg, self.cfg):
            self._model = o._load()
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

    def _decode_kwargs(self, day_callsigns, prompt: bool = True, cfg: Optional[AsrConfig] = None) -> dict:
        c = cfg or self.cfg
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
        return self.transcribe_pcm(pcm, day_callsigns, offset_s)

    def transcribe_pcm(self, pcm, day_callsigns: Sequence[str] = (), offset_s: float = 0.0) -> dict:
        """Decode 16 kHz mono float audio. Returns faster-whisper style words
        (``word, probability, start, end``), segments with ``word_span`` and,
        outside chunked mode, a ``guard`` report (windows, fallbacks, repairs,
        coverage)."""
        c = self.cfg
        duration = len(pcm) / 16000
        model = self._load()
        guard = None
        if c.chunk_s:
            segs_out = self._chunked(model, pcm, duration, day_callsigns)
        else:
            segs_out, guard = self._windowed(model, pcm, duration, day_callsigns)
        words, spans = [], []
        for s in segs_out:
            first = len(words)
            words += [
                {**w, "start": round(w["start"] + offset_s, 2), "end": round(w["end"] + offset_s, 2)}
                for w in s["words"]
            ]
            # Inclusive index range of this segment's words in ``words``.
            spans.append([first, len(words) - 1] if len(words) > first else None)
        out = {
            "model": c.model,
            "compute_type": c.compute_type,
            "duration_s": duration,
            "text": " ".join(s["text"].strip() for s in segs_out),
            "words": words,
            "segments": [
                {k: s[k] for k in ("start", "end", "avg_logprob", "no_speech_prob", "text")}
                | {"word_span": span}
                for s, span in zip(segs_out, spans, strict=True)
            ],
        }
        if guard is not None:
            guard["coverage"] = round(word_coverage(words, duration), 3)
            out["guard"] = guard
        return out

    # ---------------------------------------------------------------- decoding paths
    @staticmethod
    def _segment_dict(s, off: float) -> dict:
        return {
            "start": round(s.start + off, 2),
            "end": round(s.end + off, 2),
            "avg_logprob": s.avg_logprob,
            "no_speech_prob": s.no_speech_prob,
            "text": s.text,
            "words": [
                {
                    "word": w.word.strip(),
                    "probability": round(w.probability, 4),
                    "start": round(w.start + off, 2),
                    "end": round(w.end + off, 2),
                }
                for w in (s.words or [])
            ],
        }

    def _chunked(self, model, pcm, duration, day_callsigns) -> list[dict]:
        """Legacy experiment (``chunk_s``): our own VAD windows, decoded independently."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        c = self.cfg
        speech = get_speech_timestamps(pcm, VadOptions(**c.vad_parameters)) if c.vad_filter else []
        regions = [(t["start"] / 16000, t["end"] / 16000) for t in speech] or [(0.0, duration)]
        out = []
        for start, end in merge_regions(regions, c.chunk_s, c.chunk_gap_s, c.chunk_pad_s, duration):
            piece = pcm[int(start * 16000) : int(end * 16000)]
            segs, _ = model.transcribe(
                piece, vad_filter=False, **self._decode_kwargs(day_callsigns, prompt=c.chunk_prompt)
            )
            out += [self._segment_dict(s, start) for s in segs]
        return out

    def _decode_span(self, model, pcm, start: float, end: float, cfg: AsrConfig, day_callsigns) -> tuple:
        """One independent decode of [start, end) under the guards. Returns
        (segments, status, resume_at, rejected): status is ok, loop (a looping
        segment, not kept but returned as ``rejected``, started at resume_at)
        or timeout (wall time passed the cap; output after resume_at is missing)."""
        try:
            import ctranslate2

            ctranslate2.set_random_seed(cfg.seed)
        except ImportError:  # fake models in tests
            pass
        piece = pcm[int(start * 16000) : int(end * 16000)]
        dur = max(0.0, end - start)
        deadline = None
        if cfg.max_rtf:  # first decodes get PRIMARY_SHARE of the cap, the whole-decode budget bounds the rest
            deadline = time.monotonic() + PRIMARY_SHARE * cfg.max_rtf * max(dur, 5.0)
        budget = getattr(self, "_budget_end", None)
        if budget is not None:
            deadline = budget if deadline is None else min(deadline, budget)
        segs, _ = model.transcribe(
            piece,
            vad_filter=cfg.vad_filter,
            vad_parameters=cfg.vad_parameters,
            **self._decode_kwargs(day_callsigns, cfg=cfg),
        )
        out, n_words = [], 0
        for s in segs:  # a generator: breaking out stops decoding
            d = self._segment_dict(s, start)
            n_words += len(d["words"])
            seg_wps = SEGMENT_WPS_FACTOR * cfg.max_words_per_s
            if segment_is_loop(d, seg_wps):
                kept = collapse_repeats(d["words"])  # salvage the words before/around the loop
                d = {**d, "words": kept, "text": " ".join(w["word"] for w in kept), "collapsed": True}
                if segment_is_loop(d, seg_wps):
                    return out, "loop", max(start, min(d["start"], end)), d
                n_words = sum(len(x["words"]) for x in out) + len(kept)
            if n_words > cfg.max_words_per_s * max(dur, 1.0) + 10:
                return out, "loop", max(start, min(d["start"], end)), d
            out.append(d)
            if deadline is not None and time.monotonic() > deadline:
                return out, "timeout", max(start, min(d["end"], end)), None
        return out, "ok", end, None

    def _guarded_span(self, model, pcm, start, end, day_callsigns, guard, cfg=None) -> list[dict]:
        """Decode [start, end); if it loops or runs over time, keep what came
        before the problem and decode the rest once more with ``cfg.fallback``."""
        cfg = cfg or self.cfg
        segs, status, at, rejected = self._decode_span(model, pcm, start, end, cfg, day_callsigns)
        if status != "ok":
            entry = {
                "start": round(start, 2),
                "end": round(end, 2),
                "reason": status,
                "resume_at": round(at, 2),
            }
            if self._out_of_budget():
                entry["fallback_status"] = "skipped: wall-time budget spent"
            elif cfg.fallback and end - at > 1.0:
                fb = replace(cfg, **cfg.fallback)
                more, status2, _, _ = self._decode_span(model, pcm, at, end, fb, day_callsigns)
                entry["fallback_status"] = status2
                if status2 == "loop" and rejected is not None and not any(x["words"] for x in more):
                    # Both decodes "loop" at the same place: more likely fast real speech
                    # (test counts) than a hallucination, so keep the first pass's segment.
                    more = [rejected]
                    entry["kept_first_pass"] = True
                segs += more
            guard["fallbacks"].append(entry)
        return segs

    def _out_of_budget(self) -> bool:
        b = getattr(self, "_budget_end", None)
        return b is not None and time.monotonic() >= b

    def _windowed(self, model, pcm, duration, day_callsigns) -> tuple[list[dict], dict]:
        """Windows, then loop/timeout fallbacks and coverage repairs, all inside
        one wall-time budget of ``max_rtf`` x duration (+ the segment being
        decoded when it runs out). First decodes of each window may use
        PRIMARY_SHARE of their own share so a slow window leaves time for its fallback."""
        c = self.cfg
        t0 = time.monotonic()
        self._budget_end = t0 + c.max_rtf * max(duration, 5.0) if c.max_rtf else None
        try:
            segs, guard = self._windowed_inner(model, pcm, duration, day_callsigns)
        finally:
            self._budget_end = None
        guard["wall_s"] = round(time.monotonic() - t0, 1)
        return segs, guard

    def _windowed_inner(self, model, pcm, duration, day_callsigns) -> tuple[list[dict], dict]:
        c = self.cfg
        rms = frame_rms(pcm)
        bounds = window_bounds(rms, duration, c.window_s, c.window_search_s)
        guard: dict = {"windows": [list(b) for b in bounds], "fallbacks": [], "repairs": []}
        segs: list[dict] = []
        for a, b in bounds:
            segs += self._guarded_span(model, pcm, a, b, day_callsigns, guard)
        if c.gap_repair_s:
            words = [w for s in segs for w in s["words"]]
            repair_cfg = replace(c, condition_on_previous_text=False)
            for g0, g1 in word_gaps(words, duration, c.gap_repair_s):
                if active_fraction(rms, g0, g1) < 0.5:
                    continue
                if any(
                    f["resume_at"] <= g0 + 1.0 and f["end"] >= g1 - 1.0 and "fallback_status" in f
                    for f in guard["fallbacks"]
                ):
                    continue  # already decoded twice (first pass + fallback)
                if self._out_of_budget():
                    guard["repairs"].append(
                        {"start": g0, "end": g1, "words": 0, "skipped": "wall-time budget spent"}
                    )
                    continue
                found = self._guarded_span(
                    model, pcm, max(0.0, g0 - 1.0), min(duration, g1 + 1.0), day_callsigns, guard, repair_cfg
                )
                new = []
                for s in found:  # keep only words inside the gap (no duplicates at its edges)
                    ws = [w for w in s["words"] if g0 <= w["start"] and w["end"] <= g1 + 0.05]
                    if ws:
                        new.append({**s, "words": ws, "text": " ".join(w["word"] for w in ws)})
                n = sum(len(s["words"]) for s in new)
                guard["repairs"].append({"start": g0, "end": g1, "words": n})
                segs += new
            segs.sort(key=lambda s: s["words"][0]["start"] if s["words"] else s["start"])
        return segs, guard


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
