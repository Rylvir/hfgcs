"""Coverage and loop guards for long-recording ASR (pure Python + numpy-free).

Whisper decodes a file in 30 s steps and, on weak HF speech, its skip
heuristics (no-speech, hallucination-silence) can jump over whole steps or
straight to the end of the file; a bad step can also loop. These helpers let
``asr.Transcriber`` cut long audio into independent windows at quiet points,
spot word-free gaps that still hold signal, and judge a window's output.
``word_coverage`` follows scripts/coverage_report.py on branch decoding-bench.
"""

from __future__ import annotations

from typing import Optional, Sequence

FRAME_S = 0.1


def word_coverage(words: Sequence[dict], duration: float) -> float:
    """Seconds covered by the union of word [start, end] spans / duration."""
    spans = sorted((float(w["start"]), float(w["end"])) for w in words if w.get("end") is not None)
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / duration if duration else 0.0


def frame_rms(samples, rate: int = 16000, frame_s: float = FRAME_S) -> list[float]:
    """RMS per ``frame_s`` frame (``samples``: a sequence or numpy array of floats in [-1, 1])."""
    n = max(1, int(rate * frame_s))
    out = []
    for i in range(0, len(samples), n):
        chunk = samples[i : i + n]
        if len(chunk) == 0:
            break
        try:  # numpy fast path
            out.append(float((chunk.astype("float64") ** 2).mean() ** 0.5))
        except AttributeError:
            out.append((sum(x * x for x in chunk) / len(chunk)) ** 0.5)
    return out


SILENT_RMS = 1e-4  # about -80 dBFS: digital silence / dead air, never worth decoding


def window_bounds(
    rms: Sequence[float],
    duration: float,
    window_s: Optional[float],
    search_s: float,
    frame_s: float = FRAME_S,
) -> list[tuple[float, float]]:
    """Cut ``[0, duration]`` into windows of about ``window_s`` seconds. Each
    cut is placed at the quietest 0.5 s stretch within ``search_s`` of its
    nominal position, so cuts fall in pauses rather than inside words. Audio
    no longer than ``1.33 * window_s`` stays one window."""
    if not window_s or duration <= 1.33 * window_s:
        return [(0.0, round(duration, 3))]
    n_win = max(1, round(duration / window_s))
    step = duration / n_win
    smooth = 5  # frames (0.5 s)
    cuts = []
    for k in range(1, n_win):
        t = k * step
        lo, hi = int(max(0.0, t - search_s) / frame_s), int(min(duration, t + search_s) / frame_s)
        best, best_e = int(t / frame_s), None
        for i in range(lo, max(lo + 1, hi - smooth)):
            e = sum(rms[i : i + smooth])
            if best_e is None or e < best_e:
                best, best_e = i + smooth // 2, e
        cuts.append(round(best * frame_s, 3))
    edges = [0.0, *cuts, round(duration, 3)]
    return [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b - a > 0.05]


def word_gaps(words: Sequence[dict], duration: float, min_gap_s: float) -> list[tuple[float, float]]:
    """Word-free stretches longer than ``min_gap_s`` (including the head and tail)."""
    gaps, prev = [], 0.0
    for w in sorted(words, key=lambda w: float(w["start"])):
        if float(w["start"]) - prev > min_gap_s:
            gaps.append((round(prev, 2), round(float(w["start"]), 2)))
        prev = max(prev, float(w["end"]))
    if duration - prev > min_gap_s:
        gaps.append((round(prev, 2), round(duration, 2)))
    return gaps


def active_fraction(rms: Sequence[float], start: float, end: float, frame_s: float = FRAME_S) -> float:
    """Share of frames in [start, end) that are not digital silence."""
    seg = rms[int(start / frame_s) : max(int(start / frame_s) + 1, int(end / frame_s))]
    return sum(r > SILENT_RMS for r in seg) / len(seg) if seg else 0.0


def segment_is_loop(seg: dict, max_words_per_s: float) -> bool:
    """A Whisper segment that is a repetition loop: more words than any HFGCS
    voice produces in its time span (phonetic reads run about 1-1.5 words/s),
    or, with 12+ words, one 1-3 word unit repeated over 60 % of it."""
    words = seg.get("words") or []
    if not words:
        return False
    span = max(float(seg["end"]) - float(seg["start"]), 1.0)
    if len(words) > max_words_per_s * span + 3:
        return True
    if len(words) >= 12:
        norm = [str(w.get("word", "")).strip(" ,.?!-").lower() for w in words]
        for n in (1, 2, 3):
            units = [tuple(norm[i : i + n]) for i in range(0, len(norm) - n + 1, n)]
            if units and max(units.count(u) for u in set(units)) * n >= 0.6 * len(norm):
                return True
    return False


def collapse_repeats(words: Sequence[dict], max_ngram: int = 4, max_repeats: int = 3) -> list[dict]:
    """Cut any 1..max_ngram word unit repeated more than ``max_repeats`` times
    in a row to one occurrence (the dict-word twin of tokens.drop_repetition_loops),
    so a looping segment's real words can be kept."""
    out = list(words)

    def norm(w):
        return str(w.get("word", "")).strip(" ,.?!-").lower()

    for size in range(1, max_ngram + 1):
        i, res = 0, []
        while i < len(out):
            gram = [norm(w) for w in out[i : i + size]]
            j, reps = i + size, 1
            while len(gram) == size and [norm(w) for w in out[j : j + size]] == gram:
                reps += 1
                j += size
            if reps > max_repeats:
                res.extend(out[i : i + size])
                i = j
            else:
                res.append(out[i])
                i += 1
        out = res
    return out
