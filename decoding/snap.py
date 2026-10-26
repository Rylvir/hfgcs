"""Phonetic-word snapping: map near-miss ASR words onto the closed HFGCS
vocabulary (NATO alphabet, digit words) before parsing.

Whisper, unconstrained, writes HF-garbled phonetics as nearby English or
invented words ("Dango", "Joyette", "Cubex", "Pabra"). Inside an EAM body
every word *must* be a character, so a conservative fuzzy match recovers
some of them. Procedure words, callsign words and common English words are
never snapped. The match uses a sound key (consonant classes, vowels
dropped) plus plain edit distance; both thresholds are tuned on silver,
not gold, to avoid fitting the 8 gold EAMs.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .metrics import edit_distance
from .tokens import DIGIT_WORDS, PHONETIC

# Words that must never become characters.
PROTECTED = {
    # procedure
    "all", "stations", "this", "is", "for", "stand", "standby", "by", "message", "messages", "follows",
    "follow", "followed", "say", "again", "out", "more", "to", "skyking", "sky", "king", "time",
    "authentication", "disregard", "with", "test", "count", "testing", "complete", "completes", "do",
    "not", "answer", "of", "characters", "character", "break", "mainsail",
    # common English the model falls back to
    "the", "a", "an", "and", "now", "at", "off", "on", "in", "it", "just", "i", "im", "you", "we",
    "that", "what", "right", "okay", "ok", "yes", "no", "so", "be", "are", "was", "go",
    "united", "states", "new", "york", "america", "north", "global", "thank", "thanks",
    "ray", "brake", "your", "four", "fore", "too", "two",
}  # fmt: skip

VOCAB: dict[str, str] = {**PHONETIC, **{w: c for w, c in DIGIT_WORDS.items() if len(w) > 2}}

_CLASSES = [("bp", "B"), ("dt", "D"), ("gkcq", "K"), ("fvw", "F"), ("szx", "S"), ("mn", "M"), ("lr", "L")]


@lru_cache(maxsize=4096)
def sound_key(word: str) -> str:
    """Consonant-class skeleton: 'tango' -> 'DMK', 'dango' -> 'DMK'."""
    w = re.sub(r"[^a-z]", "", word.lower()).replace("ph", "f").replace("ck", "k").replace("qu", "k")
    w = w.replace("x", "ks").replace("j", "y")
    out = []
    for ch in w:
        if ch in "aeiouyh":
            continue
        cls = next((c for letters, c in _CLASSES if ch in letters), ch.upper())
        if not out or out[-1] != cls:
            out.append(cls)
    return "".join(out)


@lru_cache(maxsize=4096)
def snap_word(word: str, max_ratio: float = 0.34, key_ratio: float = 0.45) -> str | None:
    """Return the vocabulary word ``word`` most plausibly is, or None."""
    n = re.sub(r"[^a-z]", "", word.lower())
    if len(n) < 3 or n in PROTECTED or n in VOCAB:
        return None
    best, best_score = None, 9.0
    k = sound_key(n)
    for v in VOCAB:
        d = edit_distance(n, v) / max(len(n), len(v))
        kd = edit_distance(k, sound_key(v)) / max(len(k), len(sound_key(v)), 1)
        ok = d <= max_ratio or (kd == 0 and d <= key_ratio) or (kd <= 0.34 and d <= 0.4)
        score = d + kd
        if ok and score < best_score:
            best, best_score = v, score
    return best


def snap_words(words: list[dict], **kw) -> list[dict]:
    """Rewrite faster-whisper word dicts; snapped words keep timing and get
    their probability halved (they are guesses) plus ``snapped_from``."""
    out = []
    for w in words:
        text = str(w.get("word", ""))
        target = snap_word(text, **kw) if text else None
        if target:
            w = {
                **w,
                "word": target,
                "probability": float(w.get("probability", 1.0)) * 0.5,
                "snapped_from": text,
            }
        out.append(w)
    return out
