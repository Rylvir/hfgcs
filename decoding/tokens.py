"""Transcript tokens and phonetic-alphabet normalisation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Union

# NATO/ICAO alphabet plus spellings ASR commonly produces for HF voice.
PHONETIC = {
    "alpha": "A",
    "alfa": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliet": "J",
    "juliett": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W",
    "whisky": "W",
    "xray": "X",
    "yankee": "Y",
    "zulu": "Z",
}
DIGIT_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "tree": "3",
    "four": "4",
    "fower": "4",
    "five": "5",
    "fife": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "niner": "9",
}
TEENS_TENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

# Two-word forms that ASR splits; joined before matching.
_JOINS = {("sky", "king"): "skyking", ("x", "ray"): "xray", ("stand", "by"): "standby"}


@dataclass
class Token:
    """One ASR word. ``prob`` is the word probability (faster-whisper
    ``word.probability``); times are seconds from the start of the audio.
    ``src`` is the inclusive range ``(i, j)`` of input words (ASR word
    indices) the token came from; ``tokenize`` fills it in."""

    text: str
    prob: float = 1.0
    start: Optional[float] = None
    end: Optional[float] = None
    src: Optional[tuple[int, int]] = None

    @property
    def norm(self) -> str:
        return re.sub(r"[^a-z0-9]", "", self.text.lower())


TokenInput = Union[str, Sequence[Token], Sequence[dict]]


def tokenize(transcript: TokenInput) -> list[Token]:
    """Accept plain text, a list of Token, or faster-whisper style word dicts
    (``{"word", "probability", "start", "end"}``) and return normalised tokens.
    """
    raw: list[Token] = []
    if isinstance(transcript, str):
        words = [w for w in re.split(r"[\s,.;:!?]+", transcript) if w]
        raw = [Token(w, src=(k, k)) for k, w in enumerate(words)]
    else:
        for k, t in enumerate(transcript):
            if isinstance(t, Token):
                raw.append(t if t.src is not None else replace(t, src=(k, k)))
            else:
                raw.append(
                    Token(
                        str(t.get("word", t.get("text", ""))),
                        float(t.get("probability", t.get("prob", 1.0))),
                        t.get("start"),
                        t.get("end"),
                        (k, k),
                    )
                )
    # A single ASR word can hold several ("Alpha,Bravo"); split those, keeping
    # the parent probability and time span.
    split: list[Token] = []
    for t in raw:
        parts = [p for p in re.split(r"[\s,.;:!?]+", t.text) if p]
        for p in parts:
            split.append(Token(p, t.prob, t.start, t.end, t.src))
    out: list[Token] = []
    i = 0
    while i < len(split):
        if i + 1 < len(split):
            key = (split[i].norm, split[i + 1].norm)
            if key in _JOINS:
                a, b = split[i], split[i + 1]
                out.append(Token(_JOINS[key], min(a.prob, b.prob), a.start, b.end, src_span(a, b)))
                i += 2
                continue
        if split[i].norm:
            out.append(split[i])
        i += 1
    return drop_repetition_loops(_join_say_again(out))


def src_span(*toks: Token) -> Optional[tuple[int, int]]:
    """Inclusive input-word range covered by ``toks`` (None if none is known)."""
    srcs = [t.src for t in toks if t.src is not None]
    if not srcs:
        return None
    return (min(s[0] for s in srcs), max(s[1] for s in srcs))


def _join_say_again(toks: list[Token]) -> list[Token]:
    """ "I say again" / "I'd say again" / bare "again" -> one token
    ``sayagain`` (ASR often mangles or drops the "I say")."""
    out: list[Token] = []
    i = 0
    while i < len(toks):
        if toks[i].norm == "again":
            # "I say again", "I'd say again", "Iced it again", bare "again":
            # "again" never occurs inside an EAM body, so it marks read 2.
            first = toks[i]
            while out and out[-1].norm in ("say", "said", "i", "id", "ill", "ive", "ai", "it", "iced"):
                first = out.pop()
            prob = min(first.prob, toks[i].prob)
            out.append(Token("sayagain", prob, first.start, toks[i].end, src_span(first, toks[i])))
            i += 1
            continue
        out.append(toks[i])
        i += 1
    return out


def drop_repetition_loops(tokens: list[Token], max_ngram: int = 4, max_repeats: int = 3) -> list[Token]:
    """Remove Whisper hallucination loops ("New York City, New York City, ...").
    An n-gram (n <= max_ngram) repeated more than ``max_repeats`` times in a
    row is cut to one occurrence. EAM bodies legitimately repeat a character
    only a few times, and the "Mainsail Mainsail" / "stand by" x3 patterns
    stay within the limit."""
    out = list(tokens)
    for size in range(1, max_ngram + 1):
        i, res = 0, []
        while i < len(out):
            gram = [t.norm for t in out[i : i + size]]
            j = i + size
            reps = 1
            while len(gram) == size and [t.norm for t in out[j : j + size]] == gram:
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


def token_chars(tok: Token) -> Optional[str]:
    """Characters a token spells inside a message body, or None if it is not a
    character token. "Alpha" -> "A", "niner" -> "9", "7" -> "7", "4K" -> "4K"."""
    n = tok.norm
    raw = tok.text.strip(" ,.;:!?").lower()
    if re.fullmatch(r"(?:[a-z0-9]-){1,}[a-z0-9]", raw):
        return raw.replace("-", "").upper()  # "K-R-A-B-X" spelled out by ASR
    if n in PHONETIC:
        return PHONETIC[n]
    if n in DIGIT_WORDS:
        return DIGIT_WORDS[n]
    if n.isdigit():
        return n
    if len(n) == 1 and n.isalpha():
        return n.upper()
    if re.fullmatch(r"[a-z0-9]{2,3}", n) and any(c.isdigit() for c in n):
        return n.upper()  # e.g. "4K" written as a single ASR word
    return None


def parse_number(tokens: Iterable[Token]) -> Optional[int]:
    """Spoken number: digit-by-digit ("one two zero"), words ("thirty",
    "one hundred twenty"), or numerals ("30")."""
    norms = [t.norm for t in tokens if t.norm not in ("and",)]
    if not norms:
        return None
    if all(n in DIGIT_WORDS or n.isdigit() for n in norms) and "hundred" not in norms:
        s = "".join(DIGIT_WORDS.get(n, n) for n in norms)
        return int(s) if s else None
    total, cur, seen = 0, 0, False
    for n in norms:
        if n in DIGIT_WORDS:
            cur += int(DIGIT_WORDS[n])
            seen = True
        elif n in TEENS_TENS:
            cur += TEENS_TENS[n]
            seen = True
        elif n.isdigit():
            cur += int(n)
            seen = True
        elif n == "hundred":
            cur = (cur or 1) * 100
            seen = True
        else:
            return None
    return total + cur if seen else None
