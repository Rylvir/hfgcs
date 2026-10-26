"""Align two reads of a message and build a per-character consensus.

EAMs are read twice ("... I say again ..."). Treating the reads as two noisy
observations of the same string gives a confidence per character that is
independent of the ASR's own (often over-confident) probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

GAP = None

# Confidence policy (tunable on the eval set).
SINGLE_READ_FACTOR = 0.6  # only one read exists at all
GAP_FACTOR = 0.4  # character present in one read, missing in the other
DISAGREE_FACTOR = 0.5  # reads differ; we keep the more probable one
PREAMBLE_BONUS = 0.5  # weight of an agreeing preamble character


@dataclass
class CharObs:
    char: str
    prob: float = 1.0


@dataclass
class ConsensusChar:
    char: str
    confidence: float
    read1: Optional[str]
    read2: Optional[str]
    agree: bool
    preamble: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def align_reads(
    r1: Sequence[CharObs], r2: Sequence[CharObs]
) -> list[tuple[Optional[CharObs], Optional[CharObs]]]:
    """Levenshtein alignment; returns column pairs with None for gaps."""
    n, m = len(r1), len(r2)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if r1[i - 1].char == r2[j - 1].char else 1
            d[i][j] = min(d[i - 1][j - 1] + sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    cols = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r1[i - 1].char != r2[j - 1].char):
            cols.append((r1[i - 1], r2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            cols.append((r1[i - 1], GAP))
            i -= 1
        else:
            cols.append((GAP, r2[j - 1]))
            j -= 1
    cols.reverse()
    return cols


def consensus(
    r1: Sequence[CharObs],
    r2: Sequence[CharObs],
    preamble: Optional[str] = None,
) -> list[ConsensusChar]:
    """Combine two reads. ``preamble`` (the first six characters, heard three
    times before "message follows") is a third, partial observation used to
    break ties and lift confidence on positions 0-5."""
    if not r1 and not r2:
        return []
    if not r2 or not r1:
        only = r1 or r2
        out = [
            ConsensusChar(
                o.char, o.prob * SINGLE_READ_FACTOR, o.char if r1 else None, o.char if r2 else None, False
            )
            for o in only
        ]
    else:
        out = []
        for a, b in align_reads(r1, r2):
            if a and b and a.char == b.char:
                conf = 1 - (1 - a.prob) * (1 - b.prob)
                out.append(ConsensusChar(a.char, conf, a.char, b.char, True))
            elif a and b:
                best = a if a.prob >= b.prob else b
                out.append(ConsensusChar(best.char, best.prob * DISAGREE_FACTOR, a.char, b.char, False))
            else:
                o = a or b
                out.append(
                    ConsensusChar(
                        o.char, o.prob * GAP_FACTOR, a.char if a else None, b.char if b else None, False
                    )
                )
    if preamble:
        for idx, pc in enumerate(preamble[: len(out)]):
            c = out[idx]
            c.preamble = pc
            if pc == c.char:
                c.confidence = 1 - (1 - c.confidence) * (1 - PREAMBLE_BONUS)
            elif not c.agree and pc in (c.read1, c.read2):
                # Preamble breaks the tie between disagreeing reads.
                c.char = pc
                c.confidence = max(c.confidence, PREAMBLE_BONUS)
    for c in out:
        c.confidence = round(min(max(c.confidence, 0.0), 1.0), 3)
    return out
