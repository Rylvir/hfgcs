"""Character vote across several ASR passes of the same clip (ROVER-style).

Used as an optional second pass: when the first-pass parse looks unreliable
(see ``needs_second_pass``), the clip is decoded with extra configurations and
the resulting bodies are combined here. Pure Python, no ASR imports.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from .align import CharObs, align_reads
from .metrics import edit_distance

# Second-pass trigger, calibrated on gold + silver (docs/decoding.md): fires on
# about a third of EAMs. Structural flags (single_read, length_mismatch, ...)
# were tried as triggers too, but fire on ~85 % of current first passes.
AGREE_THRESHOLD = 0.3
SECOND_PASS_CONFIGS = ("no_vad_snap", "baseline")  # evaluate.CONFIGS names


def medoid(bodies: Sequence[str]) -> int:
    """Index of the body with the smallest total edit distance to the others;
    ties go to the earliest (pass order = first pass first)."""
    return min(range(len(bodies)), key=lambda i: (sum(edit_distance(bodies[i], b) for b in bodies), i))


def vote_bodies(bodies: Sequence[str]) -> str:
    """Combine bodies by aligning each to the medoid and voting per medoid
    column. A column is dropped when most passes have a gap there; ties keep
    the medoid's character. Insertions relative to the medoid are ignored."""
    bodies = [b for b in bodies if b]
    if not bodies:
        return ""
    if len(bodies) < 3:
        return bodies[medoid(bodies)]
    k = medoid(bodies)
    pivot = bodies[k]
    votes = [Counter({c: 1}) for c in pivot]
    gaps = [0] * len(pivot)
    for j, b in enumerate(bodies):
        if j == k:
            continue
        i = 0
        for x, y in align_reads([CharObs(c) for c in pivot], [CharObs(c) for c in b]):
            if x is None:
                continue
            if y is None:
                gaps[i] += 1
            else:
                votes[i][y.char] += 1
            i += 1
    out = []
    for i, c in enumerate(pivot):
        v = votes[i]
        if gaps[i] > sum(v.values()):
            continue
        best, n = v.most_common(1)[0]
        out.append(best if n > v[c] else c)
    return "".join(out)


def needs_second_pass(rec: Optional[dict], threshold: float = AGREE_THRESHOLD) -> bool:
    """True when a first-pass EAM record is unreliable enough to re-decode:
    fewer than ``threshold`` of its consensus columns have read 1 and read 2
    agreeing (a single read counts as 0 agreement). Clips with no EAM body
    (radio checks, noise) never trigger: on the eval they were all true
    negatives, and re-decoding them only costs CPU."""
    if not rec or rec.get("type") != "eam" or not rec.get("body"):
        return False
    return rec.get("confidence", {}).get("reads_agree_ratio", 0.0) < threshold


def apply_vote(rec: Optional[dict], others: Sequence[Optional[dict]]) -> Optional[dict]:
    """Return a copy of the first-pass record with ``body`` replaced by the
    vote over it and the other passes' bodies (flag ``second_pass_vote``).
    Per-character fields (``chars``, ``body_masked``) are left from pass 1."""
    bodies = [(r or {}).get("body") or "" for r in (rec, *others)]
    voted = vote_bodies(bodies)
    if not voted:
        return rec
    base = dict(rec) if rec else dict(next(r for r in others if r and r.get("body")))
    base["body_first_pass"] = (rec or {}).get("body", "")
    base["body"] = voted
    base["body_length"] = len(voted)
    base["flags"] = [*base.get("flags", []), "second_pass_vote"]
    return base
