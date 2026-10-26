"""Accuracy metrics for the evaluation set (see docs/decoding.md section 3)."""

from __future__ import annotations

from typing import Optional, Sequence


def edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate = edit distance / len(ref). Unreadable characters
    in the reference ('.') are dropped from both strings' comparison length
    only if the hypothesis also has them; keep labels clean instead."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def field_exact_match(ref: dict, hyp: dict, fields: Sequence[str]) -> dict[str, Optional[bool]]:
    """Per-field exact match (case-insensitive, whitespace-collapsed). None when
    the reference does not label the field."""
    out: dict[str, Optional[bool]] = {}
    for f in fields:
        r = ref.get(f)
        if r is None:
            out[f] = None
            continue
        h = hyp.get(f)
        norm = lambda v: " ".join(str(v).upper().split()) if v is not None else None  # noqa: E731
        out[f] = norm(r) == norm(h)
    return out


def detection_pr(ref_ids: set, hyp_ids: set) -> tuple[float, float]:
    """Precision / recall of message detection (ids are matched upstream)."""
    tp = len(ref_ids & hyp_ids)
    p = tp / len(hyp_ids) if hyp_ids else 0.0
    r = tp / len(ref_ids) if ref_ids else 0.0
    return p, r
