"""Preamble fusion: vote the 6-character preamble over every hearing.

An EAM's first six characters are heard up to five times: three times before
"message follows" (each followed by "stand by") and again at the start of
read 1 and read 2. The parser votes only the three pre-reads; this pass adds
the two read starts, then (optionally) writes the result back over the start
of the body. Pure post-processing on a parsed record.
"""

from __future__ import annotations

from typing import Optional

from .metrics import edit_distance
from .vote import medoid, vote_on_pivot

PREAMBLE_LENGTH = 6
MAX_PATCH_DISTANCE = 2  # only overwrite a body start that already looks like the preamble


def preamble_candidates(rec: dict) -> tuple[list[str], list[str]]:
    """(full, partial) hearings of the preamble. A pre-read longer than six is
    cut to its last six characters (junk such as a misheard "for" lands in
    front of the preamble, right before "stand by"); read starts give their
    first six. Shorter pre-reads are partial and only vote."""
    full, partial = [], []
    for r in rec.get("preamble_reads") or []:
        (full if len(r) >= PREAMBLE_LENGTH else partial).append(r[-PREAMBLE_LENGTH:] if r else r)
    if "mid_message_start" not in rec.get("flags", ()):
        for key in ("read1", "read2"):
            start = (rec.get(key) or "")[:PREAMBLE_LENGTH]
            if len(start) == PREAMBLE_LENGTH:
                full.append(start)
    return full, [p for p in partial if p]


def _patch_start(body: str, fused: str) -> Optional[str]:
    """Replace the body prefix that aligns best with ``fused`` (prefix length
    4-8, ties prefer 6) when it is within MAX_PATCH_DISTANCE edits."""
    best = None
    for k in range(PREAMBLE_LENGTH - 2, PREAMBLE_LENGTH + 3):
        if k > len(body):
            break
        d = edit_distance(body[:k], fused)
        key = (d, abs(k - PREAMBLE_LENGTH))
        if best is None or key < best[0]:
            best = (key, k)
    if best is None or best[0][0] == 0 or best[0][0] > MAX_PATCH_DISTANCE:
        return None
    return fused + body[best[1] :]


def fuse_preamble(rec: Optional[dict], patch_body: bool = True) -> Optional[dict]:
    """Return a copy of an EAM record with ``preamble`` voted over all its
    hearings (flag ``preamble_fused`` when it changed) and, if ``patch_body``,
    the aligned start of the body replaced by it when within
    MAX_PATCH_DISTANCE edits (flag ``body_start_from_preamble``)."""
    if not rec or rec.get("type") != "eam":
        return rec
    full, partial = preamble_candidates(rec)
    if not full or len(full) + len(partial) < 2:
        return rec
    k = medoid(full)
    fused = vote_on_pivot(full[k], full[:k] + full[k + 1 :] + partial)
    if len(fused) != PREAMBLE_LENGTH:
        return rec
    out = dict(rec, flags=list(rec.get("flags", [])))
    if fused != rec.get("preamble"):
        out["preamble_parser"] = rec.get("preamble")
        out["preamble"] = fused
        out["flags"].append("preamble_fused")
    body = rec.get("body") or ""
    if patch_body and body:
        patched = _patch_start(body, fused)
        if patched is not None:
            out["body"] = patched
            out["body_length"] = len(patched)
            out["flags"].append("body_start_from_preamble")
    return out
