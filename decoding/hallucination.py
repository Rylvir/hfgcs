"""Post-decode filters for Whisper prompt echoes that look like Skykings.

Live 0.3.3 writes its initial_prompt ("Skyking, Skyking, do not answer.") back
over noise, often at conf 0.0 with no codeword/time/authentication. A false
Skyking alert is the worst failure. These helpers are pure and optional: the
experiment harness and the proposed alert path call them; the live pipeline
does not until a PR lands.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

# A real Skyking codeword is one or two spoken words (e.g. "BRAVO FORCE"), not a
# phonetic body. Longer "codewords" are almost always misparsed EAM characters.
MAX_CODEWORD_WORDS = 3

ROUTINE_PATTERNS = (
    re.compile(r"\bstanding by for traffic\b", re.I),
    re.compile(r"\bstand by for traffic\b", re.I),
    re.compile(r"\bget in by the traffic\b", re.I),  # ASR garble of "standing by"
    re.compile(r"\bradio check\b", re.I),
    re.compile(r"\btest count\b", re.I),
    re.compile(r"\bhow copy\b", re.I),
    re.compile(r"\bhow do you read\b", re.I),
)


def codeword_value(rec: dict) -> str:
    cw = rec.get("codeword")
    if isinstance(cw, dict):
        return str(cw.get("value") or "")
    return str(cw or "")


def is_routine_traffic(rec: dict) -> bool:
    """Callsign + standing-by / radio-check phrasing: log, do not alert."""
    text = rec.get("transcript") or rec.get("text") or ""
    return any(p.search(text) for p in ROUTINE_PATTERNS)


def suppress_weak_skyking(rec: dict) -> dict:
    """Downgrade a Skyking that cannot be a real read to ``other``.

    Rules (any one is enough):
    - no codeword value;
    - missing both time and authentication;
    - codeword longer than MAX_CODEWORD_WORDS (misparsed EAM body);
    - confidence below 0.2 with no codeword characters that look like a name.
    """
    if rec.get("type") != "skyking":
        return rec
    flags = set(rec.get("flags") or [])
    cw = codeword_value(rec)
    conf = float(rec.get("confidence") or 0)
    # "no codeword key at all" is not weak: keyword-alert fixtures and older
    # documents omit it. Weak = parser said missing_codeword, empty value,
    # missing time+auth, or a codeword that is clearly an EAM body.
    no_cw = "missing_codeword" in flags or ("codeword" in rec and not cw)
    weak = (
        no_cw
        or {"missing_time", "missing_authentication"} <= flags
        or (cw and len(cw.split()) > MAX_CODEWORD_WORDS)
        or (conf < 0.2 and no_cw)
    )
    if not weak:
        return rec
    out = dict(rec)
    out["type"] = "other"
    out["flags"] = list(flags | {"suppressed_hallucination", "suppressed_skyking"})
    out["suppressed_from"] = "skyking"
    return out


def suppress_empty_eam(rec: dict) -> dict:
    """An EAM with no preamble, no body and no EAM procedure words is noise."""
    if rec.get("type") != "eam":
        return rec
    text = (rec.get("transcript") or rec.get("text") or "").lower()
    has_proc = "all stations" in text or "message" in text
    if rec.get("body") or rec.get("preamble") or has_proc:
        return rec
    out = dict(rec)
    out["type"] = "other"
    out["flags"] = list(set(rec.get("flags") or []) | {"suppressed_hallucination", "suppressed_eam"})
    out["suppressed_from"] = "eam"
    return out


def filter_records(records: Sequence[dict], *, suppress: bool = True) -> list[dict]:
    out = []
    for r in records:
        if suppress:
            r = suppress_weak_skyking(r)
            r = suppress_empty_eam(r)
        out.append(r)
    return out


def filter_asr_segments(
    result: dict,
    *,
    max_no_speech: Optional[float] = 0.75,
    min_logprob: Optional[float] = -0.3,
    min_word_prob: Optional[float] = None,
) -> dict:
    """Drop ASR segments that look like prompt echoes (high no_speech_prob,
    low avg_logprob). Words belonging to dropped segments are removed and
    word_span indices are rebuilt. Pure; does not need silero."""
    if max_no_speech is None and min_logprob is None and min_word_prob is None:
        return result
    segs = list(result.get("segments") or [])
    words = list(result.get("words") or [])
    keep_idx: set[int] = set()
    new_segs = []
    for s in segs:
        nsp = float(s.get("no_speech_prob") or 0)
        lp = float(s.get("avg_logprob") or 0)
        span = s.get("word_span")
        drop = False
        if max_no_speech is not None and min_logprob is not None:
            drop = nsp >= max_no_speech and lp < min_logprob
        if not drop and min_word_prob is not None and span:
            chunk = words[span[0] : span[1] + 1]
            if chunk and max(float(w.get("probability") or 0) for w in chunk) < min_word_prob:
                drop = True
        if drop:
            continue
        if span:
            keep_idx.update(range(span[0], span[1] + 1))
        new_segs.append(s)
    if len(new_segs) == len(segs):
        return result
    # rebuild words + spans
    old_to_new = {}
    new_words = []
    for i, w in enumerate(words):
        if i in keep_idx:
            old_to_new[i] = len(new_words)
            new_words.append(w)
    rebuilt = []
    for s in new_segs:
        span = s.get("word_span")
        if not span:
            rebuilt.append(s)
            continue
        mapped = [old_to_new[i] for i in range(span[0], span[1] + 1) if i in old_to_new]
        ns = dict(s)
        ns["word_span"] = [mapped[0], mapped[-1]] if mapped else None
        rebuilt.append(ns)
    out = dict(result)
    out["segments"] = rebuilt
    out["words"] = new_words
    out["text"] = " ".join(w.get("word", "") for w in new_words)
    out["dropped_by_filter"] = len(segs) - len(new_segs)
    return out


def skyking_alert_ok(
    rec: dict,
    *,
    min_confidence: float = 0.25,
    require_codeword: bool = True,
    require_time_or_auth: bool = True,
    vote_agreed: Optional[bool] = None,
    cross_channel: Optional[bool] = None,
) -> tuple[bool, str]:
    """Proposed alert gate. Returns (alert?, reason).

    Alert only when type is skyking, a short codeword is present, confidence is
    at least ``min_confidence``, and either the vote passes agreed on the
    codeword or another frequency saw the same message (``cross_channel``).
    Single-channel + low confidence -> unconfirmed (log, no alert).
    """
    if rec.get("type") != "skyking":
        return False, "not_skyking"
    if is_routine_traffic(rec):
        return False, "routine_traffic"
    flags = set(rec.get("flags") or [])
    if "suppressed_hallucination" in flags:
        return False, "suppressed"
    cw = codeword_value(rec)
    if require_codeword:
        if cw and len(cw.split()) > MAX_CODEWORD_WORDS:
            return False, "no_or_long_codeword"
        if "missing_codeword" in flags or ("codeword" in rec and not cw):
            return False, "no_or_long_codeword"
        if not cw:
            # No parser codeword field: require non-preamble content in the transcript.
            text = (rec.get("transcript") or "").lower()
            stripped = re.sub(
                r"\b(skyking|sky|king|do|not|answer|,|\.)\b", " ", text
            )
            if len(re.findall(r"[a-z0-9]+", stripped)) < 2:
                return False, "no_or_long_codeword"
    conf = float(rec.get("confidence") or 0)
    if require_time_or_auth and {"missing_time", "missing_authentication"} <= flags:
        return False, "missing_time_and_auth"
    if conf < min_confidence:
        if vote_agreed or cross_channel:
            return True, "boosted_by_agreement"
        return False, "unconfirmed_low_conf"
    if vote_agreed is False and cross_channel is False:
        return False, "disagreed_single_channel"
    if vote_agreed or cross_channel or conf >= min_confidence:
        return True, "ok"
    return False, "unconfirmed"
