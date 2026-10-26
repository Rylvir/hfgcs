"""Grammar-driven parser for HFGCS one-way broadcasts (EAM, Skyking, radio check).

Record types: ``eam``, ``skyking``, ``radio_check`` (test count / radio check)
and ``other`` (no recognisable procedure), the same names as labels.jsonl.

Procedure per Priyom, https://priyom.org/military-stations/united-states/hfgcs:

EAM
  1. "All stations, all stations, this is <callsign>, <callsign>."
  2. optional "For <callsign>", repeated twice
  3. first six characters + "Stand by", repeated altogether thrice
  4. "Message follows." | "Message of <number> characters follows."
  5. message in full   6. "I say again."   7. message in full
  8. optional "This complete message of <number> characters."
  9. "This is <callsign>, out." | "This is <callsign>, more to follow, stand by."
Skyking
  1. "Skyking, skyking, do not answer."  2. codeword, twice
  3. "Time" + minutes of the hour, two digits spelled separately
  4. "Authentication" + two letters  5. "I say again," steps 1-4  6. "This is <callsign>, out."

The parser is deliberately tolerant: ASR drops and mangles words, recordings
start late and end early. Every deviation is reported in ``flags`` rather
than raised.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Optional, Sequence

from .align import CharObs, consensus
from .metrics import edit_distance
from .tokens import Token, TokenInput, parse_number, src_span, token_chars, tokenize

SCHEMA_VERSION = "0.2"
DEFAULT_EAM_LENGTH = 30
PREAMBLE_LENGTH = 6
LOW_CONFIDENCE = 0.5

# Words that end a callsign / addressee phrase.
_FOLLOWS = ("follows", "follow", "followed", "followes")

_MARKERS = {
    "followed",
    "all",
    "stations",
    "this",
    "is",
    "for",
    "standby",
    "message",
    "follows",
    "follow",
    "sayagain",
    "i",
    "say",
    "again",
    "out",
    "more",
    "skyking",
    "time",
    "authentication",
    "disregard",
    "with",
    "test",
    "count",
    "complete",
    "do",
    "not",
    "answer",
}


# ---------------------------------------------------------------- helpers
def _norms(toks: Sequence[Token]) -> list[str]:
    return [t.norm for t in toks]


def _find(n: list[str], phrase: Sequence[str], start: int = 0, end: Optional[int] = None) -> int:
    end = len(n) if end is None else end
    k = len(phrase)
    for i in range(max(start, 0), end - k + 1):
        if n[i : i + k] == list(phrase):
            return i
    return -1


def _dedupe_repeat(words: list[str]) -> list[str]:
    """ "Mainsail Mainsail" -> ["Mainsail"]; "Red Dog Red Dog" -> ["Red", "Dog"]."""
    k = len(words)
    if k >= 2 and k % 2 == 0 and [w.lower() for w in words[: k // 2]] == [w.lower() for w in words[k // 2 :]]:
        return words[: k // 2]
    return words


def _phrase(toks: Sequence[Token], start: int, allow_chars: bool = False) -> tuple[list[Token], int]:
    """Collect name words from ``start`` until a procedure marker or a
    character token. Returns (tokens, next index)."""
    out = []
    i = start
    while i < len(toks):
        t = toks[i]
        if t.norm in _MARKERS or (not allow_chars and token_chars(t) is not None):
            break
        out.append(t)
        i += 1
    return out, i


def _name(toks: Sequence[Token]) -> Optional[str]:
    words = _dedupe_repeat([t.text.strip(",.").upper() for t in toks if t.norm])
    return " ".join(words) if words else None


def _chars(toks: Sequence[Token]) -> tuple[list[CharObs], int]:
    """Characters spelled by a token span, plus count of non-character words."""
    obs, junk = [], 0
    for t in toks:
        c = token_chars(t)
        if c is None:
            junk += 1
            continue
        obs.extend(CharObs(ch, t.prob) for ch in c)
    return obs, junk


def _mean_prob(toks: Sequence[Token]) -> float:
    return round(statistics.fmean(t.prob for t in toks), 3) if toks else 0.0


def _time_span(toks: Sequence[Token]) -> dict:
    starts = [t.start for t in toks if t.start is not None]
    ends = [t.end for t in toks if t.end is not None]
    return {"start_sec": min(starts) if starts else None, "end_sec": max(ends) if ends else None}


def _closing(toks: list[Token], n: list[str], start: int) -> dict:
    """Parse "this is <callsign>, out" / "more to follow" / "disregard"."""
    res = {
        "closing_callsign": None,
        "more_to_follow": False,
        "out": False,
        "disregard": "disregard" in n[start:],
    }
    i = _find(n, ["this", "is"], start)
    if i >= 0:
        words, _ = _phrase(toks, i + 2)
        res["closing_callsign"] = _name(words)
    res["more_to_follow"] = _find(n, ["more", "to", "follow"], start) >= 0
    res["out"] = "out" in n[start:]
    return res


def _vote_preamble(reads: list[str]) -> str:
    """Pick the preamble from up to three hearings. Prefer the most common
    whole reading; otherwise vote per position among readings of the most
    plausible length (6 if any, else the commonest), since a stray word
    ("Break", a callsign letter) shifts every later position."""
    whole, count = Counter(reads).most_common(1)[0]
    if count > 1:
        return whole
    lengths = Counter(len(r) for r in reads)
    width = PREAMBLE_LENGTH if PREAMBLE_LENGTH in lengths else lengths.most_common(1)[0][0]
    pool = [r for r in reads if len(r) == width]
    if len(pool) == 1:
        # One clean 6-char read, others garbled: take the clean one
        return pool[0]
    return "".join(Counter(r[i] for r in pool).most_common(1)[0][0] for i in range(width))


def _strip_leading_repeats(obs: list[CharObs]) -> tuple[list[CharObs], list[list[CharObs]]]:
    """If ``obs`` starts with a 5-7 character block heard 2+ times (edit
    distance <= 1 between copies), return (rest starting at the last copy,
    [earlier copies]). The last copy is kept: it is the start of read 1."""
    chars = "".join(o.char for o in obs)
    for w in (6, 5, 7):
        blocks = []
        i = 0
        while i + 2 * w <= len(chars) and edit_distance(chars[i : i + w], chars[i + w : i + 2 * w]) <= 1:
            blocks.append(obs[i : i + w])
            i += w
        if blocks:
            return obs[i:], blocks
    return obs, []


def _self_split(obs: list[CharObs], max_norm_dist: float = 0.5):
    """Split a run holding both reads into (read1, read2) at the point where
    the halves are most alike; None if no split is convincing."""
    chars = "".join(o.char for o in obs)
    n = len(chars)
    best = None
    for k in range(int(n * 0.35), int(n * 0.65) + 1):
        a, b = chars[:k], chars[k:]
        d = edit_distance(a, b) / max(len(a), len(b))
        if best is None or d < best[0]:
            best = (d, k)
    if best and best[0] <= max_norm_dist:
        return obs[: best[1]], obs[best[1] :]
    return None


# ---------------------------------------------------------------- EAM
def parse_eam(transcript: TokenInput) -> dict:
    toks = tokenize(transcript)
    n = _norms(toks)
    flags: list[str] = []
    rec: dict = {"type": "eam", "schema_version": SCHEMA_VERSION}

    # 1. header + callsign
    i_hdr = _find(n, ["all", "stations"])
    if i_hdr < 0:
        flags.append("missing_all_stations")
    i_this = _find(n, ["this", "is"], max(i_hdr, 0))
    callsign_toks: list[Token] = []
    p = max(i_hdr + 4 if i_hdr >= 0 else 0, 0)
    if i_this >= 0 and i_this - max(i_hdr, 0) <= 6:
        callsign_toks, p = _phrase(toks, i_this + 2)
    else:
        flags.append("missing_callsign")
        i_this = -1
    rec["opening_callsign"] = _name(callsign_toks)
    rec["opening_callsign_confidence"] = _mean_prob(callsign_toks)

    # 2. optional addressee: "for X, for X"
    addr_toks: list[Token] = []
    while p < len(toks) and n[p] == "for":
        words, p = _phrase(toks, p + 1)
        addr_toks.extend(words)
    rec["addressee"] = _name(addr_toks)

    # 4. "message [of N characters] follows"
    i_msg, i_follows, declared = -1, -1, None
    j = p
    while True:
        j = _find(n, ["message"], j)
        if j < 0:
            break
        f = next((k for k in range(j + 1, min(j + 8, len(n))) if n[k] in _FOLLOWS), -1)
        if f >= 0:
            i_msg, i_follows = j, f
            if f > j + 1 and n[j + 1] == "of":
                k_end = next((k for k in range(j + 2, f) if n[k] in ("characters", "character")), f)
                declared = parse_number(toks[j + 2 : k_end])
                if declared is None:
                    flags.append("unparsed_length")
            break
        j += 1
    if i_msg < 0:
        # ASR often garbles "message" ("Classic. Follow."): accept a bare
        # follow(s/ed) shortly after the last "stand by".
        sb_all = [k for k in range(p, len(n)) if n[k] == "standby"]
        if sb_all:
            last_sb = sb_all[-1]
            # last stand by *before* any read-2 marker
            again = _find(n, ["sayagain"], p)
            if again >= 0:
                before = [k for k in sb_all if k < again]
                last_sb = before[-1] if before else last_sb
            f = next((k for k in range(last_sb + 1, min(last_sb + 4, len(n))) if n[k] in _FOLLOWS), -1)
            if f >= 0:
                i_msg, i_follows = f, f
                flags.append("message_follows_inferred")
    if i_msg < 0:
        flags.append("missing_message_follows")

    # 3. preamble: (6 chars + "stand by") x3, between header and "message follows"
    pre_end = i_msg if i_msg >= 0 else len(n)
    pre_reads: list[str] = []
    seg_start = p
    sb_positions = [k for k in range(p, pre_end) if n[k] == "standby"]
    for k in sb_positions:
        obs, _ = _chars(toks[seg_start:k])
        if obs:
            pre_reads.append("".join(o.char for o in obs))
        seg_start = k + 1
    if not sb_positions:
        flags.append("missing_preamble")
    preamble = None
    if pre_reads:
        preamble = _vote_preamble(pre_reads)
        if len(preamble) != PREAMBLE_LENGTH:
            flags.append("preamble_length_%d" % len(preamble))
        if len(set(pre_reads)) > 1:
            flags.append("preamble_reads_disagree")
        if len(pre_reads) != 3:
            flags.append("preamble_repeats_%d" % len(pre_reads))
    rec["preamble"] = preamble
    rec["preamble_reads"] = pre_reads

    # 5-7. body reads
    b1_start = i_follows + 1 if i_follows >= 0 else (sb_positions[-1] + 1 if sb_positions else p)
    i_again = _find(n, ["sayagain"], b1_start)
    trailer_markers = [
        _find(n, ["this", "complete", "message"], b1_start),
        _find(n, ["this", "is"], b1_start),
        _find(n, ["disregard"], b1_start),
        _find(n, ["message"], b1_start),  # "this [mangled] message of N characters"
    ]

    def first_after(pos: int) -> int:
        cands = [m for m in trailer_markers if m >= pos]
        return min(cands) if cands else len(n)

    if i_again >= 0:
        b1_end = min(i_again, first_after(b1_start))
        b2_start = i_again + 1
        b2_end = first_after(b2_start)
    else:
        flags.append("single_read")
        b1_end = first_after(b1_start)
        b2_start = b2_end = b1_end
    read1, junk1 = _chars(toks[b1_start:b1_end])
    read2, junk2 = _chars(toks[b2_start:b2_end])
    if not sb_positions and read1:
        # "stand by" lost: the three preamble hearings run into read 1.
        stripped, reps = _strip_leading_repeats(read1)
        if reps:
            pre_reads = ["".join(o.char for o in blk) for blk in reps]
            preamble = _vote_preamble(pre_reads)
            rec["preamble"], rec["preamble_reads"] = preamble, pre_reads
            read1 = stripped
            flags.append("preamble_inferred")
    if not read2 and len(read1) >= 1.6 * (declared or DEFAULT_EAM_LENGTH) and len(read1) >= 20:
        # "I say again" lost: both reads ran together; split where the two
        # halves agree best.
        split = _self_split(read1)
        if split:
            read1, read2 = split
            flags.append("reads_split_inferred")
            if "single_read" in flags:
                flags.remove("single_read")
    if junk1 or junk2:
        flags.append("non_character_words_in_body")
    if not read1 and not read2:
        flags.append("empty_body")

    # 8. trailer length
    trailer_len = None
    i_cm = _find(n, ["this", "complete", "message"], b1_start)
    if i_cm >= 0:
        k = i_cm + 3
        if k < len(n) and n[k] == "of":
            k_end = next(
                (x for x in range(k + 1, min(k + 8, len(n))) if n[x] in ("characters", "character")), -1
            )
            if k_end > 0:
                trailer_len = parse_number(toks[k + 1 : k_end])
    if declared is None and trailer_len is not None:
        declared = trailer_len
    elif declared is not None and trailer_len is not None and declared != trailer_len:
        flags.append("declared_length_conflict")
    length_source = "announced" if declared is not None else "default"
    expected_len = declared if declared is not None else DEFAULT_EAM_LENGTH

    # consensus + confidence
    cons = consensus(read1, read2, preamble)
    body = "".join(c.char for c in cons)
    if read1 and read2 and "".join(o.char for o in read1) != "".join(o.char for o in read2):
        flags.append("reads_disagree")
    if body and len(body) != expected_len:
        flags.append("length_mismatch")
    if preamble and body and not body.startswith(preamble):
        flags.append("preamble_body_mismatch")

    confs = [c.confidence for c in cons]
    rec.update(
        {
            "declared_length": declared,
            "length_source": length_source,
            "expected_length": expected_len,
            "read1": "".join(o.char for o in read1),
            "read2": "".join(o.char for o in read2),
            "body": body,
            "body_masked": "".join(c.char if c.confidence >= LOW_CONFIDENCE else "." for c in cons),
            "body_length": len(body),
            "chars": [c.to_dict() for c in cons],
            "confidence_detail": {
                "mean": round(statistics.fmean(confs), 3) if confs else 0.0,
                "min": min(confs) if confs else 0.0,
                "low_count": sum(c < LOW_CONFIDENCE for c in confs),
                "reads_agree_ratio": round(sum(c.agree for c in cons) / len(cons), 3) if cons else 0.0,
            },
            "timing": {
                "read1": _time_span(toks[b1_start:b1_end]),
                "read2": _time_span(toks[b2_start:b2_end]),
            },
        }
    )
    rec.update(_closing(toks, n, b1_start if i_again < 0 else b2_start))
    opening, closing = rec["opening_callsign"], rec["closing_callsign"]
    if opening and closing and opening != closing:
        flags.append("closing_callsign_mismatch")
    _set_callsigns(rec, [(opening, "opening"), (rec["addressee"], "addressee"), (closing, "closing")])
    # Overall confidence: mean per-character body confidence (0 when no body).
    rec["confidence"] = rec["confidence_detail"]["mean"]
    rec["flags"] = flags
    return rec


# ---------------------------------------------------------------- Skyking
def _skyking_read(toks: list[Token], n: list[str], start: int, end: int) -> dict:
    i = start
    while i < end and n[i] == "skyking":
        i += 1
    dna = _find(n, ["do", "not", "answer"], start, end)
    if dna >= 0:
        i = dna + 3
    i_time = _find(n, ["time"], i, end)
    i_auth = _find(n, ["authentication"], i, end)
    cw_end = min([x for x in (i_time, i_auth) if x >= 0] or [end])
    cw_toks = [t for t in toks[i:cw_end] if t.norm not in _MARKERS]
    time_toks = toks[i_time + 1 : (i_auth if i_auth > i_time else end)] if i_time >= 0 else []
    time_obs, _ = _chars(time_toks)
    time_digits = "".join(o.char for o in time_obs if o.char.isdigit())[:2]
    auth_toks = toks[i_auth + 1 : end] if i_auth >= 0 else []
    auth_obs = [o for o in _chars(auth_toks)[0] if o.char.isalpha()][:2]
    return {
        "codeword": _name(cw_toks),
        "codeword_p": _mean_prob(cw_toks),
        "time": time_digits or None,
        "time_p": _mean_prob(time_toks[:2]),
        "auth": "".join(o.char for o in auth_obs) or None,
        "auth_p": round(statistics.fmean(o.prob for o in auth_obs), 3) if auth_obs else 0.0,
    }


def _field(v1, p1, v2, p2, have2: bool) -> dict:
    if not have2:
        return {
            "value": v1,
            "read1": v1,
            "read2": None,
            "agree": False,
            "confidence": round(p1 * 0.6, 3) if v1 else 0.0,
        }
    if v1 is not None and v1 == v2:
        return {
            "value": v1,
            "read1": v1,
            "read2": v2,
            "agree": True,
            "confidence": round(1 - (1 - p1) * (1 - p2), 3),
        }
    best, bp = (v1, p1) if (v1 is not None and p1 >= p2) or v2 is None else (v2, p2)
    return {
        "value": best,
        "read1": v1,
        "read2": v2,
        "agree": False,
        "confidence": round(bp * 0.5, 3) if best else 0.0,
    }


def parse_skyking(transcript: TokenInput) -> dict:
    toks = tokenize(transcript)
    n = _norms(toks)
    flags: list[str] = []
    i_sk = _find(n, ["skyking"])
    if i_sk < 0:
        flags.append("missing_skyking")
        i_sk = 0
    i_again = _find(n, ["sayagain"], i_sk)
    i_this = _find(n, ["this", "is"], i_sk)
    end_all = i_this if i_this >= 0 else len(n)
    if i_again >= 0 and i_again < end_all:
        r1 = _skyking_read(toks, n, i_sk, i_again)
        r2 = _skyking_read(toks, n, i_again + 1, end_all)
        have2 = True
    else:
        r1 = _skyking_read(toks, n, i_sk, end_all)
        r2 = {"codeword": None, "codeword_p": 0, "time": None, "time_p": 0, "auth": None, "auth_p": 0}
        have2 = False
        flags.append("single_read")
    rec = {"type": "skyking", "schema_version": SCHEMA_VERSION}
    for f in ("codeword", "time", "auth"):
        rec[f if f != "auth" else "authentication"] = _field(r1[f], r1[f + "_p"], r2[f], r2[f + "_p"], have2)
    for f in ("codeword", "time", "authentication"):
        fld = rec[f]
        if fld["value"] is None:
            flags.append("missing_" + f)
        elif have2 and not fld["agree"]:
            flags.append(f + "_reads_disagree")
    t = rec["time"]["value"]
    if t is not None and (len(t) != 2 or int(t) > 59):
        flags.append("invalid_time")
    a = rec["authentication"]["value"]
    if a is not None and len(a) != 2:
        flags.append("invalid_authentication")
    rec.update(_closing(toks, n, end_all))
    _set_callsigns(rec, [(rec["closing_callsign"], "closing")])
    # Overall confidence: the weakest of the three fields (a Skyking is only as
    # usable as its least certain field).
    detail = {f: rec[f]["confidence"] for f in ("codeword", "time", "authentication")}
    rec["confidence_detail"] = detail
    rec["confidence"] = min(detail.values())
    rec["flags"] = flags
    return rec


# ---------------------------------------------------------------- dispatcher
def _word_confidence(rec: dict, toks: Sequence[Token]) -> dict:
    """radio_check / other: overall confidence = mean ASR word probability of
    the broadcast's tokens (snapped words count at their halved probability)."""
    mean = _mean_prob(toks)
    rec["confidence"] = mean
    rec["confidence_detail"] = {"mean_word_prob": mean, "n_words": len(toks)}
    return rec


def _parse_radio_check(toks: list[Token], n: list[str]) -> dict:
    i = _find(n, ["this", "is"])
    words, _ = _phrase(toks, i + 2) if i >= 0 else ([], 0)
    rec = {
        "type": "radio_check",
        "schema_version": SCHEMA_VERSION,
        "flags": [] if i >= 0 else ["missing_callsign"],
    }
    _set_callsigns(rec, [(_name(words), "station")])
    return _word_confidence(rec, toks)


def _other(toks: list[Token], flag: str) -> dict:
    rec = {
        "type": "other",
        "schema_version": SCHEMA_VERSION,
        "text": " ".join(t.text for t in toks),
        "flags": [flag],
    }
    _set_callsigns(rec, [])
    return _word_confidence(rec, toks)


def _is_radio_check(n: list[str]) -> bool:
    """HFGCS radio check = a test count ("... test count, one two three ...")."""
    return _find(n, ["test", "count"]) >= 0


def _segment(n: list[str]) -> list[int]:
    """Start index of each broadcast in a recording. A new broadcast starts at
    "all stations" or "skyking" only after the previous one closed ("out" /
    "more to follow"), so the Skyking repeat after "I say again" stays put.
    A radio check ("this is X [with a] test count") after a closed broadcast
    starts one too."""
    starts: list[int] = []
    closed = True
    i = 0
    while i < len(n):
        is_start = n[i : i + 2] == ["all", "stations"] or n[i] == "skyking"
        if closed and starts and n[i : i + 2] == ["this", "is"]:
            is_start = _find(n, ["test", "count"], i + 2, min(len(n), i + 2 + RADIO_CHECK_LOOKAHEAD)) >= 0
            if is_start:
                starts.append(i)
                closed = False
                i += 2
                continue
        if is_start and closed:
            starts.append(i)
            closed = False
            while i < len(n) and n[i] in ("all", "stations", "skyking"):
                i += 1
            continue
        if n[i] == "out" or n[i : i + 3] == ["more", "to", "follow"]:
            closed = True
        i += 1
    return starts


UNSTRUCTURED_MIN_CHARS = 12
RADIO_CHECK_LOOKAHEAD = 8  # "this is <callsign words> with a test count"


def _set_callsigns(rec: dict, found: Sequence[tuple[Optional[str], str]]) -> None:
    """``callsigns_all`` = every callsign heard, as ``{callsign, role}``; the
    single ``callsign`` is the station identifying itself: the closing
    "this is X" (EAM, Skyking), else the opening one, else the radio-check
    station. The addressee ("for X") is never the ``callsign``."""
    rec["callsigns_all"] = [{"callsign": c, "role": role} for c, role in found if c]
    by_role = {d["role"]: d["callsign"] for d in rec["callsigns_all"]}
    rec["callsign"] = next((by_role[r] for r in ("closing", "opening", "station") if r in by_role), None)


def _longest_char_run(toks: Sequence[Token]) -> int:
    best = cur = 0
    for t in toks:
        c = token_chars(t)
        cur = cur + len(c) if c else 0
        best = max(best, cur)
    return best


MID_START_MIN_READ1 = 3  # fewer read-1 characters before "I say again" = read 2 only


def mid_message_start_kind(rec: dict) -> Optional[str]:
    """For a headless EAM (``partial_start``), say whether the clip began
    partway through the message itself rather than in the header/preamble:
    neither a preamble hearing nor "message follows" survived, yet a body did.

    Returns ``"read2_only"`` (the clip starts at or just before "I say again"),
    ``"in_read1"`` (a tail of read 1, then read 2), ``"body_only"`` (characters
    but no "I say again": inside read 1 or read 2, which one is unknown), or
    None. Missing words can also be ASR drops, so this is a flag, not proof."""
    flags = rec.get("flags", ())
    if rec.get("type") != "eam" or not rec.get("body"):
        return None
    has_follows = "missing_message_follows" not in flags
    has_preamble = bool(rec.get("preamble_reads"))
    if has_follows or has_preamble:
        return None
    if "single_read" in flags:
        return "body_only"
    if "reads_split_inferred" in flags:
        return "in_read1"
    return "read2_only" if len(rec.get("read1") or "") < MID_START_MIN_READ1 else "in_read1"


def _segment_info(toks: Sequence[Token], s: int, e: int) -> dict:
    """Token range [s, e) of one broadcast, its time span, and the inclusive
    range of *input* word indices (ASR words) it was built from."""
    span = src_span(*toks[s:e])
    return {
        "token_start": s,
        "token_end": e,
        **_time_span(toks[s:e]),
        "word_start": span[0] if span else None,
        "word_end": span[1] if span else None,
    }


def parse_transcript(transcript: TokenInput) -> list[dict]:
    """Split a recording's transcript into broadcasts and parse each one.
    Always returns a list; unrecognised audio yields one ``other`` record.
    Every record has a ``segment`` whose ``word_start``/``word_end`` index the
    input words (inclusive), so a writer can map it back to the ASR output."""
    toks = tokenize(transcript)
    n = _norms(toks)
    if not toks:
        rec = _other([], "empty_transcript")
        rec["segment"] = {"token_start": 0, "token_end": 0, "start_sec": None, "end_sec": None}
        rec["segment"].update(word_start=None, word_end=None)
        return [rec]
    starts = _segment(n)
    if not starts or starts[0] > 0:
        # Recording may start mid-message.
        head_end = starts[0] if starts else len(n)
        head = n[:head_end]
        if any(w in head for w in ("standby", "sayagain", "complete", *_FOLLOWS)) or "sayagain" in head:
            starts = [0] + starts
        elif _is_radio_check(head):
            starts = [0] + starts
        elif not starts and _longest_char_run(toks) >= UNSTRUCTURED_MIN_CHARS:
            # No procedure words survived ASR, but a long run of characters
            # did: parse as a headless EAM so the body is not lost.
            starts = [0]
        elif not starts:
            rec = _other(toks, "no_procedure_markers")
            rec["segment"] = _segment_info(toks, 0, len(toks))
            return [rec]
    out = []
    for s, e in zip(starts, starts[1:] + [len(toks)], strict=True):
        seg, segn = toks[s:e], n[s:e]
        if "skyking" in segn[:3]:
            rec = parse_skyking(seg)
        elif _is_radio_check(segn) and _find(segn, ["message"]) < 0:
            rec = _parse_radio_check(seg, segn)
        else:
            rec = parse_eam(seg)
            if s == 0 and n[:2] != ["all", "stations"]:
                rec["flags"].append("partial_start")
                kind = mid_message_start_kind(rec)
                if kind:
                    rec["flags"].append("mid_message_start")
                    rec["mid_message_start"] = kind
        rec["segment"] = _segment_info(toks, s, e)
        out.append(rec)
    return out
