"""Per-recording decode document (``<wav>.decode.json`` / one line of
``decodes.jsonl``). Pure Python; the I/O lives in ``decoding.run``.

Contract agreed with Database & Analytics (docs/decoding.md "Decode output",
JSON Schema in docs/decode.schema.json):

* top level: ``schema_version``, ``decoder_version``, ``recording``,
  ``duration_s``, ``decoded_at``, ``asr``, ``broadcasts``;
* every broadcast: ``broadcast_id`` (``<sha256[:12]>@<start_s, 0.1 s>``),
  ``type`` (eam | skyking | radio_check | other), ``start_s``/``end_s``
  (seconds from the start of the audio file), ``asr_word_span`` (inclusive
  ``[i, j]`` into ``asr.words``), ``transcript`` (that broadcast's own ASR
  text), ``callsign``, ``callsigns_all``, ``confidence`` (0-1) and
  ``confidence_detail``, followed by the parser's per-type fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

DOCUMENT_SCHEMA_VERSION = "1.0"
BROADCAST_TYPES = ("eam", "skyking", "radio_check", "other")
_FIRST_KEYS = (
    "broadcast_id",
    "type",
    "start_s",
    "end_s",
    "asr_word_span",
    "transcript",
    "callsign",
    "callsigns_all",
    "confidence",
    "confidence_detail",
)


def utc_now_iso() -> str:
    """ISO 8601 UTC with a trailing Z, whole seconds (2026-10-26T22:11:00Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def broadcast_id(sha256: str, start_s: float) -> str:
    return f"{sha256[:12]}@{round(float(start_s), 1):.1f}"


def _segment_word_spans(segments: Sequence[dict], words: Sequence[dict]) -> list[Optional[list[int]]]:
    """Inclusive word index range per ASR segment. Uses the transcriber's own
    ``word_span`` when present; otherwise (older cached output) assigns each
    word, in order, to the segment its midpoint falls in."""
    if all("word_span" in s for s in segments):
        return [list(s["word_span"]) if s["word_span"] is not None else None for s in segments]
    spans: list[Optional[list[int]]] = [None] * len(segments)
    k = 0
    for wi, w in enumerate(words):
        mid = (float(w.get("start") or 0) + float(w.get("end") or 0)) / 2
        while k + 1 < len(segments) and mid >= float(segments[k + 1]["start"]) - 1e-6:
            k += 1
        if not segments:
            break
        sp = spans[k]
        spans[k] = [wi, wi] if sp is None else [sp[0], wi]
    return spans


def build_asr(result: dict, snap: bool = True) -> dict:
    """``asr`` section from ``Decoder.decode()`` output: first-pass words
    (with timings and probabilities), segments with their word spans, and
    the second-pass info."""
    first = result.get("asr") or {}
    words_in = list(first.get("words") or [])
    snapped = words_in
    if snap:
        from .snap import snap_words

        snapped = snap_words(words_in)
    words = []
    for w, s in zip(words_in, snapped, strict=True):
        out = {
            "word": str(w.get("word", "")),
            "start": w.get("start"),
            "end": w.get("end"),
            "probability": w.get("probability"),
        }
        if s.get("snapped_from") is not None:
            out["snapped_to"] = s["word"]
        words.append(out)
    segments_in = list(first.get("segments") or [])
    spans = _segment_word_spans(segments_in, words_in)
    segments = [
        {
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "text": str(s.get("text", "")).strip(),
            "avg_logprob": s.get("avg_logprob"),
            "no_speech_prob": s.get("no_speech_prob"),
            "word_span": span,
        }
        for s, span in zip(segments_in, spans, strict=True)
    ]
    sp = dict(result.get("second_pass") or {})
    return {
        "model": first.get("model"),
        "compute_type": first.get("compute_type"),
        "config_key": first.get("config_key"),
        # Mirror of the top-level field, for loaders written against §12's
        # first draft (which read ``asr.decoder_version``).
        "decoder_version": result.get("decoder_version"),
        "day_callsigns": list(first.get("day_callsigns") or []),
        "duration_s": first.get("duration_s"),
        "asr_seconds": first.get("asr_seconds"),
        "text": str(first.get("text", "")).strip(),
        "segments": segments,
        "words": words,
        "first_pass": {
            "config_key": first.get("config_key"),
            "model": first.get("model"),
            "compute_type": first.get("compute_type"),
            "asr_seconds": first.get("asr_seconds"),
        },
        "second_pass": sp,
        # Long-recording guards (windows, loop/timeout fallbacks, coverage repairs);
        # None for ASR output cached before decoding 0.3.
        "guard": first.get("guard"),
    }


def build_broadcasts(
    records: Sequence[dict], words: Sequence[dict], sha256: str, truncated: bool = False
) -> list[dict]:
    """Wrap parser records into broadcasts (see module docstring). ``words``
    are the ``asr.words`` the records' ``segment.word_start/word_end`` index."""
    out: list[dict] = []
    seen: dict[str, int] = {}
    prev_end = 0.0
    for rec in records:
        seg = rec.get("segment") or {}
        ws, we = seg.get("word_start"), seg.get("word_end")
        span = [ws, we] if ws is not None and we is not None else None
        start = seg.get("start_sec")
        if start is None:
            start = words[ws]["start"] if span and words[ws].get("start") is not None else prev_end
        end = seg.get("end_sec")
        if end is None:
            end = words[we]["end"] if span and words[we].get("end") is not None else start
        start, end = round(float(start), 2), round(float(end), 2)
        prev_end = end
        bid = broadcast_id(sha256, start)
        if bid in seen:  # two broadcasts in the same 0.1 s: never in practice
            seen[bid] += 1
            bid = f"{bid}#{seen[bid]}"
        else:
            seen[bid] = 1
        conf = rec.get("confidence")
        conf = 0.0 if not isinstance(conf, (int, float)) else round(min(1.0, max(0.0, float(conf))), 3)
        flags = list(rec.get("flags") or [])
        if truncated and rec is records[-1] and "truncated" not in flags:
            flags.append("truncated")  # recorder hit its length cap: the last broadcast is cut
        b = {
            "broadcast_id": bid,
            "type": rec["type"],
            "start_s": start,
            "end_s": end,
            "asr_word_span": span,
            "transcript": " ".join(str(words[k].get("word", "")).strip() for k in range(ws, we + 1))
            if span
            else "",
            "callsign": rec.get("callsign"),
            "callsigns_all": list(rec.get("callsigns_all") or []),
            "confidence": conf,
            "confidence_detail": rec.get("confidence_detail") or {},
        }
        b.update({k: v for k, v in rec.items() if k not in _FIRST_KEYS})
        b["flags"] = flags
        out.append(b)
    return out


def build_document(
    result: dict,
    recording: dict,
    decoded_at: Optional[str] = None,
    snap: bool = True,
) -> dict:
    """The full per-recording document. ``result`` is ``Decoder.decode()``
    output; ``recording`` must hold ``path`` and ``sha256`` and may carry the
    Phase 0 catalog fields (copied verbatim)."""
    if not recording.get("sha256") or not recording.get("path"):
        raise ValueError("recording needs path and sha256")
    asr = build_asr(result, snap=snap)
    broadcasts = build_broadcasts(
        result.get("records") or [], asr["words"], recording["sha256"], bool(recording.get("truncated"))
    )
    duration = result.get("duration_s")
    return {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "decoder_version": result.get("decoder_version"),
        "recording": dict(recording),
        "duration_s": None if duration is None else round(float(duration), 3),
        "decoded_at": decoded_at or utc_now_iso(),
        "asr": asr,
        "broadcasts": broadcasts,
    }
