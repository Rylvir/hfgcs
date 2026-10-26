"""Fake Decoding outputs in the agreed decode output contract v1 (branch decoding-output),
built from the real parser on main so record contents match what Decoding produces.
All words, callsigns and bodies are made up."""

from decoding.eam_parser import parse_transcript

BODY_WORDS = ("quebec xray alfa bravo charlie delta echo foxtrot golf hotel juliett kilo lima mike "
              "november papa romeo sierra tango uniform victor whiskey yankee zulu two three four five "
              "six seven")
SKYKING_TEXT = ("skyking skyking do not answer sierra tango uniform time one five authentication papa quebec "
                "i say again skyking skyking do not answer sierra tango uniform time one five "
                "authentication papa quebec this is fakecall out")
TYPES = {"eam": "eam", "skyking": "skyking", "test_count": "radio_check"}


def eam_text(body: str = BODY_WORDS, callsign: str = "fakecall") -> str:
    return (f"all stations all stations this is {callsign} with a flash message stand by "
            f"message follows {body} i say again {body} this is {callsign} out")


def words(text: str, t0: float = 2.0, p: float = 0.9) -> list[dict]:
    out, t = [], t0
    for w in text.split():
        out.append({"word": w, "probability": p, "start": round(t, 2), "end": round(t + 0.4, 2)})
        t += 0.5
    return out


def _broadcast(rec: dict, ws: list[dict], sha: str) -> dict:
    """Parser record -> contract v1 broadcast (what Decoding's writer emits)."""
    seg = rec.pop("segment")
    start, end = seg["start_sec"], seg["end_sec"]
    i = next(k for k, w in enumerate(ws) if w["start"] >= start - 1e-6)
    j = max(k for k, w in enumerate(ws) if w["end"] <= end + 1e-6) + 1
    typ = TYPES.get(rec["type"], "other")
    opening = rec.get("callsign")
    closing = rec.pop("closing_callsign", None) or opening
    detail = rec.pop("confidence", None)
    if typ == "skyking":
        cs = [rec[k]["confidence"] for k in ("codeword", "time", "authentication")]
        detail = {k: rec[k]["confidence"] for k in ("codeword", "time", "authentication")}
        conf = sum(cs) / len(cs)
    else:
        conf = detail["mean"] if isinstance(detail, dict) else 0.5
    roles = [{"callsign": opening, "role": "opening"}] if opening and typ == "eam" else []
    roles.append({"callsign": closing, "role": "closing"})
    return {**rec, "type": typ, "broadcast_id": f"{sha[:12]}@{round(start, 1):.1f}",
            "start_s": start, "end_s": end, "asr_word_span": [i, j],
            "transcript": " ".join(w["word"] for w in ws[i:j]), "callsign": closing,
            "callsigns_all": roles, "confidence": round(conf, 3), "confidence_detail": detail or {}}


def contract_doc(recording: dict, text: str, *, t0: float = 2.0, p: float = 0.9,
                 decoder_version: str = "decoding/0.2.0;parser=0.1;asr=ab12cd34;second_pass=vote3@0.3",
                 decoded_at: str = "2031-01-02T04:00:00.000Z") -> dict:
    ws = words(text, t0, p)
    sha = recording["sha256"]
    recs = parse_transcript(ws)
    return {
        "schema_version": "1.0",
        "decoder_version": decoder_version,
        "recording": {"path": recording["file"], **recording},
        "duration_s": recording.get("duration_sec", 70.0),
        "decoded_at": decoded_at,
        "asr": {"model": "small.en", "words": ws,
                "segments": [{"start": ws[0]["start"], "end": ws[-1]["end"], "text": text}],
                "first_pass": {"config_key": "ab12cd34"}, "second_pass": {"fired": False, "passes": 1}},
        "broadcasts": [_broadcast(r, ws, sha) for r in recs if r.get("type") != "unknown"],
    }
