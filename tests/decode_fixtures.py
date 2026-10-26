"""Fake Decoding outputs in decode output contract v1, built with Decoding's own parser
(schema 0.2) and document writer (decoding.output.build_document, on main since #13), so the
documents match what `python -m decoding.run` writes. All words, callsigns and bodies are made up."""

from decoding.eam_parser import parse_transcript
from decoding.output import build_document

BODY_WORDS = ("quebec xray alfa bravo charlie delta echo foxtrot golf hotel juliett kilo lima mike "
              "november papa romeo sierra tango uniform victor whiskey yankee zulu two three four five "
              "six seven")
SKYKING_TEXT = ("skyking skyking do not answer sierra tango uniform time one five authentication papa quebec "
                "i say again skyking skyking do not answer sierra tango uniform time one five "
                "authentication papa quebec this is fakecall out")


def eam_text(body: str = BODY_WORDS, callsign: str = "fakecall") -> str:
    return (f"all stations all stations this is {callsign} with a flash message stand by "
            f"message follows {body} i say again {body} this is {callsign} out")


def words(text: str, t0: float = 2.0, p: float = 0.9) -> list[dict]:
    out, t = [], t0
    for w in text.split():
        out.append({"word": w, "probability": p, "start": round(t, 2), "end": round(t + 0.4, 2)})
        t += 0.5
    return out


def contract_doc(recording: dict, text: str, *, t0: float = 2.0, p: float = 0.9,
                 decoder_version: str = "decoding/0.2.0;parser=0.2;asr=ab12cd34ef56;second_pass=vote3@0.3",
                 decoded_at: str = "2031-01-02T04:00:00.000Z") -> dict:
    ws = words(text, t0, p)
    result = {
        "decoder_version": decoder_version,
        "duration_s": recording.get("duration_sec", 70.0),
        "asr": {"model": "small.en", "compute_type": "int8", "config_key": "ab12cd34ef56", "words": ws,
                "segments": [{"start": ws[0]["start"], "end": ws[-1]["end"], "text": text}]},
        "second_pass": {"mode": "vote3@0.3", "enabled": True, "fired": False, "passes": 1},
        "records": parse_transcript(ws),
    }
    return build_document(result, {"path": recording["file"], **recording}, decoded_at=decoded_at, snap=False)
