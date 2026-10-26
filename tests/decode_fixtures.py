"""Builds fake Decoding outputs (sidecar docs) with the real parser on main
(decoding.eam_parser.parse_transcript), so the loader is tested against the format
Decoding actually produces. All words, callsigns and bodies are made up."""

from decoding.eam_parser import parse_transcript

BODY_WORDS = ("quebec xray alfa bravo charlie delta echo foxtrot golf hotel juliett kilo lima mike "
              "november papa romeo sierra tango uniform victor whiskey yankee zulu two three four five "
              "six seven")


def words(text: str, t0: float = 2.0, p: float = 0.9) -> list[dict]:
    out, t = [], t0
    for w in text.split():
        out.append({"word": w, "probability": p, "start": round(t, 2), "end": round(t + 0.4, 2)})
        t += 0.5
    return out


def eam_text(body: str = BODY_WORDS, callsign: str = "fakecall") -> str:
    return (f"all stations all stations this is {callsign} with a flash message stand by "
            f"message follows {body} i say again {body} this is {callsign} out")


SKYKING_TEXT = ("skyking skyking do not answer sierra tango uniform time one five authentication papa quebec "
                "i say again skyking skyking do not answer sierra tango uniform time one five "
                "authentication papa quebec this is fakecall out")


def sidecar(recording: dict, text: str, *, model: str = "small.en", t0: float = 2.0, p: float = 0.9) -> dict:
    w = words(text, t0, p)
    return {"recording": recording,
            "asr": {"model": model, "compute_type": "int8", "text": text, "duration_s": 70.0},
            "broadcasts": parse_transcript(w)}
