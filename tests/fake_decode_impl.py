"""Fake ASR and voice gate for the decode-worker tests (loaded inside the worker's child
process through the hidden --transcribe-impl / --gate-impl hooks). No model, no ffmpeg."""

import json
import os
import time
from pathlib import Path

from decode_fixtures import eam_text, words


def _log(kind, audio):
    path = os.environ.get("FAKE_DECODE_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({"kind": kind, "audio": Path(audio).name, "pid": os.getpid()}) + "\n")


def transcribe_eam(cfg, audio, day_callsigns):
    """An EAM read twice (made-up words); behaves on `slow` / `boom` in the file name."""
    name = Path(audio).name
    _log("asr", audio)
    if "slow" in name:
        Path(os.environ["FAKE_DECODE_PIDFILE"]).write_text(str(os.getpid()))
        time.sleep(120)
    if "boom" in name:
        raise RuntimeError("fake ASR exploded")
    ws = words(eam_text(), t0=1.0)
    return {"model": cfg.model, "compute_type": cfg.compute_type, "duration_s": 60.0, "asr_seconds": 0.1,
            "text": " ".join(w["word"] for w in ws), "words": ws,
            "segments": [{"start": ws[0]["start"], "end": ws[-1]["end"], "text": "x", "avg_logprob": -0.2,
                          "no_speech_prob": 0.01, "word_span": [0, len(ws) - 1]}]}


def gate_by_name(audio, cfg):
    """No voice for files named *silent*; voice otherwise."""
    _log("gate", audio)
    if "silent" in Path(audio).name:
        return {"voice": False, "speech_s": 0.0, "method": "fake", "reason": "fake silence"}
    return {"voice": True, "speech_s": 12.0, "method": "fake", "reason": "fake speech"}
