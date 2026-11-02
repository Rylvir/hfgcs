"""Decoder.decode runs filter_records(suppress=True) before returning."""

from dataclasses import replace

from decoding.pipeline import DecodeConfig, Decoder


class _FakeTr:
    def __init__(self, words):
        self.words = words

    def __call__(self, cfg, audio, day_callsigns=()):
        return {
            "words": self.words,
            "segments": [],
            "text": " ".join(w["word"] for w in self.words),
            "duration_s": 5.0,
            "asr_seconds": 0.1,
        }


def test_decode_suppresses_prompt_echo_skyking(tmp_path):
    words = [
        {"word": w, "start": i * 0.3, "end": i * 0.3 + 0.2, "probability": 0.9}
        for i, w in enumerate("Skyking Skyking do not answer".split())
    ]
    cfg = replace(DecodeConfig(), second_pass=False, preamble_fusion=False)
    d = Decoder(cfg, transcribe=_FakeTr(words))
    out = d.decode(tmp_path / "x.wav")
    types = [r.get("type") for r in out["records"]]
    assert "skyking" not in types
