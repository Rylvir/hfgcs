from decoding.asr import AsrConfig
from decoding.pipeline import DecodeConfig, Decoder

_SPELL = {
    "A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo", "F": "Foxtrot", "G": "Golf",
    "H": "Hotel", "I": "India", "J": "Juliet", "K": "Kilo", "L": "Lima", "X": "X-ray", "Z": "Zulu",
}  # fmt: skip


def _say(chars):
    return " ".join(_SPELL[c] for c in chars)


def _eam(read1, read2):
    pre = _say(read1[:6])
    return (
        f"All stations, this is Mainsail. {pre} stand by. {pre} stand by. Message follows. "
        f"{_say(read1)}. I say again. {_say(read2)}. This is Mainsail, out."
    )


TRUE = "ABCDEFGHIJKL"


class FakeAsr:
    """Returns a scripted transcript per (vad_filter, condition_on_previous_text)."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def __call__(self, cfg: AsrConfig, audio, day_callsigns):
        k = (cfg.vad_filter, cfg.condition_on_previous_text)
        self.calls.append(k)
        words = [
            {"word": " " + w, "probability": 0.9, "start": float(i), "end": i + 0.5}
            for i, w in enumerate(self.texts[k].split())
        ]
        return {"words": words, "duration_s": 100.0, "asr_seconds": 10.0}


def test_default_config_has_second_pass_on_and_preamble_fusion_off():
    c = DecodeConfig()
    assert c.second_pass and c.second_pass_threshold == 0.3 and not c.preamble_fusion and c.snap
    assert c.asr.vad_filter is False and c.asr.condition_on_previous_text is True


def test_second_pass_fires_on_low_agreement_and_votes():
    asr = FakeAsr(
        {
            (False, True): _eam("ABCDEFGHIJXL", "ZZZZZZZZZZZZ"),  # reads disagree -> low agreement
            (False, False): _eam(TRUE, TRUE),
            (True, False): _eam(TRUE, TRUE),
        }
    )
    out = Decoder(transcribe=asr).decode("clip.opus")
    rec = [r for r in out["records"] if r["type"] == "eam"][0]
    assert out["second_pass"]["fired"] and out["second_pass"]["passes"] == 3
    assert asr.calls == [(False, True), (False, False), (True, False)]
    assert "second_pass_vote" in rec["flags"] and "body_first_pass" in rec
    assert rec["body"] == TRUE and rec["body_first_pass"] != TRUE


def test_second_pass_skipped_when_reads_agree_or_disabled():
    agree = {(False, True): _eam(TRUE, TRUE)}
    asr = FakeAsr(agree)
    out = Decoder(transcribe=asr).decode("clip.opus")
    assert not out["second_pass"]["fired"] and asr.calls == [(False, True)]
    asr = FakeAsr({(False, True): _eam(TRUE, "ZZZZZZZZZZZZ")})
    out = Decoder(DecodeConfig(second_pass=False), transcribe=asr).decode("clip.opus")
    assert not out["second_pass"]["fired"] and asr.calls == [(False, True)]


def test_no_eam_never_triggers_second_pass():
    asr = FakeAsr({(False, True): "Mainsail radio check, how copy, over."})
    out = Decoder(transcribe=asr).decode("clip.opus")
    assert not out["second_pass"]["fired"] and len(asr.calls) == 1


def test_pipeline_second_passes_match_eval_configs():
    from dataclasses import replace

    from decoding.evaluate import CONFIGS, split_config
    from decoding.vote import SECOND_PASS_CONFIGS, SECOND_PASS_PASSES

    for name, (overrides, snap) in zip(SECOND_PASS_CONFIGS, SECOND_PASS_PASSES, strict=True):
        cfg, post = split_config(CONFIGS[name])
        assert replace(AsrConfig(), **overrides).key() == cfg.key()
        assert bool(post.get("snap")) == snap
    d_cfg, d_post = split_config(CONFIGS["default"])
    assert d_cfg.key() == DecodeConfig().asr.key() and d_post["vote"] == list(SECOND_PASS_CONFIGS)


def test_transcriber_shares_model_when_compatible():
    from decoding.asr import Transcriber

    a = Transcriber(AsrConfig())
    a._model = sentinel = object()
    b = Transcriber(AsrConfig(vad_filter=True), share_with=a)
    assert b._load() is sentinel
    c = Transcriber(AsrConfig(model="medium.en"), share_with=a)
    c._model = other = object()
    assert c._load() is other
