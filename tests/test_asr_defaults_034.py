"""Live ASR defaults for decoding 0.3.4 (no Skyking seed)."""

from decoding.asr import AsrConfig, build_hotwords, resolve_prompt
from decoding.pipeline import DecodeConfig, decoder_version


def test_asr_defaults_have_no_skyking_seed():
    c = AsrConfig()
    assert c.prompt is None and c.condition_on_previous_text is False
    assert resolve_prompt(c) is None
    hot = build_hotwords(["FAKECALL"], include_skyking=c.prompt not in (None, "phonetic_only"))
    assert "Skyking" not in hot and "FAKECALL" in hot and "Mainsail" in hot


def test_decoder_version_is_034():
    v = decoder_version(DecodeConfig())
    assert v.startswith("decoding/0.3.4;")
    assert "asr=" in v
