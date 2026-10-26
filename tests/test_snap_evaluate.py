import json

from decoding.evaluate import (
    aggregate,
    baseline_machine,
    best_eam,
    load_subset,
    masked_edit_distance,
    predicted_type,
    score_clip,
)
from decoding.snap import snap_word, snap_words, sound_key


def test_sound_key_groups_voicing_pairs():
    assert sound_key("tango") == sound_key("dango")
    assert sound_key("papa") == sound_key("baba")


def test_snap_recovers_near_misses():
    for wrong, right in [
        ("dango", "tango"),
        ("romio", "romeo"),
        ("oskar", "oscar"),
        ("hotell", "hotel"),
        ("viktor", "victor"),
        ("zoolu", "zulu"),
        ("kilow", "kilo"),
        ("seara", "sierra"),
    ]:
        assert snap_word(wrong) == right, wrong


def test_snap_leaves_procedure_and_english_alone():
    for w in [
        "stand",
        "message",
        "follows",
        "again",
        "the",
        "characters",
        "mainsail",
        "manage",
        "break",
        "alpha",
        "niner",
        "x",
        "-ray",
        "brake",
        "your",
    ]:
        assert snap_word(w) is None, w


def test_snap_words_marks_and_discounts():
    out = snap_words([{"word": " Dango,", "probability": 0.8, "start": 1.0, "end": 1.2}, {"word": "stand"}])
    assert out[0]["word"] == "tango" and out[0]["probability"] == 0.4 and out[0]["snapped_from"] == " Dango,"
    assert out[1] == {"word": "stand"}


def test_masked_edit_distance_treats_dot_as_wildcard():
    assert masked_edit_distance("AB.D", "ABXD") == 0
    assert masked_edit_distance("ABCD", "ABXD") == 1
    assert masked_edit_distance("ABCD", "") == 4


def _label(body="ABCDEF1234", typ="eam", pre="ABCDEF", length=None):
    return {
        "id": "x",
        "type": typ,
        "label_status": "gold",
        "labels": {"body": body if typ == "eam" else None, "preamble": pre, "declared_length": length},
        "reference_machine": {"text": "ABCDEF12"},
    }


def test_score_and_aggregate():
    recs = [
        {"type": "eam", "body": "ABCDEF1234", "preamble": "ABCDEF", "declared_length": None, "flags": []},
        {"type": "eam", "body": "AB", "flags": []},
    ]
    assert best_eam(recs)["body"] == "ABCDEF1234"
    good = score_clip(_label(), recs)
    assert good["cer"] == 0 and good["preamble_em"] is True and good["length_em"] is None
    miss = score_clip(_label(), [])
    assert miss["cer"] == 1.0 and miss["detected"] is False
    neg = score_clip(_label(typ="radio_check"), recs)
    assert neg["false_eam"] is True
    assert good["type_hyp"] == "eam" and good["type_match"] and not neg["type_match"]
    agg = aggregate([good, miss, neg])
    assert agg["cer_micro"] == 0.5 and agg["n_eam"] == 2 and agg["false_eam"] == 1
    assert baseline_machine([_label()])["cer_micro"] == 0.2
    assert agg["type_acc"] == 1 / 3  # good matches; miss (no EAM found) and neg do not


def test_predicted_type_uses_broadcast_type_names():
    assert predicted_type([{"type": "radio_check"}]) == "radio_check"
    assert predicted_type([{"type": "other"}]) == "other"
    assert predicted_type([{"type": "skyking"}, {"type": "eam", "body": ""}]) == "skyking"
    rc = score_clip(_label(typ="radio_check"), [{"type": "radio_check"}])
    assert rc["type_match"] and not rc["false_eam"]
    auth = score_clip(_label(typ="auth"), [{"type": "other"}])
    assert auth["type_match"]


def test_load_subset_filters(tmp_path):
    rows = [
        {"label_status": "gold", "type": "eam", "audio_path": "a", "labels": {"body": "AB.C"}},
        {
            "label_status": "silver",
            "type": "eam",
            "audio_path": "b",
            "labels": {"body": "ABC"},
            "has_both_reads": True,
        },
        {
            "label_status": "silver",
            "type": "eam",
            "audio_path": "c",
            "labels": {"body": "A.C"},
            "has_both_reads": True,
        },
        {
            "label_status": "silver",
            "type": "eam",
            "audio_path": None,
            "labels": {"body": "ABC"},
            "has_both_reads": True,
        },
    ]
    p = tmp_path / "l.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r["audio_path"] for r in load_subset(p, "gold")] == ["a"]
    assert [r["audio_path"] for r in load_subset(p, "silver")] == ["b"]


def test_merge_regions_windows():
    from decoding.asr import merge_regions

    out = merge_regions([(1, 2), (2.3, 4), (10, 30)], max_len=8, gap=0.6, pad=0.3, duration=40)
    assert out[0] == (0.7, 4.3)  # first two joined (gap 0.3 <= 0.6)
    assert len(out) == 4 and all(b - a <= 8 + 0.6 + 1e-9 for a, b in out)
    assert merge_regions([(0.1, 1.0)], 8, 0.6, 0.3, 1.1) == [(0.0, 1.1)]  # clamped


def test_snap_keeps_split_xray_joinable():
    from decoding.tokens import tokenize

    words = snap_words([{"word": "X"}, {"word": "-ray,"}])
    assert [t.norm for t in tokenize(words)] == ["xray"]


def test_config_overrides_are_relative_to_v01_and_best_is_default():
    from decoding.asr import AsrConfig
    from decoding.evaluate import CONFIGS, split_config

    base, _ = split_config(CONFIGS["baseline"])
    assert base.vad_filter is True  # v0.1 rows keep their cache keys
    best, post = split_config(CONFIGS["best"])
    assert best.key() == AsrConfig().key()
    assert post == {"snap": True}
    assert split_config(CONFIGS["nv_cond_prev"])[0].key() == AsrConfig().key()
    assert base.condition_on_previous_text is False
