"""Decode output contract (docs/decoding.md "Decode output", docs/decode.schema.json)."""

import json
import re
from pathlib import Path

import pytest

from decoding import parse_transcript, tokenize
from decoding.output import BROADCAST_TYPES, broadcast_id, build_document
from decoding.pipeline import DecodeConfig, Decoder, decoder_version
from decoding.run import decode_recording, iter_inputs, recording_section, sidecar_path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs" / "decode.schema.json").read_text())
FIXTURE = Path(__file__).parent / "fixtures" / "decode_sample_two_broadcasts.decode.json"
SHA = "ab" * 32

_NATO = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima".split()
_SPELL = dict(zip("ABCDEFGHIJKL", _NATO, strict=True))
BODY = "ABCDEFGHIJKL"


def _say(chars):
    return " ".join(_SPELL[c] for c in chars)


def _eam(opening="Mainsail", closing="Mainsail", addressee=None, body=BODY):
    pre = _say(body[:6])
    head = f"All stations, this is {opening}. " if opening else "All stations. "
    if addressee:
        head += f"For {addressee}, for {addressee}. "
    tail = f" This is {closing}, out." if closing else ""
    return (
        f"{head}{pre} stand by. {pre} stand by. {pre} stand by. Message follows. "
        f"{_say(body)}. I say again. {_say(body)}.{tail}"
    )


SKYKING = (
    "Skyking, Skyking, do not answer. Redwood. Time three five. Authentication Bravo Charlie. "
    "I say again. Skyking, Skyking, do not answer. Redwood. Time three five. "
    "Authentication Bravo Charlie. This is Andrews, out."
)
RADIO_CHECK = "This is Offutt with a test count. One two three, three two one. This is Offutt, out."


def _words(text, t0=0.0, prob=0.9, step=0.5):
    """Whisper-style word dicts, one per whitespace word, ``step`` s apart."""
    return [
        {
            "word": " " + w,
            "probability": prob,
            "start": round(t0 + i * step, 2),
            "end": round(t0 + i * step + 0.4, 2),
        }
        for i, w in enumerate(text.split())
    ]


class FakeAsr:
    def __init__(self, words):
        self.words = words

    def __call__(self, cfg, audio, day_callsigns):
        n = len(self.words)
        segs = (
            [
                {
                    "start": self.words[0]["start"],
                    "end": self.words[-1]["end"],
                    "text": "x",
                    "word_span": [0, n - 1],
                }
            ]
            if n
            else []
        )
        text = " ".join(w["word"].strip() for w in self.words)
        return {
            "model": cfg.model,
            "compute_type": cfg.compute_type,
            "duration_s": 300.0,
            "asr_seconds": 1.0,
            "text": text,
            "words": self.words,
            "segments": segs,
        }


def _multi_words():
    w = _words(_eam(opening="Mainsail", closing="Andrews", addressee="Red Dog"), t0=5.0)
    w += _words(SKYKING, t0=w[-1]["end"] + 10)
    w += _words(RADIO_CHECK, t0=w[-1]["end"] + 10, prob=0.7)
    return w


def _doc(words, recording=None, cfg=None):
    dec = Decoder(cfg or DecodeConfig(), transcribe=FakeAsr(words))
    result = dec.decode(Path("clip.wav"))
    return build_document(result, recording or {"path": "/r/clip.wav", "sha256": SHA}, snap=dec.cfg.snap)


# ------------------------------------------------------------------ mini JSON Schema check
def _check(inst, sch, path="$"):
    """Validate the JSON Schema subset docs/decode.schema.json uses (CI has no
    jsonschema): $ref, anyOf, type, enum, required, properties, items,
    pattern, minimum/maximum, minItems/maxItems."""
    if "$ref" in sch:
        name = sch["$ref"].split("/")[-1]
        return _check(inst, SCHEMA["$defs"][name], path)
    if "anyOf" in sch:
        errs = [_check(inst, s, path) for s in sch["anyOf"]]
        return [] if any(not e for e in errs) else [f"{path}: no anyOf branch matches"]
    errs = []
    types = sch.get("type")
    if types:
        types = [types] if isinstance(types, str) else types
        py = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}

        def ok(t):
            if t == "number":
                return isinstance(inst, (int, float)) and not isinstance(inst, bool)
            if t == "integer":
                return isinstance(inst, int) and not isinstance(inst, bool)
            return isinstance(inst, py[t])

        if not any(ok(t) for t in types):
            return [f"{path}: {inst!r:.60} is not {types}"]
    if "enum" in sch and inst not in sch["enum"]:
        errs.append(f"{path}: {inst!r} not in {sch['enum']}")
    if isinstance(inst, dict):
        errs += [f"{path}: missing {k}" for k in sch.get("required", []) if k not in inst]
        for k, s in sch.get("properties", {}).items():
            if k in inst:
                errs += _check(inst[k], s, f"{path}.{k}")
    if isinstance(inst, list):
        if len(inst) < sch.get("minItems", 0) or len(inst) > sch.get("maxItems", 10**9):
            errs.append(f"{path}: bad length {len(inst)}")
        if "items" in sch:
            for i, x in enumerate(inst):
                errs += _check(x, sch["items"], f"{path}[{i}]")
    if isinstance(inst, str) and "pattern" in sch and not re.search(sch["pattern"], inst):
        errs.append(f"{path}: {inst!r} !~ {sch['pattern']}")
    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if "minimum" in sch and inst < sch["minimum"]:
            errs.append(f"{path}: {inst} < {sch['minimum']}")
        if "maximum" in sch and inst > sch["maximum"]:
            errs.append(f"{path}: {inst} > {sch['maximum']}")
    return errs


def _assert_contract(doc):
    assert _check(doc, SCHEMA) == []
    words = doc["asr"]["words"]
    prev = -1
    for b in doc["broadcasts"]:
        i, j = b["asr_word_span"]
        assert prev < i <= j < len(words)  # disjoint, in order
        prev = j
        assert b["transcript"] == " ".join(w["word"].strip() for w in words[i : j + 1])
        assert b["start_s"] == words[i]["start"] and b["end_s"] >= b["start_s"]
        assert b["broadcast_id"] == broadcast_id(doc["recording"]["sha256"], b["start_s"])
        assert 0.0 <= b["confidence"] <= 1.0
        json.dumps(b["callsigns_all"]), json.dumps(b["confidence_detail"])


# ------------------------------------------------------------------ tokens -> ASR words
def test_tokens_carry_source_word_indices():
    toks = tokenize([{"word": "Alpha,Bravo"}, {"word": "X"}, {"word": "ray"}, {"word": "I"}, {"word": "say"},
                     {"word": "again."}, {"word": "-"}, {"word": "Sky"}, {"word": "king"}])  # fmt: skip
    assert [(t.norm, t.src) for t in toks] == [
        ("alpha", (0, 0)), ("bravo", (0, 0)), ("xray", (1, 2)), ("sayagain", (3, 5)), ("skyking", (7, 8)),
    ]  # fmt: skip


def test_parser_segments_index_input_words():
    words = _multi_words()
    recs = parse_transcript(words)
    assert [r["type"] for r in recs] == ["eam", "skyking", "radio_check"]
    spans = [(r["segment"]["word_start"], r["segment"]["word_end"]) for r in recs]
    assert spans[0][0] == 0 and spans[-1][1] == len(words) - 1
    assert all(a[1] + 1 == b[0] for a, b in zip(spans, spans[1:], strict=False))


# ------------------------------------------------------------------ document
def test_multi_broadcast_document():
    doc = _doc(_multi_words())
    _assert_contract(doc)
    eam, sky, rc = doc["broadcasts"]
    assert [b["type"] for b in doc["broadcasts"]] == ["eam", "skyking", "radio_check"]
    assert eam["transcript"].startswith("All stations,") and eam["transcript"].endswith("Andrews, out.")
    assert sky["transcript"].startswith("Skyking,") and sky["transcript"].endswith("Andrews, out.")
    assert rc["transcript"].startswith("This is Offutt") and "test count" in rc["transcript"]
    assert eam["start_s"] == 5.0 and sky["start_s"] > eam["end_s"] and rc["start_s"] > sky["end_s"]
    assert eam["body"] == BODY and sky["codeword"]["value"] == "REDWOOD" and sky["time"]["value"] == "35"


def test_top_level_fields_and_versions():
    doc = _doc(_multi_words())
    assert list(doc)[:7] == ["schema_version", "decoder_version", "recording", "duration_s", "decoded_at",
                             "asr", "broadcasts"]  # fmt: skip
    cfg = DecodeConfig()
    assert doc["decoder_version"] == f"decoding/0.2.0;parser=0.2;asr={cfg.asr.key()};second_pass=vote3@0.3"
    assert doc["asr"]["decoder_version"] == doc["decoder_version"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc["decoded_at"])
    assert decoder_version(DecodeConfig(second_pass=False)).endswith(";second_pass=off")
    sp = doc["asr"]["second_pass"]
    assert sp["mode"] == "vote3@0.3" and sp["fired"] is False and sp["passes"] == 1
    assert doc["asr"]["segments"][0]["word_span"] == [0, len(doc["asr"]["words"]) - 1]


def test_broadcast_ids_are_stable_across_redecodes():
    words = _multi_words()
    a, b = _doc(words), _doc(words)
    ids = [x["broadcast_id"] for x in a["broadcasts"]]
    assert ids == [x["broadcast_id"] for x in b["broadcasts"]]
    assert ids[0] == "abababababab@5.0"
    # A re-decode that finds an extra broadcast earlier keeps the later ids.
    extra = _words(RADIO_CHECK, t0=0.0)
    shifted = [{**w, "start": w["start"]} for w in words]
    c = _doc(extra[:-1] + [{**extra[-1], "start": 4.0, "end": 4.4}] + shifted)
    later = [x["broadcast_id"] for x in c["broadcasts"]]
    assert set(ids[1:]) <= set(later)


def test_type_names_match_labels():
    seen = set()
    for text in (_eam(), SKYKING, RADIO_CHECK, "hello is anyone there", ""):
        doc = _doc(_words(text) if text else [])
        seen |= {b["type"] for b in doc["broadcasts"]}
    assert seen == set(BROADCAST_TYPES) == {"eam", "skyking", "radio_check", "other"}


def test_empty_transcript_still_has_numeric_start():
    doc = _doc([])
    (b,) = doc["broadcasts"]
    assert (
        b["type"] == "other" and b["start_s"] == 0.0 and b["asr_word_span"] is None and b["transcript"] == ""
    )
    assert b["confidence"] == 0.0 and _check(doc, SCHEMA) == []


@pytest.mark.parametrize(
    "text,callsign,roles",
    [
        (_eam(opening="Mainsail", closing="Andrews"), "ANDREWS",
         [("MAINSAIL", "opening"), ("ANDREWS", "closing")]),
        (_eam(opening="Mainsail", closing=None), "MAINSAIL", [("MAINSAIL", "opening")]),
        (_eam(opening=None, closing="Offutt"), "OFFUTT", [("OFFUTT", "closing")]),
        (_eam(addressee="Red Dog"), "MAINSAIL",
         [("MAINSAIL", "opening"), ("RED DOG", "addressee"), ("MAINSAIL", "closing")]),
        (SKYKING, "ANDREWS", [("ANDREWS", "closing")]),
        (RADIO_CHECK, "OFFUTT", [("OFFUTT", "station")]),
        ("hello is anyone there", None, []),
    ],
)  # fmt: skip
def test_callsign_selection(text, callsign, roles):
    (b,) = _doc(_words(text))["broadcasts"]
    assert b["callsign"] == callsign
    assert [(c["callsign"], c["role"]) for c in b["callsigns_all"]] == roles


def test_confidence_per_type():
    doc = _doc(_multi_words())
    eam, sky, rc = doc["broadcasts"]
    assert eam["confidence"] == eam["confidence_detail"]["mean"]
    assert set(eam["confidence_detail"]) == {"mean", "min", "low_count", "reads_agree_ratio"}
    assert sky["confidence"] == min(sky["confidence_detail"].values())
    assert sky["confidence_detail"] == {
        f: sky[f]["confidence"] for f in ("codeword", "time", "authentication")
    }
    assert rc["confidence"] == pytest.approx(0.7) and rc["confidence_detail"]["n_words"] > 0
    for prob in (0.0, 0.3, 1.0):
        for b in _doc(_multi_words_prob(prob))["broadcasts"]:
            assert 0.0 <= b["confidence"] <= 1.0


def _multi_words_prob(p):
    return [{**w, "probability": p} for w in _multi_words()]


def test_second_pass_keeps_first_pass_spans():
    words = _words(_eam(body=BODY).replace(f"say again. {_say(BODY)}", "say again. " + _say("LKJIHGFEDCBA")))
    doc = _doc(words)
    (b,) = [x for x in doc["broadcasts"] if x["type"] == "eam"]
    assert doc["asr"]["second_pass"]["fired"] and "second_pass_vote" in b["flags"]
    _assert_contract(doc)


def test_truncated_recording_flags_last_broadcast():
    doc = _doc(_multi_words(), {"path": "/r/x.wav", "sha256": SHA, "truncated": True})
    assert "truncated" in doc["broadcasts"][-1]["flags"]
    assert all("truncated" not in b["flags"] for b in doc["broadcasts"][:-1])


# ------------------------------------------------------------------ writer
def test_writer_sidecar_equals_jsonl_line_and_is_idempotent(tmp_path):
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    wav = rec_dir / "HFGCS_11175kHz_20261026T220000Z_rx1.wav"
    wav.write_bytes(b"RIFF fake audio")
    entry = {
        "schema": 2,
        "start_utc": "2026-10-26T22:00:05.000Z",
        "file_start_utc": "2026-10-26T22:00:00.000Z",
        "end_utc": "2026-10-26T22:05:00.000Z",
        "frequency_khz": 11175.0,
        "mode": "USB",
        "receiver_id": "rx1",
        "file": "recordings/" + wav.name,
        "sha256": None,
        "audio_rate": 8000,
        "duration_sec": 300.0,
        "preroll_sec": 5.0,
        "snr_db_peak": 21.0,
        "snr_db_mean": 12.5,
        "noise_floor_db": -90.0,
        "truncated": False,
    }
    catalog = tmp_path / "hfgcs_catalog.jsonl"
    catalog.write_text(json.dumps(entry) + "\n")
    ((audio, e, jdir),) = list(iter_inputs([catalog]))
    assert audio == wav.resolve() and e == entry and jdir == tmp_path
    dec = Decoder(transcribe=FakeAsr(_multi_words()))
    jsonl = tmp_path / "decodes.jsonl"
    doc = decode_recording(dec, audio, e, jsonl=jsonl)
    side = sidecar_path(audio)
    assert side.name == wav.name + ".decode.json"
    assert json.loads(side.read_text()) == doc == json.loads(jsonl.read_text().splitlines()[0])
    r = doc["recording"]
    assert r["path"] == str(audio) and len(r["sha256"]) == 64 and r["receiver_id"] == "rx1"
    assert r["snr_db_mean"] == 12.5 and r["end_utc"] == entry["end_utc"] and "catalog_sha256" not in r
    _assert_contract(doc)
    assert decode_recording(dec, audio, e, jsonl=jsonl) is None  # up to date: skipped
    assert len(jsonl.read_text().splitlines()) == 1
    assert decode_recording(dec, audio, e, jsonl=jsonl, force=True) is not None
    assert len(jsonl.read_text().splitlines()) == 2


def test_recording_section_reports_catalog_hash_mismatch(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    r = recording_section(f, {"file": "a.wav", "sha256": "0" * 64})
    assert r["sha256"] != "0" * 64 and r["catalog_sha256"] == "0" * 64
    bare = recording_section(f, None)
    assert bare == {"path": str(f), "sha256": r["sha256"], "file": str(f)}


# ------------------------------------------------------------------ committed sample
def test_committed_sample_fixture():
    """Real decode of a synthetic recording (gold EAMs eam-356f1a01 + 2 s +
    eam-86d80b13, missileer.watch CC0) made by `python -m decoding.run`;
    see tests/fixtures/decode_sample_two_broadcasts.README.md."""
    doc = json.loads(FIXTURE.read_text())
    _assert_contract(doc)
    assert len(doc["broadcasts"]) >= 2
    assert all(b["type"] == "eam" for b in doc["broadcasts"])
    # gold: "... This is Coupling, out" -> heard as a closing callsign
    assert doc["broadcasts"][0]["callsigns_all"][-1]["role"] == "closing"
    assert doc["recording"]["schema"] == 2 and doc["recording"]["receiver_id"] == "eval-synthetic"
    assert len({b["broadcast_id"] for b in doc["broadcasts"]}) == len(doc["broadcasts"])
    assert all(b["broadcast_id"].startswith(doc["recording"]["sha256"][:12] + "@") for b in doc["broadcasts"])
