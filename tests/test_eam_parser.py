from datetime import datetime, timedelta, timezone

import pytest

from decoding import Token, cer, link_rebroadcasts, parse_eam, parse_skyking, parse_transcript
from decoding.metrics import field_exact_match
from decoding.tokens import parse_number, tokenize

NATO = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliet",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "X-ray",
    "Y": "Yankee",
    "Z": "Zulu",
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Niner",
}
BODY = "QKR4TZ7HMB2XPLA9NDW3CVE6UJS8GY"  # 30 chars, synthetic


def spell(s):
    return " ".join(NATO[c] for c in s)


def eam(
    body=BODY,
    read2=None,
    callsign="Mainsail",
    addressee=None,
    length_phrase=None,
    trailer=True,
    closing="out",
):
    read2 = body if read2 is None else read2
    parts = [f"All stations, all stations, this is {callsign}, {callsign}."]
    if addressee:
        parts.append(f"For {addressee}, for {addressee}.")
    parts += [f"{spell(body[:6])}, stand by."] * 3
    parts.append(f"Message of {length_phrase} characters follows." if length_phrase else "Message follows.")
    parts += [spell(body), "I say again.", spell(read2)]
    if length_phrase and trailer:
        parts.append(f"This complete message of {length_phrase} characters.")
    parts.append(
        f"This is {callsign}, out." if closing == "out" else f"This is {callsign}, more to follow, stand by."
    )
    return " ".join(parts)


def words_with_probs(text, low=None):
    """Turn text into faster-whisper style word dicts; ``low`` maps word index
    -> probability."""
    low = low or {}
    return [
        {"word": w, "probability": low.get(i, 0.95), "start": i * 0.4, "end": i * 0.4 + 0.35}
        for i, w in enumerate(text.split())
    ]


# ------------------------------------------------------------ EAM: happy path
def test_eam_double_read_agreement():
    r = parse_eam(eam(addressee="Red Dog"))
    assert r["type"] == "eam"
    assert r["callsign"] == "MAINSAIL"
    assert r["addressee"] == "RED DOG"
    assert r["preamble"] == BODY[:6]
    assert r["preamble_reads"] == [BODY[:6]] * 3
    assert r["body"] == BODY
    assert r["read1"] == r["read2"] == BODY
    assert r["expected_length"] == 30 and r["length_source"] == "default"
    assert r["confidence"]["reads_agree_ratio"] == 1.0
    assert r["confidence"]["low_count"] == 0
    assert all(c["agree"] for c in r["chars"])
    assert r["out"] and r["closing_callsign"] == "MAINSAIL"
    assert r["flags"] == []


def test_agreeing_reads_raise_confidence_above_asr_prob():
    r = parse_eam(words_with_probs(eam()))
    # 0.95 per read -> 1 - 0.05^2 = 0.9975 for agreeing characters
    assert min(c["confidence"] for c in r["chars"]) >= 0.99


def test_announced_length_digit_by_digit_and_words():
    body = BODY[:20]
    r = parse_eam(eam(body=body, length_phrase="two zero"))
    assert r["declared_length"] == 20 and r["length_source"] == "announced"
    assert r["body"] == body and "length_mismatch" not in r["flags"]
    r = parse_eam(eam(body=body, length_phrase="twenty"))
    assert r["declared_length"] == 20


def test_trailer_length_used_when_header_length_missed():
    body = BODY[:24]
    text = eam(body=body, length_phrase="two four").replace(
        "Message of two four characters follows.", "Message follows."
    )
    r = parse_eam(text)
    assert r["declared_length"] == 24


def test_more_to_follow():
    r = parse_eam(eam(closing="more"))
    assert r["more_to_follow"] and not r["out"]


# ------------------------------------------------------------ EAM: disagreement
def test_mismatched_reads_lower_confidence_and_flag():
    bad = BODY[:10] + "W" + BODY[11:]  # one substitution in read 2
    text = eam(read2=bad)
    r = parse_eam(words_with_probs(text))
    assert "reads_disagree" in r["flags"]
    c = r["chars"][10]
    assert not c["agree"] and {c["read1"], c["read2"]} == {BODY[10], "W"}
    assert c["confidence"] < 0.5
    assert r["body_masked"][10] == "."
    assert r["confidence"]["low_count"] == 1
    others = [x["confidence"] for i, x in enumerate(r["chars"]) if i != 10]
    assert min(others) > 0.99


def test_disagreement_resolved_by_asr_probability():
    bad = BODY[:10] + "W" + BODY[11:]
    text = eam(read2=bad)
    words = text.split()
    # locate the 11th character word of read 2 and make it low-probability
    i_again = max(i for i, w in enumerate(words) if w == "say")
    idx = i_again + 2 + 10
    r = parse_eam(words_with_probs(text, {idx: 0.2}))
    assert r["body"][10] == BODY[10]


def test_preamble_breaks_tie_in_first_six():
    bad = "Z" + BODY[1:]
    r = parse_eam(eam(read2=bad))  # equal probs, preamble says BODY[0]
    assert r["body"][0] == BODY[0]


def test_dropped_character_in_one_read():
    r = parse_eam(eam(read2=BODY[:15] + BODY[16:]))
    assert r["body"] == BODY
    assert not r["chars"][15]["agree"] and r["chars"][15]["read2"] is None
    assert r["chars"][15]["confidence"] < 0.5


def test_length_mismatch_flag():
    r = parse_eam(eam(body=BODY[:28]))  # 28 chars but no length announced
    assert "length_mismatch" in r["flags"]


# ------------------------------------------------------------ Skyking
SKYKING = (
    "Skyking, skyking, do not answer. Kingfisher, Kingfisher. Time two five. "
    "Authentication Papa Hotel. I say again. Skyking, skyking, do not answer. "
    "Kingfisher, Kingfisher. Time two five. Authentication Papa Hotel. This is Mainsail, out."
)


def test_skyking_extraction():
    r = parse_skyking(SKYKING)
    assert r["type"] == "skyking"
    assert r["codeword"]["value"] == "KINGFISHER" and r["codeword"]["agree"]
    assert r["time"]["value"] == "25" and r["time"]["agree"]
    assert r["authentication"]["value"] == "PH" and r["authentication"]["agree"]
    assert r["callsign"] == "MAINSAIL"
    assert r["flags"] == []


def test_skyking_split_words_and_disagreeing_auth():
    text = SKYKING.replace("Skyking", "Sky king").replace(
        "Authentication Papa Hotel. This", "Authentication Papa Hotel. This", 1
    )
    text = text.replace("Authentication Papa Hotel. This is", "Authentication Papa Echo. This is")
    r = parse_transcript(text)
    assert len(r) == 1 and r[0]["type"] == "skyking"
    a = r[0]["authentication"]
    assert not a["agree"] and a["read1"] == "PH" and a["read2"] == "PE"
    assert "authentication_reads_disagree" in r[0]["flags"]


def test_skyking_invalid_time_flag():
    r = parse_skyking(SKYKING.replace("two five", "seven five"))
    assert "invalid_time" in r["flags"]


# ------------------------------------------------------------ segmentation / partial / malformed
def test_two_messages_in_one_recording():
    text = eam(closing="more") + " " + eam(body=BODY[::-1], callsign="Mainsail") + " " + SKYKING
    recs = parse_transcript(text)
    assert [r["type"] for r in recs] == ["eam", "eam", "skyking"]
    assert recs[1]["body"] == BODY[::-1]


def test_partial_start_mid_message():
    text = eam()
    cut = text[text.index("I say again") :]
    recs = parse_transcript("static noise " + spell(BODY[20:]) + " " + cut)
    assert recs[0]["type"] == "eam"
    assert "partial_start" in recs[0]["flags"]
    assert recs[0]["read2"] == BODY


def test_truncated_after_first_read():
    text = eam()
    text = text[: text.index("I say again")]
    r = parse_transcript(text)[0]
    assert r["body"] == BODY and "single_read" in r["flags"]
    # preamble (heard 3x) still supports the first six characters
    assert all(c["confidence"] < 0.7 for c in r["chars"][6:])
    assert all(c["confidence"] > 0.7 for c in r["chars"][:6])


@pytest.mark.parametrize(
    "text,flag",
    [
        ("", "empty_transcript"),
        ("   ", "empty_transcript"),
        ("hello is anyone there radio check over", "no_procedure_markers"),
    ],
)
def test_malformed_input(text, flag):
    r = parse_transcript(text)
    assert len(r) == 1 and r[0]["type"] == "unknown" and flag in r[0]["flags"]


def test_garbled_header_does_not_crash():
    r = parse_transcript("all stations all stations stand by stand by message follows I say again")[0]
    assert r["type"] == "eam"
    assert "empty_body" in r["flags"] and "missing_callsign" in r["flags"]


def test_test_count():
    r = parse_transcript(
        "This is Mainsail with a test count. Testing one two three four five, "
        "five four three two one. This is Mainsail, out."
    )[0]
    assert r["type"] == "test_count" and r["callsign"] == "MAINSAIL"


# ------------------------------------------------------------ tokens / metrics / rebroadcast
def test_tokenize_joins_and_numbers():
    toks = tokenize([Token("X"), Token("ray"), Token("Sky"), Token("king,"), Token("stand"), Token("by.")])
    assert [t.norm for t in toks] == ["xray", "skyking", "standby"]
    assert parse_number(tokenize("one two zero")) == 120
    assert parse_number(tokenize("one hundred twenty")) == 120
    assert parse_number(tokenize("30")) == 30
    assert parse_number(tokenize("banana")) is None


def test_metrics():
    assert cer("ABCDEF", "ABCDEF") == 0
    assert cer("ABCDEF", "ABXDEF") == pytest.approx(1 / 6)
    m = field_exact_match(
        {"callsign": "Mainsail", "addressee": None}, {"callsign": "MAINSAIL"}, ["callsign", "addressee"]
    )
    assert m == {"callsign": True, "addressee": None}


def test_rebroadcast_linking():
    t0 = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    near = BODY[:5] + "Z" + BODY[6:]
    recs = [
        {"id": "a", "timestamp": t0.isoformat(), "body": BODY, "frequency_khz": 11175},
        {
            "id": "b",
            "timestamp": (t0 + timedelta(seconds=2)).isoformat(),
            "body": near,
            "frequency_khz": 8992,
        },
        {
            "id": "c",
            "timestamp": (t0 + timedelta(minutes=55)).isoformat(),
            "body": BODY,
            "frequency_khz": 11175,
        },
        {
            "id": "d",
            "timestamp": (t0 + timedelta(minutes=5)).isoformat(),
            "body": BODY[::-1],
            "frequency_khz": 4724,
        },
        {"id": "e", "timestamp": (t0 + timedelta(days=3)).isoformat(), "body": BODY, "frequency_khz": 4724},
    ]
    groups = link_rebroadcasts(recs)
    by_ids = sorted(sorted(m["id"] for m in g["members"]) for g in groups)
    assert by_ids == [["a", "b", "c"], ["d"], ["e"]]
    g = next(g for g in groups if len(g["members"]) == 3)
    assert [m["relation"] for m in g["members"]] == ["first", "simulcast", "rebroadcast"]


# ------------------------------------------------ regressions from real off-air ASR (missileer clips)
def test_asr_quirks_followed_hyphen_letters_and_loops():
    body = "QH7JAPKRABXRJCVZ"
    text = eam(body=body, length_phrase="16").replace("characters follows", "characters followed")
    text = text.replace(spell("KRABX"), "K-R-A-B-X")
    text = text.replace("This complete message", "This sweet message")
    text += " New York City," * 40
    r = parse_transcript(text)
    assert len(r) == 1
    r = r[0]
    assert r["declared_length"] == 16
    assert r["read1"] == body and r["read2"] == body
    assert "missing_message_follows" not in r["flags"]


def test_preamble_vote_ignores_stray_word():
    text = eam().replace(f"{spell(BODY[:6])}, stand by.", f"Break {spell(BODY[:6])}, stand by.", 1)
    text = text.replace("Break", "Alpha Alpha", 1)  # stray letters before the first hearing
    r = parse_eam(text)
    assert r["preamble"] == BODY[:6]


def test_partial_start_detected_from_trailer_only():
    trailer = " This complete message of three zero characters. This is Mainsail, out."
    text = spell(BODY[10:]) + trailer
    r = parse_transcript(text)[0]
    assert r["type"] == "eam" and "partial_start" in r["flags"] and r["read1"] == BODY[10:]


def test_garbled_message_follows_and_id_say_again():
    # Real small.en output pattern: "Classic. Follow." and "I'd say again"
    text = eam().replace("Message follows.", "Classic. Follow.").replace("I say again.", "I'd say again")
    r = parse_eam(text)
    assert r["body"] == BODY and r["read1"] == BODY and r["read2"] == BODY
    assert "message_follows_inferred" in r["flags"] and "single_read" not in r["flags"]
    assert r["preamble_reads"] == [BODY[:6]] * 3


def test_bare_say_again_without_i():
    r = parse_eam(eam().replace("I say again.", "say again"))
    assert r["read2"] == BODY


def test_lost_say_again_reads_are_self_split():
    text = eam().replace("I say again.", "")
    r = parse_eam(text)
    assert r["read1"] == BODY and r["read2"] == BODY
    assert "reads_split_inferred" in r["flags"] and "single_read" not in r["flags"]


def test_lost_standby_preamble_repeats_are_stripped():
    text = eam().replace(", stand by.", ".").replace("Message follows.", "")
    r = parse_eam(text)
    assert r["preamble"] == BODY[:6] and "preamble_inferred" in r["flags"]
    assert r["body"] == BODY


def test_garbled_again_variants():
    for variant in ["Iced it again", "again", "I said again"]:
        r = parse_eam(eam().replace("I say again.", variant))
        assert r["read2"] == BODY, variant


def test_unstructured_long_char_run_is_parsed_headless():
    r = parse_transcript("noise noise " + spell(BODY) + " thank you")
    assert r[0]["type"] == "eam" and r[0]["body"] == BODY


def test_short_char_run_stays_unknown():
    r = parse_transcript("radio check Alpha Bravo one two over")
    assert r[0]["type"] == "unknown"


# ---- mid-message start detection
_R1 = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"
_HDR = "All stations, this is Mainsail. Alpha Bravo Charlie Delta Echo Foxtrot stand by. Message follows. "


def _eam(text):
    from decoding.eam_parser import parse_transcript

    recs = [r for r in parse_transcript(text) if r["type"] == "eam"]
    assert recs
    return recs[0]


def test_mid_start_read2_only():
    rec = _eam(f"again. {_R1}. This is Mainsail, out.")
    assert rec["mid_message_start"] == "read2_only" and "mid_message_start" in rec["flags"]


def test_mid_start_tail_of_read1_then_read2():
    rec = _eam(f"Hotel India Juliet. I say again. {_R1}. This is Mainsail, out.")
    assert rec["mid_message_start"] == "in_read1"
    assert rec["read1"] == "HIJ" and rec["read2"] == "ABCDEFGHIJ"


def test_mid_start_body_only():
    rec = _eam("Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango. Out.")
    assert rec["mid_message_start"] == "body_only"


def test_headless_but_preamble_heard_is_not_mid_message():
    rec = _eam(
        f"Alpha Bravo Charlie Delta Echo Foxtrot stand by. Message follows. {_R1}. I say again. {_R1}."
    )
    assert "partial_start" in rec["flags"]
    assert "mid_message_start" not in rec and "mid_message_start" not in rec["flags"]


def test_full_message_is_not_mid_message():
    rec = _eam(f"{_HDR}{_R1}. I say again. {_R1}. This is Mainsail, out.")
    assert "mid_message_start" not in rec["flags"] and "partial_start" not in rec["flags"]


def test_mid_start_kind_helper_edge_cases():
    from decoding.eam_parser import mid_message_start_kind

    assert mid_message_start_kind({"type": "skyking", "body": "X"}) is None
    assert mid_message_start_kind({"type": "eam", "body": "", "flags": []}) is None
    split = {"type": "eam", "body": "ABCDEF", "flags": ["missing_message_follows", "reads_split_inferred"]}
    assert mid_message_start_kind(split) == "in_read1"
