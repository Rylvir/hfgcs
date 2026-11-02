"""Hallucination filters and the proposed Skyking alert gate."""

from decoding.hallucination import (
    filter_records,
    is_routine_traffic,
    skyking_alert_ok,
    suppress_weak_skyking,
)


def _sk(**kw):
    base = {
        "type": "skyking",
        "confidence": 0.0,
        "flags": ["missing_codeword", "missing_time", "missing_authentication"],
        "codeword": {"value": ""},
        "transcript": "Skyking, Skyking, do not answer.",
    }
    base.update(kw)
    return base


def test_prompt_echo_skyking_is_suppressed():
    r = suppress_weak_skyking(_sk())
    assert r["type"] == "other" and "suppressed_skyking" in r["flags"]


def test_long_codeword_is_suppressed():
    r = suppress_weak_skyking(
        _sk(
            confidence=0.4,
            flags=["missing_time", "missing_authentication"],
            codeword={"value": "MIKE SIERRA YANKEE ZULU DELTA DELTA SIERRA"},
        )
    )
    assert r["type"] == "other"


def test_real_short_codeword_kept():
    r = suppress_weak_skyking(
        _sk(
            confidence=0.55,
            flags=[],
            codeword={"value": "BRAVO FORCE"},
            transcript=(
                "Skyking, Skyking, do not answer. Bravo Force. "
                "Time one five. Authentication Alpha Bravo."
            ),
        )
    )
    assert r["type"] == "skyking"


def test_monogram_routine_is_not_alerted():
    rec = {
        "type": "skyking",
        "confidence": 0.0,
        "flags": ["missing_time", "missing_authentication"],
        "codeword": {"value": "MONOGRAM MONOGRAM GET IN BY THE TRAFFIC"},
        "transcript": (
            "Skyking, Skyking, do not answer. Monogram, Monogram, "
            "get in by the traffic. Monogram out."
        ),
    }
    rec = filter_records([rec])[0]
    assert rec["type"] == "other"  # long codeword + missing time/auth
    assert is_routine_traffic(
        {
            "type": "other",
            "transcript": "Monogram, Monogram, standing by for traffic. Monogram out.",
        }
    )
    ok, reason = skyking_alert_ok(rec)
    assert not ok


def test_alert_requires_codeword_and_conf_or_agreement():
    good = _sk(
        confidence=0.4,
        flags=[],
        codeword={"value": "BRAVO FORCE"},
        transcript="Skyking. Bravo Force. Time one five. Authentication Alpha.",
    )
    assert skyking_alert_ok(good)[0]
    low = dict(good, confidence=0.1)
    assert skyking_alert_ok(low) == (False, "unconfirmed_low_conf")
    assert skyking_alert_ok(low, cross_channel=True)[0]


def test_standing_by_for_traffic_is_routine():
    assert is_routine_traffic(
        {"transcript": "This is Hellstorm. Hellstorm. Stand by for traffic."}
    )
    assert is_routine_traffic(
        {"transcript": "Monogram, Monogram, standing by for traffic. Monogram out."}
    )
    assert not is_routine_traffic(
        {
            "transcript": (
                "Skyking, Skyking, do not answer. Bravo Force. "
                "Time one five. Authentication Alpha Bravo."
            )
        }
    )


def test_routine_skyking_is_not_alerted():
    ok, reason = skyking_alert_ok(
        {
            "type": "skyking",
            "confidence": 0.9,
            "flags": [],
            "codeword": {"value": "MONOGRAM"},
            "transcript": "Skyking. Monogram standing by for traffic.",
        }
    )
    assert not ok and reason == "routine_traffic"
