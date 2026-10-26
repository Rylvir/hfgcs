"""Decoding's text_similarity as seen by the Database tool (scripts/hfgcs_db.py,
docs/database.md sections 3.1-3.2). Uses their resolve_similarity, unmodified."""

import importlib.util
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def db():
    path = ROOT / "scripts" / "hfgcs_db.py"
    spec = importlib.util.spec_from_file_location("hfgcs_db_for_decoding_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tool_picks_decoding_similarity(db):
    for mode in ("auto", "decoding"):
        fn, method = db.resolve_similarity(mode, db.LINK_DEFAULTS["link.min_known"])
        assert method == db.DECODING_METHOD == "decoding:text_similarity"
        assert fn("QKR4TZ7HAB", "QKR4TZ7HAB") == 1.0
    _, method = db.resolve_similarity("builtin", 0.5)
    assert method == db.BUILTIN_METHOD


def test_equal_length_substitutions_match_builtin(db):
    fn, _ = db.resolve_similarity("decoding", 0.5)
    rng = random.Random(7)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    for _ in range(300):
        a = "".join(rng.choice(alphabet) for _ in range(30))
        b = list(a)
        for i in rng.sample(range(30), rng.randint(0, 4)):
            b[i] = rng.choice(alphabet + ".?")
        b = "".join(b)
        ours, theirs = fn(a, b), db.builtin_similarity(a, b, 0.5)
        assert (ours is None) == (theirs is None)
        if theirs is not None:
            assert ours >= theirs - 1e-12  # spec: never below the built-in on equal lengths
            if "." not in b and "?" not in b:
                assert ours == pytest.approx(theirs)  # substitutions only: equal to the built-in


def test_unequal_lengths_against_link_threshold(db):
    fn, _ = db.resolve_similarity("decoding", 0.5)
    threshold = db.LINK_DEFAULTS["link.threshold"]
    m30 = "QKR4TZ7HAB2CD3EF4GH5JK6LM7NP8Q"
    assert fn(m30, m30[:-1]) == pytest.approx(1 - 1 / 30) and fn(m30, m30[:-1]) >= threshold
    two_dropped = m30[:10] + m30[12:]
    assert fn(m30, two_dropped) == pytest.approx(1 - 2 / 30) and fn(m30, two_dropped) >= threshold
    assert fn(m30, m30[:20]) < threshold  # truncated copy is not merged
    with pytest.raises(TypeError):
        fn(m30, None)
