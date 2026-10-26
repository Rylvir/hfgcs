import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from decoding.similarity import text_similarity, wildcard_edit_distance

M30 = "QKR4TZ7HMB2XPLA9NDW3CVE6UJS8GY"


def builtin_wildcard_hamming(a, b, min_known=0.5):
    """Reference copy of the DB tool's builtin:wildcard_hamming:v1 (docs/database.md section 3)."""
    if len(a) != len(b) or not a:
        return None
    compared = matches = 0
    for x, y in zip(a, b, strict=True):
        if x in ".?" or y in ".?":
            continue
        compared += 1
        matches += x == y
    if compared < max(6, min_known * len(a)):
        return None
    return matches / compared


def test_identical_is_one():
    assert text_similarity(M30, M30) == 1.0


def test_range_and_type():
    rnd = random.Random(7)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?"
    for _ in range(500):
        n, m = rnd.randint(1, 40), rnd.randint(1, 40)
        a = "".join(rnd.choice(alphabet) for _ in range(n))
        b = "".join(rnd.choice(alphabet) for _ in range(m))
        v = text_similarity(a, b)
        assert v is None or (isinstance(v, float) and 0.0 <= v <= 1.0)


def test_wildcards_match_anything():
    a = M30[:10] + "." + M30[11:20] + "?" + M30[21:]
    assert text_similarity(a, M30) == 1.0
    assert text_similarity(M30, a) == 1.0


def test_substitutions_equal_builtin_on_equal_lengths():
    b = M30[:5] + "A" + M30[6:20] + "Q" + M30[21:]
    assert (
        text_similarity(M30, b) == pytest.approx(builtin_wildcard_hamming(M30, b)) == pytest.approx(28 / 30)
    )


def test_never_below_builtin_on_equal_lengths():
    rnd = random.Random(3)
    for _ in range(300):
        b = list(M30)
        for _ in range(rnd.randint(0, 12)):
            b[rnd.randrange(30)] = rnd.choice("ABC.?7")
        b = "".join(b)
        ours, ref = text_similarity(M30, b), builtin_wildcard_hamming(M30, b)
        assert (ours is None) == (ref is None)
        if ref is not None:
            assert ours >= ref - 1e-12


def test_dropped_character_shift_scores_high_where_hamming_fails():
    shifted = M30[:8] + M30[9:] + "X"  # one char dropped early, filler at the end: same length
    assert builtin_wildcard_hamming(M30, shifted) < 0.3
    assert text_similarity(M30, shifted) >= 0.9


def test_different_lengths_supported():
    v = text_similarity(M30, M30[:-1])
    assert v == pytest.approx(29 / 30)
    assert text_similarity(M30, M30[:20]) == pytest.approx(20 / 30)  # truncated copy: low, not None
    assert text_similarity(M30, M30[:10]) is None  # 10 known < 0.5 * 30


def test_unrelated_messages_score_low():
    assert text_similarity(M30, M30[::-1]) < 0.5


@pytest.mark.parametrize(
    "a,b",
    [
        ("", M30),
        (M30, ""),
        ("." * 30, M30),
        ("?" * 30, "?" * 30),
        ("ABC", "ABC"),  # fewer than 6 known positions
        (M30[:14] + "." * 16, "." * 14 + M30[14:]),  # no position known in both
        (M30[:14] + "." * 16, M30),  # 14 < 0.5 * 30 known
    ],
)
def test_not_comparable_returns_none(a, b):
    assert text_similarity(a, b) is None


def test_min_known_boundary():
    ok = M30[:15] + "." * 15  # exactly 15 = 0.5 * 30 known
    assert text_similarity(ok, M30) == 1.0


def test_normalisation_is_idempotent_and_defensive():
    assert text_similarity("qkr4-tz7h mb2x\tpla9ndw3cve6ujs8gy", M30) == 1.0


def test_deterministic_and_symmetric():
    b = M30[:3] + M30[4:] + "?"
    vals = {text_similarity(M30, b) for _ in range(5)}
    assert len(vals) == 1
    assert text_similarity(M30, b) == text_similarity(b, M30)


def test_rejects_non_str():
    with pytest.raises(TypeError):
        text_similarity(None, M30)  # type: ignore[arg-type]


def test_banded_distance_matches_full_on_small_shifts():
    for b in [M30, M30[1:], "A" + M30, M30[:10] + M30[12:]]:
        assert wildcard_edit_distance(M30, b, band=8) == wildcard_edit_distance(M30, b)


def test_fast_enough():
    rnd = random.Random(1)
    pairs = []
    for _ in range(10_000):
        b = list(M30)
        b[rnd.randrange(30)] = "#"
        pairs.append("".join(b))
    t0 = time.perf_counter()
    for b in pairs:
        text_similarity(M30, b)
    assert time.perf_counter() - t0 < 5.0  # ~1e4 calls; typically well under 1 s


def test_import_is_lightweight():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import decoding.similarity as s; "
        "mods = ('faster_whisper', 'ctranslate2', 'numpy', 'torch', 'decoding.asr'); "
        "heavy = [m for m in mods if m in sys.modules]; "
        "print(heavy)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
