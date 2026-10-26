"""Message-body similarity for the Database tool's rebroadcast linking.

Contract (docs/database.md section 3.1 on branch db/schema-v1):
  text_similarity(a: str, b: str) -> float | None
  * inputs: normalised copies (upper-case, no spaces/hyphens); '.' = unreadable
    and '?' = unknown, both wildcards; lengths may differ (v1 passes equal).
  * returns a float in [0.0, 1.0] (1.0 = same content) or None when the pair
    cannot be compared (too few known characters).
  * pure, deterministic, no I/O or global state, fast (1e4-1e5 calls/run), and
    importing it pulls in nothing heavy (stdlib only, no package imports).

Method ``decoding:text_similarity`` (v1):
  1. Normalise defensively (idempotent on already-normalised input):
     upper-case, drop spaces, tabs and hyphens.
  2. Comparable only if both copies have at least
     ``max(MIN_KNOWN_ABS, MIN_KNOWN_FRAC * max(len))`` known characters (for
     equal lengths: positions where *both* are known), mirroring the DB
     built-in's ``max(6, 0.5 * length)`` rule.
  3. Score = the larger of
       * wildcard Hamming: matches / positions both known (equal lengths only;
         identical to ``builtin:wildcard_hamming:v1``), and
       * wildcard edit similarity: 1 - d / max(len), where d is a banded
         Levenshtein distance in which a wildcard matches anything at cost 0.
     The edit term tolerates a dropped or inserted character (ASR slips shift
     every later position, which Hamming scores as near-random).
  This is the same body comparison ``rebroadcast.link_rebroadcasts`` uses.
"""

from __future__ import annotations

WILDCARDS = frozenset(".?")
MIN_KNOWN_ABS = 6
MIN_KNOWN_FRAC = 0.5
BAND = 8  # extra diagonals beyond the length difference for the banded DP


def normalize(s: str) -> str:
    return s.upper().replace(" ", "").replace("\t", "").replace("-", "")


def _known(s: str) -> int:
    return sum(1 for c in s if c not in WILDCARDS)


def _hamming(a: str, b: str) -> tuple[int, int]:
    """(matches, positions where both are known) for equal-length strings."""
    compared = matches = 0
    for x, y in zip(a, b):  # noqa: B905 (equal lengths checked by caller)
        if x in WILDCARDS or y in WILDCARDS:
            continue
        compared += 1
        matches += x == y
    return matches, compared


def wildcard_edit_distance(a: str, b: str, band: int | None = None) -> int:
    """Levenshtein distance where '.'/'?' on either side match anything.
    With ``band`` set, only cells within |i - j| <= |len(a)-len(b)| + band are
    evaluated (exact whenever the optimal alignment stays inside the band, an
    upper bound otherwise; deterministic either way)."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return max(la, lb)
    w = None if band is None else abs(la - lb) + band
    big = la + lb + 1
    prev = [j if (w is None or j <= w) else big for j in range(lb + 1)]
    for i in range(1, la + 1):
        lo = 1 if w is None else max(1, i - w)
        hi = lb if w is None else min(lb, i + w)
        cur = [big] * (lb + 1)
        cur[0] = i if (w is None or i <= w) else big
        ca = a[i - 1]
        wa = ca in WILDCARDS
        for j in range(lo, hi + 1):
            cb = b[j - 1]
            sub = prev[j - 1] + (0 if (wa or cb in WILDCARDS or ca == cb) else 1)
            cur[j] = min(sub, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev[lb], max(la, lb))


def text_similarity(a: str, b: str) -> float | None:
    """Similarity in [0, 1] of two message copies, or None if not comparable."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("text_similarity expects two str")
    a, b = normalize(a), normalize(b)
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return None
    need = max(MIN_KNOWN_ABS, MIN_KNOWN_FRAC * max(la, lb))
    if la == lb:
        matches, compared = _hamming(a, b)
        if compared < need:
            return None
        ham = matches / compared
        if ham == 1.0:
            return 1.0
    else:
        if min(_known(a), _known(b)) < need:
            return None
        ham = 0.0
    edit = 1.0 - wildcard_edit_distance(a, b, BAND) / max(la, lb)
    score = max(ham, edit)
    return min(1.0, max(0.0, score))
