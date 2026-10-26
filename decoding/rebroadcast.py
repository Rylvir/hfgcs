"""Link airings of the same EAM heard at different times / frequencies /
receivers. About 1 in 3 EAMs is rebroadcast, mostly within 2 h (NEET INTEL
rebroadcast timeline), and simulcast means one airing is heard on several
channels at once."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

from .similarity import text_similarity

MAX_BODY_CER = 0.2  # bodies this close are treated as the same message
WINDOW = timedelta(hours=24)
SIMULCAST_SECONDS = 30  # same airing on several channels


def _ts(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def _confidence(r: dict) -> float:
    """Overall 0-1 confidence; also reads pre-0.2 records (``confidence`` dict)."""
    c = r.get("confidence")
    if isinstance(c, dict):
        return c.get("mean", 0) or 0
    return float(c) if isinstance(c, (int, float)) else 0.0


def link_rebroadcasts(
    records: Sequence[dict], max_cer: float = MAX_BODY_CER, window: timedelta = WINDOW
) -> list[dict]:
    """``records``: dicts with ``id``, ``timestamp`` (UTC ISO), ``body`` and
    optional ``frequency_khz`` / ``receiver``. Returns one group per message:
    ``{"group_id", "body", "members": [{"id", "relation"}]}`` where relation is
    ``first``, ``simulcast`` (within SIMULCAST_SECONDS of the first airing) or
    ``rebroadcast``. Union-find over pairs whose bodies are within ``max_cer``
    of each other (``similarity.text_similarity`` >= 1 - max_cer) and within
    ``window`` in time."""
    recs = [r for r in records if r.get("body")]
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            if abs(_ts(a["timestamp"]) - _ts(b["timestamp"])) > window:
                continue
            ref = max(len(a["body"]), len(b["body"]))
            if abs(len(a["body"]) - len(b["body"])) > max_cer * ref:
                continue
            sim = text_similarity(a["body"], b["body"])  # same comparison the DB tool uses
            if sim is not None and sim >= 1.0 - max_cer:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, r in enumerate(recs):
        groups.setdefault(find(i), []).append(r)
    out = []
    for members in groups.values():
        members.sort(key=lambda r: _ts(r["timestamp"]))
        t0 = _ts(members[0]["timestamp"])
        best = max(
            members,
            key=lambda r: _confidence(r),
        )
        out.append(
            {
                "group_id": "eam-" + t0.strftime("%Y%m%dT%H%M%SZ") + "-" + members[0]["body"][:6],
                "body": best["body"],
                "members": [
                    {
                        "id": m["id"],
                        "timestamp": m["timestamp"],
                        "frequency_khz": m.get("frequency_khz"),
                        "relation": "first"
                        if k == 0
                        else (
                            "simulcast"
                            if (_ts(m["timestamp"]) - t0).total_seconds() <= SIMULCAST_SECONDS
                            else "rebroadcast"
                        ),
                    }
                    for k, m in enumerate(members)
                ],
            }
        )
    out.sort(key=lambda g: g["members"][0]["timestamp"])
    return out
