"""Score the shadow squelch against what the decoder found in the live recordings.

The live squelch records and the shadow squelch only logs the captures it would have made
(docs/squelch.md). The decode worker goes through the live recordings, so for each message it
found we know when it was on the air. This checks whether the shadow squelch would have
recorded each one. It also splits the live minutes by what the decoder made of them (a
message, voice without one, no voice, not decoded yet) and shows how much of each the shadow
squelch would have kept::

    python -m hfgcs.shadow_score --catalog hfgcs_catalog.jsonl --db data/hfgcs.db \\
        [--since 2026-10-30T18:00Z] [--list]

It only reads; the database is opened read-only. The recorders never import this module, so
changing it restarts nothing on deploy.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .shadow import default_shadow_path

# A message is kept when the shadow captures cover all of it except this much (decoder word
# times and squelch frames do not line up exactly); a quarter of a very short message.
EDGE_TOLERANCE_S = 2.0
# Messages that ended less than this before the run are not scored: a shadow capture still open
# is not in the file yet, and none stays open longer than max_record_seconds (900 s by default).
GRACE_S = 16 * 60.0
DEFAULT_MIN_CONFIDENCE = 0.5
MIN_SCHEMA = 3  # messages.start_s / end_s (migration 0003)


class ScoreError(Exception):
    """A database this tool cannot use (an older schema, or not an hfgcs database)."""


KEPT, SPLIT, CUT, MISSED = "kept", "split", "cut", "missed"
OUTCOMES = (KEPT, SPLIT, CUT, MISSED)
# What the decoder made of a live recording. Without the decode state file the last three
# cannot be told apart and are reported together as NO_MESSAGE.
MESSAGE, VOICE, NO_VOICE, PENDING = "a message", "voice, no message", "no voice", "not decoded yet"
NO_MESSAGE = "no message found"
# The types the summary line is about; the table shows every type.
KEY_TYPES = ("EAM", "EAM_8888", "SKYKING")

MESSAGE_SQL = """
SELECT m.id, m.msg_type, m.callsign, m.confidence,
       coalesce(m.start_s, m.offset_sec) AS start_s, m.end_s,
       c.start_utc AS cap_start, c.audio_sha256 AS sha, c.audio_path AS file,
       c.frequency_khz AS khz, r.code AS rx,
       (m.corrected = 1 OR m.origin <> 'pipeline' OR m.review_status = 'reviewed'
        OR EXISTS (SELECT 1 FROM edit_history h WHERE h.message_id = m.id)
        OR EXISTS (SELECT 1 FROM reviews v WHERE v.message_id = m.id AND v.checked = 1)) AS human
FROM messages m
JOIN captures c ON c.id = m.capture_id
JOIN receivers r ON r.id = c.receiver_id
WHERE c.external_source IS NULL AND m.review_status <> 'rejected' AND c.end_utc >= ?
ORDER BY c.start_utc, start_s
"""


def _epoch(text: str) -> float:
    """UTC timestamp or day (``2026-10-30``, ``2026-10-30T18:00Z``); no zone means UTC."""
    t = str(text).strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    d = datetime.fromisoformat(t)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _key(rx, khz) -> tuple[str, float]:
    try:
        return (str(rx or "?"), round(float(khz or 0), 1))
    except (TypeError, ValueError):
        return (str(rx or "?"), 0.0)


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(e, dict):
                yield e


def load_live(catalog: Path) -> list[dict]:
    """The live recordings: one per schema-2 catalog line, spanning the whole file."""
    rows = []
    for e in _jsonl(catalog):
        start = e.get("file_start_utc") or e.get("start_utc")
        try:
            s, end = _epoch(start), _epoch(e["end_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        pre = e.get("preroll_sec")
        rows.append({"key": _key(e.get("receiver_id"), e.get("frequency_khz")), "s": s, "e": end,
                     "sha": e.get("sha256"), "file": e.get("file"),
                     "state_key": e.get("sha256") or (f"file:{e['file']}" if e.get("file") else None),
                     "preroll": float(pre) if isinstance(pre, (int, float)) and pre > 0 else None})
    return rows


def load_shadow(path: Path) -> tuple[list[dict], set[str]]:
    """The shadow squelch's captures (``start`` is where its activity began) and the
    "<shadow> shadow vs <live> live" pairs named in the file."""
    rows, algos = [], set()
    for e in _jsonl(path):
        try:
            s, end = _epoch(e["start_utc"]), _epoch(e["end_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"key": _key(e.get("receiver_id"), e.get("frequency_khz")), "s": s, "e": end,
                     "reason": e.get("close_reason") or "?"})
        algos.add(f"{e.get('algorithm')} shadow vs {e.get('live_algorithm')} live")
    return rows, algos


def connect_ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_messages(con: sqlite3.Connection, since: float | None) -> tuple[list[dict], int]:
    """Messages in our own recordings, placed in time (capture start + offset); rejected ones
    are left out. Returns them and how many could not be placed (no offset)."""
    try:
        version = con.execute("SELECT max(version) FROM schema_version").fetchone()[0] or 0
    except sqlite3.OperationalError:   # no schema_version table: not an hfgcs database
        version = 0
    if version < MIN_SCHEMA:
        raise ScoreError(f"database schema is version {version}; this needs {MIN_SCHEMA}. "
                         "Run `scripts/hfgcs_db.py migrate` first.")
    # a day of margin: the string compare only narrows the query, the exact cut is below
    floor = _iso(since - 86400)[:10] if since is not None else ""
    rows, unplaced = [], 0
    for r in con.execute(MESSAGE_SQL, (floor,)):
        if r["start_s"] is None:
            unplaced += 1
            continue
        t0 = _epoch(r["cap_start"]) + float(r["start_s"])
        t1 = _epoch(r["cap_start"]) + float(r["end_s"]) if r["end_s"] is not None else t0
        rows.append({"id": r["id"], "type": r["msg_type"], "callsign": r["callsign"],
                     "conf": r["confidence"], "human": bool(r["human"]), "key": _key(r["rx"], r["khz"]),
                     "t0": t0, "t1": max(t0, t1), "sha": r["sha"], "file": r["file"],
                     "offset": float(r["start_s"])})
    return rows, unplaced


def default_decode_state() -> Path:
    """Where the decode worker keeps ``decode-state.json`` (decoding/worker.py)."""
    base = os.environ.get("HFGCS_DECODE_STATE_DIR")
    if not base:
        home = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
        base = os.path.join(home, "hfgcs")
    return Path(base) / "decode-state.json"


def load_decode_state(path: Path) -> dict[str, str]:
    """Catalog key (sha256, else ``file:<path>``) -> the worker's status for that recording."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = data.get("clips") if isinstance(data, dict) else None
    if not isinstance(clips, dict):
        raise ValueError("no 'clips' object")
    return {k: str(v.get("status")) for k, v in clips.items() if isinstance(v, dict)}


def _parts(t0: float, t1: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted((max(a, t0), min(b, t1)) for a, b in spans if a < t1 and b > t0)


def _union(parts: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in parts:   # sorted by start
        if b > end:
            total += b - max(a, end)
            end = b
    return total


def classify(t0: float, t1: float, spans: list[tuple[float, float]]) -> tuple[str, float, int]:
    """How the shadow captures (``spans``) cover a message on the air from t0 to t1:
    (outcome, share covered, number of captures holding a real part of it).

    kept: all of it, in one capture; split: all of it, over several captures; cut: only part;
    missed: none."""
    dur = t1 - t0
    if dur <= 0:   # no end time: a point
        n = sum(1 for a, b in spans if a <= t0 <= b)
        return (KEPT if n else MISSED), (1.0 if n else 0.0), n
    tol = min(EDGE_TOLERANCE_S, 0.25 * dur)
    parts = _parts(t0, t1, spans)
    covered = _union(parts)
    if covered <= 0:
        return MISSED, 0.0, 0
    n = sum(1 for a, b in parts if b - a > tol)
    if dur - covered <= tol:
        return (SPLIT if n > 1 else KEPT), covered / dur, n
    return CUT, covered / dur, n


def _median(values, default: float = 0.0) -> float:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else default


def score(live: list[dict], shadow: list[dict], messages: list[dict], states: dict[str, str] | None,
          *, since: float | None = None, now: float | None = None,
          min_confidence: float = DEFAULT_MIN_CONFIDENCE, preroll: float | None = None) -> dict:
    """Score the shadow captures against the decoded messages and the live recordings.

    A shadow capture would have started its file ``preroll`` seconds before its activity, like
    the live one (default: the median ``preroll_sec`` of that channel's live recordings).
    The window starts at ``since``, else at the first shadow capture, and ends GRACE_S
    before ``now``."""
    now = time.time() if now is None else now
    cutoff = now - GRACE_S
    by_key_live = defaultdict(list)
    for r in live:
        by_key_live[r["key"]].append(r)
    all_pre = _median([r["preroll"] for r in live])
    pre = {k: (preroll if preroll is not None else _median([r["preroll"] for r in rs], all_pre))
           for k, rs in by_key_live.items()}
    spans = defaultdict(list)
    first_shadow = {}
    for r in sorted(shadow, key=lambda r: r["s"]):
        a = r["s"] - (preroll if preroll is not None else pre.get(r["key"], all_pre))
        spans[r["key"]].append((a, r["e"]))
        first_shadow.setdefault(r["key"], a)
    start = since if since is not None else (min(first_shadow.values()) if first_shadow else None)
    report = {"start": start, "start_from": "since" if since is not None else "first shadow capture",
              "cutoff": cutoff, "min_confidence": min_confidence, "states": states is not None,
              "types": defaultdict(Counter), "messages": [], "low_confidence": 0, "too_recent": 0,
              "captures": defaultdict(lambda: {"n": 0, "v1_s": 0.0, "v2_s": 0.0}),
              "channels": defaultdict(lambda: {"found": 0, "kept": 0, "not_kept": 0,
                                               "v1_s": 0.0, "v2_s": 0.0}),
              "shadow_only": {"n": 0, "s": 0.0}, "voice_skipped": []}
    if start is None:
        return report

    # messages
    live_by_sha = {r["sha"]: r for r in live if r["sha"]}
    with_message = set()   # ids of live recordings holding a counted message
    for m in messages:
        if m["t0"] < start:
            continue
        if m["t1"] > cutoff:
            report["too_recent"] += 1
            continue
        counted = m["human"] or m["conf"] is None or m["conf"] >= min_confidence
        outcome, share, n = classify(m["t0"], m["t1"], spans.get(m["key"], []))
        early = m["key"] not in first_shadow or m["t0"] < first_shadow[m["key"]]
        report["messages"].append({**m, "outcome": outcome, "share": share, "n": n,
                                   "counted": counted, "early": early})
        if not counted:
            report["low_confidence"] += 1
            continue
        report["types"][m["type"]][outcome] += 1
        ch = report["channels"][m["key"]]
        ch["found"] += 1
        ch["kept" if outcome == KEPT else "not_kept"] += 1
        cap = live_by_sha.get(m["sha"]) if m["sha"] else None
        if cap is None:   # no sha256: the live recording of that channel holding its start
            cap = next((r for r in by_key_live.get(m["key"], []) if r["s"] <= m["t0"] <= r["e"]), None)
        if cap is not None:
            with_message.add(id(cap))

    # live recordings that ended inside the window; one that began before it counts from its start
    for r in live:
        if r["e"] <= start or r["e"] > cutoff:
            continue
        if id(r) in with_message:
            cat = MESSAGE
        elif states is None:
            cat = NO_MESSAGE
        else:
            status = states.get(r["state_key"]) if r["state_key"] else None
            cat = {"no_voice": NO_VOICE, "decoded": VOICE}.get(status, PENDING)
        s = max(r["s"], start)
        v1 = r["e"] - s
        v2 = _union(_parts(s, r["e"], spans.get(r["key"], [])))
        for bucket in (report["captures"][cat], report["channels"][r["key"]]):
            bucket["v1_s"] += v1
            bucket["v2_s"] += v2
        report["captures"][cat]["n"] += 1
        if cat == VOICE and v2 <= 0:
            report["voice_skipped"].append(r)

    # shadow captures with no live recording at all
    for key, ss in spans.items():
        lv = [(r["s"], r["e"]) for r in by_key_live.get(key, [])]
        for a, b in ss:
            if a >= start and b <= cutoff and not _parts(a, b, lv):
                report["shadow_only"]["n"] += 1
                report["shadow_only"]["s"] += b - a
    return report


def _type_order(t: str) -> tuple:
    return (KEY_TYPES.index(t) if t in KEY_TYPES else len(KEY_TYPES), t)


def _minutes(seconds: float) -> str:
    m = seconds / 60
    return f"{m:.0f}" if m >= 10 else f"{m:.1f}"


def _where(key: tuple[str, float]) -> str:
    return f"{key[0]} {key[1]:g}"


def _message_line(m: dict) -> str:
    conf = "?" if m["conf"] is None else f"{m['conf']:.2f}"
    notes = [f"{m['t1'] - m['t0']:.0f} s", f"confidence {conf}"]
    if m["human"]:
        notes.append("checked by a person")
    notes.append(f"{m['share']:.0%} covered" + (f" over {m['n']} captures" if m["n"] > 1 else ""))
    if m["early"]:
        notes.append("before the first shadow capture on this channel")
    tag = m["outcome"] if m["counted"] else f"{m['outcome']}, low confidence"
    return (f"    {tag}: {_iso(m['t0'])} {_where(m['key'])} {m['type']} {m['callsign'] or '-'} "
            f"({', '.join(notes)}); listen: {m['file'] or '?'} at {m['offset']:.1f} s")


def render(report: dict, algos: set[str], *, show_list: bool = False) -> str:
    out = []
    say = out.append
    head = ", ".join(sorted(algos)) or "shadow vs live"
    if report["start"] is None:
        return f"{head}: the shadow file has no captures yet, so there is nothing to score.\n"
    how = ("from --since" if report["start_from"] == "since"
           else "from the first shadow capture; --since sets it")
    if report["cutoff"] <= report["start"]:
        return (f"{head}: nothing to score yet. The window starts at {_iso(report['start'])} ({how}), and "
                f"only what ended more than {GRACE_S / 60:.0f} min ago is scored.\n")
    say(f"{head}. Scored {_iso(report['start'])} .. {_iso(report['cutoff'])} ({how}).")

    say("")
    say(f"Messages the decoder found in the live recordings (confidence >= {report['min_confidence']:g}, "
        "or checked by a person), and whether the shadow squelch would have recorded them:")
    if report["types"]:
        say(f"  {'type':12s} {'found':>6s} " + " ".join(f"{o:>6s}" for o in OUTCOMES))
        for t in sorted(report["types"], key=_type_order):
            c = report["types"][t]
            say(f"  {t:12s} {sum(c.values()):6d} " + " ".join(f"{c[o]:6d}" for o in OUTCOMES))
        say("  kept: all of it in one capture; split: all of it, over several; cut: only part; missed: none.")
    else:
        say("  none yet")
    not_scored = []
    if report["low_confidence"]:
        not_scored.append(f"{report['low_confidence']} below the confidence floor")
    if report["too_recent"]:
        not_scored.append(f"{report['too_recent']} that ended less than {GRACE_S / 60:.0f} min ago")
    if not_scored:
        say("  Not scored: " + "; ".join(not_scored) + ".")
    early = [m for m in report["messages"] if m["counted"] and m["outcome"] != KEPT and m["early"]]
    if early:
        say(f"  {len(early)} of those not kept were on the air before the shadow squelch's first capture on "
            "their channel (or it has none there). Either that recorder was not running it yet (set --since "
            "to when it was), or it is deaf on that channel.")
    if show_list:
        for m in report["messages"]:
            if m["outcome"] != KEPT:
                say(_message_line(m))

    say("")
    say("Live recordings by what the decoder made of them, and the minutes the shadow squelch would have "
        "recorded of them:")
    cats = report["captures"]
    order = [MESSAGE, VOICE, NO_VOICE, PENDING] if report["states"] else [MESSAGE, NO_MESSAGE]
    say(f"  {'decoder found':20s} {'recordings':>10s} {'live min':>9s} {'shadow min':>10s}")
    total = {"n": 0, "v1_s": 0.0, "v2_s": 0.0}
    for cat in order:
        c = cats.get(cat, {"n": 0, "v1_s": 0.0, "v2_s": 0.0})
        for k in total:
            total[k] += c[k]
        say(f"  {cat:20s} {c['n']:10d} {c['v1_s'] / 60:9.1f} {c['v2_s'] / 60:10.1f}")
    say(f"  {'all':20s} {total['n']:10d} {total['v1_s'] / 60:9.1f} {total['v2_s'] / 60:10.1f}")
    if not report["states"]:
        say("  No decode state file, so recordings with no voice and ones not decoded yet are counted "
            "together (--decode-state).")
    so = report["shadow_only"]
    say(f"  Shadow captures with no live recording at all (no audio to check): {so['n']}, "
        f"{so['s'] / 60:.1f} min.")
    if show_list:
        for r in report["voice_skipped"]:
            say(f"    voice, no message, none of it in a shadow capture: {_iso(r['s'])} {_where(r['key'])} "
                f"({r['e'] - r['s']:.0f} s); listen: {r['file'] or '?'}")

    say("")
    say("Per receiver and channel:")
    say(f"  {'receiver':28s} {'kHz':>8s} {'messages':>8s} {'kept':>5s} {'not kept':>8s} "
        f"{'live min':>9s} {'shadow min':>10s}")
    for key, c in sorted(report["channels"].items()):
        say(f"  {key[0][:28]:28s} {key[1]:8g} {c['found']:8d} {c['kept']:5d} {c['not_kept']:8d} "
            f"{c['v1_s'] / 60:9.1f} {c['v2_s'] / 60:10.1f}")

    say("")
    types = report["types"]
    found = sum(sum(types[t].values()) for t in KEY_TYPES if t in types)
    lost = sum(types[t][o] for t in KEY_TYPES if t in types for o in (CUT, MISSED))
    split = sum(types[t][SPLIT] for t in KEY_TYPES if t in types)
    if not found:
        verdict = "The decoder has not found an EAM or a Skyking in this window yet; score again later."
    elif lost:
        verdict = (f"The shadow squelch would have cut or missed {lost} of the {found} EAM/Skyking messages "
                   "the decoder found. Listen to them (--list) before switching.")
    else:
        verdict = (f"The shadow squelch would have kept every EAM and Skyking the decoder found ({found}"
                   + (f", {split} of them split over more than one capture)." if split else ")."))
    if total["v1_s"] > 0:
        verdict += (f" It would have recorded {_minutes(total['v2_s'])} of the {_minutes(total['v1_s'])} "
                    f"live minutes ({total['v2_s'] / total['v1_s']:.0%}).")
    say("Summary: " + verdict)
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hfgcs.shadow_score", description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", default="hfgcs_catalog.jsonl", help="the live recorder catalog")
    ap.add_argument("--shadow", help="shadow file (default: <catalog stem>.squelch-shadow.jsonl beside it)")
    ap.add_argument("--db", default=os.environ.get("HFGCS_DB"),
                    help="review database with the decoded messages (default: $HFGCS_DB)")
    ap.add_argument("--decode-state",
                    help="the decode worker's decode-state.json, to tell no-voice recordings from ones "
                         "not decoded yet (default: the worker's, if it exists)")
    ap.add_argument("--since", help="start of the scored window, UTC (2026-10-30 or 2026-10-30T18:00Z); "
                                    "default: the first shadow capture")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                    help="machine decodes below this are not scored (default %(default)s; "
                         "ones a person checked always are)")
    ap.add_argument("--preroll", type=float,
                    help="seconds a shadow capture's file would start before its activity "
                         "(default: the live recordings' median preroll_sec)")
    ap.add_argument("--list", action="store_true",
                    help="list each message not kept, and voice recordings the shadow squelch would skip")
    args = ap.parse_args(argv)

    def fail(msg: str) -> int:
        print(msg, file=sys.stderr)
        return 2

    catalog = Path(args.catalog)
    shadow_path = Path(args.shadow) if args.shadow else default_shadow_path(catalog)
    if not args.db:
        return fail("no database: pass --db or set HFGCS_DB")
    for p in (catalog, shadow_path, Path(args.db)):
        if not p.is_file():
            return fail(f"no such file: {p}")
    try:
        since = _epoch(args.since) if args.since else None
    except ValueError:
        return fail(f"--since: not a UTC day or time: {args.since!r}")
    state_path = Path(args.decode_state) if args.decode_state else default_decode_state()
    states = None
    if args.decode_state or state_path.is_file():
        try:
            states = load_decode_state(state_path)
        except (OSError, ValueError) as exc:
            if args.decode_state:
                return fail(f"cannot read the decode state {state_path}: {exc}")
            print(f"note: ignoring the decode state {state_path} ({exc})", file=sys.stderr)
    shadow, algos = load_shadow(shadow_path)
    try:
        con = connect_ro(Path(args.db))
        try:
            messages, unplaced = load_messages(con, since if since is not None else
                                               min((r["s"] for r in shadow), default=None))
        finally:
            con.close()
    except ScoreError as exc:
        return fail(str(exc))
    except sqlite3.DatabaseError as exc:   # also OperationalError: cannot open it
        return fail(f"cannot read {args.db}: {exc}")
    report = score(load_live(catalog), shadow, messages, states, since=since,
                   min_confidence=args.min_confidence, preroll=args.preroll)
    sys.stdout.write(render(report, algos, show_list=args.list))
    if unplaced:
        print(f"({unplaced} message(s) in our recordings have no time offset and were not scored.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
