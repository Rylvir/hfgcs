#!/usr/bin/env python3
"""HFGCS message database tool (Python stdlib only).

Subcommands
  init            create/upgrade a database from schema.sql, then apply migrations/
  migrate         apply pending migrations/NNNN_*.sql (0002: review + edit history)
  import-neet     import a NEET INTEL-format CSV (DAY,DATE,UTC,CALLSIGN,PR,MESSAGE,Q,LN,
                  SPECIFICATION,E6,NOTES,SOURCE,SET)
  export-neet     export messages in the same CSV format
  ingest-catalog  load the recorder's hfgcs_catalog.jsonl into `captures` (schema 2 or legacy)
  link            find simulcast/rebroadcast copies and rebuild clusters
  query           run a NEET-style query string (e.g. "2601&C30&G1&PR=CZ")
  import-missileer  upsert missileer.watch's CC0 archive.json (URL or file; read-only GET)
  import-daily-counts  load hand-entered external daily EAM counts from a CSV (no fetching)
  settings        list or set tunables stored in the DB (e.g. link.threshold)

Column names and the query-token ideas follow NEET INTEL's public data; this code
is original and does not reuse theirs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import inspect
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema.sql"
NEET_COLUMNS = ["DAY", "DATE", "UTC", "CALLSIGN", "PR", "MESSAGE", "Q", "LN",
                "SPECIFICATION", "E6", "NOTES", "SOURCE", "SET"]
DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
WILDCARDS = set(".?")          # unreadable / unknown characters
NO_BODY = {"_", "%"}


# --------------------------------------------------------------------------- db
def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def cmd_init(args) -> int:
    con = connect(args.db)
    con.executescript(Path(args.schema).read_text(encoding="utf-8"))
    con.commit()
    _migrations().apply_migrations(con)
    v = con.execute("SELECT max(version) FROM schema_version").fetchone()[0]
    print(f"initialised {args.db} (schema version {v}, sqlite {sqlite3.sqlite_version})")
    return 0


def _migrations():
    """hfgcs/db_migrations.py (stdlib only), imported from the repo root."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("hfgcs.db_migrations")


def cmd_migrate(args) -> int:
    """Apply pending migrations/NNNN_*.sql to an existing database (no-op when current)."""
    con = connect(args.db)
    mig = _migrations()
    if args.status:
        print(f"schema version {mig.current_version(con)}; pending: "
              f"{[v for v, _, _ in mig.pending(con)] or 'none'}")
        return 0
    applied = mig.apply_migrations(con)
    print(f"applied migrations {applied}" if applied else "schema already current")
    print(f"schema version {mig.current_version(con)}")
    return 0


# -------------------------------------------------------------------- NEET CSV
def iso_from_neet(date: str, utc: str) -> str:
    d = dt.datetime.strptime(f"{date} {utc}", "%Y.%m.%d %H:%M")
    return d.strftime("%Y-%m-%dT%H:%M:00Z")


def classify(row: dict) -> tuple[str, str, str | None]:
    """Return (msg_type, broadcast_type, addressee) from a NEET row."""
    msg, spec, call = row["MESSAGE"].strip(), row["SPECIFICATION"].strip(), row["CALLSIGN"].strip()
    if msg == "%":
        return "NIL", "unknown", None
    if call.upper() == "SKYKING":
        return "SKYKING", "broadcast", None
    if msg == "_":
        return "OTHER", "callsign_only", None
    m = re.match(r"FOR\s+(.+)$", spec, re.I)
    if m:
        return "EAM", "directed", m.group(1).strip()
    if spec.upper().startswith("DISREGARD"):
        return "EAM", "disregard", None
    return "EAM", "broadcast", None


def cmd_import_neet(args) -> int:
    con = connect(args.db)
    receiver_id = None
    if args.receiver:
        receiver_id = ensure_receiver(con, args.receiver, kind="livestream")
    seen = Counter()
    inserted = skipped = 0
    warnings = []
    with open(args.csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in NEET_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"error: missing columns {missing}", file=sys.stderr)
            return 2
        for lineno, row in enumerate(reader, start=2):
            row = {k: (row.get(k) or "") for k in NEET_COLUMNS}
            try:
                heard = iso_from_neet(row["DATE"], row["UTC"])
            except ValueError:
                warnings.append(f"line {lineno}: bad DATE/UTC {row['DATE']!r} {row['UTC']!r}")
                skipped += 1
                continue
            day = DAYS[dt.date.fromisoformat(heard[:10]).weekday()]
            if row["DAY"] and row["DAY"].upper() != day:
                warnings.append(f"line {lineno}: DAY {row['DAY']} != computed {day}")
            ln = int(row["LN"]) if row["LN"].strip().isdigit() else None
            if ln is not None and row["MESSAGE"] not in NO_BODY and ln != len(row["MESSAGE"]):
                warnings.append(f"line {lineno}: LN {ln} != len(MESSAGE) {len(row['MESSAGE'])}")
            msg_type, btype, addressee = classify(row)
            raw = "\x1f".join(row[c] for c in NEET_COLUMNS)
            seen[raw] += 1
            key = "neet:" + hashlib.sha1(raw.encode()).hexdigest()[:20] + f"#{seen[raw]}"
            cur = con.execute(
                """INSERT OR IGNORE INTO messages
                   (heard_utc, msg_type, broadcast_type, callsign, prefix, message_text, length,
                    addressee, qualifier, specification, e6_tail, notes, source, set_label,
                    receiver_id, origin, review_status, import_key, heard_at_basis, preamble,
                    external_source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'neet_import','reviewed',?,'log_minute',?,
                           'neetintel')""",
                (heard, msg_type, btype, row["CALLSIGN"] or None, row["PR"] or None,
                 row["MESSAGE"], ln, addressee, row["Q"] or None,
                 row["SPECIFICATION"],            # raw, '' kept so export is exact
                 row["E6"] or None, row["NOTES"] or None, row["SOURCE"] or None,
                 row["SET"] or None, receiver_id, key,
                 row["MESSAGE"][:6] if msg_type == "EAM" and len(row["MESSAGE"]) >= 6 else None))
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1
    con.commit()
    for w in warnings[:20]:
        print("warning:", w, file=sys.stderr)
    print(f"imported {inserted} rows, skipped {skipped} (duplicates/bad), {len(warnings)} warnings")
    return 0


def cmd_export_neet(args) -> int:
    con = connect(args.db)
    where, params = ("", [])
    if args.query:
        where, params = neet_query_sql(args.query, alias="")
        where = "WHERE _id IN (SELECT id FROM v_messages WHERE " + where + ")"
    if args.origin:
        where += (" AND " if where else "WHERE ") + "_id IN (SELECT id FROM messages WHERE origin = ?)"
        params.append(args.origin)
    rows = con.execute(f"SELECT * FROM v_neet_export {where} ORDER BY _heard_utc, _id", params)
    out = open(args.out, "w", newline="", encoding="utf-8") if args.out != "-" else sys.stdout
    w = csv.writer(out, lineterminator="\n")
    w.writerow(NEET_COLUMNS)
    n = 0
    for r in rows:
        w.writerow([r[c] for c in NEET_COLUMNS])
        n += 1
    if out is not sys.stdout:
        out.close()
    print(f"exported {n} rows to {args.out}", file=sys.stderr)
    return 0


# ------------------------------------------------------------- scanner catalog
def ensure_receiver(con, code: str, kind: str = "other") -> int:
    con.execute("INSERT OR IGNORE INTO receivers(code, kind) VALUES (?, ?)", (code, kind))
    return con.execute("SELECT id FROM receivers WHERE code = ?", (code,)).fetchone()[0]


def to_z(ts: dt.datetime) -> str:
    ts = ts.astimezone(dt.timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def _utc(value: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t


def catalog_entry_to_capture(e: dict) -> dict:
    """Map one catalog line to `captures` columns. Two formats are accepted:

    * schema 2 (hfgcs/capture.py ``save_recording``, the recorder on main):
      ``start_utc`` (activity onset), ``file_start_utc`` (start of pre-roll),
      ``end_utc``, ``frequency_khz``, ``mode``, ``receiver_id``, ``file``,
      ``sha256``, ``audio_rate``, ``duration_sec``, ``preroll_sec``,
      ``snr_db_peak``, ``snr_db_mean``, ``noise_floor_db``, ``truncated``, ``notes``.
      The capture spans the whole file (``file_start_utc`` .. ``end_utc``) and
      ``voice_start_s`` = ``preroll_sec`` (seconds into the WAV where activity starts).
    * legacy (hfgcs_scanner.py): ``{timestamp, frequency_khz, file, duration_sec,
      noise_floor_db, notes}``; ``timestamp`` is the END of the capture.
    """
    if "end_utc" in e and ("start_utc" in e or "file_start_utc" in e):
        start = _utc(e.get("file_start_utc") or e["start_utc"])
        end = _utc(e["end_utc"])
        dur = e.get("duration_sec")
        dur = float(dur) if dur is not None else (end - start).total_seconds()
        snr = e.get("snr_db_mean", e.get("snr_db"))
        return {
            "start": start, "end": end, "duration": dur, "receiver_code": e.get("receiver_id"),
            "mode": e.get("mode") or "USB", "snr_db": snr, "noise_floor_db": e.get("noise_floor_db"),
            "sha256": e.get("sha256"), "sample_rate_hz": e.get("audio_rate"),
            "voice_start_s": e.get("preroll_sec"), "notes": e.get("notes"),
        }
    end = _utc(e["timestamp"])
    dur = float(e.get("duration_sec") or 0)
    return {
        "start": end - dt.timedelta(seconds=dur), "end": end, "duration": dur, "receiver_code": None,
        "mode": e.get("mode") or "USB", "snr_db": e.get("snr_db"), "noise_floor_db": e.get("noise_floor_db"),
        "sha256": None, "sample_rate_hz": None, "voice_start_s": None, "notes": e.get("notes"),
    }


def cmd_ingest_catalog(args) -> int:
    """Load the recorder's catalog JSONL into `captures` (both formats, see
    catalog_entry_to_capture). Idempotent: rows are keyed by the audio SHA-256
    (from the entry, else hashed from the file) and by the own-capture unique index."""
    con = connect(args.db)
    receivers: dict[str, int] = {}

    def rid_for(code):
        code = code or args.receiver
        if code not in receivers:
            receivers[code] = ensure_receiver(con, code, kind=args.kind)
        return receivers[code]

    base = Path(args.recordings_root) if args.recordings_root else Path(args.catalog).parent
    n = dup = 0
    with open(args.catalog, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            e = json.loads(line)
            c = catalog_entry_to_capture(e)
            rel = e.get("file")
            f = None
            if rel:
                f = Path(rel) if Path(rel).is_absolute() else base / rel
                if not f.is_file() and (base / Path(rel).name).is_file():
                    f = base / Path(rel).name   # catalog written with record_dir-relative paths
            sha = c["sha256"] or (hashlib.sha256(f.read_bytes()).hexdigest() if f and f.is_file() else None)
            cur = con.execute(
                """INSERT OR IGNORE INTO captures(receiver_id, frequency_khz, mode, start_utc, end_utc,
                       duration_sec, noise_floor_db, snr_db, audio_path, audio_sha256, sample_rate_hz,
                       voice_start_s, voice_start_method, source_catalog, raw_json, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid_for(c["receiver_code"]), float(e["frequency_khz"]), c["mode"], to_z(c["start"]),
                 to_z(c["end"]), c["duration"], c["noise_floor_db"], c["snr_db"], rel, sha,
                 c["sample_rate_hz"], c["voice_start_s"],
                 "recorder_preroll" if c["voice_start_s"] is not None else None,
                 f"{Path(args.catalog).name}:{lineno}", line.strip(), c["notes"]))
            n += cur.rowcount
            dup += 1 - cur.rowcount
    con.commit()
    who = ", ".join(sorted(receivers)) or args.receiver
    print(f"ingested {n} captures ({dup} already present) for receiver(s) {who}")
    return 0


# ------------------------------------------------------------------ dedup/link
# Built-in defaults; the `settings` table overrides them and CLI flags override both.
LINK_DEFAULTS = {"link.simulcast_sec": 120.0, "link.window_hours": 48.0,
                 "link.threshold": 0.90, "link.min_known": 0.5, "link.similarity": "auto",
                 "link.max_len_diff": 2}
BUILTIN_METHOD = "builtin:wildcard_hamming:v1"
DECODING_METHOD = "decoding:text_similarity"


def load_settings(con) -> dict:
    vals = dict(LINK_DEFAULTS)
    try:
        for k, v in con.execute("SELECT key, value FROM settings"):
            if k in vals:
                vals[k] = type(LINK_DEFAULTS[k])(v)
    except sqlite3.OperationalError:
        pass
    return vals


def builtin_similarity(a: str, b: str, min_known: float = 0.5) -> float | None:
    """Wildcard-aware Hamming similarity for equal-length strings; None = not comparable."""
    if len(a) != len(b) or not a:
        return None
    compared = matches = 0
    for x, y in zip(a, b, strict=True):  # equal lengths checked above
        if x in WILDCARDS or y in WILDCARDS:
            continue
        compared += 1
        matches += x == y
    if compared < max(6, min_known * len(a)):
        return None
    return matches / compared


similarity = builtin_similarity   # backwards-compatible name


def resolve_similarity(mode: str, min_known: float):
    """Return (fn(a, b) -> float|None, method_name).

    mode 'auto' uses decoding.similarity.text_similarity when importable, else the
    built-in; 'decoding' requires it; 'builtin' never imports it. Expected interface:

        text_similarity(a: str, b: str) -> float | None
          a, b : str, normalised copies (upper-case, no spaces/tabs/hyphens); '.' and
                 '?' are wildcards; lengths may differ by up to link.max_len_diff
          returns a float in [0.0, 1.0] (1.0 = same content), or None when the two
                 copies cannot be compared; raises TypeError on non-str (a caller bug);
                 pure, deterministic, no I/O. See docs/database.md section 3.1.
    """
    if mode not in ("auto", "builtin", "decoding"):
        raise ValueError(f"unknown similarity mode {mode!r}")
    builtin = (lambda a, b: builtin_similarity(a, b, min_known)), BUILTIN_METHOD
    if mode == "builtin":
        return builtin
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from decoding.similarity import text_similarity  # provided by Decoding (optional)
    except ImportError:
        if mode == "decoding":
            raise
        return builtin

    try:
        takes_min_known = "min_known" in inspect.signature(text_similarity).parameters
    except (TypeError, ValueError):
        takes_min_known = False
    kwargs = {"min_known": min_known} if takes_min_known else {}

    def checked(a, b):
        v = text_similarity(a, b, **kwargs)
        if v is None:
            return None
        v = float(v)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"text_similarity returned {v}, outside [0, 1]")
        return v
    return checked, DECODING_METHOD


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def consensus(texts: list[str]) -> str:
    """Per-position majority over the copies of the modal length (ties -> longer)."""
    lengths = Counter(len(t) for t in texts)
    modal = max(lengths, key=lambda n: (lengths[n], n))
    out = []
    for chars in zip(*[t for t in texts if len(t) == modal], strict=True):
        known = [c for c in chars if c not in WILDCARDS]
        out.append(Counter(known).most_common(1)[0][0] if known else ".")
    return "".join(out)


def cmd_link(args) -> int:
    con = connect(args.db)
    cfg = load_settings(con)
    for flag, key in (("simulcast_sec", "link.simulcast_sec"), ("window_hours", "link.window_hours"),
                      ("threshold", "link.threshold"), ("min_known", "link.min_known"),
                      ("similarity", "link.similarity"), ("max_len_diff", "link.max_len_diff")):
        if getattr(args, flag) is not None:
            cfg[key] = getattr(args, flag)
    sim, method = resolve_similarity(cfg["link.similarity"], cfg["link.min_known"])
    simulcast_sec, window_s, threshold = (cfg["link.simulcast_sec"], cfg["link.window_hours"] * 3600,
                                          cfg["link.threshold"])
    # unequal lengths are only compared by a method that scores them (Decoding's);
    # the built-in Hamming score is defined for equal lengths only
    max_len_diff = int(cfg["link.max_len_diff"]) if method == DECODING_METHOD else 0
    rows = con.execute(
        """SELECT id, heard_utc, text_norm, prefix, receiver_id, frequency_khz,
                  (julianday(heard_utc) - 2440587.5) * 86400.0 AS t
           FROM messages
           WHERE msg_type IN ('EAM','SKYKING') AND review_status <> 'rejected'
             AND text_norm NOT IN ('', '_', '%')
           ORDER BY heard_utc, id""").fetchall()
    links = []
    j0 = 0
    for i, a in enumerate(rows):            # rows are in time order; b is always the earlier copy
        while rows[j0]["t"] < a["t"] - window_s:
            j0 += 1
        la = len(a["text_norm"])
        for b in rows[j0:i]:
            if abs(len(b["text_norm"]) - la) > max_len_diff:
                continue
            if a["prefix"] and b["prefix"] and "?" not in a["prefix"] + b["prefix"] \
                    and a["prefix"] != b["prefix"]:
                continue
            s = sim(b["text_norm"], a["text_norm"])
            if s is None or s < threshold:
                continue
            d = a["t"] - b["t"]
            kind = "simulcast" if d <= simulcast_sec else "rebroadcast"
            links.append((b["id"], a["id"], kind, round(d, 1), round(s, 4)))
    # all similarity calls are done before any write, so a failing similarity
    # function leaves the DB untouched. Only our own unconfirmed links are rebuilt;
    # source-asserted ('external:*') and human-confirmed links are kept.
    con.execute("DELETE FROM message_links WHERE confirmed = 0 AND method IN (?, ?)",
                (BUILTIN_METHOD, DECODING_METHOD))
    con.execute("UPDATE messages SET cluster_id = NULL")
    con.execute("DELETE FROM message_clusters")
    con.executemany(
        """INSERT OR IGNORE INTO message_links(message_a, message_b, link_kind, delta_sec,
               similarity, method) VALUES (?,?,?,?,?,?)""", [link + (method,) for link in links])
    # clusters from all links (incl. confirmed/manual ones kept from earlier runs)
    dsu, sim_dsu = DSU(), DSU()
    for a, b, kind in con.execute("SELECT message_a, message_b, link_kind FROM message_links"):
        dsu.union(a, b)
        if kind == "simulcast":
            sim_dsu.union(a, b)
    info = {r["id"]: r for r in rows}
    members = defaultdict(list)
    for mid in list(dsu.p):
        members[dsu.find(mid)].append(mid)
    for ids in members.values():
        ids = [i for i in ids if i in info]
        if len(ids) < 2:
            continue
        ids.sort(key=lambda i: (info[i]["heard_utc"], i))
        texts = [info[i]["text_norm"] for i in ids]
        canon = consensus(texts)
        airings = len({sim_dsu.find(i) for i in ids})
        cur = con.execute(
            """INSERT INTO message_clusters(canonical_text, prefix, length, first_heard_utc,
                   last_heard_utc, n_messages, n_airings) VALUES (?,?,?,?,?,?,?)""",
            (canon, info[ids[0]]["prefix"], len(canon),
             info[ids[0]]["heard_utc"], info[ids[-1]]["heard_utc"], len(ids), airings))
        con.executemany("UPDATE messages SET cluster_id = ? WHERE id = ?",
                        [(cur.lastrowid, i) for i in ids])
    con.commit()
    kinds = Counter(link[2] for link in links)
    ncl = con.execute("SELECT count(*) FROM message_clusters").fetchone()[0]
    print(f"compared {len(rows)} messages with {method} (max_len_diff {max_len_diff}, "
          f"simulcast<={simulcast_sec:g}s, "
          f"window {window_s / 3600:g}h, threshold {threshold:g}): {len(links)} links {dict(kinds)}, "
          f"{ncl} clusters")
    return 0



# ------------------------------------------------------------ missileer.watch
MISSILEER_URL = "https://data.missileer.watch/public/archive.json"
GRADE_TO_TYPE = {"message": "EAM", "unconfirmed": "EAM", "fragment": "EAM",
                 "radio-check": "RADIOCHECK", "auth": "AUTH", "traffic": "TRAFFIC", "8888": "EAM_8888"}


def _z(ts):
    """'2026-10-26T09:33:22.000Z' -> '2026-10-26T09:33:22Z' (drop zero millis)."""
    if not ts:
        return None
    return ts.replace(".000Z", "Z")


def _json(v):
    return None if v is None else json.dumps(v, separators=(",", ":"))


def fetch_missileer(location: str, etag: str | None):
    """Read-only GET with If-None-Match; returns (data|None, etag, last_modified, status)."""
    if not re.match(r"https?://", location):
        return json.loads(Path(location).read_text(encoding="utf-8")), None, None, "ok"
    req = urllib.request.Request(location, headers={
        "User-Agent": "hfgcs-db-importer/0.1 (+https://github.com/Rylvir/hfgcs)",
        "Accept": "application/json"})
    if etag:
        req.add_header("If-None-Match", etag)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r), r.headers.get("ETag"), r.headers.get("Last-Modified"), "ok"
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return None, etag, None, "not_modified"
        raise


def cmd_import_missileer(args) -> int:
    con = connect(args.db)
    prev = con.execute("SELECT etag FROM import_runs WHERE source='missileer' AND status='ok' "
                       "AND location=? ORDER BY id DESC LIMIT 1", (args.source,)).fetchone()
    data, etag, lastmod, status = fetch_missileer(args.source, None if args.force or not prev else prev[0])
    if status == "not_modified":
        con.execute("INSERT INTO import_runs(source, location, etag, status, rows_seen, rows_upserted) "
                    "VALUES ('missileer', ?, ?, 'not_modified', 0, 0)", (args.source, etag))
        con.commit()
        print("missileer: not modified since last import")
        return 0
    if data.get("license") != "CC0-1.0":
        print(f"refusing: feed license is {data.get('license')!r}, expected CC0-1.0", file=sys.stderr)
        return 3
    seen = up = links = 0
    for e in data.get("messages", []):
        if e.get("collapsed"):
            continue
        mtype = GRADE_TO_TYPE.get(e.get("grade"), "OTHER")
        review = ("reviewed" if e.get("corrected") else
                  "needs_review" if e.get("unconfirmed") or e.get("ear_queue") else "auto")
        text = e.get("text") or ""
        recs = e.get("receptions") or [{"id": e["id"], "heard_at": e.get("heard_at"),
                                        "heard_at_basis": e.get("heard_at_basis"),
                                        "clip_start_at": e.get("clip_start_at"), "station": e.get("station"),
                                        "frequency_khz": e.get("frequency_khz"), "first": True,
                                        "coreception": False, "clip_url": e.get("clip_url")}]
        first_id = None
        for r in recs:
            seen += 1
            station = r.get("station") or e.get("station") or "unknown"
            rid = ensure_receiver(con, f"missileer:{station}", kind="kiwisdr")
            con.execute("UPDATE receivers SET url = coalesce(url, 'https://missileer.watch') WHERE id = ?",
                        (rid,))
            freq = float(r.get("frequency_khz") or e.get("frequency_khz") or 0) or None
            cap_id = None
            if r.get("clip_url") and r.get("clip_start_at") and freq:
                start = _z(r["clip_start_at"])
                dur = r.get("clip_duration_s")
                end = None
                if dur is not None:
                    t0 = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end = to_z(t0 + dt.timedelta(seconds=float(dur)))
                con.execute(
                    """INSERT INTO captures(receiver_id, frequency_khz, start_utc, end_utc, duration_sec,
                           audio_url, voice_start_s, external_source, external_id, classification)
                       VALUES (?,?,?,?,?,?,?,'missileer',?,'voice')
                       ON CONFLICT(external_source, external_id) DO UPDATE SET
                           audio_url=excluded.audio_url, voice_start_s=excluded.voice_start_s,
                           end_utc=excluded.end_utc, duration_sec=excluded.duration_sec""",
                    (rid, freq, start, end, dur, r["clip_url"], r.get("voice_start_s"),
                     r["clip_url"]))
                cap_id = con.execute("SELECT id FROM captures WHERE external_source='missileer' "
                                     "AND external_id=?", (r["clip_url"],)).fetchone()[0]
            heard = _z(r.get("heard_at") or e.get("heard_at"))
            pre = e.get("preamble") or (text[:6] if len(text) >= 6 else None)
            is_first = 1 if r.get("first") else 0
            vals = dict(
                capture_id=cap_id, receiver_id=rid, frequency_khz=freq, heard_utc=heard,
                heard_at_basis=r.get("heard_at_basis") or "unknown",
                msg_type=mtype, callsign=None, prefix=(pre[:2] if pre and "?" not in pre[:2] else None),
                message_text=text, length=e.get("char_count"),
                preamble=pre, preamble_vote=e.get("preamble_vote"),
                announced_length=e.get("announced_length"),
                announced_length_machine=e.get("announced_length_machine"),
                group_count=e.get("group_count"), partial=int(bool(e.get("partial"))),
                malformed=int(bool(e.get("malformed"))), grade=e.get("grade"),
                is_first_reception=is_first, coreception=int(bool(r.get("coreception"))),
                char_times=_json(r.get("char_times")), anchors=_json(r.get("anchors")),
                position_notes=_json(e.get("position_notes") if is_first else None),
                machine_text=e.get("machine_text"), corrected=int(bool(e.get("corrected"))),
                correction_source=e.get("correction_source"), ear_queue=int(bool(e.get("ear_queue"))),
                ear_queue_reason=e.get("ear_queue_reason"), review_status=review,
                origin="missileer_import", external_source="missileer", external_id=e["id"],
                source="missileer.watch (CC0)",
                import_key=f"missileer:{e['id']}:{r.get('id') or station}")
            cols = ", ".join(vals)
            upd = ", ".join(f"{k}=excluded.{k}" for k in vals if k not in ("import_key", "origin"))
            con.execute(f"INSERT INTO messages({cols}) VALUES ({', '.join('?' * len(vals))}) "
                        f"ON CONFLICT(import_key) DO UPDATE SET {upd}", list(vals.values()))
            mid = con.execute("SELECT id FROM messages WHERE import_key=?",
                              (vals["import_key"],)).fetchone()[0]
            up += 1
            if first_id is None:          # receptions are oldest first, first copy leading
                first_id, first_heard = mid, heard
                continue
            a, b = sorted((first_id, mid))
            d = (dt.datetime.fromisoformat(heard.replace("Z", "+00:00")) -
                 dt.datetime.fromisoformat(first_heard.replace("Z", "+00:00"))).total_seconds()
            con.execute("""INSERT OR IGNORE INTO message_links(message_a, message_b, link_kind, delta_sec,
                               similarity, method) VALUES (?,?,?,?,1.0,'external:missileer')""",
                        (a, b, "simulcast" if r.get("coreception") else "rebroadcast", abs(d)))
            links += 1
    con.execute("""INSERT INTO import_runs(source, location, etag, last_modified, generated_at, status,
                       rows_seen, rows_upserted, notes) VALUES ('missileer',?,?,?,?,'ok',?,?,?)""",
                (args.source, etag, lastmod, data.get("generated_at"), seen, up,
                 f"{len(data.get('hidden') or [])} hidden stubs skipped; {links} reception links"))
    con.commit()
    print(f"missileer: {len(data.get('messages', []))} entries, {seen} receptions upserted, "
          f"{links} reception links (generated_at {data.get('generated_at')})")
    return 0


def cmd_import_daily_counts(args) -> int:
    """Hand-entered counts only (e.g. copied by a person from warwatchintel's public page).
    This tool never fetches external count pages."""
    con = connect(args.db)
    n = 0
    with open(args.csv, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                """INSERT INTO external_daily_counts(source, date_utc, metric, count, day_boundary,
                       method, collected_by, url, notes) VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source, date_utc, metric) DO UPDATE SET count=excluded.count,
                       collected_utc=strftime('%Y-%m-%dT%H:%M:%SZ','now'), method=excluded.method,
                       collected_by=excluded.collected_by, notes=excluded.notes""",
                (row["source"], row["date_utc"], row.get("metric") or "eams", int(row["count"]),
                 row.get("day_boundary") or "UTC", "manual",
                 row.get("collected_by"), row.get("url"), row.get("notes")))
            n += 1
    con.commit()
    print(f"loaded {n} external daily counts")
    return 0

def cmd_settings(args) -> int:
    con = connect(args.db)
    if args.key:
        if args.key not in LINK_DEFAULTS:
            print(f"unknown setting {args.key!r}; known: {', '.join(LINK_DEFAULTS)}", file=sys.stderr)
            return 2
        type(LINK_DEFAULTS[args.key])(args.value)  # validate
        if args.key == "link.similarity" and args.value not in ("auto", "builtin", "decoding"):
            print("link.similarity must be auto, builtin or decoding", file=sys.stderr)
            return 2
        con.execute("""INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE
                       SET value=excluded.value, updated_utc=strftime('%Y-%m-%dT%H:%M:%SZ','now')""",
                    (args.key, args.value))
        con.commit()
    for k, v in load_settings(con).items():
        print(f"{k}\t{v}")
    return 0


# ------------------------------------------------------------ NEET-style query
TOKEN_HELP = """\
Tokens joined with '&' (all must match; message-level semantics):
  YYMMDD  date prefix, partials ok (2601 = Jan 2026); '-' = any digit (--06 = every June)
  C30     length 30;  C32,27 = length 32 or 27
  G1      group 1;    G23 = group 2 or 3
  R       not yet reviewed (review_status auto/needs_review)
  S<code> receiver code (e.g. Slocal-hfplus);  S-<code> = exclude receiver
  PR=XX   prefix      CS=NAME  callsign (LIKE, case-insensitive)
  F=11175 frequency kHz   T=EAM|SKYKING|OTHER   E6  has an E-6B tail
  "text"  full-text (trigram, >= 3 chars) over message/callsign/addressee/notes
"""


def neet_query_sql(q: str, alias: str = "") -> tuple[str, list]:
    p = alias + "." if alias else ""
    clauses, params = [], []
    for raw in [t.strip() for t in q.split("&") if t.strip()]:
        tok = raw.rstrip("x") if re.fullmatch(r"[0-9-]+x|C[\d,]+x", raw) else raw
        if re.fullmatch(r"[0-9-]{1,6}", tok):
            clauses.append(f"{p}yymmdd GLOB ?")
            params.append(tok.replace("-", "?") + "*")
        elif m := re.fullmatch(r"C([\d,]+)", tok, re.I):
            ns = [int(n) for n in m.group(1).split(",") if n]
            clauses.append(f"{p}text_length IN ({','.join('?' * len(ns))})")
            params += ns
        elif m := re.fullmatch(r"G([1-5]+)", tok, re.I):
            gs = [int(c) for c in m.group(1)]
            clauses.append(f"{p}group_no IN ({','.join('?' * len(gs))})")
            params += gs
        elif tok.upper() == "R":
            clauses.append(f"{p}review_status IN ('auto','needs_review')")
        elif tok.upper() == "E6":
            clauses.append(f"coalesce({p}e6_tail,'') <> ''")
        elif m := re.fullmatch(r"S(-?)(.+)", tok):
            clauses.append(f"coalesce({p}receiver,'') {'<>' if m.group(1) else '='} ?")
            params.append(m.group(2))
        elif m := re.fullmatch(r"(PR|CS|F|T)=(.+)", tok, re.I):
            k, v = m.group(1).upper(), m.group(2).strip()
            if k == "PR":
                clauses.append(f"{p}prefix = ?")
                params.append(v.upper())
            elif k == "CS":
                clauses.append(f"{p}callsign LIKE ?")
                params.append(v)
            elif k == "F":
                clauses.append(f"abs({p}frequency_khz - ?) < 0.5")
                params.append(float(v))
            else:
                clauses.append(f"{p}msg_type = ?")
                params.append(v.upper())
        elif m := re.fullmatch(r'"(.+)"', tok):
            clauses.append(f"{p}id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?)")
            params.append('"' + m.group(1).replace('"', '""') + '"')
        else:
            raise ValueError(f"unknown query token {raw!r}\n{TOKEN_HELP}")
    return (" AND ".join(clauses) or "1"), params


def cmd_query(args) -> int:
    con = connect(args.db)
    where, params = neet_query_sql(args.q)
    if args.show_sql:
        print(where, params, file=sys.stderr)
    if args.days:
        sql = (f"SELECT date_utc, count(*) AS n, group_concat(DISTINCT prefix) AS prefixes "
               f"FROM v_messages WHERE {where} GROUP BY date_utc ORDER BY date_utc")
    else:
        sql = (f"SELECT heard_utc, callsign, prefix, message_text, text_length, group_no, "
               f"addressee, receiver, frequency_khz FROM v_messages WHERE {where} "
               f"ORDER BY heard_utc LIMIT {int(args.limit)}")
    for r in con.execute(sql, params):
        print("\t".join("" if v is None else str(v) for v in r))
    return 0


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("HFGCS_DB", "hfgcs.db"),
                    help="SQLite file (default: $HFGCS_DB, else ./hfgcs.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--schema", default=str(SCHEMA))
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("migrate", help="apply pending migrations/NNNN_*.sql (e.g. 0002 review tables)")
    s.add_argument("--status", action="store_true", help="only show the version and pending migrations")
    s.set_defaults(fn=cmd_migrate)
    s = sub.add_parser("import-neet")
    s.add_argument("csv")
    s.add_argument("--receiver", help="receiver code to attach (e.g. neet-livestream)")
    s.set_defaults(fn=cmd_import_neet)
    s = sub.add_parser("export-neet")
    s.add_argument("out", help="output CSV path or -")
    s.add_argument("--query", help="NEET-style filter")
    s.add_argument("--origin", choices=["pipeline", "neet_import", "manual"])
    s.set_defaults(fn=cmd_export_neet)
    s = sub.add_parser("ingest-catalog")
    s.add_argument("catalog")
    s.add_argument("--receiver", default="local-hfplus")
    s.add_argument("--kind", default="local_sdr")
    s.add_argument("--recordings-root", help="directory the catalog's 'file' paths are relative to")
    s.set_defaults(fn=cmd_ingest_catalog)
    s = sub.add_parser("link", help="flags override the settings table, which overrides built-ins")
    s.add_argument("--simulcast-sec", type=float, help="default 120 (settings: link.simulcast_sec)")
    s.add_argument("--window-hours", type=float, help="default 48 (settings: link.window_hours)")
    s.add_argument("--threshold", type=float, help="default 0.90 (settings: link.threshold)")
    s.add_argument("--min-known", type=float, help="default 0.5 (settings: link.min_known)")
    s.add_argument("--max-len-diff", type=int,
                   help="default 2 (settings: link.max_len_diff); used only with the decoding similarity")
    s.add_argument("--similarity", choices=["auto", "builtin", "decoding"],
                   help="default auto (settings: link.similarity)")
    s.set_defaults(fn=cmd_link)
    s = sub.add_parser("query", epilog=TOKEN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("q")
    s.add_argument("--days", action="store_true", help="aggregate per day")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--show-sql", action="store_true")
    s.set_defaults(fn=cmd_query)
    s = sub.add_parser("import-missileer")
    s.add_argument("source", nargs="?", default=MISSILEER_URL, help="URL or local archive.json")
    s.add_argument("--force", action="store_true", help="ignore stored ETag")
    s.set_defaults(fn=cmd_import_missileer)
    s = sub.add_parser("import-daily-counts", help="hand-entered CSV only")
    s.add_argument("csv")
    s.set_defaults(fn=cmd_import_daily_counts)
    s = sub.add_parser("settings")
    s.add_argument("key", nargs="?")
    s.add_argument("value", nargs="?")
    s.set_defaults(fn=cmd_settings)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
