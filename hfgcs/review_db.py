"""Data access for the private review app (hfgcs/review_app.py). Stdlib only.

Everything that touches the database lives here so it can be tested without HTTP:
listing recordings with their decoded messages, saving edits with an append-only
history (migration 0002), the checked flag and notes, gold-label export, and safe
resolution of audio paths inside the recordings directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import IO, Iterator, Optional

from . import db_migrations

REVIEW_SCHEMA_VERSION = 2
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".opus": "audio/ogg", ".ogg": "audio/ogg",
               ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
NEET_TIMECARD_URL = "https://neetintel.github.io/dtcp_gallery.html?{yymmdd}"
GOLD_EXPORT_SCHEMA = 1

# field -> (max length or None for int, normaliser)
EDITABLE: dict[str, tuple[Optional[int], str]] = {
    "transcript": (20000, "text"),
    "message_text": (2000, "upper"),
    "callsign": (64, "upper"),
    "prefix": (8, "upper"),
    "addressee": (64, "upper"),
    "length": (None, "int"),
    "msg_type": (32, "code"),
}
# Decoding's label "type" vocabulary (decoding/ingest_missileer.py).
LABEL_TYPE = {"EAM": "eam", "RADIOCHECK": "radio_check", "AUTH": "auth", "TRAFFIC": "traffic",
              "SKYKING": "skyking"}


class ReviewError(ValueError):
    """Invalid input from a reviewer (shown back as HTTP 400)."""


class Conflict(RuntimeError):
    """The message changed since the form was rendered (HTTP 409)."""


def now_z() -> str:
    t = dt.datetime.now(dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def connect(path: str | os.PathLike, *, migrate: bool = False) -> sqlite3.Connection:
    """Open the DB (must exist and have the base schema). Applies pending migrations
    when ``migrate`` is true; otherwise refuses to run without the review tables."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"database not found: {p} (create it with scripts/hfgcs_db.py init)")
    con = sqlite3.connect(str(p), timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 10000")
    if migrate:
        db_migrations.apply_migrations(con)
    elif any(v == REVIEW_SCHEMA_VERSION for v, _, _ in db_migrations.pending(con)):
        con.close()
        raise RuntimeError(f"{p} lacks the review tables; run `scripts/hfgcs_db.py --db {p} migrate`"
                           " (or start the app with --migrate)")
    return con


def neet_timecard_url(date_utc: str) -> str:
    """Link to NEET INTEL's Daily Timecard gallery searched for one UTC day.
    The gallery seeds its search from the query string (``?YYMMDD``); see docs/review-ui.md."""
    d = dt.date.fromisoformat(date_utc[:10])
    return NEET_TIMECARD_URL.format(yymmdd=d.strftime("%y%m%d"))


# ------------------------------------------------------------------ listing
def frequencies(con: sqlite3.Connection) -> list[float]:
    return [r[0] for r in con.execute("SELECT DISTINCT frequency_khz FROM captures "
                                      "WHERE external_source IS NULL ORDER BY frequency_khz")]


def _filters(date: Optional[str], freq: Optional[float], unchecked: bool) -> tuple[str, list]:
    where, params = ["c.external_source IS NULL"], []
    if date:
        dt.date.fromisoformat(date)  # raises ValueError on junk
        where.append("substr(c.start_utc, 1, 10) = ?")
        params.append(date)
    if freq is not None:
        where.append("abs(c.frequency_khz - ?) < 0.05")
        params.append(float(freq))
    if unchecked:
        where.append("""(NOT EXISTS (SELECT 1 FROM messages m WHERE m.capture_id = c.id)
                         OR EXISTS (SELECT 1 FROM messages m LEFT JOIN reviews rv ON rv.message_id = m.id
                                    WHERE m.capture_id = c.id AND coalesce(rv.checked, 0) = 0))""")
    return " AND ".join(where), params


def list_recordings(con: sqlite3.Connection, *, date: Optional[str] = None, freq: Optional[float] = None,
                    unchecked: bool = False, limit: int = 25, offset: int = 0) -> tuple[int, list[dict]]:
    """Own captures newest first, each with its messages (oldest first) and review state."""
    where, params = _filters(date, freq, unchecked)
    total = con.execute(f"SELECT count(*) FROM captures c WHERE {where}", params).fetchone()[0]
    rows = con.execute(
        f"""SELECT c.id, c.start_utc, c.end_utc, c.frequency_khz, c.mode, c.duration_sec, c.snr_db,
                   c.noise_floor_db, c.audio_path, c.audio_sha256, c.voice_start_s, r.code AS receiver
            FROM captures c LEFT JOIN receivers r ON r.id = c.receiver_id
            WHERE {where} ORDER BY c.start_utc DESC, c.id DESC LIMIT ? OFFSET ?""",
        [*params, int(limit), int(offset)]).fetchall()
    recs = [dict(r) for r in rows]
    by_id = {r["id"]: r for r in recs}
    for r in recs:
        r["messages"] = []
    if by_id:
        marks = ",".join("?" * len(by_id))
        for m in con.execute(
                f"""SELECT v.id, v.heard_utc, v.msg_type, v.callsign, v.prefix, v.message_text, v.length,
                           v.text_length, v.addressee, v.group_no, v.confidence, v.review_status,
                           m.capture_id, m.transcript, m.machine_text, m.corrected, m.origin,
                           coalesce(rv.checked, 0) AS checked, rv.notes AS review_notes, rv.reviewer,
                           rv.reviewed_utc,
                           (SELECT coalesce(max(h.id), 0) FROM edit_history h
                            WHERE h.message_id = m.id) AS version
                    FROM v_messages v JOIN messages m ON m.id = v.id
                    LEFT JOIN reviews rv ON rv.message_id = m.id
                    WHERE m.capture_id IN ({marks}) ORDER BY v.heard_utc, v.id""", list(by_id)):
            by_id[m["capture_id"]]["messages"].append(dict(m))
    return total, recs


def message_types(con: sqlite3.Connection) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in con.execute("SELECT code, label FROM message_types ORDER BY code")]


def history(con: sqlite3.Connection, message_id: int) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM edit_history WHERE message_id = ? ORDER BY id", (message_id,))]


def message_version(con: sqlite3.Connection, message_id: int) -> int:
    return con.execute("SELECT coalesce(max(id), 0) FROM edit_history WHERE message_id = ?",
                       (message_id,)).fetchone()[0]


# ------------------------------------------------------------------ editing
def _norm(con: sqlite3.Connection, field: str, raw) -> object:
    limit, kind = EDITABLE[field]
    if raw is None:
        return None
    s = str(raw).replace("\r\n", "\n").strip()
    if s == "":
        if kind == "code":
            raise ReviewError("message type is required")
        return None
    if kind == "int":
        if not s.isdigit() or len(s) > 5:
            raise ReviewError(f"{field} must be a whole number")
        return int(s)
    if limit is not None and len(s) > limit:
        raise ReviewError(f"{field} is longer than {limit} characters")
    if kind == "upper":
        s = " ".join(s.upper().split())
    if kind == "code":
        if con.execute("SELECT 1 FROM message_types WHERE code = ?", (s,)).fetchone() is None:
            raise ReviewError(f"unknown message type {s!r}")
    return s


def _text(v) -> Optional[str]:
    return None if v is None else str(v)


def _log(con, change_id, editor, source, table, message_id, field, old, new) -> None:
    con.execute("""INSERT INTO edit_history(editor, editor_source, change_id, table_name, message_id,
                                            field, old_value, new_value)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (editor, source, change_id, table, message_id, field, _text(old), _text(new)))


def save_review(con: sqlite3.Connection, message_id: int, *, fields: dict, checked: Optional[bool],
                notes: Optional[str], editor: str, editor_source: str = "form",
                expected_version: Optional[int] = None) -> int:
    """Apply a reviewer's edits in one transaction. Every changed value is appended to
    ``edit_history``. Returns the number of changed fields (0 = nothing to do)."""
    editor = (editor or "").strip()[:200]
    if not editor:
        raise ReviewError("editor name is required")
    unknown = set(fields) - set(EDITABLE)
    if unknown:
        raise ReviewError(f"not editable: {sorted(unknown)}")
    if notes is not None and len(notes) > 5000:
        raise ReviewError("notes are longer than 5000 characters")
    with con:
        con.execute("BEGIN IMMEDIATE")  # read-check-write is atomic against other writers
        row = con.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            raise KeyError(message_id)
        if expected_version is not None and message_version(con, message_id) != expected_version:
            raise Conflict(f"message {message_id} was changed by someone else; reload and try again")
        change_id = uuid.uuid4().hex
        n = 0
        updates: dict[str, object] = {}
        for f, raw in fields.items():
            new = _norm(con, f, raw)
            if new != row[f]:
                updates[f] = new
        if "message_text" in updates and not row["corrected"]:
            if row["machine_text"] is None and row["message_text"] is not None:
                updates["machine_text"] = row["message_text"]
            updates["corrected"] = 1
            updates["correction_source"] = "review-ui"
        rev = con.execute("SELECT * FROM reviews WHERE message_id = ?", (message_id,)).fetchone()
        old_checked = bool(rev["checked"]) if rev else False
        old_notes = rev["notes"] if rev else None
        new_notes = old_notes if notes is None else (notes.replace("\r\n", "\n").strip() or None)
        new_checked = old_checked if checked is None else bool(checked)
        if new_checked != old_checked:
            updates["review_status"] = "reviewed" if new_checked else "needs_review"
        for f, v in updates.items():
            _log(con, change_id, editor, editor_source, "messages", message_id, f, row[f], v)
            n += 1
        if updates:
            cols = ", ".join(f"{f} = ?" for f in updates)
            con.execute(f"UPDATE messages SET {cols}, updated_utc = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                        f"WHERE id = ?", [*updates.values(), message_id])
        review_changes = {}
        if new_checked != old_checked:
            review_changes["checked"] = (int(old_checked) if rev else None, int(new_checked))
        if new_notes != old_notes:
            review_changes["notes"] = (old_notes, new_notes)
        for f, (old, new) in review_changes.items():
            _log(con, change_id, editor, editor_source, "reviews", message_id, f, old, new)
            n += 1
        if review_changes:
            ts = now_z()
            con.execute(
                """INSERT INTO reviews(message_id, checked, notes, reviewer, reviewed_utc)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(message_id) DO UPDATE SET checked = excluded.checked, notes = excluded.notes,
                       reviewer = excluded.reviewer, reviewed_utc = excluded.reviewed_utc,
                       updated_utc = strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
                (message_id, int(new_checked), new_notes, editor, ts))
        return n


def create_message(con: sqlite3.Connection, capture_id: int, *, editor: str,
                   editor_source: str = "form") -> int:
    """Add an empty manual message to a capture that has none, so it can be transcribed by ear."""
    editor = (editor or "").strip()[:200]
    if not editor:
        raise ReviewError("editor name is required")
    with con:
        con.execute("BEGIN IMMEDIATE")
        cap = con.execute("SELECT id, start_utc, voice_start_s FROM captures WHERE id = ? "
                          "AND external_source IS NULL", (capture_id,)).fetchone()
        if cap is None:
            raise KeyError(capture_id)
        cur = con.execute(
            """INSERT INTO messages(capture_id, heard_utc, heard_at_basis, msg_type, origin, review_status)
               VALUES (?, ?, 'clip_start', 'EAM', 'manual', 'needs_review')""",
            (capture_id, cap["start_utc"]))
        mid = cur.lastrowid
        _log(con, uuid.uuid4().hex, editor, editor_source, "messages", mid, "created", None,
             f"manual message for capture {capture_id}")
        return mid


# ------------------------------------------------------------------ audio
def resolve_audio(recordings_dir: str | os.PathLike, audio_path: Optional[str]) -> Optional[Path]:
    """Return the audio file for a capture's ``audio_path`` only if it is an existing
    regular file of a known audio type inside ``recordings_dir`` (after resolving
    symlinks and ``..``); otherwise None. Relative paths are tried against the
    recordings dir as stored, then by file name (the recorder stores
    ``<record_dir>/<name>``)."""
    if not audio_path or "\x00" in audio_path:
        return None
    root = Path(recordings_dir).resolve()
    p = Path(audio_path)
    candidates = [p] if p.is_absolute() else [root / p, root / p.name]
    for c in candidates:
        try:
            real = c.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if real.is_file() and real.is_relative_to(root) and real.suffix.lower() in AUDIO_TYPES:
            return real
    return None


# ------------------------------------------------------------------ gold export
def _split(key: str, test_fraction: float = 0.3) -> str:
    """Same deterministic dev/test split rule as decoding/ingest_missileer.split_for."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "dev"


def gold_records(con: sqlite3.Connection,
                 recordings_dir: Optional[str | os.PathLike] = None) -> Iterator[dict]:
    """Checked messages as label records in Decoding's labels.jsonl shape (see
    docs/review-ui.md, "Gold labels")."""
    for r in con.execute("SELECT * FROM v_gold_labels ORDER BY heard_utc, message_id"):
        audio = resolve_audio(recordings_dir, r["audio_path"]) if recordings_dir else None
        typ = LABEL_TYPE.get(r["msg_type"], "other")
        body = r["text_norm"] or None
        source_id = f"msg{r['message_id']}"
        yield {
            "id": f"hfgcs:{source_id}",
            "export_schema": GOLD_EXPORT_SCHEMA,
            "source": "hfgcs-review",
            "source_id": source_id,
            "message_id": r["message_id"],
            "capture_id": r["capture_id"],
            "type": typ,
            "msg_type": r["msg_type"],
            "label_status": "gold",
            "labels": {
                "body": body if r["msg_type"] in ("EAM", "EAM_8888") else None,
                "preamble": r["preamble"] or (body[:6] if body and r["msg_type"] == "EAM" else None),
                "declared_length": r["length"] if r["length"] is not None else r["announced_length"],
                "callsign": r["callsign"],
                "addressee": r["addressee"],
                "prefix": r["prefix"],
                "read1": None,
                "read2": None,
            },
            "transcript": r["transcript"],
            "reference_machine": {"text": r["machine_text"] if r["corrected"] else r["message_text"],
                                  "confidence": r["confidence"]},
            "corrected": bool(r["corrected"]),
            "audio_path": str(audio) if audio else None,
            "audio_file": r["audio_path"],
            "audio_sha256": r["audio_sha256"],
            "heard_at_utc": r["heard_utc"],
            "clip_start_at_utc": r["capture_start_utc"],
            "voice_start_s": r["voice_start_s"],
            "frequency_khz": r["frequency_khz"],
            "receiver": r["receiver"],
            "reviewer": r["reviewer"],
            "reviewed_utc": r["reviewed_utc"],
            "notes": r["review_notes"],
            "split": _split(source_id),
        }


def write_gold_jsonl(con: sqlite3.Connection, out: IO[str], recordings_dir=None) -> int:
    n = 0
    for rec in gold_records(con, recordings_dir):
        out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        n += 1
    return n
