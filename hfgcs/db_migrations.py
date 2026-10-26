"""Apply numbered SQL migrations (``migrations/NNNN_name.sql``) to an HFGCS SQLite DB.

``schema.sql`` is version 1 and is re-runnable. Each later file ``NNNN_*.sql`` is
applied once, in order, inside a transaction, and recorded in ``schema_version``.
Stdlib only; used by ``scripts/hfgcs_db.py migrate`` and the review app.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "migrations"
_NAME = re.compile(r"^(\d{4})_([A-Za-z0-9_]+)\.sql$")


def available(migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, str, Path]]:
    out = []
    for p in sorted(Path(migrations_dir).glob("*.sql")):
        m = _NAME.match(p.name)
        if m:
            out.append((int(m.group(1)), m.group(2), p))
    return out


def current_version(con: sqlite3.Connection) -> int:
    try:
        v = con.execute("SELECT max(version) FROM schema_version").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    return int(v or 0)


def pending(con: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, str, Path]]:
    done = set()
    try:
        done = {r[0] for r in con.execute("SELECT version FROM schema_version")}
    except sqlite3.OperationalError:
        pass
    return [m for m in available(migrations_dir) if m[0] not in done]


def apply_migrations(con: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[int]:
    """Apply every pending migration; return the versions applied. The base schema
    (version 1, ``schema.sql``) must already be present."""
    if current_version(con) < 1:
        raise RuntimeError("database has no schema; run `scripts/hfgcs_db.py init` first")
    applied = []
    for version, name, path in pending(con, migrations_dir):
        sql = path.read_text(encoding="utf-8")
        con.commit()
        try:
            con.execute("BEGIN")
            # executescript would COMMIT first; run statements one by one inside our transaction.
            for stmt in _statements(sql):
                con.execute(stmt)
            con.execute("INSERT INTO schema_version(version, note) VALUES (?, ?)", (version, name))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        applied.append(version)
    return applied


def _statements(sql: str) -> list[str]:
    """Split a migration into statements with sqlite3.complete_statement, which
    understands strings, comments and CREATE TRIGGER ... BEGIN ... END bodies."""
    out, buf = [], ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip() and not all(ln.strip().startswith("--") or not ln.strip() for ln in buf.splitlines()):
        raise ValueError("incomplete SQL statement at end of migration")
    return out
