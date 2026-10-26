"""ingest-catalog with the recorder's schema-2 catalog (hfgcs/capture.py save_recording),
plus migrations 0002. Fake data only."""

import contextlib
import importlib.util
import io
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("hfgcs_db_catalog_tests", ROOT / "scripts" / "hfgcs_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(tool, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = tool.main(list(argv))
    return rc, out.getvalue()


def test_schema2_catalog_ingest(tool, tmp_path):
    db = str(tmp_path / "t.db")
    assert run(tool, "--db", db, "init")[0] == 0
    rc, out = run(tool, "--db", db, "ingest-catalog", str(FIX / "recorder_catalog_v2_fake.jsonl"))
    assert rc == 0 and "ingested 2 captures" in out and "fake-rx-a" in out and "fake-rx-b" in out
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("""SELECT c.*, r.code FROM captures c JOIN receivers r ON r.id = c.receiver_id
                          ORDER BY c.start_utc""").fetchall()
    a = rows[0]
    assert a["code"] == "fake-rx-a" and a["frequency_khz"] == 8992.0
    assert a["start_utc"] == "2031-01-02T03:04:05.000Z"      # file start (pre-roll included)
    assert a["end_utc"] == "2031-01-02T03:05:15.000Z"
    assert a["voice_start_s"] == 1.5 and a["voice_start_method"] == "recorder_preroll"
    assert a["snr_db"] == 14.2 and a["noise_floor_db"] == -73.1 and a["sample_rate_hz"] == 8000
    assert a["audio_sha256"] == "a" * 64 and a["duration_sec"] == 70.0
    assert a["audio_path"].endswith("HFGCS_8992kHz_20310102T030406Z_fake-rx-a.wav")
    assert rows[1]["code"] == "fake-rx-b"
    rc, out = run(tool, "--db", db, "ingest-catalog", str(FIX / "recorder_catalog_v2_fake.jsonl"))
    assert "ingested 0 captures (2 already present)" in out


def test_init_applies_migrations_and_migrate_upgrades_v1(tool, tmp_path):
    db = str(tmp_path / "v1.db")
    con = sqlite3.connect(db)
    con.executescript((ROOT / "schema.sql").read_text())   # a v1 database as created before 0002
    con.close()
    rc, out = run(tool, "--db", db, "migrate", "--status")
    assert "schema version 1" in out and "[2, 3]" in out
    rc, out = run(tool, "--db", db, "migrate")
    assert rc == 0 and "applied migrations [2, 3]" in out
    rc, out = run(tool, "--db", db, "migrate")
    assert "already current" in out
    con = sqlite3.connect(db)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    assert {"reviews", "edit_history", "v_gold_labels", "trg_edit_history_no_update"} <= names
    db2 = str(tmp_path / "fresh.db")
    run(tool, "--db", db2, "init")
    assert sqlite3.connect(db2).execute("SELECT max(version) FROM schema_version").fetchone()[0] == 3
