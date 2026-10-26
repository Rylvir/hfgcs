"""`hfgcs_db.py load-decodes`: Decoding's sidecars / decodes.jsonl -> messages.
Fake data only (recorder catalog fixture + parser output on made-up words)."""

import contextlib
import importlib
import importlib.util
import io
import json
import sqlite3
from pathlib import Path

import pytest
from decode_fixtures import BODY_WORDS, SKYKING_TEXT, eam_text, sidecar

from hfgcs import review_db

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
CATALOG = [json.loads(x) for x in (FIX / "recorder_catalog_v2_fake.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("hfgcs_db_load_tests", ROOT / "scripts" / "hfgcs_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(tool, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = tool.main([str(a) for a in argv])
    return rc, out.getvalue()


@pytest.fixture()
def db(tool, tmp_path):
    p = tmp_path / "hfgcs.db"
    run(tool, "--db", p, "init")
    run(tool, "--db", p, "ingest-catalog", FIX / "recorder_catalog_v2_fake.jsonl")
    return p


def write_sidecar(dirpath: Path, doc: dict) -> Path:
    name = Path(doc["recording"]["file"]).name + ".decode.json"
    f = dirpath / name
    f.write_text(json.dumps(doc))
    return f


def q(db, sql, *params):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def test_load_sidecars_and_jsonl_then_link(tool, db, tmp_path):
    side = tmp_path / "recordings"
    side.mkdir()
    write_sidecar(side, sidecar(CATALOG[0], eam_text()))
    jsonl = tmp_path / "decodes.jsonl"
    doc_b = sidecar(CATALOG[1], eam_text() + " " + SKYKING_TEXT + " uh test test", t0=3.0)
    jsonl.write_text(json.dumps(doc_b) + "\n")
    rc, out = run(tool, "--db", db, "load-decodes", side, jsonl)
    assert rc == 0, out
    assert "inserted 3" in out and "recordings 2" in out
    msgs = q(db, """SELECT m.*, c.audio_sha256, r.code FROM messages m JOIN captures c ON c.id = m.capture_id
                    JOIN receivers r ON r.id = m.receiver_id ORDER BY m.id""")
    eam_a = msgs[0]
    assert eam_a["code"] == "fake-rx-a" and eam_a["audio_sha256"] == "a" * 64
    assert eam_a["msg_type"] == "EAM" and eam_a["callsign"] == "FAKECALL"
    assert eam_a["message_text"] == "QXABCDEFGHJKLMNPRSTUVWYZ234567" and eam_a["prefix"] == "QX"
    assert eam_a["heard_utc"] == "2031-01-02T03:04:07.000Z"      # file_start_utc + segment.start_sec (2.0)
    assert eam_a["heard_at_basis"] == "voice_onset" and eam_a["offset_sec"] == 2.0
    assert eam_a["origin"] == "pipeline" and eam_a["frequency_khz"] == 8992.0
    assert eam_a["decoder_version"] == "parser:0.1|asr:small.en|compute_type:int8"
    assert 0.9 < eam_a["confidence"] <= 1.0 and len(json.loads(eam_a["char_confidence"])) == 30
    assert eam_a["transcript"].startswith("all stations")       # single broadcast -> whole ASR text
    sk = [m for m in msgs if m["msg_type"] == "SKYKING"][0]
    assert (sk["skyking_codeword"], sk["skyking_time"], sk["skyking_auth"]) == \
        ("SIERRA TANGO UNIFORM", "15", "PQ")
    assert sk["transcript"] is None                              # several broadcasts in one recording
    assert "link" in out.lower() or "links" in out.lower()
    links = q(db, "SELECT * FROM message_links")
    assert [(lk["message_a"], lk["message_b"], lk["link_kind"]) for lk in links] == [(1, 2, "simulcast")]
    assert len(q(db, "SELECT * FROM message_decodes WHERE applied = 1")) == 3
    # Idempotent: the same files again change nothing.
    rc, out = run(tool, "--db", db, "load-decodes", side, jsonl)
    assert "unchanged 3" in out and "inserted" not in out
    assert len(q(db, "SELECT * FROM messages")) == 3 and len(q(db, "SELECT * FROM message_decodes")) == 3


def test_redecode_updates_untouched_but_keeps_human_edits(tool, db, tmp_path):
    side = tmp_path / "rec"
    side.mkdir()
    write_sidecar(side, sidecar(CATALOG[0], eam_text()))
    jl = tmp_path / "b.jsonl"
    jl.write_text(json.dumps(sidecar(CATALOG[1], eam_text())) + "\n")
    run(tool, "--db", db, "load-decodes", side, jl, "--no-link")
    # A person corrects message 1 in the review app; message 2 is left alone.
    con = review_db.connect(db)
    review_db.save_review(con, 1, fields={"message_text": "QXABCDEFGHJKLMNPRSTUVWYZ234568",
                                          "callsign": "HUMANCALL"}, checked=True, notes=None, editor="tester")
    con.close()
    new_body = BODY_WORDS.replace("hotel", "india")
    write_sidecar(side, sidecar(CATALOG[0], eam_text(new_body), model="medium.en"))
    jl.write_text(json.dumps(sidecar(CATALOG[1], eam_text(new_body), model="medium.en")) + "\n")
    rc, out = run(tool, "--db", db, "load-decodes", side, jl, "--no-link")
    assert "kept_human 1" in out and "updated 1" in out
    m1, m2 = q(db, "SELECT * FROM messages ORDER BY id")
    assert m1["message_text"] == "QXABCDEFGHJKLMNPRSTUVWYZ234568" and m1["callsign"] == "HUMANCALL"
    assert m1["review_status"] == "reviewed"
    assert m1["machine_text"] == "QXABCDEFGIJKLMNPRSTUVWYZ234567"   # newest machine copy kept beside it
    assert m1["decoder_version"] == "parser:0.1|asr:medium.en|compute_type:int8"
    assert m2["message_text"] == "QXABCDEFGIJKLMNPRSTUVWYZ234567" and m2["machine_text"] is None
    d = q(db, "SELECT message_id, applied, machine_text FROM message_decodes ORDER BY id")
    assert [(r["message_id"], r["applied"]) for r in d] == [(1, 1), (2, 1), (1, 0), (2, 1)]
    assert q(db, "SELECT count(*) AS n FROM reviews WHERE checked = 1")[0]["n"] == 1
    # The review page shows the not-applied machine copy.
    con = review_db.connect(db)
    _, recs = review_db.list_recordings(con)
    m = [x for r in recs for x in r["messages"] if x["id"] == 1][0]
    assert m["pending_machine_text"] == "QXABCDEFGIJKLMNPRSTUVWYZ234567"
    con.close()


def test_checked_only_row_is_protected(tool, db, tmp_path):
    side = tmp_path / "rec"
    side.mkdir()
    write_sidecar(side, sidecar(CATALOG[0], eam_text()))
    run(tool, "--db", db, "load-decodes", side, "--no-link")
    con = review_db.connect(db)
    review_db.save_review(con, 1, fields={}, checked=True, notes=None, editor="tester")
    con.close()
    write_sidecar(side, sidecar(CATALOG[0], eam_text(BODY_WORDS.replace("golf", "oscar"))))
    rc, out = run(tool, "--db", db, "load-decodes", side, "--no-link")
    assert "kept_human 1" in out
    assert q(db, "SELECT message_text FROM messages")[0][0] == "QXABCDEFGHJKLMNPRSTUVWYZ234567"


def test_match_by_file_name_create_capture_and_unmatched(tool, tmp_path):
    p = tmp_path / "x.db"
    run(tool, "--db", p, "init")
    side = tmp_path / "rec"
    side.mkdir()
    # Embedded catalog entry, no capture yet: created from the sidecar's recording entry.
    write_sidecar(side, sidecar(CATALOG[0], eam_text()))
    rc, out = run(tool, "--db", p, "load-decodes", side, "--no-link")
    assert "captures_created 1" in out and "inserted 1" in out
    assert q(p, "SELECT snr_peak_db FROM captures")[0][0] == 21.4
    # Bare Decoder.decode() output ({records, ...}) matched only by the sidecar's file name.
    entry = dict(CATALOG[1])
    cap_name = Path(entry["file"]).name
    con = sqlite3.connect(p)
    con.execute("INSERT INTO receivers(code) VALUES ('fake-rx-b')")
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, audio_path)
                   VALUES (2, 11175.0, '2031-01-02T03:04:05.500Z', ?)""", ("recordings/" + cap_name,))
    con.commit()
    con.close()
    bare = {"records": sidecar(entry, SKYKING_TEXT)["broadcasts"], "second_pass": {}, "duration_s": 70.0}
    (side / (cap_name + ".decode.json")).write_text(json.dumps(bare))
    (side / "nothing_here.wav.decode.json").write_text(json.dumps(bare))
    rc, out = run(tool, "--db", p, "load-decodes", side, "--no-link", "--strict")
    assert rc == 1 and "no capture found for 1" in out and "nothing_here" in out
    assert q(p, "SELECT msg_type FROM messages WHERE capture_id = 2")[0][0] == "SKYKING"


def test_requires_migration(tool, tmp_path):
    p = tmp_path / "v1.db"
    con = sqlite3.connect(p)
    con.executescript((ROOT / "schema.sql").read_text())
    con.close()
    with pytest.raises(SystemExit, match="migrate"):
        run(tool, "--db", p, "load-decodes", tmp_path)


def test_peak_snr_backfill_by_migration(tool, tmp_path):
    p = tmp_path / "v2.db"
    con = sqlite3.connect(p)
    con.executescript((ROOT / "schema.sql").read_text())
    con.close()
    mig = importlib.import_module("hfgcs.db_migrations")
    con = sqlite3.connect(p)
    v2_only = [m for m in mig.available() if m[0] <= 2]
    for _version, _name, path in v2_only:   # bring the DB to version 2 only
        for stmt in mig._statements(path.read_text()):
            con.execute(stmt)
        con.execute("INSERT INTO schema_version(version, note) VALUES (?, ?)", (_version, _name))
    con.execute("INSERT INTO receivers(id, code) VALUES (1, 'fake-rx-a')")
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, snr_db, raw_json)
                   VALUES (1, 8992.0, '2031-01-02T03:04:05.000Z', 14.2, ?)""", (json.dumps(CATALOG[0]),))
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, raw_json)
                   VALUES (1, 8992.0, '2031-01-02T04:00:00.000Z', '{"timestamp": "x"}')""")
    con.commit()
    assert mig.apply_migrations(con) == [3]
    assert [r[0] for r in con.execute("SELECT snr_peak_db FROM captures ORDER BY id")] == [21.4, None]

