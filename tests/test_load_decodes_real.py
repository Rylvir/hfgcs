"""`load-decodes` against Decoding's real output on main (#13): the committed two-broadcast
sample decode (docs/decode.schema.json, parser 0.2) and documents built by Decoding's own writer
for the cases its PR lists as differences from docs/database.md section 12."""

import contextlib
import importlib.util
import io
import json
import sqlite3
from pathlib import Path

import pytest
from decode_fixtures import contract_doc, eam_text

from hfgcs import review_db

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SAMPLE = FIX / "decode_sample_two_broadcasts.decode.json"
CATALOG = [json.loads(x) for x in (FIX / "recorder_catalog_v2_fake.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("hfgcs_db_real_tests", ROOT / "scripts" / "hfgcs_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(tool, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = tool.main([str(a) for a in argv])
    return rc, out.getvalue()


def rows(db, sql, *params):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


def dump(db) -> list[str]:
    con = sqlite3.connect(db)
    try:
        return list(con.iterdump())
    finally:
        con.close()


@pytest.fixture()
def db(tool, tmp_path):
    p = tmp_path / "hfgcs.db"
    run(tool, "--db", p, "init")
    return p


def test_real_sample_document(tool, db, tmp_path):
    doc = json.loads(SAMPLE.read_text())
    assert doc["decoder_version"].split(";")[1] == "parser=0.2"
    assert doc["asr"]["decoder_version"] == doc["decoder_version"] and "first_pass" in doc["asr"]
    # Capture created from the catalog fields Decoding copies into `recording`.
    rc, out = run(tool, "--db", db, "load-decodes", "--strict", "--no-link", SAMPLE)
    assert rc == 0 and "inserted 2" in out and "captures_created 1" in out, out
    msgs = rows(db, "SELECT * FROM messages ORDER BY start_s")
    assert len(msgs) == 2
    for m, b in zip(msgs, doc["broadcasts"], strict=True):
        assert m["msg_type"] == "EAM" and m["broadcast_id"] == b["broadcast_id"]
        assert m["transcript"] == b["transcript"] and m["start_s"] == b["start_s"]
        assert m["end_s"] == b["end_s"] and m["confidence"] == b["confidence"]
        assert m["callsign"] == b["callsign"] and m["message_text"] == b["body"]
        assert m["decoder_version"] == doc["decoder_version"] and m["decoded_utc"] == doc["decoded_at"]
        assert m["asr_model"] == doc["asr"]["model"] and m["review_status"] == "needs_review"
    assert [m["start_s"] for m in msgs] == [0.0, 204.52]
    assert msgs[0]["callsign"] == "CUPWAY" and msgs[1]["callsign"] is None
    assert msgs[0]["partial"] == 1 and msgs[1]["partial"] == 0          # partial_start flag
    assert msgs[1]["heard_utc"] == "2026-09-23T17:36:17.520Z"           # file_start_utc + 204.52 s
    cap = rows(db, "SELECT * FROM captures")[0]
    assert cap["audio_sha256"] == doc["recording"]["sha256"] and cap["audio_path"] == doc["recording"]["file"]
    before = dump(db)
    rc, out = run(tool, "--db", db, "load-decodes", "--strict", "--no-link", SAMPLE)
    assert rc == 0 and "unchanged 2" in out and dump(db) == before

    # A person corrects one message and checks the other; a different re-decode keeps both.
    con = review_db.connect(db)
    review_db.save_review(con, msgs[1]["id"], fields={"message_text": "SJVSJV9CLNXX"}, checked=True,
                          notes="by ear", editor="tester")
    review_db.save_review(con, msgs[0]["id"], fields={}, checked=True, notes=None, editor="tester")
    con.close()
    redo = json.loads(SAMPLE.read_text())
    redo["decoded_at"] = "2026-10-27T01:00:00Z"
    for b, body in zip(redo["broadcasts"], ("A4REDECODED", "SJREDECODED"), strict=True):
        b["body"], b["transcript"] = body, b["transcript"] + " redecoded"
    f = tmp_path / "redo.decode.json"
    f.write_text(json.dumps(redo))
    rc, out = run(tool, "--db", db, "load-decodes", "--strict", "--no-link", f)
    assert rc == 0 and "kept_human 2" in out, out
    after = rows(db, "SELECT * FROM messages ORDER BY start_s")
    assert [m["message_text"] for m in after] == [msgs[0]["message_text"], "SJVSJV9CLNXX"]
    assert [m["transcript"] for m in after] == [m["transcript"] for m in msgs]
    assert [m["machine_text"] for m in after] == ["A4REDECODED", "SJREDECODED"]
    assert [m["review_status"] for m in after] == ["reviewed", "reviewed"]
    pending = rows(db, "SELECT * FROM message_decodes WHERE applied = 0 ORDER BY message_id")
    assert [p["machine_text"] for p in pending] == ["A4REDECODED", "SJREDECODED"]
    assert all(p["decoded_at"] == "2026-10-27T01:00:00Z" for p in pending)


def test_contract_differences_listed_by_decoding(tool, db, tmp_path):
    run(tool, "--db", db, "ingest-catalog", FIX / "recorder_catalog_v2_fake.jsonl")
    rec = dict(CATALOG[0], truncated=True)
    text = eam_text() + " this is fakecall radio check test count one two three four five out"
    doc = contract_doc(rec, text)
    eam, rchk = doc["broadcasts"]
    # A radio check after a closed broadcast; its callsign falls back to the station.
    assert (eam["type"], rchk["type"]) == ("eam", "radio_check")
    assert [r["role"] for r in rchk["callsigns_all"]] == ["station"] and rchk["callsign"]
    assert "truncated" in rchk["flags"] and "truncated" not in eam["flags"]
    # The recording's own hash differs from the catalog's: Decoding keeps the catalog one aside.
    doc["recording"]["catalog_sha256"] = doc["recording"]["sha256"]
    doc["recording"]["sha256"] = "f" * 64
    doc["recording"]["path"] = "/elsewhere/" + Path(rec["file"]).name
    # A colliding broadcast id gets a #n suffix.
    eam["broadcast_id"] = rchk["broadcast_id"].split("@")[0] + "@2.0#2"
    side = tmp_path / "x.decode.json"
    side.write_text(json.dumps(doc))
    rc, out = run(tool, "--db", db, "load-decodes", "--strict", "--no-link", "--no-create-captures", side)
    assert rc == 0 and "inserted 2" in out, out
    a, b = rows(db, "SELECT * FROM messages ORDER BY start_s")
    assert a["broadcast_id"].endswith("#2") and a["import_key"] == "decode:" + a["broadcast_id"]
    assert b["msg_type"] == "RADIOCHECK" and b["broadcast_type"] == "directed"
    assert b["callsign"] == rchk["callsign"] and b["partial"] == 1 and b["review_status"] == "needs_review"
    assert a["partial"] == 0 and a["capture_id"] == b["capture_id"] == 1
    rc, out = run(tool, "--db", db, "load-decodes", "--strict", "--no-link", "--no-create-captures", side)
    assert "unchanged 2" in out


def test_capture_found_by_catalog_sha256_alone(tool, db):
    run(tool, "--db", db, "ingest-catalog", FIX / "recorder_catalog_v2_fake.jsonl")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        entry = {"sha256": "f" * 64, "catalog_sha256": CATALOG[1]["sha256"]}
        cap = tool._find_capture(con, entry, Path("unrelated.json"))
        assert cap is not None and cap["audio_sha256"] == CATALOG[1]["sha256"]
        assert tool._find_capture(con, {"sha256": "f" * 64}, Path("unrelated.json")) is None
    finally:
        con.close()
