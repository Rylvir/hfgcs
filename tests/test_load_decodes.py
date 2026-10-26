"""`hfgcs_db.py load-decodes` against Decoding's output contract v1 (decoding-output):
sidecars / decodes.jsonl -> messages. Fake data only (recorder catalog fixture + parser
output on made-up words, wrapped by tests/decode_fixtures.py)."""

import contextlib
import importlib
import importlib.util
import io
import json
import sqlite3
from pathlib import Path

import pytest
from decode_fixtures import BODY_WORDS, SKYKING_TEXT, contract_doc, eam_text

from hfgcs import review_db

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
CATALOG = [json.loads(x) for x in (FIX / "recorder_catalog_v2_fake.jsonl").read_text().splitlines()]
V1 = "decoding/0.2.0;parser=0.1;asr=ab12cd34;second_pass=vote3@0.3"
V2 = "decoding/0.3.0;parser=0.1;asr=ef56ab78;second_pass=off"
BODY = "QXABCDEFGHJKLMNPRSTUVWYZ234567"            # what the parser reads from BODY_WORDS
BODY_INDIA = BODY.replace("H", "I")


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
    f = dirpath / (Path(doc["recording"]["path"]).name + ".decode.json")
    f.write_text(json.dumps(doc))
    return f


def q(db, sql, *params):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def test_contract_fields_multi_broadcast_and_link(tool, db, tmp_path):
    side = tmp_path / "recordings"
    side.mkdir()
    write_sidecar(side, contract_doc(CATALOG[0], eam_text()))
    jsonl = tmp_path / "decodes.jsonl"
    doc_b = contract_doc(CATALOG[1], eam_text() + " " + SKYKING_TEXT + " uh test test", t0=3.0)
    assert [b["type"] for b in doc_b["broadcasts"]] == ["eam", "skyking"]
    jsonl.write_text(json.dumps(doc_b) + "\n")
    rc, out = run(tool, "--db", db, "load-decodes", side, jsonl)
    assert rc == 0, out
    assert "inserted 3" in out and "recordings 2" in out
    msgs = q(db, "SELECT m.*, r.code FROM messages m JOIN receivers r ON r.id = m.receiver_id ORDER BY m.id")
    a = msgs[0]
    assert a["code"] == "fake-rx-a" and a["msg_type"] == "EAM" and a["message_text"] == BODY
    assert a["prefix"] == "QX" and a["frequency_khz"] == 8992.0 and a["origin"] == "pipeline"
    assert a["broadcast_id"] == "aaaaaaaaaaaa@2.0" and a["import_key"] == "decode:aaaaaaaaaaaa@2.0"
    assert a["start_s"] == 2.0 and a["offset_sec"] == 2.0 and a["end_s"] > 40
    assert a["heard_utc"] == "2031-01-02T03:04:07.000Z"           # file_start_utc + start_s
    assert a["decoder_version"] == V1 and a["decoded_utc"] == "2031-01-02T04:00:00.000Z"
    assert a["asr_model"] == "small.en" and 0.9 < a["confidence"] <= 1.0
    assert json.loads(a["confidence_detail"])["reads_agree_ratio"] == 1.0
    assert a["callsign"] == "FAKECALL"
    assert json.loads(a["callsigns_all"]) == [{"callsign": "FAKECALL", "role": "opening"},
                                              {"callsign": "FAKECALL", "role": "closing"}]
    assert a["transcript"].startswith("all stations") and a["transcript"].endswith("fakecall out")
    # Two broadcasts in one recording: each gets its own transcript slice.
    b_eam, b_sk = msgs[1], msgs[2]
    assert b_eam["transcript"].endswith("fakecall out") and "skyking" not in b_eam["transcript"]
    assert b_sk["msg_type"] == "SKYKING" and b_sk["transcript"].startswith("skyking skyking")
    assert (b_sk["skyking_codeword"], b_sk["skyking_time"], b_sk["skyking_auth"]) == \
        ("SIERRA TANGO UNIFORM", "15", "PQ")
    assert set(json.loads(b_sk["confidence_detail"])) == {"codeword", "time", "authentication"}
    links = q(db, "SELECT message_a, message_b, link_kind FROM message_links")
    assert [tuple(lk) for lk in links] == [(1, 2, "simulcast")]
    assert len(q(db, "SELECT * FROM message_decodes WHERE applied = 1")) == 3
    # Idempotent: the same files again change nothing.
    rc, out = run(tool, "--db", db, "load-decodes", side, jsonl)
    assert "unchanged 3" in out and "inserted" not in out
    assert len(q(db, "SELECT * FROM messages")) == 3 and len(q(db, "SELECT * FROM message_decodes")) == 3


def test_match_by_sha_and_start_within_2s(tool, db, tmp_path):
    side = tmp_path / "rec"
    side.mkdir()
    write_sidecar(side, contract_doc(CATALOG[0], eam_text()))
    run(tool, "--db", db, "load-decodes", side, "--no-link")
    # Re-decode: segmentation moved by 0.8 s (new broadcast_id string) -> same row, updated.
    write_sidecar(side, contract_doc(CATALOG[0], eam_text(BODY_WORDS.replace("hotel", "india")), t0=2.8,
                                     decoder_version=V2))
    rc, out = run(tool, "--db", db, "load-decodes", side, "--no-link")
    assert "updated 1" in out and "inserted" not in out
    (m,) = q(db, "SELECT * FROM messages")
    assert m["message_text"] == BODY_INDIA and m["start_s"] == 2.8 and m["broadcast_id"] == "aaaaaaaaaaaa@2.8"
    assert m["import_key"] == "decode:aaaaaaaaaaaa@2.0"           # identity of the row is unchanged
    # 3 s away is a different broadcast.
    write_sidecar(side, contract_doc(CATALOG[0], eam_text(), t0=5.9, decoder_version=V2))
    rc, out = run(tool, "--db", db, "load-decodes", side, "--no-link")
    assert "inserted 1" in out and len(q(db, "SELECT * FROM messages")) == 2


def test_redecode_keeps_human_edits(tool, db, tmp_path):
    side = tmp_path / "rec"
    side.mkdir()
    write_sidecar(side, contract_doc(CATALOG[0], eam_text()))
    jl = tmp_path / "b.jsonl"
    jl.write_text(json.dumps(contract_doc(CATALOG[1], eam_text())) + "\n")
    run(tool, "--db", db, "load-decodes", side, jl, "--no-link")
    con = review_db.connect(db)       # a person corrects message 1; message 2 is left alone
    review_db.save_review(con, 1, fields={"message_text": BODY[:-1] + "8", "callsign": "HUMANCALL",
                                          "transcript": "by ear"}, checked=True, notes=None, editor="tester")
    con.close()
    new = eam_text(BODY_WORDS.replace("hotel", "india"))
    write_sidecar(side, contract_doc(CATALOG[0], new, t0=2.4, decoder_version=V2))
    jl.write_text(json.dumps(contract_doc(CATALOG[1], new, decoder_version=V2)) + "\n")
    rc, out = run(tool, "--db", db, "load-decodes", side, jl, "--no-link")
    assert "kept_human 1" in out and "updated 1" in out
    m1, m2 = q(db, "SELECT * FROM messages ORDER BY id")
    assert (m1["message_text"], m1["callsign"], m1["transcript"]) == (BODY[:-1] + "8", "HUMANCALL", "by ear")
    assert m1["review_status"] == "reviewed" and m1["start_s"] == 2.0      # human row: timing kept too
    assert m1["machine_text"] == BODY_INDIA and m1["decoder_version"] == V2
    assert m2["message_text"] == BODY_INDIA and m2["machine_text"] is None and m2["decoder_version"] == V2
    d = q(db, "SELECT message_id, applied, broadcast_id FROM message_decodes ORDER BY id")
    assert [(r["message_id"], r["applied"]) for r in d] == [(1, 1), (2, 1), (1, 0), (2, 1)]
    assert d[2]["broadcast_id"] == "aaaaaaaaaaaa@2.4"
    assert q(db, "SELECT count(*) AS n FROM reviews WHERE checked = 1")[0]["n"] == 1
    con = review_db.connect(db)
    _, recs = review_db.list_recordings(con)
    m = [x for r in recs for x in r["messages"] if x["id"] == 1][0]
    assert m["pending_machine_text"] == BODY_INDIA
    con.close()


def test_checked_only_row_is_protected(tool, db, tmp_path):
    side = tmp_path / "rec"
    side.mkdir()
    write_sidecar(side, contract_doc(CATALOG[0], eam_text()))
    run(tool, "--db", db, "load-decodes", side, "--no-link")
    con = review_db.connect(db)
    review_db.save_review(con, 1, fields={}, checked=True, notes=None, editor="tester")
    con.close()
    redo = eam_text(BODY_WORDS.replace("golf", "oscar"))
    write_sidecar(side, contract_doc(CATALOG[0], redo, decoder_version=V2))
    rc, out = run(tool, "--db", db, "load-decodes", side, "--no-link")
    assert "kept_human 1" in out
    assert q(db, "SELECT message_text FROM messages")[0][0] == BODY


def test_create_capture_name_match_and_rejected_old_shape(tool, tmp_path):
    p = tmp_path / "x.db"
    run(tool, "--db", p, "init")
    side = tmp_path / "rec"
    side.mkdir()
    write_sidecar(side, contract_doc(CATALOG[0], eam_text()))     # no capture yet: made from `recording`
    rc, out = run(tool, "--db", p, "load-decodes", side, "--no-link")
    assert "captures_created 1" in out and "inserted 1" in out
    assert q(p, "SELECT snr_peak_db FROM captures")[0][0] == 21.4
    # Capture without a checksum, matched by the recording's file name.
    entry = CATALOG[1]
    name = Path(entry["file"]).name
    con = sqlite3.connect(p)
    con.execute("INSERT INTO receivers(code) VALUES ('fake-rx-b')")
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, audio_path)
                   VALUES (2, 11175.0, '2031-01-02T03:04:05.500Z', ?)""", ("recordings/" + name,))
    con.commit()
    con.close()
    doc = contract_doc({"file": entry["file"], "sha256": "c" * 64}, SKYKING_TEXT)
    doc["recording"] = {"path": "/elsewhere/" + name, "sha256": "c" * 64}
    (side / (name + ".decode.json")).write_text(json.dumps(doc))
    (side / "nothing_here.wav.decode.json").write_text(json.dumps(doc | {"recording": {"path": "x.wav"}}))
    # The #6-era shape (no schema_version / decoder_version, `records` list) is refused.
    old = {"records": doc["broadcasts"], "second_pass": {}, "duration_s": 70.0}
    (side / "old_shape.wav.decode.json").write_text(json.dumps(old))
    rc, out = run(tool, "--db", p, "load-decodes", side, "--no-link", "--strict")
    assert rc == 1 and "no capture found for 1" in out and "nothing_here" in out
    assert "skipped 1 document(s) not in decode contract v1" in out and "old_shape" in out
    sk = q(p, "SELECT * FROM messages WHERE capture_id = 2")[0]
    assert sk["msg_type"] == "SKYKING" and sk["broadcast_id"] == "cccccccccccc@2.0"


def test_radio_check_and_bad_confidence(tool, db, tmp_path):
    doc = contract_doc(CATALOG[0], "fakecall this is fake two radio check over")
    roles = [{"callsign": "FAKECALL", "role": "called"}, {"callsign": "FAKE TWO", "role": "closing"}]
    doc["broadcasts"] = [{"type": "radio_check", "start_s": 2.0, "end_s": 5.0, "asr_word_span": [0, 8],
                          "broadcast_id": "aaaaaaaaaaaa@2.0",
                          "transcript": "fakecall this is fake two radio check",
                          "callsign": "FAKE TWO", "callsigns_all": roles,
                          "confidence": 0.7, "confidence_detail": {"callsign": 0.7}}]
    f = tmp_path / "rc.jsonl"
    bad = json.loads(json.dumps(doc))
    bad["recording"]["sha256"] = "b" * 64
    bad["broadcasts"][0]["confidence"] = 7
    f.write_text(json.dumps(doc) + "\n" + json.dumps(bad) + "\n")
    rc, out = run(tool, "--db", db, "load-decodes", f, "--no-link")
    assert "inserted 1" in out and "skipped 1" in out and "0-1" in out
    m = q(db, "SELECT * FROM messages")[0]
    assert m["msg_type"] == "RADIOCHECK" and m["callsign"] == "FAKE TWO"
    assert m["review_status"] == "needs_review" and json.loads(m["callsigns_all"]) == roles


def test_requires_migration(tool, tmp_path):
    p = tmp_path / "v1.db"
    con = sqlite3.connect(p)
    con.executescript((ROOT / "schema.sql").read_text())
    con.close()
    with pytest.raises(SystemExit, match="migrate"):
        run(tool, "--db", p, "load-decodes", tmp_path)


def test_peak_snr_backfill_by_migration(tmp_path):
    p = tmp_path / "v2.db"
    con = sqlite3.connect(p)
    con.executescript((ROOT / "schema.sql").read_text())
    mig = importlib.import_module("hfgcs.db_migrations")
    for version, name, path in [m for m in mig.available() if m[0] <= 2]:   # bring the DB to v2 only
        for stmt in mig._statements(path.read_text()):
            con.execute(stmt)
        con.execute("INSERT INTO schema_version(version, note) VALUES (?, ?)", (version, name))
    con.execute("INSERT INTO receivers(id, code) VALUES (1, 'fake-rx-a')")
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, snr_db, raw_json)
                   VALUES (1, 8992.0, '2031-01-02T03:04:05.000Z', 14.2, ?)""", (json.dumps(CATALOG[0]),))
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, raw_json)
                   VALUES (1, 8992.0, '2031-01-02T04:00:00.000Z', '{"timestamp": "x"}')""")
    con.commit()
    assert mig.apply_migrations(con) == [3]
    assert [r[0] for r in con.execute("SELECT snr_peak_db FROM captures ORDER BY id")] == [21.4, None]
