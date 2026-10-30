"""Scoring the shadow squelch against the decoded messages (hfgcs/shadow_score.py). All data is fake."""

import contextlib
import hashlib
import importlib
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from hfgcs import shadow_score as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
hfgcs_db = importlib.import_module("hfgcs_db")

T = ss._epoch("2026-10-01T10:00:00Z")
KEY = ("kiwi-a", 4724.0)
NOW = T + 86400


def live(s, e, sha, pre=2.0, key=KEY):
    return {"key": key, "s": T + s, "e": T + e, "sha": sha, "file": f"rec/{sha}.wav", "state_key": sha,
            "preroll": pre}


def shadow(s, e, key=KEY):
    return {"key": key, "s": T + s, "e": T + e, "reason": "hang"}


def msg(t0, t1, typ="EAM", conf=0.9, human=False, sha=None, key=KEY, mid=1):
    return {"id": mid, "type": typ, "callsign": "QUICKLINE", "conf": conf, "human": human, "key": key,
            "t0": T + t0, "t1": T + t1, "sha": sha, "file": f"rec/{sha}.wav", "offset": 1.0}


def test_classify_kept_split_cut_missed():
    assert ss.classify(10, 90, [(0, 100)]) == (ss.KEPT, 1.0, 1)
    assert ss.classify(10, 90, [(0, 89)])[0] == ss.KEPT            # within the 2 s edge tolerance
    out, share, n = ss.classify(10, 90, [(0, 50), (52, 100)])      # all but 2 s, in two captures
    assert (out, n) == (ss.SPLIT, 2) and share == pytest.approx(78 / 80)
    out, share, n = ss.classify(10, 90, [(0, 50), (60, 100)])      # a 10 s hole
    assert (out, n) == (ss.CUT, 2) and share == pytest.approx(70 / 80)
    assert ss.classify(10, 90, [(95, 200)]) == (ss.MISSED, 0.0, 0)
    assert ss.classify(10, 12, [(0, 11)])[0] == ss.CUT             # a short message: a quarter is the limit
    assert ss.classify(50, 50, [(0, 100)])[0] == ss.KEPT           # no end time: a point
    assert ss.classify(50, 50, [(60, 100)])[0] == ss.MISSED


def test_messages_are_scored_by_type_with_preroll_and_confidence():
    sh = [shadow(100, 300)]                     # activity from 100 s; its file would start at 98 s
    lv = [live(0, 900, "a")]
    msgs = [msg(99, 290, sha="a", mid=1),                         # starts in the pre-roll: kept
            msg(400, 500, sha="a", mid=2),                        # missed
            msg(600, 605, typ="SKYKING", conf=0.0, sha="a", mid=3),     # a hallucination in noise
            msg(700, 710, typ="RADIOCHECK", conf=0.1, human=True, sha="a", mid=4),  # checked: scored
            msg(86400 - 300, 86400 - 200, sha="a", mid=5),        # ended 3 min before the run
            msg(50, 60, sha="a", mid=6)]                          # before the window
    r = ss.score(lv, sh, msgs, None, now=NOW)
    assert r["start"] == T + 98 and r["start_from"] == "first shadow capture"
    assert dict(r["types"]["EAM"]) == {ss.KEPT: 1, ss.MISSED: 1}
    assert dict(r["types"]["RADIOCHECK"]) == {ss.MISSED: 1}
    assert "SKYKING" not in r["types"] and r["low_confidence"] == 1 and r["too_recent"] == 1
    assert {m["id"] for m in r["messages"]} == {1, 2, 3, 4}
    assert not next(m for m in r["messages"] if m["id"] == 2)["early"]
    assert r["channels"][KEY]["found"] == 3 and r["channels"][KEY]["kept"] == 1

    r = ss.score(lv, sh, msgs, None, since=T, now=NOW)     # --since: the early one is scored, flagged
    assert dict(r["types"]["EAM"]) == {ss.KEPT: 1, ss.MISSED: 2}
    assert next(m for m in r["messages"] if m["id"] == 6)["early"]
    assert ss.score(lv, sh, msgs, None, now=NOW, min_confidence=0)["types"]["SKYKING"][ss.MISSED] == 1
    # the pre-roll counts: a message 3 s ahead of the shadow squelch's activity is 1 s short with
    # 2 s of pre-roll (inside the edge tolerance) and 3 s short without it
    ahead = [msg(97, 290, sha="a")]
    assert dict(ss.score(lv, sh, ahead, None, since=T, now=NOW)["types"]["EAM"]) == {ss.KEPT: 1}
    assert dict(ss.score(lv, sh, ahead, None, since=T, now=NOW, preroll=0)["types"]["EAM"]) == {ss.CUT: 1}


def test_live_minutes_split_by_what_the_decoder_found():
    sh = [shadow(100, 300), shadow(5000, 5100)]       # the second has no live recording
    lv = [live(0, 900, "a"),            # holds a decoded EAM; begins before the window (98 s)
          live(900, 1800, "b"),         # no voice
          live(2000, 2030, "c"),        # voice the decoder found no message in
          live(3000, 3100, "d")]        # not decoded yet
    msgs = [msg(110, 280, sha="a")]
    states = {"a": "decoded", "b": "no_voice", "c": "decoded"}
    r = ss.score(lv, sh, msgs, states, now=NOW)
    cats = r["captures"]
    assert cats[ss.MESSAGE] == {"n": 1, "v1_s": pytest.approx(802), "v2_s": pytest.approx(202)}
    assert cats[ss.NO_VOICE] == {"n": 1, "v1_s": pytest.approx(900), "v2_s": 0.0}
    assert cats[ss.VOICE]["n"] == 1 and cats[ss.PENDING]["n"] == 1
    assert [x["sha"] for x in r["voice_skipped"]] == ["c"]
    assert r["shadow_only"] == {"n": 1, "s": pytest.approx(102)}
    # without the decode state, the last three are one group
    r = ss.score(lv, sh, msgs, None, now=NOW)
    assert r["captures"][ss.NO_MESSAGE]["n"] == 3 and r["captures"][ss.MESSAGE]["n"] == 1
    # a message whose recording has no sha256 is placed in the channel's recording by time
    lv[0]["sha"] = None
    r = ss.score(lv, sh, [msg(110, 280)], states, now=NOW)
    assert r["captures"][ss.MESSAGE]["n"] == 1


def test_summary_and_list_text():
    sh = [shadow(100, 300)]
    lv = [live(0, 900, "a")]
    text = ss.render(ss.score(lv, sh, [msg(110, 280, sha="a")], None, now=NOW), {"v2 shadow vs v1 live"})
    assert text.startswith("v2 shadow vs v1 live. Scored 2026-10-01T10:01:38Z")
    assert "Summary: The shadow squelch would have kept every EAM and Skyking the decoder found (1)." in text
    msgs = [msg(110, 280, sha="a"), msg(400, 500, sha="a", mid=2)]
    text = ss.render(ss.score(lv, sh, msgs, None, now=NOW), set(), show_list=True)
    assert "would have cut or missed 1 of the 2 EAM/Skyking messages" in text
    assert "    missed: 2026-10-01T10:06:40Z kiwi-a 4724 EAM QUICKLINE (100 s, confidence 0.90, 0% covered)" \
        "; listen: rec/a.wav at 1.0 s" in text
    assert "nothing to score" in ss.render(ss.score(lv, [], [], None, now=NOW), set())
    assert "nothing to score yet" in ss.render(ss.score(lv, sh, [], None, now=T + 600), set())


# --- the command, on a real (temporary) database ------------------------------------------------

def _db(tmp_path):
    db = tmp_path / "hfgcs.db"
    with contextlib.redirect_stdout(io.StringIO()):
        assert hfgcs_db.main(["--db", str(db), "init"]) == 0
    con = sqlite3.connect(db)
    con.execute("INSERT INTO receivers(code, kind) VALUES ('kiwi-a', 'kiwisdr')")
    con.execute("INSERT INTO external_sources(code, name) VALUES ('missileer', 'missileer.watch') "
                "ON CONFLICT DO NOTHING")
    return db, con


def _capture(con, start, end, sha, external=None):
    con.execute("""INSERT INTO captures(receiver_id, frequency_khz, start_utc, end_utc, audio_path,
                       audio_sha256, external_source, external_id) VALUES (1, 4724.0, ?, ?, ?, ?, ?, ?)""",
                (start, end, f"rec/{sha}.wav", sha, external, sha if external else None))
    return con.execute("SELECT id FROM captures WHERE audio_sha256 = ?", (sha,)).fetchone()[0]


def _message(con, cap, start_s, end_s, typ="EAM", conf=0.93, status="auto"):
    con.execute("""INSERT INTO messages(capture_id, heard_utc, msg_type, start_s, end_s, confidence, callsign,
                       review_status) VALUES (?, '2026-10-01T10:00:00.000Z', ?, ?, ?, ?, 'QUICKLINE', ?)""",
                (cap, typ, start_s, end_s, conf, status))


def _run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = ss.main([str(a) for a in argv])
    return rc, out.getvalue(), err.getvalue()


def _files(tmp_path, shadow_rows):
    catalog = tmp_path / "hfgcs_catalog.jsonl"
    rows = [{"receiver_id": "kiwi-a", "frequency_khz": 4724.0, "file_start_utc": s, "start_utc": s,
             "end_utc": e, "sha256": sha, "file": f"rec/{sha}.wav", "preroll_sec": 2.0}
            for s, e, sha in (("2026-10-01T10:00:00.000Z", "2026-10-01T10:15:00.000Z", "a" * 64),
                              ("2026-10-01T10:15:00.000Z", "2026-10-01T10:30:00.000Z", "b" * 64))]
    catalog.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "hfgcs_catalog.squelch-shadow.jsonl").write_text(
        "".join(json.dumps({"receiver_id": "kiwi-a", "frequency_khz": 4724.0, "algorithm": "v2",
                            "live_algorithm": "v1", "start_utc": s, "end_utc": e,
                            "close_reason": "hang"}) + "\n" for s, e in shadow_rows))
    state = tmp_path / "decode-state.json"
    state.write_text(json.dumps({"schema": 1, "clips": {"a" * 64: {"status": "decoded"},
                                                        "b" * 64: {"status": "no_voice"}}}))
    return catalog, state


def test_command_reads_the_database_without_changing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("HFGCS_DECODE_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "no-state-here"))
    db, con = _db(tmp_path)
    a = _capture(con, "2026-10-01T10:00:00.000Z", "2026-10-01T10:15:00.000Z", "a" * 64)
    _message(con, a, 300.0, 450.0)                            # 10:05:00-10:07:30, in the shadow capture
    _message(con, a, 600.0, 700.0)                            # 10:10, missed
    _message(con, a, 750.0, 800.0, status="rejected")         # a person rejected it: not scored
    _message(con, a, 820.0, 830.0, typ="SKYKING", conf=0.0)   # below the confidence floor
    ext = _capture(con, "2026-10-01T10:00:00.000Z", "2026-10-01T10:15:00.000Z", "e" * 64,
                   external="missileer")
    _message(con, ext, 100.0, 200.0)                          # someone else's recording: not ours to score
    con.commit()
    con.close()
    catalog, state = _files(tmp_path, [("2026-10-01T10:04:59.000Z", "2026-10-01T10:07:40.000Z")])
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    rc, out, _ = _run("--catalog", catalog, "--db", db, "--decode-state", state, "--list")
    assert rc == 0
    eam = next(line for line in out.splitlines() if line.strip().startswith("EAM"))
    assert eam.split() == ["EAM", "2", "1", "0", "0", "1"]
    assert "Not scored: 1 below the confidence floor." in out
    assert "missed: 2026-10-01T10:10:00Z kiwi-a 4724 EAM QUICKLINE" in out
    assert "missed, low confidence: 2026-10-01T10:13:40Z kiwi-a 4724 SKYKING" in out
    no_voice = next(line for line in out.splitlines() if line.strip().startswith("no voice"))
    assert no_voice.split()[2:] == ["1", "15.0", "0.0"]
    assert "would have cut or missed 1 of the 2 EAM/Skyking messages" in out
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before

    # no decode state given and none at the worker's default place: two groups only
    rc, out, _ = _run("--catalog", catalog, "--db", db)
    assert rc == 0 and "no message found" in out and "No decode state file" in out
    # the worker's state file is picked up from $HFGCS_DECODE_STATE_DIR
    monkeypatch.setenv("HFGCS_DECODE_STATE_DIR", str(tmp_path))
    (tmp_path / "decode-state.json").write_text(state.read_text())
    rc, out, _ = _run("--catalog", catalog, "--db", db)
    assert rc == 0 and "not decoded yet" in out
    # a broken default state file is skipped with a note, an explicit one is an error
    (tmp_path / "decode-state.json").write_text("{not json")
    rc, out, err = _run("--catalog", catalog, "--db", db)
    assert rc == 0 and "ignoring the decode state" in err
    rc, _, err = _run("--catalog", catalog, "--db", db, "--decode-state", tmp_path / "decode-state.json")
    assert rc == 2 and "cannot read the decode state" in err


def test_command_errors_and_an_empty_shadow_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HFGCS_DB", raising=False)
    db, con = _db(tmp_path)
    con.close()
    catalog, _ = _files(tmp_path, [])
    rc, out, _ = _run("--catalog", catalog, "--db", db)
    assert rc == 0 and "nothing to score" in out
    assert _run("--catalog", catalog)[0] == 2                                  # no database
    assert _run("--catalog", tmp_path / "missing.jsonl", "--db", db)[0] == 2
    rc, _, err = _run("--catalog", catalog, "--db", db, "--since", "yesterday")
    assert rc == 2 and "--since" in err
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    rc, _, err = _run("--catalog", catalog, "--db", empty)
    assert rc == 2 and "schema is version 0" in err
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a database" * 100)
    rc, _, err = _run("--catalog", catalog, "--db", junk)
    assert rc == 2 and "cannot read" in err
    monkeypatch.setenv("HFGCS_DB", str(db))
    assert _run("--catalog", catalog)[0] == 0                                  # $HFGCS_DB
