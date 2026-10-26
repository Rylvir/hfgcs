"""Private review app (hfgcs/review_app.py, hfgcs/review_db.py). Fake fixtures only:
a temporary DB, fake receivers/captures/messages and dummy audio bytes."""

import contextlib
import http.client
import importlib.util
import io
import json
import re
import sqlite3
import threading
from pathlib import Path
from urllib.parse import urlencode

import pytest

from hfgcs import review_app, review_db

ROOT = Path(__file__).resolve().parents[1]
FAKE_WAV = b"RIFF" + b"\x00" * 60 + b"fake-audio-bytes" * 64


def _tool():
    spec = importlib.util.spec_from_file_location("hfgcs_db_review_tests", ROOT / "scripts" / "hfgcs_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def env(tmp_path):
    db = tmp_path / "data" / "hfgcs.db"
    db.parent.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        assert _tool().main(["--db", str(db), "init"]) == 0
    rec = tmp_path / "recordings"
    rec.mkdir()
    (rec / "HFGCS_8992kHz_20310102T030405Z_fake-rx.wav").write_bytes(FAKE_WAV)
    (rec / "HFGCS_11175kHz_20310103T101010Z_fake-rx.wav").write_bytes(FAKE_WAV)
    (tmp_path / "secret.wav").write_bytes(b"outside the recordings dir")
    (rec / "link.wav").symlink_to(tmp_path / "secret.wav")
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("INSERT INTO receivers(id, code, kind) VALUES (1, 'fake-rx', 'local_sdr')")
    caps = [  # id, freq, start, audio_path
        (1, 8992.0, "2031-01-02T03:04:05.000Z", "recordings/HFGCS_8992kHz_20310102T030405Z_fake-rx.wav"),
        (2, 11175.0, "2031-01-03T10:10:10.000Z", "HFGCS_11175kHz_20310103T101010Z_fake-rx.wav"),
        (3, 8992.0, "2031-01-03T12:00:00.000Z", "../secret.wav"),
        (4, 4724.0, "2031-01-01T01:00:00.000Z", str(tmp_path / "secret.wav")),
        (5, 4724.0, "2031-01-01T02:00:00.000Z", "link.wav"),
    ]
    for cid, f, start, path in caps:
        con.execute("""INSERT INTO captures(id, receiver_id, frequency_khz, start_utc, duration_sec,
                                            audio_path, voice_start_s) VALUES (?,1,?,?,60,?,1.5)""",
                    (cid, f, start, path))
    con.execute("""INSERT INTO messages(id, capture_id, heard_utc, msg_type, callsign, prefix, message_text,
                                        length, transcript, confidence, origin)
                   VALUES (10, 1, '2031-01-02T03:04:07.000Z', 'EAM', 'FAKECALL', 'QX', 'QXABCDEFGHJKLMNP', 16,
                           'quebec x-ray alfa bravo', 0.71, 'pipeline')""")
    con.execute("""INSERT INTO messages(id, capture_id, heard_utc, msg_type, message_text, origin)
                   VALUES (11, 2, '2031-01-03T10:10:12.000Z', 'RADIOCHECK', NULL, 'pipeline')""")
    con.commit()
    con.close()
    s = review_app.Settings(db=db, recordings=rec, host="127.0.0.1", port=0, secret=b"t" * 32)
    srv = review_app.make_server(s)
    s.port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"db": db, "rec": rec, "settings": s, "port": s.port, "tmp": tmp_path}
    srv.shutdown()
    srv.server_close()


def req(e, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", e["port"], timeout=10)
    h = {"Host": f"127.0.0.1:{e['port']}", **(headers or {})}
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def page(e, path="/"):
    r, data = req(e, "GET", path)
    assert r.status == 200
    html = data.decode()
    cookie = r.getheader("Set-Cookie")
    tok = re.search(r"name=csrf value='([^']+)'", html)
    return html, cookie, tok.group(1) if tok else None


def post(e, path, form, *, token, cookie_token=None, origin="same", extra=None):
    origin_hdr = {"same": {"Origin": f"http://127.0.0.1:{e['port']}"},
                  "evil": {"Origin": "http://evil.example"}, None: {}}[origin]
    headers = {"Content-Type": "application/x-www-form-urlencoded", **origin_hdr, **(extra or {})}
    ct = token if cookie_token is None else cookie_token
    if ct:
        headers["Cookie"] = f"{review_app.CSRF_COOKIE}={ct}"
    body = urlencode({**form, **({"csrf": token} if token else {})})
    return req(e, "POST", path, body, headers)


def db(e):
    con = sqlite3.connect(e["db"])
    con.row_factory = sqlite3.Row
    return con


def save_form(**kw):
    base = {"msg_type": "EAM", "callsign": "FAKECALL", "prefix": "QX", "length": "16", "addressee": "",
            "message_text": "QXABCDEFGHJKLMNP", "transcript": "quebec x-ray alfa bravo", "notes": "",
            "checked_present": "1", "editor": "tester", "return": "/"}
    base.update(kw)
    return base


# ------------------------------------------------------------------ listing & filters
def test_list_newest_first_with_player_fields_and_neet_link(env):
    html, cookie, tok = page(env)
    assert cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie and tok
    order = [int(x) for x in re.findall(r"<article class=rec id='c(\d+)'", html)]
    assert order == [3, 2, 1, 5, 4]
    assert "<audio controls preload=none src='/audio/1#t=1.50'>" in html
    assert "QXABCDEFGHJKLMNP" in html and "FAKECALL" in html and "quebec x-ray alfa bravo" in html
    assert "confidence <b>0.71</b>" in html and "16 chars" in html
    assert "https://neetintel.github.io/dtcp_gallery.html?310102" in html
    assert "https://neetintel.github.io/dtcp_gallery.html?310103" in html


def test_filters_date_frequency_unchecked(env):
    html, _, _ = page(env, "/?date=2031-01-03")
    assert re.findall(r"id='c(\d+)'", html) == ["3", "2"]
    html, _, _ = page(env, "/?freq=8992")
    assert re.findall(r"id='c(\d+)'", html) == ["3", "1"]
    html, _, _ = page(env, "/?date=2031-01-03&freq=11175")
    assert re.findall(r"id='c(\d+)'", html) == ["2"]
    r, _ = req(env, "GET", "/?date=not-a-date")
    assert r.status == 400


def test_neet_url_pattern():
    assert review_db.neet_timecard_url("2026-10-25") == "https://neetintel.github.io/dtcp_gallery.html?261025"
    assert review_db.neet_timecard_url("2031-01-02T03:04:05Z").endswith("?310102")


# ------------------------------------------------------------------ edit, history, checked
def test_edit_records_history_and_keeps_machine_copy(env):
    _, _, tok = page(env)
    r, _ = post(env, "/message/10", save_form(message_text="QXABCDEFGHJKLMNQ", callsign="fakecall two",
                                               transcript="corrected words", notes="first note", version="0"),
                token=tok)
    assert r.status == 303 and r.getheader("Location").startswith("/?saved=1")
    con = db(env)
    m = con.execute("SELECT * FROM messages WHERE id = 10").fetchone()
    assert m["message_text"] == "QXABCDEFGHJKLMNQ" and m["callsign"] == "FAKECALL TWO"
    assert m["machine_text"] == "QXABCDEFGHJKLMNP" and m["corrected"] == 1
    assert m["correction_source"] == "review-ui" and m["transcript"] == "corrected words"
    hist = {(h["table_name"], h["field"]): h for h in con.execute("SELECT * FROM edit_history")}
    assert hist[("messages", "message_text")]["old_value"] == "QXABCDEFGHJKLMNP"
    assert hist[("messages", "message_text")]["new_value"] == "QXABCDEFGHJKLMNQ"
    assert hist[("messages", "callsign")]["editor"] == "tester"
    assert hist[("reviews", "notes")]["new_value"] == "first note"
    assert len({h["change_id"] for h in hist.values()}) == 1
    assert all(h["editor_source"] == "form" for h in hist.values())
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE edit_history SET new_value = 'x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM edit_history")
    con.rollback()  # release the write lock the refused statements opened
    html, _, _ = page(env, "/message/10/history")
    assert "QXABCDEFGHJKLMNQ" in html and "tester" in html
    # A stale form (old version number) is refused instead of overwriting.
    r, _ = post(env, "/message/10", save_form(callsign="OTHER", version="0"), token=tok)
    assert r.status == 409
    # Saving the same values again writes no history.
    n = con.execute("SELECT count(*) FROM edit_history").fetchone()[0]
    ver = con.execute("SELECT max(id) FROM edit_history").fetchone()[0]
    r, _ = post(env, "/message/10", save_form(message_text="QXABCDEFGHJKLMNQ", callsign="FAKECALL TWO",
                                               transcript="corrected words", notes="first note",
                                               version=str(ver)), token=tok)
    assert r.status == 303 and con.execute("SELECT count(*) FROM edit_history").fetchone()[0] == n


def test_invalid_input_rejected(env):
    _, _, tok = page(env)
    r, _ = post(env, "/message/10", save_form(length="12a"), token=tok)
    assert r.status == 400
    r, _ = post(env, "/message/10", save_form(msg_type="NOPE"), token=tok)
    assert r.status == 400
    r, _ = post(env, "/message/999", save_form(), token=tok)
    assert r.status == 404


def test_checked_flag_and_unchecked_filter(env):
    _, _, tok = page(env)
    r, _ = post(env, "/message/10", save_form(checked="1"), token=tok)
    assert r.status == 303
    con = db(env)
    rv = con.execute("SELECT * FROM reviews WHERE message_id = 10").fetchone()
    assert rv["checked"] == 1 and rv["reviewer"] == "tester" and rv["reviewed_utc"].endswith("Z")
    assert con.execute("SELECT review_status FROM messages WHERE id = 10").fetchone()[0] == "reviewed"
    html, _, _ = page(env, "/?unchecked=1")
    assert "id='c1'" not in html and "id='c2'" in html and "id='c3'" in html  # c3: no message yet
    r, _ = post(env, "/message/10", save_form(), token=tok)  # checkbox absent -> unchecked
    assert con.execute("SELECT checked FROM reviews WHERE message_id = 10").fetchone()[0] == 0
    assert con.execute("SELECT review_status FROM messages WHERE id = 10").fetchone()[0] == "needs_review"
    rows = con.execute("SELECT old_value, new_value FROM edit_history WHERE field = 'checked' "
                       "ORDER BY id").fetchall()
    assert [tuple(x) for x in rows] == [(None, "1"), ("1", "0")]


def test_add_message_to_capture_without_one(env):
    html, _, tok = page(env)
    assert "action='/capture/3/message'" in html
    r, _ = post(env, "/capture/3/message", {"editor": "tester", "return": "/#c3"}, token=tok)
    assert r.status == 303
    con = db(env)
    m = con.execute("SELECT * FROM messages WHERE capture_id = 3").fetchone()
    assert m["origin"] == "manual" and m["review_status"] == "needs_review"
    first = con.execute("SELECT field FROM edit_history WHERE message_id = ?", (m["id"],)).fetchone()
    assert first[0] == "created"


def test_cf_access_header_only_when_trusted(env):
    _, _, tok = page(env)
    hdr = {review_app.CF_ACCESS_HEADER: "someone@example.org"}
    post(env, "/message/10", save_form(notes="n1"), token=tok, extra=hdr)
    con = db(env)
    assert con.execute("SELECT editor, editor_source FROM edit_history ORDER BY id DESC").fetchone()[:] == \
        ("tester", "form")
    env["settings"].trust_cf_access = True
    post(env, "/message/10", save_form(notes="n2"), token=tok, extra=hdr)
    assert con.execute("SELECT editor, editor_source FROM edit_history ORDER BY id DESC").fetchone()[:] == \
        ("someone@example.org", "cf_access")


# ------------------------------------------------------------------ gold export
def test_gold_export_cli_and_endpoint(env, tmp_path):
    _, _, tok = page(env)
    post(env, "/message/10", save_form(checked="1", message_text="QXABCDEFGHJKLMNQ", notes="by ear"),
         token=tok)
    r, data = req(env, "GET", "/export/gold.jsonl")
    assert r.status == 200 and r.getheader("Content-Type").startswith("application/x-ndjson")
    recs = [json.loads(x) for x in data.decode().splitlines()]
    assert len(recs) == 1
    g = recs[0]
    assert g["id"] == "hfgcs:msg10" and g["label_status"] == "gold" and g["type"] == "eam"
    assert g["labels"]["body"] == "QXABCDEFGHJKLMNQ" and g["labels"]["preamble"] == "QXABCD"
    assert g["labels"]["declared_length"] == 16 and g["labels"]["callsign"] == "FAKECALL"
    assert g["reference_machine"]["text"] == "QXABCDEFGHJKLMNP" and g["corrected"] is True
    assert Path(g["audio_path"]).is_file() and Path(g["audio_path"]).is_relative_to(env["rec"].resolve())
    assert g["voice_start_s"] == 1.5 and g["notes"] == "by ear" and g["split"] in ("dev", "test")
    out = tmp_path / "gold.jsonl"
    with contextlib.redirect_stderr(io.StringIO()):
        rc = review_app.main(["--db", str(env["db"]), "export-gold", "--out", str(out),
                              "--recordings", str(env["rec"])])
    assert rc == 0 and [json.loads(x) for x in out.read_text().splitlines()] == recs


# ------------------------------------------------------------------ CSRF / origin / host
def test_csrf_and_origin_rejections(env):
    _, _, tok = page(env)
    form = save_form(callsign="HACKED")
    assert post(env, "/message/10", form, token=None)[0].status == 403                 # no token at all
    assert post(env, "/message/10", form, token=tok, cookie_token="x.y")[0].status == 403  # cookie mismatch
    forged = "abc." + "0" * 32
    assert post(env, "/message/10", form, token=forged, cookie_token=forged)[0].status == 403  # unsigned
    assert post(env, "/message/10", form, token=tok, origin="evil")[0].status == 403      # cross-origin
    assert post(env, "/message/10", form, token=tok, origin=None)[0].status == 403        # no Origin/Referer
    assert post(env, "/message/10", form, token=tok, origin=None,
                extra={"Referer": "http://evil.example/x"})[0].status == 403
    assert post(env, "/message/10", form, token=tok, extra={"Sec-Fetch-Site": "cross-site"})[0].status == 403
    assert db(env).execute("SELECT callsign FROM messages WHERE id = 10").fetchone()[0] == "FAKECALL"
    assert db(env).execute("SELECT count(*) FROM edit_history").fetchone()[0] == 0
    # Same-origin Referer without Origin is accepted.
    r, _ = post(env, "/message/10", form, token=tok, origin=None,
                extra={"Referer": f"http://127.0.0.1:{env['port']}/"})
    assert r.status == 303


def test_unexpected_host_header_rejected(env):
    r, _ = req(env, "GET", "/", headers={"Host": "attacker.example"})
    assert r.status == 400


def test_security_headers(env):
    r, _ = req(env, "GET", "/")
    assert "frame-ancestors 'none'" in r.getheader("Content-Security-Policy")
    assert r.getheader("X-Frame-Options") == "DENY" and r.getheader("Cache-Control") == "no-store"


# ------------------------------------------------------------------ audio & traversal
def test_audio_served_with_range(env):
    r, data = req(env, "GET", "/audio/1")
    assert r.status == 200 and data == FAKE_WAV and r.getheader("Content-Type") == "audio/wav"
    r, data = req(env, "GET", "/audio/2")       # stored as a bare file name
    assert r.status == 200 and data == FAKE_WAV
    r, data = req(env, "GET", "/audio/1", headers={"Range": "bytes=4-9"})
    assert r.status == 206 and data == FAKE_WAV[4:10]
    assert r.getheader("Content-Range") == f"bytes 4-9/{len(FAKE_WAV)}"
    r, _ = req(env, "GET", "/audio/1", headers={"Range": "bytes=99999-"})
    assert r.status == 416


def test_audio_path_traversal_blocked(env):
    for cid in (3, 4, 5):   # ../secret.wav, absolute path outside, symlink escaping the dir
        r, data = req(env, "GET", f"/audio/{cid}")
        assert r.status == 404 and b"outside the recordings dir" not in data
    for p in ("/audio/..%2f..%2fsecret.wav", "/audio/../secret.wav", "/audio/1/../../secret.wav",
              "/audio/%2e%2e/secret.wav", "/audio/999"):
        r, data = req(env, "GET", p)
        assert r.status == 404 and b"outside the recordings dir" not in data
    rec = env["rec"]
    assert review_db.resolve_audio(rec, "../secret.wav") is None
    assert review_db.resolve_audio(rec, str(env["tmp"] / "secret.wav")) is None
    assert review_db.resolve_audio(rec, "link.wav") is None
    assert review_db.resolve_audio(rec, "/etc/passwd") is None
    assert review_db.resolve_audio(rec, "x\x00.wav") is None


# ------------------------------------------------------------------ bind policy & config
def test_bind_refused_on_all_interfaces(env, capsys):
    for host in ("0.0.0.0", "::", ""):
        with pytest.raises(SystemExit):
            review_app.check_bind(host, False)
    review_app.check_bind("0.0.0.0", True)      # explicit flag
    review_app.check_bind("192.168.2.214", False)
    rc = review_app.main(["--db", str(env["db"]), "serve", "--host", "0.0.0.0", "--port", "0"])
    assert rc == 2 and "--allow-all-interfaces" in capsys.readouterr().err


def test_defaults_and_env(monkeypatch):
    s = review_app.settings_from_env({})
    assert (s.host, s.port) == ("127.0.0.1", 8090)
    assert str(s.db).endswith("hfgcs-app/data/hfgcs.db")
    assert str(s.recordings).endswith("hfgcs-app/recordings")
    s = review_app.settings_from_env({"HFGCS_REVIEW_HOST": "192.168.2.214", "HFGCS_REVIEW_PORT": "8090",
                                      "HFGCS_DB": "/x/hfgcs.db", "HFGCS_REVIEW_TRUST_CF_ACCESS": "1",
                                      "HFGCS_REVIEW_ALLOWED_HOSTS": "hilltop.lan:8090"})
    assert s.host == "192.168.2.214" and s.trust_cf_access and str(s.db) == "/x/hfgcs.db"
    assert s.host_whitelist() == {"192.168.2.214:8090", "hilltop.lan:8090"}


def test_refuses_db_without_review_tables(tmp_path):
    p = tmp_path / "v1.db"
    con = sqlite3.connect(p)
    con.executescript((ROOT / "schema.sql").read_text())
    con.close()
    with pytest.raises(RuntimeError, match="migrate"):
        review_db.connect(p)
    review_db.connect(p, migrate=True).close()
    review_db.connect(p).close()
