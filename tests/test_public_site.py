"""The PUBLIC page (public_site/ + hfgcs/static_site.py) must never expose recordings,
transcripts, the database, the private review app, or the receiver's location."""

import http.client
import re
import threading
from pathlib import Path

import pytest

from hfgcs import static_site

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public_site"
ALLOWED_SUFFIXES = {".html", ".css", ".svg", ".txt", ".ico", ".png", ".webmanifest"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac", ".webm", ".mp4", ".iq", ".raw"}
FORBIDDEN = [  # references to private parts of the system
    r"hfgcs\.db", r"\.sqlite", r"\bsqlite\b", r"\.db\b", r"datasette", r"review_app", r"review_db",
    r"/audio/", r"gold\.jsonl", r"hfgcs_catalog", r"recordings/", r"edit_history",
    r":8090\b", r":8092\b", r"\b8090\b", r"\b8092\b", r"192\.168\.", r"\b10\.\d+\.\d+\.\d+\b",
    r"127\.0\.0\.1", r"localhost", r"\bhilltop\b", r"Cf-Access", r"HFGCS_(DB|REVIEW|RECORDINGS)",
    r"<audio\b", r"<video\b", r"<source\b",
]
COORDINATES = [
    r"-?\b\d{1,3}\.\d{2,}\s*[,;/ ]\s*-?\d{1,3}\.\d{2,}\b",          # 38.91, -121.07 style pairs
    r"\b\d{1,3}\s*°",                                              # degrees
    r"\b\d{1,3}\.\d+\s*[NSEW]\b",                                  # 38.9N
    r"\b\d{1,3}\s*[°d]?\s*\d{1,2}\s*['′]\s*(\d{1,2}(\.\d+)?\s*[\"″])?\s*[NSEW]\b",  # DMS
    r"\b(lat|lon|lng|latitude|longitude)\b\s*[:=]",
    r"geo:\s*-?\d", r"maps\.google|google\.[a-z.]+/maps|openstreetmap\.org|maps\.apple|bing\.com/maps",
    r"[?&](ll|q)=-?\d{1,3}\.\d+", r"#map=\d", r"\b[A-R]{2}\d{2}[a-x]{2}\b",  # Maidenhead locator
]


def public_files():
    return [p for p in PUBLIC.rglob("*") if p.is_file()]


def test_public_dir_exists_with_index():
    assert (PUBLIC / "index.html").is_file()
    assert "https://waterfall.scanplacer.com" in (PUBLIC / "index.html").read_text()


def test_no_audio_or_unexpected_files():
    for p in public_files():
        assert p.suffix.lower() not in AUDIO_SUFFIXES, f"audio file in public_site: {p}"
        assert p.suffix.lower() in ALLOWED_SUFFIXES, f"unexpected file type in public_site: {p}"
        assert not p.name.startswith("."), f"hidden file in public_site: {p}"
        head = p.read_bytes()[:16]
        assert not head.startswith((b"RIFF", b"OggS", b"fLaC", b"ID3", b"SQLite format")), p


@pytest.mark.parametrize("pattern", FORBIDDEN + COORDINATES)
def test_no_private_references_or_coordinates(pattern):
    rx = re.compile(pattern, re.I if pattern not in COORDINATES[-1:] else 0)
    for p in public_files():
        if p.suffix.lower() in {".png", ".ico"}:
            continue
        text = p.read_text(encoding="utf-8")
        m = rx.search(text)
        assert m is None, f"{p.name}: forbidden match {m.group(0)!r} for {pattern!r}"


def test_static_server_module_is_isolated():
    src = (ROOT / "hfgcs" / "static_site.py").read_text()
    for bad in ("review_app", "review_db", "sqlite3", "HFGCS_DB", "recordings", "import hfgcs"):
        assert bad not in src, bad


@pytest.fixture()
def server(tmp_path):
    site = tmp_path / "site"
    (site / "sub").mkdir(parents=True)
    (site / "empty").mkdir()
    (site / "index.html").write_text("<p>home</p>")
    (site / "404.html").write_text("<p>nope</p>")
    (site / "style.css").write_text("p{}")
    (site / "sub" / "index.html").write_text("<p>sub</p>")
    (site / ".env").write_text("SECRET=1")
    (site / "notes.md").write_text("draft")
    (tmp_path / "outside.html").write_text("outside")
    (site / "escape.html").symlink_to(tmp_path / "outside.html")
    srv = static_site.make_server(site, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def get(port, path, method="GET"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


def test_static_server_serves_only_site_files(server):
    r, b = get(server, "/")
    assert r.status == 200 and b == b"<p>home</p>"
    assert "frame-ancestors" in r.getheader("Content-Security-Policy")
    assert get(server, "/style.css")[0].getheader("Content-Type").startswith("text/css")
    assert get(server, "/sub/")[1] == b"<p>sub</p>"
    for p in ("/empty/", "/sub", "/.env", "/notes.md", "/escape.html", "/../outside.html",
              "/%2e%2e/outside.html", "/sub/..%2f..%2foutside.html", "/nope.html"):
        r, b = get(server, p)
        assert r.status == 404 and b"outside" not in b and b"SECRET" not in b and b"Index of" not in b, p
    assert get(server, "/", "POST")[0].status == 405


def test_static_bind_refused_on_all_interfaces(tmp_path):
    (tmp_path / "index.html").write_text("x")
    with pytest.raises(SystemExit):
        static_site.make_server(tmp_path, "0.0.0.0", 0)
    assert (static_site.DEFAULT_HOST, static_site.DEFAULT_PORT) == ("127.0.0.1", 8091)


def test_real_public_site_served(tmp_path):
    srv = static_site.make_server(PUBLIC, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r, b = get(srv.server_address[1], "/")
        assert r.status == 200 and b"HFGCS Monitoring Project" in b
        assert get(srv.server_address[1], "/robots.txt")[0].status == 200
    finally:
        srv.shutdown()
        srv.server_close()
