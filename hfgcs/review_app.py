"""Private review app: listen to recordings, correct transcripts, mark gold labels.

Stdlib only (http.server + sqlite3); no JavaScript. LAN-only by design and completely
separate from the public site (hfgcs/static_site.py); never put it behind the tunnel.

    python -m hfgcs.review_app serve            # 127.0.0.1:8090 unless HFGCS_REVIEW_HOST/PORT
    python -m hfgcs.review_app export-gold --out labels.jsonl

See docs/review-ui.md for configuration, security notes and the gold-label format.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import os
import re
import secrets
import sqlite3
import sys
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlsplit

from . import review_db as rdb

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_DB = "~/hfgcs-app/data/hfgcs.db"
DEFAULT_RECORDINGS = "~/hfgcs-app/recordings"
ALL_INTERFACES = {"0.0.0.0", "::", "[::]", "", "*"}
CF_ACCESS_HEADER = "Cf-Access-Authenticated-User-Email"
CSRF_COOKIE = "hfgcs_csrf"
EDITOR_COOKIE = "hfgcs_editor"
MAX_BODY = 64 * 1024
PAGE_SIZE = 25
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--mute:#5d6b7d;--line:#d7dde5;--ok:#1d7a46;
--warn:#9a5b00;--acc:#1f5fbf}
*{box-sizing:border-box}body{margin:0;
font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);
color:var(--ink)}
header.top{position:sticky;top:0;z-index:2;background:#14202e;color:#fff;padding:.5rem 1rem;
display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
header.top h1{font-size:1.05rem;margin:0}header.top a{color:#cfe0ff}
main{max-width:1200px;margin:0 auto;padding:.75rem}
form.filters{display:flex;flex-wrap:wrap;gap:.5rem;align-items:end;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:.6rem;margin-bottom:.75rem}
form.filters label{display:flex;flex-direction:column;font-size:.8rem;color:var(--mute)}
input,select,textarea,button{font:inherit}input,select,textarea{border:1px solid var(--line);
border-radius:6px;padding:.35rem .45rem;background:#fff;max-width:100%}
textarea{width:100%;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
button{border:0;border-radius:6px;padding:.45rem .9rem;background:var(--acc);color:#fff;
cursor:pointer}
h2.day{font-size:1rem;margin:1.2rem 0 .5rem;display:flex;gap:.75rem;flex-wrap:wrap;
align-items:baseline}
h2.day a{font-weight:normal;font-size:.9rem}
article.rec{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.7rem;
margin-bottom:.75rem}
.rec-head{display:flex;flex-wrap:wrap;gap:.35rem 1rem;font-size:.9rem;color:var(--mute);
margin-bottom:.5rem}
.rec-head b{color:var(--ink)}
.rec-body{display:grid;grid-template-columns:minmax(240px,320px) 1fr;gap:.8rem}
.player audio{width:100%}.player p{margin:.3rem 0;font-size:.85rem;color:var(--mute)}
.msg{border-top:1px dashed var(--line);padding-top:.5rem;margin-top:.5rem}.msg:first-child{border:0;
margin:0;padding:0}
.fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:.4rem}
.fields label,.wide label{display:flex;flex-direction:column;font-size:.78rem;color:var(--mute)}
.ro{font-size:.85rem;color:var(--mute);margin:.35rem 0}.wide{margin-top:.4rem}
.actions{display:flex;flex-wrap:wrap;gap:.75rem;align-items:center;margin-top:.5rem}
.badge{display:inline-block;border-radius:999px;padding:0 .5rem;font-size:.78rem;border:1px solid}
.badge.ok{color:var(--ok);border-color:var(--ok)}.badge.todo{color:var(--warn);
border-color:var(--warn)}
nav.pager{display:flex;gap:1rem;justify-content:center;margin:1rem 0}
table.hist{border-collapse:collapse;width:100%;font-size:.85rem}
table.hist td,table.hist th{border:1px solid var(--line);padding:.3rem;vertical-align:top;
text-align:left}
.err{background:#fde8e8;border:1px solid #e3a0a0;padding:.6rem;border-radius:8px}
@media (max-width:760px){.rec-body{grid-template-columns:1fr}main{padding:.4rem}
header.top{position:static}}
"""


class BindRefused(SystemExit):
    pass


def check_bind(host: str, allow_all_interfaces: bool) -> None:
    """Refuse an all-interfaces bind unless explicitly allowed (--allow-all-interfaces)."""
    if host.strip() in ALL_INTERFACES and not allow_all_interfaces:
        raise BindRefused(f"refusing to bind to {host or '(empty)'!s}: this exposes the review app on every "
                          "interface. Bind to a specific LAN address (e.g. HFGCS_REVIEW_HOST=192.168.2.214) "
                          "or pass --allow-all-interfaces if you really mean it.")


def _env_bool(name: str, env=os.environ) -> bool:
    return env.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    db: Path = Path(os.path.expanduser(DEFAULT_DB))
    recordings: Path = Path(os.path.expanduser(DEFAULT_RECORDINGS))
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    allow_all_interfaces: bool = False
    trust_cf_access: bool = False
    allowed_hosts: list[str] = field(default_factory=list)
    default_editor: str = ""
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    migrate: bool = False

    def host_whitelist(self) -> Optional[set[str]]:
        """Accepted Host headers (DNS-rebinding defence), or None to accept any
        (only when bound to all interfaces without HFGCS_REVIEW_ALLOWED_HOSTS)."""
        hosts = {h.strip().lower() for h in self.allowed_hosts if h.strip()}
        if self.host.strip() not in ALL_INTERFACES:
            h = f"[{self.host}]" if ":" in self.host else self.host
            hosts |= {f"{h}:{self.port}".lower()}
            if self.host in ("127.0.0.1", "::1", "localhost"):
                hosts |= {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        return hosts or None


def settings_from_env(env=os.environ) -> Settings:
    s = Settings()
    s.db = Path(os.path.expanduser(env.get("HFGCS_DB", DEFAULT_DB)))
    s.recordings = Path(os.path.expanduser(env.get("HFGCS_RECORDINGS", DEFAULT_RECORDINGS)))
    s.host = env.get("HFGCS_REVIEW_HOST", DEFAULT_HOST)
    s.port = int(env.get("HFGCS_REVIEW_PORT", DEFAULT_PORT))
    s.trust_cf_access = _env_bool("HFGCS_REVIEW_TRUST_CF_ACCESS", env)
    s.allowed_hosts = [h for h in env.get("HFGCS_REVIEW_ALLOWED_HOSTS", "").split(",") if h.strip()]
    s.default_editor = env.get("HFGCS_REVIEW_EDITOR", "").strip()
    if env.get("HFGCS_REVIEW_SECRET"):
        s.secret = env["HFGCS_REVIEW_SECRET"].encode()
    return s


# ------------------------------------------------------------------ CSRF
def new_csrf_token(secret: bytes) -> str:
    nonce = secrets.token_urlsafe(24)
    sig = hmac.new(secret, nonce.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{nonce}.{sig}"


def csrf_valid(secret: bytes, cookie_token: Optional[str], form_token: Optional[str]) -> bool:
    """Double-submit token: the form value must equal the cookie, and be signed by us."""
    if not cookie_token or not form_token or not hmac.compare_digest(cookie_token, form_token):
        return False
    nonce, _, sig = form_token.partition(".")
    want = hmac.new(secret, nonce.encode(), hashlib.sha256).hexdigest()[:32]
    return bool(nonce) and hmac.compare_digest(sig, want)


def same_origin(headers, host: str) -> bool:
    """True if the request provably comes from a page on this origin (Origin, else Referer;
    Sec-Fetch-Site when the browser sends it). Requests with neither header are refused."""
    host = (host or "").lower()
    sfs = headers.get("Sec-Fetch-Site")
    if sfs and sfs not in ("same-origin", "none"):
        return False
    origin = headers.get("Origin")
    if origin is not None:
        return origin != "null" and urlsplit(origin).netloc.lower() == host
    ref = headers.get("Referer")
    return bool(ref) and urlsplit(ref).netloc.lower() == host


# ------------------------------------------------------------------ rendering
def esc(v) -> str:
    return "" if v is None else html.escape(str(v), quote=True)


def _page(title: str, body: str) -> str:
    return ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<meta name=robots content='noindex,nofollow'>"
            f"<title>{esc(title)}</title><link rel=stylesheet href='/static/review.css'></head><body>"
            "<header class=top><h1>HFGCS review (private)</h1><a href='/'>Recordings</a>"
            "<a href='/export/gold.jsonl'>Gold labels (JSONL)</a></header>"
            f"<main>{body}</main></body></html>")


def _fmt(v, nd=1, unit="") -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def render_message(m: dict, *, csrf: str, types, editor: str, cf_editor: Optional[str], ret: str) -> str:
    opts = "".join(f"<option value='{esc(c)}'{' selected' if c == m['msg_type'] else ''}>{esc(c)}</option>"
                   for c, _ in types)
    state = ("<span class='badge ok'>checked</span>" if m["checked"]
             else "<span class='badge todo'>unchecked</span>")
    who = (f"<span class=ro>editing as <b>{esc(cf_editor)}</b> (Cloudflare Access)</span>" if cf_editor else
           f"<label>Your name <input name=editor value='{esc(editor)}' maxlength=200 required "
           "autocomplete=nickname></label>")
    conf = m["confidence"]
    return f"""<form class=msg method=post action='/message/{m['id']}' id='m{m['id']}'>
<input type=hidden name=csrf value='{esc(csrf)}'><input type=hidden name=version value='{m['version']}'>
<input type=hidden name=return value='{esc(ret)}'><input type=hidden name=checked_present value=1>
<div class=ro>Message #{m['id']} · heard {esc(m['heard_utc'])} · {state} · origin {esc(m['origin'])}
 · group <b>{_fmt(m['group_no'])}</b> · confidence <b>{'–' if conf is None else f'{conf:.2f}'}</b>
 · {_fmt(m['text_length'])} chars{' · corrected' if m['corrected'] else ''}</div>
<div class=fields>
<label>Type <select name=msg_type>{opts}</select></label>
<label>Callsign <input name=callsign value='{esc(m['callsign'])}' maxlength=64></label>
<label>Prefix <input name=prefix value='{esc(m['prefix'])}' maxlength=8 size=4></label>
<label>Length <input name=length value='{esc(m['length'])}' inputmode=numeric pattern='[0-9]*' size=4></label>
<label>Addressee <input name=addressee value='{esc(m['addressee'])}' maxlength=64></label>
</div>
<div class=wide><label>Message text <textarea name=message_text rows=2 spellcheck=false>
{esc(m['message_text'])}</textarea></label></div>
<div class=wide><label>Transcript <textarea name=transcript rows=4>
{esc(m['transcript'])}</textarea></label></div>
{f"<div class=ro>machine copy: <code>{esc(m['machine_text'])}</code></div>" if m['machine_text'] else ''}
{f"<div class=ro>newer machine decode ({esc(m['decoder_version'])}), not applied over your edits: "
 f"<code>{esc(m['pending_machine_text'])}</code></div>"
 if m['pending_machine_text'] and m['pending_machine_text'] != m['message_text'] else ''}
<div class=wide><label>Notes <textarea name=notes rows=2>{esc(m['review_notes'])}</textarea></label></div>
<div class=actions><label><input type=checkbox name=checked value=1{' checked' if m['checked'] else ''}>
 Checked: transcript and fields are correct (gold)</label>{who}<button type=submit>Save</button>
<a href='/message/{m['id']}/history'>history</a></div>
</form>"""


def render_list(con: sqlite3.Connection, q: dict, *, csrf: str, editor: str, cf_editor: Optional[str],
                notice: str = "") -> str:
    date = (q.get("date") or "").strip() or None
    freq_s = (q.get("freq") or "").strip()
    freq = float(freq_s) if freq_s else None
    unchecked = q.get("unchecked") == "1"
    page = max(1, int(q.get("page") or 1))
    total, recs = rdb.list_recordings(con, date=date, freq=freq, unchecked=unchecked,
                                      limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
    types = rdb.message_types(con)
    fopts = "<option value=''>all</option>" + "".join(
        f"<option value='{f:g}'{' selected' if freq is not None and abs(f - freq) < 0.05 else ''}>"
        f"{f:g} kHz</option>" for f in rdb.frequencies(con))
    keep = {k: v for k, v in (("date", date), ("freq", freq_s), ("unchecked", "1" if unchecked else None))
            if v}
    out = [notice, f"""<form class=filters method=get action='/'>
<label>Date (UTC) <input type=date name=date value='{esc(date)}'></label>
<label>Frequency <select name=freq>{fopts}</select></label>
<label style='flex-direction:row;gap:.35rem;align-items:center'><input type=checkbox name=unchecked value=1
{' checked' if unchecked else ''}> unchecked only</label>
<button type=submit>Filter</button> <a href='/'>reset</a>
<span class=ro>{total} recording(s)</span></form>"""]
    day = None
    for r in recs:
        d = r["start_utc"][:10]
        if d != day:
            day = d
            yymmdd = d[2:4] + d[5:7] + d[8:10]
            out.append(f"<h2 class=day>{esc(d)} UTC <a href='{esc(rdb.neet_timecard_url(d))}' "
                       f"rel='noreferrer noopener' target=_blank>NEET INTEL daily timecard "
                       f"({esc(yymmdd)})</a></h2>")
        ret = "/?" + urlencode({**keep, "page": page}) + f"#c{r['id']}"
        t = r["voice_start_s"]
        frag = f"#t={t:.2f}" if isinstance(t, (int, float)) and t > 0 else ""
        player = "<p>no audio file</p>"
        if r["audio_path"]:
            player = (f"<audio controls preload=none src='/audio/{r['id']}{frag}'></audio>"
                      f"<p>{esc(Path(r['audio_path']).name)}</p>")
        msgs = "".join(render_message(m, csrf=csrf, types=types, editor=editor, cf_editor=cf_editor, ret=ret)
                       for m in r["messages"])
        if not r["messages"]:
            msgs = (f"<form method=post action='/capture/{r['id']}/message' class=msg>"
                    f"<input type=hidden name=csrf value='{esc(csrf)}'>"
                    f"<input type=hidden name=return value='{esc(ret)}'>"
                    "<p class=ro>No decoded message for this recording.</p>"
                    + (f"<span class=ro>as <b>{esc(cf_editor)}</b></span>" if cf_editor else
                       f"<label class=ro>Your name <input name=editor value='{esc(editor)}' required>"
                       "</label>")
                    + " <button type=submit>Add a message to transcribe</button></form>")
        out.append(f"""<article class=rec id='c{r['id']}'><div class=rec-head>
<b>{esc(r['start_utc'][11:19])} UTC</b><span>{_fmt(r['frequency_khz'], 1)} kHz {esc(r['mode'])}</span>
<span>{esc(r['receiver'])}</span><span>{_fmt(r['duration_sec'], 1, ' s')}</span>
<span>SNR {_fmt(r['snr_db'], 1, ' dB')} mean / {_fmt(r['snr_peak_db'], 1, ' dB')} peak</span>
<span>capture #{r['id']}</span></div>
<div class=rec-body><div class=player>{player}</div><div class=msgs>{msgs}</div></div></article>""")
    if not recs:
        out.append("<p>No recordings match.</p>")
    pages = []
    if page > 1:
        pages.append(f"<a href='/?{esc(urlencode({**keep, 'page': page - 1}))}'>← newer</a>")
    if page * PAGE_SIZE < total:
        pages.append(f"<a href='/?{esc(urlencode({**keep, 'page': page + 1}))}'>older →</a>")
    out.append(f"<nav class=pager>{''.join(pages)}</nav>")
    return _page("HFGCS review", "\n".join(out))


def render_history(con, message_id: int) -> str:
    rows = rdb.history(con, message_id)
    tr = "".join(f"<tr><td>{esc(h['edited_utc'])}</td><td>{esc(h['editor'])} ({esc(h['editor_source'])})</td>"
                 f"<td>{esc(h['table_name'])}.{esc(h['field'])}</td><td>{esc(h['old_value'])}</td>"
                 f"<td>{esc(h['new_value'])}</td></tr>" for h in rows)
    return _page(f"History of message {message_id}",
                 f"<h2>Edit history, message #{message_id}</h2><table class=hist><tr><th>When (UTC)</th>"
                 f"<th>Who</th><th>Field</th><th>Old</th><th>New</th></tr>{tr}</table>"
                 + ("" if rows else "<p>No edits yet.</p>"))


# ------------------------------------------------------------------ HTTP
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "hfgcs-review"
    sys_version = ""
    settings: Settings  # set by make_server

    # --- helpers
    def _con(self) -> sqlite3.Connection:
        return rdb.connect(self.settings.db)

    def _cookies(self) -> SimpleCookie:
        c = SimpleCookie()
        try:
            c.load(self.headers.get("Cookie", ""))
        except CookieError:
            pass
        return c

    def _cookie(self, name: str) -> Optional[str]:
        m = self._cookies().get(name)
        return m.value if m else None

    def _headers_common(self, ctype: str, length: int, *, cache: str = "no-store") -> None:
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'self'; media-src 'self'; img-src 'self'; "
                         "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")

    def _send(self, status: int, body: str | bytes, ctype: str = "text/html; charset=utf-8",
              extra: Optional[dict] = None, cookies: Optional[list[str]] = None) -> None:
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self._headers_common(ctype, len(data))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        for c in cookies or []:
            self.send_header("Set-Cookie", c)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, status: int, msg: str) -> None:
        self._send(status, _page("Error", f"<p class=err>{esc(msg)}</p><p><a href='/'>back</a></p>"))

    def _host_ok(self) -> bool:
        allowed = self.settings.host_whitelist()
        return allowed is None or (self.headers.get("Host") or "").lower() in allowed

    def _csrf_for_page(self) -> tuple[str, list[str]]:
        tok = self._cookie(CSRF_COOKIE)
        if tok and csrf_valid(self.settings.secret, tok, tok):
            return tok, []
        tok = new_csrf_token(self.settings.secret)
        return tok, [f"{CSRF_COOKIE}={tok}; Path=/; HttpOnly; SameSite=Strict"]

    def _cf_editor(self) -> Optional[str]:
        if not self.settings.trust_cf_access:
            return None
        v = (self.headers.get(CF_ACCESS_HEADER) or "").strip()
        return v[:200] or None

    # --- dispatch
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.BAD_REQUEST, "unexpected Host header")
        url = urlsplit(self.path)
        path = url.path
        q = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        try:
            if path == "/static/review.css":
                return self._send(200, CSS, "text/css; charset=utf-8", {"Cache-Control": "max-age=300"})
            if path == "/healthz":
                return self._send(200, "ok\n", "text/plain; charset=utf-8")
            if path == "/":
                csrf, cookies = self._csrf_for_page()
                cf = self._cf_editor()
                editor = self._cookie(EDITOR_COOKIE) or self.settings.default_editor
                notice = ("<p class=ro>Saved.</p>" if q.get("saved") else "")
                with self._con() as con:
                    body = render_list(con, q, csrf=csrf, editor=editor, cf_editor=cf, notice=notice)
                return self._send(200, body, cookies=cookies)
            m = re.fullmatch(r"/audio/(\d{1,12})", path)
            if m:
                return self._audio(int(m.group(1)))
            m = re.fullmatch(r"/message/(\d{1,12})/history", path)
            if m:
                with self._con() as con:
                    return self._send(200, render_history(con, int(m.group(1))))
            if path == "/export/gold.jsonl":
                import io
                buf = io.StringIO()
                with self._con() as con:
                    rdb.write_gold_jsonl(con, buf, self.settings.recordings)
                return self._send(200, buf.getvalue(), "application/x-ndjson; charset=utf-8",
                                  {"Content-Disposition": "attachment; filename=hfgcs_gold_labels.jsonl"})
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, f"bad request: {e}")
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.BAD_REQUEST, "unexpected Host header")
        if not same_origin(self.headers, self.headers.get("Host", "")):
            return self._error(HTTPStatus.FORBIDDEN, "cross-origin request refused")
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/x-www-form-urlencoded":
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "form posts only")
        form = {k: v[-1] for k, v in parse_qs(self.rfile.read(n).decode("utf-8", "replace"),
                                                  keep_blank_values=True).items()}
        if not csrf_valid(self.settings.secret, self._cookie(CSRF_COOKIE), form.get("csrf")):
            return self._error(HTTPStatus.FORBIDDEN, "missing or invalid CSRF token; reload the page")
        cf = self._cf_editor()
        if cf:
            editor, source = cf, "cf_access"
        elif form.get("editor", "").strip():
            editor, source = form["editor"].strip(), "form"
        else:
            editor, source = self.settings.default_editor, "default"
        ret = form.get("return") or "/"
        if not ret.startswith("/") or ret.startswith("//") or "\\" in ret:
            ret = "/"
        cookies = [] if cf or source != "form" else [
            f"{EDITOR_COOKIE}={_cookie_safe(editor)}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000"]
        path = urlsplit(self.path).path
        try:
            m = re.fullmatch(r"/message/(\d{1,12})", path)
            if m:
                fields = {f: form.get(f) for f in rdb.EDITABLE if f in form}
                checked = (form.get("checked") == "1") if "checked_present" in form else None
                version = int(form["version"]) if form.get("version", "").isdigit() else None
                with self._con() as con:
                    rdb.save_review(con, int(m.group(1)), fields=fields, checked=checked,
                                    notes=form.get("notes"), editor=editor, editor_source=source,
                                    expected_version=version)
                return self._redirect(_with_saved(ret), cookies)
            m = re.fullmatch(r"/capture/(\d{1,12})/message", path)
            if m:
                with self._con() as con:
                    rdb.create_message(con, int(m.group(1)), editor=editor, editor_source=source)
                return self._redirect(ret, cookies)
        except rdb.ReviewError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except rdb.Conflict as e:
            return self._error(HTTPStatus.CONFLICT, str(e))
        except KeyError:
            return self._error(HTTPStatus.NOT_FOUND, "no such record")
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def _redirect(self, location: str, cookies: list[str]) -> None:
        self._send(HTTPStatus.SEE_OTHER, "", "text/plain", {"Location": location}, cookies)

    def _audio(self, capture_id: int) -> None:
        with self._con() as con:
            row = con.execute("SELECT audio_path FROM captures WHERE id = ? AND external_source IS NULL",
                              (capture_id,)).fetchone()
        f = rdb.resolve_audio(self.settings.recordings, row["audio_path"]) if row else None
        if f is None:
            return self._error(HTTPStatus.NOT_FOUND, "audio not found")
        size = f.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            mm = _RANGE.match(rng.strip())
            if not mm or (not mm.group(1) and not mm.group(2)):
                return self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "", "text/plain",
                                  {"Content-Range": f"bytes */{size}"})
            if mm.group(1):
                start = int(mm.group(1))
                end = min(int(mm.group(2)), size - 1) if mm.group(2) else size - 1
            else:
                start = max(0, size - int(mm.group(2)))
            if start > end or start >= size:
                return self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "", "text/plain",
                                  {"Content-Range": f"bytes */{size}"})
            status = 206
        length = end - start + 1
        self.send_response(status)
        self._headers_common(rdb.AUDIO_TYPES[f.suffix.lower()], length, cache="private, max-age=3600")
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(f, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def _cookie_safe(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9@._+\-]", "_", v)[:100]


def _with_saved(ret: str) -> str:
    path, _, frag = ret.partition("#")
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}saved=1" + (f"#{frag}" if frag else "")


def make_server(settings: Settings) -> ThreadingHTTPServer:
    """Build (but do not start) the server. Enforces the bind policy and checks the DB."""
    check_bind(settings.host, settings.allow_all_interfaces)
    con = rdb.connect(settings.db, migrate=settings.migrate)
    con.close()
    handler = type("BoundReviewHandler", (ReviewHandler,), {"settings": settings})
    host = settings.host.strip("[]")
    if host in ("", "*"):
        host = "0.0.0.0"
    srv_cls = ThreadingHTTPServer
    if ":" in host:
        import socket

        srv_cls = type("V6Server", (ThreadingHTTPServer,), {"address_family": socket.AF_INET6})
    srv = srv_cls((host, settings.port), handler)
    srv.daemon_threads = True
    return srv


def main(argv=None) -> int:
    env = settings_from_env()
    ap = argparse.ArgumentParser(prog="python -m hfgcs.review_app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(env.db), help="SQLite DB (env HFGCS_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the review web app")
    s.add_argument("--host", default=env.host,
                   help=f"bind address (env HFGCS_REVIEW_HOST, default {DEFAULT_HOST})")
    s.add_argument("--port", type=int, default=env.port,
                   help=f"(env HFGCS_REVIEW_PORT, default {DEFAULT_PORT})")
    s.add_argument("--recordings", default=str(env.recordings), help="audio root (env HFGCS_RECORDINGS)")
    s.add_argument("--allow-all-interfaces", action="store_true",
                   help="permit binding 0.0.0.0/:: (refused otherwise)")
    s.add_argument("--trust-cf-access", action="store_true", default=env.trust_cf_access,
                   help=f"take the editor name from {CF_ACCESS_HEADER} (env HFGCS_REVIEW_TRUST_CF_ACCESS=1)")
    s.add_argument("--migrate", action="store_true", help="apply pending DB migrations at start-up")
    e = sub.add_parser("export-gold", help="write checked messages as gold labels (JSONL)")
    e.add_argument("--out", default="-", help="output file or - for stdout")
    e.add_argument("--recordings", default=str(env.recordings),
                   help="audio root used to fill absolute audio_path (env HFGCS_RECORDINGS)")
    args = ap.parse_args(argv)
    if args.cmd == "export-gold":
        con = rdb.connect(args.db)
        if args.out == "-":
            n = rdb.write_gold_jsonl(con, sys.stdout, args.recordings)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                n = rdb.write_gold_jsonl(con, fh, args.recordings)
        print(f"exported {n} gold label(s)", file=sys.stderr)
        return 0
    env.db, env.host, env.port = Path(args.db), args.host, args.port
    env.recordings, env.allow_all_interfaces = Path(args.recordings), args.allow_all_interfaces
    env.trust_cf_access, env.migrate = args.trust_cf_access, args.migrate
    try:
        srv = make_server(env)
    except BindRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"HFGCS review app on http://{args.host}:{args.port}/  db={env.db}  recordings={env.recordings}"
          f"  cf-access={'trusted' if env.trust_cf_access else 'ignored'}", file=sys.stderr, flush=True)
    if env.host_whitelist() is None:
        print("warning: bound to all interfaces with no HFGCS_REVIEW_ALLOWED_HOSTS; Host header not checked",
              file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
