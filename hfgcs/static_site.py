"""Minimal static server for the PUBLIC project page (public_site/). Stdlib only.

Serves files from one directory and nothing else: no directory listings, no dotfiles,
only a short list of static file types, GET/HEAD only. It has no access to the database,
captured audio or the private review page, and must stay that way (tests enforce it).

    python -m hfgcs.static_site            # 127.0.0.1:8091 unless HFGCS_PUBLIC_HOST/PORT
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "public_site"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
ALL_INTERFACES = {"0.0.0.0", "::", "[::]", "", "*"}
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}
CSP = ("default-src 'none'; style-src 'self'; img-src 'self'; font-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def check_bind(host: str, allow_all_interfaces: bool) -> None:
    if host.strip() in ALL_INTERFACES and not allow_all_interfaces:
        raise SystemExit(f"refusing to bind to {host or '(empty)'}: pass --allow-all-interfaces to expose "
                         "the public page on every interface (set HFGCS_PUBLIC_HOST to one address instead)")


def resolve(root: Path, url_path: str) -> Optional[Path]:
    """Map a URL path to a file under ``root`` or None. ``/`` and ``/dir/`` map to
    ``index.html``; anything hidden, outside root, or of another type is None."""
    path = unquote(urlsplit(url_path).path)
    if "\x00" in path or "\\" in path:
        return None
    parts = [p for p in path.split("/") if p]
    if any(p.startswith(".") for p in parts):
        return None
    rootr = root.resolve()
    target = rootr.joinpath(*parts) if parts else rootr
    try:
        target = target.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if target.is_dir():
        if not path.endswith("/"):
            return None
        target = target / "index.html"
        if not target.is_file():
            return None  # never list a directory
    if not target.is_relative_to(rootr) or not target.is_file() or target.suffix.lower() not in TYPES:
        return None
    return target


class StaticHandler(BaseHTTPRequestHandler):
    server_version = "hfgcs-public"
    sys_version = ""
    root: Path = DEFAULT_DIR

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-cache") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        f = resolve(self.root, self.path)
        if f is None:
            page = self.root / "404.html"
            body = page.read_bytes() if page.is_file() else b"Not found\n"
            return self._send(HTTPStatus.NOT_FOUND, body,
                              TYPES[".html"] if page.is_file() else TYPES[".txt"])
        cache = "public, max-age=300" if f.suffix != ".html" else "no-cache"
        self._send(HTTPStatus.OK, f.read_bytes(), TYPES[f.suffix.lower()], cache)

    def do_HEAD(self) -> None:
        self.do_GET()

    def _not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _not_allowed


def make_server(directory: Path, host: str, port: int,
                allow_all_interfaces: bool = False) -> ThreadingHTTPServer:
    check_bind(host, allow_all_interfaces)
    directory = Path(directory)
    if not (directory / "index.html").is_file():
        raise SystemExit(f"{directory} has no index.html")
    handler = type("BoundStaticHandler", (StaticHandler,), {"root": directory.resolve()})
    h = host.strip("[]") or "0.0.0.0"
    cls = ThreadingHTTPServer
    if ":" in h:
        cls = type("V6Server", (ThreadingHTTPServer,), {"address_family": socket.AF_INET6})
    srv = cls((h, port), handler)
    srv.daemon_threads = True
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hfgcs.static_site", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.environ.get("HFGCS_PUBLIC_DIR", str(DEFAULT_DIR)))
    ap.add_argument("--host", default=os.environ.get("HFGCS_PUBLIC_HOST", DEFAULT_HOST))
    ap.add_argument("--port", type=int, default=int(os.environ.get("HFGCS_PUBLIC_PORT", DEFAULT_PORT)))
    ap.add_argument("--allow-all-interfaces", action="store_true")
    a = ap.parse_args(argv)
    srv = make_server(Path(a.dir), a.host, a.port, a.allow_all_interfaces)
    print(f"HFGCS public page on http://{a.host}:{a.port}/ serving {Path(a.dir).resolve()}",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
