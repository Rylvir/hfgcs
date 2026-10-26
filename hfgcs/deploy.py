"""Safe deploys on hilltop: restart only what changed, never mid-capture, never in a quiet window.

Why: OpenWebRX stops the SDR when its last client disconnects, so every restart
of ``hfgcs-8992`` (OpenWebRX's only client) leaves a 2-3 s hole in the audio.
A KiwiSDR restart drops a volunteer's slot and opens a new session. So a deploy
must restart a recorder only when its own code changed, only while it is idle,
never inside a known event window, and the Kiwi units one at a time.

Entry point: ``deploy/deploy.sh`` (runs ``scripts/hfgcs_deploy.py``); see
docs/deploy.md. This module is stdlib only and every external effect (git,
systemctl, the clock) goes through a small object the tests replace.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path

from .runstate import parse_utc, read_state, runtime_dir, safe_name, state_path, utc_iso

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

# -- units and roles -------------------------------------------------------------------------

OPENWEBRX, KIWI, REVIEW, PUBLIC = "openwebrx", "kiwisdr", "review", "public"
RECORDERS = frozenset({OPENWEBRX, KIWI})
WEB = frozenset({REVIEW, PUBLIC})
ALL_ROLES = RECORDERS | WEB

# Changed path (fnmatch pattern, repo-relative) -> roles whose units must restart.
# First match wins. Anything that matches nothing (docs, tests, decoding/, scripts/,
# the deploy tool itself, ...) restarts nothing.
PATH_RULES: list[tuple[str, frozenset[str]]] = [
    ("hfgcs/sources/openwebrx.py", frozenset({OPENWEBRX})),
    ("hfgcs/sources/kiwisdr.py", frozenset({KIWI})),
    ("hfgcs/sources/base.py", RECORDERS),
    ("hfgcs/sources/__init__.py", RECORDERS),
    ("hfgcs/dsp.py", RECORDERS),  # shared DSP
    ("hfgcs/capture.py", RECORDERS),  # squelch/recording pipeline
    ("hfgcs/runstate.py", RECORDERS),
    ("hfgcs/config.py", RECORDERS),
    ("hfgcs_scanner.py", RECORDERS),
    ("hfgcs/review_app.py", frozenset({REVIEW})),
    ("hfgcs/review_db.py", frozenset({REVIEW})),
    ("hfgcs/db_migrations.py", frozenset({REVIEW})),
    ("migrations/*", frozenset({REVIEW})),
    ("hfgcs/static_site.py", frozenset({PUBLIC})),
    ("public_site/*", frozenset({PUBLIC})),
    ("deploy/bin/hfgcs-python", WEB),  # the web units' ExecStart wrapper
    ("hfgcs/__init__.py", ALL_ROLES),
]
UNIT_FILE_DIR = "deploy/systemd/"


def role_of(unit: str) -> str | None:
    """Role by naming convention: hfgcs-<kHz> (OpenWebRX), hfgcs-kiwi-*, hfgcs-review, hfgcs-public."""
    unit = safe_name(unit)
    if unit == "hfgcs-review":
        return REVIEW
    if unit == "hfgcs-public":
        return PUBLIC
    if unit.startswith("hfgcs-kiwi-"):
        return KIWI
    if re.fullmatch(r"hfgcs-\d+(\.\d+)?", unit):
        return OPENWEBRX
    return None


def roles_for_path(path: str) -> frozenset[str]:
    for pattern, roles in PATH_RULES:
        if fnmatch.fnmatchcase(path, pattern):
            return roles
    return frozenset()


def unit_file_unit(path: str) -> str | None:
    """``deploy/systemd/hfgcs-review.service`` -> ``hfgcs-review``."""
    if path.startswith(UNIT_FILE_DIR) and path.endswith(".service") and "/" not in path[len(UNIT_FILE_DIR):]:
        return path[len(UNIT_FILE_DIR): -len(".service")]
    return None


def order_key(unit: str) -> tuple:
    """Web units first, then Kiwi units one by one, OpenWebRX (hfgcs-8992) last."""
    rank = {REVIEW: 0, PUBLIC: 1, None: 2, KIWI: 3, OPENWEBRX: 4}[role_of(unit)]
    return (rank, unit)


def dependencies(pyproject_text: str | None) -> tuple:
    """The runtime dependency lists of a pyproject.toml (None if missing/unparseable)."""
    if not pyproject_text:
        return ()
    try:
        project = _toml.loads(pyproject_text).get("project", {})
    except _toml.TOMLDecodeError:
        return ("<unparseable>", pyproject_text)
    return (
        tuple(project.get("dependencies", ())),
        tuple(sorted((k, tuple(v)) for k, v in project.get("optional-dependencies", {}).items())),
        project.get("requires-python"),
    )


@dataclass
class UnitPlan:
    unit: str
    role: str | None
    reasons: list[str] = field(default_factory=list)
    forced: bool = False
    unit_file: str | None = None  # repo path of a changed unit file to install

    @property
    def recorder(self) -> bool:
        return self.role in RECORDERS


def plan_units(
    changed: list[str],
    units: list[str],
    *,
    deps_changed: bool = False,
    force: tuple[str, ...] = (),
    pending: dict[str, list[str]] | None = None,
) -> tuple[list[UnitPlan], list[str]]:
    """Which of ``units`` must restart for ``changed`` paths, in restart order.

    Returns ``(plans, notes)``. ``force`` names units to restart whatever changed
    (``all`` = every unit); ``pending`` carries restarts deferred by an earlier run.
    """
    known = {safe_name(u) for u in units}
    plans: dict[str, UnitPlan] = {}
    notes: list[str] = []

    def add(unit: str, reason: str) -> UnitPlan:
        p = plans.setdefault(unit, UnitPlan(unit, role_of(unit)))
        if reason not in p.reasons:
            p.reasons.append(reason)
        return p

    for path in changed:
        uf = unit_file_unit(path)
        if uf is not None:
            if uf in known:
                add(uf, f"unit file {path}").unit_file = path
            else:
                notes.append(f"{path}: unit {uf} is not installed here; not installed or started")
            continue
        roles = roles_for_path(path)
        for unit in known:
            if role_of(unit) in roles:
                add(unit, path)
    if deps_changed:
        for unit in known:
            if role_of(unit) is not None:
                add(unit, "pyproject.toml dependencies")
    for unit, reasons in (pending or {}).items():
        unit = safe_name(unit)
        if unit in known:
            for r in reasons:
                add(unit, f"pending from an earlier deploy: {r}")
        else:
            notes.append(f"pending restart of {unit} dropped: unit not found")
    for f in force:
        targets = sorted(known) if f == "all" else [safe_name(f)]
        for unit in targets:
            if unit not in known:
                notes.append(f"--force {f}: no such unit")
                continue
            add(unit, "--force").forced = True
    return sorted(plans.values(), key=lambda p: order_key(p.unit)), notes


# -- quiet (event) windows -------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    start: datetime | None = None  # absolute window (UTC)
    end: datetime | None = None
    daily_start: dtime | None = None  # recurring daily window (UTC), may wrap midnight
    daily_end: dtime | None = None
    note: str = ""

    def contains(self, now: datetime) -> bool:
        now = now.astimezone(timezone.utc)
        if self.start is not None:
            return self.start <= now < self.end
        t = now.time().replace(tzinfo=None)
        if self.daily_start < self.daily_end:
            return self.daily_start <= t < self.daily_end
        return t >= self.daily_start or t < self.daily_end  # wraps midnight

    def end_after(self, now: datetime) -> datetime:
        """When this window (containing ``now``) ends."""
        if self.start is not None:
            return self.end
        end = datetime.combine(now.astimezone(timezone.utc).date(), self.daily_end, timezone.utc)
        return end if end > now else end + timedelta(days=1)

    @property
    def label(self) -> str:
        if self.start is not None:
            text = f"{utc_iso(self.start)[:-8]}Z/{utc_iso(self.end)[:-8]}Z"
        else:
            text = f"daily {self.daily_start:%H:%M}-{self.daily_end:%H:%M} UTC"
        return f"{text} ({self.note})" if self.note else text


def _utc(text: str) -> datetime:
    t = text.strip()
    if not re.search(r"(Z|[+-]\d\d:?\d\d)$", t):
        raise ValueError(f"window time {text!r} must be UTC: end it with Z (e.g. 2026-10-26T23:45Z)")
    if re.fullmatch(r".*T\d\d:\d\d(Z|[+-].*)", t):  # no seconds: Python 3.10 needs them
        t = re.sub(r"(T\d\d:\d\d)", r"\1:00", t, count=1)
    return parse_utc(t)


def parse_window(spec: str, note: str = "") -> Window:
    """``2026-10-26T23:45Z/2026-10-27T07:00Z`` (absolute) or ``23:45-07:00`` (daily, UTC)."""
    spec = spec.strip()
    if "/" in spec:
        a, b = spec.split("/", 1)
        start, end = _utc(a), _utc(b)
        if end <= start:
            raise ValueError(f"window {spec!r} ends before it starts")
        return Window(start=start, end=end, note=note)
    m = re.fullmatch(r"(\d{1,2}):(\d\d)\s*-\s*(\d{1,2}):(\d\d)(\s*Z|\s*UTC)?", spec)
    if not m:
        raise ValueError(f"bad window {spec!r}: use START/END in UTC or a daily HH:MM-HH:MM (UTC)")
    a = dtime(int(m.group(1)), int(m.group(2)))
    b = dtime(int(m.group(3)), int(m.group(4)))
    if a == b:
        raise ValueError(f"daily window {spec!r} is empty")
    return Window(daily_start=a, daily_end=b, note=note)


def _entries(value, key: str) -> list[tuple[str, str]]:
    out = []
    for item in value or ():
        if isinstance(item, str):
            out.append((item, ""))
        elif isinstance(item, dict) and isinstance(item.get("span"), str):
            out.append((item["span"], str(item.get("note", ""))))
        else:
            raise ValueError(f"{key}: each entry is a string or {{span = \"...\", note = \"...\"}}")
    return out


def load_windows(path: Path | None) -> list[Window]:
    """Read quiet windows from TOML (``windows = [...]``, ``daily = [...]``); missing file: none."""
    if path is None or not Path(path).exists():
        return []
    with open(path, "rb") as fh:
        data = _toml.load(fh)
    unknown = set(data) - {"windows", "daily"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} (use 'windows' and 'daily')")
    out = [parse_window(s, n) for s, n in _entries(data.get("windows"), "windows")]
    for s, n in _entries(data.get("daily"), "daily"):
        w = parse_window(s, n)
        if w.start is not None:
            raise ValueError(f"daily entry {s!r} must be HH:MM-HH:MM")
        out.append(w)
    return out


def active_window(windows: list[Window], now: datetime) -> Window | None:
    for w in windows:
        if w.contains(now):
            return w
    return None


# -- systemd -----------------------------------------------------------------------------------

class Systemctl:
    """``systemctl --user`` (binary from $HFGCS_SYSTEMCTL for tests)."""

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("HFGCS_SYSTEMCTL") or "systemctl"
        self.calls: list[list[str]] = []

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        cmd = [self.binary, "--user", *args]
        self.calls.append(list(args))
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    def list_units(self) -> list[str]:
        names = set()
        for args in (("list-unit-files", "hfgcs-*.service", "--no-legend", "--no-pager"),
                     ("list-units", "hfgcs-*.service", "--all", "--no-legend", "--no-pager")):
            r = self._run(*args)
            for line in r.stdout.splitlines():
                for word in line.replace("●", " ").split():
                    if word.startswith("hfgcs") and word.endswith(".service"):
                        names.add(word[: -len(".service")])
        return sorted(names)

    def is_active(self, unit: str) -> str:
        return self._run("is-active", f"{unit}.service").stdout.strip() or "unknown"

    def main_pid(self, unit: str) -> int:
        out = self._run("show", "-p", "MainPID", "--value", f"{unit}.service").stdout.strip()
        try:
            return int(out)
        except ValueError:
            return 0

    def restart(self, unit: str) -> tuple[bool, str]:
        r = self._run("restart", f"{unit}.service")
        return r.returncode == 0, (r.stderr or r.stdout).strip()

    def daemon_reload(self) -> tuple[bool, str]:
        r = self._run("daemon-reload")
        return r.returncode == 0, (r.stderr or r.stdout).strip()


# -- waiting on recorders ----------------------------------------------------------------------

@dataclass
class Clock:
    monotonic: object = time.monotonic
    sleep: object = time.sleep
    now: object = lambda: datetime.now(timezone.utc)  # noqa: E731


@dataclass
class Status:
    """What the recorder's state file says about the running process."""

    kind: str  # "idle", "capturing", "unknown", "inactive"
    detail: str = ""
    idle_s: float = 0.0
    last_audio: datetime | None = None


def unit_status(unit: str, systemctl, now: datetime, state_dir: Path | None = None) -> Status:
    active = systemctl.is_active(unit)
    if active not in ("active", "reloading", "activating"):
        return Status("inactive", f"unit is {active}")
    pid = systemctl.main_pid(unit)
    path = state_path(unit, state_dir)
    data = read_state(path)
    if data is None:
        return Status("unknown", f"no state file {path} (recorder predates state files?)")
    if pid and data.get("pid") != pid:
        return Status("unknown", f"state file {path} is from pid {data.get('pid')}, unit runs pid {pid}")
    try:
        since = parse_utc(data.get("since_utc")) or now
        last_audio = parse_utc(data.get("last_audio_utc"))
    except ValueError:
        return Status("unknown", f"state file {path} has bad timestamps")
    state = data.get("state")
    if state == "capturing":
        return Status("capturing", f"capturing since {utc_iso(since)}", 0.0, last_audio)
    if state in ("idle", "starting"):
        idle = max(0.0, (now - since).total_seconds())
        return Status("idle", f"{state} for {idle:.0f} s", idle, last_audio)
    return Status("unknown", f"state file says {state!r}")


@dataclass
class Outcome:
    unit: str
    result: str  # restarted, skipped, deferred, failed, dry-run
    detail: str = ""


def wait_idle(
    unit: str,
    systemctl,
    clock: Clock,
    *,
    min_idle_s: float,
    timeout_s: float,
    windows: list[Window],
    forced: bool,
    state_dir: Path | None = None,
    poll_s: float = 2.0,
    say=print,
) -> Outcome | None:
    """Block until ``unit`` has been idle >= ``min_idle_s``; None = go ahead, else why not.

    A capture in progress is never interrupted, not even with --force; --force only
    lifts quiet windows and accepts a recorder that writes no state file yet.
    """
    deadline = clock.monotonic() + timeout_s
    announced = None
    while True:
        now = clock.now()
        w = None if forced else active_window(windows, now)
        if w is not None:
            until = utc_iso(w.end_after(now))
            return Outcome(unit, "deferred", f"inside quiet window {w.label} (until {until})")
        st = unit_status(unit, systemctl, now, state_dir)
        if st.kind == "inactive":
            return Outcome(unit, "skipped", f"{st.detail}; left as it is")
        if st.kind == "unknown":
            if forced:
                say(f"  {unit}: {st.detail}; --force given, restarting without an idle check")
                return None
            return Outcome(unit, "skipped", f"{st.detail}; cannot tell if it is capturing "
                                            f"(use --force {unit} during a quiet time)")
        if st.kind == "idle" and st.idle_s >= min_idle_s:
            return None
        if clock.monotonic() >= deadline:
            return Outcome(unit, "skipped", f"still {st.detail} after waiting {timeout_s:.0f} s")
        if announced != st.kind:
            say(f"  {unit}: {st.detail}; waiting for {min_idle_s:.0f} s idle (up to {timeout_s:.0f} s)")
            announced = st.kind
        clock.sleep(poll_s)


def wait_streaming(
    unit: str,
    systemctl,
    clock: Clock,
    *,
    restarted_at: datetime,
    timeout_s: float = 60.0,
    state_dir: Path | None = None,
    poll_s: float = 2.0,
) -> str:
    """After a restart: wait until the unit is active and its new process has audio, or the timeout."""
    deadline = clock.monotonic() + timeout_s
    while True:
        active = systemctl.is_active(unit)
        if active == "active":
            pid = systemctl.main_pid(unit)
            data = read_state(state_path(unit, state_dir)) or {}
            try:
                last = parse_utc(data.get("last_audio_utc"))
            except ValueError:
                last = None
            if data.get("pid") == pid and last is not None and last >= restarted_at:
                return f"streaming (first audio {utc_iso(last)})"
        elif active == "failed":
            return "unit failed after restart"
        if clock.monotonic() >= deadline:
            return f"{active}, no audio yet after {timeout_s:.0f} s; moved on"
        clock.sleep(poll_s)


# -- git and the deploy as a whole ----------------------------------------------------------------

class Git:
    def __init__(self, repo: Path):
        self.repo = Path(repo)

    def run(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
        return r.stdout.strip()

    def sha(self, ref: str) -> str:
        return self.run("rev-parse", "--verify", f"{ref}^{{commit}}")

    def show(self, rev: str, path: str) -> str | None:
        cmd = ["git", "-C", str(self.repo), "show", f"{rev}:{path}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    def changed(self, a: str, b: str) -> list[str]:
        if a == b:
            return []
        out = self.run("diff", "--name-only", "--no-renames", a, b)
        return [ln for ln in out.splitlines() if ln]

    def dirty(self) -> list[str]:
        return [ln for ln in self.run("status", "--porcelain", "--untracked-files=no").splitlines() if ln]

    def is_ancestor(self, a: str, b: str) -> bool:
        cmd = ["git", "-C", str(self.repo), "merge-base", "--is-ancestor", a, b]
        return subprocess.run(cmd, capture_output=True).returncode == 0

    def fast_forward(self, target: str) -> None:
        branch = self.run("symbolic-ref", "-q", "--short", "HEAD", check=False)
        if branch:
            self.run("merge", "--ff-only", target)
        else:
            self.run("checkout", "-q", "--detach", target)


def default_state_home() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "hfgcs"


def default_db(repo: Path) -> Path:
    """$HFGCS_DB, else HFGCS_DB from ~/.config/hfgcs/review.env, else <repo>/data/hfgcs.db."""
    if os.environ.get("HFGCS_DB"):
        return Path(os.path.expanduser(os.environ["HFGCS_DB"]))
    env = Path(os.path.expanduser("~/.config/hfgcs/review.env"))
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.fullmatch(r"\s*HFGCS_DB\s*=\s*(.+?)\s*", line)
            if m:
                return Path(os.path.expanduser(m.group(1).strip("'\"")))
    return Path(repo) / "data" / "hfgcs.db"


def pending_migrations(db: Path, migrations_dir: Path) -> list[tuple[int, str, Path]] | None:
    """Pending migrations from ``migrations_dir`` (None if the DB does not exist)."""
    from .db_migrations import pending

    if not Path(db).exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pending(con, migrations_dir)
    finally:
        con.close()


def backup_db(db: Path, dest_dir: Path, tag: str, now: datetime) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{Path(db).stem}-{now:%Y%m%dT%H%M%SZ}-{tag}.db"
    src = sqlite3.connect(str(db))
    try:
        out = sqlite3.connect(str(dest))
        try:
            src.backup(out)
        finally:
            out.close()
    finally:
        src.close()
    return dest


def apply_migrations_to(db: Path, migrations_dir: Path) -> list[int]:
    from .db_migrations import apply_migrations

    con = sqlite3.connect(str(db))
    try:
        return apply_migrations(con, migrations_dir)
    finally:
        con.close()


class PendingStore:
    """Restarts that were skipped or deferred, so the next run still does them."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict[str, list[str]]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return {k: list(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def save(self, data: dict[str, list[str]]) -> None:
        from .runstate import write_json_atomic

        if data:
            write_json_atomic(self.path, data)
        else:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


@dataclass
class Options:
    repo: Path
    ref: str = "origin/main"
    remote: str = "origin"
    fetch: bool = True
    dry_run: bool = False
    force: tuple[str, ...] = ()
    from_sha: str | None = None  # set when re-executed after the fast-forward
    min_idle_s: float = 30.0
    idle_timeout_s: float = 600.0
    stream_timeout_s: float = 60.0
    poll_s: float = 2.0
    windows_file: Path | None = None
    extra_windows: tuple[str, ...] = ()
    db: Path | None = None
    migrate: bool = True
    pip_spec: str = ".[openwebrx]"
    python: str = sys.executable
    install_units: bool = True
    unit_dir: Path = Path(os.path.expanduser("~/.config/systemd/user"))
    state_home: Path = field(default_factory=default_state_home)
    state_dir: Path | None = None  # recorder state files (default $XDG_RUNTIME_DIR/hfgcs)
    units: tuple[str, ...] = ()  # explicit unit list (default: discover hfgcs-*.service)


class Deployer:
    def __init__(self, opts: Options, *, git=None, systemctl=None, clock: Clock | None = None,
                 run=subprocess.run, out=None):
        self.o = opts
        self.git = git or Git(opts.repo)
        self.sc = systemctl or Systemctl()
        self.clock = clock or Clock()
        self._run = run
        self._out = out or sys.stdout
        self.outcomes: list[Outcome] = []
        self.pending = PendingStore(opts.state_home / "pending-restarts.json")

    def say(self, msg: str = "") -> None:
        print(msg, file=self._out, flush=True)

    # -- phase 1: update the checkout ------------------------------------------------------
    def update(self) -> tuple[str, str]:
        """Fetch and fast-forward. Returns (deployed sha before, target sha)."""
        dirty = self.git.dirty()
        if dirty and not self.o.dry_run:
            raise RuntimeError("the checkout has local changes; commit or stash them first:\n  "
                               + "\n  ".join(dirty))
        old = self.git.sha("HEAD")
        if self.o.fetch:
            self.git.run("fetch", "--quiet", self.o.remote)
        target = self.git.sha(self.o.ref)
        if old != target and not self.git.is_ancestor(old, target):
            raise RuntimeError(f"{self.o.ref} ({target[:7]}) is not a fast-forward of the deployed "
                               f"{old[:7]}; refusing (check out the wanted commit by hand)")
        if not self.o.dry_run and old != target:
            self.git.fast_forward(target)
        return old, target

    # -- phase 2: act on the change --------------------------------------------------------
    def windows(self) -> list[Window]:
        ws = load_windows(self.o.windows_file)
        ws += [parse_window(s) for s in self.o.extra_windows]
        return ws

    def run(self) -> int:
        o = self.o
        if o.from_sha:
            old, target = self.git.sha(o.from_sha), self.git.sha("HEAD")
        else:
            old, target = self.update()
        mode = " (dry run: nothing is changed)" if o.dry_run else ""
        self.say(f"hfgcs deploy{mode}")
        self.say(f"  deployed before: {old}")
        n = len(self.git.run("rev-list", f"{old}..{target}").split()) if old != target else 0
        self.say(f"  target:          {target} ({o.ref}; {n} new commit{'s' * (n != 1)})")
        changed = self.git.changed(old, target)
        windows = self.windows()
        now = self.clock.now()
        w = active_window(windows, now)
        self.say(f"  quiet window now: {w.label if w else 'none'} ({len(windows)} configured)")
        self.say(f"  changed files: {len(changed)}")
        for p in changed:
            self.say(f"    {p}")

        old_py, new_py = self.git.show(old, "pyproject.toml"), self.git.show(target, "pyproject.toml")
        reinstall = old_py != new_py
        deps_changed = reinstall and dependencies(old_py) != dependencies(new_py)
        errors = 0
        if reinstall:
            cmd = [o.python, "-m", "pip", "install", "--quiet", "-e", o.pip_spec]
            self.say(f"  pyproject.toml changed: reinstall ({' '.join(cmd)})"
                     + ("; dependencies changed, every unit restarts" if deps_changed else ""))
            if not o.dry_run:
                r = self._run(cmd, cwd=str(o.repo), capture_output=True, text=True)
                if r.returncode != 0:
                    self.say(f"  REINSTALL FAILED; no unit restarted:\n{r.stderr or r.stdout}")
                    self._record(old, target, changed)
                    return 1
        else:
            self.say("  pyproject.toml unchanged: no reinstall")

        if o.migrate and not self._migrate(old, now):
            self._record(old, target, changed)
            return 1

        units = list(o.units) or self.sc.list_units()
        pending = self.pending.load()
        plans, notes = plan_units(changed, units, deps_changed=deps_changed, force=o.force, pending=pending)
        for note in notes:
            self.say(f"  note: {note}")
        idle_units = sorted(set(map(safe_name, units)) - {p.unit for p in plans})
        self.say("  plan (one at a time, in this order):" if plans else "  plan: no unit needs a restart")
        for i, p in enumerate(plans, 1):
            how = (f"restart when idle >= {o.min_idle_s:.0f} s" if p.recorder else "restart")
            if p.unit_file:
                how = f"install {p.unit_file}, daemon-reload, " + how
            extra = ""
            if p.recorder:
                st = unit_status(p.unit, self.sc, now, o.state_dir)
                extra = f" [now: {st.kind}, {st.detail}]"
                if w is not None and not p.forced:
                    extra += " [would be DEFERRED: quiet window]"
            self.say(f"    {i}. {p.unit}: {how}{' (forced)' if p.forced else ''}; because "
                     f"{', '.join(p.reasons)}{extra}")
        if idle_units:
            self.say(f"  not restarted: {', '.join(idle_units)}")
        if o.dry_run:
            self.outcomes = [Outcome(p.unit, "dry-run", "would restart") for p in plans]
            self.summary()
            return 0

        errors += self._install_unit_files(plans)
        for p in plans:
            self.outcomes.append(self._restart(p, windows))
        left = {oc.unit: next(p.reasons for p in plans if p.unit == oc.unit)
                for oc in self.outcomes if oc.result in ("skipped", "deferred", "failed")}
        # A unit that is simply stopped needs no restart later.
        left = {u: r for u, r in left.items()
                if not any(oc.unit == u and "left as it is" in oc.detail for oc in self.outcomes)}
        self.pending.save({u: [x.removeprefix("pending from an earlier deploy: ") for x in r]
                           for u, r in left.items()})
        self.summary()
        self._record(old, target, changed)
        errors += sum(1 for oc in self.outcomes if oc.result == "failed")
        if errors:
            return 1
        return 3 if left else 0

    def _migrate(self, old: str, now: datetime) -> bool:
        o = self.o
        db = o.db or default_db(o.repo)
        try:
            todo = pending_migrations(db, o.repo / "migrations")
        except sqlite3.Error as exc:
            self.say(f"  migrations: cannot read {db}: {exc}; stopping before any restart")
            return False
        if todo is None:
            self.say(f"  migrations: no database at {db}; skipped")
            return True
        if not todo:
            self.say(f"  migrations: none pending ({db})")
            return True
        names = ", ".join(f"{v:04d}_{n}" for v, n, _ in todo)
        if o.dry_run:
            self.say(f"  migrations pending: {names} (would back up {db} first)")
            return True
        try:
            backup = backup_db(db, Path(db).parent / "backups", old[:7], now)
            self.say(f"  database backed up to {backup}")
            applied = apply_migrations_to(db, o.repo / "migrations")
        except (sqlite3.Error, RuntimeError, ValueError, OSError) as exc:
            self.say(f"  MIGRATION FAILED ({names}): {exc}; no unit restarted")
            return False
        self.say(f"  migrations applied: {', '.join(f'{v:04d}' for v in applied)}")
        return True

    def _install_unit_files(self, plans: list[UnitPlan]) -> int:
        changed = [p for p in plans if p.unit_file]
        if not changed:
            return 0
        if self.o.install_units:
            for p in changed:
                dest = self.o.unit_dir / f"{p.unit}.service"
                if not dest.exists():
                    self.say(f"  {p.unit}: {dest} not installed; repo unit file not copied")
                    continue
                src = self.o.repo / p.unit_file
                if dest.read_bytes() != src.read_bytes():
                    bak = dest.with_name(dest.name + f".bak-{self.clock.now():%Y%m%dT%H%M%SZ}")
                    shutil.copy2(dest, bak)
                    shutil.copyfile(src, dest)
                    self.say(f"  {p.unit}: installed {p.unit_file} (old copy kept as {bak.name})")
        ok, msg = self.sc.daemon_reload()
        self.say("  systemctl --user daemon-reload: " + ("ok" if ok else f"FAILED {msg}"))
        return 0 if ok else 1

    def _restart(self, p: UnitPlan, windows: list[Window]) -> Outcome:
        o = self.o
        self.say(f"- {p.unit}")
        if p.recorder:
            blocked = wait_idle(p.unit, self.sc, self.clock, min_idle_s=o.min_idle_s,
                                timeout_s=o.idle_timeout_s, windows=windows, forced=p.forced,
                                state_dir=o.state_dir, poll_s=o.poll_s, say=self.say)
            if blocked is not None:
                self.say(f"  {blocked.result.upper()}: {blocked.detail}")
                return blocked
        elif self.sc.is_active(p.unit) not in ("active", "activating", "reloading") and not p.forced:
            oc = Outcome(p.unit, "skipped", f"unit is {self.sc.is_active(p.unit)}; left as it is")
            self.say(f"  SKIPPED: {oc.detail}")
            return oc
        at = self.clock.now()
        ok, msg = self.sc.restart(p.unit)
        if not ok:
            self.say(f"  FAILED: systemctl --user restart {p.unit}: {msg}")
            return Outcome(p.unit, "failed", msg)
        detail = f"at {utc_iso(at)}"
        if p.recorder:
            ready = wait_streaming(p.unit, self.sc, self.clock, restarted_at=at,
                                   timeout_s=o.stream_timeout_s, state_dir=o.state_dir, poll_s=o.poll_s)
            detail += f"; {ready}"
            if ready.startswith("unit failed"):
                self.say(f"  FAILED: {detail}")
                return Outcome(p.unit, "failed", detail)
        self.say(f"  restarted {detail}")
        return Outcome(p.unit, "restarted", detail)

    def summary(self) -> None:
        self.say("summary:")
        if not self.outcomes:
            self.say("  nothing restarted")
        for result in ("restarted", "dry-run", "deferred", "skipped", "failed"):
            for oc in self.outcomes:
                if oc.result == result:
                    self.say(f"  {result:9s} {oc.unit}: {oc.detail}")
        if any(oc.result in ("deferred", "skipped") for oc in self.outcomes):
            self.say("  (deferred/skipped restarts are remembered; run deploy/deploy.sh again later)")

    def _record(self, old: str, target: str, changed: list[str]) -> None:
        try:
            self.o.state_home.mkdir(parents=True, exist_ok=True)
            entry = {"utc": utc_iso(self.clock.now()), "from": old, "to": target, "changed": changed,
                     "outcomes": [oc.__dict__ for oc in self.outcomes]}
            with open(self.o.state_home / "deploy-history.jsonl", "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            (self.o.state_home / "deployed-sha").write_text(target + "\n")
        except OSError as exc:  # pragma: no cover
            self.say(f"  (could not record the deploy: {exc})")


__all__ = [
    "Clock", "Deployer", "Options", "Outcome", "Status", "Systemctl", "UnitPlan", "Window",
    "active_window", "load_windows", "order_key", "parse_window", "plan_units", "role_of",
    "roles_for_path", "runtime_dir", "unit_status", "wait_idle", "wait_streaming",
]
