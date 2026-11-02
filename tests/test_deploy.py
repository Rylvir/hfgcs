"""Deploy script: path -> unit mapping, quiet windows, idle waits, ordering, dry run.

Everything external is fake: a throwaway git repo, a fake systemctl and a fake clock.
"""

import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hfgcs import deploy as d
from hfgcs.runstate import RunState, utc_iso, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 26, 20, 0, 0, tzinfo=timezone.utc)
KIWIS = [f"hfgcs-kiwi-{c}" for c in (11175, 4724, 6739, 13200, 15016)]
UNITS = ["hfgcs-8992", *KIWIS, "hfgcs-review", "hfgcs-public"]


# -- fakes -------------------------------------------------------------------------------------
class FakeClock(d.Clock):
    def __init__(self, start=T0):
        self.t = 0.0
        self.start = start
        self.hooks = []  # (at_t, fn) run when time passes at_t
        super().__init__(
            monotonic=lambda: self.t, sleep=self._sleep, now=lambda: self.start + timedelta(seconds=self.t)
        )

    def _sleep(self, s):
        self.t += s
        due = [h for h in self.hooks if h[0] <= self.t]
        self.hooks = [h for h in self.hooks if h[0] > self.t]
        for _, fn in sorted(due, key=lambda h: h[0]):
            fn()

    def at(self, t, fn):
        self.hooks.append((t, fn))


class FakeSystemctl:
    """Units with a pid each; restart writes a fresh state file and, ``audio_after`` s later, audio."""

    def __init__(self, clock, state_dir, units=UNITS, audio_after=6.0):
        self.clock, self.state_dir, self.audio_after = clock, Path(state_dir), audio_after
        self.units = {u: {"active": "active", "pid": 100 + i} for i, u in enumerate(units)}
        self.log = []

    def list_units(self):
        return sorted(self.units)

    def is_active(self, unit):
        return self.units[unit]["active"]

    def main_pid(self, unit):
        return self.units[unit]["pid"]

    def daemon_reload(self):
        self.log.append(("daemon-reload", None, self.clock.t))
        return True, ""

    def restart(self, unit):
        u = self.units[unit]
        u["pid"] += 1000
        u["active"] = "active"
        self.log.append(("restart", unit, self.clock.t))
        if d.role_of(unit) in d.RECORDERS:
            st = RunState(unit, self.state_dir / f"{unit}.state", now=self.clock.now, pid=u["pid"])
            st.set("idle", receiver="rx", channel_khz=1.0)
            self.clock.at(self.clock.t + self.audio_after, st.audio)
        return True, ""


def write_state(state_dir, unit, pid, state="idle", since=T0 - timedelta(minutes=5), last_audio=T0):
    write_json_atomic(Path(state_dir) / f"{unit}.state", {
        "schema": 1, "name": unit, "pid": pid, "state": state, "since_utc": utc_iso(since),
        "last_audio_utc": utc_iso(last_audio), "receiver": "rx",
    })


def idle_everywhere(sc, state_dir):
    for unit, u in sc.units.items():
        if d.role_of(unit) in d.RECORDERS:
            write_state(state_dir, unit, u["pid"])


# -- path -> unit mapping -------------------------------------------------------------------------
@pytest.mark.parametrize("path, roles", [
    ("hfgcs/sources/openwebrx.py", {"openwebrx"}),
    ("hfgcs/sources/kiwisdr.py", {"kiwisdr"}),
    ("hfgcs/dsp.py", {"openwebrx", "kiwisdr"}),
    ("hfgcs/capture.py", {"openwebrx", "kiwisdr"}),
    ("hfgcs/config.py", {"openwebrx", "kiwisdr"}),
    ("hfgcs/shadow.py", {"openwebrx", "kiwisdr"}),
    ("hfgcs/shadow_score.py", set()),
    ("hfgcs_scanner.py", {"openwebrx", "kiwisdr"}),
    ("hfgcs/sources/base.py", {"openwebrx", "kiwisdr"}),
    ("hfgcs/review_app.py", {"review"}),
    ("hfgcs/review_db.py", {"review"}),
    ("migrations/0004_x.sql", {"review"}),
    ("hfgcs/static_site.py", {"public"}),
    ("public_site/index.html", {"public"}),
    ("docs/deploy.md", set()),
    ("README.md", set()),
    ("tests/test_dsp.py", set()),
    ("decoding/asr.py", {"decode"}),
    ("decoding/worker.py", {"decode"}),
    ("deploy/bin/hfgcs-python", {"review", "public", "decode"}),
    ("scripts/hfgcs_db.py", set()),
    ("hfgcs/deploy.py", set()),
    ("hfgcs/sources/soapy.py", set()),
    ("config.example.toml", set()),
])
def test_roles_for_path(path, roles):
    assert set(d.roles_for_path(path)) == roles


# Source modules only local runs load (an SDR on this machine, file replay). No deployed unit
# uses them, so changing them restarts nothing.
LOCAL_ONLY = {"hfgcs/sources/soapy.py", "hfgcs/sources/file.py"}


def _repo_file(parts: list[str]) -> str | None:
    for rel in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
        if (ROOT / rel).is_file():
            return rel
    return None


def recorder_modules() -> set[str]:
    """Repo files the recorder (hfgcs_scanner.py) imports, including imports inside functions
    and the __init__.py of each package on the way."""
    seen, todo = set(), ["hfgcs_scanner.py"]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        pkg = rel[: -len(".py")].split("/")[:-1]   # the folder, for a module and an __init__.py alike
        names = []
        for node in ast.walk(ast.parse((ROOT / rel).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names += [a.name.split(".") for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = pkg[: len(pkg) - node.level + 1] if node.level else []
                mod = base + (node.module.split(".") if node.module else [])
                names.append(mod)
                names += [mod + [a.name] for a in node.names]   # "from . import x" may name a module
        for name in names:
            for i in range(1, len(name) + 1):   # each package on the way, then the module
                hit = _repo_file(name[:i])
                if hit:
                    todo.append(hit)
    return seen


def test_every_module_the_recorders_import_restarts_them():
    mods = recorder_modules()
    # the walk sees function-level imports (config.shadow_path, make_source)
    assert {"hfgcs/shadow.py", "hfgcs/sources/kiwisdr.py", "hfgcs/sources/openwebrx.py"} <= mods
    assert not any(m.startswith("decoding/") for m in mods)   # the recorders never import the decoder
    assert "hfgcs/shadow_score.py" not in mods                # a report tool: changing it restarts nothing
    unmapped = sorted(m for m in mods - LOCAL_ONLY if not d.roles_for_path(m) & d.RECORDERS)
    assert unmapped == [], f"recorder code a deploy would not restart the recorders for: {unmapped}"


def test_role_of_units():
    assert d.role_of("hfgcs-8992") == "openwebrx"
    assert d.role_of("hfgcs-8992.service") == "openwebrx"
    assert d.role_of("hfgcs-kiwi-11175") == "kiwisdr"
    assert d.role_of("hfgcs-review") == "review"
    assert d.role_of("hfgcs-public") == "public"
    assert d.role_of("hfgcs-datasette") is None
    assert d.role_of("hfgcs-decode") == "decode"
    assert d.role_of("hfgcs-decode-backfill") == "oneshot"


def names(plans):
    return [p.unit for p in plans]


def test_docs_and_tests_restart_nothing():
    plans, notes = d.plan_units(["docs/deploy.md", "tests/test_x.py", "README.md"], UNITS)
    assert plans == [] and notes == []


def test_openwebrx_change_restarts_only_8992():
    plans, _ = d.plan_units(["hfgcs/sources/openwebrx.py", "docs/sources.md"], UNITS)
    assert names(plans) == ["hfgcs-8992"]
    assert plans[0].reasons == ["hfgcs/sources/openwebrx.py"] and plans[0].recorder


def test_kiwi_change_restarts_only_kiwi_units_in_name_order():
    plans, _ = d.plan_units(["hfgcs/sources/kiwisdr.py"], UNITS)
    assert names(plans) == sorted(KIWIS)


def test_shared_dsp_restarts_recorders_web_first_kiwi_then_8992_last():
    plans, _ = d.plan_units(["hfgcs/dsp.py", "hfgcs/review_app.py", "public_site/style.css"], UNITS)
    assert names(plans) == ["hfgcs-review", "hfgcs-public", *sorted(KIWIS), "hfgcs-8992"]


def test_unit_file_change_installs_and_restarts_that_unit():
    plans, notes = d.plan_units(
        ["deploy/systemd/hfgcs-review.service", "deploy/systemd/hfgcs-datasette.service"], UNITS
    )
    assert names(plans) == ["hfgcs-review"]
    assert plans[0].unit_file == "deploy/systemd/hfgcs-review.service"
    assert any("hfgcs-datasette is not installed" in n for n in notes)


def test_force_pending_and_dependency_changes():
    plans, notes = d.plan_units([], UNITS, force=("hfgcs-8992", "nope"))
    assert names(plans) == ["hfgcs-8992"] and plans[0].forced
    assert any("--force nope" in n for n in notes)
    plans, _ = d.plan_units([], UNITS, pending={"hfgcs-kiwi-4724": ["hfgcs/dsp.py"]})
    assert names(plans) == ["hfgcs-kiwi-4724"] and "earlier deploy" in plans[0].reasons[0]
    plans, _ = d.plan_units(["pyproject.toml"], UNITS, deps_changed=True)
    assert len(plans) == len(UNITS)
    plans, _ = d.plan_units(["pyproject.toml"], UNITS, deps_changed=False)
    assert plans == []
    plans, _ = d.plan_units([], UNITS, force=("all",))
    assert len(plans) == len(UNITS) and all(p.forced for p in plans)


def test_dependencies_ignores_tool_sections():
    a = '[project]\nname="x"\ndependencies=["numpy>=1"]\n[tool.ruff]\nline-length=100\n'
    b = a.replace("100", "110")
    c = a.replace("numpy>=1", "numpy>=2")
    assert d.dependencies(a) == d.dependencies(b)
    assert d.dependencies(a) != d.dependencies(c)


# -- quiet windows ---------------------------------------------------------------------------------
def test_parse_absolute_window():
    w = d.parse_window("2026-10-26T23:45Z/2026-10-27T07:00Z", "exercise")
    assert w.start == datetime(2026, 10, 26, 23, 45, tzinfo=timezone.utc)
    assert w.contains(datetime(2026, 10, 27, 3, 0, tzinfo=timezone.utc))
    assert not w.contains(datetime(2026, 10, 27, 7, 0, tzinfo=timezone.utc))  # end is exclusive
    assert not w.contains(datetime(2026, 10, 26, 23, 44, tzinfo=timezone.utc))
    assert "exercise" in w.label and "2026-10-26T23:45Z" in w.label
    # offsets are converted to UTC
    w2 = d.parse_window("2026-10-26T16:45-07:00/2026-10-27T00:00-07:00")
    assert w2.start == w.start


def test_parse_daily_window_wraps_midnight():
    w = d.parse_window("23:45-02:00")
    at = lambda h, m: datetime(2026, 10, 26, h, m, tzinfo=timezone.utc)  # noqa: E731
    assert w.contains(at(23, 50)) and w.contains(at(1, 59)) and not w.contains(at(2, 0))
    assert not w.contains(at(12, 0))
    assert w.end_after(at(23, 50)) == datetime(2026, 10, 27, 2, 0, tzinfo=timezone.utc)
    day = d.parse_window("12:00-13:00 UTC")
    assert day.contains(at(12, 30)) and not day.contains(at(13, 30))


@pytest.mark.parametrize("bad", [
    "2026-10-26T23:45/2026-10-27T07:00",  # no zone
    "2026-10-27T07:00Z/2026-10-26T23:45Z",  # backwards
    "25:00-26:00", "10:00-10:00", "tomorrow",
])
def test_bad_windows_are_rejected(bad):
    with pytest.raises(ValueError):
        d.parse_window(bad)


def test_load_windows_toml(tmp_path):
    f = tmp_path / "quiet-windows.toml"
    f.write_text(
        'windows = ["2026-10-26T23:45Z/2026-10-27T07:00Z",\n'
        '  { span = "2026-11-01T00:00Z/2026-11-01T01:00Z", note = "GT exercise" }]\n'
        'daily = ["06:00-06:30"]\n'
    )
    ws = d.load_windows(f)
    assert len(ws) == 3 and ws[1].note == "GT exercise" and ws[2].daily_start is not None
    assert d.active_window(ws, datetime(2026, 10, 27, 1, 0, tzinfo=timezone.utc)) is ws[0]
    assert d.active_window(ws, datetime(2026, 10, 28, 6, 10, tzinfo=timezone.utc)) is ws[2]
    assert d.active_window(ws, datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)) is None
    assert d.load_windows(tmp_path / "missing.toml") == []
    f.write_text('window = []\n')
    with pytest.raises(ValueError, match="unknown keys"):
        d.load_windows(f)


# -- idle wait -------------------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    return clock, sc, tmp_path


def wait(unit, sc, clock, state_dir, **kw):
    args = dict(min_idle_s=30, timeout_s=600, windows=[], forced=False, state_dir=state_dir, poll_s=2,
                say=lambda m: None)
    args.update(kw)
    return d.wait_idle(unit, sc, clock, **args)


def test_idle_long_enough_goes_ahead(env):
    clock, sc, sd = env
    write_state(sd, "hfgcs-8992", sc.main_pid("hfgcs-8992"))
    assert wait("hfgcs-8992", sc, clock, sd) is None
    assert clock.t == 0


def test_waits_for_capture_to_end_then_min_idle(env):
    clock, sc, sd = env
    pid = sc.main_pid("hfgcs-8992")
    write_state(sd, "hfgcs-8992", pid, state="capturing", since=T0)
    # the transmission ends 40 s in; the recorder writes idle at that moment
    clock.at(40, lambda: write_state(sd, "hfgcs-8992", pid, since=clock.now()))
    assert wait("hfgcs-8992", sc, clock, sd) is None
    assert 70 <= clock.t <= 74  # 40 s capturing + 30 s idle (2 s polls)


def test_capture_that_never_ends_times_out_and_skips(env):
    clock, sc, sd = env
    write_state(sd, "hfgcs-8992", sc.main_pid("hfgcs-8992"), state="capturing", since=T0)
    oc = wait("hfgcs-8992", sc, clock, sd, timeout_s=120)
    assert oc.result == "skipped" and "capturing" in oc.detail and clock.t >= 120
    # --force does not interrupt a capture either
    oc = wait("hfgcs-8992", sc, clock, sd, timeout_s=10, forced=True)
    assert oc.result == "skipped"


def test_unknown_state_skips_unless_forced(env):
    clock, sc, sd = env
    oc = wait("hfgcs-8992", sc, clock, sd)
    assert oc.result == "skipped" and "no state file" in oc.detail and "--force hfgcs-8992" in oc.detail
    write_state(sd, "hfgcs-8992", pid=1)  # stale file from an earlier process
    oc = wait("hfgcs-8992", sc, clock, sd)
    assert oc.result == "skipped" and "from pid 1" in oc.detail
    assert wait("hfgcs-8992", sc, clock, sd, forced=True) is None


def test_inactive_unit_is_left_alone(env):
    clock, sc, sd = env
    sc.units["hfgcs-8992"]["active"] = "inactive"
    oc = wait("hfgcs-8992", sc, clock, sd)
    assert oc.result == "skipped" and "inactive" in oc.detail


def test_quiet_window_defers_unless_forced(env):
    clock, sc, sd = env
    write_state(sd, "hfgcs-8992", sc.main_pid("hfgcs-8992"))
    windows = [d.parse_window("2026-10-26T19:00Z/2026-10-26T21:00Z", "exercise")]
    oc = wait("hfgcs-8992", sc, clock, sd, windows=windows)
    assert oc.result == "deferred" and "exercise" in oc.detail and "2026-10-26T21:00:00.000Z" in oc.detail
    assert wait("hfgcs-8992", sc, clock, sd, windows=windows, forced=True) is None


def test_window_that_starts_while_waiting_defers(env):
    clock, sc, sd = env
    write_state(sd, "hfgcs-8992", sc.main_pid("hfgcs-8992"), state="capturing", since=T0)
    windows = [d.parse_window("2026-10-26T20:01Z/2026-10-26T22:00Z")]
    oc = wait("hfgcs-8992", sc, clock, sd, windows=windows)
    assert oc.result == "deferred" and 60 <= clock.t <= 62


def test_wait_streaming(env):
    clock, sc, sd = env
    at = clock.now()
    sc.restart("hfgcs-kiwi-4724")
    msg = d.wait_streaming("hfgcs-kiwi-4724", sc, clock, restarted_at=at, state_dir=sd, poll_s=1)
    assert msg.startswith("streaming") and 6 <= clock.t <= 7
    sc.audio_after = 1000
    at = clock.now()
    sc.restart("hfgcs-kiwi-6739")
    msg = d.wait_streaming("hfgcs-kiwi-6739", sc, clock, restarted_at=at, state_dir=sd, poll_s=1)
    assert "no audio yet after 60 s" in msg


# -- whole deploys against a throwaway repo -------------------------------------------------------
def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "hfgcs-app"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    for rel in ("hfgcs/dsp.py", "hfgcs/sources/openwebrx.py", "hfgcs/sources/kiwisdr.py",
                "hfgcs/review_app.py", "docs/deploy.md", "deploy/systemd/hfgcs-review.service"):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text("# v1\n")
    (r / "pyproject.toml").write_text('[project]\nname = "hfgcs"\ndependencies = ["numpy"]\n')
    git(r, "add", "-A")
    git(r, "commit", "-qm", "v1")
    git(r, "branch", "target")
    return r


def advance(repo, files, msg="change"):
    git(repo, "checkout", "-q", "target")
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    git(repo, "checkout", "-q", "main")
    return git(repo, "rev-parse", "target")


def deployer(repo, tmp_path, clock, sc, out, **kw):
    opts = d.Options(repo=repo, ref="target", fetch=False, windows_file=None, db=tmp_path / "none.db",
                     state_home=tmp_path / "state-home", state_dir=tmp_path, poll_s=1.0,
                     unit_dir=tmp_path / "units", **kw)
    return d.Deployer(opts, systemctl=sc, clock=clock, out=out)


class Out:
    def __init__(self):
        self.lines = []

    def write(self, s):
        self.lines.append(s)

    def flush(self):
        pass

    @property
    def text(self):
        return "".join(self.lines)


def test_dry_run_prints_plan_and_changes_nothing(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    idle_everywhere(sc, tmp_path)
    before = git(repo, "rev-parse", "HEAD")
    target = advance(repo, {"hfgcs/sources/openwebrx.py": "# v2\n", "docs/deploy.md": "# v2\n"})
    out = Out()
    rc = deployer(repo, tmp_path, clock, sc, out, dry_run=True).run()
    text = out.text
    assert rc == 0
    assert git(repo, "rev-parse", "HEAD") == before  # not fast-forwarded
    assert not [x for x in sc.log if x[0] == "restart"]
    assert "dry run" in text and f"deployed before: {before}" in text and target in text
    assert "1. hfgcs-8992: restart when idle >= 30 s; because hfgcs/sources/openwebrx.py [now: idle" in text
    assert "hfgcs-kiwi-11175" in text.split("not restarted:")[1]
    assert "pyproject.toml unchanged: no reinstall" in text
    assert "dry-run   hfgcs-8992" in text
    assert not (tmp_path / "state-home" / "deployed-sha").exists()


def test_dry_run_shows_deferral_in_a_quiet_window(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    idle_everywhere(sc, tmp_path)
    advance(repo, {"hfgcs/sources/openwebrx.py": "# v2\n"})
    out = Out()
    deployer(repo, tmp_path, clock, sc, out, dry_run=True,
             extra_windows=("2026-10-26T19:45Z/2026-10-27T07:00Z",)).run()
    assert "quiet window now: 2026-10-26T19:45Z/2026-10-27T07:00Z" in out.text
    assert "would be DEFERRED" in out.text


def test_deploy_restarts_kiwi_one_at_a_time_and_8992_last(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path, audio_after=7.0)
    idle_everywhere(sc, tmp_path)
    target = advance(repo, {"hfgcs/dsp.py": "# v2\n", "docs/deploy.md": "# v2\n"})
    out = Out()
    rc = deployer(repo, tmp_path, clock, sc, out).run()
    assert rc == 0, out.text
    assert git(repo, "rev-parse", "HEAD") == target
    restarts = [(u, t) for kind, u, t in sc.log if kind == "restart"]
    assert [u for u, _ in restarts] == [*sorted(KIWIS), "hfgcs-8992"]
    # each restart waited for the previous unit's first audio (7 s), never overlapping
    gaps = [b[1] - a[1] for a, b in zip(restarts, restarts[1:], strict=False)]
    assert all(g >= 7 for g in gaps), gaps
    assert "hfgcs-review" not in [u for u, _ in restarts]
    assert out.text.count("restarted ") >= 6 and "streaming (first audio" in out.text
    home = tmp_path / "state-home"
    hist = [json.loads(x) for x in (home / "deploy-history.jsonl").read_text().splitlines()]
    assert hist[-1]["to"] == target and (home / "deployed-sha").read_text().strip() == target


def test_kiwi_that_never_streams_still_moves_on_after_60_s(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path, audio_after=10_000)
    idle_everywhere(sc, tmp_path)
    advance(repo, {"hfgcs/sources/kiwisdr.py": "# v2\n"})
    out = Out()
    assert deployer(repo, tmp_path, clock, sc, out).run() == 0
    restarts = [t for kind, u, t in sc.log if kind == "restart"]
    assert len(restarts) == 5 and all(b - a >= 60 for a, b in zip(restarts, restarts[1:], strict=False))
    assert "no audio yet after 60 s; moved on" in out.text


def test_busy_and_quiet_units_are_remembered_for_the_next_run(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    idle_everywhere(sc, tmp_path)
    write_state(tmp_path, "hfgcs-kiwi-4724", sc.main_pid("hfgcs-kiwi-4724"), state="capturing", since=T0)
    advance(repo, {"hfgcs/sources/kiwisdr.py": "# v2\n"})
    out = Out()
    rc = deployer(repo, tmp_path, clock, sc, out, idle_timeout_s=20).run()
    assert rc == 3
    assert "skipped   hfgcs-kiwi-4724: still capturing" in out.text
    pending = json.loads((tmp_path / "state-home" / "pending-restarts.json").read_text())
    assert pending == {"hfgcs-kiwi-4724": ["hfgcs/sources/kiwisdr.py"]}
    # next run: nothing new in git, but the remembered restart happens once it is idle
    write_state(tmp_path, "hfgcs-kiwi-4724", sc.main_pid("hfgcs-kiwi-4724"))
    sc.log.clear()
    out = Out()
    assert deployer(repo, tmp_path, clock, sc, out).run() == 0
    assert [u for k, u, _ in sc.log if k == "restart"] == ["hfgcs-kiwi-4724"]
    assert not (tmp_path / "state-home" / "pending-restarts.json").exists()


def test_quiet_window_defers_recorders_but_not_web_units(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    idle_everywhere(sc, tmp_path)
    advance(repo, {"hfgcs/sources/openwebrx.py": "# v2\n", "hfgcs/review_app.py": "# v2\n"})
    out = Out()
    rc = deployer(repo, tmp_path, clock, sc, out, extra_windows=("19:00-21:00",)).run()
    assert rc == 3
    assert [u for k, u, _ in sc.log if k == "restart"] == ["hfgcs-review"]
    assert "deferred  hfgcs-8992: inside quiet window daily 19:00-21:00 UTC" in out.text
    # --force lifts the window
    out = Out()
    rc = deployer(repo, tmp_path, clock, sc, out, extra_windows=("19:00-21:00",), force=("hfgcs-8992",)).run()
    assert rc == 0 and [u for k, u, _ in sc.log if k == "restart"][-1] == "hfgcs-8992"


def test_unit_file_change_is_installed_with_backup_and_reloaded(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    units = tmp_path / "units"
    units.mkdir()
    (units / "hfgcs-review.service").write_text("# v1\n")
    advance(repo, {"deploy/systemd/hfgcs-review.service": "# v2\n"})
    out = Out()
    assert deployer(repo, tmp_path, clock, sc, out).run() == 0
    assert (units / "hfgcs-review.service").read_text() == "# v2\n"
    assert [p.read_text() for p in units.glob("hfgcs-review.service.bak-*")] == ["# v1\n"]
    kinds = [(k, u) for k, u, _ in sc.log]
    assert kinds == [("daemon-reload", None), ("restart", "hfgcs-review")]


def test_pyproject_change_reinstalls_and_dependency_change_restarts_all(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    idle_everywhere(sc, tmp_path)
    advance(repo, {"pyproject.toml": '[project]\nname = "hfgcs"\ndependencies = ["numpy>=2"]\n'})
    ran = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    out = Out()
    opts = d.Options(repo=repo, ref="target", fetch=False, windows_file=None, db=tmp_path / "none.db",
                     state_home=tmp_path / "sh", state_dir=tmp_path, poll_s=1.0, python="/venv/python")
    rc = d.Deployer(opts, systemctl=sc, clock=clock, out=out,
                    run=lambda cmd, **kw: ran.append(cmd) or R()).run()
    assert rc == 0, out.text
    assert ran == [["/venv/python", "-m", "pip", "install", "--quiet", "-e", ".[openwebrx]"]]
    assert len([1 for k, *_ in sc.log if k == "restart"]) == len(UNITS)


def test_migrations_back_up_the_db_first(repo, tmp_path):
    import sqlite3

    (repo / "migrations").mkdir()
    (repo / "migrations" / "0002_x.sql").write_text("CREATE TABLE x (a INTEGER);\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "mig")
    git(repo, "branch", "-f", "target")
    db = tmp_path / "data" / "hfgcs.db"
    db.parent.mkdir()
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, note TEXT);"
                      "INSERT INTO schema_version VALUES (1, 'base');")
    con.close()
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    out = Out()
    opts = d.Options(repo=repo, ref="target", fetch=False, windows_file=None, db=db,
                     state_home=tmp_path / "sh", state_dir=tmp_path)
    assert d.Deployer(opts, systemctl=sc, clock=clock, out=out).run() == 0, out.text
    backups = list((db.parent / "backups").glob("hfgcs-*.db"))
    assert len(backups) == 1
    old = sqlite3.connect(backups[0])
    assert [r[0] for r in old.execute("SELECT version FROM schema_version")] == [1]
    new = sqlite3.connect(db)
    assert [r[0] for r in new.execute("SELECT version FROM schema_version ORDER BY 1")] == [1, 2]
    assert "migrations applied: 0002" in out.text


def test_refuses_dirty_checkout_and_non_fast_forward(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    (repo / "hfgcs/dsp.py").write_text("# local edit\n")
    with pytest.raises(RuntimeError, match="local changes"):
        deployer(repo, tmp_path, clock, sc, Out()).run()
    git(repo, "checkout", "-q", "--", "hfgcs/dsp.py")
    advance(repo, {"docs/deploy.md": "# other\n"})
    (repo / "README.md").write_text("diverged\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "local commit")
    with pytest.raises(RuntimeError, match="not a fast-forward"):
        deployer(repo, tmp_path, clock, sc, Out()).run()


# -- the command line, with a fake systemctl binary -------------------------------------------------
FAKE_SYSTEMCTL = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_SYSTEMCTL_LOG"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")
assert args[0] == "--user", args
cmd = args[1]
if cmd in ("list-unit-files", "list-units"):
    for u in ("hfgcs-8992", "hfgcs-kiwi-11175", "hfgcs-review", "hfgcs-public"):
        print(f"{u}.service enabled enabled")
elif cmd == "is-active":
    print("active")
elif cmd == "show":
    print("4242")
else:
    sys.exit(1)
"""


def test_cli_dry_run_with_fake_systemctl(repo, tmp_path):
    fake = tmp_path / "systemctl"
    fake.write_text(FAKE_SYSTEMCTL)
    fake.chmod(0o755)
    advance(repo, {"hfgcs/sources/kiwisdr.py": "# v2\n"})
    write_state(tmp_path, "hfgcs-kiwi-11175", 4242)
    env = {**os.environ, "HFGCS_SYSTEMCTL": str(fake), "FAKE_SYSTEMCTL_LOG": str(tmp_path / "sc.log"),
           "HFGCS_RUNTIME_DIR": str(tmp_path), "XDG_STATE_HOME": str(tmp_path / "xs"),
           "XDG_CONFIG_HOME": str(tmp_path / "xc"), "HFGCS_DB": str(tmp_path / "none.db")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hfgcs_deploy.py"), "--repo", str(repo),
                        "--ref", "target", "--no-fetch", "--dry-run"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "1. hfgcs-kiwi-11175: restart when idle >= 30 s; because hfgcs/sources/kiwisdr.py" in r.stdout
    assert "[now: idle, idle for" in r.stdout
    assert "not restarted: hfgcs-8992, hfgcs-public, hfgcs-review" in r.stdout
    calls = [json.loads(x) for x in (tmp_path / "sc.log").read_text().splitlines()]
    assert not any(c[1] in ("restart", "daemon-reload") for c in calls)


def test_deploy_sh_wrapper_runs_the_script():
    env = {**os.environ, "HFGCS_PYTHON": sys.executable}
    cmd = [str(ROOT / "deploy" / "deploy.sh"), "--help"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "--dry-run" in r.stdout and "--force UNIT" in r.stdout


# -- decode service (hfgcs-decode, hfgcs-decode-backfill) -------------------------------------------
DECODE_UNITS = [*UNITS, "hfgcs-decode", "hfgcs-decode-backfill"]


def test_decoding_change_restarts_only_the_decode_worker():
    plans, _ = d.plan_units(["decoding/worker.py", "decoding/asr.py", "docs/decoding.md"], DECODE_UNITS)
    assert names(plans) == ["hfgcs-decode"]
    assert not any(p.recorder for p in plans)


def test_new_decode_unit_file_restarts_nothing_while_not_installed():
    changed = ["deploy/systemd/hfgcs-decode.service", "deploy/systemd/hfgcs-decode-backfill.service",
               "deploy/env/decode.env.example", "decoding/worker.py", "decoding/voice_gate.py"]
    plans, notes = d.plan_units(changed, UNITS)   # hilltop today: no decode units installed
    assert plans == []
    assert any("hfgcs-decode is not installed" in n for n in notes)


def test_backfill_unit_file_is_installed_but_never_started():
    plans, _ = d.plan_units(["deploy/systemd/hfgcs-decode-backfill.service"], DECODE_UNITS)
    assert names(plans) == ["hfgcs-decode-backfill"] and plans[0].install_only
    plans, notes = d.plan_units([], DECODE_UNITS, force=("all", "hfgcs-decode-backfill"),
                                pending={"hfgcs-decode-backfill": ["x"]})
    assert "hfgcs-decode-backfill" not in names(plans) and "hfgcs-decode" in names(plans)
    assert any("one-shot" in n for n in notes)


def _py(extras='openwebrx = ["websockets>=12"]', deps='["numpy"]'):
    return f'[project]\nname = "hfgcs"\ndependencies = {deps}\n[project.optional-dependencies]\n{extras}\n'


def test_dependency_roles_by_extra():
    base = _py()
    assert d.dependency_roles(base, base) == frozenset()
    with_decode = _py('openwebrx = ["websockets>=12"]\ndecode = ["faster-whisper"]')
    assert d.dependency_roles(base, with_decode) == {"decode"}
    assert d.dependency_roles(base, _py('openwebrx = ["websockets>=13"]')) == {"openwebrx"}
    assert d.dependency_roles(base, _py('openwebrx = ["websockets>=12"]\ndev = ["pytest"]')) == frozenset()
    assert d.dependency_roles(base, _py(deps='["numpy>=2"]')) == d.ALL_ROLES
    assert d.dependency_roles(base, _py('openwebrx = ["websockets>=12"]\nnewthing = ["x"]')) == d.ALL_ROLES
    plans, _ = d.plan_units(["pyproject.toml"], DECODE_UNITS, dep_roles=frozenset({"decode"}))
    assert names(plans) == ["hfgcs-decode"]


def test_adding_the_decode_extra_restarts_no_recorder(repo, tmp_path):
    """The deploy of this very change on hilltop: new extra + new unit files + decoding/ code."""
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path)
    idle_everywhere(sc, tmp_path)
    advance(repo, {"pyproject.toml": '[project]\nname = "hfgcs"\ndependencies = ["numpy"]\n'
                   '[project.optional-dependencies]\ndecode = ["faster-whisper>=1.1"]\n',
                   "decoding/worker.py": "# v2\n", "deploy/systemd/hfgcs-decode.service": "# v1\n"})
    ran = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    out = Out()
    opts = d.Options(repo=repo, ref="target", fetch=False, windows_file=None, db=tmp_path / "none.db",
                     state_home=tmp_path / "sh", state_dir=tmp_path, poll_s=1.0, python="/venv/python",
                     unit_dir=tmp_path / "units")
    rc = d.Deployer(opts, systemctl=sc, clock=clock, out=out,
                    run=lambda cmd, **kw: ran.append(cmd) or R()).run()
    assert rc == 0, out.text
    assert ran == [["/venv/python", "-m", "pip", "install", "--quiet", "-e", ".[openwebrx]"]]
    assert [k for k, *_ in sc.log] == []           # no restart, no daemon-reload
    assert "hfgcs-decode is not installed" in out.text


def test_decode_unit_installed_gets_decode_extra_and_only_it_restarts(repo, tmp_path):
    clock = FakeClock()
    sc = FakeSystemctl(clock, tmp_path, units=DECODE_UNITS)
    idle_everywhere(sc, tmp_path)
    advance(repo, {"pyproject.toml": '[project]\nname = "hfgcs"\ndependencies = ["numpy"]\n'
                   '[project.optional-dependencies]\ndecode = ["faster-whisper>=1.2"]\n',
                   "decoding/pipeline.py": "# v2\n"})
    ran = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    out = Out()
    opts = d.Options(repo=repo, ref="target", fetch=False, windows_file=None, db=tmp_path / "none.db",
                     state_home=tmp_path / "sh", state_dir=tmp_path, poll_s=1.0, python="/venv/python",
                     extra_windows=("2026-10-26T19:00Z/2026-10-27T07:00Z",))   # inside a quiet window
    rc = d.Deployer(opts, systemctl=sc, clock=clock, out=out,
                    run=lambda cmd, **kw: ran.append(cmd) or R()).run()
    assert rc == 0, out.text
    assert ran == [["/venv/python", "-m", "pip", "install", "--quiet", "-e", ".[openwebrx,decode]"]]
    assert [(k, u) for k, u, _ in sc.log] == [("restart", "hfgcs-decode")]
