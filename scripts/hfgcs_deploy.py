#!/usr/bin/env python3
"""Deploy the HFGCS checkout on hilltop and restart only what changed.

Run it as the service user (never sudo), normally through ``deploy/deploy.sh``:

    deploy/deploy.sh --dry-run          # show the plan; changes nothing
    deploy/deploy.sh                    # fast-forward to origin/main and restart what changed
    deploy/deploy.sh --force hfgcs-8992 # also restart this unit (lifts quiet windows, not the idle wait)

Steps: record the deployed SHA, ``git fetch`` and fast-forward to ``--ref``
(default origin/main), reinstall if pyproject.toml changed, back up the DB and
apply pending migrations, then restart only the affected units: web units
first, Kiwi recorders one at a time (each must stream again, or 60 s pass,
before the next), hfgcs-8992 last. A recorder is restarted only after it has
been idle (no capture open) for --min-idle seconds and never inside a quiet
window. See docs/deploy.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hfgcs.deploy import Deployer, Options  # noqa: E402


def default_windows_file() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "hfgcs" / "quiet-windows.toml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy/deploy.sh",
        description="Fast-forward the HFGCS checkout and restart only the units whose code changed, "
        "waiting for idle recorders and respecting quiet windows. Never run with sudo.",
    )
    p.add_argument("--ref", default="origin/main", help="target ref (default: %(default)s)")
    p.add_argument("--remote", default="origin", help="remote to fetch (default: %(default)s)")
    p.add_argument("--no-fetch", action="store_true", help="do not git fetch first")
    p.add_argument("--dry-run", "-n", action="store_true", help="print the plan; change nothing")
    p.add_argument(
        "--force", action="append", default=[], metavar="UNIT",
        help="restart UNIT even if nothing it uses changed ('all' = every hfgcs unit); repeatable. "
        "For recorders it lifts quiet windows and accepts a missing state file, but still waits "
        "for an open capture to finish",
    )
    p.add_argument("--min-idle", type=float, default=30.0, metavar="S",
                   help="a recorder must have been idle this long before a restart (default: %(default)s)")
    p.add_argument("--idle-timeout", type=float, default=600.0, metavar="S",
                   help="give up waiting for an idle recorder after this long; it is skipped and "
                   "remembered for the next run (default: %(default)s)")
    p.add_argument("--stream-timeout", type=float, default=60.0, metavar="S",
                   help="after restarting a recorder, wait this long for audio before the next unit "
                   "(default: %(default)s)")
    p.add_argument("--quiet-windows", type=Path, default=None, metavar="FILE",
                   help="TOML with UTC quiet windows (default: ~/.config/hfgcs/quiet-windows.toml; "
                   "see deploy/quiet-windows.example.toml)")
    p.add_argument("--quiet-window", action="append", default=[], metavar="SPEC",
                   help="extra quiet window, e.g. 2026-10-26T23:45Z/2026-10-27T07:00Z or 23:45-07:00 "
                   "(daily, UTC); repeatable")
    p.add_argument("--db", type=Path, default=None,
                   help="SQLite DB to migrate (default: $HFGCS_DB, review.env, then data/hfgcs.db)")
    p.add_argument("--no-migrate", action="store_true", help="do not apply pending DB migrations")
    p.add_argument("--pip-spec", default=".[openwebrx]",
                   help="what pip install -e reinstalls when pyproject.toml changed (default: %(default)s, "
                   "plus the decode extra when an hfgcs-decode unit is installed)")
    p.add_argument("--no-install-units", action="store_true",
                   help="do not copy changed deploy/systemd/*.service files over installed ones")
    p.add_argument("--unit", action="append", default=[], metavar="UNIT",
                   help="consider only these units (default: every installed hfgcs-*.service)")
    p.add_argument("--repo", type=Path, default=ROOT, help="checkout to deploy (default: %(default)s)")
    p.add_argument(
        "--since", "--from-sha", dest="from_sha", metavar="SHA",
        help="do not fetch or move the checkout; treat SHA as the previously deployed commit and act on "
        "SHA..HEAD (used internally after the fast-forward; also for the first deploy after a manual pull)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print("deploy: refusing to run as root; run it as the service user (systemctl --user units)",
              file=sys.stderr)
        return 2
    opts = Options(
        repo=args.repo.resolve(),
        ref=args.ref,
        remote=args.remote,
        fetch=not args.no_fetch,
        dry_run=args.dry_run,
        force=tuple(args.force),
        from_sha=args.from_sha,
        min_idle_s=args.min_idle,
        idle_timeout_s=args.idle_timeout,
        stream_timeout_s=args.stream_timeout,
        windows_file=args.quiet_windows or default_windows_file(),
        extra_windows=tuple(args.quiet_window),
        db=args.db,
        migrate=not args.no_migrate,
        pip_spec=args.pip_spec,
        install_units=not args.no_install_units,
        units=tuple(args.unit),
    )
    dep = Deployer(opts)
    try:
        if not opts.dry_run and not opts.from_sha:
            # Update the checkout, then run the *new* version of this script for the rest,
            # so a deploy always acts with the code it deploys.
            old, target = dep.update()
            script = opts.repo / "scripts" / "hfgcs_deploy.py"
            os.execv(sys.executable, [sys.executable, str(script), *argv, "--from-sha", old])
        return dep.run()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"deploy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
