"""Shadow squelch: write its decisions and compare them with the live catalog.

The recorder can run a second squelch on the same stream without recording
(``squelch_shadow`` in the config, docs/squelch.md). Each capture it would have
made becomes one JSON line in the shadow file. This module appends those lines
and reports how they line up with what the live squelch recorded::

    python -m hfgcs.shadow --catalog hfgcs_catalog.jsonl [--shadow FILE] [--since 2026-10-30] [--list]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def default_shadow_path(catalog: Path) -> Path:
    """The shadow file beside a catalog: ``<catalog stem>.squelch-shadow.jsonl``."""
    catalog = Path(catalog)
    return catalog.with_name(f"{catalog.stem}.squelch-shadow.jsonl")


def append_events(events: list[dict], path: Path) -> None:
    """Append one JSON line per shadow capture (creates the file and its folder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _t(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _load(path: Path, since: str | None) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            start, end = e.get("file_start_utc") or e.get("start_utc"), e.get("end_utc")
            if not start or not end or (since and start[:10] < since):
                continue
            rows.append({"rx": e.get("receiver_id") or "?", "khz": float(e.get("frequency_khz") or 0),
                         "s": _t(start), "e": _t(end), "start": start, "end": end,
                         "reason": e.get("close_reason") or ("cap" if e.get("truncated") else "hang"),
                         "snr": e.get("snr_db_peak")})
    return rows


def _overlaps(a: dict, others: list[dict]) -> bool:
    return any(o["s"] < a["e"] and o["e"] > a["s"] for o in others)


def compare(live: list[dict], shadow: list[dict]) -> dict:
    """Per (receiver, kHz): counts and seconds recorded by each, and the captures one
    of them made with no overlapping capture from the other."""
    out = {}
    keys = sorted({(r["rx"], r["khz"]) for r in live + shadow})
    for key in keys:
        lv = [r for r in live if (r["rx"], r["khz"]) == key]
        sh = [r for r in shadow if (r["rx"], r["khz"]) == key]
        reasons: dict[str, int] = defaultdict(int)
        for r in sh:
            reasons[r["reason"]] += 1
        out[key] = {
            "live": len(lv), "live_s": sum(r["e"] - r["s"] for r in lv),
            "shadow": len(sh), "shadow_s": sum(r["e"] - r["s"] for r in sh),
            "live_only": [r for r in lv if not _overlaps(r, sh)],
            "shadow_only": [r for r in sh if not _overlaps(r, lv)],
            "shadow_reasons": dict(reasons),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hfgcs.shadow", description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", default="hfgcs_catalog.jsonl", help="the live recorder catalog")
    ap.add_argument("--shadow", help="shadow file (default: <catalog stem>.squelch-shadow.jsonl beside it)")
    ap.add_argument("--since", help="first UTC day to compare, YYYY-MM-DD")
    ap.add_argument("--list", action="store_true", help="list each capture only one squelch made")
    args = ap.parse_args(argv)
    catalog = Path(args.catalog)
    shadow_path = Path(args.shadow) if args.shadow else default_shadow_path(catalog)
    for p in (catalog, shadow_path):
        if not p.exists():
            print(f"no such file: {p}", file=sys.stderr)
            return 2
    shadow = _load(shadow_path, args.since)
    if shadow:   # compare over the span the shadow file covers
        first = min(r["s"] for r in shadow)
        live = [r for r in _load(catalog, args.since) if r["e"] >= first]
    else:
        live = _load(catalog, args.since)
    algos = set()
    with open(shadow_path) as fh:
        for line in fh:
            try:
                e = json.loads(line)
                algos.add(f"{e.get('algorithm')} shadow vs {e.get('live_algorithm')} live")
            except json.JSONDecodeError:
                pass
    print(f"{', '.join(sorted(algos)) or 'shadow'}: {len(live)} live and {len(shadow)} shadow captures"
          + (f" since {args.since}" if args.since else ""))
    print(f"{'receiver':28s} {'kHz':>8s} {'live':>5s} {'live min':>9s} {'shadow':>6s} {'shad min':>9s} "
          f"{'live only':>9s} {'shad only':>9s}  shadow closes")
    for (rx, khz), c in compare(live, shadow).items():
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(c["shadow_reasons"].items())) or "-"
        print(f"{rx[:28]:28s} {khz:8g} {c['live']:5d} {c['live_s'] / 60:9.1f} {c['shadow']:6d} "
              f"{c['shadow_s'] / 60:9.1f} {len(c['live_only']):9d} {len(c['shadow_only']):9d}  {reasons}")
        if args.list:
            for kind in ("live_only", "shadow_only"):
                for r in c[kind]:
                    print(f"    {kind.replace('_', ' ')}: {r['start']} .. {r['end']} "
                          f"({r['e'] - r['s']:.0f} s, {r['reason']}, SNR peak {r['snr']})")
    print("live only: recorded now, the shadow squelch would have skipped it (noise, or a miss?)."
          " shadow only: the shadow squelch would have recorded it and the live one did not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
