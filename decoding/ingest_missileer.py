"""Seed eval-set candidates from the missileer.watch public archive feed.

Feed: https://data.missileer.watch/public/archive.json (schema version 1,
self-describing ``_schema`` block). License: the feed's ``license`` field is
``CC0-1.0`` and the site footer says "Our recordings and decodes are released
under CC0 1.0 (public domain dedication); attribution appreciated:
missileer.watch". We keep attribution on every row anyway.

Label quality, which matters for evaluation:
  * ``corrected: true`` rows carry a listener correction (``text``) and the
    machine copy it replaced (``machine_text``): ``gold`` (human-corrected;
    team decision 2026-10-26; not yet re-checked by our own ear).
  * every other row's ``text`` is missileer's own machine copy: ``silver``.
    Good for detection / structure checks and as a competitor baseline,
    never treated as ground truth for headline CER.

Usage (small samples only; this never bulk-downloads audio):

    python -m decoding.ingest_missileer --sample 5 --out evalset/candidates/missileer.jsonl
    python -m decoding.ingest_missileer --sample 3 --download-audio evalset/audio --max-audio 3

Full eval-set build (labels for every row, audio for gold, optionally all,
under a byte budget; writes labels.jsonl + manifest.jsonl + manifest.csv):

    python -m decoding.ingest_missileer --build /workspace/hfgcs-data/eval --audio all --max-mb 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

FEED_URL = "https://data.missileer.watch/public/archive.json"
SOURCE = "missileer.watch"
USER_AGENT = "hfgcs-eval-ingest/0.1 (+https://github.com/Rylvir/hfgcs)"
EXPECTED_LICENSE = "CC0-1.0"
MAX_AUDIO_DOWNLOADS = 10  # hard cap for --sample mode: sampling, not mirroring
DEFAULT_BUDGET_MB = 300  # --build mode byte budget for audio


def fetch_feed(url: str = FEED_URL, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        return json.load(resp)


def _clean(text: Optional[str]) -> Optional[str]:
    """missileer marks unheard characters with '?'; our labels use '.'
    (NEET INTEL convention) so CER code can skip them."""
    return text.replace("?", ".") if text else text


def to_label(row: dict, feed: dict) -> dict:
    """Map one archive row to our eval-label record (docs/decoding.md section 3)."""
    corrected = bool(row.get("corrected"))
    text = row.get("text") or ""
    anchors = row.get("anchors") or []
    kinds = [a.get("kind") for a in anchors]
    return {
        "id": f"mw:{row['id']}",
        "source": SOURCE,
        "source_url": FEED_URL,
        "source_id": row["id"],
        "license": feed.get("license"),
        "attribution": feed.get("attribution"),
        "feed_generated_at": feed.get("generated_at"),
        "audio_url": row.get("clip_url"),
        "audio_format": "opus" if (row.get("clip_url") or "").endswith(".opus") else None,
        "audio_path": None,
        "audio_sha256": None,
        "heard_at_utc": row.get("heard_at"),
        "heard_at_basis": row.get("heard_at_basis"),
        "clip_start_at_utc": row.get("clip_start_at"),
        "voice_start_s": row.get("voice_start_s"),
        "frequency_khz": row.get("frequency_khz"),
        "receiver": row.get("station"),
        "reception_count": row.get("reception_count"),
        "coreceptions": sum(1 for r in row.get("receptions") or [] if r.get("coreception")),
        # missileer's 'grade' -> our 'type'
        "type": {
            "message": "eam",
            "unconfirmed": "eam",
            "fragment": "eam",
            "radio-check": "radio_check",
            "auth": "auth",
            "traffic": "traffic",
            "8888": "other",
        }.get(row.get("grade"), "unknown"),
        "label_status": "gold" if corrected else "silver",
        "labels": {
            "body": _clean(text) if row.get("grade") in ("message", "unconfirmed", "fragment") else None,
            "preamble": _clean(row.get("preamble")),
            "declared_length": row.get("announced_length"),
            "callsign": None,  # not in the feed: label by ear
            "addressee": None,  # not in the feed: label by ear
            "read1": None,
            "read2": None,
        },
        "reference_machine": {
            "text": _clean(row.get("machine_text") if corrected else text),
            "char_times": row.get("char_times"),
            "position_notes": row.get("position_notes") or [],
        },
        "anchors": anchors,
        "has_both_reads": "i_say_again" in kinds,
        "flags": [
            f
            for f, on in (
                ("partial", row.get("partial")),
                ("malformed", row.get("malformed")),
                ("unconfirmed", row.get("unconfirmed")),
                ("ear_queue", row.get("ear_queue")),
                ("has_unheard_chars", "?" in text),
            )
            if on
        ],
        "correction_source": row.get("correction_source"),
        "split": split_for(row["id"]),
    }


def split_for(key: str, test_fraction: float = 0.3) -> str:
    """Deterministic dev/test split (stable across re-ingests)."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "dev"


def select(rows: Iterable[dict], n: int) -> list[dict]:
    """Best candidates first: corrected rows, then complete message rows with
    audio and both reads anchored. Newest first inside each tier."""
    rows = [r for r in rows if r.get("clip_url")]

    def tier(r):
        kinds = [a.get("kind") for a in r.get("anchors") or []]
        if r.get("corrected"):
            return 0
        if r.get("grade") == "message" and "?" not in (r.get("text") or "") and "i_say_again" in kinds:
            return 1
        if r.get("grade") == "message":
            return 2
        return 3

    rows.sort(key=lambda r: r.get("heard_at") or "", reverse=True)
    rows.sort(key=tier)
    return rows[:n]


def download_audio(label: dict, out_dir: Path) -> dict:
    url = label["audio_url"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (label["source_id"] + Path(url).suffix)
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            dest.write_bytes(resp.read())
    label["audio_path"] = str(dest)
    label["audio_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    return label


def _download(url: str, dest: Path) -> int:
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.read())
            tmp.rename(dest)
    return dest.stat().st_size


def manifest_row(label: dict, retrieved_at: str) -> dict:
    return {
        "clip_id": label["id"],
        "source_id": label["source_id"],
        "source_url": label["audio_url"],
        "feed_url": label["source_url"],
        "license": label["license"],
        "attribution": label["attribution"],
        "retrieved_at": retrieved_at,
        "status": label["label_status"],
        "type": label["type"],
        "split": label["split"],
        "sha256": label["audio_sha256"],
        "bytes": label.get("audio_bytes"),
        "path": label["audio_path"],
    }


def build_eval_set(
    feed: dict, root: Path, audio: str = "gold", max_mb: float = DEFAULT_BUDGET_MB, delay_s: float = 0.2
) -> dict:
    """Write ``root``/labels.jsonl, manifest.jsonl, manifest.csv and audio under
    ``root``/audio/{gold,silver}. ``audio``: "none" | "gold" | "all". Gold
    audio is always fetched first; silver stops at the byte budget."""
    import csv
    import time
    from datetime import datetime, timezone

    if feed.get("license") != EXPECTED_LICENSE:
        raise ValueError(f"feed license is {feed.get('license')!r}, expected {EXPECTED_LICENSE}")
    root = Path(root)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    (root / "feed").mkdir(parents=True, exist_ok=True)
    snap = root / "feed" / f"archive_{feed.get('generated_at', 'unknown').replace(':', '')}.json"
    snap.write_text(json.dumps(feed))
    labels = [to_label(r, feed) for r in feed.get("messages", [])]
    labels.sort(key=lambda lab: (lab["label_status"] != "gold", lab["heard_at_utc"] or ""))
    budget = max_mb * 1e6
    used = 0
    skipped = 0
    for lab in labels:
        want = audio == "all" or (audio == "gold" and lab["label_status"] == "gold")
        if not want or not lab["audio_url"]:
            continue
        dest = root / "audio" / lab["label_status"] / (lab["source_id"] + Path(lab["audio_url"]).suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if lab["label_status"] != "gold" and used >= budget:
            skipped += 1
            continue
        fresh = not dest.exists()
        size = _download(lab["audio_url"], dest)
        used += size
        lab["audio_path"] = str(dest)
        lab["audio_bytes"] = size
        lab["audio_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if fresh and delay_s:
            time.sleep(delay_s)
    with open(root / "labels.jsonl", "w") as fh:
        for lab in labels:
            fh.write(json.dumps(lab) + "\n")
    rows = [manifest_row(lab, retrieved_at) for lab in labels]
    with open(root / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["clip_id"])
        w.writeheader()
        w.writerows(rows)
    return {
        "rows": len(labels),
        "gold": sum(lab["label_status"] == "gold" for lab in labels),
        "silver": sum(lab["label_status"] == "silver" for lab in labels),
        "audio_files": sum(1 for lab in labels if lab["audio_path"]),
        "audio_bytes": used,
        "skipped_over_budget": skipped,
        "feed_snapshot": str(snap),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feed", default=FEED_URL, help="feed URL or local JSON path")
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="JSONL output (default stdout)")
    ap.add_argument("--download-audio", type=Path, default=None, metavar="DIR")
    ap.add_argument("--max-audio", type=int, default=3)
    ap.add_argument("--build", type=Path, default=None, metavar="ROOT", help="build the full eval set")
    ap.add_argument("--audio", choices=["none", "gold", "all"], default="gold")
    ap.add_argument("--max-mb", type=float, default=DEFAULT_BUDGET_MB)
    args = ap.parse_args(argv)

    feed = json.loads(Path(args.feed).read_text()) if Path(args.feed).exists() else fetch_feed(args.feed)
    if feed.get("license") != EXPECTED_LICENSE:
        print(
            f"refusing: feed license is {feed.get('license')!r}, expected {EXPECTED_LICENSE}", file=sys.stderr
        )
        return 2
    if args.build:
        print(json.dumps(build_eval_set(feed, args.build, args.audio, args.max_mb), indent=1))
        return 0
    labels = [to_label(r, feed) for r in select(feed.get("messages", []), args.sample)]
    if args.download_audio:
        for lab in labels[: min(args.max_audio, MAX_AUDIO_DOWNLOADS)]:
            download_audio(lab, args.download_audio)
    lines = "".join(json.dumps(lab) + "\n" for lab in labels)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(lines)
    else:
        sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
