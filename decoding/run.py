"""Decode recordings and write the per-recording documents.

    python -m decoding.run recordings/hfgcs_catalog.jsonl      # Phase 0 catalog (schema 2)
    python -m decoding.run clip.wav [clip2.opus ...]            # bare audio files

For each recording this writes ``<audio>.decode.json`` next to the audio and
appends the identical document as one line to ``decodes.jsonl`` (default:
beside the catalog, or beside the first audio file; ``--jsonl`` overrides).
A recording whose sidecar already has the same sha256 and decoder_version is
skipped (``--force`` re-decodes), so re-running over a growing catalog is
idempotent. See docs/decoding.md "Decode output" and docs/decode.schema.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from .output import build_document
from .pipeline import DecodeConfig, Decoder

CATALOG_FIELDS_NOTE = "copied verbatim from the Phase 0 catalog line (schema 2)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_catalog_file(entry: dict, catalog: Path) -> Path:
    """The audio file of a catalog line: absolute as written, else relative to
    the catalog's directory, else just its name in that directory, else
    relative to the working directory."""
    f = Path(entry["file"])
    if f.is_absolute():
        return f
    for cand in (catalog.parent / f, catalog.parent / f.name, f):
        if cand.is_file():
            return cand.resolve()
    return (catalog.parent / f).resolve()


def iter_inputs(paths: Sequence[Path]) -> Iterator[tuple[Path, Optional[dict], Path]]:
    """Yield (audio path, catalog entry or None, default decodes.jsonl dir)."""
    for p in paths:
        p = Path(p)
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        e = json.loads(line)
                        if e.get("file"):
                            yield resolve_catalog_file(e, p), e, p.parent
        else:
            yield p.resolve(), None, p.resolve().parent


def recording_section(audio: Path, entry: Optional[dict]) -> dict:
    """``recording``: path + sha256 of the file actually decoded, then every
    catalog field. A catalog sha256 that disagrees with the file is kept as
    ``catalog_sha256`` (and reported), never silently trusted."""
    sha = sha256_file(audio)
    rec = {"path": str(audio), "sha256": sha}
    if entry:
        for k, v in entry.items():
            if k == "sha256":
                if v and v != sha:
                    rec["catalog_sha256"] = v
                    print(f"warning: {audio}: catalog sha256 {v[:12]} != file {sha[:12]}", file=sys.stderr)
                continue
            rec[k] = v
    else:
        rec["file"] = str(audio)
    return rec


def sidecar_path(audio: Path) -> Path:
    return audio.with_name(audio.name + ".decode.json")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_document(doc: dict, audio: Path, jsonl: Optional[Path]) -> Path:
    """Sidecar (pretty, atomic) + one compact line in ``jsonl``. Same content."""
    side = sidecar_path(audio)
    _write_atomic(side, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    if jsonl is not None:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    return side


def cached_transcriber(decoder: Decoder, cache: Path) -> Callable:
    """Wrap the decoder's ASR with an on-disk cache laid out like
    ``decoding.evaluate`` (``<cache>/<AsrConfig.key()>/<audio stem>.json``)."""
    inner = decoder._asr

    def transcribe(cfg, audio, day_callsigns):
        f = Path(cache) / cfg.key() / (Path(audio).stem + ".json")
        sha = sha256_file(Path(audio))
        if f.exists():
            r = json.loads(f.read_text())
            # Entries written here carry the audio hash; evaluate.py's do not.
            if r.get("audio_sha256") in (None, sha):
                return r
        t0 = time.time()
        r = inner(cfg, audio, day_callsigns)
        r.setdefault("asr_seconds", round(time.time() - t0, 1))
        r["audio_sha256"] = sha
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r))
        return r

    return transcribe


def decode_recording(
    decoder: Decoder,
    audio: Path,
    entry: Optional[dict] = None,
    day_callsigns: Sequence[str] = (),
    jsonl: Optional[Path] = None,
    force: bool = False,
    decoded_at: Optional[str] = None,
) -> Optional[dict]:
    """Decode one recording and write its sidecar (+ decodes.jsonl line).
    Returns the document, or None when an up-to-date sidecar already exists."""
    audio = Path(audio)
    recording = recording_section(audio, entry)
    side = sidecar_path(audio)
    if not force and side.exists():
        try:
            old = json.loads(side.read_text())
        except (OSError, ValueError):
            old = {}
        from .pipeline import decoder_version

        if (old.get("recording") or {}).get("sha256") == recording["sha256"] and old.get(
            "decoder_version"
        ) == decoder_version(decoder.cfg):
            return None
    result = decoder.decode(audio, day_callsigns)
    doc = build_document(result, recording, decoded_at=decoded_at, snap=decoder.cfg.snap)
    write_document(doc, audio, jsonl)
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="catalog .jsonl files and/or audio files")
    ap.add_argument("--jsonl", type=Path, help="decodes.jsonl to append to (default: beside the input)")
    ap.add_argument("--no-jsonl", action="store_true", help="write sidecars only")
    ap.add_argument("--force", action="store_true", help="re-decode even if the sidecar is up to date")
    ap.add_argument("--asr-cache", type=Path, help="cache ASR output here (evaluate.py layout)")
    ap.add_argument("--day-callsign", action="append", default=[], help="today's callsigns (ASR prompt)")
    ap.add_argument("--model", help="override AsrConfig.model (e.g. distil-small.en)")
    ap.add_argument("--cpu-threads", type=int, help="override AsrConfig.cpu_threads")
    ap.add_argument("--no-second-pass", action="store_true", help="disable the vote3 second pass")
    args = ap.parse_args(argv)

    cfg = DecodeConfig()
    if args.model:
        cfg.asr.model = args.model
    if args.cpu_threads:
        cfg.asr.cpu_threads = args.cpu_threads
    if args.no_second_pass:
        cfg.second_pass = False
    decoder = Decoder(cfg)
    if args.asr_cache:
        decoder._transcribe = cached_transcriber(decoder, args.asr_cache)
    n = skipped = failed = 0
    for audio, entry, default_dir in iter_inputs(args.inputs):
        jsonl = None if args.no_jsonl else (args.jsonl or default_dir / "decodes.jsonl")
        if not audio.is_file():
            print(f"missing audio: {audio}", file=sys.stderr)
            failed += 1
            continue
        try:
            doc = decode_recording(decoder, audio, entry, args.day_callsign, jsonl, args.force)
        except Exception as e:  # keep going over a catalog; report at the end
            print(f"failed: {audio}: {e}", file=sys.stderr)
            failed += 1
            continue
        if doc is None:
            skipped += 1
            continue
        n += 1
        kinds = ", ".join(f"{b['type']}@{b['start_s']}" for b in doc["broadcasts"])
        print(f"{audio.name}: {len(doc['broadcasts'])} broadcast(s) [{kinds}]")
    print(f"decoded {n}, up to date {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
