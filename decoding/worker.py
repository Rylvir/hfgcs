"""Decode service: watch the recorder's catalog, decode new captures, load them into the DB.

    python -m decoding.worker                 # watch loop (the hfgcs-decode systemd unit)
    python -m decoding.worker --once          # one pass over new captures, then exit
    python -m decoding.worker --backfill      # every capture in the catalog (also older ones), then exit
    python -m decoding.worker --check         # dependencies, ffmpeg and the ASR model; exit 0 if ready
    python -m decoding.worker --prefetch      # download the ASR model (small.en, ~460 MB) and exit
    python -m decoding.worker --status        # what the state file says, per capture

One worker, one clip at a time, oldest first. For each capture in ``hfgcs_catalog.jsonl``:

1. **Voice gate** (``decoding.voice_gate``): silero-vad on the filtered audio. A clip with
   less than ``--min-speech`` seconds of speech is recorded as ``no_voice`` (skipped, not an
   error) and not decoded.
2. **Decode** with the default pipeline (``decoding.pipeline.Decoder``: ASR ``small.en``,
   parser, and the vote3 second pass when the reads disagree) and write the contract-v1
   document (``<wav>.decode.json`` sidecar + a ``decodes.jsonl`` line; docs/decode.schema.json).
   Gate and decode run in a child process with a **wall-time cap** (``--timeout-factor`` x the
   clip length, clamped to ``--timeout-min``..``--timeout-max``); a child over the cap is
   killed (its whole process group) and the clip is recorded as ``timeout``.
3. **Load**: ``scripts/hfgcs_db.py --db $HFGCS_DB load-decodes <sidecar>``, so the result shows
   on the review page. load-decodes never overwrites rows a person edited or checked.

Idempotent: a state file (``~/.local/state/hfgcs/decode-state.json``) keyed by the recording's
sha256 remembers each verdict with the decoder version; re-runs skip finished clips, and the
sidecar check in ``decoding.run`` and load-decodes' own dedup make a repeated decode or load a
no-op. A clip that failed or timed out is retried ``--max-attempts`` times in all (``--retry-failed``
resets that). A new decoder version (``decoding.__version__``, ASR config or parser schema)
makes every clip due again.

The watch loop handles captures that end after the worker first started; older ones are
left to ``--backfill`` (run it once, e.g. through hfgcs-decode-backfill.service), so a first
start never begins with hours of backlog. A lock file keeps a backfill and the watch loop
from ever decoding two clips at once.

The worker itself is stdlib only; the child needs the ``decode`` extra
(``pip install -e '.[decode]'``: faster-whisper, and imageio-ffmpeg as a fallback when the
``ffmpeg`` binary is missing) and the model in the Hugging Face cache (``--prefetch``).
See docs/decoding.md "Decode service" and docs/deploy.md.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("hfgcs.decode")

REPO = Path(__file__).resolve().parent.parent
EXIT_CONFIG = 78  # EX_CONFIG: missing dependency or model; the unit does not restart on it
STATE_SCHEMA = 1
DONE_STATUSES = ("decoded", "no_voice")
RETRY_STATUSES = ("failed", "timeout", "missing")

INSTALL_HELP = """\
The decode service needs the 'decode' extra and the ASR model. On hilltop, as scribe:
    cd ~/hfgcs-app
    .venv/bin/python -m pip install -e '.[openwebrx,decode]'
    .venv/bin/python -m decoding.worker --prefetch      # downloads {model} (~460 MB for small.en)
    .venv/bin/python -m decoding.worker --check
(docs/decoding.md "Decode service")."""


# -- configuration -----------------------------------------------------------------------------

def _env(name: str, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return type(default)(v) if default is not None else v


def default_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "hfgcs"


@dataclass
class WorkerConfig:
    catalog: Path = REPO / "hfgcs_catalog.jsonl"
    recordings_root: Optional[Path] = None  # base of the catalog's "file" paths (default: catalog dir)
    db: Optional[Path] = None  # None: decode only, nothing loaded
    jsonl: Optional[Path] = None  # decodes.jsonl (default: beside the catalog)
    state_dir: Path = field(default_factory=default_state_dir)
    model: str = "small.en"
    cpu_threads: int = 3
    second_pass: bool = True
    day_callsigns: tuple = ()
    gate: bool = True
    gate_threshold: float = 0.3
    min_speech_s: float = 1.0
    timeout_factor: float = 10.0
    timeout_min_s: float = 120.0
    timeout_max_s: float = 7200.0  # >= 3 passes x max_rtf 2 x 900 s clips + load (decoding 0.3)
    max_attempts: int = 2
    retry_failed: bool = False
    poll_s: float = 30.0
    ingest: bool = True  # run ingest-catalog when the catalog grows (captures show without decodes too)
    python: str = sys.executable
    db_tool: Path = REPO / "scripts" / "hfgcs_db.py"
    load_timeout_s: float = 600.0
    # Test hooks ("module:attr"): fake ASR / voice gate inside the child.
    transcribe_impl: Optional[str] = None
    gate_impl: Optional[str] = None

    @property
    def state_file(self) -> Path:
        return self.state_dir / "decode-state.json"

    @property
    def history_file(self) -> Path:
        return self.state_dir / "decode-history.jsonl"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "decode.lock"

    def root(self) -> Path:
        return Path(self.recordings_root) if self.recordings_root else Path(self.catalog).parent

    def decode_config(self):
        from .pipeline import DecodeConfig

        cfg = DecodeConfig()
        cfg.asr = replace(cfg.asr, model=self.model, cpu_threads=self.cpu_threads)
        cfg.second_pass = self.second_pass
        return cfg

    def decoder_version(self) -> str:
        from .pipeline import decoder_version

        return decoder_version(self.decode_config())

    def gate_config(self):
        from .voice_gate import GateConfig

        return GateConfig(enabled=self.gate, threshold=self.gate_threshold, min_speech_s=self.min_speech_s)

    def gate_version(self) -> str:
        return self.gate_config().version()

    def timeout_for(self, duration_s: Optional[float]) -> float:
        """Wall-time cap for one clip: factor x length, clamped to [min, max]."""
        if not duration_s or duration_s <= 0:
            return self.timeout_max_s
        return max(self.timeout_min_s, min(self.timeout_max_s, self.timeout_factor * float(duration_s)))


def summarize(counts) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing to do"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# -- capture discovery ---------------------------------------------------------------------------

@dataclass
class Capture:
    key: str  # catalog sha256, else "file:<path>"
    audio: Path
    entry: dict
    line: int
    started: Optional[datetime]
    ended: Optional[datetime]
    duration_s: Optional[float]

    @property
    def name(self) -> str:
        return self.audio.name


def read_catalog(catalog: Path, root: Optional[Path] = None) -> list[Capture]:
    """Every recording in the catalog, oldest first (file start, then catalog line). A line
    that is not JSON, or has no ``file``, is logged and skipped. Later duplicates of a key
    are ignored."""
    from .run import resolve_catalog_file

    catalog = Path(catalog)
    if not catalog.exists():
        return []
    out: dict[str, Capture] = {}
    with open(catalog, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                if line.endswith("\n"):  # a half-written last line is simply not there yet
                    log.warning("catalog %s:%d is not JSON; skipped", catalog.name, lineno)
                continue
            if not isinstance(e, dict) or not e.get("file"):
                continue
            if root is not None and not Path(e["file"]).is_absolute():
                audio = (Path(root) / e["file"]).resolve()
                if not audio.exists() and (Path(root) / Path(e["file"]).name).exists():
                    audio = (Path(root) / Path(e["file"]).name).resolve()
            else:
                audio = resolve_catalog_file(e, catalog)
            key = e.get("sha256") or f"file:{e['file']}"
            if key in out:
                continue
            ended = _parse_utc(e.get("end_utc") or e.get("timestamp"))
            started = _parse_utc(e.get("file_start_utc") or e.get("start_utc"))
            dur = e.get("duration_sec")
            try:
                dur = float(dur) if dur is not None else None
            except (TypeError, ValueError):
                dur = None
            if started is None and ended is not None and dur:
                from datetime import timedelta

                started = ended - timedelta(seconds=dur)
            out[key] = Capture(key, audio, e, lineno, started, ended, dur)
    far = datetime.max.replace(tzinfo=timezone.utc)
    return sorted(out.values(), key=lambda c: (c.started or c.ended or far, c.line))


# -- state -----------------------------------------------------------------------------------------

class State:
    """``decode-state.json``: ``{"schema": 1, "watch_from_utc": ..., "clips": {key: record}}``.
    Written atomically; always re-read under the lock before a clip is processed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {"schema": STATE_SCHEMA, "watch_from_utc": None, "clips": {}}

    def load(self) -> "State":
        try:
            d = json.loads(self.path.read_text())
            if isinstance(d, dict) and isinstance(d.get("clips"), dict):
                self.data = {"schema": STATE_SCHEMA, "watch_from_utc": d.get("watch_from_utc"),
                             "clips": d["clips"]}
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            bad = self.path.with_name(self.path.name + f".bad-{int(time.time())}")
            log.error("state file %s unreadable (%s); moved to %s and starting fresh", self.path, exc, bad)
            with contextlib.suppress(OSError):
                self.path.replace(bad)
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    @property
    def clips(self) -> dict:
        return self.data["clips"]

    @property
    def watch_from(self) -> Optional[datetime]:
        return _parse_utc(self.data.get("watch_from_utc"))

    def set_watch_from(self, when: str) -> None:
        self.data["watch_from_utc"] = when


def next_action(rec: Optional[dict], cfg: WorkerConfig, version: str, gate_version: str) -> Optional[str]:
    """What a capture still needs: ``decode``, ``load`` (decoded but not in the DB yet) or None."""
    if not rec:
        return "decode"
    status = rec.get("status")
    if status == "decoded" and rec.get("decoder_version") == version:
        if cfg.db is not None and not rec.get("loaded"):
            return "load"
        return None
    if status == "no_voice" and rec.get("gate_version") == gate_version:
        return None
    if status in RETRY_STATUSES and rec.get("decoder_version") == version and not cfg.retry_failed:
        if int(rec.get("attempts", 0)) >= cfg.max_attempts:
            return None  # given up; --retry-failed or a new decoder version tries again
    return "decode"


def pending(captures: list[Capture], state: State, cfg: WorkerConfig, *, backfill: bool) -> list[tuple]:
    """``[(capture, action)]`` still to do, oldest first. Without ``backfill`` only captures
    that ended at or after ``watch_from_utc`` count."""
    version, gv = cfg.decoder_version(), cfg.gate_version()
    since = state.watch_from
    out = []
    for c in captures:
        if not backfill and since is not None and (c.ended is None or c.ended < since):
            continue
        action = next_action(state.clips.get(c.key), cfg, version, gv)
        if action:
            out.append((c, action))
    return out


# -- the child process (gate + decode) ---------------------------------------------------------------

def _import_attr(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def ensure_ffmpeg() -> Optional[str]:
    """Path of an ffmpeg binary, putting imageio-ffmpeg's static build on PATH (as ``ffmpeg``)
    when the system has none. None if neither exists."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # optional (decode extra)

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - not installed, or no binary for this platform
        return None
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "hfgcs" / "bin"
    cache.mkdir(parents=True, exist_ok=True)
    link = cache / "ffmpeg"
    if not link.exists() or link.resolve() != Path(exe).resolve():
        with contextlib.suppress(FileNotFoundError):
            link.unlink()
        link.symlink_to(exe)
    os.environ["PATH"] = f"{cache}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(link)


def model_status(model: str) -> tuple[bool, str]:
    """(ready, detail) for the faster-whisper model, without downloading anything."""
    p = Path(os.path.expanduser(model))
    if p.is_dir():
        return (p / "model.bin").exists(), f"local model directory {p}"
    try:
        from faster_whisper.utils import download_model
    except ImportError as exc:
        return False, f"faster-whisper is not installed ({exc})"
    try:
        path = download_model(model, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - huggingface_hub raises several types
        return False, f"model {model!r} is not in the Hugging Face cache ({type(exc).__name__})"
    return True, f"model {model!r} at {path}"


def check_environment(cfg: WorkerConfig) -> list[str]:
    """Problems that stop any decode (empty list = ready)."""
    problems = []
    if cfg.transcribe_impl is None:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            problems.append(f"faster-whisper is not installed ({exc})")
        else:
            ok, detail = model_status(cfg.model)
            if not ok:
                problems.append(detail)
    needs_ffmpeg = cfg.transcribe_impl is None or (cfg.gate and cfg.gate_impl is None)
    if needs_ffmpeg and ensure_ffmpeg() is None:
        problems.append("ffmpeg not found (install the decode extra for imageio-ffmpeg, "
                        "or apt install ffmpeg)")
    return problems


def check_in_subprocess(cfg: WorkerConfig) -> list[str]:
    """check_environment in a child, so the long-running parent never imports faster-whisper."""
    cmd = [cfg.python, "-m", "decoding.worker", "_check", "--model", cfg.model]
    if not cfg.gate:
        cmd.append("--no-gate")
    for flag, v in (("--transcribe-impl", cfg.transcribe_impl), ("--gate-impl", cfg.gate_impl)):
        if v:
            cmd += [flag, v]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH")) if p)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(REPO), env=env)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return [f"environment check failed to run: {exc}"]


def child_main(args) -> int:
    """Gate and decode one recording; write the outcome to ``args.result`` as JSON."""
    cfg = config_from_args(args)
    result: dict = {"status": "failed"}
    t0 = time.monotonic()
    try:
        problems = check_environment(cfg)
        if problems:
            result = {"status": "config_error", "error": "; ".join(problems)}
            return EXIT_CONFIG
        audio = Path(args.audio)
        entry = json.loads(args.entry) if args.entry else None
        gate_cfg = cfg.gate_config()
        if cfg.gate:
            gate = _import_attr(cfg.gate_impl) if cfg.gate_impl else None
            from .voice_gate import check_voice

            g = gate(audio, gate_cfg) if gate else check_voice(audio, gate_cfg)
            result["gate"] = g
            result["gate_seconds"] = round(time.monotonic() - t0, 1)
            if not g.get("voice", True):
                result["status"] = "no_voice"
                return 0
        from .pipeline import Decoder
        from .run import decode_recording

        transcribe = _import_attr(cfg.transcribe_impl) if cfg.transcribe_impl else None
        decoder = Decoder(cfg.decode_config(), transcribe=transcribe)
        doc = decode_recording(decoder, audio, entry, list(cfg.day_callsigns),
                               jsonl=Path(args.jsonl) if args.jsonl else None)
        side = audio.with_name(audio.name + ".decode.json")
        if doc is None:
            result.update(status="decoded", up_to_date=True, sidecar=str(side))
            with contextlib.suppress(OSError, ValueError):
                doc = json.loads(side.read_text())
        else:
            result.update(status="decoded", up_to_date=False, sidecar=str(side))
        if doc:
            bs = doc.get("broadcasts") or []
            asr = doc.get("asr") or {}
            sp = asr.get("second_pass") or {}
            result.update(
                broadcasts=len(bs),
                kinds=[f"{b.get('type')}@{b.get('start_s')}" for b in bs],
                decoder_version=doc.get("decoder_version"),
                asr_seconds=(asr.get("first_pass") or {}).get("asr_seconds", asr.get("asr_seconds")),
                second_pass_fired=sp.get("fired"),
                duration_s=doc.get("duration_s"),
            )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which keeps going
        result = {**result, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        return 1
    finally:
        result["child_seconds"] = round(time.monotonic() - t0, 1)
        Path(args.result).write_text(json.dumps(result))


# -- the parent: run children, load, keep state --------------------------------------------------------

@dataclass
class ChildResult:
    status: str  # decoded, no_voice, failed, timeout, config_error, interrupted
    seconds: float
    data: dict = field(default_factory=dict)
    returncode: Optional[int] = None
    output: str = ""


class ChildRunner:
    """Runs ``python -m decoding.worker _child ...`` in its own process group with a wall-time cap."""

    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.interrupted = False

    def command(self, c: Capture, result: Path, jsonl: Optional[Path]) -> list[str]:
        cfg = self.cfg
        cmd = [cfg.python, "-m", "decoding.worker", "_child", "--audio", str(c.audio),
               "--entry", json.dumps(c.entry), "--result", str(result),
               "--model", cfg.model, "--cpu-threads", str(cfg.cpu_threads),
               "--gate-threshold", str(cfg.gate_threshold), "--min-speech", str(cfg.min_speech_s)]
        if jsonl:
            cmd += ["--jsonl", str(jsonl)]
        if not cfg.gate:
            cmd.append("--no-gate")
        if not cfg.second_pass:
            cmd.append("--no-second-pass")
        for cs in cfg.day_callsigns:
            cmd += ["--day-callsign", cs]
        if cfg.transcribe_impl:
            cmd += ["--transcribe-impl", cfg.transcribe_impl]
        if cfg.gate_impl:
            cmd += ["--gate-impl", cfg.gate_impl]
        return cmd

    def run(self, c: Capture, timeout_s: float, jsonl: Optional[Path]) -> ChildResult:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH")) if p)
        # One worker: keep every BLAS/OpenMP pool at the configured thread count.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(var, str(self.cfg.cpu_threads))
        with tempfile.TemporaryDirectory(prefix="hfgcs-decode-") as td:
            res = Path(td) / "result.json"
            cmd = self.command(c, res, jsonl)
            t0 = time.monotonic()
            self.interrupted = False
            with open(Path(td) / "out.txt", "w+") as out:
                self.proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=out,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = self.proc.wait(timeout=timeout_s)
                    status = None
                except subprocess.TimeoutExpired:
                    self.kill()
                    rc, status = self.proc.returncode, "timeout"
                out.seek(0)
                text = out.read()[-4000:]
            seconds = round(time.monotonic() - t0, 1)
            # Stopped by us, or SIGTERM/SIGINT from systemd stopping the unit: not the clip's fault.
            if self.interrupted or (status is None and rc in (-signal.SIGTERM, -signal.SIGINT)
                                    and not res.exists()):
                return ChildResult("interrupted", seconds, {}, rc, text)
            if status == "timeout":
                return ChildResult("timeout", seconds, {"timeout_s": round(timeout_s, 1)}, rc, text)
            try:
                data = json.loads(res.read_text())
            except (OSError, ValueError):
                data = {"status": "failed", "error": f"child exited {rc} without a result"}
            return ChildResult(data.get("status", "failed"), seconds, data, rc, text)

    def kill(self, grace_s: float = 5.0) -> None:
        """SIGTERM the child's process group, then SIGKILL after ``grace_s``."""
        p = self.proc
        if p is None or p.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, sig)
            try:
                p.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def interrupt(self) -> None:
        self.interrupted = True
        self.kill(grace_s=2.0)


def run_db_tool(cfg: WorkerConfig, *argv: str) -> tuple[bool, str]:
    cmd = [cfg.python, str(cfg.db_tool), "--db", str(cfg.db), *argv]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.load_timeout_s, cwd=str(REPO))
    except subprocess.TimeoutExpired:
        return False, f"{argv[0]} timed out after {cfg.load_timeout_s:.0f} s"
    except OSError as exc:
        return False, str(exc)
    text = (r.stdout + r.stderr).strip()
    return r.returncode == 0, text


def load_decodes(cfg: WorkerConfig, sidecar: Path) -> tuple[bool, str]:
    """``hfgcs_db.py --db DB load-decodes SIDECAR --recordings-root ROOT``."""
    return run_db_tool(cfg, "load-decodes", str(sidecar), "--recordings-root", str(cfg.root()))


def ingest_catalog(cfg: WorkerConfig) -> tuple[bool, str]:
    """``hfgcs_db.py --db DB ingest-catalog CATALOG`` (idempotent) so every capture is on the
    review page, decoded or not."""
    return run_db_tool(cfg, "ingest-catalog", str(cfg.catalog), "--recordings-root", str(cfg.root()))


class ConfigError(RuntimeError):
    pass


class Worker:
    def __init__(self, cfg: WorkerConfig, *, runner: Optional[ChildRunner] = None,
                 loader: Callable = load_decodes, ingester: Callable = ingest_catalog,
                 sleep: Callable = time.sleep):
        self.cfg = cfg
        self.runner = runner or ChildRunner(cfg)
        self.loader = loader
        self.ingester = ingester
        self.sleep = sleep
        self.stop = False
        self._catalog_sig = None

    # locking: one clip at a time across every worker process (watch loop + a backfill)
    @contextlib.contextmanager
    def lock(self):
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.lock_file, "a+") as fh:
            waited = False
            while True:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self.stop:
                        raise InterruptedError("stopping") from None
                    if not waited:
                        log.info("another decode worker holds %s; waiting for its clip to finish",
                                 self.cfg.lock_file)
                        waited = True
                    self.sleep(1.0)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def state(self) -> State:
        return State(self.cfg.state_file).load()

    def maybe_ingest(self) -> None:
        cfg = self.cfg
        if not (cfg.ingest and cfg.db is not None):
            return
        try:
            st = Path(cfg.catalog).stat()
        except FileNotFoundError:
            return
        sig = (st.st_size, st.st_mtime_ns)
        if sig == self._catalog_sig:
            return
        ok, text = self.ingester(cfg)
        if ok:
            self._catalog_sig = sig
            log.info("catalog ingest: %s", text.splitlines()[-1] if text else "ok")
        else:
            log.warning("catalog ingest failed (will retry): %s", text[-500:])

    def run_pass(self, *, backfill: bool = False) -> Counter:
        """Process everything currently due, one clip at a time; return counts by outcome."""
        cfg = self.cfg
        counts: Counter = Counter()
        self.maybe_ingest()
        todo = pending(read_catalog(cfg.catalog, cfg.recordings_root), self.state(), cfg, backfill=backfill)
        if todo:
            log.info("%d capture(s) to process%s", len(todo), " (backfill)" if backfill else "")
        for c, _ in todo:
            if self.stop:
                break
            try:
                with self.lock():
                    st = self.state()  # another worker may have done it meanwhile
                    action = next_action(st.clips.get(c.key), cfg, cfg.decoder_version(), cfg.gate_version())
                    if action is None:
                        counts["done_elsewhere"] += 1
                        continue
                    outcome = self.process(c, action, st)
            except InterruptedError:
                break
            counts[outcome] += 1
        return counts

    def process(self, c: Capture, action: str, st: State) -> str:
        cfg = self.cfg
        version = cfg.decoder_version()
        rec = dict(st.clips.get(c.key) or {})
        if action == "decode":
            if not c.audio.is_file():
                attempts = self._attempts(rec, version) + 1
                rec.update(status="missing", file=str(c.audio), decoder_version=version, attempts=attempts,
                           at=utc_now(), error="audio file not found")
                self._save(st, c, rec)
                log.warning("missing audio %s (attempt %d/%d)", c.audio, attempts, cfg.max_attempts)
                return "missing"
            timeout = cfg.timeout_for(c.duration_s)
            jsonl = cfg.jsonl or Path(cfg.catalog).parent / "decodes.jsonl"
            log.info("decoding %s (%.1f s audio, cap %.0f s)", c.name, c.duration_s or 0, timeout)
            r = self.runner.run(c, timeout, jsonl)
            if r.status == "interrupted":
                log.info("%s: interrupted by shutdown; it will be decoded on the next start", c.name)
                return "interrupted"
            if r.status == "config_error":
                raise ConfigError(r.data.get("error") or "decoder not ready")
            rate = f", RTF {r.seconds / c.duration_s:.2f}" if c.duration_s else ""
            base = {"file": str(c.audio), "decoder_version": version, "at": utc_now(),
                    "wall_seconds": r.seconds, "duration_s": c.duration_s}
            if r.status == "no_voice":
                g = r.data.get("gate") or {}
                rec = {**base, "status": "no_voice", "gate_version": cfg.gate_version(), "gate": g,
                       "attempts": 0}
                self._save(st, c, rec)
                log.info("skipped %s: no voice (%s) in %.1f s", c.name, g.get("reason"), r.seconds)
                return "no_voice"
            if r.status == "decoded":
                rec = {**base, "status": "decoded", "loaded": False, "attempts": 0,
                       "sidecar": r.data.get("sidecar"), "broadcasts": r.data.get("broadcasts"),
                       "kinds": r.data.get("kinds"), "asr_seconds": r.data.get("asr_seconds"),
                       "second_pass_fired": r.data.get("second_pass_fired"), "gate": r.data.get("gate")}
                self._save(st, c, rec)
                log.info("decoded %s in %.1f s wall%s: %s broadcast(s) %s%s", c.name, r.seconds, rate,
                         rec["broadcasts"], rec["kinds"] or "",
                         " (sidecar already up to date)" if r.data.get("up_to_date") else "")
            else:
                attempts = self._attempts(rec, version) + 1
                err = r.data.get("error") or (f"over the {timeout:.0f} s cap; killed" if r.status == "timeout"
                                              else f"exit {r.returncode}")
                rec = {**base, "status": r.status, "attempts": attempts, "error": err,
                       "output_tail": r.output[-1500:]}
                self._save(st, c, rec)
                log.warning("%s %s after %.1f s (attempt %d/%d): %s", r.status.upper(), c.name, r.seconds,
                            attempts, cfg.max_attempts, err)
                return r.status
        # load (fresh decode, or one whose load failed before)
        if cfg.db is None:
            return "decoded" if action == "decode" else "noop"
        sidecar = Path(rec.get("sidecar") or c.audio.with_name(c.audio.name + ".decode.json"))
        ok, text = self.loader(cfg, sidecar)
        rec["loaded"] = ok
        rec["load_output"] = text[-1000:]
        self._save(st, c, rec)
        if ok:
            log.info("loaded %s: %s", c.name, " | ".join(text.splitlines()[:2]))
        else:
            log.warning("load-decodes failed for %s (will retry): %s", c.name, text[-500:])
        if action == "decode":
            return "decoded" if ok else "decoded_not_loaded"
        return "loaded" if ok else "load_failed"

    @staticmethod
    def _attempts(rec: dict, version: str) -> int:
        if rec.get("status") in RETRY_STATUSES and rec.get("decoder_version") == version:
            return int(rec.get("attempts", 0))
        return 0

    def _save(self, st: State, c: Capture, rec: dict) -> None:
        st.clips[c.key] = rec
        st.save()
        hist = {"utc": utc_now(), "key": c.key, "file": c.audio.name, **{
            k: rec.get(k) for k in ("status", "wall_seconds", "duration_s", "broadcasts", "kinds",
                                    "loaded", "error", "attempts")}}
        with contextlib.suppress(OSError):
            with open(self.cfg.history_file, "a") as fh:
                fh.write(json.dumps(hist) + "\n")

    def init_watch_from(self) -> None:
        with self.lock():
            st = self.state()
            if st.watch_from is None:
                st.set_watch_from(utc_now())
                st.save()
                log.info("first start: watching captures that end after %s; run --backfill for older ones",
                         st.data["watch_from_utc"])

    def watch(self) -> int:
        self.init_watch_from()
        while not self.stop:
            counts = self.run_pass()
            if counts:
                log.info("pass done: %s", summarize(counts))
            slept = 0.0
            while not self.stop and slept < self.cfg.poll_s:
                self.sleep(min(1.0, self.cfg.poll_s - slept))
                slept += 1.0
        return 0

    def backfill(self) -> int:
        total: Counter = Counter()
        while not self.stop:
            counts = self.run_pass(backfill=True)
            total += counts
            if not counts or set(counts) <= {"done_elsewhere"}:
                break
            work = sum(v for k, v in counts.items() if k not in ("missing", "timeout", "failed"))
            if not work:
                break
        log.info("backfill done: %s", summarize(total))
        return 1 if total.get("failed") or total.get("timeout") else 0

    def request_stop(self, *_):
        self.stop = True
        self.runner.interrupt()


# -- CLI --------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m decoding.worker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="one pass over new captures, then exit")
    mode.add_argument("--backfill", action="store_true",
                      help="every capture in the catalog, including those before the watch started; exits")
    mode.add_argument("--check", action="store_true", help="check dependencies, ffmpeg and the model; exit")
    mode.add_argument("--prefetch", action="store_true", help="download the ASR model, then exit")
    mode.add_argument("--status", action="store_true", help="print the state file per capture; exit")
    env_catalog = os.environ.get("HFGCS_CATALOG")
    ap.add_argument("--catalog", type=Path,
                    default=Path(env_catalog) if env_catalog else REPO / "hfgcs_catalog.jsonl",
                    help="recorder catalog (default: $HFGCS_CATALOG, else <repo>/hfgcs_catalog.jsonl)")
    env_root = os.environ.get("HFGCS_RECORDINGS_ROOT")
    ap.add_argument("--recordings-root", type=Path, default=Path(env_root) if env_root else None,
                    help="base of the catalog's 'file' paths "
                    "(default: $HFGCS_RECORDINGS_ROOT, else the catalog's dir)")
    env_db = os.environ.get("HFGCS_DB")
    ap.add_argument("--db", type=Path, default=Path(env_db) if env_db else None,
                    help="SQLite DB for load-decodes (default: $HFGCS_DB; none = write sidecars only)")
    ap.add_argument("--jsonl", type=Path, help="decodes.jsonl to append to (default: beside the catalog)")
    env_state = os.environ.get("HFGCS_DECODE_STATE_DIR")
    ap.add_argument("--state-dir", type=Path, default=Path(env_state) if env_state else default_state_dir(),
                    help="state, history and lock files (default: ~/.local/state/hfgcs)")
    ap.add_argument("--model", default=_env("HFGCS_DECODE_MODEL", "small.en"),
                    help="ASR model (default: $HFGCS_DECODE_MODEL, else small.en)")
    ap.add_argument("--cpu-threads", type=int, default=_env("HFGCS_DECODE_THREADS", 3),
                    help="ASR threads (default: $HFGCS_DECODE_THREADS, else 3)")
    ap.add_argument("--no-second-pass", action="store_true", help="disable the vote3 second pass")
    ap.add_argument("--day-callsign", action="append", default=[], help="today's callsigns (ASR prompt)")
    ap.add_argument("--no-gate", action="store_true", help="decode every clip (no voice gate)")
    ap.add_argument("--gate-threshold", type=float, default=_env("HFGCS_DECODE_GATE_THRESHOLD", 0.3),
                    help="silero speech threshold (default 0.3)")
    ap.add_argument("--min-speech", type=float, default=_env("HFGCS_DECODE_MIN_SPEECH_S", 1.0),
                    help="seconds of detected speech a clip needs to be decoded (default 1.0)")
    ap.add_argument("--timeout-factor", type=float, default=_env("HFGCS_DECODE_TIMEOUT_FACTOR", 10.0),
                    help="wall-time cap per clip = factor x clip length (default 10)")
    ap.add_argument("--timeout-min", type=float, default=_env("HFGCS_DECODE_TIMEOUT_MIN_S", 120.0),
                    help="lower bound of the cap, s (default 120: model load + short clips)")
    ap.add_argument("--timeout-max", type=float, default=_env("HFGCS_DECODE_TIMEOUT_MAX_S", 7200.0),
                    help="upper bound of the cap, s (default 7200: 3 passes x 2x a 900 s clip)")
    ap.add_argument("--max-attempts", type=int, default=_env("HFGCS_DECODE_MAX_ATTEMPTS", 2),
                    help="tries per clip for failures/timeouts (default 2)")
    ap.add_argument("--retry-failed", action="store_true", help="try failed/timed-out clips again")
    ap.add_argument("--poll", type=float, default=_env("HFGCS_DECODE_POLL_S", 30.0),
                    help="watch loop poll interval, s (default 30)")
    ap.add_argument("--no-ingest", action="store_true",
                    help="do not run ingest-catalog when the catalog grows")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--transcribe-impl", help=argparse.SUPPRESS)
    ap.add_argument("--gate-impl", help=argparse.SUPPRESS)
    return ap


def build_child_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m decoding.worker _child")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--entry")
    ap.add_argument("--result", required=True)
    ap.add_argument("--jsonl")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--cpu-threads", type=int, default=3)
    ap.add_argument("--no-second-pass", action="store_true")
    ap.add_argument("--day-callsign", action="append", default=[])
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.3)
    ap.add_argument("--min-speech", type=float, default=1.0)
    ap.add_argument("--transcribe-impl")
    ap.add_argument("--gate-impl")
    return ap


def config_from_args(args) -> WorkerConfig:
    g = getattr
    cfg = WorkerConfig(
        model=args.model, cpu_threads=args.cpu_threads, second_pass=not args.no_second_pass,
        day_callsigns=tuple(args.day_callsign), gate=not args.no_gate, gate_threshold=args.gate_threshold,
        min_speech_s=args.min_speech, transcribe_impl=args.transcribe_impl, gate_impl=args.gate_impl,
    )
    if g(args, "catalog", None) is not None:
        cfg = replace(
            cfg, catalog=args.catalog, recordings_root=args.recordings_root, db=args.db, jsonl=args.jsonl,
            state_dir=args.state_dir, timeout_factor=args.timeout_factor, timeout_min_s=args.timeout_min,
            timeout_max_s=args.timeout_max, max_attempts=args.max_attempts, retry_failed=args.retry_failed,
            poll_s=args.poll, ingest=not args.no_ingest,
        )
    return cfg


def print_status(cfg: WorkerConfig) -> int:
    st = State(cfg.state_file).load()
    caps = read_catalog(cfg.catalog, cfg.recordings_root)
    version, gv = cfg.decoder_version(), cfg.gate_version()
    print(f"catalog {cfg.catalog}: {len(caps)} capture(s); state {cfg.state_file}; "
          f"watching captures that end after {st.data.get('watch_from_utc') or '(not started yet)'}")
    print(f"decoder_version {version}")
    since = st.watch_from
    counts: Counter = Counter()
    for c in caps:
        rec = st.clips.get(c.key) or {}
        action = next_action(rec or None, cfg, version, gv)
        backlog = since is not None and (c.ended is None or c.ended < since)
        label = rec.get("status", "new") + (f" (next: {action})" if action else "")
        if action and backlog:
            label += " [backlog: --backfill]"
        counts[rec.get("status", "new")] += 1
        extra = f" {rec.get('wall_seconds')} s" if rec.get("wall_seconds") is not None else ""
        print(f"  {c.name}: {label}{extra}")
    print("totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_child"]:
        return child_main(build_child_parser().parse_args(argv[1:]))
    if argv[:1] == ["_check"]:
        cargs = build_child_parser().parse_args(["--audio", "-", "--result", "-", *argv[1:]])
        print(json.dumps(check_environment(config_from_args(cargs))))
        return 0
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    logging.Formatter.converter = time.gmtime  # log times in UTC
    cfg = config_from_args(args)
    if args.status:
        return print_status(cfg)
    if args.prefetch:
        try:
            from faster_whisper.utils import download_model
        except ImportError as exc:
            print(f"faster-whisper is not installed ({exc}).\n" + INSTALL_HELP.format(model=cfg.model))
            return EXIT_CONFIG
        print(f"model {cfg.model} ready at {download_model(cfg.model)}")
        return 0
    problems = check_environment(cfg) if args.check else check_in_subprocess(cfg)
    if args.check:
        for p in problems:
            print(f"NOT READY: {p}")
        if problems:
            print(INSTALL_HELP.format(model=cfg.model))
            return EXIT_CONFIG
        print(f"ready: model {cfg.model}, ffmpeg {shutil.which('ffmpeg') or ensure_ffmpeg()}, "
              f"decoder_version {cfg.decoder_version()}")
        return 0
    if problems:
        log.error("decoder not ready: %s\n%s", "; ".join(problems), INSTALL_HELP.format(model=cfg.model))
        return EXIT_CONFIG
    if cfg.db is None:
        log.warning("no --db / $HFGCS_DB: decodes are written as sidecars but not loaded into the DB")
    log.info("decode worker: catalog %s, db %s, model %s, %d thread(s), gate %s, cap %gx clip in [%g, %g] s",
             cfg.catalog, cfg.db, cfg.model, cfg.cpu_threads, cfg.gate_version(), cfg.timeout_factor,
             cfg.timeout_min_s, cfg.timeout_max_s)
    worker = Worker(cfg)
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    try:
        if args.backfill:
            return worker.backfill()
        if args.once:
            worker.init_watch_from()
            counts = worker.run_pass()
            log.info("pass done: %s", summarize(counts))
            return 1 if counts.get("failed") or counts.get("timeout") else 0
        return worker.watch()
    except ConfigError as exc:
        log.error("decoder not ready: %s\n%s", exc, INSTALL_HELP.format(model=cfg.model))
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
