"""Keyword alerts for HFGCS decodes: match, clip audio, push to ntfy with an attachment.

Watches ``decodes.jsonl`` (or replays it with ``--dry-run``). Replaces the ad-hoc
``hfgcs-skyking-watch.py`` which posted Skyking text with no audio and no confidence
floor, so Whisper prompt-hallucinations on truncated noise clips paged the phone.

Bomber callsigns that are also common English words (doom, rogue, skull) only match
with callsign context (digits after the word, or nearby procedure words). See the
example TOML and docs/keyword-alerts.md.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]

# America/Los_Angeles offset is applied via zoneinfo when available.
try:
    from zoneinfo import ZoneInfo
    _PT = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover
    _PT = timezone(timedelta(hours=-8), name="PST")

DEFAULT_CONFIG = Path.home() / ".config/hfgcs/keyword-alerts.toml"
DEFAULT_STATE = Path.home() / ".local/state/hfgcs/keyword-ntfy.offset"
DEFAULT_DEDUPE = Path.home() / ".local/state/hfgcs/keyword-ntfy-dedupe.json"
POLL_S = 2.0
NTFY_SH_ATTACH_MAX = 2 * 1024 * 1024  # public ntfy.sh per-file limit
DEFAULT_PAD_S = 12.0
DEFAULT_DEDUPE_S = 120.0
DEFAULT_MIN_CONF = 0.25
DEFAULT_MIN_WORD_PROB = 0.35

# Procedure words that make a short bomber callsign look like traffic, not prose.
_PROCEDURE_BIGRAMS = (
    ("this", "is"),
    ("radio", "check"),
    ("how", "copy"),
    ("how", "do"),
    ("do", "you"),
    ("stand", "by"),
    ("all", "stations"),
    ("more", "to"),
    ("test", "count"),
    ("do", "not"),
)


@dataclass
class Keyword:
    name: str
    patterns: list[re.Pattern[str]]
    require_callsign_context: bool = False
    min_confidence: float = DEFAULT_MIN_CONF
    priority: str = "4"
    tags: str = "rotating_light"


@dataclass
class AlertConfig:
    keywords: list[Keyword] = field(default_factory=list)
    pad_s: float = DEFAULT_PAD_S
    dedupe_s: float = DEFAULT_DEDUPE_S
    min_word_prob: float = DEFAULT_MIN_WORD_PROB
    ntfy_server: str = "https://ntfy.sh"
    ntfy_topic: str = ""
    ntfy_token: str = ""
    attach_max_bytes: int = NTFY_SH_ATTACH_MAX
    review_base_url: str = ""  # e.g. http://192.168.2.214:8090
    recordings_root: Path = Path(".")
    enabled: bool = True


@dataclass(frozen=True)
class Match:
    keyword: str
    matched_text: str
    start_s: float
    end_s: float
    confidence: float
    word_prob: float
    transcript: str
    broadcast_type: str
    require_context_ok: bool = True


@dataclass
class Alert:
    matches: list[Match]
    frequency_khz: Optional[float]
    receiver_id: str
    file: Path
    decoded_at: str
    start_utc: str
    channels: list[str] = field(default_factory=list)  # filled when deduped
    capture_start_utc: str = ""


def _compile_patterns(raw: Iterable[str]) -> list[re.Pattern[str]]:
    out = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        if p.startswith("/") and p.endswith("/") and len(p) > 2:
            out.append(re.compile(p[1:-1], re.IGNORECASE))
        else:
            # fuzzy spacing: "sky king" / "skyking"
            esc = re.escape(p).replace(r"\ ", r"[\s\-]*")
            out.append(re.compile(rf"(?<![a-z0-9]){esc}(?![a-z0-9])", re.IGNORECASE))
    return out


def default_keywords() -> list[Keyword]:
    """Built-in list matching the user's research; override via TOML."""
    return [
        Keyword("skyking", _compile_patterns([
            "skyking", "sky king", "sky kin", "skyking skyking do not answer",
        ]), min_confidence=DEFAULT_MIN_CONF, priority="5", tags="warning,loud"),
        Keyword("skymaster", _compile_patterns(["skymaster", "sky master"]), priority="5", tags="warning"),
        Keyword("skybird", _compile_patterns(["skybird", "sky bird"]), priority="5", tags="warning"),
        Keyword("nightwatch", _compile_patterns([
            "nightwatch", "night watch", r"/night\s*watch\s*0?[1-4](\s*(lima|charlie))?/",
            "NW1", "NW2", "NW3", "NW4",
        ]), priority="5", tags="airplane"),
        Keyword("gordo", _compile_patterns([r"/gordo\s*\d{1,2}/", "gordo"]),
                require_callsign_context=True, priority="5", tags="rotating_light"),
        Keyword("doom", _compile_patterns(["doom"]), require_callsign_context=True, tags="airplane"),
        Keyword("rogue", _compile_patterns(["rogue"]), require_callsign_context=True, tags="airplane"),
        Keyword("skull", _compile_patterns(["skull"]), require_callsign_context=True, tags="airplane"),
        Keyword("baffy", _compile_patterns(["baffy"]), tags="airplane"),
        Keyword("panga", _compile_patterns(["panga"]), tags="airplane"),
        Keyword("pyrex", _compile_patterns(["pyrex"]), tags="airplane"),
        Keyword("zolar", _compile_patterns(["zolar"]), tags="airplane"),
    ]


def load_config(path: Optional[Path] = None, *, env: Optional[dict] = None) -> AlertConfig:
    env = env or os.environ
    cfg = AlertConfig(
        keywords=default_keywords(),
        ntfy_server=env.get("NTFY_SERVER", "https://ntfy.sh"),
        ntfy_topic=env.get("NTFY_TOPIC", ""),
        ntfy_token=env.get("NTFY_TOKEN", "") or env.get("NTFY_ACCESS_TOKEN", ""),
        review_base_url=env.get("HFGCS_REVIEW_URL", "").rstrip("/"),
        recordings_root=Path(env.get("HFGCS_RECORDINGS", env.get("HFGCS_RECORDINGS_ROOT", "."))).expanduser(),
        attach_max_bytes=int(env.get("NTFY_ATTACH_MAX", str(NTFY_SH_ATTACH_MAX))),
    )
    path = path or Path(env.get("HFGCS_KEYWORD_ALERTS", str(DEFAULT_CONFIG)))
    if path and path.is_file():
        data = _toml.loads(path.read_text())
        cfg.pad_s = float(data.get("pad_s", cfg.pad_s))
        cfg.dedupe_s = float(data.get("dedupe_s", cfg.dedupe_s))
        cfg.min_word_prob = float(data.get("min_word_prob", cfg.min_word_prob))
        cfg.enabled = bool(data.get("enabled", True))
        if "review_base_url" in data:
            cfg.review_base_url = str(data["review_base_url"]).rstrip("/")
        if "attach_max_bytes" in data:
            cfg.attach_max_bytes = int(data["attach_max_bytes"])
        kws = data.get("keywords")
        if isinstance(kws, list) and kws:
            loaded = []
            for item in kws:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                loaded.append(Keyword(
                    name=str(item["name"]),
                    patterns=_compile_patterns(item.get("patterns") or [item["name"]]),
                    require_callsign_context=bool(item.get("require_callsign_context", False)),
                    min_confidence=float(item.get("min_confidence", DEFAULT_MIN_CONF)),
                    priority=str(item.get("priority", "4")),
                    tags=str(item.get("tags", "rotating_light")),
                ))
            if loaded:
                cfg.keywords = loaded
    if env.get("NTFY_SERVER"):
        cfg.ntfy_server = env["NTFY_SERVER"]
    if env.get("NTFY_TOPIC"):
        cfg.ntfy_topic = env["NTFY_TOPIC"]
    return cfg


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _has_callsign_context(text: str, match: re.Match[str]) -> bool:
    """Digits after the keyword, or a procedure bigram in a short window around it."""
    if re.search(r"\b\d{1,3}\b", text[match.end(): match.end() + 12]):
        return True
    # Also "DOOM12" / "DOOM-12" glued to the match
    if re.match(r"[\s\-]*\d{1,3}\b", text[match.end(): match.end() + 8]):
        return True
    span = text[max(0, match.start() - 48): match.end() + 48]
    words = re.findall(r"[a-z0-9]+", _norm_text(span))
    for a, b in _PROCEDURE_BIGRAMS:
        for i in range(len(words) - 1):
            if words[i] == a and words[i + 1] == b:
                return True
    return False


def _word_times(doc: dict, broadcast: dict) -> tuple[float, float, float]:
    """Return (start_s, end_s, max_word_probability) for the broadcast."""
    start = float(broadcast.get("start_s") or 0.0)
    end = float(broadcast.get("end_s") or start)
    words = (doc.get("asr") or {}).get("words") or []
    span = broadcast.get("asr_word_span") or [0, -1]
    probs: list[float] = []
    if isinstance(span, (list, tuple)) and len(span) == 2 and span[1] >= span[0] and words:
        chunk = words[int(span[0]): int(span[1]) + 1]
        probs = [float(w.get("probability") or 0) for w in chunk]
        if chunk:
            try:
                start = float(chunk[0].get("start", start))
                end = float(chunk[-1].get("end", end))
            except (TypeError, ValueError):
                pass
    return start, max(end, start), (max(probs) if probs else 0.0)


def _is_prompt_hallucination(doc: dict, broadcast: dict, word_prob: float, min_word_prob: float) -> bool:
    """Whisper often repeats 'Skyking, skyking, do not answer' on truncated / noisy audio."""
    rec = doc.get("recording") or {}
    raw = broadcast.get("transcript") or ""
    text = _norm_text(raw)
    if "skyking" not in text and "do not answer" not in text:
        return False
    conf = float(broadcast.get("confidence") or 0)
    flags = set(broadcast.get("flags") or [])
    missing = {"missing_codeword", "missing_time", "missing_authentication"} & flags
    skyking_n = text.count("skyking")
    # Low word probability + zero confidence: classic prompt echo.
    if conf <= 0 and word_prob < min_word_prob:
        return True
    # Empty Skyking fields at conf 0: not a real read, even if Whisper is "sure".
    if conf <= 0 and len(missing) >= 2:
        # Preamble-only / loop / "i say again" spam
        stripped = re.sub(
            r"\b(skyking|do|not|answer|i|say|again|this|is|out|statement)\b|[.,]",
            " ",
            text,
        )
        leftover = re.findall(r"[a-z0-9]+", stripped)
        if skyking_n >= 3 and len(leftover) <= 2:
            return True
        if skyking_n >= 1 and len(leftover) == 0:
            return True
        # Radio-check / test-count after a phantom preamble
        if re.search(r"\b(test count|radio check|how copy|how do you read)\b", text):
            return True
        # Truncated file with no codeword/time/auth
        if rec.get("truncated") and len(missing) >= 3:
            return True
    return False


def find_matches(doc: dict, cfg: AlertConfig) -> list[Match]:
    out: list[Match] = []
    for b in doc.get("broadcasts") or []:
        transcript = b.get("transcript") or ""
        if not transcript.strip():
            continue
        conf = float(b.get("confidence") or 0)
        start_s, end_s, word_prob = _word_times(doc, b)
        if _is_prompt_hallucination(doc, b, word_prob, cfg.min_word_prob):
            continue
        # Proposed Skyking gate (decoding.hallucination): require a short codeword,
        # and either confidence or agreement. Routine traffic is never alerted.
        if str(b.get("type") or "") == "skyking" or "skyking" in _norm_text(transcript)[:40]:
            from decoding.hallucination import is_routine_traffic, skyking_alert_ok

            if is_routine_traffic(b):
                continue
            ok, _reason = skyking_alert_ok(b, min_confidence=DEFAULT_MIN_CONF)
            # Keep legacy keyword matching for non-skyking keywords; for the
            # skyking keyword itself, only proceed when the gate says alert.
            b = dict(b)
            b["_skyking_gate_ok"] = ok
        norm = _norm_text(transcript)
        for kw in cfg.keywords:
            if conf < kw.min_confidence and conf != 0:
                # conf==0 is common on partial parses; fall back to word prob
                if word_prob < max(kw.min_confidence, cfg.min_word_prob):
                    continue
            elif conf < kw.min_confidence:
                if word_prob < max(kw.min_confidence, cfg.min_word_prob):
                    continue
            if kw.name == "skyking" and b.get("_skyking_gate_ok") is False:
                continue
            for pat in kw.patterns:
                m = pat.search(norm)
                if not m:
                    # also try original transcript for NW1-style tokens
                    m = pat.search(transcript)
                if not m:
                    continue
                ctx_ok = True
                if kw.require_callsign_context:
                    ctx_ok = _has_callsign_context(norm, m) or _has_callsign_context(
                        _norm_text(transcript), m
                    )
                    if not ctx_ok:
                        continue
                out.append(Match(
                    keyword=kw.name,
                    matched_text=m.group(0),
                    start_s=start_s,
                    end_s=end_s,
                    confidence=conf,
                    word_prob=word_prob,
                    transcript=transcript.strip(),
                    broadcast_type=str(b.get("type") or ""),
                    require_context_ok=ctx_ok,
                ))
                break
    return out


def resolve_audio(doc: dict, recordings_root: Path) -> Optional[Path]:
    rec = doc.get("recording") or {}
    file = rec.get("file") or doc.get("file") or ""
    if not file:
        return None
    p = Path(file)
    if not p.is_absolute():
        p = recordings_root / p
        if not p.exists():
            p = recordings_root / Path(file).name
    return p if p.is_file() else None


def alert_from_doc(doc: dict, cfg: AlertConfig) -> Optional[Alert]:
    matches = find_matches(doc, cfg)
    if not matches:
        return None
    audio = resolve_audio(doc, cfg.recordings_root)
    if audio is None:
        return None  # never alert without audio on disk
    rec = doc.get("recording") or {}
    # Prefer the highest-confidence match for the clip window
    best = max(matches, key=lambda m: (m.confidence, m.word_prob, m.end_s - m.start_s))
    start_utc = ""
    for b in doc.get("broadcasts") or []:
        if b.get("transcript") == best.transcript and b.get("start_utc"):
            start_utc = b["start_utc"]
            break
    if not start_utc:
        # derive from recording start + offset when present
        start_utc = rec.get("start_utc") or doc.get("decoded_at") or ""
    return Alert(
        matches=matches,
        frequency_khz=rec.get("frequency_khz"),
        receiver_id=str(rec.get("receiver_id") or ""),
        file=audio,
        decoded_at=str(doc.get("decoded_at") or ""),
        start_utc=str(start_utc),
        capture_start_utc=str(rec.get("start_utc") or ""),
        channels=[_channel_label(rec.get("frequency_khz"), rec.get("receiver_id"))],
    )


def _channel_label(khz: Any, receiver: Any) -> str:
    try:
        k = f"{float(khz):g}"
    except (TypeError, ValueError):
        k = "?"
    return f"{k} kHz ({receiver or 'unknown'})"


def _parse_utc(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def dedupe_key(alert: Alert) -> str:
    kws = ",".join(sorted({m.keyword for m in alert.matches}))
    # rough text fingerprint: first 80 chars of best transcript
    text = max(alert.matches, key=lambda m: m.confidence).transcript.lower()
    text = re.sub(r"\s+", " ", text)[:80]
    return f"{kws}|{text}"


def merge_deduped(existing: Alert, new: Alert) -> Alert:
    existing.matches.extend(new.matches)
    for ch in new.channels:
        if ch not in existing.channels:
            existing.channels.append(ch)
    # keep the higher-confidence audio
    old_best = max(existing.matches, key=lambda m: (m.confidence, m.word_prob))
    new_best = max(new.matches, key=lambda m: (m.confidence, m.word_prob))
    if (new_best.confidence, new_best.word_prob) > (old_best.confidence, old_best.word_prob):
        existing.file = new.file
        existing.frequency_khz = new.frequency_khz
        existing.receiver_id = new.receiver_id
        existing.start_utc = new.start_utc or existing.start_utc
    return existing


class DedupeStore:
    def __init__(self, path: Path, window_s: float):
        self.path = path
        self.window_s = window_s
        self._items: dict[str, dict] = {}
        if path.is_file():
            try:
                self._items = json.loads(path.read_text())
            except json.JSONDecodeError:
                self._items = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, indent=0))

    def accept(self, alert: Alert) -> tuple[bool, Optional[Alert]]:
        """Return (should_send_now, alert_to_send). May hold briefly — we send immediately
        but skip duplicates within the window; first wins with channel list updated in state."""
        key = dedupe_key(alert)
        now = _parse_utc(alert.start_utc) or _parse_utc(alert.decoded_at) or datetime.now(timezone.utc)
        # purge old
        keep = {}
        for k, v in self._items.items():
            t = _parse_utc(v.get("at", ""))
            if t and (now - t).total_seconds() <= self.window_s * 3:
                keep[k] = v
        self._items = keep
        prev = self._items.get(key)
        if prev:
            t = _parse_utc(prev.get("at", ""))
            if t and abs((now - t).total_seconds()) <= self.window_s:
                channels = list(dict.fromkeys([*(prev.get("channels") or []), *alert.channels]))
                prev["channels"] = channels
                self._items[key] = prev
                self.save()
                return False, None  # duplicate simulcast
        self._items[key] = {
            "at": (alert.start_utc or alert.decoded_at or now.isoformat()),
            "channels": list(alert.channels),
            "keywords": sorted({m.keyword for m in alert.matches}),
        }
        self.save()
        return True, alert


def cut_wav_segment(src: Path, start_s: float, end_s: float, pad_s: float) -> bytes:
    """Return mono PCM WAV bytes for [start-pad, end+pad], clipped to the file."""
    with wave.open(str(src), "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        nch = w.getnchannels()
        nframes = w.getnframes()
        dur = nframes / float(rate) if rate else 0
        a = max(0.0, start_s - pad_s)
        b = min(dur, end_s + pad_s)
        if b <= a:
            a, b = 0.0, min(dur, max(pad_s * 2, 1.0))
        w.setpos(int(a * rate))
        frames = w.readframes(max(1, int((b - a) * rate)))
        if nch > 1:
            # Average channels without audioop (removed in 3.13).
            mv = memoryview(frames).cast("B")
            mono = bytearray()
            step = width * nch
            for i in range(0, len(mv), step):
                # take first channel only (USB voice is usually mono already)
                mono.extend(mv[i:i + width])
            frames = bytes(mono)
            nch = 1
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(nch)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(frames)
        return buf.getvalue()


def encode_mp3(wav_bytes: bytes, *, bitrate: str = "32k") -> Optional[bytes]:
    """Encode WAV bytes to a small MP3. Prefer lame, then imageio-ffmpeg, then ffmpeg."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav_path = td_path / "clip.wav"
        mp3_path = td_path / "clip.mp3"
        wav_path.write_bytes(wav_bytes)
        lame = shutil.which("lame")
        if lame:
            r = subprocess.run(
                [lame, "--silent", "-b", bitrate.rstrip("k"), str(wav_path), str(mp3_path)],
                capture_output=True,
            )
            if r.returncode == 0 and mp3_path.is_file():
                return mp3_path.read_bytes()
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if ffmpeg:
            r = subprocess.run(
                [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate,
                 str(mp3_path)],
                capture_output=True,
            )
            if r.returncode == 0 and mp3_path.is_file():
                return mp3_path.read_bytes()
    return None


def build_clip(alert: Alert, cfg: AlertConfig) -> tuple[Optional[bytes], str, str]:
    """Return (mp3_or_wav_bytes, filename, content_hint)."""
    best = max(alert.matches, key=lambda m: (m.confidence, m.word_prob, m.end_s - m.start_s))
    try:
        wav_bytes = cut_wav_segment(alert.file, best.start_s, best.end_s, cfg.pad_s)
    except Exception:
        return None, "", ""
    mp3 = encode_mp3(wav_bytes)
    khz = f"{alert.frequency_khz:g}" if isinstance(alert.frequency_khz, (int, float)) else "x"
    if mp3 and len(mp3) <= cfg.attach_max_bytes:
        return mp3, f"hfgcs-{khz}-{best.keyword}.mp3", "audio/mpeg"
    if mp3 and len(mp3) > cfg.attach_max_bytes:
        # retry lower bitrate
        mp3 = encode_mp3(wav_bytes, bitrate="24k")
        if mp3 and len(mp3) <= cfg.attach_max_bytes:
            return mp3, f"hfgcs-{khz}-{best.keyword}.mp3", "audio/mpeg"
    if len(wav_bytes) <= cfg.attach_max_bytes:
        return wav_bytes, f"hfgcs-{khz}-{best.keyword}.wav", "audio/wav"
    return None, "", ""  # too big; caller falls back to review link only


def format_times(utc_s: str) -> str:
    dt = _parse_utc(utc_s)
    if not dt:
        return utc_s or "(unknown time)"
    pt = dt.astimezone(_PT)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC / {pt.strftime('%Y-%m-%d %I:%M:%S %p %Z')}"


def format_message(alert: Alert, cfg: AlertConfig, *, attached: bool) -> tuple[str, str, str, str]:
    kws = sorted({m.keyword for m in alert.matches})
    best = max(alert.matches, key=lambda m: (m.confidence, m.word_prob))
    khz = f"{alert.frequency_khz:g}" if isinstance(alert.frequency_khz, (int, float)) else "?"
    title = f"HFGCS {', '.join(kws)} {khz} kHz"
    snippet = best.transcript
    if len(snippet) > 280:
        snippet = snippet[:277] + "..."
    lines = [
        format_times(alert.start_utc or alert.decoded_at),
        "Channels: " + ", ".join(alert.channels),
        f"Matched: {', '.join(kws)} ({best.matched_text!r})",
        f"Confidence: {best.confidence:.2f} (word p={best.word_prob:.2f})",
        "",
        snippet,
    ]
    if cfg.review_base_url:
        # Review app indexes by capture; without DB id, link home + filename
        lines.append("")
        lines.append(f"Review: {cfg.review_base_url}/  ({alert.file.name})")
    if not attached:
        lines.append("")
        lines.append(f"Audio on disk: {alert.file.name} (attachment skipped: too large or encoder missing)")
    # pick priority/tags from first matching keyword config
    priority, tags = "4", "rotating_light"
    for kw in cfg.keywords:
        if kw.name in kws:
            priority, tags = kw.priority, kw.tags
            break
    return title, "\n".join(lines)[:1800], priority, tags


def ntfy_publish(
    cfg: AlertConfig,
    *,
    title: str,
    message: str,
    priority: str,
    tags: str,
    attachment: Optional[bytes] = None,
    filename: str = "",
    dry_run: bool = False,
) -> str:
    """POST/PUT to ntfy. Returns a short status string. Never prints tokens."""
    if not cfg.ntfy_topic:
        return "skipped: no NTFY_TOPIC"
    url = cfg.ntfy_server.rstrip("/") + "/" + cfg.ntfy_topic
    headers = {
        "Title": title.encode("utf-8", "replace").decode("latin-1", "replace"),
        "Priority": priority,
        "Tags": tags,
        "Markdown": "yes",
    }
    if cfg.ntfy_token:
        headers["Authorization"] = f"Bearer {cfg.ntfy_token}"
    if dry_run:
        return f"dry-run: would post {title!r} attach={bool(attachment)} bytes={len(attachment or b'')}"
    data = message.encode()
    method = "POST"
    if attachment and filename:
        # PUT body = file; message/title as headers (ntfy attachment upload)
        headers["Filename"] = filename
        msg = message.replace("\n", " ").encode("utf-8", "replace")
        headers["Message"] = msg.decode("latin-1", "replace")[:4000]
        data = attachment
        method = "PUT"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        return "posted"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200]
        if attachment and exc.code in (413, 400, 403):
            # retry without attachment
            return ntfy_publish(cfg, title=title, message=message + f"\n\n(attach failed: HTTP {exc.code})",
                                priority=priority, tags=tags, dry_run=dry_run)
        return f"error HTTP {exc.code}: {body}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"error: {exc}"


def process_doc(
    doc: dict, cfg: AlertConfig, dedupe: Optional[DedupeStore], *, dry_run: bool = False,
) -> Optional[str]:
    if not cfg.enabled:
        return None
    alert = alert_from_doc(doc, cfg)
    if alert is None:
        return None
    if dedupe is not None:
        ok, alert = dedupe.accept(alert)
        if not ok or alert is None:
            return "deduped"
    clip, filename, _ = build_clip(alert, cfg)
    attached = clip is not None
    title, message, priority, tags = format_message(alert, cfg, attached=attached)
    return ntfy_publish(
        cfg, title=title, message=message, priority=priority, tags=tags,
        attachment=clip, filename=filename, dry_run=dry_run,
    )


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def dry_run_report(path: Path, cfg: AlertConfig) -> dict[str, Any]:
    counts: dict[str, int] = {}
    examples: list[dict] = []
    skipped_no_audio = 0
    skipped_hallucination = 0
    total_docs = 0
    for doc in iter_jsonl(path):
        total_docs += 1
        # count hallucinations separately for skyking-like broadcasts
        for b in doc.get("broadcasts") or []:
            _, _, wp = _word_times(doc, b)
            if _is_prompt_hallucination(doc, b, wp, cfg.min_word_prob):
                skipped_hallucination += 1
        matches = find_matches(doc, cfg)
        if not matches:
            continue
        audio = resolve_audio(doc, cfg.recordings_root)
        if audio is None:
            skipped_no_audio += 1
            continue
        for m in matches:
            counts[m.keyword] = counts.get(m.keyword, 0) + 1
        best = max(matches, key=lambda m: (m.confidence, m.word_prob))
        if len(examples) < 40:
            rec = doc.get("recording") or {}
            examples.append({
                "keyword": best.keyword,
                "matched": best.matched_text,
                "confidence": best.confidence,
                "word_prob": round(best.word_prob, 3),
                "khz": rec.get("frequency_khz"),
                "file": audio.name,
                "transcript": best.transcript[:160],
                "type": best.broadcast_type,
            })
    return {
        "docs": total_docs,
        "counts": counts,
        "alerts": sum(counts.values()),
        "skipped_no_audio": skipped_no_audio,
        "skipped_hallucination_broadcasts": skipped_hallucination,
        "examples": examples,
    }


def follow_jsonl(path: Path, cfg: AlertConfig, state: Path, dedupe: DedupeStore) -> None:
    state.parent.mkdir(parents=True, exist_ok=True)
    offset = int(state.read_text()) if state.exists() else path.stat().st_size
    pending = ""
    while True:
        if not path.exists():
            time.sleep(POLL_S)
            continue
        size = path.stat().st_size
        if size < offset:
            offset = 0
            pending = ""
        with path.open() as fh:
            fh.seek(offset)
            chunk = fh.read()
            offset = fh.tell()
        state.write_text(str(offset))
        if not chunk:
            time.sleep(POLL_S)
            continue
        pending += chunk
        if pending.endswith("\n"):
            complete, pending = pending, ""
        else:
            complete, sep, pending = pending.rpartition("\n")
            if not sep:
                continue
        for line in complete.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                print("bad json line, skipped", file=sys.stderr, flush=True)
                continue
            status = process_doc(doc, cfg, dedupe, dry_run=False)
            if status:
                khz = (doc.get("recording") or {}).get("frequency_khz")
                print(f"alert {status} {khz}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HFGCS keyword alerts → ntfy (with audio clip)")
    ap.add_argument("--config", type=Path, default=None, help="keyword-alerts.toml")
    ap.add_argument("--decodes", type=Path, default=None, help="decodes.jsonl")
    ap.add_argument("--recordings", type=Path, default=None, help="recordings root")
    ap.add_argument("--dry-run", action="store_true", help="scan decodes; print what would alert (no ntfy)")
    ap.add_argument("--watch", action="store_true", help="tail decodes.jsonl and push alerts")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--dedupe-state", type=Path, default=DEFAULT_DEDUPE)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.recordings:
        cfg.recordings_root = args.recordings.expanduser()
    decodes = args.decodes or Path(os.environ.get("HFGCS_DECODES", "decodes.jsonl"))
    if args.dry_run:
        if not cfg.recordings_root.exists():
            # try beside decodes
            maybe = decodes.resolve().parent / "recordings"
            if maybe.is_dir():
                cfg.recordings_root = maybe
            elif (decodes.resolve().parent).is_dir():
                cfg.recordings_root = decodes.resolve().parent
        report = dry_run_report(decodes, cfg)
        print(json.dumps(report, indent=2))
        print("\nSummary:", file=sys.stderr)
        print(f"  docs scanned: {report['docs']}", file=sys.stderr)
        print(f"  keyword hits: {report['counts']}", file=sys.stderr)
        skipped = report["skipped_hallucination_broadcasts"]
        print(f"  hallucination broadcasts skipped: {skipped}", file=sys.stderr)
        print(f"  matched but no audio file: {report['skipped_no_audio']}", file=sys.stderr)
        return 0
    if args.watch:
        if not cfg.ntfy_topic:
            print("NTFY_TOPIC is required for --watch", file=sys.stderr)
            return 2
        dedupe = DedupeStore(args.dedupe_state, cfg.dedupe_s)
        follow_jsonl(decodes, cfg, args.state, dedupe)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
