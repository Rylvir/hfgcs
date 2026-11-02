"""Tests for hfgcs.keyword_alerts."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from hfgcs import keyword_alerts as ka
from hfgcs.deploy import plan_units, role_of, roles_for_path


def _wav(path: Path, seconds: float = 2.0, rate: int = 8000) -> Path:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)
    return path


def _doc(transcript: str, *, conf: float = 0.8, word_prob: float = 0.9, truncated: bool = False,
         start_s: float = 1.0, end_s: float = 3.0, dur: float = 10.0, khz: float = 11175.0,
         btype: str = "other", flags=None, file: str = "clip.wav") -> dict:
    words = [
        {"word": w, "start": start_s + i * 0.2, "end": start_s + i * 0.2 + 0.15, "probability": word_prob}
        for i, w in enumerate(transcript.split()[:20])
    ]
    return {
        "decoded_at": "2026-11-02T01:00:00Z",
        "duration_s": dur,
        "recording": {
            "file": file,
            "frequency_khz": khz,
            "receiver_id": "kiwi-test",
            "truncated": truncated,
            "start_utc": "2026-11-02T00:59:00Z",
        },
        "asr": {"words": words, "text": transcript},
        "broadcasts": [{
            "type": btype,
            "transcript": transcript,
            "confidence": conf,
            "start_s": start_s,
            "end_s": end_s,
            "asr_word_span": [0, max(0, len(words) - 1)],
            "flags": flags or [],
        }],
    }


def test_skyking_variant_match(tmp_path):
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path)
    _wav(tmp_path / "clip.wav")
    doc = _doc("Sky king, sky king, do not answer. Alpha time two five authentication Papa Hotel.",
               btype="skyking", conf=0.7)
    ms = ka.find_matches(doc, cfg)
    assert any(m.keyword == "skyking" for m in ms)


def test_doom_requires_context(tmp_path):
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path)
    _wav(tmp_path / "clip.wav")
    prose = _doc("The sense of doom hung over the valley.")
    assert ka.find_matches(prose, cfg) == []
    callsign = _doc("This is Doom one two, radio check.")
    assert any(m.keyword == "doom" for m in ka.find_matches(callsign, cfg))


def test_prompt_hallucination_dropped(tmp_path):
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path, min_word_prob=0.35)
    _wav(tmp_path / "clip.wav")
    doc = _doc(
        "Skyking, Skyking, do not answer. Skyking, Skyking, do not answer.",
        conf=0.0, word_prob=0.12, truncated=True, start_s=880, end_s=899, dur=900,
        btype="skyking",
        flags=["single_read", "missing_codeword", "missing_time", "missing_authentication", "truncated"],
    )
    assert ka.find_matches(doc, cfg) == []
    assert ka.alert_from_doc(doc, cfg) is None


def test_no_alert_without_wav(tmp_path):
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path)
    doc = _doc("Skyking, skyking, do not answer. Kingfisher time two five authentication Papa Hotel.",
               btype="skyking", conf=0.8, file="missing.wav")
    assert ka.find_matches(doc, cfg)  # text matches
    assert ka.alert_from_doc(doc, cfg) is None  # but no audio


def test_alert_with_clip(tmp_path):
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path, pad_s=0.5)
    _wav(tmp_path / "clip.wav", seconds=5.0)
    doc = _doc("Nightwatch zero one, radio check.", conf=0.9, start_s=1.0, end_s=2.5)
    alert = ka.alert_from_doc(doc, cfg)
    assert alert is not None and any(m.keyword == "nightwatch" for m in alert.matches)
    clip, name, hint = ka.build_clip(alert, cfg)
    assert clip and name.endswith((".mp3", ".wav")) and len(clip) > 100


def test_dedupe_simulcast(tmp_path):
    store = ka.DedupeStore(tmp_path / "dedupe.json", window_s=120)
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path)
    _wav(tmp_path / "a.wav")
    _wav(tmp_path / "b.wav")
    d1 = _doc("This is Zolar one one.", conf=0.8, file="a.wav", khz=11175)
    d2 = _doc("This is Zolar one one.", conf=0.6, file="b.wav", khz=8992)
    a1 = ka.alert_from_doc(d1, cfg)
    a2 = ka.alert_from_doc(d2, cfg)
    ok1, _ = store.accept(a1)
    ok2, _ = store.accept(a2)
    assert ok1 and not ok2


def test_dry_run_report(tmp_path):
    cfg = ka.AlertConfig(keywords=ka.default_keywords(), recordings_root=tmp_path)
    _wav(tmp_path / "clip.wav")
    path = tmp_path / "decodes.jsonl"
    docs = [
        _doc("Skyking, skyking, do not answer. Kingfisher time two five authentication Papa Hotel.",
             btype="skyking", conf=0.8),
        _doc("The sense of doom hung over the valley.", conf=0.9),
    ]
    path.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    report = ka.dry_run_report(path, cfg)
    assert report["counts"].get("skyking") == 1
    assert "doom" not in report["counts"]


def test_deploy_mapping():
    assert role_of("hfgcs-keyword-ntfy") == "alert"
    assert roles_for_path("hfgcs/keyword_alerts.py") == frozenset({"alert"})
    units = ["hfgcs-8992", "hfgcs-kiwi-11175", "hfgcs-decode", "hfgcs-keyword-ntfy", "hfgcs-review"]
    plans, _ = plan_units(["hfgcs/keyword_alerts.py"], units)
    assert [p.unit for p in plans] == ["hfgcs-keyword-ntfy"]
    plans, _ = plan_units(["deploy/systemd/hfgcs-keyword-ntfy.service"], units)
    assert [p.unit for p in plans] == ["hfgcs-keyword-ntfy"]
