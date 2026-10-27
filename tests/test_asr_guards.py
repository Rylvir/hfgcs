"""Coverage, loop and wall-time guards in decoding.asr (no Whisper needed: a
fake model stands in for faster-whisper)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from decoding.asr import AsrConfig, Transcriber
from decoding.coverage import (
    active_fraction,
    frame_rms,
    segment_is_loop,
    window_bounds,
    word_coverage,
    word_gaps,
)

RATE = 16000
PHON = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]


def _noise(seconds: float, quiet: tuple[float, float] | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(int(seconds * RATE))).astype("float32")
    if quiet:
        x[int(quiet[0] * RATE) : int(quiet[1] * RATE)] *= 0.01
    return x


def _seg(t0: float, texts: list[str], step: float = 0.8):
    words = [
        SimpleNamespace(word=" " + w, probability=0.9, start=t0 + i * step, end=t0 + i * step + 0.5)
        for i, w in enumerate(texts)
    ]
    return SimpleNamespace(
        start=t0,
        end=t0 + len(texts) * step,
        text=" ".join(texts),
        avg_logprob=-0.3,
        no_speech_prob=0.1,
        words=words,
    )


class FakeModel:
    """Every call: one 8-word segment every 10 s of the piece, but only up to
    ``stop_after`` seconds (a Whisper skip that drops the rest), and an
    optional runaway loop from ``loop_at`` seconds. Records each call."""

    def __init__(self, stop_after=None, loop_at=None, loop_only_without_vad=True, tick=None):
        self.stop_after, self.loop_at, self.loop_only_without_vad = stop_after, loop_at, loop_only_without_vad
        self.tick = tick or (lambda: None)
        self.calls = []

    def transcribe(self, piece, vad_filter=False, vad_parameters=None, **kw):
        dur = len(piece) / RATE
        self.calls.append(
            {"dur": round(dur, 2), "vad_filter": vad_filter, "cond": kw["condition_on_previous_text"]}
        )

        def gen():
            t = 0.0
            while t + 9 <= dur:
                if self.stop_after is not None and t >= self.stop_after and len(self.calls) == 1:
                    return
                self.tick()
                looping = self.loop_at is not None and t >= self.loop_at
                if looping and (vad_filter is False or not self.loop_only_without_vad):
                    yield _seg(t, ["Delta"] * 60, step=0.1)  # 60 words in 6 s
                else:
                    yield _seg(t, PHON)
                t += 10.0

        return gen(), None


def _tr(model, **over) -> Transcriber:
    tr = Transcriber(replace(AsrConfig(), **over))
    tr._model = model
    return tr


# ---------------------------------------------------------------- pure helpers
def test_word_coverage_and_gaps():
    words = [{"start": 0.0, "end": 1.0}, {"start": 0.5, "end": 2.0}, {"start": 50.0, "end": 51.0}]
    assert abs(word_coverage(words, 100.0) - 0.03) < 1e-9
    assert word_gaps(words, 100.0, 12.0) == [(2.0, 50.0), (51.0, 100.0)]
    assert word_gaps([], 10.0, 12.0) == []


def test_window_cuts_land_in_quiet_stretch():
    x = _noise(600.0, quiet=(290.0, 293.0))
    b = window_bounds(frame_rms(x), 600.0, 300.0, 20.0)
    assert len(b) == 2 and 290.0 <= b[0][1] <= 293.0 and b[1] == (b[0][1], 600.0)
    assert window_bounds(frame_rms(x), 600.0, 500.0, 20.0) == [(0.0, 600.0)]  # < 1.33 x window
    assert window_bounds([], 600.0, None, 20.0) == [(0.0, 600.0)]


def test_active_fraction_ignores_digital_silence():
    x = np.concatenate([_noise(5.0), np.zeros(5 * RATE, dtype="float32")])
    rms = frame_rms(x)
    assert active_fraction(rms, 0.0, 5.0) == 1.0 and active_fraction(rms, 5.0, 10.0) == 0.0


def test_segment_is_loop():
    ok = {"start": 0.0, "end": 8.0, "words": [{"word": w} for w in PHON]}
    dense = {"start": 0.0, "end": 2.0, "words": [{"word": w} for w in PHON * 2]}
    rep = {"start": 0.0, "end": 20.0, "words": [{"word": w} for w in ["Delta.", "X-ray.", "X-ray."] * 6]}
    assert not segment_is_loop(ok, 3.0)
    assert segment_is_loop(dense, 3.0) and segment_is_loop(rep, 3.0)


# ---------------------------------------------------------------- transcriber guards
def test_dropout_gap_is_redecoded():
    """First decode stops after 30 s of a 120 s clip (the eam-f27e2d58 shape):
    the coverage guard decodes the word-free rest on its own."""
    m = FakeModel(stop_after=30.0)
    r = _tr(m).transcribe_pcm(_noise(120.0))
    assert r["words"][-1]["end"] > 100.0
    assert r["guard"]["repairs"] and r["guard"]["repairs"][0]["words"] > 0
    assert len(m.calls) == 2 and m.calls[1]["cond"] is False
    starts = [w["start"] for w in r["words"]]
    assert starts == sorted(starts) and len(starts) == len(set(starts))  # merged in order, no duplicates
    for s in r["segments"]:  # word_span still indexes the merged list
        i, j = s["word_span"]
        assert r["words"][i]["start"] >= s["start"] - 0.01


def test_silent_gap_is_not_redecoded():
    m = FakeModel(stop_after=30.0)
    x = _noise(120.0)
    x[40 * RATE :] = 0.0
    r = _tr(m).transcribe_pcm(x)
    assert len(m.calls) == 1 and r["guard"]["repairs"] == []


def test_loop_resumes_with_fallback():
    """A runaway loop at 60 s: segments before it are kept, the looping
    segment is dropped, and the rest is decoded with the fallback (VAD on)."""
    m = FakeModel(loop_at=60.0)
    r = _tr(m, gap_repair_s=None).transcribe_pcm(_noise(120.0))
    fb = r["guard"]["fallbacks"]
    assert fb == [{"start": 0.0, "end": 120.0, "reason": "loop", "resume_at": 60.0, "fallback_status": "ok"}]
    assert m.calls[1] == {"dur": 60.0, "vad_filter": True, "cond": False}
    assert len(r["words"]) == 6 * 8 + 6 * 8  # 6 good segments + 6 from the fallback, no loop words
    assert all(len(s["text"].split()) == 8 for s in r["segments"])


def test_wall_time_cap(monkeypatch):
    """Each segment takes 20 s of (fake) wall time. A 60 s clip has a 2 x 60 =
    120 s budget; its first decode may use 0.6 of it (72 s), then the fallback
    gets the rest, and nothing runs past the budget."""
    now = [0.0]
    monkeypatch.setattr("decoding.asr.time.monotonic", lambda: now[0])
    m = FakeModel(tick=lambda: now.__setitem__(0, now[0] + 20.0))
    r = _tr(m).transcribe_pcm(_noise(60.0))
    fbs = r["guard"]["fallbacks"]
    assert fbs[0]["reason"] == "timeout" and 0 < fbs[0]["resume_at"] < 60.0
    assert m.calls[1]["vad_filter"] is True
    assert r["guard"]["wall_s"] <= 2 * 60 + 20  # budget + the segment in flight


def test_budget_spent_skips_fallback_and_repairs(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("decoding.asr.time.monotonic", lambda: now[0])
    m = FakeModel(stop_after=10.0, tick=lambda: now.__setitem__(0, now[0] + 200.0))
    r = _tr(m).transcribe_pcm(_noise(100.0))
    assert len(m.calls) == 1
    assert r["guard"]["fallbacks"][0]["fallback_status"].startswith("skipped")
    assert all(x.get("skipped") for x in r["guard"]["repairs"])


def test_long_audio_is_windowed():
    m = FakeModel()
    r = _tr(m, window_s=100.0, gap_repair_s=None).transcribe_pcm(_noise(400.0))
    assert len(r["guard"]["windows"]) == 4 and len(m.calls) == 4
    assert all(c["dur"] <= 130 for c in m.calls)
    assert r["words"][-1]["end"] > 380


def test_config_key_covers_guards_and_compute_type():
    base = AsrConfig()
    assert base.compute_type == "float32"
    keys = {
        replace(base, **o).key()
        for o in (
            {},
            {"compute_type": "int8"},
            {"window_s": None},
            {"gap_repair_s": None},
            {"max_rtf": None},
            {"seed": 1},
            {"fallback": {}},
        )  # fmt: skip
    }
    assert len(keys) == 7
    assert replace(base, cpu_threads=1).key() == base.key()  # float32 output does not depend on threads


# ---------------------------------------------------------------- committed real decode
FIXTURE = Path(__file__).parent / "fixtures" / "decode_sample_two_broadcasts.decode.json"


def test_fixture_covers_the_whole_recording():
    """The committed sample decode (two gold EAMs, 374 s) must reach both
    broadcasts and the end of the audio; a decode that drops a minute or more
    (the int8 failure: 31 words, one EAM) fails here."""
    doc = json.loads(FIXTURE.read_text())
    words, dur = doc["asr"]["words"], doc["duration_s"]
    assert [b["type"] for b in doc["broadcasts"]] == ["eam", "eam"]
    assert len(words) >= 150
    assert words[-1]["end"] > 0.9 * dur
    assert 0.2 <= word_coverage(words, dur) <= 0.7
    gaps = word_gaps(words, dur, 30.0)
    assert len(gaps) <= 1  # only the silence between the two clips
