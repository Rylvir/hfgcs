"""Repeatable ASR + parser evaluation on the eval set (docs/decoding.md section 3).

    python -m decoding.evaluate --labels /workspace/hfgcs-data/eval/labels.jsonl \\
        --subset gold --config baseline --config best

ASR output is cached per (config key, clip) under ``--cache``, so parser /
post-processing changes re-score in seconds. Scoring (``score_*``) is pure
Python and unit-tested; only ``transcribe_cached`` needs faster-whisper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .eam_parser import parse_transcript
from .snap import snap_words

# ---------------------------------------------------------------- configs
# Each entry: AsrConfig overrides + post-processing switches.
_VAD_DEFAULT = {
    "threshold": 0.5,
    "min_speech_duration_ms": 0,
    "min_silence_duration_ms": 2000,
    "speech_pad_ms": 400,
}
_NOSKIP = {"log_prob_threshold": None, "no_speech_threshold": None, "compression_ratio_threshold": None}

CONFIGS: dict[str, dict] = {
    "baseline": {},  # v0.1 defaults from the first smoke test
    # prompt / hotwords
    "no_prompt": {"prompt": None},
    "no_prompt_no_hot": {"prompt": None, "hotwords": False},
    "phonetic_prompt": {"prompt": "phonetic"},
    "phonetic_prompt_no_hot": {"prompt": "phonetic", "hotwords": False},
    # VAD
    "no_vad": {"vad_filter": False},
    "vad_fw_default": {"vad_parameters": _VAD_DEFAULT},
    "vad_t25": {"vad_parameters": {**_VAD_DEFAULT, "threshold": 0.25, "min_silence_duration_ms": 1500}},
    "vad_t10_pad": {"vad_parameters": {**_VAD_DEFAULT, "threshold": 0.1, "speech_pad_ms": 800}},
    "vad_t50_s500": {"vad_parameters": {**_VAD_DEFAULT, "min_silence_duration_ms": 500}},
    # hallucination / skip guards
    "noskip": _NOSKIP,
    "no_hst": {"hallucination_silence_threshold": None},
    "cond_prev": {"condition_on_previous_text": True},
    "rep_pen": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 0},
    # preprocessing
    "pre_none": {"preprocess": "none"},
    "pre_bp": {"preprocess": "bp"},
    "pre_denoise": {"preprocess": "bp_denoise_norm"},
    "pre_wide": {"preprocess": "wide_norm"},
    "pre_loudnorm": {"preprocess": "bp_loudnorm"},
    # chunked decoding (our own VAD windows, decoded independently)
    "chunk8": {"chunk_s": 8.0},
    "chunk5": {"chunk_s": 5.0},
    "chunk12": {"chunk_s": 12.0},
    "chunk8_prompt": {"chunk_s": 8.0, "chunk_prompt": True, "prompt": "phonetic"},
    "chunk8_noskip": {"chunk_s": 8.0, **_NOSKIP},
    "no_vad_noskip": {"vad_filter": False, **_NOSKIP},
    "chunk8_novad": {"chunk_s": 8.0, "vad_filter": False},
    "chunk5_novad": {"chunk_s": 5.0, "vad_filter": False},
    "chunk15_novad": {"chunk_s": 15.0, "vad_filter": False},
    "chunk8_novad_noskip": {"chunk_s": 8.0, "vad_filter": False, **_NOSKIP},
    # post-processing only (same ASR cache as the config without post_)
    "baseline_snap": {"post_snap": True},
    "no_vad_snap": {"vad_filter": False, "post_snap": True},
    # no-VAD family (silero drops weak HF speech; see results table)
    "nv_no_prompt": {"vad_filter": False, "prompt": None},
    "nv_phonetic": {"vad_filter": False, "prompt": "phonetic"},
    "nv_no_hot": {"vad_filter": False, "hotwords": False},
    "nv_pre_none": {"vad_filter": False, "preprocess": "none"},
    "nv_pre_bp": {"vad_filter": False, "preprocess": "bp"},
    "nv_pre_denoise": {"vad_filter": False, "preprocess": "bp_denoise_norm"},
    "nv_pre_wide": {"vad_filter": False, "preprocess": "wide_norm"},
    "nv_cond_prev": {"vad_filter": False, "condition_on_previous_text": True},
    "nv_no_hst": {"vad_filter": False, "hallucination_silence_threshold": None},
    # closed vocabulary
    "vocab": {"vocab_constrain": True},
    "vocab_novad": {"vocab_constrain": True, "vad_filter": False},
    "vocab_chunk8_novad": {"vocab_constrain": True, "vad_filter": False, "chunk_s": 8.0},
}


def masked_edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein where '.' in the reference (unheard by the labeller)
    matches any hypothesis character at zero cost."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            sub = 0 if (r == h or r == ".") else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def best_eam(records: list[dict]) -> Optional[dict]:
    eams = [r for r in records if r.get("type") == "eam" and r.get("body")]
    return max(eams, key=lambda r: len(r["body"])) if eams else None


def score_clip(label: dict, records: list[dict]) -> dict:
    lab = label["labels"]
    rec = best_eam(records)
    out = {"id": label["id"], "type": label["type"], "status": label["label_status"]}
    if label["type"] == "eam" and lab.get("body"):
        ref = lab["body"]
        hyp = rec["body"] if rec else ""
        out.update(
            ref=ref,
            hyp=hyp,
            edits=masked_edit_distance(ref, hyp),
            ref_len=len(ref),
            detected=bool(rec),
            preamble_ref=lab.get("preamble"),
            preamble_hyp=(rec.get("preamble") or hyp[:6]) if rec else None,
            length_ref=lab.get("declared_length"),
            length_hyp=rec.get("declared_length") if rec else None,
            flags=rec.get("flags") if rec else [],
        )
        out["cer"] = out["edits"] / out["ref_len"]
        pr = lab.get("preamble")
        out["preamble_em"] = None if not pr or "." in pr else (out["preamble_hyp"] == pr)
        out["length_em"] = (
            None if lab.get("declared_length") is None else (out["length_hyp"] == out["length_ref"])
        )
    else:
        out["false_eam"] = bool(rec and len(rec["body"]) >= 10)
    return out


def aggregate(rows: list[dict]) -> dict:
    e = [r for r in rows if "cer" in r]
    neg = [r for r in rows if "false_eam" in r]

    def rate(key):
        vals = [r[key] for r in e if r.get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    edits = sum(r["edits"] for r in e)
    chars = sum(r["ref_len"] for r in e)
    pre, npre = rate("preamble_em")
    ln, nln = rate("length_em")
    return {
        "n_eam": len(e),
        "cer_micro": edits / chars if chars else None,
        "cer_median": sorted(r["cer"] for r in e)[len(e) // 2] if e else None,
        "preamble_em": pre,
        "n_preamble": npre,
        "length_em": ln,
        "n_length": nln,
        "detected": sum(r["detected"] for r in e) / len(e) if e else None,
        "n_neg": len(neg),
        "false_eam": sum(r["false_eam"] for r in neg),
    }


def baseline_machine(labels: list[dict]) -> dict:
    """missileer's own machine copy scored the same way (competitor baseline)."""
    rows = []
    for lab in labels:
        if lab["type"] != "eam" or not lab["labels"].get("body"):
            continue
        hyp = (lab.get("reference_machine") or {}).get("text") or ""
        ref = lab["labels"]["body"]
        rows.append({"edits": masked_edit_distance(ref, hyp), "ref_len": len(ref)})
    chars = sum(r["ref_len"] for r in rows)
    return {"cer_micro": sum(r["edits"] for r in rows) / chars if chars else None, "n": len(rows)}


# ---------------------------------------------------------------- subsets
def load_subset(path: Path, subset: str) -> list[dict]:
    labels = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    labels = [lab for lab in labels if lab.get("audio_path")]
    if subset == "gold":
        return [lab for lab in labels if lab["label_status"] == "gold"]
    if subset == "silver":
        # Silver = machine copy: only complete, fully readable two-read EAMs.
        return [
            lab
            for lab in labels
            if lab["label_status"] == "silver"
            and lab["type"] == "eam"
            and lab["labels"].get("body")
            and "." not in lab["labels"]["body"]
            and lab.get("has_both_reads")
        ]
    raise ValueError(subset)


# ---------------------------------------------------------------- running
def split_config(overrides: dict):
    from .asr import AsrConfig

    asr_over = {k: v for k, v in overrides.items() if not k.startswith("post_")}
    post = {k[5:]: v for k, v in overrides.items() if k.startswith("post_")}
    return replace(AsrConfig(), **asr_over), post


def transcribe_cached(transcriber, label: dict, cache: Path) -> dict:
    key = transcriber.cfg.key()
    f = cache / key / (label["source_id"] + ".json")
    if f.exists():
        return json.loads(f.read_text())
    f.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = transcriber.transcribe(Path(label["audio_path"]))
    r["asr_seconds"] = round(time.time() - t0, 1)
    f.write_text(json.dumps(r))
    return r


def postprocess(words: list[dict], post: dict) -> list[dict]:
    if post.get("snap"):
        words = snap_words(words)
    return words


def evaluate(labels: list[dict], name: str, overrides: dict, cache: Path, cached_only: bool = False) -> dict:
    from .asr import Transcriber

    cfg, post = split_config(overrides)
    tr = Transcriber(cfg)
    rows, asr_s, audio_s = [], 0.0, 0.0
    missing = 0
    for lab in labels:
        if cached_only and not (cache / cfg.key() / (lab["source_id"] + ".json")).exists():
            missing += 1
            continue
        r = transcribe_cached(tr, lab, cache)
        asr_s += r.get("asr_seconds", 0)
        audio_s += r.get("duration_s", 0)
        rows.append(score_clip(lab, parse_transcript(postprocess(r["words"], post))))
    agg = aggregate(rows)
    agg.update(config=name, asr_key=cfg.key(), rtf=(asr_s / audio_s) if audio_s else None, missing=missing)
    return {"summary": agg, "rows": rows}


def fmt_row(s: dict) -> str:
    def p(v):
        return "–" if v is None else f"{v:.3f}"

    return (
        f"| {s['config']} | {s['n_eam']} | {p(s['cer_micro'])} | {p(s['cer_median'])} | "
        f"{p(s['preamble_em'])} ({s['n_preamble']}) | {p(s['length_em'])} ({s['n_length']}) | "
        f"{p(s['detected'])} | {s['false_eam']}/{s['n_neg']} | {p(s['rtf'])} |"
    )


HEADER = (
    "| config | EAMs | body CER (micro) | CER median | preamble EM | length EM | detected | "
    "false EAM on negatives | RTF |\n|---|---|---|---|---|---|---|---|---|"
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate ASR + parser on the HFGCS eval set")
    ap.add_argument("--labels", type=Path, default=Path("/workspace/hfgcs-data/eval/labels.jsonl"))
    ap.add_argument("--subset", choices=["gold", "silver"], default="gold")
    ap.add_argument("--config", action="append", default=None, help=f"one of {sorted(CONFIGS)}")
    ap.add_argument("--cache", type=Path, default=Path("/workspace/hfgcs-data/eval/cache"))
    ap.add_argument("--out", type=Path, default=None, help="write per-clip JSON results here")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cached-only", action="store_true", help="score only clips already in the cache")
    args = ap.parse_args(argv)
    labels = load_subset(args.labels, args.subset)[: args.limit]
    m = baseline_machine(labels)
    print(f"subset={args.subset} clips={len(labels)} missileer machine CER={m['cer_micro']} (n={m['n']})")
    print(HEADER)
    results = {}
    for name in args.config or ["baseline"]:
        res = evaluate(labels, name, CONFIGS[name], args.cache, args.cached_only)
        if res["summary"]["missing"]:
            print(f"<!-- {name}: {res['summary']['missing']} clips not cached, skipped -->")
        results[name] = res
        print(fmt_row(res["summary"]), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
