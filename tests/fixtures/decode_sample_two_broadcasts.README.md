# decode_sample_two_broadcasts.decode.json

A real decode document (contract v1, `docs/decode.schema.json`) written by `python -m decoding.run` on
2026-10-26 with the default decoder of decoding 0.3.1 (`small.en` float32 on CPU, long-recording guards, vote3
second pass), from a fresh (uncached) decode at decoding 0.3.1. The 0.2 fixture came from int8 and is not reproducible.
`tests/test_asr_guards.py::test_fixture_covers_the_whole_recording` checks its coverage.

The input was synthetic: two gold eval clips from the missileer.watch feed (CC0-1.0, attribution appreciated:
missileer.watch), joined into one 8 kHz PCM_16 WAV the way the Phase 0 recorder writes them:

    eam-356f1a01.opus (EAM, ends "This is Coupling, out") + 2 s silence + eam-86d80b13.opus (EAM, "All stations ...")

The audio is not committed. It lives outside the repo at
`/workspace/hfgcs-data/eval/samples/HFGCS_11175kHz_20260923T173253Z_eval-synthetic.wav`, with its schema-2
catalog line in `catalog.jsonl` there. To regenerate the fixture:

    cd /workspace/hfgcs-data/eval/samples
    PYTHONPATH=<repo> /workspace/.venv-asr/bin/python -m decoding.run catalog.jsonl --asr-cache $(mktemp -d) --force
    cp HFGCS_11175kHz_20260923T173253Z_eval-synthetic.wav.decode.json <repo>/tests/fixtures/decode_sample_two_broadcasts.decode.json

The catalog times are invented (the first clip's start time), and its SNR fields are null.
