# Third-party material in the decoding workstream

| Item | Where | License | Notes |
|---|---|---|---|
| missileer.watch archive feed rows | `tests/fixtures/missileer_archive_sample.json` (4 unmodified rows); eval set built locally by `decoding/ingest_missileer.py` (not committed) | CC0-1.0 (feed `license` field and site footer, verified 2026-10-26) | Attribution appreciated: missileer.watch. |
| brisi-jt/skywatch (https://github.com/brisi-jt/skywatch), Copyright (c) 2026 James Towns | Design reference only for `decoding/asr.py`: the lazy faster-whisper import, hotwords + `initial_prompt` callsign biasing and the two-model A/B evaluation harness | MIT | **No code copied** (checked 2026-10-26). Our wrapper already loads lazily, and skywatch's `verbalize.py` maps ICAO airline codes to radio names, which does not apply to HFGCS word callsigns. If code is copied later, keep its MIT notice in the file header and list the file here. |
| Priyom HFGCS page (https://priyom.org/military-stations/united-states/hfgcs) | Procedure quoted in `docs/decoding.md` and `decoding/eam_parser.py` docstrings | Website; short quotations with citation | Used for facts only. |
| faster-whisper (SYSTRAN), silero-vad (via faster-whisper), CTranslate2 | Optional runtime dependencies of `decoding/asr.py` | MIT | Not vendored. |
| Whisper `small.en` weights (OpenAI, via Systran/faster-whisper-small.en) | Downloaded at runtime | MIT | Not vendored. |
