# Third-party material

| Item | Where | License | Notes |
|---|---|---|---|
| missileer.watch archive feed rows | `tests/fixtures/missileer_archive_sample.json` (4 unmodified rows); eval set built locally by `decoding/ingest_missileer.py` (not committed) | CC0-1.0 (feed `license` field and site footer, verified 2026-10-26) | Attribution appreciated: missileer.watch. |
| brisi-jt/skywatch (https://github.com/brisi-jt/skywatch), Copyright (c) 2026 James Towns | Design reference only for `decoding/asr.py`: the lazy faster-whisper import, hotwords + `initial_prompt` callsign biasing and the two-model A/B evaluation harness | MIT | **No code copied** (checked 2026-10-26). Our wrapper already loads lazily, and skywatch's `verbalize.py` maps ICAO airline codes to radio names, which does not apply to HFGCS word callsigns. If code is copied later, keep its MIT notice in the file header and list the file here. |
| Priyom HFGCS page (https://priyom.org/military-stations/united-states/hfgcs) | Procedure quoted in `docs/decoding.md` and `decoding/eam_parser.py` docstrings | Website; short quotations with citation | Used for facts only. |
| faster-whisper (SYSTRAN), silero-vad (via faster-whisper), CTranslate2 | Optional runtime dependencies of `decoding/asr.py` | MIT | Not vendored. |
| Whisper `small.en` weights (OpenAI, via Systran/faster-whisper-small.en) | Downloaded at runtime | MIT | Not vendored. |

## Audio sources (`hfgcs/sources/`)

| Item | Where | License | Notes |
|---|---|---|---|
| OpenWebRX (jketterl/openwebrx; the server at waterfall.scanplacer.com runs the OpenWebRX+ fork, assumed protocol-compatible) | Protocol reference for `hfgcs/sources/openwebrx.py` (websocket handshake, `connectionproperties`, `dspcontrol`, binary frame types, sync-framed audio ADPCM) | AGPL-3.0 | **Read for protocol understanding only; no code copied** (checked 2026-10-26). |
| csdr (jketterl/csdr) `adpcm.cpp` | Reference for the audio ADPCM sync framing (`SYNC`, step index, predictor, ~1000 codec bytes) | BSD-style (Retzler/Ketterl; Kientzle; CWI) | No code copied. The IMA ADPCM algorithm itself is the public Intel/DVI standard; our encoder/decoder are written from that spec. |
| kiwiclient (jks-prv/kiwiclient) `kiwirecorder.py` | Run as an external program by `hfgcs/sources/kiwisdr.py` | No top-level LICENSE (GitHub detects none); `kiwi/rigctld.py` is GPL-2.0-or-later; bundled `mod_pywebsocket` is BSD | **Not vendored, not imported.** Users install it themselves; we only exec it and read its stdout WAV stream. |
| Chipensaw/HFGCS-Monitor-KiwiSDR `ETIQUETTE.md` | KiwiSDR etiquette rules adapted in `hfgcs/sources/kiwisdr.py` and `docs/sources.md` | Apache-2.0 | Rules re-implemented in our own words and code; no code copied, so no NOTICE file is required. Credited here and in `docs/sources.md`. |
