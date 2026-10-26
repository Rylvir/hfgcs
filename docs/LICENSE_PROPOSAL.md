# License proposal (decision pending)

The repository has **no LICENSE yet**, which means nobody else may legally
reuse, modify or redistribute the code. This note compares three options so
the owner can choose. No LICENSE file is added by this PR.

## Constraints

- We want to reuse ideas and possibly code from
  [Chipensaw/HFGCS-Monitor-KiwiSDR](https://github.com/Chipensaw/HFGCS-Monitor-KiwiSDR)
  (**Apache-2.0**). Copied Apache-2.0 code must keep its copyright and
  license notices, carry the upstream `NOTICE` content, and mark modified
  files as changed (Apache-2.0 §4).
- NEET INTEL repositories have **no license**: ideas and data formats only,
  no code, with credit. Nothing in this repo is copied from them.
- Likely future dependencies: kiwiclient (MIT), OpenWebRX (AGPL-3.0, only
  talked to over its websocket, not linked), gr-ale / LinuxALE (GPL),
  PC-ALE (MIT), faster-whisper (MIT).

## Comparison

| | MIT | Apache-2.0 | GPL-3.0 |
|---|---|---|---|
| Type | permissive | permissive + patent grant | strong copyleft |
| Can include Apache-2.0 code (Chipensaw) | yes, but the combined work must still honour Apache-2.0 for those files (notices, NOTICE) | **yes, same license, simplest** | yes (Apache-2.0 is GPL-3.0-compatible, one way) |
| Can include GPL code (gr-ale, LinuxALE) | no (only as a separate process/tool) | no (only as a separate process/tool) | yes |
| Others can reuse our code in closed tools | yes | yes | no |
| Explicit patent grant / retaliation | no | yes | yes |
| Notice burden | minimal | NOTICE file + change marks | source offer + license text |

## Recommendation

**Apache-2.0.** It matches the best reference project, so code from
HFGCS-Monitor-KiwiSDR can be reused under one license with one NOTICE file;
it stays permissive for community projects (e.g. NEET INTEL) to consume our
data tools; and it adds an explicit patent grant. GPL decoders (gr-ale,
LinuxALE) should be run as separate processes fed by our audio, which keeps
the core Apache-2.0. MIT is an acceptable alternative if the owner prefers
the shortest license; GPL-3.0 only makes sense if we decide to link GPL ALE
code directly.

If Apache-2.0 is chosen: add `LICENSE` (verbatim text), a `NOTICE` file, set
`license = "Apache-2.0"` in `pyproject.toml`, and add SPDX headers.
