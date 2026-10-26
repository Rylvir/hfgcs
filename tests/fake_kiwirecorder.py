"""Stand-in for kiwiclient's kiwirecorder.py in tests (never touches the network).

Behaviour is chosen by the server host name:
  busy.*      -> prints kiwiclient's too_busy error and exits 1
  dead.*      -> exits 1 without output (connect failure)
  anything    -> writes a --nc-wav stream: 12 kHz mono, 1 kHz tone, $FAKE_KIWI_SECONDS long
Every invocation's argv is appended to $FAKE_KIWI_LOG as one JSON line.
"""

import json
import os
import struct
import sys

import numpy as np

argv = sys.argv[1:]
with open(os.environ["FAKE_KIWI_LOG"], "a") as fh:
    fh.write(json.dumps(argv) + "\n")
host = argv[argv.index("-s") + 1]
if host.startswith("busy."):
    sys.stderr.write(f"ERROR: KiwiTooBusyError: {host}: all 4 client slots taken\n")
    sys.exit(1)
if host.startswith("dead."):
    sys.stderr.write("Failed to connect, giving up\n")
    sys.exit(1)

rate = 12000
out = sys.stdout.buffer
fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
out.write(b"RIFF" + struct.pack("<I", 0x7FFFFFF0) + b"WAVE")
out.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
out.write(b"data" + struct.pack("<I", 0x7FFFFFF0 - 36))
t = np.arange(int(float(os.environ.get("FAKE_KIWI_SECONDS", "1.0")) * rate)) / rate
pcm = (0.3 * 32767 * np.sin(2 * np.pi * 1000 * t)).astype("<i2")
for i in range(0, len(pcm), 600):
    out.write(pcm[i : i + 600].tobytes())
    out.flush()
