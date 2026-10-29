"""Stand-in for kiwiclient's kiwirecorder.py in tests (never touches the network).

Behaviour is chosen by the server host name:
  busy.*      -> prints kiwiclient's too_busy error and exits 1
  dead.*      -> exits 1 without output (connect failure)
  stall.*     -> writes the WAV header, then no samples (stays connected for 30 s)
  zero.*      -> writes all-zero samples (a silent receiver)
  kick.*      -> streams, then prints kiwiclient v1.11's in-process reconnect line
                 ("<host>:<port> server closed the connection cleanly. Reconnecting after 5 seconds")
                 and, like kiwirecorder, reconnects after $FAKE_KIWI_RECONNECT_S (default 5) s,
                 logging ["RECONNECT", host] to $FAKE_KIWI_LOG
  stubborn.*  -> like kick.*, but ignores SIGINT (so only SIGTERM stops it)
  kickexit.*  -> streams, prints the "server closed the connection" line, then exits 0
  redirect.*  -> streams, prints kiwiclient's redirect warning and keeps going (as if connected elsewhere)
  anything    -> writes a --nc-wav stream: 12 kHz mono, 1 kHz tone, $FAKE_KIWI_SECONDS long
With --progress, every block written is followed on stderr by kiwiclient v1.11's
"\rBlock: %08x, RSSI: %6.1f " line, the value from $FAKE_KIWI_RSSI (default -110.0).
Every invocation's argv is appended to $FAKE_KIWI_LOG as one JSON line.
"""

import json
import os
import signal
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
if host.startswith("stubborn."):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if host.startswith("dead."):
    sys.stderr.write("Failed to connect, giving up\n")
    sys.exit(1)

import time  # noqa: E402

rate = int(os.environ.get("FAKE_KIWI_RATE", "12000"))  # real kiwirecorder writes e.g. 11999
out = sys.stdout.buffer
fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
out.write(b"RIFF" + struct.pack("<I", 0x7FFFFFF0) + b"WAVE")
out.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
out.write(b"data" + struct.pack("<I", 0x7FFFFFF0 - 36))
if host.startswith("stall."):
    out.flush()
    time.sleep(30)
    sys.exit(0)
t = np.arange(int(float(os.environ.get("FAKE_KIWI_SECONDS", "1.0")) * rate)) / rate
pcm = (0.3 * 32767 * np.sin(2 * np.pi * 1000 * t)).astype("<i2")
if host.startswith("zero."):
    pcm[:] = 0
progress = "--progress" in argv
rssi = float(os.environ.get("FAKE_KIWI_RSSI", "-110.0"))
for i in range(0, len(pcm), 600):
    out.write(pcm[i : i + 600].tobytes())
    out.flush()
    if progress:
        sys.stderr.write("\rBlock: %08x, RSSI: %6.1f " % (i // 600, rssi))
        sys.stderr.flush()

port = argv[argv.index("-p") + 1]
if host.split(".")[0] in ("kick", "stubborn", "kickexit"):
    msg = " Reconnecting after 5 seconds" if host.split(".")[0] != "kickexit" else ""
    time.sleep(0.5)  # mid-session, after the stream has started
    sys.stderr.write(f"{host}:{port} server closed the connection cleanly.{msg}\n")
    sys.stderr.flush()
    if host.startswith("kickexit."):
        sys.exit(0)
    time.sleep(float(os.environ.get("FAKE_KIWI_RECONNECT_S", "5")))
    with open(os.environ["FAKE_KIWI_LOG"], "a") as fh:
        fh.write(json.dumps(["RECONNECT", host]) + "\n")
    out.write(pcm.tobytes())
    out.flush()
    time.sleep(30)
elif host.startswith("redirect."):
    sys.stderr.write(f"{host}:{port} Too busy now. Redirecting to other.example:8073\n")
    sys.stderr.flush()
    for _ in range(30):
        out.write(pcm[:1200].tobytes())
        out.flush()
        time.sleep(1)
