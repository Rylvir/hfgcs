# Connect to an OpenWebRX+ waterfall and measure HFGCS USB channels.
# The local receiver is published at https://waterfall.scanplacer.com
# and also listens on http://127.0.0.1:8073.

import argparse
import asyncio
import json
import sys
import time

import numpy as np

from hfgcs.config import HFGCS_CHANNELS
from hfgcs.sources.openwebrx import IMA_INDEX, IMA_STEP, ws_url

# IMA-ADPCM tables and the URL helper are shared with the audio source.
# Each FFT frame resets the codec and discards COMPRESS_FFT_PAD_N leading samples.
FFT_PAD = 10

# Names as shown by this receiver. OpenWebRX sends "sdr_id|profile_id".
# The channel list comes from the shared config (all six HFGCS channels).
HFGCS_PROFILES = [(ch.profile_name, int(ch.dial_hz)) for ch in HFGCS_CHANNELS]

# OpenWebRX bans clients that switch profiles faster than about 10 seconds.
PROFILE_GAP_SECONDS = 11.0
DWELL_SECONDS = 2.5


class ImaAdpcm:
    def reset(self):
        self.step_index = 0
        self.predictor = 0
        self.step = 0

    def decode(self, data: bytes) -> np.ndarray:
        self.reset()
        out = np.empty(len(data) * 2, np.int16)
        j = 0
        for byte in data:
            out[j] = self._nibble(byte & 0x0F)
            out[j + 1] = self._nibble((byte >> 4) & 0x0F)
            j += 2
        return out

    def _nibble(self, nibble: int) -> int:
        self.step_index = min(88, max(0, self.step_index + IMA_INDEX[nibble]))
        diff = self.step >> 3
        if nibble & 1:
            diff += self.step >> 2
        if nibble & 2:
            diff += self.step >> 1
        if nibble & 4:
            diff += self.step
        if nibble & 8:
            diff = -diff
        self.predictor = min(32767, max(-32768, self.predictor + diff))
        self.step = IMA_STEP[self.step_index]
        return self.predictor


def fft_to_db(payload: bytes, codec: ImaAdpcm) -> np.ndarray:
    decoded = codec.decode(payload).astype(np.float32)
    if len(decoded) <= FFT_PAD:
        return np.array([], np.float32)
    return decoded[FFT_PAD:] / 100.0


def slice_stats(frame: np.ndarray, center_hz: float, samp_rate: float, dial_hz: float):
    n = len(frame)
    start = center_hz - samp_rate / 2.0

    def bins(f0, f1):
        b0 = int((f0 - start) / samp_rate * n)
        b1 = int((f1 - start) / samp_rate * n)
        return max(0, min(n, b0)), max(0, min(n, b1))

    usb0, usb1 = bins(dial_hz + 150, dial_hz + 2750)
    usb = frame[usb0:usb1] if usb1 > usb0 else frame[:0]
    peak_i = int(np.argmax(frame))
    peak_hz = start + (peak_i + 0.5) * samp_rate / n
    return {
        "noise_db": float(np.median(frame)),
        "usb_db": float(np.median(usb)) if len(usb) else float("nan"),
        "usb_peak_db": float(np.max(usb)) if len(usb) else float("nan"),
        "peak_db": float(frame[peak_i]),
        "peak_khz": peak_hz / 1000.0,
        "usb_bins": int(len(usb)),
    }


async def _recv_until(ws, deadline, state):
    while time.time() < deadline:
        timeout = max(0.1, deadline - time.time())
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return
        _take(message, state)


def _take(message, state):
    if isinstance(message, str):
        if message.startswith("CLIENT DE SERVER"):
            state["handshake"] = message
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        kind = payload.get("type")
        if kind == "config":
            state["config"].update(payload.get("value") or {})
        elif kind == "profiles":
            state["profiles"] = payload.get("value") or []
        elif kind == "smeter":
            state["smeter"].append(float(payload.get("value")))
        elif kind in ("sdr_error", "backoff", "log_message", "demodulator_error"):
            state["notes"].append(f"{kind}: {payload.get('value') or payload.get('reason')}")
        return
    if not message:
        return
    kind = message[0]
    if kind == 1 and state["config"].get("fft_compression") == "adpcm":
        frame = fft_to_db(message[1:], state["codec"])
        if len(frame):
            state["frames"].append(frame)
    elif kind == 1 and state["config"].get("fft_compression") == "none":
        frame = np.frombuffer(message[1:], dtype=np.float32)
        if len(frame):
            state["frames"].append(frame.copy())


async def survey(http_url: str):
    import websockets  # optional extra: pip install .[openwebrx]

    url = ws_url(http_url)
    origin = http_url if "://" in http_url else "http://127.0.0.1:8073"
    print(f"Connecting to {url}")
    # OpenWebRX does not answer WebSocket ping frames, so the client keepalive
    # closes a healthy waterfall after one ping interval.
    async with websockets.connect(
        url,
        origin=origin,
        max_size=8_000_000,
        open_timeout=15,
        ping_interval=None,
        ping_timeout=None,
    ) as ws:
        await ws.send("SERVER DE CLIENT client=hfgcs-waterfall type=receiver")
        state = {
            "handshake": "",
            "config": {},
            "profiles": [],
            "frames": [],
            "smeter": [],
            "notes": [],
            "codec": ImaAdpcm(),
        }
        await _recv_until(ws, time.time() + 5, state)
        if not state["handshake"]:
            raise RuntimeError("OpenWebRX did not acknowledge the websocket")
        print(state["handshake"])
        if not state["profiles"]:
            await _recv_until(ws, time.time() + 5, state)
        by_suffix = {}
        for item in state["profiles"]:
            by_suffix.setdefault(item.get("name", ""), item.get("id"))
        print(f"Profiles advertised: {len(state['profiles'])}")

        results = []
        last_switch = time.time()
        for name, dial_hz in HFGCS_PROFILES:
            profile_id = next(
                (pid for label, pid in by_suffix.items() if label.endswith(name) and pid),
                None,
            )
            if profile_id is None:
                # Not fatal: a receiver need not carry every channel.
                results.append({"name": name, "skipped": "profile not advertised"})
                print(f"{name}: profile not advertised")
                continue
            gap_deadline = last_switch + PROFILE_GAP_SECONDS
            if gap_deadline > time.time():
                print(f"Waiting {gap_deadline - time.time():.0f}s before switching to {name}")
                await _recv_until(ws, gap_deadline, state)
            state["frames"].clear()
            state["smeter"].clear()
            state["config"].pop("center_freq", None)
            await ws.send(json.dumps({
                "type": "selectprofile",
                "params": {"profile": profile_id},
            }))
            last_switch = time.time()
            await _recv_until(ws, time.time() + 8, state)
            await ws.send(json.dumps({"type": "dspcontrol", "action": "start"}))
            await ws.send(json.dumps({
                "type": "dspcontrol",
                "params": {
                    "mod": "usb",
                    "low_cut": 150,
                    "high_cut": 2750,
                    "offset_freq": dial_hz - float(state["config"].get("center_freq") or dial_hz),
                    "squelch_level": -150,
                },
            }))
            await _recv_until(ws, time.time() + DWELL_SECONDS, state)
            center = float(state["config"].get("center_freq") or 0)
            samp = float(state["config"].get("samp_rate") or 0)
            if not state["frames"] or samp <= 0 or center <= 0:
                results.append({
                    "name": name,
                    "error": "no waterfall frames",
                    "center_hz": center,
                    "notes": list(state["notes"]),
                })
                print(f"{name}: no waterfall frames (center={center}, notes={state['notes'][-3:]})")
                continue
            stacked = np.median(np.stack(state["frames"][-12:]), axis=0)
            stats = slice_stats(stacked, center, samp, dial_hz)
            stats["name"] = name
            stats["center_khz"] = center / 1000.0
            stats["frames"] = len(state["frames"])
            if state["smeter"]:
                stats["smeter_db"] = float(np.median(state["smeter"]))
            results.append(stats)
            sm = f"  smeter {stats['smeter_db']:6.1f}" if "smeter_db" in stats else ""
            print(
                f"{name:12}  center {stats['center_khz']:8.1f} kHz  "
                f"noise {stats['noise_db']:6.1f}  USB {stats['usb_db']:6.1f}  "
                f"USB peak {stats['usb_peak_db']:6.1f}  "
                f"band peak {stats['peak_db']:6.1f} at {stats['peak_khz']:.1f} kHz  "
                f"frames {stats['frames']}{sm}"
            )
        if state["notes"]:
            print("Server notes:")
            for note in state["notes"][-8:]:
                print(f"  {note}")
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure HFGCS channels on an OpenWebRX waterfall")
    parser.add_argument(
        "--url",
        default="https://waterfall.scanplacer.com",
        help="OpenWebRX base URL (default: the public waterfall)",
    )
    args = parser.parse_args(argv)
    try:
        results = asyncio.run(survey(args.url))
    except Exception as exc:
        print(f"Waterfall test failed: {exc}", file=sys.stderr)
        return 1
    failed = [row for row in results if row.get("error")]
    measured = [row for row in results if not row.get("error") and not row.get("skipped")]
    if failed or not measured:
        return 1
    print(f"Waterfall test collected {len(measured)} HFGCS profiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
