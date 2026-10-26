# Connect to an OpenWebRX+ waterfall and measure HFGCS USB channels.
# The local receiver is published at https://waterfall.scanplacer.com
# and also listens on http://127.0.0.1:8073.

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import websockets

# OpenWebRX IMA-ADPCM tables (htdocs/lib/AudioEngine.js). Each FFT frame
# resets the codec and discards COMPRESS_FFT_PAD_N leading samples.
IMA_INDEX = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]
IMA_STEP = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]
FFT_PAD = 10

# Names as shown by this receiver. OpenWebRX sends "sdr_id|profile_id".
HFGCS_PROFILES = [
    ("HFGCS 4724", 4_724_000),
    ("HFGCS 6739", 6_739_000),
    ("HFGCS 8992", 8_992_000),
    ("HFGCS 11175", 11_175_000),
    ("HFGCS 15016", 15_016_000),
]

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


def ws_url(http_url: str) -> str:
    url = http_url.rstrip("/")
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):] + "/ws/"
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):] + "/ws/"
    if url.startswith("wss://") or url.startswith("ws://"):
        return url if url.endswith("/ws/") else url.rstrip("/") + "/ws/"
    raise SystemExit(f"Unsupported waterfall URL: {http_url}")


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
                results.append({"name": name, "error": "profile not advertised"})
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
    if failed or not results:
        return 1
    print(f"Waterfall test collected {len(results)} HFGCS profiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
