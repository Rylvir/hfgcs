"""A tiny local stand-in for an OpenWebRX server (no network beyond 127.0.0.1).

It speaks just enough of the websocket protocol for the source tests: the
handshake, config/profiles/smeter messages, dspcontrol/selectprofile handling
and type-2 audio frames (raw int16 or sync-framed ADPCM).
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

from hfgcs.sources.openwebrx import ImaAdpcmCodec, encode_sync_adpcm

PROFILES = {
    "airspy|hfgcs-11175": {"center_freq": 11_175_000, "samp_rate": 768_000, "name": "HFGCS 11175"},
    "airspy|hfgcs-13200": {"center_freq": 13_200_000, "samp_rate": 768_000, "name": "HFGCS 13200"},
    "airspy|hfgcs-15016": {"center_freq": 15_016_000, "samp_rate": 768_000, "name": "HFGCS 15016"},
    "airspy|wide-11000": {"center_freq": 11_000_000, "samp_rate": 768_000, "name": "Wide 11000"},
}


def tone_pcm(freq_hz, seconds, rate=12_000, amp=0.3, phase=0.0):
    t = np.arange(int(round(seconds * rate))) / rate
    return (amp * 32767 * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.int16)


class MockOpenWebRX:
    """Run with ``with MockOpenWebRX(...) as srv: srv.url``."""

    def __init__(
        self,
        profile="airspy|hfgcs-11175",
        compression="adpcm",
        audio=None,  # callable(conn_index) -> list of (pcm int16, smeter linear or None)
        events=None,  # {(conn_index, frame_index): ("config", {...}) | ("close", None)}
        frame_samples=240,
    ):
        self.profile = profile
        self.compression = compression
        self.audio = audio or (lambda conn: [(tone_pcm(1000, 2.0), 1e-3)])
        self.events = events or {}
        self.frame_samples = frame_samples
        self.received: list[list] = []  # per connection: text messages received
        self.headers: list[dict] = []
        self.connections = 0
        self._server = None

    # -- server plumbing -----------------------------------------------------------
    def __enter__(self):
        from websockets.sync.server import serve

        self._server = serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._thread.join(5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def config_msg(self, profile=None):
        profile = profile or self.profile
        p = PROFILES[profile]
        sdr, pid = profile.split("|")
        return json.dumps({
            "type": "config",
            "value": {
                "center_freq": p["center_freq"],
                "samp_rate": p["samp_rate"],
                "sdr_id": sdr,
                "profile_id": pid,
                "audio_compression": self.compression,
                "fft_compression": "adpcm",
                "start_mod": "usb",
            },
        })

    def all_sent(self) -> list[dict]:
        return [json.loads(m) for conn in self.received for m in conn if m.startswith("{")]

    # -- one client --------------------------------------------------------------------
    def _handler(self, ws):
        conn = self.connections
        self.connections += 1
        log: list = []
        self.received.append(log)
        try:
            self.headers.append(dict(ws.request.headers))
        except Exception:
            self.headers.append({})
        streaming = False
        frames = []
        codec, counter = ImaAdpcmCodec(), 0
        for pcm, smeter in self.audio(conn):
            for i in range(0, len(pcm), self.frame_samples):
                frames.append((pcm[i : i + self.frame_samples], smeter))
        idx = 0
        try:
            while True:
                try:
                    msg = ws.recv(timeout=0.002 if streaming else 0.05)
                except TimeoutError:
                    msg = None
                if isinstance(msg, str):
                    log.append(msg)
                    if msg.startswith("SERVER DE CLIENT"):
                        ws.send("CLIENT DE SERVER server=openwebrx version=v1.2.126-mock")
                        ws.send(json.dumps({
                            "type": "profiles",
                            "value": [{"id": k, "name": v["name"]} for k, v in PROFILES.items()],
                        }))
                        ws.send(self.config_msg())
                    else:
                        data = json.loads(msg)
                        if data.get("type") == "dspcontrol" and data.get("action") == "start":
                            streaming = True
                        elif data.get("type") == "selectprofile":
                            self.profile = data["params"]["profile"]
                            ws.send(self.config_msg())
                if streaming and idx < len(frames):
                    ev = self.events.get((conn, idx))
                    if ev and ev[0] == "close":
                        ws.close()
                        return
                    if ev and ev[0] == "config":
                        self.profile = ev[1]
                        ws.send(self.config_msg())
                    pcm, smeter = frames[idx]
                    if smeter is not None:
                        ws.send(json.dumps({"type": "smeter", "value": smeter}))
                    if self.compression == "adpcm":
                        payload, counter = encode_sync_adpcm(pcm, codec, counter)
                    else:
                        payload = pcm.astype("<i2").tobytes()
                    ws.send(b"\x02" + payload)
                    ws.send(b"\x01" + bytes(64))  # a waterfall frame we must ignore
                    idx += 1
                elif streaming:
                    time.sleep(0.01)
        except Exception:
            return
