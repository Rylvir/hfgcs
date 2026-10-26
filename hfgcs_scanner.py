# HFGCS scanner for an Airspy HF+ Discovery (or any SoapySDR device).
#
# Scans the HFGCS channels from hfgcs.config, opens a voice-slot squelch on
# activity, records USB audio with pre-roll until the signal drops for the
# hang time, and appends one JSON line per recording to the catalog.
#
# SoapySDR comes from the system (sudo apt install python3-soapysdr); it is
# imported lazily so the DSP and tests run without it.

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np

from hfgcs.capture import ChannelCapture, save_recording
from hfgcs.config import Config, load_config
from hfgcs.dsp import power_db

SETTLE_SECONDS = 0.12  # discard samples right after a retune
BLOCK_SAMPLES = 8192


def utc_now():
    return datetime.now(timezone.utc)


def _soapy():
    try:
        import SoapySDR
    except ImportError as exc:  # pragma: no cover - needs hardware stack
        raise RuntimeError(
            "SoapySDR Python bindings not found (sudo apt install python3-soapysdr)"
        ) from exc
    return SoapySDR


def setup_device(cfg: Config):
    SoapySDR = _soapy()
    sdr = SoapySDR.Device(cfg.soapy_args)
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, cfg.iq_rate)
    # Tuner AGC off by default: it makes dB squelch thresholds meaningless.
    try:
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, cfg.tuner_agc)
    except Exception:
        pass
    return sdr


class HFGCSScanner:
    def __init__(self, cfg: Config):
        SoapySDR = _soapy()
        self.SoapySDR = SoapySDR
        self.cfg = cfg
        self.sdr = setup_device(cfg)
        self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        self._wait_for_samples()
        self.captures = {f: ChannelCapture(cfg, f) for f in cfg.channels_khz}

    def _wait_for_samples(self, timeout=2.0):
        """The HF+ stream returns timeouts until the first buffer is ready."""
        end = time.time() + timeout
        while time.time() < end:
            if len(self.read_iq()):
                return

    def read_iq(self, nsamps=BLOCK_SAMPLES):
        buff = np.zeros(nsamps, np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [buff], nsamps)
        if sr.ret > 0:
            return buff[: sr.ret]
        return np.array([], np.complex64)

    def tune(self, freq_khz):
        """Tune once, then drop the samples produced while the PLL settles."""
        self.sdr.setFrequency(self.SoapySDR.SOAPY_SDR_RX, 0, freq_khz * 1000)
        end = time.time() + SETTLE_SECONDS
        while time.time() < end:
            self.read_iq()

    def dwell(self, freq_khz):
        cap = self.captures[freq_khz]
        self.tune(freq_khz)
        cap.retune(utc_now())
        start = time.time()
        while time.time() - start < self.cfg.dwell_seconds or cap.active:
            iq = self.read_iq()
            if len(iq) == 0:
                continue
            for rec in cap.feed(iq):
                self._save(rec)
        for rec in cap.flush():
            self._save(rec)

    def _save(self, rec):
        entry = save_recording(rec, self.cfg.record_dir, self.cfg.catalog_file)
        print(
            f"  -> {entry['file']} ({entry['duration_sec']:.1f}s, "
            f"SNR peak {entry['snr_db_peak']} dB, start {entry['start_utc']})"
        )

    def scan_once(self):
        for freq in self.cfg.channels_khz:
            print(f"[{utc_now().isoformat()}] {freq:.1f} kHz")
            self.dwell(freq)

    def probe(self, seconds=0.8):
        """Open the device, measure the voice slot on each channel, release it."""
        try:
            info = dict(self.sdr.getHardwareInfo())
        except Exception:
            info = {}
        label = info.get("label") or info.get("driver") or "airspyhf"
        print(f"Device: {label} serial={info.get('serial', '?')}")
        print(f"Sample rate: {self.sdr.getSampleRate(self.SoapySDR.SOAPY_SDR_RX, 0):.0f} Hz")
        for freq in self.cfg.channels_khz:
            cap = self.captures[freq]
            self.tune(freq)
            cap.retune(utc_now())
            powers = []
            end = time.time() + seconds
            while time.time() < end:
                iq = self.read_iq()
                if len(iq):
                    _, p = cap.demod.process(iq)
                    powers.append(power_db(p))
            level = np.median(powers) if powers else float("nan")
            print(f"  {freq:8.1f} kHz  voice-slot power {level:6.1f} dBFS")

    def run(self):
        print("HFGCS scanner started. Ctrl+C to stop.")
        print(f"Receiver {self.cfg.receiver_id}; channels {self.cfg.channels_khz} kHz")
        try:
            while True:
                self.scan_once()
        except KeyboardInterrupt:
            print("\nStopping...")

    def close(self):
        for call in (self.sdr.deactivateStream, self.sdr.closeStream):
            try:
                call(self.rx_stream)
            except Exception:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="HFGCS scanner for Airspy HF+ / SoapySDR")
    parser.add_argument("--config", help="TOML file overriding hfgcs.config defaults")
    parser.add_argument("--probe", action="store_true", help="Measure each channel and exit")
    parser.add_argument("--probe-seconds", type=float, default=0.8)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        scanner = HFGCSScanner(cfg)
    except Exception as exc:
        print(f"Could not open the SDR: {exc}", file=sys.stderr)
        print("If OpenWebRX is using the radio, stop that client or use hfgcs_waterfall.py.", file=sys.stderr)
        return 1
    try:
        if args.probe:
            scanner.probe(args.probe_seconds)
        else:
            scanner.run()
    finally:
        scanner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
