# HFGCS Swiss Army Tool - Starter
# Airspy HF+ Discovery + SoapySDR
# Per-channel adaptive squelch, scan, activity recording

# Dependencies (Linux Mint / Ubuntu):
#   sudo apt install python3-pip python3-numpy python3-scipy python3-soundfile soapysdr-tools
#   # Then build/install libairspyhf + SoapyAirspyHF (see OpenWebRX wiki notes)
#   pip install numpy scipy soundfile

import argparse
import sys
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import numpy as np
from scipy import signal
import soundfile as sf
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

# ====================== CONFIG ======================
FREQUENCIES_KHZ = [
    4724.0,   # Night primary
    8992.0,
    11175.0,  # Day primary / most active
    15016.0,
    # Add more as needed (ALE list, backups, etc.)
]

SAMPLE_RATE = 192000          # Safe rate for Airspy HF+ Discovery
AUDIO_RATE = 8000             # Output WAV sample rate
USB_BANDWIDTH = 3000          # USB filter bandwidth (Hz)
DWELL_SECONDS = 2.0           # Time spent on each frequency while scanning
HOLD_SECONDS = 8.0            # Minimum hold time once activity detected
PREBUFFER_SECONDS = 1.5       # Audio pre-buffer
NOISE_FLOOR_ALPHA = 0.02      # EMA smoothing for noise floor (lower = slower)
SNR_THRESHOLD_DB = 8.0        # Activity threshold above noise floor
MIN_ACTIVITY_SECONDS = 0.8    # Ignore very short bursts
RECORD_DIR = Path("recordings")
LOG_FILE = Path("hfgcs_catalog.jsonl")

# ====================== HELPERS ======================
def utc_now():
    return datetime.now(timezone.utc).isoformat()

def setup_device():
    args = dict(driver="airspyhf")
    sdr = SoapySDR.Device(args)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setGainMode(SOAPY_SDR_RX, 0, True)  # AGC on
    return sdr

def make_usb_demod(sample_rate, audio_rate, bandwidth):
    """Return a function that turns complex IQ into real USB audio."""
    # Simple USB: shift, lowpass, resample
    shift_freq = bandwidth / 2.0
    t = np.arange(1024) / sample_rate  # small chunk size used later
    # Precompute filter
    nyq = sample_rate / 2
    b, a = signal.butter(4, bandwidth / nyq, btype='low')
    return b, a, shift_freq

def demod_usb(iq, b, a, shift_freq, sample_rate, audio_rate):
    if len(iq) == 0:
        return np.array([])
    t = np.arange(len(iq)) / sample_rate
    shifted = iq * np.exp(-1j * 2 * np.pi * shift_freq * t)
    filtered = signal.lfilter(b, a, shifted)
    # Take real part (USB) and resample
    real = np.real(filtered)
    if sample_rate != audio_rate:
        num = int(len(real) * audio_rate / sample_rate)
        real = signal.resample(real, num)
    # Normalize
    if np.max(np.abs(real)) > 0:
        real = real / np.max(np.abs(real)) * 0.9
    return real.astype(np.float32)

def power_db(iq):
    if len(iq) == 0:
        return -100.0
    return 10 * np.log10(np.mean(np.abs(iq)**2) + 1e-12)

# ====================== MAIN SCANNER ======================
class HFGCSScanner:
    def __init__(self):
        self.sdr = setup_device()
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        self._wait_for_samples()
        self.noise_floors = {f: -80.0 for f in FREQUENCIES_KHZ}
        self.b, self.a, self.shift = make_usb_demod(SAMPLE_RATE, AUDIO_RATE, USB_BANDWIDTH)
        RECORD_DIR.mkdir(exist_ok=True)
        self.catalog = []

    def _wait_for_samples(self, timeout=2.0):
        """The HF+ stream returns timeouts until the first buffer is ready."""
        end = time.time() + timeout
        while time.time() < end:
            if len(self.read_iq()):
                return

    def read_iq(self, nsamps=16384):
        buff = np.zeros(nsamps, np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [buff], nsamps)
        if sr.ret > 0:
            return buff[:sr.ret]
        return np.array([], np.complex64)

    def measure_noise(self, freq_khz, seconds=1.0):
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, freq_khz * 1000)
        time.sleep(0.15)  # settle
        powers = []
        end = time.time() + seconds
        while time.time() < end:
            iq = self.read_iq()
            if len(iq):
                powers.append(power_db(iq))
        if powers:
            return np.median(powers)
        return -80.0

    def scan_once(self):
        for freq in FREQUENCIES_KHZ:
            print(f"[{utc_now()}] Tuning {freq} kHz")
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, freq * 1000)
            time.sleep(0.12)

            # Update noise floor slowly
            noise = self.measure_noise(freq, 0.4)
            self.noise_floors[freq] = (
                NOISE_FLOOR_ALPHA * noise +
                (1 - NOISE_FLOOR_ALPHA) * self.noise_floors[freq]
            )
            threshold = self.noise_floors[freq] + SNR_THRESHOLD_DB

            # Dwell and look for activity
            start = time.time()
            activity_start = None
            prebuffer = deque(maxlen=int(PREBUFFER_SECONDS * AUDIO_RATE))
            recording = []
            is_active = False

            while time.time() - start < DWELL_SECONDS or is_active:
                iq = self.read_iq(8192)
                if len(iq) == 0:
                    continue
                p = power_db(iq)
                audio = demod_usb(iq, self.b, self.a, self.shift, SAMPLE_RATE, AUDIO_RATE)

                if p > threshold:
                    if activity_start is None:
                        activity_start = time.time()
                    if time.time() - activity_start > MIN_ACTIVITY_SECONDS:
                        is_active = True
                        recording.extend(audio)
                else:
                    activity_start = None
                    if not is_active:
                        prebuffer.extend(audio)

                if is_active and (time.time() - start > DWELL_SECONDS + HOLD_SECONDS):
                    break

                if not is_active and time.time() - start > DWELL_SECONDS:
                    break

            if is_active and recording:
                self.save_recording(freq, list(prebuffer) + recording)

    def save_recording(self, freq_khz, audio_samples):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = RECORD_DIR / f"HFGCS_{freq_khz:.0f}kHz_{ts}.wav"
        sf.write(filename, np.array(audio_samples), AUDIO_RATE)
        entry = {
            "timestamp": utc_now(),
            "frequency_khz": freq_khz,
            "file": str(filename),
            "duration_sec": len(audio_samples) / AUDIO_RATE,
            "noise_floor_db": round(self.noise_floors[freq_khz], 1),
            "notes": "USB voice candidate - review for EAM"
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  → Recorded {filename.name} ({entry['duration_sec']:.1f}s)")

    def close(self):
        try:
            self.sdr.deactivateStream(self.rx_stream)
        except Exception:
            pass
        try:
            self.sdr.closeStream(self.rx_stream)
        except Exception:
            pass

    def probe(self, seconds=0.8):
        """Open the Airspy HF+, measure each channel, and release the device."""
        try:
            info = dict(self.sdr.getHardwareInfo())
        except Exception:
            info = {}
        label = info.get("label") or info.get("driver") or "airspyhf"
        serial = info.get("serial", "?")
        rate = self.sdr.getSampleRate(SOAPY_SDR_RX, 0)
        print(f"Device: {label} serial={serial}")
        print(f"Sample rate: {rate:.0f} Hz")
        for freq in FREQUENCIES_KHZ:
            noise = self.measure_noise(freq, seconds)
            print(f"  {freq:8.1f} kHz  full-band power {noise:6.1f} dBFS")

    def run(self):
        print("HFGCS Scanner started. Ctrl+C to stop.")
        print(f"Frequencies: {FREQUENCIES_KHZ}")
        try:
            while True:
                self.scan_once()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description="HFGCS scanner for Airspy HF+ Discovery")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Tune each HFGCS channel, print power, and exit",
    )
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=0.8,
        help="Seconds to measure on each channel during --probe",
    )
    args = parser.parse_args(argv)
    try:
        scanner = HFGCSScanner()
    except Exception as exc:
        print(f"Could not open Airspy HF+: {exc}", file=sys.stderr)
        print("If OpenWebRX is using the radio, stop that client or use hfgcs_waterfall.py.", file=sys.stderr)
        return 1
    try:
        if args.probe:
            scanner.probe(args.probe_seconds)
        else:
            scanner.run()
    finally:
        if args.probe:
            scanner.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
