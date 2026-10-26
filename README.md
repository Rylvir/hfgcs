# HFGCS Swiss Army Tool (Starter)

Python + SoapySDR scanner for Airspy HF+ Discovery focused on HFGCS voice (EAM) monitoring.

## Features in this starter
- Scans primary HFGCS frequencies
- Per-frequency adaptive noise floor + SNR squelch
- Activity-triggered USB audio recording with pre-buffer
- JSONL catalog (`hfgcs_catalog.jsonl`) + WAV files in `recordings/`
- Clear places to add spectral classification or external decoders later

## Setup on Linux Mint 22.x

1. Install system packages:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-numpy python3-scipy python3-soundfile soapysdr-tools libusb-1.0-0-dev cmake
   ```

2. Build Airspy HF+ support (required once):
   ```bash
   git clone https://github.com/airspy/airspyhf.git
   cd airspyhf && mkdir build && cd build
   cmake -DINSTALL_UDEV_RULES=ON ..
   make && sudo make install
   sudo ldconfig
   cd ../..

   git clone https://github.com/pothosware/SoapyAirspyHF.git
   cd SoapyAirspyHF && mkdir build && cd build
   cmake .. && make && sudo make install
   ```

3. Verify device:
   ```bash
   SoapySDRUtil --probe="driver=airspyhf"
   ```

4. Python deps:
   ```bash
   python3 -m venv --system-site-packages .venv
   .venv/bin/pip install -e ".[openwebrx]"     # or: -r requirements.txt
   sudo apt install python3-soapysdr
   ```
   Developers: `.venv/bin/pip install -e ".[dev]" && .venv/bin/pytest` (no SDR needed).
   On this machine the Airspy HF+ libraries and SoapyAirspyHF module are already installed. `python3-soapysdr` is not in the system Python path, so the project venv carries a copy of that module extracted from the Ubuntu package.

5. Initial device check (releases the radio when it finishes):
   ```bash
   .venv/bin/python hfgcs_scanner.py --probe
   ```
   OpenWebRX is already using this Airspy HF+ for the public waterfall. A local probe and the waterfall cannot both hold the USB device. Stop the OpenWebRX client, or use the waterfall test below.

6. Remote waterfall test against the OpenWebRX receiver at https://waterfall.scanplacer.com:
   ```bash
   .venv/bin/python hfgcs_waterfall.py
   # same radio, on the machine itself:
   .venv/bin/python hfgcs_waterfall.py --url http://127.0.0.1:8073
   ```
   The test selects each HFGCS profile, reads the ADPCM waterfall, and prints noise floor plus the USB voice slice (150–2750 Hz). Profile changes are spaced 11 seconds apart so OpenWebRX does not treat the client as a bot.

7. Continuous scanner (only when OpenWebRX is not using the HF+):
   ```bash
   .venv/bin/python hfgcs_scanner.py
   ```

8. Record without opening the Airspy, through OpenWebRX's own demodulator
   (only channels inside the profile the server is on; it never switches
   profiles unless you pass `--allow-profile-switch`):
   ```bash
   .venv/bin/python hfgcs_scanner.py --source openwebrx --channel 11175
   ```
   Other sources: `--source kiwisdr` (via kiwiclient, with etiquette limits)
   and `--source file` (WAV/IQ replay). See [docs/sources.md](docs/sources.md).

## Configuration and output
All tools share `hfgcs/config.py`: the six HFGCS channels (4724, 6739, 8992,
11175, 13200, 15016 kHz), receiver ID, squelch, hang time, pre-roll and AGC.
Override any value with a TOML file: `hfgcs_scanner.py --config config.toml`
(see `config.example.toml`).

Each recording is a 16-bit WAV in `recordings/` plus one JSON line in
`hfgcs_catalog.jsonl` with `start_utc` (activity onset, UTC ISO 8601),
`file_start_utc`, `end_utc`, `frequency_khz`, `receiver_id`, `snr_db_peak`,
`snr_db_mean`, `noise_floor_db`, `preroll_sec`, `duration_sec`, `sha256`.

## DSP notes
- USB demodulation is Weaver-style: mix the passband centre to 0 Hz, complex
  elliptic low-pass (the sideband filter), decimate 192 k -> 8 k, mix back up.
  Tones keep their frequency and the opposite sideband is rejected by > 80 dB.
- Mixer phase, filter memory and decimator phase persist across blocks; one
  attack/decay AGC replaces per-block normalisation.
- The squelch measures the 3 kHz voice slot, opens after `min_activity_seconds`,
  stays open until the signal has been below the close level for
  `hang_seconds`, and freezes its noise floor while open. The audio from the
  activity window and `preroll_seconds` before it are kept.
- Tuner AGC is off by default (`tuner_agc`) so dB thresholds are stable.

## Next improvements (easy to add)
- Better USB demod (liquid-dsp or more precise filter)
- Whisper / ASR transcription of recordings for automatic EAM character extraction
- Spectral classifier to tag voice vs digital
- ALE decoder hook
- Web UI or simple dashboard
- Multi-device support later

Recordings and the JSONL log are structured for easy review or contribution to projects like NEET INTEL.

## License

MIT, see [LICENSE](LICENSE). If Apache-2.0 code (for example from
Chipensaw/HFGCS-Monitor-KiwiSDR) is reused later, it keeps its original
license header and NOTICE alongside it.
