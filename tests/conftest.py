import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IQ_RATE = 192_000
AUDIO_RATE = 8_000
T0 = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)


def tone(freq_hz, seconds, amp=1.0, rate=IQ_RATE, phase=0.0):
    t = np.arange(int(round(seconds * rate))) / rate
    return amp * np.exp(1j * (2 * np.pi * freq_hz * t + phase))


def noise(seconds, amp, seed=0, rate=IQ_RATE):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    return amp * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def chunks(x, size=8192):
    for i in range(0, len(x), size):
        yield x[i : i + size]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
