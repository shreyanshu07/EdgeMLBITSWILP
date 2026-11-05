"""
preprocessing.py — LogiEdge On-Device Preprocessing Pipeline
==============================================================

Implements the three-stage preprocessing pipeline required by Task C2:

    1. Filtering      : 5-sample moving average on temperature and vibration
    2. Feature extraction : 30-second sliding window, 10-second step, producing
                            a 6-value feature vector per window
    3. Normalisation  : z-score using FIXED statistics computed once from 10
                         minutes of clean Normal-class data (training_stats.npy),
                         loaded at runtime -- NEVER recomputed from live data.

Feature vector (order matters -- this exact order is used everywhere else
in the project: training, conversion, inference, drift monitoring):

    [0] temp_mean              (deg C)
    [1] temp_std                (deg C)
    [2] temp_rate_of_change     (deg C / minute)
    [3] vibration_rms           (g)
    [4] vibration_peak          (g)
    [5] vibration_kurtosis      (unitless, excess kurtosis)

Course context: AIML ZG535 Mini-Project, Component C (Module 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

WINDOW_S = 30
STEP_S = 10
MOVING_AVG_N = 5

FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_rate_of_change_per_min",
    "vibration_rms", "vibration_peak", "vibration_kurtosis",
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Stage 1: Filtering
# ---------------------------------------------------------------------------

def moving_average(values: List[float], n: int = MOVING_AVG_N) -> np.ndarray:
    """
    Simple causal 5-sample moving average. The first (n-1) samples use a
    shrinking window (so output length == input length, no NaNs at the edges
    -- important because the edge device cannot "look into the future").
    """
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - n + 1)
        out[i] = values[lo:i + 1].mean()
    return out


# ---------------------------------------------------------------------------
# Stage 2: Feature extraction
# ---------------------------------------------------------------------------

def _excess_kurtosis(x: np.ndarray) -> float:
    """Manual excess-kurtosis (Fisher definition, normal == 0.0) -- avoids a
    SciPy dependency on constrained edge hardware."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    mean = x.mean()
    std = x.std()
    if std < 1e-9:
        return 0.0
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / (std ** 4) - 3.0)


def extract_features(temp_window: np.ndarray, temp_t: np.ndarray,
                      vib_window: np.ndarray) -> np.ndarray:
    """
    Compute the 6-value feature vector for one 30-second window.

    temp_window : filtered temperature samples within the window (1 Hz -> ~30 samples)
    temp_t      : timestamps (seconds) matching temp_window, used for rate-of-change
    vib_window  : filtered vibration_rms samples within the window (0.5 Hz -> ~15 samples)
    """
    temp_mean = float(np.mean(temp_window))
    temp_std = float(np.std(temp_window))

    # Rate of change: linear regression slope of temperature vs time,
    # converted from deg C/s to deg C/min.
    if len(temp_window) >= 2 and (temp_t[-1] - temp_t[0]) > 0:
        slope_per_s = np.polyfit(temp_t, temp_window, 1)[0]
    else:
        slope_per_s = 0.0
    temp_rate_of_change = float(slope_per_s * 60.0)

    vibration_rms = float(np.sqrt(np.mean(np.square(vib_window))))
    vibration_peak = float(np.max(np.abs(vib_window)))
    vibration_kurtosis = _excess_kurtosis(vib_window)

    return np.array([
        temp_mean, temp_std, temp_rate_of_change,
        vibration_rms, vibration_peak, vibration_kurtosis,
    ], dtype=float)


@dataclass
class WindowedFeatureExtractor:
    """
    Stateful sliding-window extractor: feed it raw (t, value) samples for
    each stream as they arrive, and it emits a feature vector every
    STEP_S seconds once a full WINDOW_S of data is available.
    """
    window_s: float = WINDOW_S
    step_s: float = STEP_S
    _temp_buf: List[Tuple[float, float]] = field(default_factory=list)
    _vib_buf: List[Tuple[float, float]] = field(default_factory=list)
    _next_emit_t: float = field(default=None)

    def add_temperature(self, t: float, value: float) -> None:
        self._temp_buf.append((t, value))

    def add_vibration(self, t: float, value: float) -> None:
        self._vib_buf.append((t, value))

    def _trim(self, buf: List[Tuple[float, float]], now: float) -> None:
        cutoff = now - self.window_s
        while buf and buf[0][0] < cutoff:
            buf.pop(0)

    def poll(self, now: float) -> Optional[np.ndarray]:
        """
        Call periodically (e.g. once per second) with the current simulated
        time. Returns a feature vector exactly every `step_s` seconds once
        `window_s` of history has accumulated, else None.
        """
        if self._next_emit_t is None:
            self._next_emit_t = self.window_s  # first full window available at t=window_s

        self._trim(self._temp_buf, now)
        self._trim(self._vib_buf, now)

        if now < self._next_emit_t:
            return None

        self._next_emit_t += self.step_s

        if len(self._temp_buf) < 2 or len(self._vib_buf) < 2:
            return None

        temp_t = np.array([p[0] for p in self._temp_buf])
        temp_v = moving_average([p[1] for p in self._temp_buf])
        vib_v = moving_average([p[1] for p in self._vib_buf])

        return extract_features(temp_v, temp_t, vib_v)


# ---------------------------------------------------------------------------
# Stage 3: Normalisation
# ---------------------------------------------------------------------------

def compute_training_stats(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (mean, std) per feature from clean Normal-class data.
    Call ONCE, offline, on >= 10 minutes of Normal-class windows, then
    persist with save_training_stats(). Never call this at inference time.
    """
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std < 1e-6] = 1e-6  # guard against divide-by-zero on a constant feature
    return mean, std


def save_training_stats(path: str, mean: np.ndarray, std: np.ndarray) -> None:
    np.save(path, np.stack([mean, std]))  # shape (2, N_FEATURES)


def load_training_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    mean, std = arr[0], arr[1]
    return mean, std


def normalise(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalisation using FIXED, pre-computed statistics."""
    return (features - mean) / std


if __name__ == "__main__":
    # Minimal self-test: fabricate a clean sine-free signal and confirm
    # moving_average / extract_features / normalise run without error.
    import sys

    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 1.0)
    temp = 4.0 + rng.normal(0, 0.3, size=len(t))
    vib_t = np.arange(0, 30, 2.0)
    vib = 0.45 + rng.normal(0, 0.05, size=len(vib_t))

    temp_f = moving_average(temp.tolist())
    vib_f = moving_average(vib.tolist())
    feats = extract_features(temp_f, t, vib_f)
    print("Feature vector:", dict(zip(FEATURE_NAMES, feats.round(4))))

    mean, std = compute_training_stats(np.tile(feats, (50, 1)) + rng.normal(0, 0.01, (50, N_FEATURES)))
    norm = normalise(feats, mean, std)
    print("Normalised:", norm.round(3))
    print("preprocessing.py self-test OK", file=sys.stderr)
