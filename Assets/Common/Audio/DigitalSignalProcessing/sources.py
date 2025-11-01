# Created on 2024-07-08
# Author: ChatGPT
# Description: Excitation source generators for voice synthesis.
"""Excitation source generators."""
from __future__ import annotations

from math import pi, sqrt

import numpy as np

from .constants import DTYPE
from .core import _db_to_lin
from .filters import _bandpass_biquad_coeff, _biquad_process, _lip_radiation

__all__ = ["_glottal_source", "_gen_band_noise"]

DEFAULT_DRIFT_CENTS = 12.0
DEFAULT_DRIFT_RETURN_RATE = 0.35
DEFAULT_VIBRATO_DEPTH_CENTS = 0.0
DEFAULT_VIBRATO_FREQUENCY_HZ = 5.5
DEFAULT_TREMOR_DEPTH_CENTS = 0.0
DEFAULT_TREMOR_FREQUENCY_HZ = 8.5
JITTER_AR_POLE = 0.999
SHIMMER_DECAY_FREQ_HZ = 800.0
TWO_PI = 2.0 * pi


def _glottal_source(
    f0: float,
    dur_s: float,
    sr: int,
    jitter_cents: float = 10.0,
    shimmer_db: float = 0.8,
    *,
    drift_cents: float = DEFAULT_DRIFT_CENTS,
    drift_return_rate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibrato_depth_cents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibrato_frequency_hz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremor_depth_cents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremor_frequency_hz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
) -> np.ndarray:
    """Saw-like glottal source with composite cents modulation."""
    rng = np.random.default_rng()
    n = max(0, int(dur_s * sr))
    if n == 0:
        return np.zeros(0, dtype=DTYPE)

    jn = rng.standard_normal(n).astype(DTYPE)
    pole = JITTER_AR_POLE
    js = np.empty_like(jn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * jn[i]
        js[i] = acc
    jitter_cents_series = float(jitter_cents) * js.astype(np.float64)

    # Ornstein-Uhlenbeck drift in cents
    drift = np.zeros(n, dtype=DTYPE)
    if drift_cents > 0.0 and drift_return_rate > 0.0:
        theta = float(drift_return_rate)
        dt = 1.0 / float(sr)
        sigma = float(drift_cents) * sqrt(2.0 * theta)
        sqrt_dt = sqrt(dt)
        state = 0.0
        for i in range(n):
            state += theta * (-state) * dt + sigma * sqrt_dt * rng.standard_normal()
            drift[i] = state
    drift_cents_series = drift.astype(np.float64)

    t = np.arange(n, dtype=np.float64) / float(sr)
    vibrato_cents = np.zeros(n, dtype=np.float64)
    if vibrato_depth_cents != 0.0 and vibrato_frequency_hz > 0.0:
        vib_phase = rng.uniform(0.0, TWO_PI)
        vibrato_cents = (
            float(vibrato_depth_cents)
            * np.sin(TWO_PI * float(vibrato_frequency_hz) * t + vib_phase)
        )

    tremor_cents = np.zeros(n, dtype=np.float64)
    if tremor_depth_cents != 0.0 and tremor_frequency_hz > 0.0:
        tremor_phase = rng.uniform(0.0, TWO_PI)
        tremor_cents = (
            float(tremor_depth_cents)
            * np.sin(TWO_PI * float(tremor_frequency_hz) * t + tremor_phase)
        )

    composite_cents = (
        jitter_cents_series
        + drift_cents_series
        + vibrato_cents
        + tremor_cents
    )
    inst_f = float(f0) * (2.0 ** (composite_cents / 1200.0))

    phase = np.cumsum(TWO_PI * inst_f / float(sr), dtype=np.float64)
    saw = (2.0 * ((phase / TWO_PI) % 1.0) - 1.0).astype(DTYPE)

    sn = rng.standard_normal(n).astype(DTYPE)
    ss = np.empty_like(sn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * sn[i]
        ss[i] = acc
    amp = (_db_to_lin(shimmer_db) ** ss).astype(DTYPE)

    raw = saw * amp

    decay = np.exp(-TWO_PI * SHIMMER_DECAY_FREQ_HZ / float(sr))
    y = np.empty_like(raw)
    s = 0.0
    for i in range(n):
        s = (1 - decay) * raw[i] + decay * s
        y[i] = s
    return y


def _gen_band_noise(
    dur_s: float,
    sr: int,
    center: float,
    Q: float,
    *,
    use_lip_radiation: bool = True,
) -> np.ndarray:
    """Generate band-limited noise around ``center`` Hz."""
    rng = np.random.default_rng()
    n = max(0, int(dur_s * sr))
    if n == 0:
        return np.zeros(0, dtype=DTYPE)
    noise = rng.standard_normal(n).astype(DTYPE)
    b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(center, Q, sr)
    y = _biquad_process(noise, b0, b1, b2, a1, a2)
    return _lip_radiation(y) if use_lip_radiation else y
