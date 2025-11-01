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
    pole = 0.999
    js = np.empty_like(jn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * jn[i]
        js[i] = acc
    jitter_semitones = (jitter_cents / 100.0) * js

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
    drift_semitones = drift / 100.0

    t = np.arange(n, dtype=np.float64) / float(sr)
    vibrato_semitones = np.zeros(n, dtype=np.float64)
    if vibrato_depth_cents != 0.0 and vibrato_frequency_hz > 0.0:
        vib_phase = rng.uniform(0.0, 2.0 * pi)
        vibrato_semitones = (
            (vibrato_depth_cents / 100.0)
            * np.sin(2.0 * pi * vibrato_frequency_hz * t + vib_phase)
        )

    tremor_semitones = np.zeros(n, dtype=np.float64)
    if tremor_depth_cents != 0.0 and tremor_frequency_hz > 0.0:
        tremor_phase = rng.uniform(0.0, 2.0 * pi)
        tremor_semitones = (
            (tremor_depth_cents / 100.0)
            * np.sin(2.0 * pi * tremor_frequency_hz * t + tremor_phase)
        )

    composite_semitones = (
        jitter_semitones.astype(np.float64)
        + drift_semitones.astype(np.float64)
        + vibrato_semitones
        + tremor_semitones
    )
    inst_f = f0 * (2.0 ** (composite_semitones / 12.0))

    phase = np.cumsum(2.0 * pi * inst_f / sr, dtype=np.float64)
    saw = (2.0 * ((phase / (2.0 * pi)) % 1.0) - 1.0).astype(DTYPE)

    sn = rng.standard_normal(n).astype(DTYPE)
    ss = np.empty_like(sn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * sn[i]
        ss[i] = acc
    amp = (_db_to_lin(shimmer_db) ** ss).astype(DTYPE)

    raw = saw * amp

    decay = np.exp(-2.0 * pi * 800.0 / sr)
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
