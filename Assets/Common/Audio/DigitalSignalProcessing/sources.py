# Created on 2024-07-08
# Author: ChatGPT
# Description: Excitation source generators for voice synthesis.
"""Excitation source generators."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from typing import Optional

import numpy as np

from .constants import DTYPE, EPS
from .core import _db_to_lin
from .filters import _bandpass_biquad_coeff, _biquad_process, _lip_radiation

__all__ = ["GlottalSourceResult", "_glottal_source", "_gen_band_noise"]

DEFAULT_DRIFT_CENTS = 8.0
DEFAULT_DRIFT_RETURN_RATE = 0.38
DEFAULT_VIBRATO_DEPTH_CENTS = 25.0
DEFAULT_VIBRATO_FREQUENCY_HZ = 5.5
DEFAULT_TREMOR_DEPTH_CENTS = 7.0
DEFAULT_TREMOR_FREQUENCY_HZ = 9.5
DEFAULT_TREMOLO_DEPTH_DB = 0.3
DEFAULT_TREMOLO_FREQUENCY_HZ = 5.5
DEFAULT_AM_OU_SIGMA = 0.028
DEFAULT_AM_OU_TAU = 0.22
DEFAULT_AM_OU_CLIP_MULTIPLE = 3.5
DEFAULT_SHIMMER_PERCENT = 0.035
_SHIMMER_STD_LIMIT_MULTIPLIER = 4.0
_MIN_SHIMMER_FACTOR = 0.2
_MAX_SHIMMER_FACTOR = 4.0
_MIN_AMPLITUDE_ENVELOPE = 0.05
_MAX_AMPLITUDE_ENVELOPE = 8.0
_GLOTTAL_LP_CUTOFF_HZ = 800.0
DEFAULT_PERIOD_JITTER_STD = 0.006
DEFAULT_PERIOD_JITTER_LIMIT = 0.035
_MIN_PERIOD_SCALE = 0.6


@dataclass(frozen=True)
class GlottalSourceResult:
    """Signal and control trajectories emitted by the glottal source."""

    signal: np.ndarray
    instantaneous_frequency: np.ndarray
    amplitude_envelope: np.ndarray
    cents_modulation: np.ndarray


def _generate_ou_series(
    sample_count: int,
    dt: float,
    tau_seconds: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    clip_multiple: Optional[float] = None,
) -> np.ndarray:
    """Generate a discrete OU trajectory with optional clipping."""

    if sample_count <= 0 or tau_seconds <= 0.0 or sigma <= 0.0:
        return np.zeros(max(sample_count, 0), dtype=np.float64)

    alpha = float(np.exp(-dt / tau_seconds))
    alpha = np.clip(alpha, 0.0, 0.999999)
    noise_std = float(sigma) * sqrt(max(0.0, 1.0 - alpha * alpha))
    limit = None
    if clip_multiple is not None and clip_multiple > 0.0:
        limit = float(sigma) * float(clip_multiple)

    series = np.empty(sample_count, dtype=np.float64)
    state = 0.0
    for index in range(sample_count):
        state = alpha * state + rng.normal(0.0, noise_std)
        if limit is not None:
            state = float(np.clip(state, -limit, limit))
        series[index] = state
    return series


def _sample_shimmer_track(
    phase: np.ndarray,
    shimmer_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return per-period amplitude scalars implementing shimmer."""

    if shimmer_std <= 0.0 or phase.size == 0:
        return np.ones_like(phase, dtype=np.float64)

    cycle_ids = np.floor(phase / (2.0 * pi)).astype(np.int64)
    cycle_ids -= int(cycle_ids.min(initial=0))
    cycle_count = int(cycle_ids.max(initial=0)) + 1
    factors = 1.0 + rng.normal(0.0, shimmer_std, size=cycle_count + 1)
    spread = _SHIMMER_STD_LIMIT_MULTIPLIER * shimmer_std
    lo = max(_MIN_SHIMMER_FACTOR, 1.0 - spread)
    hi = min(_MAX_SHIMMER_FACTOR, 1.0 + spread)
    factors = np.clip(factors, lo, hi)
    return factors[cycle_ids]


def _sample_period_scale(
    rng: np.random.Generator,
    std: float,
    limit: float,
) -> float:
    """Return a per-period multiplicative factor for pitch jitter."""

    if std <= 0.0:
        return 1.0

    scale = 1.0 + rng.normal(0.0, std)
    if limit > 0.0:
        scale = float(np.clip(scale, 1.0 - limit, 1.0 + limit))
    return float(max(scale, _MIN_PERIOD_SCALE))


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
    tremolo_depth_db: float = DEFAULT_TREMOLO_DEPTH_DB,
    tremolo_frequency_hz: float = DEFAULT_TREMOLO_FREQUENCY_HZ,
    shimmer_percent: Optional[float] = None,
    amplitude_ou_sigma: float = DEFAULT_AM_OU_SIGMA,
    amplitude_ou_tau: float = DEFAULT_AM_OU_TAU,
    amplitude_ou_clip_multiple: float = DEFAULT_AM_OU_CLIP_MULTIPLE,
    period_jitter_std: float = DEFAULT_PERIOD_JITTER_STD,
    period_jitter_limit: float = DEFAULT_PERIOD_JITTER_LIMIT,
) -> GlottalSourceResult:
    """Saw-like glottal source enriched with humanising modulations."""

    rng = np.random.default_rng()
    n = max(0, int(dur_s * sr))
    if n == 0:
        empty_signal = np.zeros(0, dtype=DTYPE)
        empty_control = np.zeros(0, dtype=np.float64)
        return GlottalSourceResult(empty_signal, empty_control, empty_signal, empty_control)

    jn = rng.standard_normal(n).astype(DTYPE)
    pole = 0.999
    js = np.empty_like(jn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * jn[i]
        js[i] = acc
    jitter_semitones = (jitter_cents / 100.0) * js

    drift = np.zeros(n, dtype=np.float64)
    if drift_cents > 0.0 and drift_return_rate > 0.0:
        theta = float(drift_return_rate)
        dt = 1.0 / float(sr)
        sigma = float(drift_cents) * sqrt(2.0 * theta)
        sqrt_dt = sqrt(dt)
        state = 0.0
        for index in range(n):
            state += theta * (-state) * dt + sigma * sqrt_dt * rng.standard_normal()
            drift[index] = state
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
    base_inst_f = f0 * (2.0 ** (composite_semitones / 12.0))

    cycle_phase = float(rng.uniform(0.0, 2.0 * pi))
    cycle_index = 0
    current_scale = _sample_period_scale(rng, float(period_jitter_std), float(period_jitter_limit))
    cycle_positions = np.empty(n, dtype=np.float64)
    phase_track = np.empty(n, dtype=np.float64)
    freq_track = np.empty(n, dtype=np.float64)
    for idx, base_freq in enumerate(base_inst_f):
        scale = current_scale
        freq = max(base_freq * scale, EPS)
        freq_track[idx] = freq
        delta = 2.0 * pi * freq / sr
        cycle_phase += delta
        while cycle_phase >= 2.0 * pi:
            cycle_phase -= 2.0 * pi
            cycle_index += 1
            current_scale = _sample_period_scale(
                rng,
                float(period_jitter_std),
                float(period_jitter_limit),
            )
        cycle_positions[idx] = cycle_phase / (2.0 * pi)
        phase_track[idx] = 2.0 * pi * (cycle_index + cycle_positions[idx])

    saw = (2.0 * cycle_positions - 1.0).astype(np.float64)

    shimmer_std = shimmer_percent if shimmer_percent is not None else max(_db_to_lin(shimmer_db) - 1.0, 0.0)
    shimmer_track = _sample_shimmer_track(phase_track, float(shimmer_std), rng)

    dt = 1.0 / float(sr)
    amp_ou = _generate_ou_series(
        n,
        dt,
        float(amplitude_ou_tau),
        float(amplitude_ou_sigma),
        rng,
        clip_multiple=amplitude_ou_clip_multiple,
    )
    amp_ou = np.clip(1.0 + amp_ou, _MIN_AMPLITUDE_ENVELOPE, _MAX_AMPLITUDE_ENVELOPE)

    tremolo = np.ones(n, dtype=np.float64)
    if tremolo_depth_db > 0.0 and tremolo_frequency_hz > 0.0:
        tremolo_phase = rng.uniform(0.0, 2.0 * pi)
        tremolo = 10.0 ** (
            (tremolo_depth_db * np.sin(2.0 * pi * tremolo_frequency_hz * t + tremolo_phase))
            / 20.0
        )

    amplitude_envelope = shimmer_track * amp_ou * tremolo
    raw = saw * amplitude_envelope

    decay = exp(-2.0 * pi * _GLOTTAL_LP_CUTOFF_HZ / sr)
    y = np.empty_like(raw, dtype=np.float64)
    state = 0.0
    for index, sample in enumerate(raw):
        state = (1.0 - decay) * sample + decay * state
        y[index] = state

    signal = y.astype(DTYPE, copy=False)
    amp_env = amplitude_envelope.astype(DTYPE, copy=False)
    freq_track = freq_track.astype(np.float64, copy=False)
    with np.errstate(divide='ignore'):
        cents_track = 1200.0 * np.log2(np.maximum(freq_track, EPS) / max(f0, EPS))
    cents_track = np.nan_to_num(cents_track, nan=0.0, posinf=0.0, neginf=0.0)
    return GlottalSourceResult(signal, freq_track, amp_env, cents_track)


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
