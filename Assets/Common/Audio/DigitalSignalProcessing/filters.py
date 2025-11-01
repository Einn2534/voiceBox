# Created on 2024-08-28
# Created by ChatGPT
# Description: Filter design and processing helpers.
"""Filter design and processing helpers."""
from __future__ import annotations

from math import cos, exp, pi, sin, sqrt
from typing import Iterable, Sequence, Tuple

import numpy as np

from .articulation import _area_profile_to_reflections, _reflection_to_lpc
from .constants import DTYPE, EPS
from .core import _clamp, _ensure_array

__all__ = [
    "_bandpass_biquad_coeff",
    "_notch_biquad_coeff",
    "_biquad_process",
    "_one_pole_lp",
    "_lip_radiation",
    "_apply_formant_filters",
    "_apply_nasal_antiresonances",
    "_apply_all_pole_filter",
    "_apply_kelly_lochbaum_filter",
    "_pre_emphasis",
    "_apply_breath_noise_coloration",
]


_RBJ_MIN_Q = 0.5
_BREATH_NOISE_HPF_CUTOFF_HZ = 2000.0
_BREATH_NOISE_HPF_Q = 1.0 / sqrt(2.0)
_BREATH_NOISE_BPF_CENTER_HZ = 5000.0
_BREATH_NOISE_BPF_Q = 1.4
_BREATH_NOISE_LPF_CUTOFF_HZ = 8000.0
_BREATH_NOISE_LPF_Q = 1.0 / sqrt(2.0)
_NYQUIST_SAFETY = 0.48
_FORMANT_MIN_FREQ_HZ = 40.0
_FORMANT_MIN_BW_HZ = 15.0
_FORMANT_MAX_BW_RATIO = 0.45


def _bandpass_biquad_coeff(f0: float, Q: float, sr: int) -> Tuple[float, float, float, float, float]:
    """RBJ band-pass filter coefficients (constant peak gain variant)."""
    w0 = 2.0 * pi * f0 / sr
    alpha = sin(w0) / (2.0 * max(Q, 0.5))
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos(w0)
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _lowpass_biquad_coeff(cutoff: float, Q: float, sr: int) -> Tuple[float, float, float, float, float]:
    """RBJ low-pass filter coefficients."""

    cutoff = float(max(cutoff, 0.0))
    if sr <= 0 or cutoff <= 0.0:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    w0 = 2.0 * pi * cutoff / sr
    alpha = sin(w0) / (2.0 * max(Q, _RBJ_MIN_Q))
    cos_w0 = cos(w0)
    b0 = (1.0 - cos_w0) * 0.5
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _highpass_biquad_coeff(cutoff: float, Q: float, sr: int) -> Tuple[float, float, float, float, float]:
    """RBJ high-pass filter coefficients."""

    cutoff = float(max(cutoff, 0.0))
    if sr <= 0 or cutoff <= 0.0:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    w0 = 2.0 * pi * cutoff / sr
    alpha = sin(w0) / (2.0 * max(Q, _RBJ_MIN_Q))
    cos_w0 = cos(w0)
    b0 = (1.0 + cos_w0) * 0.5
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _notch_biquad_coeff(f0: float, bw: float, sr: int) -> Tuple[float, float, float, float, float]:
    """Design a biquad notch filter centred at ``f0`` with bandwidth ``bw``."""

    f0 = float(f0)
    bw = float(max(bw, 1.0))
    if sr <= 0:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    theta = 2.0 * pi * f0 / sr
    theta = float(np.clip(theta, 1e-4, np.pi - 1e-4))
    r = float(np.exp(-pi * bw / sr))
    r = float(_clamp(r, 0.0, 0.9995))
    b0 = 1.0
    b1 = -2.0 * cos(theta)
    b2 = 1.0
    a1 = -2.0 * r * cos(theta)
    a2 = r * r

    sum_b = b0 + b1 + b2
    sum_a = 1.0 + a1 + a2
    if abs(sum_b) > EPS:
        gain = sum_a / sum_b
        b0 *= gain
        b1 *= gain
        b2 *= gain
    return b0, b1, b2, a1, a2


def _biquad_process(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    """Process ``x`` with a single biquad filter."""
    x = _ensure_array(x)
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def _bandpass_biquad_coeff_track(freq: np.ndarray, Q: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RBJ band-pass coefficients for time-varying parameters."""

    freq = np.asarray(freq, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if sr <= 0:
        zeros = np.zeros_like(freq)
        return zeros, zeros, zeros, zeros, zeros

    w0 = 2.0 * np.pi * freq / float(sr)
    w0 = np.clip(w0, 1e-4, np.pi * 0.995)
    alpha = np.sin(w0) / (2.0 * np.maximum(Q, _RBJ_MIN_Q))
    b0 = alpha
    b1 = np.zeros_like(alpha)
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_process_timevarying(
    x: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
) -> np.ndarray:
    """Process ``x`` with per-sample biquad coefficients."""

    x = _ensure_array(x).astype(np.float64, copy=False)
    length = len(x)
    if not (len(b0) == len(b1) == len(b2) == len(a1) == len(a2) == length):
        raise ValueError('Coefficient tracks must match the source length')

    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(length):
        xi = x[i]
        yi = b0[i] * xi + b1[i] * x1 + b2[i] * x2 - a1[i] * y1 - a2[i] * y2
        y[i] = yi
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi
    return y


def _apply_time_varying_formant_filters(
    src: np.ndarray,
    formants: np.ndarray,
    bws: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Apply band-pass filters whose centre and bandwidth vary per-sample."""

    x = _ensure_array(src).astype(np.float64, copy=False)
    if x.size == 0:
        return x

    if formants.shape != bws.shape:
        raise ValueError('Formant and bandwidth trajectories must share a shape')

    if formants.shape[0] != x.size:
        raise ValueError('Formant trajectories must match the source length')

    total = np.zeros_like(x)
    max_freq = float(sr) * _NYQUIST_SAFETY
    max_bw = float(sr) * _FORMANT_MAX_BW_RATIO

    for idx in range(formants.shape[1]):
        base_freq = float(formants[0, idx])
        freq_track = np.nan_to_num(
            formants[:, idx],
            nan=base_freq,
            posinf=max_freq,
            neginf=_FORMANT_MIN_FREQ_HZ,
        )
        freq_track = np.clip(freq_track, _FORMANT_MIN_FREQ_HZ, max_freq)

        base_bw = float(bws[0, idx])
        bw_track = np.nan_to_num(
            bws[:, idx],
            nan=base_bw,
            posinf=max_bw,
            neginf=_FORMANT_MIN_BW_HZ,
        )
        bw_track = np.clip(bw_track, _FORMANT_MIN_BW_HZ, max_bw)

        Q = np.maximum(freq_track / np.maximum(bw_track, _FORMANT_MIN_BW_HZ), _RBJ_MIN_Q)
        b0, b1, b2, a1, a2 = _bandpass_biquad_coeff_track(freq_track, Q, sr)
        total += _biquad_process_timevarying(x, b0, b1, b2, a1, a2)

    return _lip_radiation(total.astype(DTYPE, copy=False))


def _one_pole_lp(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """Simple one-pole low-pass filter."""
    x = _ensure_array(x)
    if len(x) == 0:
        return x
    decay = exp(-2.0 * pi * cutoff / sr)
    y = np.empty_like(x)
    s = 0.0
    for i, xi in enumerate(x):
        s = (1 - decay) * xi + decay * s
        y[i] = s
    return y


def _lip_radiation(x: np.ndarray) -> np.ndarray:
    """Approximate lip radiation by differentiation."""
    x = _ensure_array(x)
    n = len(x)
    if n == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - x[:-1]
    return y


def _apply_formant_filters(src: np.ndarray, formants: Sequence[float], bws: Sequence[float], sr: int) -> np.ndarray:
    if sr <= 0:
        raise ValueError('Sample rate must be positive')

    formants_arr = np.asarray(formants)
    bws_arr = np.asarray(bws)

    if formants_arr.ndim == 2 or bws_arr.ndim == 2:
        if formants_arr.ndim != 2 or bws_arr.ndim != 2:
            raise ValueError('Formant and bandwidth data must have matching dimensions')
        return _apply_time_varying_formant_filters(src, formants_arr, bws_arr, sr)

    formants_arr = formants_arr.astype(np.float64, copy=False).ravel()
    bws_arr = bws_arr.astype(np.float64, copy=False).ravel()
    if formants_arr.size != bws_arr.size:
        raise ValueError('Formant and bandwidth arrays must share length')

    out = np.zeros_like(_ensure_array(src), dtype=DTYPE)
    max_freq = sr * _NYQUIST_SAFETY
    max_bw = sr * _FORMANT_MAX_BW_RATIO
    for f, bw in zip(formants_arr, bws_arr):
        freq = float(np.clip(f, _FORMANT_MIN_FREQ_HZ, max_freq))
        bw_val = float(np.clip(bw, _FORMANT_MIN_BW_HZ, max_bw))
        Q = max(_RBJ_MIN_Q, freq / max(bw_val, _FORMANT_MIN_BW_HZ))
        b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(freq, Q, sr)
        out += _biquad_process(src, b0, b1, b2, a1, a2)
    return _lip_radiation(out)


def _apply_nasal_antiresonances(
    src: np.ndarray,
    zero_freqs: Sequence[float],
    zero_bws: Sequence[float],
    sr: int,
    *,
    depth: float = 1.0,
) -> np.ndarray:
    src = _ensure_array(src)
    if len(src) == 0:
        return src
    if zero_freqs is None or zero_bws is None:
        return src

    zero_freqs = np.asarray(zero_freqs).ravel()
    zero_bws = np.asarray(zero_bws).ravel()

    if zero_freqs.size == 0 or zero_bws.size == 0:
        return src

    depth = _clamp(depth, 0.0, 1.0)
    if depth <= EPS:
        return src

    filtered = src.copy()
    for fz, bw in zip(zero_freqs, zero_bws):
        if not np.isfinite(fz) or not np.isfinite(bw):
            continue
        if fz <= 0.0 or fz >= sr * 0.48:
            continue
        b0, b1, b2, a1, a2 = _notch_biquad_coeff(fz, bw, sr)
        filtered = _biquad_process(filtered, b0, b1, b2, a1, a2)

    if depth >= 1.0:
        return filtered.astype(DTYPE, copy=False)
    mixed = (1.0 - depth) * src + depth * filtered
    return mixed.astype(DTYPE, copy=False)


def _apply_all_pole_filter(src: np.ndarray, a_coeffs: np.ndarray) -> np.ndarray:
    x = np.asarray(src, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError('src must be one-dimensional')

    order = len(a_coeffs) - 1
    if order <= 0:
        return x.astype(DTYPE, copy=False)

    y = np.empty_like(x, dtype=np.float64)
    for n in range(len(x)):
        acc = x[n]
        for k in range(1, order + 1):
            if n - k >= 0:
                acc -= a_coeffs[k] * y[n - k]
        y[n] = acc
    return y.astype(DTYPE, copy=False)


def _apply_kelly_lochbaum_filter(
    src: np.ndarray,
    area_profile: Sequence[float],
    sr: int,
    *,
    lipReflection: float = -0.85,
    wallLoss: float = 0.996,
    applyRadiation: bool = True,
) -> np.ndarray:
    if len(src) == 0:
        return np.zeros(0, dtype=DTYPE)

    refl = _area_profile_to_reflections(area_profile, lipReflection=lipReflection)
    if wallLoss != 1.0:
        refl = np.clip(refl * float(wallLoss), -0.999, 0.999)

    a_poly = _reflection_to_lpc(refl)
    y = _apply_all_pole_filter(src, a_poly)
    if applyRadiation:
        y = _lip_radiation(y)
    return y


def _apply_breath_noise_coloration(x: np.ndarray, sr: int) -> np.ndarray:
    """Emphasise 2–8 kHz content using HPF→BPF→LPF processing.

    Args:
        x: Noise signal to colour.
        sr: Sample rate associated with ``x``.
    """

    x = _ensure_array(x)
    if len(x) == 0 or sr <= 0:
        return x

    hp_cut = min(_BREATH_NOISE_HPF_CUTOFF_HZ, float(sr) * _NYQUIST_SAFETY)
    bp_center = float(_BREATH_NOISE_BPF_CENTER_HZ)
    lp_cut = min(_BREATH_NOISE_LPF_CUTOFF_HZ, float(sr) * _NYQUIST_SAFETY)

    if bp_center >= float(sr) * 0.5:
        bp_center = float(sr) * _NYQUIST_SAFETY

    y = x
    b0, b1, b2, a1, a2 = _highpass_biquad_coeff(hp_cut, _BREATH_NOISE_HPF_Q, sr)
    y = _biquad_process(y, b0, b1, b2, a1, a2)

    b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(bp_center, _BREATH_NOISE_BPF_Q, sr)
    y = _biquad_process(y, b0, b1, b2, a1, a2)

    b0, b1, b2, a1, a2 = _lowpass_biquad_coeff(lp_cut, _BREATH_NOISE_LPF_Q, sr)
    y = _biquad_process(y, b0, b1, b2, a1, a2)

    return y.astype(DTYPE, copy=False)


def _pre_emphasis(x: np.ndarray, coefficient: float = 0.85) -> np.ndarray:
    x = _ensure_array(x)
    n = len(x)
    if n == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coefficient * x[:-1]
    return y
