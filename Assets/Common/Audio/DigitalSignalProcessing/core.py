# Created on 2024-08-28
# Created by ChatGPT
# Description: Core numeric helpers shared across the DSP modules.
"""Core numeric helpers shared across the DSP modules."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .constants import DTYPE, PEAK_DEFAULT, EPS

__all__ = [
    "_clamp",
    "_clamp01",
    "_ms_to_samples",
    "_db_to_lin",
    "_ensure_array",
    "_normalize_peak",
    "_apply_fade",
    "_add_breath_noise",
    "_soft_limit",
]


_BREATH_GATE_SHAPE_POWER = 2.0
_BREATH_GATE_MIN_FREQUENCY = 10.0
_SOFT_LIMIT_MIN_DRIVE_DB = -24.0
_SOFT_LIMIT_MAX_DRIVE_DB = 24.0


def _soft_limit(sig: np.ndarray, drive_db: float = 0.0) -> np.ndarray:
    """Apply an arctangent soft limiter with configurable drive."""

    sig = _ensure_array(sig)
    if sig.size == 0:
        return sig

    drive_db = float(np.clip(drive_db, _SOFT_LIMIT_MIN_DRIVE_DB, _SOFT_LIMIT_MAX_DRIVE_DB))
    drive = 10.0 ** (drive_db / 20.0)
    driven = drive * sig
    limited = np.arctan(driven) / np.arctan(drive)
    return limited.astype(DTYPE, copy=False)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _ms_to_samples(ms: float, sr: int) -> int:
    """Convert milliseconds to a number of samples (floor, min 0)."""
    if ms <= 0:
        return 0
    return int(sr * (ms / 1000.0))


def _db_to_lin(db: float) -> float:
    """Convert dB to linear gain."""
    return 10.0 ** (db / 20.0)


def _ensure_array(x: np.ndarray, *, dtype=DTYPE) -> np.ndarray:
    """Ensure ``x`` is of the requested dtype and contiguous."""
    if x.dtype != dtype:
        return x.astype(dtype, copy=False)
    return x


def _normalize_peak(sig: np.ndarray, target: float = PEAK_DEFAULT) -> np.ndarray:
    """Apply peak normalisation (guarding against silence)."""
    sig = _ensure_array(sig)
    peak = float(np.max(np.abs(sig)) + EPS)
    scale = target / peak if peak > 0 else 1.0
    return (sig * scale).astype(DTYPE, copy=False)


def _apply_fade(
    sig: np.ndarray,
    sr: int,
    *,
    attack_ms: float = 5.0,
    release_ms: float = 8.0,
) -> np.ndarray:
    """Apply linear attack/release fades."""
    sig = _ensure_array(sig)
    n = len(sig)
    if n == 0:
        return sig
    out = sig.copy()

    a = _ms_to_samples(attack_ms, sr)
    r = _ms_to_samples(release_ms, sr)
    if 0 < a < n:
        out[:a] *= np.linspace(0.0, 1.0, a, dtype=DTYPE)
    if 0 < r < n:
        out[-r:] *= np.linspace(1.0, 0.0, r, dtype=DTYPE)
    return out


def _add_breath_noise(
    sig: np.ndarray,
    level_db: float,
    sr: Optional[int] = None,
    *,
    f0_track: Optional[Union[float, np.ndarray]] = None,
    hnr_target_db: Optional[float] = None,
    harmonic: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add breath noise using HNR and glottal gating when available.

    Args:
        sig: Harmonic component that receives the noise contribution.
        level_db: Legacy breath level in dB relative to ``sig`` RMS (negative).
        sr: Sample rate for noise coloration and gating.
        f0_track: Fundamental frequency in Hz (scalar or per-sample array).
        hnr_target_db: Desired harmonic-to-noise ratio in dB.
        harmonic: Optional reference used to measure harmonic power.
    """

    def _compute_gate(n_samples: int) -> Optional[np.ndarray]:
        if sr is None or sr <= 0:
            return None
        if f0_track is None:
            return None
        if isinstance(f0_track, (int, float)):
            freqs = np.full(n_samples, float(f0_track), dtype=np.float64)
        else:
            arr = np.asarray(f0_track, dtype=np.float64).ravel()
            if arr.size == 0:
                return None
            if arr.size != n_samples:
                pos = np.linspace(0.0, 1.0, arr.size, endpoint=False, dtype=np.float64)
                tgt = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=np.float64)
                arr = np.interp(tgt, pos, arr, left=arr[0], right=arr[-1])
            freqs = arr
        freqs = np.clip(freqs, 0.0, float(sr) * 0.5)
        if np.all(freqs <= _BREATH_GATE_MIN_FREQUENCY):
            return None
        phase_inc = 2.0 * np.pi * freqs / float(sr)
        phase = np.cumsum(phase_inc, dtype=np.float64)
        gate = np.maximum(0.0, np.sin(phase))
        if np.max(gate) <= EPS:
            return None
        shaped = gate ** _BREATH_GATE_SHAPE_POWER
        peak = float(np.max(shaped))
        if peak <= EPS:
            return None
        return (shaped / peak).astype(DTYPE, copy=False)

    sig = _ensure_array(sig)
    n = len(sig)
    if n == 0:
        return sig

    rng = np.random.default_rng()
    noise = rng.standard_normal(n).astype(DTYPE)

    if sr is not None and sr > 0:
        from .filters import _apply_breath_noise_coloration

        noise = _apply_breath_noise_coloration(noise, sr)

    gate = _compute_gate(n)
    if gate is not None:
        noise = noise * gate

    harmonic_ref = _ensure_array(harmonic) if harmonic is not None else sig
    harmonic_power = float(
        np.mean(np.asarray(harmonic_ref, dtype=np.float64) ** 2) + EPS
    )

    target_rms: Optional[float] = None
    if hnr_target_db is not None and np.isfinite(hnr_target_db):
        ratio = 10.0 ** (hnr_target_db / 10.0)
        ratio = max(ratio, EPS)
        noise_power = harmonic_power / ratio
        target_rms = float(np.sqrt(max(noise_power, 0.0)))
    elif level_db < 0.0:
        harmonic_rms = float(np.sqrt(harmonic_power))
        target_rms = harmonic_rms * _db_to_lin(level_db)

    if target_rms is None or target_rms <= 0.0:
        return sig

    noise_rms = float(
        np.sqrt(np.mean(np.asarray(noise, dtype=np.float64) ** 2) + EPS)
    )
    if noise_rms <= 0.0:
        return sig

    scaled_noise = noise * (target_rms / noise_rms)
    return (sig + scaled_noise).astype(DTYPE, copy=False)
