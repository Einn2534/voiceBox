"""Core numeric helpers shared across the DSP modules."""
from __future__ import annotations

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
]


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


def _add_breath_noise(sig: np.ndarray, level_db: float) -> np.ndarray:
    """Add a breath noise component when ``level_db`` is negative."""
    rng = np.random.default_rng()
    sig = _ensure_array(sig)
    if level_db >= 0 or len(sig) == 0:
        return sig
    noise = rng.standard_normal(len(sig)).astype(DTYPE)
    rms = float(np.sqrt(np.mean(sig * sig) + EPS))
    target = rms * _db_to_lin(level_db)
    n_rms = float(np.sqrt(np.mean(noise * noise) + EPS))
    if n_rms > 0:
        sig = sig + noise * (target / n_rms)
    return sig.astype(DTYPE, copy=False)
