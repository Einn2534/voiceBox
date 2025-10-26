"""Excitation source generators."""
from __future__ import annotations

from math import pi
from typing import Tuple

import numpy as np

from .constants import DTYPE
from .core import _db_to_lin
from .filters import _bandpass_biquad_coeff, _biquad_process, _lip_radiation

__all__ = ["_glottal_source", "_gen_band_noise"]


def _glottal_source(
    f0: float,
    dur_s: float,
    sr: int,
    jitter_cents: float = 10.0,
    shimmer_db: float = 0.8,
) -> np.ndarray:
    """Saw-like glottal source with simple jitter/shimmer."""
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
    cents = (jitter_cents / 100.0) * js
    inst_f = f0 * (2.0 ** (cents / 12.0))

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
