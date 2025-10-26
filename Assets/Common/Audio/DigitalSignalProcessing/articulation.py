"""Vocal-tract and nasal coupling helpers."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    DEFAULT_TRACT_SECTIONS,
    SPEED_OF_SOUND_CM_S,
    VOCAL_TRACT_LENGTH_CM,
    _NEUTRAL_TRACT_AREA_CM2,
    EPS,
    NasalCoupling,
)
from .core import _clamp, _clamp01

__all__ = [
    "_sanitize_nasal_coupling",
    "_nasal_branch_zeros",
    "_estimate_nasal_formants",
    "_resample_profile",
    "generate_kelly_lochbaum_profile",
    "_reflection_to_lpc",
    "_area_profile_to_reflections",
    "area_profile_to_formants",
]


def _sanitize_nasal_coupling(coupling: Optional[NasalCoupling]) -> NasalCoupling:
    base = coupling or NasalCoupling()
    return replace(
        base,
        port_open=_clamp01(base.port_open),
        vowel_leak=_clamp01(base.vowel_leak),
        nostril_area_cm2=max(0.1, float(base.nostril_area_cm2)),
        nasal_cavity_length_cm=max(5.0, float(base.nasal_cavity_length_cm)),
        nasal_cavity_area_cm2=max(0.5, float(base.nasal_cavity_area_cm2)),
        sinus_cavity_length_cm=max(1.0, float(base.sinus_cavity_length_cm)),
        sinus_cavity_area_cm2=max(0.1, float(base.sinus_cavity_area_cm2)),
        sinus_coupling_area_cm2=max(0.05, float(base.sinus_coupling_area_cm2)),
        loss_db_per_meter=max(0.0, float(base.loss_db_per_meter)),
    )


def _nasal_branch_zeros(
    coupling: NasalCoupling,
    sr: int,
    *,
    port_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = _sanitize_nasal_coupling(coupling)
    if port_override is not None:
        cfg = replace(cfg, port_open=_clamp01(port_override))
    port = cfg.port_open
    if sr <= 0 or port <= EPS:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    eff_len = max(5.0, cfg.nasal_cavity_length_cm + 0.35 * cfg.nostril_area_cm2)
    base_freq = SPEED_OF_SOUND_CM_S / (4.0 * eff_len)
    area_ratio = cfg.nasal_cavity_area_cm2 / max(cfg.nostril_area_cm2, 0.1)
    damping = cfg.loss_db_per_meter * 38.0

    zeros: List[float] = []
    bws: List[float] = []
    for idx in range(3):
        freq = base_freq * (2 * idx + 1)
        if freq >= sr * 0.48:
            break
        q = max(1.8, area_ratio * (1.35 + 1.25 * port))
        bw = freq / q + damping
        zeros.append(freq)
        bws.append(bw)

    if cfg.sinus_coupling_area_cm2 > 0.05 and cfg.sinus_cavity_area_cm2 > 0.05:
        eff_sinus_len = max(
            1.0,
            cfg.sinus_cavity_length_cm
            + 0.18 * (cfg.sinus_cavity_area_cm2 / max(cfg.sinus_coupling_area_cm2, 0.05)),
        )
        sinus_freq = SPEED_OF_SOUND_CM_S / (4.0 * eff_sinus_len)
        if sinus_freq < sr * 0.48:
            q_s = max(
                1.6,
                (cfg.sinus_cavity_area_cm2 / max(cfg.sinus_coupling_area_cm2, 0.05))
                * (1.05 + 0.6 * port),
            )
            bw_s = sinus_freq / q_s + damping * 0.8
            zeros.append(sinus_freq)
            bws.append(bw_s)

    if not zeros:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    zero_arr = np.asarray(zeros, dtype=np.float64)
    bw_arr = np.asarray(bws, dtype=np.float64)
    order = np.argsort(zero_arr)
    return zero_arr[order], bw_arr[order]


def _estimate_nasal_formants(
    consonant: str,
    coupling: NasalCoupling,
    *,
    port_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = _sanitize_nasal_coupling(coupling)
    if port_override is not None:
        cfg = replace(cfg, port_open=_clamp01(port_override))
    port = cfg.port_open

    nasal_len = max(6.0, cfg.nasal_cavity_length_cm + 0.6 * cfg.nostril_area_cm2)
    base_freq = SPEED_OF_SOUND_CM_S / (4.0 * nasal_len)
    scale = 1.4 + 1.1 * port + 0.03 * cfg.nostril_area_cm2
    f1 = base_freq / scale
    oral_front = 4.3 if consonant == 'n' else 5.6
    oral_mid = 3.1 if consonant == 'n' else 3.9
    scale2 = 2.0 + 0.8 * (1.0 - port) + 0.03 * cfg.nasal_cavity_area_cm2
    scale3 = 2.05 + 0.6 * (1.0 - port) + 0.02 * (
        cfg.nasal_cavity_area_cm2 + cfg.sinus_coupling_area_cm2
    )
    f2 = SPEED_OF_SOUND_CM_S / (2.0 * oral_front * scale2)
    f3 = SPEED_OF_SOUND_CM_S / (2.0 * oral_mid * scale3)

    if consonant == 'm':
        f1 *= 0.92
        f2 *= 0.88
        scale_offset = 1.0 + 0.04 * cfg.nostril_area_cm2
        f3 *= 0.94 / scale_offset

    f1 = float(np.clip(f1, 150.0, 450.0))
    f2 = float(np.clip(f2, 1100.0, 2300.0))
    f3 = float(np.clip(f3, 2000.0, 3200.0))

    loss_term = cfg.loss_db_per_meter * 22.0
    bw1 = 70.0 + 130.0 * port + loss_term
    bw2 = 150.0 + 120.0 * port + loss_term * 0.6
    bw3 = 210.0 + 110.0 * port + loss_term * 0.5
    if consonant == 'm':
        bw1 += 12.0
        bw2 += 8.0

    return (
        np.array([f1, f2, f3], dtype=np.float64),
        np.array([bw1, bw2, bw3], dtype=np.float64),
    )


def _resample_profile(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) == size:
        return values.copy()
    positions = np.linspace(0.0, len(values) - 1.0, num=len(values))
    target = np.linspace(0.0, len(values) - 1.0, num=size)
    return np.interp(target, positions, values).astype(np.float64)


def generate_kelly_lochbaum_profile(
    *,
    numSections: int = DEFAULT_TRACT_SECTIONS,
    articulation: Optional[Dict[str, float]] = None,
    jaw: float = 0.0,
    tongueBody: float = 0.0,
    tongueTip: float = 0.0,
    lipHeight: float = 0.0,
    lipProtrusion: float = 0.0,
    pharynx: float = 0.0,
    smoothing: bool = True,
) -> np.ndarray:
    """Generate a Kelly-Lochbaum area profile from simplified articulation controls."""

    params = {
        'jaw': jaw,
        'tongueBody': tongueBody,
        'tongueTip': tongueTip,
        'lipHeight': lipHeight,
        'lipProtrusion': lipProtrusion,
        'pharynx': pharynx,
    }
    if articulation:
        params.update(articulation)

    params = {k: _clamp(float(v), -1.0, 1.0) for k, v in params.items()}

    base = _resample_profile(_NEUTRAL_TRACT_AREA_CM2, max(2, int(numSections)))
    sections = len(base)
    idx = np.arange(sections, dtype=np.float64)

    areas = base.copy()

    if params['jaw'] != 0.0:
        areas *= np.exp(0.35 * params['jaw'])

    if params['pharynx'] != 0.0:
        weight = np.linspace(1.0, 0.2, sections)
        areas *= np.exp(-0.6 * params['pharynx'] * weight)

    if params['tongueBody'] != 0.0:
        center = sections * (0.45 + 0.1 * params['tongueBody'])
        width = max(1.5, 3.5 - params['tongueBody'] * 1.0)
        bump = np.exp(-0.5 * ((idx - center) / width) ** 2)
        areas *= 1.0 - 0.35 * params['tongueBody'] * bump

    if params['tongueTip'] != 0.0:
        center = sections * 0.72
        width = 2.2
        bump = np.exp(-0.5 * ((idx - center) / width) ** 2)
        areas *= 1.0 - 0.25 * params['tongueTip'] * bump

    if params['lipHeight'] != 0.0:
        weights = np.clip((idx - (sections - 3)) / 3.0, 0.0, 1.0)
        areas *= 1.0 - 0.5 * params['lipHeight'] * weights

    if params['lipProtrusion'] != 0.0:
        protrude = 1.0 - 0.4 * params['lipProtrusion']
        areas[-3:] *= protrude

    areas = np.clip(areas, 0.05, None)

    if smoothing and sections > 2:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        padded = np.pad(areas, (1, 1), mode='edge')
        areas = np.convolve(padded, kernel, mode='valid')

    return areas.astype(np.float64, copy=False)


def _reflection_to_lpc(reflection: Iterable[float]) -> np.ndarray:
    a = np.array([1.0], dtype=np.float64)
    for k in reflection:
        k = float(_clamp(k, -0.999, 0.999))
        m = len(a)
        a_new = np.empty(m + 1, dtype=np.float64)
        a_new[0] = 1.0
        for i in range(1, m):
            a_new[i] = a[i] + k * a[m - i]
        a_new[m] = k
        a = a_new
    return a


def _area_profile_to_reflections(
    area_profile: Sequence[float],
    *,
    lipReflection: float = -0.85,
) -> np.ndarray:
    """Convert a Kelly-Lochbaum area profile to reflection coefficients."""

    areas = np.asarray(area_profile, dtype=np.float64)
    if areas.ndim != 1 or len(areas) < 2:
        raise ValueError('area_profile must be a 1-D sequence with >=2 elements')

    refl: List[float] = []
    for left, right in zip(areas[:-1], areas[1:]):
        denom = left + right
        if denom <= EPS:
            refl.append(0.0)
        else:
            refl.append((right - left) / denom)
    refl.append(float(_clamp(lipReflection, -0.999, 0.0)))
    return np.asarray(refl, dtype=np.float64)


def area_profile_to_formants(
    area_profile: Sequence[float],
    *,
    nFormants: int = 3,
    lipReflection: float = -0.85,
    wallLoss: float = 0.996,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate formant frequencies/bandwidths from a Kelly-Lochbaum area profile."""

    areas = np.asarray(area_profile, dtype=np.float64)
    if areas.ndim != 1 or len(areas) < 2:
        raise ValueError('area_profile must be a 1-D sequence with >=2 elements')

    refl = _area_profile_to_reflections(areas, lipReflection=lipReflection)

    a_poly = _reflection_to_lpc(refl)
    roots = np.roots(a_poly)
    roots = roots[np.abs(roots) > EPS]
    roots = roots[np.imag(roots) >= 0.0]

    section_len_cm = VOCAL_TRACT_LENGTH_CM / float(len(areas))
    fs = SPEED_OF_SOUND_CM_S / (2.0 * section_len_cm)

    freq = np.angle(roots) * fs / (2.0 * np.pi)
    damp = np.abs(roots) * wallLoss
    bw = -fs * np.log(np.clip(damp, EPS, 0.9999)) / np.pi

    valid = np.isfinite(freq) & np.isfinite(bw) & (freq > 0.0) & (bw > 0.0)
    freq = freq[valid]
    bw = bw[valid]

    if len(freq) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    order = np.argsort(freq)
    freq = freq[order][:nFormants]
    bw = bw[order][:nFormants]
    return freq.astype(np.float64), bw.astype(np.float64)
