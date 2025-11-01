# Created on 2024-06-08
# Created by ChatGPT
"""High-level synthesis routines built on the DSP helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .articulation import (
    _estimate_nasal_formants,
    _nasal_branch_zeros,
    _resample_profile,
    _sanitize_nasal_coupling,
    area_profile_to_formants,
    generate_kelly_lochbaum_profile,
)
from .constants import (
    CV_TOKEN_MAP,
    GLIDE_ONSETS,
    LIQUID_ONSETS,
    NASAL_PRESETS,
    NASAL_TOKEN_MAP,
    PAUSE_TOKEN,
    PEAK_DEFAULT,
    VOWEL_TABLE,
    _NEUTRAL_TRACT_AREA_CM2,
    DTYPE,
    EPS,
    NasalCoupling,
    SpeakerProfile,
)
from .core import (
    _add_breath_noise,
    _apply_fade,
    _clamp,
    _clamp01,
    _ms_to_samples,
    _normalize_peak,
)
from .filters import (
    _apply_formant_filters,
    _apply_kelly_lochbaum_filter,
    _apply_nasal_antiresonances,
    _one_pole_lp,
    _pre_emphasis,
)
from .io import write_wav
from .sources import (
    DEFAULT_DRIFT_CENTS,
    DEFAULT_DRIFT_RETURN_RATE,
    DEFAULT_TREMOR_DEPTH_CENTS,
    DEFAULT_TREMOR_FREQUENCY_HZ,
    DEFAULT_VIBRATO_DEPTH_CENTS,
    DEFAULT_VIBRATO_FREQUENCY_HZ,
    _gen_band_noise,
    _glottal_source,
)

@dataclass(frozen=True)
class TokenProsody:
    """Container for per-token prosodic overrides used during synthesis."""

    f0: Optional[float] = None
    vowelMilliseconds: Optional[float] = None
    consonantMilliseconds: Optional[float] = None
    preMilliseconds: Optional[float] = None
    overlapMilliseconds: Optional[float] = None
    gapMilliseconds: Optional[float] = None
    durationScale: Optional[float] = None


_MIN_VOWEL_DURATION_MS = 120.0
_MIN_SCALED_VOWEL_MS = 40.0
_MIN_NASAL_DURATION_MS = 80.0
_NASAL_DURATION_RATIO = 0.6
_PAUSE_MULTIPLIER = 3.0
_MIN_PAUSE_DURATION_MS = 120.0


__all__ = [
    "TokenProsody",
    "synth_vowel",
    "synth_fricative",
    "synth_plosive",
    "synth_affricate",
    "synth_nasal",
    "synth_cv",
    "synth_vowel_with_onset",
    "synth_cv_to_wav",
    "synth_phrase_to_wav",
    "synth_token_sequence",
    "synth_tokens_to_wav",
]


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, *, overlap_ms: float = 30.0) -> np.ndarray:
    if len(a) == 0:
        return b.astype(DTYPE, copy=False)
    if len(b) == 0:
        return a.astype(DTYPE, copy=False)
    overlap = max(0, _ms_to_samples(overlap_ms, sr))
    if overlap == 0:
        return np.concatenate([a, b]).astype(DTYPE, copy=False)
    left = a[:-overlap] if overlap < len(a) else np.zeros(0, dtype=DTYPE)
    right = b[overlap:] if overlap < len(b) else np.zeros(0, dtype=DTYPE)
    fade_out = np.linspace(1.0, 0.0, overlap, dtype=DTYPE)
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=DTYPE)
    blended = a[-overlap:] * fade_out + b[:overlap] * fade_in
    return np.concatenate([left, blended, right]).astype(DTYPE, copy=False)


def _synth_vowel_fixed(
    formants: Sequence[float],
    bws: Sequence[float],
    f0: float,
    dur_s: float,
    sr: int,
    *,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
    breathHnrDb: Optional[float] = None,
    driftCents: float = DEFAULT_DRIFT_CENTS,
    driftReturnRate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibratoDepthCents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibratoFrequencyHz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremorDepthCents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremorFrequencyHz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
    areaProfile: Optional[Sequence[float]] = None,
    articulation: Optional[Dict[str, float]] = None,
    kellySections: Optional[int] = None,
    useLegacyFormantFilter: bool = True,
    waveguideLipReflection: float = -0.85,
    waveguideWallLoss: float = 0.996,
) -> np.ndarray:
    """Synthesize a steady-state vowel for the provided tract configuration.

    Args:
        formants: Target formant frequencies in Hz.
        bws: Corresponding bandwidths in Hz.
        f0: Base fundamental frequency in Hz.
        dur_s: Duration in seconds.
        sr: Sample rate in Hz.
        jitterCents: Amount of fast pitch variation in cents.
        shimmerDb: Amplitude modulation depth in dB.
        breathLevelDb: Legacy breath level relative to harmonic RMS.
        breathHnrDb: Optional HNR target used to derive noise gain.
        driftCents: Slow pitch drift depth in cents.
        driftReturnRate: Rate at which drift recenters.
        vibratoDepthCents: Vibrato depth in cents.
        vibratoFrequencyHz: Vibrato speed in Hz.
        tremorDepthCents: Tremor depth in cents.
        tremorFrequencyHz: Tremor speed in Hz.
        areaProfile: Optional custom area function for Kelly-Lochbaum.
        articulation: Optional articulation overrides for waveguide.
        kellySections: Number of waveguide sections to use.
        useLegacyFormantFilter: Toggle between formant and waveguide model.
        waveguideLipReflection: Lip reflection coefficient for waveguide.
        waveguideWallLoss: Wall loss factor for waveguide.
    """

    src = _glottal_source(
        f0,
        dur_s,
        sr,
        jitterCents,
        shimmerDb,
        drift_cents=driftCents,
        drift_return_rate=driftReturnRate,
        vibrato_depth_cents=vibratoDepthCents,
        vibrato_frequency_hz=vibratoFrequencyHz,
        tremor_depth_cents=tremorDepthCents,
        tremor_frequency_hz=tremorFrequencyHz,
    )
    formants = np.asarray(formants, dtype=np.float64)
    bws = np.asarray(bws, dtype=np.float64)
    if useLegacyFormantFilter:
        y = _apply_formant_filters(src, formants, bws, sr)
    else:
        sections = int(kellySections) if kellySections else len(_NEUTRAL_TRACT_AREA_CM2)
        profile: Optional[np.ndarray]
        if areaProfile is not None:
            profile = np.asarray(areaProfile, dtype=np.float64)
            if profile.ndim != 1:
                raise ValueError('areaProfile must be one-dimensional')
            if sections and len(profile) != sections:
                profile = _resample_profile(profile.astype(np.float64), sections)
        elif articulation is not None:
            profile = generate_kelly_lochbaum_profile(numSections=sections, articulation=articulation or {})
        else:
            profile = generate_kelly_lochbaum_profile(numSections=sections, articulation={})
        y = _apply_kelly_lochbaum_filter(
            src,
            profile,
            sr,
            lipReflection=waveguideLipReflection,
            wallLoss=waveguideWallLoss,
            applyRadiation=True,
        )
    y = _add_breath_noise(
        y,
        breathLevelDb,
        sr,
        f0_track=f0,
        hnr_target_db=breathHnrDb,
    )
    return _normalize_peak(y, PEAK_DEFAULT)


def synth_vowel(
    vowel: str = 'a',
    f0: float = 120.0,
    durationSeconds: float = 1.0,
    sampleRate: int = 22050,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
    breathHnrDb: Optional[float] = None,
    *,
    driftCents: float = DEFAULT_DRIFT_CENTS,
    driftReturnRate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibratoDepthCents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibratoFrequencyHz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremorDepthCents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremorFrequencyHz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
    kellyBlend: Optional[float] = None,
    articulation: Optional[Dict[str, float]] = None,
    areaProfile: Optional[Sequence[float]] = None,
    kellySections: Optional[int] = None,
    useLegacyFormantFilter: bool = True,
    waveguideLipReflection: float = -0.85,
    waveguideWallLoss: float = 0.996,
    speakerProfile: Optional[SpeakerProfile] = None,
) -> np.ndarray:
    """Synthesize a vowel tone using the requested articulatory settings.

    Args:
        vowel: Target vowel symbol.
        f0: Fundamental frequency in Hz.
        durationSeconds: Output duration in seconds.
        sampleRate: Rendering sample rate in Hz.
        jitterCents: Fast pitch jitter amount in cents.
        shimmerDb: Amplitude jitter in dB.
        breathLevelDb: Legacy breath gain relative to harmonics.
        breathHnrDb: Optional HNR target driving breath noise gain.
        driftCents: Slow pitch drift amount in cents.
        driftReturnRate: Re-centering rate for pitch drift.
        vibratoDepthCents: Vibrato depth in cents.
        vibratoFrequencyHz: Vibrato frequency in Hz.
        tremorDepthCents: Tremor depth in cents.
        tremorFrequencyHz: Tremor frequency in Hz.
        kellyBlend: Blend factor between formant filter and waveguide.
        articulation: Optional articulation overrides for waveguide.
        areaProfile: Custom vocal tract area function.
        kellySections: Number of sections for waveguide.
        useLegacyFormantFilter: Flag selecting legacy formant filtering.
        waveguideLipReflection: Lip reflection coefficient for waveguide.
        waveguideWallLoss: Wall damping factor for waveguide.
        speakerProfile: Optional speaker profile overrides.
    """
    assert vowel in VOWEL_TABLE, f"unsupported vowel: {vowel}"
    src = _glottal_source(
        f0,
        durationSeconds,
        sampleRate,
        jitterCents,
        shimmerDb,
        drift_cents=driftCents,
        drift_return_rate=driftReturnRate,
        vibrato_depth_cents=vibratoDepthCents,
        vibrato_frequency_hz=vibratoFrequencyHz,
        tremor_depth_cents=tremorDepthCents,
        tremor_frequency_hz=tremorFrequencyHz,
    )
    spec = VOWEL_TABLE[vowel]

    if kellyBlend is None:
        blend = 1.0 if (articulation is not None or areaProfile is not None) else 0.0
    else:
        blend = float(kellyBlend)
    blend = _clamp(blend, 0.0, 1.0)
    formants = np.array(spec['F'], dtype=np.float64)
    bws = np.array(spec['BW'], dtype=np.float64)

    nasal_zeros: Tuple[np.ndarray, np.ndarray] = (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    nasal_leak_depth = 0.0
    if speakerProfile is not None:
        coupling = _sanitize_nasal_coupling(speakerProfile.nasal_coupling)
        leak = coupling.vowel_leak
        if leak > EPS:
            base_port = coupling.port_open
            effective_port = leak if base_port <= EPS else leak * base_port
            zeros, zero_bw = _nasal_branch_zeros(coupling, sampleRate, port_override=effective_port)
            if len(zeros) > 0:
                nasal_zeros = (zeros, zero_bw)
                nasal_leak_depth = _clamp(leak * (0.7 + 0.3 * base_port), 0.0, 1.0)

    k_sections = int(kellySections) if kellySections else len(_NEUTRAL_TRACT_AREA_CM2)
    profile_custom: Optional[np.ndarray] = None
    if areaProfile is not None:
        profile_custom = np.asarray(areaProfile, dtype=np.float64)
        if profile_custom.ndim != 1:
            raise ValueError('areaProfile must be one-dimensional')
        if k_sections and len(profile_custom) != k_sections:
            profile_custom = _resample_profile(profile_custom.astype(np.float64), k_sections)
    elif articulation is not None:
        profile_custom = generate_kelly_lochbaum_profile(numSections=k_sections, articulation=articulation or {})

    new_formants: Optional[np.ndarray] = None
    new_bw: Optional[np.ndarray] = None
    if profile_custom is not None:
        try:
            new_formants, new_bw = area_profile_to_formants(
                profile_custom,
                nFormants=len(formants),
                lipReflection=waveguideLipReflection,
                wallLoss=waveguideWallLoss,
            )
        except Exception:
            new_formants, new_bw = None, None

    if new_formants is not None and len(new_formants) > 0 and new_bw is not None:
        if len(new_formants) < len(formants):
            pad = len(formants) - len(new_formants)
            new_formants = np.pad(new_formants, (0, pad), mode='edge')
            new_bw = np.pad(new_bw, (0, pad), mode='edge')
        elif len(new_formants) > len(formants):
            new_formants = new_formants[:len(formants)]
            new_bw = new_bw[:len(formants)]
        formants = (1.0 - blend) * formants + blend * new_formants
        bws = (1.0 - blend) * bws + blend * new_bw

    if useLegacyFormantFilter:
        y = _apply_formant_filters(src, formants, bws, sampleRate)
    else:
        neutral_profile: Optional[np.ndarray] = None

        def _neutral_profile() -> np.ndarray:
            nonlocal neutral_profile
            if neutral_profile is None:
                neutral_profile = generate_kelly_lochbaum_profile(numSections=k_sections, articulation={})
            return neutral_profile

        if profile_custom is None:
            profile_waveguide = _neutral_profile()
        else:
            if blend <= 0.0:
                profile_waveguide = _neutral_profile()
            elif blend >= 1.0:
                profile_waveguide = profile_custom
            else:
                profile_waveguide = np.clip((1.0 - blend) * _neutral_profile() + blend * profile_custom, 0.05, None)
        y = _apply_kelly_lochbaum_filter(
            src,
            profile_waveguide,
            sampleRate,
            lipReflection=waveguideLipReflection,
            wallLoss=waveguideWallLoss,
            applyRadiation=True,
        )
    y = _add_breath_noise(
        y,
        breathLevelDb,
        sampleRate,
        f0_track=f0,
        hnr_target_db=breathHnrDb,
    )
    if nasal_leak_depth > EPS and nasal_zeros[0].size > 0:
        y = _apply_nasal_antiresonances(
            y,
            nasal_zeros[0],
            nasal_zeros[1],
            sampleRate,
            depth=nasal_leak_depth,
        )
    return _normalize_peak(y, PEAK_DEFAULT)


def synth_fricative(
    consonant: str = 's',
    durationSeconds: float = 0.16,
    sampleRate: int = 22050,
    levelDb: float = -12.0,
) -> np.ndarray:
    c = consonant.lower()
    if c == 's':
        center, Q, lip, lp = 6500.0, 3.0, True, None
    elif c == 'sh':
        center, Q, lip, lp = 3800.0, 2.4, True, 4200.0
    elif c == 'h':
        center, Q, lip, lp = 1800.0, 1.4, False, 2300.0
    elif c == 'f':
        center, Q, lip, lp = 950.0, 1.3, False, 2100.0
    else:
        raise ValueError("synth_fricative: supported consonants are 's','sh','h','f'.")

    y = _gen_band_noise(durationSeconds, sampleRate, center, Q, use_lip_radiation=lip)
    if lp is not None:
        y = _one_pole_lp(y, cutoff=lp, sr=sampleRate)
    if c == 'h':
        y = _one_pole_lp(y, cutoff=1700.0, sr=sampleRate)

    attack = 6 if c in ('h', 'f') else 4
    release = 16 if c in ('h', 'f') else 12
    y = _apply_fade(y, sampleRate, attack_ms=attack, release_ms=release)

    peak = 0.6 if c not in ('h', 'f') else (0.5 if c == 'h' else 0.55)
    y = _normalize_peak(y, peak)
    gain = 10.0 ** (levelDb / 20.0)
    if gain != 1.0:
        y = (y * gain).astype(DTYPE, copy=False)
    return y


def synth_plosive(
    consonant: str = 't',
    sampleRate: int = 22050,
    closureMilliseconds: Optional[float] = None,
    burstMilliseconds: Optional[float] = None,
    aspirationMilliseconds: Optional[float] = None,
    levelDb: float = -10.0,
) -> np.ndarray:
    c = consonant.lower()
    if c not in ('t', 'k'):
        raise ValueError("synth_plosive: supported only 't' or 'k'.")

    if c == 't':
        closure = 24 if closureMilliseconds is None else closureMilliseconds
        burst = 12 if burstMilliseconds is None else burstMilliseconds
        asp = 18 if aspirationMilliseconds is None else aspirationMilliseconds
        b_center, b_Q = 4500.0, 1.2
        a_center, a_Q = 6000.0, 0.9
    else:
        closure = 32 if closureMilliseconds is None else closureMilliseconds
        burst = 14 if burstMilliseconds is None else burstMilliseconds
        asp = 24 if aspirationMilliseconds is None else aspirationMilliseconds
        b_center, b_Q = 1700.0, 1.2
        a_center, a_Q = 2500.0, 0.9

    closure_seg = np.zeros(_ms_to_samples(closure, sampleRate), dtype=DTYPE)

    burst_len = _ms_to_samples(burst, sampleRate)
    burst_noise = _gen_band_noise(burst_len / sampleRate, sampleRate, b_center, b_Q, use_lip_radiation=False)
    tau = max(1.0, burst_len / 4.0)
    env = np.exp(-np.arange(burst_len, dtype=DTYPE) / tau)
    burst_seg = (burst_noise[:burst_len] * env).astype(DTYPE, copy=False)

    asp_len = _ms_to_samples(asp, sampleRate)
    asp_noise = _gen_band_noise(asp_len / sampleRate, sampleRate, a_center, a_Q, use_lip_radiation=False)
    asp_env = np.exp(-np.linspace(0.0, 1.0, max(1, asp_len), dtype=DTYPE) * 3.5)
    aspiration_seg = (asp_noise[:asp_len] * asp_env).astype(DTYPE, copy=False)

    release = _apply_fade(burst_seg, sampleRate, attack_ms=2.0, release_ms=4.0)
    aspiration_seg = _apply_fade(aspiration_seg, sampleRate, attack_ms=1.0, release_ms=asp)

    seq = [closure_seg, release, aspiration_seg]
    y = np.concatenate(seq).astype(DTYPE, copy=False)
    gain = 10.0 ** (levelDb / 20.0)
    if gain != 1.0:
        y = (y * gain).astype(DTYPE, copy=False)
    return _normalize_peak(y, 0.65 if c == 't' else 0.7)


def synth_affricate(
    consonant: str = 'ch',
    sampleRate: int = 22050,
    closureMilliseconds: Optional[float] = None,
    fricativeMilliseconds: Optional[float] = None,
    levelDb: float = -11.0,
) -> np.ndarray:
    c = consonant.lower()
    if c not in ('ch', 'ts'):
        raise ValueError("synth_affricate: supported only 'ch' or 'ts'.")

    if c == 'ch':
        closure = 26 if closureMilliseconds is None else closureMilliseconds
        burst_ms = 10
        fric_ms = 95.0 if fricativeMilliseconds is None else fricativeMilliseconds
        fric = synth_fricative('sh', durationSeconds=fric_ms / 1000.0, sampleRate=sampleRate, levelDb=-16.0)
    else:
        closure = 24 if closureMilliseconds is None else closureMilliseconds
        burst_ms = 9
        fric_ms = 90.0 if fricativeMilliseconds is None else fricativeMilliseconds
        fric = synth_fricative('s', durationSeconds=fric_ms / 1000.0, sampleRate=sampleRate, levelDb=-16.0)

    plosive = synth_plosive(
        't',
        sampleRate=sampleRate,
        closureMilliseconds=closure,
        burstMilliseconds=burst_ms,
        aspirationMilliseconds=0.0,
        levelDb=levelDb,
    )
    y = _crossfade(plosive, fric, sampleRate, overlap_ms=14)
    return _normalize_peak(y, 0.6)


def synth_nasal(
    consonant: str = 'n',
    f0: float = 120.0,
    durationMilliseconds: float = 90.0,
    sampleRate: int = 22050,
    *,
    nasalCoupling: Optional[NasalCoupling] = None,
    speakerProfile: Optional[SpeakerProfile] = None,
    portOpen: Optional[float] = None,
    nostrilAreaCm2: Optional[float] = None,
    breathLevelDb: float = -38.0,
) -> np.ndarray:
    c = consonant.lower()
    if c not in NASAL_PRESETS:
        raise ValueError("synth_nasal: supported nasals are 'n' or 'm'.")

    dur_s = max(20.0, float(durationMilliseconds)) / 1000.0
    coupling = nasalCoupling
    if coupling is None and speakerProfile is not None:
        coupling = speakerProfile.nasal_coupling
    coupling = _sanitize_nasal_coupling(coupling)
    if portOpen is not None:
        coupling = _sanitize_nasal_coupling(replace(coupling, port_open=_clamp01(portOpen)))
    if nostrilAreaCm2 is not None:
        coupling = _sanitize_nasal_coupling(
            replace(coupling, nostril_area_cm2=max(0.1, float(nostrilAreaCm2)))
        )

    formants, bws = _estimate_nasal_formants(c, coupling)
    zeros, zero_bw = _nasal_branch_zeros(coupling, sampleRate)

    breath_level = float(breathLevelDb)
    breath_level += (1.0 - coupling.port_open) * 6.0
    y = _synth_vowel_fixed(
        formants,
        bws,
        f0,
        dur_s,
        sampleRate,
        jitterCents=4.0,
        shimmerDb=0.4,
        breathLevelDb=breath_level,
    )
    depth = _clamp(0.75 + 0.2 * coupling.port_open, 0.0, 1.0)
    if zeros.size > 0:
        y = _apply_nasal_antiresonances(y, zeros, zero_bw, sampleRate, depth=depth)

    y = _apply_fade(
        y,
        sampleRate,
        attack_ms=8.0 if c == 'n' else 10.0,
        release_ms=22.0 if c == 'n' else 28.0,
    )
    gain = 0.6 if c == 'n' else 0.68
    gain += 0.06 * (coupling.port_open - 0.8)
    gain = _clamp(gain, 0.45, 0.82)
    return _normalize_peak(y * gain, 0.5)


def synth_vowel_with_onset(
    vowel: str,
    f0: float,
    sampleRate: int,
    totalMilliseconds: int = 240,
    onsetMilliseconds: int = 45,
    onsetFormants: Optional[Sequence[float]] = None,
    onsetBandwidthScale: float = 0.85,
    vowelModel: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    spec = VOWEL_TABLE[vowel]
    targetF, targetBW = spec['F'], spec['BW']
    vowel_kwargs = dict(vowelModel or {})
    vowel_kwargs.setdefault('jitterCents', 6.0)
    vowel_kwargs.setdefault('shimmerDb', 0.6)
    vowel_kwargs.setdefault('breathLevelDb', -40.0)
    vowel_kwargs.setdefault('breathHnrDb', None)
    vowel_kwargs.setdefault('driftCents', DEFAULT_DRIFT_CENTS)
    vowel_kwargs.setdefault('driftReturnRate', DEFAULT_DRIFT_RETURN_RATE)
    vowel_kwargs.setdefault('vibratoDepthCents', DEFAULT_VIBRATO_DEPTH_CENTS)
    vowel_kwargs.setdefault('vibratoFrequencyHz', DEFAULT_VIBRATO_FREQUENCY_HZ)
    vowel_kwargs.setdefault('tremorDepthCents', DEFAULT_TREMOR_DEPTH_CENTS)
    vowel_kwargs.setdefault('tremorFrequencyHz', DEFAULT_TREMOR_FREQUENCY_HZ)
    waveguide_opts = {
        'areaProfile': vowel_kwargs.get('areaProfile'),
        'articulation': vowel_kwargs.get('articulation'),
        'kellySections': vowel_kwargs.get('kellySections'),
        'useLegacyFormantFilter': vowel_kwargs.get('useLegacyFormantFilter', True),
        'waveguideLipReflection': vowel_kwargs.get('waveguideLipReflection', -0.85),
        'waveguideWallLoss': vowel_kwargs.get('waveguideWallLoss', 0.996),
        'jitterCents': vowel_kwargs['jitterCents'],
        'shimmerDb': vowel_kwargs['shimmerDb'],
        'breathLevelDb': vowel_kwargs['breathLevelDb'],
        'breathHnrDb': vowel_kwargs['breathHnrDb'],
        'driftCents': vowel_kwargs['driftCents'],
        'driftReturnRate': vowel_kwargs['driftReturnRate'],
        'vibratoDepthCents': vowel_kwargs['vibratoDepthCents'],
        'vibratoFrequencyHz': vowel_kwargs['vibratoFrequencyHz'],
        'tremorDepthCents': vowel_kwargs['tremorDepthCents'],
        'tremorFrequencyHz': vowel_kwargs['tremorFrequencyHz'],
    }

    if not onsetFormants:
        return synth_vowel(
            vowel=vowel,
            f0=f0,
            durationSeconds=totalMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )

    total_ms = int(max(10, totalMilliseconds))
    onset_ms = int(max(1, min(onsetMilliseconds, total_ms - 1)))
    sustain_ms = max(0, total_ms - onset_ms)

    onsetBW = [bw * float(onsetBandwidthScale) for bw in targetBW]
    onset = _synth_vowel_fixed(
        onsetFormants,
        onsetBW,
        f0,
        onset_ms / 1000.0,
        sampleRate,
        **waveguide_opts,
    )
    onset = _apply_fade(onset, sampleRate, attack_ms=6.0, release_ms=min(12.0, onset_ms * 0.5))

    if sustain_ms <= 0:
        return onset

    ov_ms = min(max(6.0, onset_ms * 0.45), 14.0, float(sustain_ms))
    rest_len_ms = sustain_ms + max(0.0, ov_ms)

    if vowel_kwargs:
        sustain = synth_vowel(
            vowel=vowel,
            f0=f0,
            durationSeconds=rest_len_ms / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
    else:
        sustain = _synth_vowel_fixed(
            targetF,
            targetBW,
            f0,
            rest_len_ms / 1000.0,
            sampleRate,
            **waveguide_opts,
        )
    sustain = _apply_fade(sustain, sampleRate, attack_ms=4.0, release_ms=12.0)

    return _crossfade(onset, sustain, sampleRate, overlap_ms=ov_ms)


def synth_cv(
    cons: str,
    vowel: str,
    f0: float = 120.0,
    sampleRate: int = 22050,
    preMilliseconds: int = 0,
    consonantMilliseconds: Optional[int] = None,
    vowelMilliseconds: int = 240,
    overlapMilliseconds: int = 30,
    useOnsetTransition: bool = False,
    vowelModel: Optional[Dict[str, Any]] = None,
    speakerProfile: Optional[SpeakerProfile] = None,
) -> np.ndarray:
    c = cons.lower()
    v = vowel.lower()
    if v not in VOWEL_TABLE:
        raise ValueError("Unknown vowel for synth_cv")

    head = np.zeros(_ms_to_samples(preMilliseconds, sampleRate), dtype=DTYPE)
    vowel_kwargs = dict(vowelModel or {})
    vowel_kwargs.setdefault('breathHnrDb', None)
    if speakerProfile is not None:
        vowel_kwargs.setdefault('speakerProfile', speakerProfile)
    speaker_profile: Optional[SpeakerProfile] = vowel_kwargs.get('speakerProfile')
    nasal_coupling = speaker_profile.nasal_coupling if speaker_profile is not None else None

    if c in ('', 'pau', PAUSE_TOKEN):
        vowel_kwargs.setdefault('durationSeconds', vowelMilliseconds / 1000.0)
        vow = synth_vowel(vowel=v, f0=f0, sampleRate=sampleRate, **vowel_kwargs)
        return _normalize_peak(np.concatenate([head, vow]).astype(DTYPE), PEAK_DEFAULT)

    if c == 's':
        cons_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 90.0
        s_part = synth_fricative('s', durationSeconds=cons_ms / 1000.0, sampleRate=sampleRate, levelDb=-16.0)
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(s_part, vow, sampleRate, overlap_ms=max(18, overlapMilliseconds))

    elif c == 'sh':
        cons_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 110.0
        s_part = synth_fricative('sh', durationSeconds=cons_ms / 1000.0, sampleRate=sampleRate, levelDb=-18.0)
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(s_part, vow, sampleRate, overlap_ms=max(24, overlapMilliseconds))

    elif c == 'h':
        cons_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 80.0
        s_part = synth_fricative('h', durationSeconds=cons_ms / 1000.0, sampleRate=sampleRate, levelDb=-14.0)
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(s_part, vow, sampleRate, overlap_ms=max(18, overlapMilliseconds))

    elif c == 'f':
        cons_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 90.0
        s_part = synth_fricative('f', durationSeconds=cons_ms / 1000.0, sampleRate=sampleRate, levelDb=-14.0)
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(s_part, vow, sampleRate, overlap_ms=max(20, overlapMilliseconds))

    elif c == 't':
        plosive_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 60.0
        tap = synth_plosive('t', sampleRate=sampleRate, levelDb=-12.0)
        tail = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=(vowelMilliseconds + plosive_ms) / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(tap, tail, sampleRate, overlap_ms=max(16, overlapMilliseconds))

    elif c == 'k':
        plosive_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 72.0
        tap = synth_plosive('k', sampleRate=sampleRate, levelDb=-11.0)
        tail = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=(vowelMilliseconds + plosive_ms) / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(tap, tail, sampleRate, overlap_ms=max(18, overlapMilliseconds))

    elif c == 'ch':
        cons_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 120.0
        aff = synth_affricate('ch', sampleRate=sampleRate, closureMilliseconds=cons_ms * 0.35)
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(aff, vow, sampleRate, overlap_ms=max(25, overlapMilliseconds))

    elif c == 'ts':
        cons_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 120.0
        aff = synth_affricate('ts', sampleRate=sampleRate, closureMilliseconds=cons_ms * 0.3)
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(aff, vow, sampleRate, overlap_ms=max(25, overlapMilliseconds))

    elif c == 'n':
        nasal_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 90.0
        nasal = synth_nasal(
            'n',
            f0=f0,
            durationMilliseconds=nasal_ms,
            sampleRate=sampleRate,
            nasalCoupling=nasal_coupling,
            speakerProfile=speaker_profile,
        )
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(nasal, vow, sampleRate, overlap_ms=max(25, overlapMilliseconds))

    elif c == 'm':
        nasal_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 110.0
        nasal = synth_nasal(
            'm',
            f0=f0,
            durationMilliseconds=nasal_ms,
            sampleRate=sampleRate,
            nasalCoupling=nasal_coupling,
            speakerProfile=speaker_profile,
        )
        vow = synth_vowel(
            vowel=v,
            f0=f0,
            durationSeconds=vowelMilliseconds / 1000.0,
            sampleRate=sampleRate,
            **vowel_kwargs,
        )
        out = _crossfade(nasal, vow, sampleRate, overlap_ms=max(28, overlapMilliseconds))

    elif c == 'w':
        onset_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 44.0
        onset_ms = max(24.0, min(onset_ms, float(vowelMilliseconds) - 8.0))
        onsetF = GLIDE_ONSETS['w'].get(v, GLIDE_ONSETS['w']['a'])
        out = synth_vowel_with_onset(
            v,
            f0,
            sampleRate,
            totalMilliseconds=vowelMilliseconds,
            onsetMilliseconds=int(onset_ms),
            onsetFormants=onsetF,
            onsetBandwidthScale=1.18,
            vowelModel=vowel_kwargs,
        )
        out = _pre_emphasis(out, coefficient=0.86)
        out = _add_breath_noise(
            out,
            level_db=-36.0,
            sr=sampleRate,
            f0_track=f0,
        )
        out = _normalize_peak(out, PEAK_DEFAULT)

    elif c == 'y':
        onset_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 40.0
        onset_ms = max(24.0, min(onset_ms, float(vowelMilliseconds) - 10.0))
        onsetF = GLIDE_ONSETS['y'].get(v, GLIDE_ONSETS['y']['a'])
        out = synth_vowel_with_onset(
            v,
            f0,
            sampleRate,
            totalMilliseconds=vowelMilliseconds,
            onsetMilliseconds=int(onset_ms),
            onsetFormants=onsetF,
            onsetBandwidthScale=1.10,
            vowelModel=vowel_kwargs,
        )
        out = _pre_emphasis(out, coefficient=0.84)
        out = _add_breath_noise(
            out,
            level_db=-38.0,
            sr=sampleRate,
            f0_track=f0,
        )
        out = _normalize_peak(out, PEAK_DEFAULT)

    elif c == 'r':
        tap = synth_plosive('t', sampleRate=sampleRate, closureMilliseconds=12.0, burstMilliseconds=6.0,
                            aspirationMilliseconds=4.0, levelDb=-20.0)
        onsetF = LIQUID_ONSETS.get(v, LIQUID_ONSETS['a'])
        vow = synth_vowel_with_onset(
            v,
            f0,
            sampleRate,
            totalMilliseconds=vowelMilliseconds,
            onsetMilliseconds=36,
            onsetFormants=onsetF,
            vowelModel=vowel_kwargs,
        )
        out = _crossfade(tap, vow, sampleRate, overlap_ms=12)

    else:
        raise ValueError("synth_cv: unsupported consonant for synth_cv")

    combined = np.concatenate([head, out]).astype(DTYPE)
    return _normalize_peak(combined, PEAK_DEFAULT)


def synth_cv_to_wav(
    cons: str,
    vowel: str,
    outPath: str,
    f0: float = 120.0,
    sampleRate: int = 22050,
    preMilliseconds: int = 0,
    consonantMilliseconds: Optional[int] = None,
    vowelMilliseconds: int = 240,
    overlapMilliseconds: int = 30,
    useOnsetTransition: bool = False,
    vowelModel: Optional[Dict[str, Any]] = None,
    speakerProfile: Optional[SpeakerProfile] = None,
) -> str:
    waveform = synth_cv(
        cons,
        vowel,
        f0=f0,
        sampleRate=sampleRate,
        preMilliseconds=preMilliseconds,
        consonantMilliseconds=consonantMilliseconds,
        vowelMilliseconds=vowelMilliseconds,
        overlapMilliseconds=overlapMilliseconds,
        useOnsetTransition=useOnsetTransition,
        vowelModel=vowelModel,
        speakerProfile=speakerProfile,
    )
    return write_wav(outPath, waveform, sampleRate=sampleRate)


def synth_phrase_to_wav(
    vowels: Sequence[str],
    outPath: str,
    f0: float = 120.0,
    unitMilliseconds: int = 220,
    gapMilliseconds: int = 30,
    sampleRate: int = 22050,
    vowelModel: Optional[Dict[str, Any]] = None,
    speakerProfile: Optional[SpeakerProfile] = None,
) -> str:
    chunks: List[np.ndarray] = []
    vowel_kwargs = dict(vowelModel or {})
    if speakerProfile is not None:
        vowel_kwargs.setdefault('speakerProfile', speakerProfile)

    for v in map(str.lower, vowels):
        if v in VOWEL_TABLE:
            seg = synth_vowel(
                v,
                f0=f0,
                durationSeconds=unitMilliseconds / 1000.0,
                sampleRate=sampleRate,
                **vowel_kwargs,
            )
            chunks.append(seg)
            chunks.append(np.zeros(_ms_to_samples(gapMilliseconds, sampleRate), dtype=DTYPE))
    if not chunks:
        chunks = [np.zeros(int(sampleRate * 0.3), dtype=DTYPE)]
    y = np.concatenate(chunks)
    return write_wav(outPath, y, sampleRate=sampleRate)


def synth_token_sequence(
    tokens: Sequence[str],
    *,
    f0: float = 120.0,
    sampleRate: int = 22050,
    vowelMilliseconds: int = 240,
    overlapMilliseconds: int = 30,
    gapMilliseconds: int = 40,
    useOnsetTransition: bool = False,
    vowelModel: Optional[Dict[str, Any]] = None,
    speakerProfile: Optional[SpeakerProfile] = None,
    tokenProsody: Optional[Sequence[Optional[TokenProsody]]] = None,
) -> np.ndarray:
    """Synthesize a token sequence with optional per-token prosody overrides.

    Args:
        tokens: Phonetic tokens to synthesize.
        f0: Baseline pitch in Hertz.
        sampleRate: Target sampling rate.
        vowelMilliseconds: Default vowel duration in milliseconds.
        overlapMilliseconds: Default consonant-vowel overlap in milliseconds.
        gapMilliseconds: Default silence inserted between tokens in milliseconds.
        useOnsetTransition: Whether to synthesize vowel onsets explicitly.
        vowelModel: Optional vowel synthesis overrides.
        speakerProfile: Speaker profile used for nasal coupling and timbre.
        tokenProsody: Optional overrides for pitch and timing per token index.
    """

    segs: List[np.ndarray] = []
    default_gap_ms = max(0.0, float(gapMilliseconds))
    default_vowel_ms = max(_MIN_VOWEL_DURATION_MS, float(vowelMilliseconds))

    vowel_kwargs = dict(vowelModel or {})
    if speakerProfile is not None:
        vowel_kwargs.setdefault('speakerProfile', speakerProfile)
    speaker_profile: Optional[SpeakerProfile] = vowel_kwargs.get('speakerProfile')
    nasal_coupling = speaker_profile.nasal_coupling if speaker_profile is not None else None

    for idx, tok in enumerate(tokens):
        t = tok.strip().lower()
        if not t:
            continue

        prosody = None
        if tokenProsody is not None and idx < len(tokenProsody):
            prosody = tokenProsody[idx]

        token_f0 = float(f0) if prosody is None or prosody.f0 is None else float(prosody.f0)

        token_vowel_ms = default_vowel_ms
        if prosody is not None and prosody.vowelMilliseconds is not None:
            token_vowel_ms = max(_MIN_SCALED_VOWEL_MS, float(prosody.vowelMilliseconds))
        elif prosody is not None and prosody.durationScale is not None:
            token_vowel_ms = max(
                _MIN_SCALED_VOWEL_MS,
                default_vowel_ms * max(0.05, float(prosody.durationScale)),
            )

        token_overlap_ms = int(overlapMilliseconds)
        if prosody is not None and prosody.overlapMilliseconds is not None:
            token_overlap_ms = max(0, int(prosody.overlapMilliseconds))

        token_gap_ms = default_gap_ms
        if prosody is not None and prosody.gapMilliseconds is not None:
            token_gap_ms = max(0.0, float(prosody.gapMilliseconds))
        elif prosody is not None and prosody.durationScale is not None:
            token_gap_ms = max(0.0, default_gap_ms * max(0.0, float(prosody.durationScale)))

        token_gap_samples = _ms_to_samples(int(token_gap_ms), sampleRate)

        if t == PAUSE_TOKEN:
            pause_ms = max(default_gap_ms * _PAUSE_MULTIPLIER, _MIN_PAUSE_DURATION_MS)
            if prosody is not None:
                if prosody.vowelMilliseconds is not None:
                    pause_ms = max(0.0, float(prosody.vowelMilliseconds))
                elif prosody.durationScale is not None and prosody.gapMilliseconds is None:
                    pause_ms = max(
                        0.0,
                        pause_ms * max(0.0, float(prosody.durationScale)),
                    )
                elif prosody.gapMilliseconds is not None:
                    pause_ms = max(0.0, float(prosody.gapMilliseconds))
            segs.append(np.zeros(_ms_to_samples(int(pause_ms), sampleRate), dtype=DTYPE))
            continue

        if t in CV_TOKEN_MAP:
            ck, vk = CV_TOKEN_MAP[t]
            cv_kwargs: Dict[str, Any] = {
                'f0': token_f0,
                'sampleRate': sampleRate,
                'vowelMilliseconds': int(token_vowel_ms),
                'overlapMilliseconds': token_overlap_ms,
                'useOnsetTransition': useOnsetTransition,
                'vowelModel': vowel_kwargs,
                'speakerProfile': speaker_profile,
            }
            if prosody is not None and prosody.preMilliseconds is not None:
                cv_kwargs['preMilliseconds'] = max(0, int(prosody.preMilliseconds))
            if prosody is not None and prosody.consonantMilliseconds is not None:
                cv_kwargs['consonantMilliseconds'] = max(0, int(prosody.consonantMilliseconds))
            seg = synth_cv(ck, vk, **cv_kwargs)

        elif t in VOWEL_TABLE:
            seg = synth_vowel(
                t,
                f0=token_f0,
                durationSeconds=max(_MIN_SCALED_VOWEL_MS, token_vowel_ms) / 1000.0,
                sampleRate=sampleRate,
                **vowel_kwargs,
            )

        elif t in NASAL_TOKEN_MAP:
            nasal_ms = max(
                _MIN_NASAL_DURATION_MS,
                int(token_vowel_ms * _NASAL_DURATION_RATIO),
            )
            if prosody is not None and prosody.consonantMilliseconds is not None:
                nasal_ms = max(_MIN_NASAL_DURATION_MS, int(prosody.consonantMilliseconds))
            seg = synth_nasal(
                NASAL_TOKEN_MAP[t],
                f0=token_f0,
                durationMilliseconds=nasal_ms,
                sampleRate=sampleRate,
                nasalCoupling=nasal_coupling,
                speakerProfile=speaker_profile,
            )
        else:
            raise ValueError(f"Unsupported token '{tok}' for synthesis")

        if segs and token_gap_samples > 0:
            segs.append(np.zeros(token_gap_samples, dtype=DTYPE))

        segs.append(seg.astype(DTYPE, copy=False))

    if not segs:
        return np.zeros(_ms_to_samples(int(_MIN_PAUSE_DURATION_MS), sampleRate), dtype=DTYPE)

    y = np.concatenate(segs).astype(DTYPE, copy=False)
    return _normalize_peak(y, PEAK_DEFAULT)


def synth_tokens_to_wav(
    tokens: Sequence[str],
    outPath: str,
    *,
    f0: float = 120.0,
    sampleRate: int = 22050,
    vowelMilliseconds: int = 240,
    overlapMilliseconds: int = 30,
    gapMilliseconds: int = 40,
    useOnsetTransition: bool = False,
    vowelModel: Optional[Dict[str, Any]] = None,
    speakerProfile: Optional[SpeakerProfile] = None,
    tokenProsody: Optional[Sequence[Optional[TokenProsody]]] = None,
) -> str:
    """Render a token sequence to disk while allowing per-token prosody overrides.

    Args:
        tokens: Phonetic tokens to synthesize.
        outPath: Output WAV file path.
        f0: Baseline pitch in Hertz.
        sampleRate: Target sampling rate.
        vowelMilliseconds: Default vowel duration in milliseconds.
        overlapMilliseconds: Default consonant-vowel overlap in milliseconds.
        gapMilliseconds: Default silence inserted between tokens in milliseconds.
        useOnsetTransition: Whether to synthesize vowel onsets explicitly.
        vowelModel: Optional vowel synthesis overrides.
        speakerProfile: Speaker profile used for nasal coupling and timbre.
        tokenProsody: Optional overrides for pitch and timing per token index.
    """

    y = synth_token_sequence(
        tokens,
        f0=f0,
        sampleRate=sampleRate,
        vowelMilliseconds=vowelMilliseconds,
        overlapMilliseconds=overlapMilliseconds,
        gapMilliseconds=gapMilliseconds,
        useOnsetTransition=useOnsetTransition,
        vowelModel=vowelModel,
        speakerProfile=speakerProfile,
        tokenProsody=tokenProsody,
    )
    return write_wav(outPath, y, sampleRate=sampleRate)
