# Created on 2024-09-19
# Created by ChatGPT
# Description: High-level synthesis routines built on the DSP helpers.
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
    FormantOuPhaseParams,
    FormantOuSettings,
    FormantPerturbationParams,
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
    _soft_limit,
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
    DEFAULT_AM_OU_CLIP_MULTIPLE,
    DEFAULT_AM_OU_SIGMA,
    DEFAULT_AM_OU_TAU,
    DEFAULT_DRIFT_CENTS,
    DEFAULT_DRIFT_RETURN_RATE,
    DEFAULT_SHIMMER_PERCENT,
    DEFAULT_TREMOLO_DEPTH_DB,
    DEFAULT_TREMOLO_FREQUENCY_HZ,
    DEFAULT_TREMOR_DEPTH_CENTS,
    DEFAULT_TREMOR_FREQUENCY_HZ,
    DEFAULT_VIBRATO_DEPTH_CENTS,
    DEFAULT_VIBRATO_FREQUENCY_HZ,
    GlottalSourceResult,
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
    f0MultiplierTrack: Optional[Sequence[float]] = None
    amplitudeMultiplierTrack: Optional[Sequence[float]] = None
    formantOffsetTrack: Optional[Sequence[Sequence[float]]] = None
    formantTargetTrack: Optional[Sequence[Sequence[float]]] = None
    breathLevelOffsetTrack: Optional[Sequence[float]] = None
    breathHnrTrack: Optional[Sequence[float]] = None


_MIN_VOWEL_DURATION_MS = 120.0
_MIN_SCALED_VOWEL_MS = 40.0
_MIN_NASAL_DURATION_MS = 80.0
_NASAL_DURATION_RATIO = 0.6
_PAUSE_MULTIPLIER = 3.0
_MIN_PAUSE_DURATION_MS = 120.0
_DURATION_OU_THETA = 0.45
_DURATION_OU_SIGMA = 0.018
_DURATION_JITTER_STD = 0.006
_DURATION_SCALE_LIMIT = 0.08
_PHRASE_SCALE_MIN = 0.6
_PHRASE_SCALE_MAX = 1.4
_INHALE_MIN_MS = 120.0
_INHALE_ACTIVE_RATIO = 0.65
_INHALE_ATTACK_RATIO = 0.3
_INHALE_LEVEL_DB = -28.0
_INHALE_SILENCE_RATIO = 0.2
_INHALE_NOISE_CUTOFF_HZ = 2600.0
_CONSONANT_BASE_DURATION_MS: Dict[str, float] = {
    's': 90.0,
    'sh': 110.0,
    'h': 80.0,
    'f': 90.0,
    't': 60.0,
    'k': 72.0,
    'ch': 120.0,
    'ts': 120.0,
    'n': 90.0,
    'm': 110.0,
    'w': 44.0,
    'y': 40.0,
}


__all__ = [
    "DynamicControl",
    "FrameControlFrame",
    "SegmentControlPlan",
    "LowRateControlTargets",
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


@dataclass(frozen=True)
class DynamicControl:
    """Optional per-sample control tracks applied during synthesis."""

    f0Multiplier: Optional[np.ndarray] = None
    amplitudeMultiplier: Optional[np.ndarray] = None
    formantOffsetHz: Optional[np.ndarray] = None
    formantTargetHz: Optional[np.ndarray] = None
    breathLevelOffsetDb: Optional[np.ndarray] = None
    breathHnrOffsetDb: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FrameControlFrame:
    """Frame-level control snapshot combining targets and stochastic drift."""

    startSample: int
    endSample: int
    f0Multiplier: float
    amplitudeMultiplier: float
    formantOffsetHz: np.ndarray
    formantTargetHz: np.ndarray
    breathLevelOffsetDb: float
    breathHnrOffsetDb: float


@dataclass(frozen=True)
class SegmentControlPlan:
    """Frame-wise plan paired with the per-sample dynamic control tracks."""

    totalSamples: int
    dynamicControl: DynamicControl
    frames: Tuple[FrameControlFrame, ...]


@dataclass(frozen=True)
class LowRateControlTargets:
    """Target trajectories anchoring the stochastic low-rate controller."""

    f0Multiplier: Optional[Sequence[float]] = None
    amplitudeMultiplier: Optional[Sequence[float]] = None
    formantOffsetHz: Optional[Sequence[Sequence[float]]] = None
    formantTargetHz: Optional[Sequence[Sequence[float]]] = None
    breathLevelOffsetDb: Optional[Sequence[float]] = None
    breathHnrOffsetDb: Optional[Sequence[float]] = None


_CONTROL_FRAME_RATE_HZ = 200.0
_CONTROL_SMOOTHING_MS = 22.0
_CONTROL_FORMANT_COUNT = 3
_CONTROL_F0_SIGMA_CENTS = 6.0
_CONTROL_F0_TAU_MS = 220.0
_CONTROL_F0_CLIP_MULTIPLE = 3.0
_CONTROL_AMP_SIGMA_DB = 1.5
_CONTROL_AMP_TAU_MS = 260.0
_CONTROL_AMP_CLIP_MULTIPLE = 3.0
_CONTROL_FORMANT_SIGMA_HZ = 40.0
_CONTROL_FORMANT_TAU_MS = 200.0
_CONTROL_FORMANT_CLIP_MULTIPLE = 3.0
_CONTROL_BREATH_SIGMA_DB = 2.4
_CONTROL_BREATH_TAU_MS = 320.0
_CONTROL_BREATH_CLIP_MULTIPLE = 3.5
_CONTROL_HNR_SIGMA_DB = 2.0
_CONTROL_HNR_TAU_MS = 340.0
_CONTROL_HNR_CLIP_MULTIPLE = 3.0
_CONTROL_MIN_AMP = 0.35
_CONTROL_MAX_AMP = 2.6
_CONTROL_MIN_F0_MULT = 0.6
_CONTROL_MAX_F0_MULT = 1.6
_FINAL_PEAK_TARGET = 10.0 ** (-1.0 / 20.0)
_SOFT_LIMIT_DRIVE_DB = 1.5
_BASE_F0_MULTIPLIER = 1.0
_BASE_AMPLITUDE_MULTIPLIER = 1.0
_BASE_FORMANT_OFFSET_HZ = 0.0


def _frame_length_samples(sample_rate: int, frame_rate: float) -> int:
    """Return the integer frame length in samples for the low-rate controller."""

    if sample_rate <= 0:
        return 0
    if frame_rate <= 0.0:
        return sample_rate
    length = int(round(sample_rate / float(frame_rate)))
    return max(1, length)


def _match_control_track(track: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
    """Resize ``track`` to ``target_len`` samples using linear interpolation."""

    if track is None:
        return None
    if target_len <= 0:
        return None
    arr = np.asarray(track, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size == target_len:
            return arr.astype(np.float64, copy=False)
        src_pos = np.linspace(0.0, 1.0, arr.size, dtype=np.float64, endpoint=True)
        dst_pos = np.linspace(0.0, 1.0, target_len, dtype=np.float64, endpoint=False)
        resized = np.interp(dst_pos, src_pos, arr)
        return resized.astype(np.float64, copy=False)
    if arr.ndim == 2:
        cols: List[np.ndarray] = []
        for column in arr.T:
            resized_column = _match_control_track(column, target_len)
            if resized_column is None:
                continue
            cols.append(resized_column)
        if not cols:
            return None
        stacked = np.stack(cols, axis=1)
        return stacked.astype(np.float64, copy=False)
    raise ValueError("Control track must be one- or two-dimensional")


def _build_frame_control_plan(
    sample_count: int,
    sample_rate: int,
    control: DynamicControl,
    frame_rate: float,
    formant_count: int,
) -> Tuple[FrameControlFrame, ...]:
    """Aggregate per-sample controls into ~5 ms frame snapshots."""

    if sample_count <= 0 or sample_rate <= 0:
        return tuple()

    frame_length = _frame_length_samples(sample_rate, frame_rate)
    if frame_length <= 0:
        return tuple()

    f0_track = _match_control_track(control.f0Multiplier, sample_count)
    if f0_track is None:
        f0_track = np.full(sample_count, _BASE_F0_MULTIPLIER, dtype=np.float64)

    amp_track = _match_control_track(control.amplitudeMultiplier, sample_count)
    if amp_track is None:
        amp_track = np.full(sample_count, _BASE_AMPLITUDE_MULTIPLIER, dtype=np.float64)

    formant_track = _match_control_track(control.formantOffsetHz, sample_count)
    if formant_track is None:
        formant_track = np.full(
            (sample_count, max(formant_count, 1)),
            _BASE_FORMANT_OFFSET_HZ,
            dtype=np.float64,
        )
    elif formant_track.ndim == 1:
        formant_track = formant_track[:, None]

    formant_target_track = _match_control_track(control.formantTargetHz, sample_count)
    if formant_target_track is None:
        formant_target_track = np.zeros((sample_count, max(formant_count, 1)), dtype=np.float64)
    elif formant_target_track.ndim == 1:
        formant_target_track = formant_target_track[:, None]

    active_formants = max(0, min(int(formant_count), formant_track.shape[1]))
    if active_formants == 0:
        formant_track = np.zeros((sample_count, 0), dtype=np.float64)
        formant_target_track = np.zeros((sample_count, 0), dtype=np.float64)
    else:
        formant_track = formant_track[:, :active_formants]
        formant_target_track = formant_target_track[:, :active_formants]

    breath_track = _match_control_track(control.breathLevelOffsetDb, sample_count)
    if breath_track is None:
        breath_track = np.zeros(sample_count, dtype=np.float64)

    hnr_track = _match_control_track(control.breathHnrOffsetDb, sample_count)
    if hnr_track is None:
        hnr_track = np.zeros(sample_count, dtype=np.float64)

    frames: List[FrameControlFrame] = []
    for start in range(0, sample_count, frame_length):
        end = min(start + frame_length, sample_count)
        if end <= start:
            continue
        frame_slice = slice(start, end)
        frame_f0 = float(np.mean(f0_track[frame_slice]))
        frame_amp = float(np.mean(amp_track[frame_slice]))
        if formant_track.size > 0:
            frame_formant = np.mean(formant_track[frame_slice, :], axis=0)
            frame_formant_targets = np.mean(formant_target_track[frame_slice, :], axis=0)
        else:
            frame_formant = np.zeros((0,), dtype=np.float64)
            frame_formant_targets = np.zeros((0,), dtype=np.float64)
        frame_breath_offset = float(np.mean(breath_track[frame_slice]))
        frame_hnr_offset = float(np.mean(hnr_track[frame_slice]))
        frames.append(
            FrameControlFrame(
                startSample=int(start),
                endSample=int(end),
                f0Multiplier=frame_f0,
                amplitudeMultiplier=frame_amp,
                formantOffsetHz=frame_formant.astype(np.float64, copy=False),
                formantTargetHz=frame_formant_targets.astype(np.float64, copy=False),
                breathLevelOffsetDb=frame_breath_offset,
                breathHnrOffsetDb=frame_hnr_offset,
            )
        )

    return tuple(frames)


def _smooth_control_track(track: np.ndarray, sample_rate: int, smoothing_ms: float) -> np.ndarray:
    """Apply single-pole smoothing to the supplied control track."""

    if track.size == 0 or sample_rate <= 0 or smoothing_ms <= 0.0:
        return track
    tau_seconds = max(float(smoothing_ms), 0.0) / 1000.0
    if tau_seconds <= 0.0:
        return track
    alpha = float(np.exp(-1.0 / (tau_seconds * float(sample_rate))))
    alpha = np.clip(alpha, 0.0, 0.999999)
    smoothed = np.empty_like(track, dtype=np.float64)
    if track.ndim == 1:
        state = float(track[0])
        smoothed[0] = state
        for index in range(1, track.size):
            state = (1.0 - alpha) * float(track[index]) + alpha * state
            smoothed[index] = state
    else:
        state = track[0].astype(np.float64, copy=True)
        smoothed[0] = state
        for index in range(1, track.shape[0]):
            state = (1.0 - alpha) * track[index] + alpha * state
            smoothed[index] = state
    return smoothed


def _apply_amplitude_control(segment: np.ndarray, control: Optional[DynamicControl]) -> np.ndarray:
    """Scale ``segment`` by the amplitude multiplier stored in ``control``."""

    if control is None or control.amplitudeMultiplier is None:
        return segment
    amp_track = _match_control_track(control.amplitudeMultiplier, len(segment))
    if amp_track is None:
        return segment
    scaled = segment.astype(np.float64, copy=False) * amp_track
    return scaled.astype(DTYPE, copy=False)


def _ensure_1d_array(values: Sequence[float]) -> np.ndarray:
    """Return ``values`` as a 1D float64 ``ndarray`` for interpolation helpers."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float64)
    return arr.reshape(-1).astype(np.float64, copy=False)


def _normalize_formant_targets(values: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce formant target sequences to a 2D float64 array."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.array([[float(arr)]], dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :].astype(np.float64, copy=False)
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    raise ValueError("Formant targets must be 1D or 2D sequences")


def _prosody_to_control_targets(prosody: TokenProsody) -> Optional[LowRateControlTargets]:
    """Derive low-rate control targets from ``prosody`` when available."""

    has_targets = any(
        value is not None
        for value in (
            prosody.f0MultiplierTrack,
            prosody.amplitudeMultiplierTrack,
            prosody.formantOffsetTrack,
            prosody.formantTargetTrack,
            prosody.breathLevelOffsetTrack,
            prosody.breathHnrTrack,
        )
    )
    if not has_targets:
        return None
    return LowRateControlTargets(
        f0Multiplier=prosody.f0MultiplierTrack,
        amplitudeMultiplier=prosody.amplitudeMultiplierTrack,
        formantOffsetHz=prosody.formantOffsetTrack,
        formantTargetHz=prosody.formantTargetTrack,
        breathLevelOffsetDb=prosody.breathLevelOffsetTrack,
        breathHnrOffsetDb=prosody.breathHnrTrack,
    )


class _LowRateControlGenerator:
    """Generate smooth 200 Hz control trajectories for synthesis parameters."""

    def __init__(
        self,
        *,
        frameRate: float = _CONTROL_FRAME_RATE_HZ,
        smoothingMs: float = _CONTROL_SMOOTHING_MS,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.frameRate = float(frameRate)
        self.frameDt = 1.0 / self.frameRate if self.frameRate > 0.0 else 0.0
        self.smoothingMs = float(smoothingMs)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.f0State = 0.0
        self.ampState = 0.0
        self.breathState = 0.0
        self.hnrState = 0.0
        self.formantState = np.zeros(_CONTROL_FORMANT_COUNT, dtype=np.float64)

    def _ou_series(
        self,
        length: int,
        tau_ms: float,
        sigma: float,
        state: np.ndarray,
        clip_multiple: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if length <= 0:
            return np.zeros((0,) + state.shape, dtype=np.float64), state
        tau_seconds = max(float(tau_ms), 1e-3) / 1000.0
        if self.frameDt <= 0.0:
            series = np.repeat(state[None, ...], length, axis=0)
            return series, state
        alpha = float(np.exp(-self.frameDt / tau_seconds))
        alpha = np.clip(alpha, 0.0, 0.999999)
        noise_std = float(sigma) * float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
        clip_limit = float(sigma) * float(clip_multiple)
        series = np.empty((length,) + state.shape, dtype=np.float64)
        current = state.astype(np.float64, copy=True)
        for index in range(length):
            noise = self.rng.normal(0.0, noise_std, size=state.shape)
            current = alpha * current + noise
            if clip_limit > 0.0:
                current = np.clip(current, -clip_limit, clip_limit)
            series[index] = current
        return series, current

    def next(
        self,
        sampleCount: int,
        sampleRate: int,
        formantCount: int,
        *,
        targets: Optional[LowRateControlTargets] = None,
    ) -> DynamicControl:
        """Return per-sample control tracks for the requested segment."""

        sampleCount = int(max(sampleCount, 0))
        if sampleCount == 0 or sampleRate <= 0 or self.frameRate <= 0.0:
            return DynamicControl()

        duration_seconds = sampleCount / float(sampleRate)
        frame_count = max(2, int(np.ceil(duration_seconds * self.frameRate)) + 1)

        f0_series, f0_state = self._ou_series(
            frame_count,
            _CONTROL_F0_TAU_MS,
            _CONTROL_F0_SIGMA_CENTS,
            np.array([self.f0State], dtype=np.float64),
            _CONTROL_F0_CLIP_MULTIPLE,
        )
        self.f0State = float(f0_state[-1])

        amp_series, amp_state = self._ou_series(
            frame_count,
            _CONTROL_AMP_TAU_MS,
            _CONTROL_AMP_SIGMA_DB,
            np.array([self.ampState], dtype=np.float64),
            _CONTROL_AMP_CLIP_MULTIPLE,
        )
        self.ampState = float(amp_state[-1])

        breath_series, breath_state = self._ou_series(
            frame_count,
            _CONTROL_BREATH_TAU_MS,
            _CONTROL_BREATH_SIGMA_DB,
            np.array([self.breathState], dtype=np.float64),
            _CONTROL_BREATH_CLIP_MULTIPLE,
        )
        self.breathState = float(breath_state[-1])

        active_formant_count = max(1, min(int(formantCount), _CONTROL_FORMANT_COUNT))
        formant_state = self.formantState[:active_formant_count]
        formant_series, formant_next = self._ou_series(
            frame_count,
            _CONTROL_FORMANT_TAU_MS,
            _CONTROL_FORMANT_SIGMA_HZ,
            formant_state,
            _CONTROL_FORMANT_CLIP_MULTIPLE,
        )
        self.formantState[:active_formant_count] = formant_next[-1]

        hnr_series, hnr_state = self._ou_series(
            frame_count,
            _CONTROL_HNR_TAU_MS,
            _CONTROL_HNR_SIGMA_DB,
            np.array([self.hnrState], dtype=np.float64),
            _CONTROL_HNR_CLIP_MULTIPLE,
        )
        self.hnrState = float(hnr_state[-1])

        frame_positions = np.linspace(0.0, 1.0, frame_count, dtype=np.float64, endpoint=True)
        sample_positions = np.linspace(0.0, 1.0, sampleCount, dtype=np.float64, endpoint=False)

        target_f0_track: Optional[np.ndarray] = None
        target_amp_track: Optional[np.ndarray] = None
        target_formant_track: Optional[np.ndarray] = None
        target_formant_absolute: Optional[np.ndarray] = None
        target_breath_track: Optional[np.ndarray] = None
        target_hnr_track: Optional[np.ndarray] = None

        if targets is not None:
            if targets.f0Multiplier is not None:
                raw_f0_target = _match_control_track(
                    _ensure_1d_array(targets.f0Multiplier),
                    frame_count,
                )
                if raw_f0_target is not None:
                    clamped = np.clip(
                        raw_f0_target.astype(np.float64, copy=False),
                        _CONTROL_MIN_F0_MULT,
                        _CONTROL_MAX_F0_MULT,
                    )
                    with np.errstate(divide='ignore'):
                        target_f0_track = 1200.0 * np.log2(clamped)

            if targets.amplitudeMultiplier is not None:
                raw_amp_target = _match_control_track(
                    _ensure_1d_array(targets.amplitudeMultiplier),
                    frame_count,
                )
                if raw_amp_target is not None:
                    clamped = np.clip(
                        raw_amp_target.astype(np.float64, copy=False),
                        _CONTROL_MIN_AMP,
                        _CONTROL_MAX_AMP,
                    )
                    with np.errstate(divide='ignore'):
                        target_amp_track = 20.0 * np.log10(np.maximum(clamped, EPS))

            if targets.formantOffsetHz is not None:
                normalized = _normalize_formant_targets(targets.formantOffsetHz)
                raw_formant = _match_control_track(normalized, frame_count)
                if raw_formant is not None:
                    target_formant_track = raw_formant.astype(np.float64, copy=False)

            if targets.formantTargetHz is not None:
                normalized = _normalize_formant_targets(targets.formantTargetHz)
                raw_absolute = _match_control_track(normalized, frame_count)
                if raw_absolute is not None:
                    target_formant_absolute = raw_absolute.astype(np.float64, copy=False)

            if targets.breathLevelOffsetDb is not None:
                raw_breath = _match_control_track(
                    _ensure_1d_array(targets.breathLevelOffsetDb),
                    frame_count,
                )
                if raw_breath is not None:
                    target_breath_track = raw_breath.astype(np.float64, copy=False)

            if targets.breathHnrOffsetDb is not None:
                raw_hnr = _match_control_track(
                    _ensure_1d_array(targets.breathHnrOffsetDb),
                    frame_count,
                )
                if raw_hnr is not None:
                    target_hnr_track = raw_hnr.astype(np.float64, copy=False)

        frame_f0_track = f0_series[:, 0]
        if target_f0_track is not None:
            frame_f0_track = frame_f0_track + target_f0_track

        f0_track = np.interp(sample_positions, frame_positions, frame_f0_track)
        f0_track = _smooth_control_track(f0_track, sampleRate, self.smoothingMs)
        f0_multiplier = np.clip(
            2.0 ** (f0_track / 1200.0),
            _CONTROL_MIN_F0_MULT,
            _CONTROL_MAX_F0_MULT,
        )

        frame_amp_track = amp_series[:, 0]
        if target_amp_track is not None:
            frame_amp_track = frame_amp_track + target_amp_track

        amp_track = np.interp(sample_positions, frame_positions, frame_amp_track)
        amp_track = _smooth_control_track(amp_track, sampleRate, self.smoothingMs)
        amp_multiplier = np.clip(
            10.0 ** (amp_track / 20.0),
            _CONTROL_MIN_AMP,
            _CONTROL_MAX_AMP,
        )

        frame_formant_series = formant_series[:, :active_formant_count]
        if target_formant_track is not None:
            if target_formant_track.ndim == 1:
                target_formant_track = target_formant_track[:, None]
            if target_formant_track.shape[1] < active_formant_count:
                pad_width = active_formant_count - target_formant_track.shape[1]
                target_formant_track = np.pad(
                    target_formant_track,
                    ((0, 0), (0, pad_width)),
                    mode='edge',
                )
            elif target_formant_track.shape[1] > active_formant_count:
                target_formant_track = target_formant_track[:, :active_formant_count]
            frame_formant_series = frame_formant_series + target_formant_track

        formant_interp = np.empty((sampleCount, active_formant_count), dtype=np.float64)
        for column in range(active_formant_count):
            interp_column = np.interp(
                sample_positions,
                frame_positions,
                frame_formant_series[:, column],
            )
            formant_interp[:, column] = _smooth_control_track(
                interp_column,
                sampleRate,
                self.smoothingMs,
            )

        formant_absolute_interp: Optional[np.ndarray] = None
        if target_formant_absolute is not None:
            if target_formant_absolute.ndim == 1:
                target_formant_absolute = target_formant_absolute[:, None]
            if target_formant_absolute.shape[1] < active_formant_count:
                pad_width = active_formant_count - target_formant_absolute.shape[1]
                target_formant_absolute = np.pad(
                    target_formant_absolute,
                    ((0, 0), (0, pad_width)),
                    mode='edge',
                )
            elif target_formant_absolute.shape[1] > active_formant_count:
                target_formant_absolute = target_formant_absolute[:, :active_formant_count]
            formant_absolute_interp = np.empty((sampleCount, active_formant_count), dtype=np.float64)
            for column in range(active_formant_count):
                interp_column = np.interp(
                    sample_positions,
                    frame_positions,
                    target_formant_absolute[:, column],
                )
                formant_absolute_interp[:, column] = _smooth_control_track(
                    interp_column,
                    sampleRate,
                    self.smoothingMs,
                )

        frame_breath_track = breath_series[:, 0]
        if target_breath_track is not None:
            frame_breath_track = frame_breath_track + target_breath_track

        breath_interp = np.interp(sample_positions, frame_positions, frame_breath_track)
        breath_interp = _smooth_control_track(breath_interp, sampleRate, self.smoothingMs)

        frame_hnr_track = hnr_series[:, 0]
        if target_hnr_track is not None:
            frame_hnr_track = frame_hnr_track + target_hnr_track

        hnr_interp = np.interp(sample_positions, frame_positions, frame_hnr_track)
        hnr_interp = _smooth_control_track(hnr_interp, sampleRate, self.smoothingMs)

        if active_formant_count < _CONTROL_FORMANT_COUNT:
            pad_width = _CONTROL_FORMANT_COUNT - active_formant_count
            formant_interp = np.pad(formant_interp, ((0, 0), (0, pad_width)), mode='constant')
            if formant_absolute_interp is None:
                formant_absolute_interp = np.zeros_like(formant_interp)
            else:
                formant_absolute_interp = np.pad(
                    formant_absolute_interp,
                    ((0, 0), (0, pad_width)),
                    mode='constant',
                )

        if formant_absolute_interp is None:
            formant_absolute_interp = np.zeros_like(formant_interp)

        return DynamicControl(
            f0Multiplier=f0_multiplier.astype(np.float64, copy=False),
            amplitudeMultiplier=amp_multiplier.astype(np.float64, copy=False),
            formantOffsetHz=formant_interp.astype(np.float64, copy=False),
            formantTargetHz=formant_absolute_interp.astype(np.float64, copy=False),
            breathLevelOffsetDb=breath_interp.astype(np.float64, copy=False),
            breathHnrOffsetDb=hnr_interp.astype(np.float64, copy=False),
        )


def _render_segment_control_plan(
    sample_count: int,
    sample_rate: int,
    formant_count: int,
    generator: _LowRateControlGenerator,
    targets: Optional[LowRateControlTargets],
) -> SegmentControlPlan:
    """Produce per-sample control and frame plan for a synthesis segment."""

    control = generator.next(
        sample_count,
        sample_rate,
        formant_count,
        targets=targets,
    )
    frames = _build_frame_control_plan(
        sample_count,
        sample_rate,
        control,
        generator.frameRate,
        formant_count,
    )
    return SegmentControlPlan(
        totalSamples=int(sample_count),
        dynamicControl=control,
        frames=frames,
    )

_FORMANT_MIN_FREQ_HZ = 40.0
_FORMANT_MAX_FREQ_RATIO = 0.48
_FORMANT_MIN_BW_HZ = 15.0
_FORMANT_MAX_BW_RATIO = 0.45


def _resolve_formant_ou_phase(
    speaker_profile: Optional[SpeakerProfile],
    vowel: Optional[str],
    phase: str,
) -> Optional[FormantOuPhaseParams]:
    """Return the OU parameters for a given vowel phase if available."""

    if speaker_profile is None:
        return None

    settings: FormantOuSettings = speaker_profile.formant_ou
    if settings is None:
        return None

    lookup_key = (vowel or '').lower()
    model = settings.perVowel.get(lookup_key)
    if model is None:
        model = settings.default

    candidate = getattr(model, phase, None)
    if candidate is None:
        candidate = model.sustain
    return candidate


def _formant_ou_enabled(params: Optional[FormantOuPhaseParams]) -> bool:
    """Return True when the OU configuration yields non-zero modulation."""

    if params is None:
        return False
    freq = params.frequency
    bw = params.bandwidth
    return (freq.sigma > EPS and freq.tauMilliseconds > 0.0) or (bw.sigma > EPS and bw.tauMilliseconds > 0.0)


def _smooth_tracks(tracks: np.ndarray, sr: int, smoothing_ms: float) -> np.ndarray:
    """Apply first-order low-pass smoothing to each column of ``tracks``."""

    if tracks.size == 0 or smoothing_ms <= 0.0 or sr <= 0:
        return tracks

    tau_s = max(float(smoothing_ms), 0.0) / 1000.0
    if tau_s <= 0.0:
        return tracks

    alpha = float(np.exp(-1.0 / (tau_s * sr)))
    alpha = np.clip(alpha, 0.0, 0.999999)
    smoothed = np.empty_like(tracks, dtype=np.float64)
    state = tracks[0].astype(np.float64, copy=False)
    smoothed[0] = state
    for idx in range(1, tracks.shape[0]):
        state = (1.0 - alpha) * tracks[idx] + alpha * state
        smoothed[idx] = state
    return smoothed


def _generate_ou_offsets(
    num_samples: int,
    series_count: int,
    sr: int,
    params: FormantPerturbationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate OU noise offsets for ``series_count`` parallel trajectories."""

    sigma = max(float(params.sigma), 0.0)
    if sigma <= EPS or num_samples <= 0 or sr <= 0:
        return np.zeros((max(num_samples, 0), series_count), dtype=np.float64)

    tau_s = max(float(params.tauMilliseconds), 1e-3) / 1000.0
    dt = 1.0 / float(sr)
    alpha = float(np.exp(-dt / tau_s))
    alpha = np.clip(alpha, 0.0, 0.999999)
    noise_std = sigma * float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    clip_limit = sigma * max(float(params.clipMultiple), 0.0)

    offsets = np.zeros((num_samples, series_count), dtype=np.float64)
    state = np.zeros(series_count, dtype=np.float64)
    for idx in range(num_samples):
        noise = rng.normal(0.0, noise_std, size=series_count)
        state = alpha * state + noise
        if clip_limit > 0.0:
            state = np.clip(state, -clip_limit, clip_limit)
        offsets[idx] = state

    if params.smoothingMilliseconds > 0.0:
        offsets = _smooth_tracks(offsets, sr, params.smoothingMilliseconds)
        if clip_limit > 0.0:
            offsets = np.clip(offsets, -clip_limit, clip_limit)
    return offsets


def _make_formant_tracks(
    base_formants: Sequence[float],
    base_bws: Sequence[float],
    sample_count: int,
    sr: int,
    params: Optional[FormantOuPhaseParams],
    rng: Optional[np.random.Generator] = None,
    *,
    external_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create formant and bandwidth trajectories with optional OU modulation."""

    formants = np.asarray(base_formants, dtype=np.float64).ravel()
    bws = np.asarray(base_bws, dtype=np.float64).ravel()
    if formants.size != bws.size:
        raise ValueError('Formant and bandwidth arrays must share length')

    if not _formant_ou_enabled(params) and external_offsets is None:
        return formants, bws

    total_samples = max(int(sample_count), 0)
    if total_samples <= 0 or sr <= 0:
        return formants, bws

    if _formant_ou_enabled(params):
        generator = rng if rng is not None else np.random.default_rng()
        freq_offsets = _generate_ou_offsets(total_samples, formants.size, sr, params.frequency, generator)
        bw_offsets = _generate_ou_offsets(total_samples, bws.size, sr, params.bandwidth, generator)
        freq_tracks = formants[None, :] + freq_offsets
        bw_tracks = bws[None, :] + bw_offsets
    else:
        freq_tracks = np.repeat(formants[None, :], total_samples, axis=0)
        bw_tracks = np.repeat(bws[None, :], total_samples, axis=0)

    if external_offsets is not None:
        offsets = np.asarray(external_offsets, dtype=np.float64)
        if offsets.ndim == 1:
            offsets = offsets[:, None]
        if offsets.shape[0] != total_samples:
            resized_offsets = _match_control_track(offsets, total_samples)
            if resized_offsets is not None:
                offsets = resized_offsets if resized_offsets.ndim == 2 else resized_offsets[:, None]
        if offsets.shape[1] < freq_tracks.shape[1]:
            pad_width = freq_tracks.shape[1] - offsets.shape[1]
            offsets = np.pad(offsets, ((0, 0), (0, pad_width)), mode='edge')
        if offsets.shape[1] > freq_tracks.shape[1]:
            offsets = offsets[:, : freq_tracks.shape[1]]
        freq_tracks = freq_tracks + offsets

    max_freq = sr * _FORMANT_MAX_FREQ_RATIO
    max_bw = sr * _FORMANT_MAX_BW_RATIO
    freq_tracks = np.clip(freq_tracks, _FORMANT_MIN_FREQ_HZ, max_freq)
    bw_tracks = np.clip(bw_tracks, _FORMANT_MIN_BW_HZ, max_bw)
    return freq_tracks, bw_tracks


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


def _sample_duration_scale(
    previous_state: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Generate a smooth duration multiplier using an OU process with jitter."""

    drift = -_DURATION_OU_THETA * previous_state
    state = previous_state + drift + rng.normal(0.0, _DURATION_OU_SIGMA)
    deviation = np.clip(
        state + rng.normal(0.0, _DURATION_JITTER_STD),
        -_DURATION_SCALE_LIMIT,
        _DURATION_SCALE_LIMIT,
    )
    scale = 1.0 + float(deviation)
    return scale, float(state)


def _build_phrase_scale_lookup(
    tokens: Sequence[str],
    phrase_scale_bounds: Optional[Sequence[Tuple[float, float]]],
) -> Dict[int, float]:
    """Return per-token phrase scales interpolated between provided bounds."""

    if not tokens:
        return {}

    normalized_tokens = [str(t).strip().lower() for t in tokens]
    phrase_indices: List[List[int]] = []
    current: List[int] = []
    for idx, token in enumerate(normalized_tokens):
        if token == PAUSE_TOKEN:
            if current:
                phrase_indices.append(current)
                current = []
            continue
        current.append(idx)
    if current:
        phrase_indices.append(current)

    if not phrase_indices:
        return {}

    bounds = list(phrase_scale_bounds or [])
    if bounds:
        bounds = [
            (
                float(np.clip(start, _PHRASE_SCALE_MIN, _PHRASE_SCALE_MAX)),
                float(np.clip(end, _PHRASE_SCALE_MIN, _PHRASE_SCALE_MAX)),
            )
            for start, end in bounds
        ]

    lookup: Dict[int, float] = {}
    for phrase_idx, token_indices in enumerate(phrase_indices):
        if not token_indices:
            continue
        if bounds:
            start_scale, end_scale = bounds[min(phrase_idx, len(bounds) - 1)]
        else:
            start_scale = end_scale = 1.0
        denom = max(1, len(token_indices) - 1)
        for local_pos, token_idx in enumerate(token_indices):
            alpha = local_pos / denom
            interp_scale = (1.0 - alpha) * start_scale + alpha * end_scale
            lookup[token_idx] = float(np.clip(interp_scale, _PHRASE_SCALE_MIN, _PHRASE_SCALE_MAX))
    return lookup


def _synth_pause_with_inhale(
    pause_ms: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create an inhalation-style pause with shaped noise and silence padding."""

    total_samples = _ms_to_samples(int(pause_ms), sample_rate)
    if total_samples == 0:
        return np.zeros(0, dtype=DTYPE)

    breath_ms = max(_INHALE_MIN_MS, pause_ms * _INHALE_ACTIVE_RATIO)
    breath_samples = min(total_samples, _ms_to_samples(int(breath_ms), sample_rate))
    if breath_samples <= 0:
        return np.zeros(total_samples, dtype=DTYPE)

    noise = rng.standard_normal(breath_samples).astype(DTYPE)
    shaped = _one_pole_lp(noise, cutoff=_INHALE_NOISE_CUTOFF_HZ, sr=sample_rate)

    attack_samples = max(1, int(breath_samples * _INHALE_ATTACK_RATIO))
    release_samples = max(1, breath_samples - attack_samples)
    attack_env = np.linspace(0.0, 1.0, attack_samples, dtype=DTYPE)
    release_env = np.linspace(1.0, 0.0, release_samples, dtype=DTYPE)
    envelope = np.concatenate([attack_env, release_env])
    envelope = envelope[:breath_samples]
    if envelope.size < breath_samples:
        envelope = np.pad(envelope, (0, breath_samples - envelope.size), mode='edge')

    amplitude = 10.0 ** (_INHALE_LEVEL_DB / 20.0)
    breath = shaped * envelope * amplitude

    remaining_samples = max(0, total_samples - breath_samples)
    leading_silence = int(remaining_samples * (_INHALE_SILENCE_RATIO / 2.0))
    trailing_silence = remaining_samples - leading_silence
    if leading_silence < 0:
        leading_silence = 0
    if trailing_silence < 0:
        trailing_silence = 0

    head = np.zeros(leading_silence, dtype=DTYPE)
    tail = np.zeros(trailing_silence, dtype=DTYPE)
    pause = np.concatenate([head, breath, tail]).astype(DTYPE, copy=False)
    if pause.size < total_samples:
        pause = np.pad(pause, (0, total_samples - pause.size))
    elif pause.size > total_samples:
        pause = pause[:total_samples]
    return pause


def _synth_vowel_fixed(
    formants: Sequence[float],
    bws: Sequence[float],
    f0: float,
    dur_s: float,
    sr: int,
    *,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    shimmerPercent: Optional[float] = None,
    tremoloDepthDb: float = DEFAULT_TREMOLO_DEPTH_DB,
    tremoloFrequencyHz: float = DEFAULT_TREMOLO_FREQUENCY_HZ,
    amplitudeOuSigma: float = DEFAULT_AM_OU_SIGMA,
    amplitudeOuTau: float = DEFAULT_AM_OU_TAU,
    amplitudeOuClipMultiple: float = DEFAULT_AM_OU_CLIP_MULTIPLE,
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
    formantOuPhase: Optional[FormantOuPhaseParams] = None,
    dynamicControls: Optional[DynamicControl] = None,
) -> np.ndarray:
    """Synthesize a steady-state vowel for the provided tract configuration.

    Args:
        formants: Target formant frequencies in Hz.
        bws: Corresponding bandwidths in Hz.
        f0: Base fundamental frequency in Hz.
        dur_s: Duration in seconds.
        sr: Sample rate in Hz.
        jitterCents: Amount of fast pitch variation in cents.
        shimmerDb: Legacy shimmer depth in dB.
        shimmerPercent: Optional shimmer depth expressed as linear fraction.
        tremoloDepthDb: Tremolo depth in dB.
        tremoloFrequencyHz: Tremolo rate in Hz.
        amplitudeOuSigma: OU sigma controlling slow amplitude wander.
        amplitudeOuTau: OU time constant in seconds for amplitude wander.
        amplitudeOuClipMultiple: Clip multiple used for amplitude OU limiting.
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
        formantOuPhase: OU modulation parameters for legacy formant filters.
        dynamicControls: Optional per-sample control tracks for F0, amplitude,
            formant offsets, and breath noise adjustments.
    """

    sample_count = max(0, int(dur_s * sr))
    freq_multiplier: Optional[np.ndarray] = None
    amp_multiplier: Optional[np.ndarray] = None
    formant_offset_track: Optional[np.ndarray] = None
    breath_offset_db = 0.0
    breath_hnr_offset_db = 0.0
    formant_absolute_track: Optional[np.ndarray] = None
    if dynamicControls is not None:
        freq_multiplier = _match_control_track(dynamicControls.f0Multiplier, sample_count)
        amp_multiplier = _match_control_track(dynamicControls.amplitudeMultiplier, sample_count)
        formant_offset_track = _match_control_track(dynamicControls.formantOffsetHz, sample_count)
        formant_absolute_track = _match_control_track(dynamicControls.formantTargetHz, sample_count)
        breath_track = _match_control_track(dynamicControls.breathLevelOffsetDb, sample_count)
        breath_hnr_track = _match_control_track(dynamicControls.breathHnrOffsetDb, sample_count)
        if breath_track is not None and breath_track.size > 0:
            breath_offset_db = float(np.mean(breath_track))
        if breath_hnr_track is not None and breath_hnr_track.size > 0:
            breath_hnr_offset_db = float(np.mean(breath_hnr_track))
        if freq_multiplier is not None:
            freq_multiplier = np.clip(freq_multiplier, _CONTROL_MIN_F0_MULT, _CONTROL_MAX_F0_MULT)
        if amp_multiplier is not None:
            amp_multiplier = np.clip(amp_multiplier, _CONTROL_MIN_AMP, _CONTROL_MAX_AMP)
        if formant_offset_track is not None and formant_offset_track.ndim == 2:
            if formant_offset_track.shape[1] < len(formants):
                pad_width = len(formants) - formant_offset_track.shape[1]
                formant_offset_track = np.pad(
                    formant_offset_track,
                    ((0, 0), (0, pad_width)),
                    mode='edge',
                )
            if formant_offset_track.shape[1] > len(formants):
                formant_offset_track = formant_offset_track[:, : len(formants)]

    source: GlottalSourceResult = _glottal_source(
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
        tremolo_depth_db=tremoloDepthDb,
        tremolo_frequency_hz=tremoloFrequencyHz,
        shimmer_percent=shimmerPercent,
        amplitude_ou_sigma=amplitudeOuSigma,
        amplitude_ou_tau=amplitudeOuTau,
        amplitude_ou_clip_multiple=amplitudeOuClipMultiple,
        frequency_multiplier=freq_multiplier,
        amplitude_multiplier=amp_multiplier,
    )
    src = source.signal
    formants = np.asarray(formants, dtype=np.float64)
    bws = np.asarray(bws, dtype=np.float64)
    if useLegacyFormantFilter:
        freq_targets, bw_targets = _make_formant_tracks(
            formants,
            bws,
            len(src),
            sr,
            formantOuPhase,
            external_offsets=formant_offset_track,
        )
        if formant_absolute_track is not None and formant_absolute_track.size > 0:
            if formant_absolute_track.ndim == 1:
                formant_absolute_track = formant_absolute_track[:, None]
            if formant_absolute_track.shape[1] < freq_targets.shape[1]:
                pad_width = freq_targets.shape[1] - formant_absolute_track.shape[1]
                formant_absolute_track = np.pad(
                    formant_absolute_track,
                    ((0, 0), (0, pad_width)),
                    mode='edge',
                )
            elif formant_absolute_track.shape[1] > freq_targets.shape[1]:
                formant_absolute_track = formant_absolute_track[:, : freq_targets.shape[1]]
            freq_targets = formant_absolute_track
        y = _apply_formant_filters(src, freq_targets, bw_targets, sr)
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
    effective_breath_level = breathLevelDb + breath_offset_db
    effective_hnr = breathHnrDb
    if effective_hnr is not None:
        effective_hnr = float(effective_hnr)
    if breath_hnr_offset_db != 0.0:
        if effective_hnr is None:
            effective_hnr = float(breath_hnr_offset_db)
        else:
            effective_hnr = float(effective_hnr) + float(breath_hnr_offset_db)

    y = _add_breath_noise(
        y,
        effective_breath_level,
        sr,
        f0_track=source.instantaneous_frequency,
        hnr_target_db=effective_hnr,
    )
    return _normalize_peak(y, PEAK_DEFAULT)


def synth_vowel(
    vowel: str = 'a',
    f0: float = 120.0,
    durationSeconds: float = 1.0,
    sampleRate: int = 22050,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    shimmerPercent: Optional[float] = DEFAULT_SHIMMER_PERCENT,
    tremoloDepthDb: float = DEFAULT_TREMOLO_DEPTH_DB,
    tremoloFrequencyHz: float = DEFAULT_TREMOLO_FREQUENCY_HZ,
    amplitudeOuSigma: float = DEFAULT_AM_OU_SIGMA,
    amplitudeOuTau: float = DEFAULT_AM_OU_TAU,
    amplitudeOuClipMultiple: float = DEFAULT_AM_OU_CLIP_MULTIPLE,
    breathLevelDb: float = -40.0,
    breathHnrDb: Optional[float] = 20.0,
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
    dynamicControls: Optional[DynamicControl] = None,
) -> np.ndarray:
    """Synthesize a vowel tone using the requested articulatory settings.

    Args:
        vowel: Target vowel symbol.
        f0: Fundamental frequency in Hz.
        durationSeconds: Output duration in seconds.
        sampleRate: Rendering sample rate in Hz.
        jitterCents: Fast pitch jitter amount in cents.
        shimmerDb: Legacy shimmer specification in dB.
        shimmerPercent: Optional shimmer magnitude expressed as fraction.
        tremoloDepthDb: Tremolo depth in dB.
        tremoloFrequencyHz: Tremolo rate in Hz.
        amplitudeOuSigma: OU sigma for slow amplitude wander.
        amplitudeOuTau: OU time constant for amplitude wander in seconds.
        amplitudeOuClipMultiple: Clip multiple used for amplitude OU limiting.
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
        dynamicControls: Optional low-rate control tracks applied during rendering.
    """
    assert vowel in VOWEL_TABLE, f"unsupported vowel: {vowel}"
    sample_count = max(0, int(durationSeconds * sampleRate))
    freq_multiplier: Optional[np.ndarray] = None
    amp_multiplier: Optional[np.ndarray] = None
    formant_offset_track: Optional[np.ndarray] = None
    formant_absolute_track: Optional[np.ndarray] = None
    breath_offset_db = 0.0
    breath_hnr_offset_db = 0.0
    if dynamicControls is not None:
        freq_multiplier = _match_control_track(dynamicControls.f0Multiplier, sample_count)
        amp_multiplier = _match_control_track(dynamicControls.amplitudeMultiplier, sample_count)
        formant_offset_track = _match_control_track(dynamicControls.formantOffsetHz, sample_count)
        formant_absolute_track = _match_control_track(dynamicControls.formantTargetHz, sample_count)
        breath_track = _match_control_track(dynamicControls.breathLevelOffsetDb, sample_count)
        breath_hnr_track = _match_control_track(dynamicControls.breathHnrOffsetDb, sample_count)
        if breath_track is not None and breath_track.size > 0:
            breath_offset_db = float(np.mean(breath_track))
        if breath_hnr_track is not None and breath_hnr_track.size > 0:
            breath_hnr_offset_db = float(np.mean(breath_hnr_track))
        if freq_multiplier is not None:
            freq_multiplier = np.clip(freq_multiplier, _CONTROL_MIN_F0_MULT, _CONTROL_MAX_F0_MULT)
        if amp_multiplier is not None:
            amp_multiplier = np.clip(amp_multiplier, _CONTROL_MIN_AMP, _CONTROL_MAX_AMP)
        if formant_offset_track is not None and formant_offset_track.ndim == 2:
            if formant_offset_track.shape[1] < len(VOWEL_TABLE[vowel]['F']):
                pad_width = len(VOWEL_TABLE[vowel]['F']) - formant_offset_track.shape[1]
                formant_offset_track = np.pad(
                    formant_offset_track,
                    ((0, 0), (0, pad_width)),
                    mode='edge',
                )
            if formant_offset_track.shape[1] > len(VOWEL_TABLE[vowel]['F']):
                formant_offset_track = formant_offset_track[:, : len(VOWEL_TABLE[vowel]['F'])]

    source: GlottalSourceResult = _glottal_source(
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
        tremolo_depth_db=tremoloDepthDb,
        tremolo_frequency_hz=tremoloFrequencyHz,
        shimmer_percent=shimmerPercent,
        amplitude_ou_sigma=amplitudeOuSigma,
        amplitude_ou_tau=amplitudeOuTau,
        amplitude_ou_clip_multiple=amplitudeOuClipMultiple,
        frequency_multiplier=freq_multiplier,
        amplitude_multiplier=amp_multiplier,
    )
    src = source.signal
    spec = VOWEL_TABLE[vowel]

    if kellyBlend is None:
        blend = 1.0 if (articulation is not None or areaProfile is not None) else 0.0
    else:
        blend = float(kellyBlend)
    blend = _clamp(blend, 0.0, 1.0)
    formants = np.array(spec['F'], dtype=np.float64)
    bws = np.array(spec['BW'], dtype=np.float64)
    sustain_phase = _resolve_formant_ou_phase(speakerProfile, vowel, 'sustain')

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
        freq_targets, bw_targets = _make_formant_tracks(
            formants,
            bws,
            len(src),
            sampleRate,
            sustain_phase,
            external_offsets=formant_offset_track,
        )
        if formant_absolute_track is not None and formant_absolute_track.size > 0:
            if formant_absolute_track.ndim == 1:
                formant_absolute_track = formant_absolute_track[:, None]
            if formant_absolute_track.shape[1] < freq_targets.shape[1]:
                pad_width = freq_targets.shape[1] - formant_absolute_track.shape[1]
                formant_absolute_track = np.pad(
                    formant_absolute_track,
                    ((0, 0), (0, pad_width)),
                    mode='edge',
                )
            elif formant_absolute_track.shape[1] > freq_targets.shape[1]:
                formant_absolute_track = formant_absolute_track[:, : freq_targets.shape[1]]
            freq_targets = formant_absolute_track
        y = _apply_formant_filters(src, freq_targets, bw_targets, sampleRate)
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
    effective_breath_level = breathLevelDb + breath_offset_db
    effective_hnr = breathHnrDb
    if effective_hnr is not None:
        effective_hnr = float(effective_hnr)
    if breath_hnr_offset_db != 0.0:
        if effective_hnr is None:
            effective_hnr = float(breath_hnr_offset_db)
        else:
            effective_hnr = float(effective_hnr) + float(breath_hnr_offset_db)

    y = _add_breath_noise(
        y,
        effective_breath_level,
        sampleRate,
        f0_track=source.instantaneous_frequency,
        hnr_target_db=effective_hnr,
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
    dynamicControls: Optional[DynamicControl] = None,
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
    ou_phase = _resolve_formant_ou_phase(speakerProfile, None, 'sustain')

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
        formantOuPhase=ou_phase,
        dynamicControls=dynamicControls,
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
    vowel_kwargs.setdefault('shimmerPercent', DEFAULT_SHIMMER_PERCENT)
    vowel_kwargs.setdefault('tremoloDepthDb', DEFAULT_TREMOLO_DEPTH_DB)
    vowel_kwargs.setdefault('tremoloFrequencyHz', DEFAULT_TREMOLO_FREQUENCY_HZ)
    vowel_kwargs.setdefault('amplitudeOuSigma', DEFAULT_AM_OU_SIGMA)
    vowel_kwargs.setdefault('amplitudeOuTau', DEFAULT_AM_OU_TAU)
    vowel_kwargs.setdefault('amplitudeOuClipMultiple', DEFAULT_AM_OU_CLIP_MULTIPLE)
    vowel_kwargs.setdefault('breathLevelDb', -40.0)
    vowel_kwargs.setdefault('breathHnrDb', 20.0)
    vowel_kwargs.setdefault('driftCents', DEFAULT_DRIFT_CENTS)
    vowel_kwargs.setdefault('driftReturnRate', DEFAULT_DRIFT_RETURN_RATE)
    vowel_kwargs.setdefault('vibratoDepthCents', DEFAULT_VIBRATO_DEPTH_CENTS)
    vowel_kwargs.setdefault('vibratoFrequencyHz', DEFAULT_VIBRATO_FREQUENCY_HZ)
    vowel_kwargs.setdefault('tremorDepthCents', DEFAULT_TREMOR_DEPTH_CENTS)
    vowel_kwargs.setdefault('tremorFrequencyHz', DEFAULT_TREMOR_FREQUENCY_HZ)
    speaker_profile: Optional[SpeakerProfile] = vowel_kwargs.get('speakerProfile')
    onset_phase = _resolve_formant_ou_phase(speaker_profile, vowel, 'onset')
    sustain_phase = _resolve_formant_ou_phase(speaker_profile, vowel, 'sustain')

    waveguide_opts = {
        'areaProfile': vowel_kwargs.get('areaProfile'),
        'articulation': vowel_kwargs.get('articulation'),
        'kellySections': vowel_kwargs.get('kellySections'),
        'useLegacyFormantFilter': vowel_kwargs.get('useLegacyFormantFilter', True),
        'waveguideLipReflection': vowel_kwargs.get('waveguideLipReflection', -0.85),
        'waveguideWallLoss': vowel_kwargs.get('waveguideWallLoss', 0.996),
        'jitterCents': vowel_kwargs['jitterCents'],
        'shimmerDb': vowel_kwargs['shimmerDb'],
        'shimmerPercent': vowel_kwargs['shimmerPercent'],
        'tremoloDepthDb': vowel_kwargs['tremoloDepthDb'],
        'tremoloFrequencyHz': vowel_kwargs['tremoloFrequencyHz'],
        'amplitudeOuSigma': vowel_kwargs['amplitudeOuSigma'],
        'amplitudeOuTau': vowel_kwargs['amplitudeOuTau'],
        'amplitudeOuClipMultiple': vowel_kwargs['amplitudeOuClipMultiple'],
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
        formantOuPhase=onset_phase,
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
            formantOuPhase=sustain_phase,
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
    vowel_kwargs.setdefault('breathHnrDb', 20.0)
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
    tokenControlTargets: Optional[Sequence[Optional[LowRateControlTargets]]] = None,
    phraseDurationScaleBounds: Optional[Sequence[Tuple[float, float]]] = None,
    returnControlPlan: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Tuple[SegmentControlPlan, ...]]:
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
        tokenControlTargets: Optional low-rate control targets per token.
        phraseDurationScaleBounds: Optional per-phrase (start, end) duration scales.
        returnControlPlan: When True, return the synthesized audio along with the
            frame-level control plan for each segment.
    """

    segs: List[np.ndarray] = []
    segmentPlans: List[SegmentControlPlan] = []
    default_gap_ms = max(0.0, float(gapMilliseconds))
    default_vowel_ms = max(_MIN_VOWEL_DURATION_MS, float(vowelMilliseconds))

    vowel_kwargs = dict(vowelModel or {})
    if speakerProfile is not None:
        vowel_kwargs.setdefault('speakerProfile', speakerProfile)
    speaker_profile: Optional[SpeakerProfile] = vowel_kwargs.get('speakerProfile')
    nasal_coupling = speaker_profile.nasal_coupling if speaker_profile is not None else None
    rng = np.random.default_rng()
    control_generator = _LowRateControlGenerator()
    phrase_scale_lookup = _build_phrase_scale_lookup(tokens, phraseDurationScaleBounds)
    ou_state = 0.0

    for idx, tok in enumerate(tokens):
        t = tok.strip().lower()
        if not t:
            continue

        prosody = None
        if tokenProsody is not None and idx < len(tokenProsody):
            prosody = tokenProsody[idx]

        token_f0 = float(f0) if prosody is None or prosody.f0 is None else float(prosody.f0)

        phrase_scale = phrase_scale_lookup.get(idx, 1.0)

        control_targets: Optional[LowRateControlTargets] = None
        if tokenControlTargets is not None and idx < len(tokenControlTargets):
            control_targets = tokenControlTargets[idx]
        if control_targets is None and prosody is not None:
            control_targets = _prosody_to_control_targets(prosody)

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
            pause_ms *= phrase_scale
            pause_samples = max(1, _ms_to_samples(int(round(pause_ms)), sampleRate))
            pause_plan = _render_segment_control_plan(
                pause_samples,
                sampleRate,
                _CONTROL_FORMANT_COUNT,
                control_generator,
                control_targets,
            )
            control = pause_plan.dynamicControl
            pause_seg = _synth_pause_with_inhale(pause_ms, sampleRate, rng)
            pause_seg = _apply_amplitude_control(pause_seg, control)
            segs.append(pause_seg)
            if returnControlPlan:
                segmentPlans.append(pause_plan)
            ou_state = 0.0
            continue

        duration_scale, ou_state = _sample_duration_scale(ou_state, rng)
        composite_scale = phrase_scale * duration_scale

        base_vowel_ms = default_vowel_ms
        if prosody is not None and prosody.vowelMilliseconds is not None:
            base_vowel_ms = max(_MIN_SCALED_VOWEL_MS, float(prosody.vowelMilliseconds))
        elif prosody is not None and prosody.durationScale is not None:
            base_vowel_ms = max(
                _MIN_SCALED_VOWEL_MS,
                default_vowel_ms * max(0.05, float(prosody.durationScale)),
            )
        token_vowel_ms = max(_MIN_SCALED_VOWEL_MS, base_vowel_ms * composite_scale)

        base_overlap_ms = float(overlapMilliseconds)
        if prosody is not None and prosody.overlapMilliseconds is not None:
            base_overlap_ms = max(0.0, float(prosody.overlapMilliseconds))
        token_overlap_ms = max(0, int(round(base_overlap_ms * composite_scale)))

        base_gap_ms = default_gap_ms
        if prosody is not None and prosody.gapMilliseconds is not None:
            base_gap_ms = max(0.0, float(prosody.gapMilliseconds))
        elif prosody is not None and prosody.durationScale is not None:
            base_gap_ms = max(0.0, default_gap_ms * max(0.0, float(prosody.durationScale)))
        gap_ms = max(0.0, base_gap_ms * composite_scale)
        token_gap_samples = _ms_to_samples(int(round(gap_ms)), sampleRate)

        pre_ms: Optional[float] = None
        if prosody is not None and prosody.preMilliseconds is not None:
            pre_ms = max(0.0, float(prosody.preMilliseconds) * composite_scale)

        consonant_ms: Optional[float] = None
        if prosody is not None and prosody.consonantMilliseconds is not None:
            consonant_ms = max(0.0, float(prosody.consonantMilliseconds) * composite_scale)

        expected_samples = max(1, _ms_to_samples(int(round(token_vowel_ms)), sampleRate))
        control: DynamicControl

        if t in CV_TOKEN_MAP:
            ck, vk = CV_TOKEN_MAP[t]
            if consonant_ms is None:
                base_cons = _CONSONANT_BASE_DURATION_MS.get(ck)
                if base_cons is not None:
                    consonant_ms = max(0.0, base_cons * composite_scale)
            expected_ms = token_vowel_ms
            if consonant_ms is not None:
                expected_ms += float(consonant_ms)
            elif ck in _CONSONANT_BASE_DURATION_MS:
                expected_ms += _CONSONANT_BASE_DURATION_MS[ck] * composite_scale
            if pre_ms is not None:
                expected_ms += float(pre_ms)
            expected_samples = max(1, _ms_to_samples(int(round(expected_ms)), sampleRate))
            segment_plan = _render_segment_control_plan(
                expected_samples,
                sampleRate,
                _CONTROL_FORMANT_COUNT,
                control_generator,
                control_targets,
            )
            control = segment_plan.dynamicControl
            if returnControlPlan:
                segmentPlans.append(segment_plan)
            control_f0_scale = _BASE_F0_MULTIPLIER
            if segment_plan.frames:
                frame_f0_values = [frame.f0Multiplier for frame in segment_plan.frames]
                control_f0_scale = float(
                    np.clip(
                        np.mean(frame_f0_values),
                        _CONTROL_MIN_F0_MULT,
                        _CONTROL_MAX_F0_MULT,
                    )
                )
            adjusted_f0 = token_f0 * control_f0_scale
            token_vowel_kwargs = dict(vowel_kwargs)
            breath_offset_db = 0.0
            if control.breathLevelOffsetDb is not None and control.breathLevelOffsetDb.size > 0:
                breath_offset_db = float(np.mean(control.breathLevelOffsetDb))
            if breath_offset_db != 0.0:
                base_breath = token_vowel_kwargs.get('breathLevelDb', -40.0)
                token_vowel_kwargs['breathLevelDb'] = float(base_breath) + breath_offset_db
            hnr_offset_db = 0.0
            if control.breathHnrOffsetDb is not None and control.breathHnrOffsetDb.size > 0:
                hnr_offset_db = float(np.mean(control.breathHnrOffsetDb))
            if hnr_offset_db != 0.0:
                base_hnr = token_vowel_kwargs.get('breathHnrDb', 20.0)
                if base_hnr is None:
                    token_vowel_kwargs['breathHnrDb'] = float(hnr_offset_db)
                else:
                    token_vowel_kwargs['breathHnrDb'] = float(base_hnr) + hnr_offset_db
            cv_kwargs: Dict[str, Any] = {
                'f0': adjusted_f0,
                'sampleRate': sampleRate,
                'vowelMilliseconds': int(round(token_vowel_ms)),
                'overlapMilliseconds': token_overlap_ms,
                'useOnsetTransition': useOnsetTransition,
                'vowelModel': token_vowel_kwargs,
                'speakerProfile': speaker_profile,
            }
            if pre_ms is not None:
                cv_kwargs['preMilliseconds'] = int(round(pre_ms))
            if consonant_ms is not None:
                cv_kwargs['consonantMilliseconds'] = int(round(consonant_ms))
            seg = synth_cv(ck, vk, **cv_kwargs)
            seg = _apply_amplitude_control(seg, control)

        elif t in VOWEL_TABLE:
            segment_plan = _render_segment_control_plan(
                expected_samples,
                sampleRate,
                _CONTROL_FORMANT_COUNT,
                control_generator,
                control_targets,
            )
            control = segment_plan.dynamicControl
            if returnControlPlan:
                segmentPlans.append(segment_plan)
            token_vowel_kwargs = dict(vowel_kwargs)
            seg = synth_vowel(
                t,
                f0=token_f0,
                durationSeconds=max(_MIN_SCALED_VOWEL_MS, token_vowel_ms) / 1000.0,
                sampleRate=sampleRate,
                dynamicControls=control,
                **token_vowel_kwargs,
            )

        elif t in NASAL_TOKEN_MAP:
            nasal_ms = max(
                _MIN_NASAL_DURATION_MS,
                int(round(token_vowel_ms * _NASAL_DURATION_RATIO)),
            )
            if consonant_ms is not None:
                nasal_ms = max(_MIN_NASAL_DURATION_MS, int(round(consonant_ms)))
            expected_samples = max(1, _ms_to_samples(nasal_ms, sampleRate))
            segment_plan = _render_segment_control_plan(
                expected_samples,
                sampleRate,
                _CONTROL_FORMANT_COUNT,
                control_generator,
                control_targets,
            )
            control = segment_plan.dynamicControl
            if returnControlPlan:
                segmentPlans.append(segment_plan)
            seg = synth_nasal(
                NASAL_TOKEN_MAP[t],
                f0=token_f0,
                durationMilliseconds=nasal_ms,
                sampleRate=sampleRate,
                nasalCoupling=nasal_coupling,
                speakerProfile=speaker_profile,
                dynamicControls=control,
            )
        else:
            raise ValueError(f"Unsupported token '{tok}' for synthesis")

        if segs and token_gap_samples > 0:
            segs.append(np.zeros(token_gap_samples, dtype=DTYPE))

        segs.append(seg.astype(DTYPE, copy=False))

    if not segs:
        empty = np.zeros(_ms_to_samples(int(_MIN_PAUSE_DURATION_MS), sampleRate), dtype=DTYPE)
        if returnControlPlan:
            return empty, tuple()
        return empty

    y = np.concatenate(segs).astype(DTYPE, copy=False)
    y = _soft_limit(y, _SOFT_LIMIT_DRIVE_DB)
    normalized = _normalize_peak(y, _FINAL_PEAK_TARGET)
    if returnControlPlan:
        return normalized, tuple(segmentPlans)
    return normalized


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
    tokenControlTargets: Optional[Sequence[Optional[LowRateControlTargets]]] = None,
    phraseDurationScaleBounds: Optional[Sequence[Tuple[float, float]]] = None,
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
        tokenControlTargets: Optional low-rate control targets per token.
        phraseDurationScaleBounds: Optional per-phrase (start, end) duration scales.
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
        tokenControlTargets=tokenControlTargets,
        phraseDurationScaleBounds=phraseDurationScaleBounds,
    )
    return write_wav(outPath, y, sampleRate=sampleRate)
