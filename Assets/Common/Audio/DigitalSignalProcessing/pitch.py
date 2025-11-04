# Created on 2024-05-09
# Author: ChatGPT
# Description: Pitch contour generation utilities for voice synthesis.
"""Pitch contour generation helpers following log-frequency best practices."""
from __future__ import annotations

from dataclasses import dataclass
from math import log2, pi
from typing import Optional, Sequence, Tuple

import numpy as np

from .synthesis import LowRateControlTargets


@dataclass(frozen=True)
class PitchAnchor:
    """Anchor describing the desired log-frequency at a specific timestamp."""

    timeSeconds: float
    lf0Cents: float


@dataclass(frozen=True)
class PhraseShape:
    """Phrase-level shaping parameters applied between start and end times."""

    startSeconds: float
    endSeconds: float
    slopeCentsPerSecond: float = -60.0
    startOffsetCents: float = -20.0
    finalOffsetCents: float = -120.0
    questionRiseCents: Optional[float] = None


@dataclass(frozen=True)
class PitchEvent:
    """Localised pitch event such as an accent or micro-prosodic gesture."""

    timeSeconds: float
    amountCents: float
    widthSeconds: float


@dataclass(frozen=True)
class VibratoSpec:
    """Sinusoidal vibrato configuration applied after smoothing."""

    depthCents: float = 0.0
    frequencyHz: float = 5.5
    startSeconds: float = 0.0
    endSeconds: Optional[float] = None
    phaseRadians: float = 0.0


@dataclass(frozen=True)
class PitchDriftSpec:
    """Ornstein–Uhlenbeck drift specification applied post smoothing."""

    sigmaCents: float = 12.0
    returnRate: float = 0.35
    clipMultiple: float = 3.0


@dataclass(frozen=True)
class PitchCurve:
    """Generated log-frequency trajectory and its exponential counterpart."""

    timeSeconds: np.ndarray
    lf0Cents: np.ndarray
    f0Hz: np.ndarray


_DEFAULT_CONTROL_RATE = 200.0
_MIN_DURATION = 1e-6
_EPSILON = 1e-9
_DEFAULT_RATE_LIMIT = 1000.0
_DEFAULT_SHORT_UNVOICED_S = 0.12
_LONG_UV_HOLD_RATIO = 0.65


def hz_to_lf0(frequencyHz: float) -> float:
    """Convert Hertz to cents relative to one Hertz."""

    return 1200.0 * log2(max(frequencyHz, _EPSILON))


def lf0_to_hz(lf0Cents: float) -> float:
    """Convert log-frequency cents back into Hertz."""

    return 2.0 ** (lf0Cents / 1200.0)


def generate_pitch_curve(
    anchors: Sequence[PitchAnchor],
    durationSeconds: float,
    *,
    baselineHz: float,
    controlRate: float = _DEFAULT_CONTROL_RATE,
    phraseShapes: Optional[Sequence[PhraseShape]] = None,
    unvoicedSpans: Optional[Sequence[Tuple[float, float]]] = None,
    accentEvents: Optional[Sequence[PitchEvent]] = None,
    microEvents: Optional[Sequence[PitchEvent]] = None,
    vibrato: Optional[VibratoSpec] = None,
    drift: Optional[PitchDriftSpec] = None,
    followerTauSeconds: float = 0.12,
    rateLimitCentsPerSecond: float = _DEFAULT_RATE_LIMIT,
    finalLowpassTauSeconds: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> PitchCurve:
    """Generate a smoothed log-frequency contour from anchors and events."""

    if not anchors:
        raise ValueError("At least one pitch anchor is required")

    validDuration = max(float(durationSeconds), _MIN_DURATION)
    step = 1.0 / max(controlRate, _DEFAULT_CONTROL_RATE)
    frameCount = max(1, int(round(validDuration / step)))
    timeAxis = np.linspace(0.0, step * frameCount, frameCount, endpoint=False, dtype=np.float64)

    sortedAnchors = sorted(anchors, key=lambda anchor: anchor.timeSeconds)
    anchorTimes = np.array([anchor.timeSeconds for anchor in sortedAnchors], dtype=np.float64)
    anchorLf0 = np.array([anchor.lf0Cents for anchor in sortedAnchors], dtype=np.float64)

    if phraseShapes:
        anchorLf0 = _apply_phrase_shapes(anchorTimes, anchorLf0, phraseShapes)

    targetLf0 = np.interp(
        timeAxis,
        anchorTimes,
        anchorLf0,
        left=anchorLf0[0],
        right=anchorLf0[-1],
    )

    if unvoicedSpans:
        targetLf0 = _bridge_unvoiced_segments(
            timeAxis,
            targetLf0,
            unvoicedSpans,
            defaultSlope=-60.0,
        )

    smoothedLf0 = _critically_damped_follow(
        targetLf0,
        step,
        followerTauSeconds,
    )

    if rateLimitCentsPerSecond > 0.0:
        smoothedLf0 = _apply_rate_limit(smoothedLf0, step, rateLimitCentsPerSecond)

    combinedLf0 = smoothedLf0.copy()

    if accentEvents:
        combinedLf0 += _render_events(timeAxis, accentEvents)

    if microEvents:
        combinedLf0 += _render_events(timeAxis, microEvents)

    if finalLowpassTauSeconds > 0.0:
        combinedLf0 = _single_pole_lowpass(combinedLf0, step, finalLowpassTauSeconds)

    if drift:
        combinedLf0 += _generate_drift(trackLength=combinedLf0.size, dt=step, spec=drift, rng=rng)

    if vibrato and vibrato.depthCents != 0.0 and vibrato.frequencyHz > 0.0:
        combinedLf0 += _render_vibrato(timeAxis, vibrato)

    baseLf0 = hz_to_lf0(baselineHz)
    absoluteLf0 = combinedLf0 + baseLf0
    frequencyTrack = lf0_to_hz_array(absoluteLf0)

    return PitchCurve(timeAxis, absoluteLf0, frequencyTrack)


def lf0_to_hz_array(lf0Cents: np.ndarray) -> np.ndarray:
    """Vectorised cents-to-Hertz conversion."""

    return np.power(2.0, lf0Cents / 1200.0, dtype=np.float64)


def generate_f0_targets(
    anchors: Sequence[PitchAnchor],
    durationSeconds: float,
    *,
    baselineHz: float,
    controlRate: float = _DEFAULT_CONTROL_RATE,
    phraseShapes: Optional[Sequence[PhraseShape]] = None,
    unvoicedSpans: Optional[Sequence[Tuple[float, float]]] = None,
    accentEvents: Optional[Sequence[PitchEvent]] = None,
    microEvents: Optional[Sequence[PitchEvent]] = None,
    vibrato: Optional[VibratoSpec] = None,
    drift: Optional[PitchDriftSpec] = None,
    followerTauSeconds: float = 0.12,
    rateLimitCentsPerSecond: float = _DEFAULT_RATE_LIMIT,
    finalLowpassTauSeconds: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[PitchCurve, LowRateControlTargets]:
    """Generate a pitch curve and wrap it in ``LowRateControlTargets``."""

    curve = generate_pitch_curve(
        anchors,
        durationSeconds,
        baselineHz=baselineHz,
        controlRate=controlRate,
        phraseShapes=phraseShapes,
        unvoicedSpans=unvoicedSpans,
        accentEvents=accentEvents,
        microEvents=microEvents,
        vibrato=vibrato,
        drift=drift,
        followerTauSeconds=followerTauSeconds,
        rateLimitCentsPerSecond=rateLimitCentsPerSecond,
        finalLowpassTauSeconds=finalLowpassTauSeconds,
        rng=rng,
    )

    multipliers = np.clip(curve.f0Hz / max(baselineHz, _EPSILON), 0.001, 16.0)
    return curve, LowRateControlTargets(f0Multiplier=multipliers.tolist())


def _apply_phrase_shapes(
    anchorTimes: np.ndarray,
    anchorLf0: np.ndarray,
    phraseShapes: Sequence[PhraseShape],
) -> np.ndarray:
    adjusted = anchorLf0.copy()
    for phrase in phraseShapes:
        start = float(phrase.startSeconds)
        end = float(phrase.endSeconds)
        if end <= start:
            continue
        mask = (anchorTimes >= start) & (anchorTimes <= end)
        if not np.any(mask):
            continue
        indices = np.where(mask)[0]
        firstIndex = int(indices[0])
        lastIndex = int(indices[-1])
        startTime = anchorTimes[firstIndex]
        adjusted[firstIndex] += float(phrase.startOffsetCents)
        adjusted[lastIndex] += float(
            phrase.questionRiseCents if phrase.questionRiseCents is not None else phrase.finalOffsetCents
        )
        slope = float(phrase.slopeCentsPerSecond)
        adjusted[indices] += slope * (anchorTimes[indices] - startTime)
    return adjusted


def _bridge_unvoiced_segments(
    timeAxis: np.ndarray,
    lf0Track: np.ndarray,
    spans: Sequence[Tuple[float, float]],
    *,
    defaultSlope: float,
    shortThreshold: float = _DEFAULT_SHORT_UNVOICED_S,
) -> np.ndarray:
    bridged = lf0Track.copy()
    totalSamples = lf0Track.size
    for start, end in spans:
        spanStart = max(float(start), 0.0)
        spanEnd = max(spanStart, float(end))
        if spanEnd <= spanStart:
            continue
        startIndex = int(np.searchsorted(timeAxis, spanStart, side="left"))
        endIndex = int(np.searchsorted(timeAxis, spanEnd, side="right"))
        if endIndex <= startIndex:
            continue
        prevIndex = startIndex - 1
        nextIndex = endIndex if endIndex < totalSamples else totalSamples - 1
        prevValue = bridged[prevIndex] if prevIndex >= 0 else bridged[min(nextIndex, totalSamples - 1)]
        nextValue = bridged[nextIndex] if nextIndex < totalSamples else prevValue
        spanDuration = timeAxis[min(endIndex, totalSamples - 1)] - timeAxis[startIndex]
        if spanDuration <= shortThreshold and prevIndex >= 0 and nextIndex < totalSamples:
            interp = np.linspace(prevValue, nextValue, endIndex - prevIndex, endpoint=False)
            bridged[startIndex:endIndex] = interp[1:]
            continue
        indices = np.arange(startIndex, endIndex, dtype=np.int64)
        holdLength = max(1, int(round(len(indices) * _LONG_UV_HOLD_RATIO)))
        holdTimes = timeAxis[indices[:holdLength]] - timeAxis[indices[0]]
        holdValues = prevValue + defaultSlope * holdTimes
        holdValues = np.clip(
            holdValues,
            min(prevValue, nextValue) - 480.0,
            max(prevValue, nextValue) + 480.0,
        )
        bridged[indices[:holdLength]] = holdValues
        rampLength = len(indices) - holdLength
        if rampLength > 0:
            rampStartValue = holdValues[-1] if holdValues.size > 0 else prevValue
            ramp = np.linspace(rampStartValue, nextValue, rampLength + 1, endpoint=True)[1:]
            bridged[indices[holdLength:]] = ramp
    return bridged


def _critically_damped_follow(target: np.ndarray, dt: float, tauSeconds: float) -> np.ndarray:
    omega = 1.0 / max(tauSeconds, _MIN_DURATION)
    position = float(target[0])
    velocity = 0.0
    followed = np.empty_like(target, dtype=np.float64)
    followed[0] = position
    for index in range(1, target.size):
        goal = float(target[index])
        error = position - goal
        acceleration = -2.0 * omega * velocity - (omega * omega) * error
        velocity += acceleration * dt
        position += velocity * dt
        followed[index] = position
    return followed


def _apply_rate_limit(track: np.ndarray, dt: float, maxRate: float) -> np.ndarray:
    limited = track.copy()
    limit = maxRate * dt
    if limit <= 0.0:
        return limited
    for index in range(1, limited.size):
        delta = limited[index] - limited[index - 1]
        if delta > limit:
            limited[index] = limited[index - 1] + limit
        elif delta < -limit:
            limited[index] = limited[index - 1] - limit
    return limited


def _render_events(timeAxis: np.ndarray, events: Sequence[PitchEvent]) -> np.ndarray:
    envelope = np.zeros_like(timeAxis, dtype=np.float64)
    for event in events:
        width = max(float(event.widthSeconds), _MIN_DURATION)
        sigma = width / 2.354820045
        if sigma <= 0.0:
            continue
        distance = timeAxis - float(event.timeSeconds)
        envelope += float(event.amountCents) * np.exp(-0.5 * (distance / sigma) ** 2)
    return envelope


def _single_pole_lowpass(track: np.ndarray, dt: float, tauSeconds: float) -> np.ndarray:
    alpha = float(np.exp(-dt / max(tauSeconds, _MIN_DURATION)))
    state = float(track[0])
    filtered = np.empty_like(track, dtype=np.float64)
    filtered[0] = state
    for index in range(1, track.size):
        state = (1.0 - alpha) * float(track[index]) + alpha * state
        filtered[index] = state
    return filtered


def _generate_drift(trackLength: int, dt: float, spec: PitchDriftSpec, rng: Optional[np.random.Generator]) -> np.ndarray:
    generator = rng if rng is not None else np.random.default_rng()
    theta = max(float(spec.returnRate), _MIN_DURATION)
    sigma = max(float(spec.sigmaCents), 0.0)
    if sigma == 0.0:
        return np.zeros(trackLength, dtype=np.float64)
    alpha = np.exp(-theta * dt)
    variance = (sigma ** 2) * (1.0 - np.exp(-2.0 * theta * dt))
    noiseStd = np.sqrt(max(variance, 0.0))
    limit = None
    if spec.clipMultiple > 0.0:
        limit = float(spec.clipMultiple) * sigma
    state = 0.0
    driftTrack = np.empty(trackLength, dtype=np.float64)
    for index in range(trackLength):
        state = alpha * state + noiseStd * generator.standard_normal()
        if limit is not None:
            state = float(np.clip(state, -limit, limit))
        driftTrack[index] = state
    return driftTrack


def _render_vibrato(timeAxis: np.ndarray, spec: VibratoSpec) -> np.ndarray:
    start = max(float(spec.startSeconds), 0.0)
    end = float(spec.endSeconds) if spec.endSeconds is not None else timeAxis[-1] if timeAxis.size > 0 else start
    activeMask = (timeAxis >= start) & (timeAxis <= end)
    vibrato = np.zeros_like(timeAxis, dtype=np.float64)
    if not np.any(activeMask):
        return vibrato
    phase = 2.0 * pi * float(spec.frequencyHz) * (timeAxis[activeMask] - start) + float(spec.phaseRadians)
    vibrato[activeMask] = float(spec.depthCents) * np.sin(phase)
    return vibrato


__all__ = [
    "PitchAnchor",
    "PhraseShape",
    "PitchEvent",
    "VibratoSpec",
    "PitchDriftSpec",
    "PitchCurve",
    "hz_to_lf0",
    "lf0_to_hz",
    "lf0_to_hz_array",
    "generate_pitch_curve",
    "generate_f0_targets",
]
