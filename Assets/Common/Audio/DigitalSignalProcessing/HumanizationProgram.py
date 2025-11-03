# Created on 2024-12-05
# Author: ChatGPT
# Description: Recommended humanisation programs for the synthesis pipeline.
"""Humanisation program helpers for configuring natural vocal synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .constants import (
    FormantOuModel,
    FormantOuPhaseParams,
    FormantOuSettings,
    FormantPerturbationParams,
    SpeakerProfile,
)

__all__ = [
    "PitchProgram",
    "AmplitudeProgram",
    "NoiseProgram",
    "FormantProgram",
    "TimingProgram",
    "HumanizationProgram",
    "build_humanization_program",
    "list_available_programs",
]

# Pitch defaults
SPEECH_JITTER_CENTS = 7.0
SPEECH_PERIOD_JITTER_STD = 0.007
SPEECH_PERIOD_JITTER_LIMIT = 0.04
SPEECH_DRIFT_CENTS = 9.0
SPEECH_DRIFT_RETURN_RATE = 0.34
SPEECH_VIBRATO_DEPTH_CENTS = 26.0
SPEECH_VIBRATO_FREQUENCY_HZ = 5.6
SPEECH_TREMOR_DEPTH_CENTS = 8.5
SPEECH_TREMOR_FREQUENCY_HZ = 10.5

BREATHY_JITTER_CENTS = 9.0
BREATHY_PERIOD_JITTER_STD = 0.011
BREATHY_PERIOD_JITTER_LIMIT = 0.05
BREATHY_DRIFT_CENTS = 11.0
BREATHY_DRIFT_RETURN_RATE = 0.28
BREATHY_VIBRATO_DEPTH_CENTS = 18.0
BREATHY_VIBRATO_FREQUENCY_HZ = 5.0
BREATHY_TREMOR_DEPTH_CENTS = 6.0
BREATHY_TREMOR_FREQUENCY_HZ = 9.0

TENSE_JITTER_CENTS = 5.0
TENSE_PERIOD_JITTER_STD = 0.005
TENSE_PERIOD_JITTER_LIMIT = 0.03
TENSE_DRIFT_CENTS = 7.0
TENSE_DRIFT_RETURN_RATE = 0.42
TENSE_VIBRATO_DEPTH_CENTS = 20.0
TENSE_VIBRATO_FREQUENCY_HZ = 5.8
TENSE_TREMOR_DEPTH_CENTS = 9.0
TENSE_TREMOR_FREQUENCY_HZ = 12.0

# Amplitude defaults
SPEECH_SHIMMER_DB = 0.9
SPEECH_SHIMMER_PERCENT = 0.04
SPEECH_TREMOLO_DEPTH_DB = 0.35
SPEECH_TREMOLO_FREQUENCY_HZ = 5.6
SPEECH_AM_OU_SIGMA = 0.032
SPEECH_AM_OU_TAU = 0.24
SPEECH_AM_OU_CLIP = 3.2

BREATHY_SHIMMER_DB = 1.2
BREATHY_SHIMMER_PERCENT = 0.05
BREATHY_TREMOLO_DEPTH_DB = 0.4
BREATHY_TREMOLO_FREQUENCY_HZ = 5.2
BREATHY_AM_OU_SIGMA = 0.038
BREATHY_AM_OU_TAU = 0.28
BREATHY_AM_OU_CLIP = 3.4

TENSE_SHIMMER_DB = 0.6
TENSE_SHIMMER_PERCENT = 0.03
TENSE_TREMOLO_DEPTH_DB = 0.28
TENSE_TREMOLO_FREQUENCY_HZ = 5.8
TENSE_AM_OU_SIGMA = 0.026
TENSE_AM_OU_TAU = 0.22
TENSE_AM_OU_CLIP = 3.0

# Noise defaults
SPEECH_BREATH_LEVEL_DB = -38.0
SPEECH_BREATH_HNR_DB = 18.0
SPEECH_TILT_DB_PER_OCT = -8.5
SPEECH_INHALE_DURATION_MS = 260.0
SPEECH_PAUSE_SILENCE_MS = 120.0

BREATHY_BREATH_LEVEL_DB = -34.0
BREATHY_BREATH_HNR_DB = 12.0
BREATHY_TILT_DB_PER_OCT = -10.0
BREATHY_INHALE_DURATION_MS = 320.0
BREATHY_PAUSE_SILENCE_MS = 150.0

TENSE_BREATH_LEVEL_DB = -42.0
TENSE_BREATH_HNR_DB = 24.0
TENSE_TILT_DB_PER_OCT = -7.0
TENSE_INHALE_DURATION_MS = 180.0
TENSE_PAUSE_SILENCE_MS = 100.0

# Formant defaults
SPEECH_FORMANT_SIGMA = 22.0
SPEECH_FORMANT_TAU_MS = 180.0
SPEECH_BW_SIGMA = 16.0
SPEECH_BW_TAU_MS = 210.0
SPEECH_FORMANT_CLIP = 2.6
SPEECH_FORMANT_SMOOTH_MS = 18.0

BREATHY_FORMANT_SIGMA = 28.0
BREATHY_FORMANT_TAU_MS = 230.0
BREATHY_BW_SIGMA = 20.0
BREATHY_BW_TAU_MS = 240.0
BREATHY_FORMANT_CLIP = 2.8
BREATHY_FORMANT_SMOOTH_MS = 22.0

TENSE_FORMANT_SIGMA = 18.0
TENSE_FORMANT_TAU_MS = 160.0
TENSE_BW_SIGMA = 14.0
TENSE_BW_TAU_MS = 190.0
TENSE_FORMANT_CLIP = 2.4
TENSE_FORMANT_SMOOTH_MS = 14.0

# Timing defaults
SPEECH_DURATION_DRIFT_STD = 0.018
SPEECH_DURATION_JITTER_STD = 0.007
SPEECH_PHRASE_ACCEL = 0.012

BREATHY_DURATION_DRIFT_STD = 0.022
BREATHY_DURATION_JITTER_STD = 0.009
BREATHY_PHRASE_ACCEL = 0.010

TENSE_DURATION_DRIFT_STD = 0.015
TENSE_DURATION_JITTER_STD = 0.005
TENSE_PHRASE_ACCEL = 0.014


@dataclass(frozen=True)
class PitchProgram:
    """Pitch-related modulation targets used during synthesis."""

    jitterCents: float
    periodJitterStd: float
    periodJitterLimit: float
    driftCents: float
    driftReturnRate: float
    vibratoDepthCents: float
    vibratoFrequencyHz: float
    tremorDepthCents: float
    tremorFrequencyHz: float

    def to_glottal_kwargs(self) -> Dict[str, float]:
        """Return keyword arguments for ``_glottal_source`` pitch controls."""

        return {
            "jitterCents": self.jitterCents,
            "periodJitterStd": self.periodJitterStd,
            "periodJitterLimit": self.periodJitterLimit,
            "driftCents": self.driftCents,
            "driftReturnRate": self.driftReturnRate,
            "vibratoDepthCents": self.vibratoDepthCents,
            "vibratoFrequencyHz": self.vibratoFrequencyHz,
            "tremorDepthCents": self.tremorDepthCents,
            "tremorFrequencyHz": self.tremorFrequencyHz,
        }


@dataclass(frozen=True)
class AmplitudeProgram:
    """Slow and per-period amplitude modulation targets."""

    shimmerDb: float
    shimmerPercent: float
    tremoloDepthDb: float
    tremoloFrequencyHz: float
    amplitudeOuSigma: float
    amplitudeOuTau: float
    amplitudeOuClipMultiple: float

    def to_glottal_kwargs(self) -> Dict[str, float]:
        """Return keyword arguments for amplitude-related glottal controls."""

        return {
            "shimmerDb": self.shimmerDb,
            "shimmerPercent": self.shimmerPercent,
            "tremoloDepthDb": self.tremoloDepthDb,
            "tremoloFrequencyHz": self.tremoloFrequencyHz,
            "amplitudeOuSigma": self.amplitudeOuSigma,
            "amplitudeOuTau": self.amplitudeOuTau,
            "amplitudeOuClipMultiple": self.amplitudeOuClipMultiple,
        }


@dataclass(frozen=True)
class NoiseProgram:
    """Noise gain, tilt, and breathing cues."""

    breathLevelDb: float
    breathHnrDb: float
    spectralTiltDbPerOctave: float
    inhaleDurationMs: float
    pauseSilenceMs: float


@dataclass(frozen=True)
class FormantProgram:
    """OU parameters describing formant and bandwidth wander."""

    frequencySigma: float
    frequencyTauMs: float
    bandwidthSigma: float
    bandwidthTauMs: float
    clipMultiple: float
    smoothingMs: float

    def to_phase_params(self) -> FormantOuPhaseParams:
        """Create ``FormantOuPhaseParams`` from the stored OU magnitudes."""

        freq_params = FormantPerturbationParams(
            sigma=self.frequencySigma,
            tauMilliseconds=self.frequencyTauMs,
            clipMultiple=self.clipMultiple,
            smoothingMilliseconds=self.smoothingMs,
        )
        bw_params = FormantPerturbationParams(
            sigma=self.bandwidthSigma,
            tauMilliseconds=self.bandwidthTauMs,
            clipMultiple=self.clipMultiple,
            smoothingMilliseconds=self.smoothingMs,
        )
        return FormantOuPhaseParams(frequency=freq_params, bandwidth=bw_params)


@dataclass(frozen=True)
class TimingProgram:
    """Timing modulation strengths for phrase level shaping."""

    durationDriftStd: float
    durationJitterStd: float
    phraseAccelerationPerSecond: float


@dataclass(frozen=True)
class HumanizationProgram:
    """Bundle of modulation programs describing a full humanisation setup."""

    name: str
    summary: str
    pitch: PitchProgram
    amplitude: AmplitudeProgram
    noise: NoiseProgram
    formant: FormantProgram
    timing: TimingProgram
    checklist: Tuple[str, ...]

    def synth_kwargs(self) -> Dict[str, float]:
        """Create keyword arguments for :func:`synth_vowel` and similar APIs."""

        kwargs: Dict[str, float] = {}
        kwargs.update(self.pitch.to_glottal_kwargs())
        kwargs.update(self.amplitude.to_glottal_kwargs())
        kwargs["breathLevelDb"] = self.noise.breathLevelDb
        kwargs["breathHnrDb"] = self.noise.breathHnrDb
        return kwargs

    def create_speaker_profile(self, base: Optional[SpeakerProfile] = None) -> SpeakerProfile:
        """Return a speaker profile with formant OU matching the program.

        Args:
            base: Optional template speaker profile to clone nasal coupling from.
        """

        template = base if base is not None else SpeakerProfile()
        sustain_params = self.formant.to_phase_params()
        per_vowel = {
            key: FormantOuModel(sustain=sustain_params)
            for key in template.formant_ou.perVowel.keys()
        }
        formant_settings = FormantOuSettings(
            default=FormantOuModel(sustain=sustain_params),
            perVowel=per_vowel,
        )
        return SpeakerProfile(
            nasal_coupling=template.nasal_coupling,
            formant_ou=formant_settings,
        )

    def describe_steps(self) -> List[str]:
        """Return the recommended implementation steps for documentation or UI."""

        return list(self.checklist)


def _speech_program() -> HumanizationProgram:
    """Return the default conversational speech program."""

    pitch = PitchProgram(
        jitterCents=SPEECH_JITTER_CENTS,
        periodJitterStd=SPEECH_PERIOD_JITTER_STD,
        periodJitterLimit=SPEECH_PERIOD_JITTER_LIMIT,
        driftCents=SPEECH_DRIFT_CENTS,
        driftReturnRate=SPEECH_DRIFT_RETURN_RATE,
        vibratoDepthCents=SPEECH_VIBRATO_DEPTH_CENTS,
        vibratoFrequencyHz=SPEECH_VIBRATO_FREQUENCY_HZ,
        tremorDepthCents=SPEECH_TREMOR_DEPTH_CENTS,
        tremorFrequencyHz=SPEECH_TREMOR_FREQUENCY_HZ,
    )
    amplitude = AmplitudeProgram(
        shimmerDb=SPEECH_SHIMMER_DB,
        shimmerPercent=SPEECH_SHIMMER_PERCENT,
        tremoloDepthDb=SPEECH_TREMOLO_DEPTH_DB,
        tremoloFrequencyHz=SPEECH_TREMOLO_FREQUENCY_HZ,
        amplitudeOuSigma=SPEECH_AM_OU_SIGMA,
        amplitudeOuTau=SPEECH_AM_OU_TAU,
        amplitudeOuClipMultiple=SPEECH_AM_OU_CLIP,
    )
    noise = NoiseProgram(
        breathLevelDb=SPEECH_BREATH_LEVEL_DB,
        breathHnrDb=SPEECH_BREATH_HNR_DB,
        spectralTiltDbPerOctave=SPEECH_TILT_DB_PER_OCT,
        inhaleDurationMs=SPEECH_INHALE_DURATION_MS,
        pauseSilenceMs=SPEECH_PAUSE_SILENCE_MS,
    )
    formant = FormantProgram(
        frequencySigma=SPEECH_FORMANT_SIGMA,
        frequencyTauMs=SPEECH_FORMANT_TAU_MS,
        bandwidthSigma=SPEECH_BW_SIGMA,
        bandwidthTauMs=SPEECH_BW_TAU_MS,
        clipMultiple=SPEECH_FORMANT_CLIP,
        smoothingMs=SPEECH_FORMANT_SMOOTH_MS,
    )
    timing = TimingProgram(
        durationDriftStd=SPEECH_DURATION_DRIFT_STD,
        durationJitterStd=SPEECH_DURATION_JITTER_STD,
        phraseAccelerationPerSecond=SPEECH_PHRASE_ACCEL,
    )
    checklist = (
        "F0: Add ±{:.1f} cents vibrato around {:.1f} Hz with OU drift.".format(
            SPEECH_VIBRATO_DEPTH_CENTS, SPEECH_VIBRATO_FREQUENCY_HZ
        ),
        "Amplitude: Blend shimmer {:.0%} with {:.1f} dB tremolo.".format(
            SPEECH_SHIMMER_PERCENT, SPEECH_TREMOLO_DEPTH_DB
        ),
        "Noise: Mix breath to target HNR {:.1f} dB with {:.1f} dB/oct tilt.".format(
            SPEECH_BREATH_HNR_DB, SPEECH_TILT_DB_PER_OCT
        ),
        "Formants: OU wander σ={:.1f} Hz, τ={:.0f} ms (frequency).".format(
            SPEECH_FORMANT_SIGMA, SPEECH_FORMANT_TAU_MS
        ),
        "Timing: Apply drift σ={:.3f}, jitter σ={:.3f}, pause {:.0f} ms + inhale.".format(
            SPEECH_DURATION_DRIFT_STD, SPEECH_DURATION_JITTER_STD, SPEECH_PAUSE_SILENCE_MS
        ),
    )
    return HumanizationProgram(
        name="speech",
        summary="汎用的な会話声向けの人間化プリセット",
        pitch=pitch,
        amplitude=amplitude,
        noise=noise,
        formant=formant,
        timing=timing,
        checklist=checklist,
    )


def _breathy_program() -> HumanizationProgram:
    """Return a breathy voice program with stronger noise and slower drift."""

    pitch = PitchProgram(
        jitterCents=BREATHY_JITTER_CENTS,
        periodJitterStd=BREATHY_PERIOD_JITTER_STD,
        periodJitterLimit=BREATHY_PERIOD_JITTER_LIMIT,
        driftCents=BREATHY_DRIFT_CENTS,
        driftReturnRate=BREATHY_DRIFT_RETURN_RATE,
        vibratoDepthCents=BREATHY_VIBRATO_DEPTH_CENTS,
        vibratoFrequencyHz=BREATHY_VIBRATO_FREQUENCY_HZ,
        tremorDepthCents=BREATHY_TREMOR_DEPTH_CENTS,
        tremorFrequencyHz=BREATHY_TREMOR_FREQUENCY_HZ,
    )
    amplitude = AmplitudeProgram(
        shimmerDb=BREATHY_SHIMMER_DB,
        shimmerPercent=BREATHY_SHIMMER_PERCENT,
        tremoloDepthDb=BREATHY_TREMOLO_DEPTH_DB,
        tremoloFrequencyHz=BREATHY_TREMOLO_FREQUENCY_HZ,
        amplitudeOuSigma=BREATHY_AM_OU_SIGMA,
        amplitudeOuTau=BREATHY_AM_OU_TAU,
        amplitudeOuClipMultiple=BREATHY_AM_OU_CLIP,
    )
    noise = NoiseProgram(
        breathLevelDb=BREATHY_BREATH_LEVEL_DB,
        breathHnrDb=BREATHY_BREATH_HNR_DB,
        spectralTiltDbPerOctave=BREATHY_TILT_DB_PER_OCT,
        inhaleDurationMs=BREATHY_INHALE_DURATION_MS,
        pauseSilenceMs=BREATHY_PAUSE_SILENCE_MS,
    )
    formant = FormantProgram(
        frequencySigma=BREATHY_FORMANT_SIGMA,
        frequencyTauMs=BREATHY_FORMANT_TAU_MS,
        bandwidthSigma=BREATHY_BW_SIGMA,
        bandwidthTauMs=BREATHY_BW_TAU_MS,
        clipMultiple=BREATHY_FORMANT_CLIP,
        smoothingMs=BREATHY_FORMANT_SMOOTH_MS,
    )
    timing = TimingProgram(
        durationDriftStd=BREATHY_DURATION_DRIFT_STD,
        durationJitterStd=BREATHY_DURATION_JITTER_STD,
        phraseAccelerationPerSecond=BREATHY_PHRASE_ACCEL,
    )
    checklist = (
        "F0: ゆっくり漂う±{:.1f} centsドリフトとビブラート {:.1f} Hz.".format(
            BREATHY_DRIFT_CENTS, BREATHY_VIBRATO_FREQUENCY_HZ
        ),
        "Amplitude: シマー {:.0%}・トレモロ {:.1f} dBを同期させる.".format(
            BREATHY_SHIMMER_PERCENT, BREATHY_TREMOLO_DEPTH_DB
        ),
        "Noise: HNR {:.1f} dB とティルト {:.1f} dB/oct を意識する.".format(
            BREATHY_BREATH_HNR_DB, BREATHY_TILT_DB_PER_OCT
        ),
        "Formants: σ={:.1f}/{:.1f} Hz, τ={:.0f}/{:.0f} msで滑らかに.".format(
            BREATHY_FORMANT_SIGMA,
            BREATHY_BW_SIGMA,
            BREATHY_FORMANT_TAU_MS,
            BREATHY_BW_TAU_MS,
        ),
        "Timing: ポーズ {:.0f} ms + 吸気 {:.0f} ms のブリージングを入れる.".format(
            BREATHY_PAUSE_SILENCE_MS,
            BREATHY_INHALE_DURATION_MS,
        ),
    )
    return HumanizationProgram(
        name="breathy",
        summary="息っぽいソフトな声質向けプリセット",
        pitch=pitch,
        amplitude=amplitude,
        noise=noise,
        formant=formant,
        timing=timing,
        checklist=checklist,
    )


def _tense_program() -> HumanizationProgram:
    """Return an energetic tense voice program with tighter control."""

    pitch = PitchProgram(
        jitterCents=TENSE_JITTER_CENTS,
        periodJitterStd=TENSE_PERIOD_JITTER_STD,
        periodJitterLimit=TENSE_PERIOD_JITTER_LIMIT,
        driftCents=TENSE_DRIFT_CENTS,
        driftReturnRate=TENSE_DRIFT_RETURN_RATE,
        vibratoDepthCents=TENSE_VIBRATO_DEPTH_CENTS,
        vibratoFrequencyHz=TENSE_VIBRATO_FREQUENCY_HZ,
        tremorDepthCents=TENSE_TREMOR_DEPTH_CENTS,
        tremorFrequencyHz=TENSE_TREMOR_FREQUENCY_HZ,
    )
    amplitude = AmplitudeProgram(
        shimmerDb=TENSE_SHIMMER_DB,
        shimmerPercent=TENSE_SHIMMER_PERCENT,
        tremoloDepthDb=TENSE_TREMOLO_DEPTH_DB,
        tremoloFrequencyHz=TENSE_TREMOLO_FREQUENCY_HZ,
        amplitudeOuSigma=TENSE_AM_OU_SIGMA,
        amplitudeOuTau=TENSE_AM_OU_TAU,
        amplitudeOuClipMultiple=TENSE_AM_OU_CLIP,
    )
    noise = NoiseProgram(
        breathLevelDb=TENSE_BREATH_LEVEL_DB,
        breathHnrDb=TENSE_BREATH_HNR_DB,
        spectralTiltDbPerOctave=TENSE_TILT_DB_PER_OCT,
        inhaleDurationMs=TENSE_INHALE_DURATION_MS,
        pauseSilenceMs=TENSE_PAUSE_SILENCE_MS,
    )
    formant = FormantProgram(
        frequencySigma=TENSE_FORMANT_SIGMA,
        frequencyTauMs=TENSE_FORMANT_TAU_MS,
        bandwidthSigma=TENSE_BW_SIGMA,
        bandwidthTauMs=TENSE_BW_TAU_MS,
        clipMultiple=TENSE_FORMANT_CLIP,
        smoothingMs=TENSE_FORMANT_SMOOTH_MS,
    )
    timing = TimingProgram(
        durationDriftStd=TENSE_DURATION_DRIFT_STD,
        durationJitterStd=TENSE_DURATION_JITTER_STD,
        phraseAccelerationPerSecond=TENSE_PHRASE_ACCEL,
    )
    checklist = (
        "F0: タイトな±{:.1f} centsジッターと {:.1f} Hz ビブラート.".format(
            TENSE_JITTER_CENTS, TENSE_VIBRATO_FREQUENCY_HZ
        ),
        "Amplitude: シマー {:.0%}, トレモロ {:.1f} dBで芯を保つ.".format(
            TENSE_SHIMMER_PERCENT, TENSE_TREMOLO_DEPTH_DB
        ),
        "Noise: HNR {:.1f} dB、ティルト {:.1f} dB/oct でクリアに.".format(
            TENSE_BREATH_HNR_DB, TENSE_TILT_DB_PER_OCT
        ),
        "Formants: σ={:.1f}/{:.1f} Hz、τ={:.0f}/{:.0f} msで機敏に.".format(
            TENSE_FORMANT_SIGMA,
            TENSE_BW_SIGMA,
            TENSE_FORMANT_TAU_MS,
            TENSE_BW_TAU_MS,
        ),
        "Timing: σ={:.3f}/{:.3f}、ポーズ {:.0f} ms でキレを出す.".format(
            TENSE_DURATION_DRIFT_STD,
            TENSE_DURATION_JITTER_STD,
            TENSE_PAUSE_SILENCE_MS,
        ),
    )
    return HumanizationProgram(
        name="tense",
        summary="張りのあるボイス向けプリセット",
        pitch=pitch,
        amplitude=amplitude,
        noise=noise,
        formant=formant,
        timing=timing,
        checklist=checklist,
    )


_PROGRAM_BUILDERS = {
    "speech": _speech_program,
    "breathy": _breathy_program,
    "tense": _tense_program,
}


def build_humanization_program(style: str = "speech") -> HumanizationProgram:
    """Construct a :class:`HumanizationProgram` for the requested style.

    Args:
        style: Identifier of the preset (``speech``, ``breathy``, or ``tense``).
    """

    normalized = style.strip().lower()
    if normalized not in _PROGRAM_BUILDERS:
        available = ", ".join(sorted(_PROGRAM_BUILDERS))
        raise ValueError(f"Unknown humanization style '{style}'. Available: {available}")
    return _PROGRAM_BUILDERS[normalized]()


def list_available_programs() -> List[str]:
    """Return the available humanisation program identifiers."""

    return sorted(_PROGRAM_BUILDERS.keys())
