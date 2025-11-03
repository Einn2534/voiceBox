"""Modularised digital signal processing helpers for the voiceBox project."""
from __future__ import annotations

from .constants import (
    DTYPE,
    EPS,
    PEAK_DEFAULT,
    CV_TOKEN_MAP,
    NASAL_TOKEN_MAP,
    PAUSE_TOKEN,
    VOWEL_TABLE,
    NasalCoupling,
    SpeakerProfile,
)
from .HumanizationProgram import (
    HumanizationProgram,
    build_humanization_program,
    list_available_programs,
)
from .synthesis import (
    synth_affricate,
    synth_cv,
    synth_cv_to_wav,
    synth_fricative,
    synth_nasal,
    synth_phrase_to_wav,
    synth_plosive,
    synth_token_sequence,
    synth_tokens_to_wav,
    synth_vowel,
    synth_vowel_with_onset,
    synth_vowel_with_program,
)
from .text import normalize_token_sequence, text_to_tokens
from .io import write_wav

__all__ = [
    "DTYPE",
    "EPS",
    "PEAK_DEFAULT",
    "CV_TOKEN_MAP",
    "NASAL_TOKEN_MAP",
    "PAUSE_TOKEN",
    "VOWEL_TABLE",
    "NasalCoupling",
    "SpeakerProfile",
    "text_to_tokens",
    "normalize_token_sequence",
    "HumanizationProgram",
    "build_humanization_program",
    "list_available_programs",
    "synth_vowel",
    "synth_vowel_with_program",
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
    "write_wav",
]
