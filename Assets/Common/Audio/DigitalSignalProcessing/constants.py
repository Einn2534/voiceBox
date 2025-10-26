"""Shared data structures and lookup tables for the DSP helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "DTYPE",
    "PEAK_DEFAULT",
    "EPS",
    "VOCAL_TRACT_LENGTH_CM",
    "DEFAULT_TRACT_SECTIONS",
    "SPEED_OF_SOUND_CM_S",
    "_NEUTRAL_TRACT_AREA_CM2",
    "NasalCoupling",
    "SpeakerProfile",
    "VOWEL_TABLE",
    "CV_TOKEN_MAP",
    "NASAL_TOKEN_MAP",
    "PAUSE_TOKEN",
    "_VALID_TOKENS",
    "_MAX_ROMAJI_TOKEN_LENGTH",
    "_KANA_BASE_MAP",
    "_VOICED_KANA_MAP",
    "_HAND_DAKUTEN_MAP",
    "_KANA_DIGRAPH_MAP",
    "_PUNCTUATION_CHARS",
    "GLIDE_ONSETS",
    "LIQUID_ONSETS",
    "NASAL_PRESETS",
]

DTYPE = np.float32
PEAK_DEFAULT = 0.9
EPS = 1e-12

VOCAL_TRACT_LENGTH_CM = 17.5
DEFAULT_TRACT_SECTIONS = 20
SPEED_OF_SOUND_CM_S = 35000.0

_NEUTRAL_TRACT_AREA_CM2 = np.array([
    2.2, 2.3, 2.5, 2.8, 3.1,
    3.4, 3.5, 3.3, 3.0, 2.7,
    2.4, 2.2, 1.9, 1.7, 1.5,
    1.3, 1.1, 0.95, 0.85, 0.8,
], dtype=np.float64)


@dataclass(frozen=True)
class NasalCoupling:
    """Configuration parameters describing the nasal/sinus branching network."""

    port_open: float = 1.0
    vowel_leak: float = 0.0
    nostril_area_cm2: float = 0.7
    nasal_cavity_length_cm: float = 14.5
    nasal_cavity_area_cm2: float = 2.4
    sinus_cavity_length_cm: float = 5.5
    sinus_cavity_area_cm2: float = 4.2
    sinus_coupling_area_cm2: float = 0.5
    loss_db_per_meter: float = 1.6


@dataclass(frozen=True)
class SpeakerProfile:
    """Minimal speaker description used by the procedural voice model."""

    nasal_coupling: NasalCoupling = field(default_factory=NasalCoupling)


# ---- 母音プリセット（成人中性声の目安） ----
VOWEL_TABLE: Dict[str, Dict[str, Sequence[float]]] = {
    'a': {'F': [800, 1150, 2900], 'BW': [90, 110, 150]},
    'i': {'F': [350, 2000, 3000], 'BW': [60, 100, 150]},
    'u': {'F': [325, 700, 2530],  'BW': [60, 90, 140]},
    'e': {'F': [400, 1700, 2600], 'BW': [70, 100, 150]},
    'o': {'F': [450, 800, 2830],  'BW': [80, 110, 150]},
}

# ローマ字トークン → (子音, 母音)
CV_TOKEN_MAP: Dict[str, Tuple[str, str]] = {
    'ka': ('k', 'a'),  'ki': ('k', 'i'),  'ku': ('k', 'u'),  'ke': ('k', 'e'),  'ko': ('k', 'o'),
    'sa': ('s', 'a'),  'shi': ('sh', 'i'), 'su': ('s', 'u'), 'se': ('s', 'e'), 'so': ('s', 'o'),
    'ta': ('t', 'a'),  'chi': ('ch', 'i'), 'tsu': ('ts', 'u'), 'te': ('t', 'e'), 'to': ('t', 'o'),
    'na': ('n', 'a'),  'ni': ('n', 'i'),  'nu': ('n', 'u'),  'ne': ('n', 'e'),  'no': ('n', 'o'),
    'ha': ('h', 'a'),  'hi': ('h', 'i'),  'fu': ('f', 'u'),  'he': ('h', 'e'),  'ho': ('h', 'o'),
    'ma': ('m', 'a'),  'mi': ('m', 'i'),  'mu': ('m', 'u'),  'me': ('m', 'e'),  'mo': ('m', 'o'),
    'ya': ('y', 'a'),  'yu': ('y', 'u'),  'yo': ('y', 'o'),
    'ra': ('r', 'a'),  'ri': ('r', 'i'),  'ru': ('r', 'u'),  're': ('r', 'e'),  'ro': ('r', 'o'),
    'wa': ('w', 'a'),  'wo': ('w', 'o'),
}

NASAL_TOKEN_MAP: Dict[str, str] = {'n': 'n', 'nn': 'n', 'm': 'm'}
PAUSE_TOKEN = 'pau'

_VALID_TOKENS = set(CV_TOKEN_MAP) | set(VOWEL_TABLE) | set(NASAL_TOKEN_MAP)
_MAX_ROMAJI_TOKEN_LENGTH = max(len(token) for token in _VALID_TOKENS)

# ひらがな変換・合字
_KANA_BASE_MAP: Dict[str, List[str]] = {
    'あ': ['a'], 'い': ['i'], 'う': ['u'], 'え': ['e'], 'お': ['o'],
    'か': ['ka'], 'き': ['ki'], 'く': ['ku'], 'け': ['ke'], 'こ': ['ko'],
    'さ': ['sa'], 'し': ['shi'], 'す': ['su'], 'せ': ['se'], 'そ': ['so'],
    'た': ['ta'], 'ち': ['chi'], 'つ': ['tsu'], 'て': ['te'], 'と': ['to'],
    'な': ['na'], 'に': ['ni'], 'ぬ': ['nu'], 'ね': ['ne'], 'の': ['no'],
    'は': ['ha'], 'ひ': ['hi'], 'ふ': ['fu'], 'へ': ['he'], 'ほ': ['ho'],
    'ま': ['ma'], 'み': ['mi'], 'む': ['mu'], 'め': ['me'], 'も': ['mo'],
    'や': ['ya'], 'ゆ': ['yu'], 'よ': ['yo'],
    'ら': ['ra'], 'り': ['ri'], 'る': ['ru'], 'れ': ['re'], 'ろ': ['ro'],
    'わ': ['wa'], 'ゐ': ['i'], 'ゑ': ['e'], 'を': ['wo'],
    'ん': ['n'],
    'ぁ': ['a'], 'ぃ': ['i'], 'ぅ': ['u'], 'ぇ': ['e'], 'ぉ': ['o'],
    'ゎ': ['wa'], 'ゕ': ['ka'], 'ゖ': ['ke'],
    'ゔ': ['u'],
}
_VOICED_KANA_MAP: Dict[str, str] = {
    'が': 'か', 'ぎ': 'き', 'ぐ': 'く', 'げ': 'け', 'ご': 'こ',
    'ざ': 'さ', 'じ': 'し', 'ず': 'す', 'ぜ': 'せ', 'ぞ': 'そ',
    'だ': 'た', 'ぢ': 'ち', 'づ': 'つ', 'で': 'て', 'ど': 'と',
    'ば': 'は', 'び': 'ひ', 'ぶ': 'ふ', 'べ': 'へ', 'ぼ': 'ほ',
}
_HAND_DAKUTEN_MAP: Dict[str, str] = {'ぱ': 'は', 'ぴ': 'ひ', 'ぷ': 'ふ', 'ぺ': 'へ', 'ぽ': 'ほ'}

_KANA_DIGRAPH_MAP: Dict[str, List[str]] = {
    'きゃ': ['ki', 'ya'], 'きゅ': ['ki', 'yu'], 'きょ': ['ki', 'yo'],
    'ぎゃ': ['ki', 'ya'], 'ぎゅ': ['ki', 'yu'], 'ぎょ': ['ki', 'yo'],
    'しゃ': ['shi', 'ya'], 'しゅ': ['shi', 'yu'], 'しょ': ['shi', 'yo'],
    'じゃ': ['shi', 'ya'], 'じゅ': ['shi', 'yu'], 'じょ': ['shi', 'yo'],
    'ちゃ': ['chi', 'ya'], 'ちゅ': ['chi', 'yu'], 'ちょ': ['chi', 'yo'],
    'にゃ': ['ni', 'ya'], 'にゅ': ['ni', 'yu'], 'にょ': ['ni', 'yo'],
    'ひゃ': ['hi', 'ya'], 'ひゅ': ['hi', 'yu'], 'ひょ': ['hi', 'yo'],
    'びゃ': ['hi', 'ya'], 'びゅ': ['hi', 'yu'], 'びょ': ['hi', 'yo'],
    'ぴゃ': ['hi', 'ya'], 'ぴゅ': ['hi', 'yu'], 'ぴょ': ['hi', 'yo'],
    'みゃ': ['mi', 'ya'], 'みゅ': ['mi', 'yu'], 'みょ': ['mi', 'yo'],
    'りゃ': ['ri', 'ya'], 'りゅ': ['ri', 'yu'], 'りょ': ['ri', 'yo'],
}

_PUNCTUATION_CHARS = set('、。，．,.!?！？；：:;…‥・「」『』（）()[]{}<>')
_PUNCTUATION_CHARS.update({'"', "'", '“', '”', '‘', '’', '—'})

GLIDE_ONSETS: Dict[str, Dict[str, Sequence[float]]] = {
    'w': {
        'a': [420.0, 1000.0, 2300.0],
        'i': [400.0, 1200.0, 2500.0],
        'u': [380.0,  900.0, 2200.0],
        'e': [410.0, 1100.0, 2350.0],
        'o': [420.0,  950.0, 2300.0],
    },
    'y': {
        'a': [380.0, 2600.0, 3300.0],
        'i': [360.0, 2900.0, 3500.0],
        'u': [370.0, 2650.0, 3300.0],
        'e': [380.0, 2800.0, 3400.0],
        'o': [380.0, 2500.0, 3200.0],
    },
}

LIQUID_ONSETS: Dict[str, Sequence[float]] = {
    'a': [480.0, 1350.0, 2100.0],
    'i': [380.0, 1800.0, 2200.0],
    'u': [420.0, 1500.0, 2100.0],
    'e': [430.0, 1600.0, 2200.0],
    'o': [460.0, 1400.0, 2150.0],
}

NASAL_PRESETS: Dict[str, Dict[str, Sequence[float]]] = {
    'n': {'F': [250.0, 1900.0, 2800.0], 'BW': [80.0, 160.0, 220.0]},
    'm': {'F': [220.0, 1600.0, 2500.0], 'BW': [70.0, 150.0, 210.0]},
}
