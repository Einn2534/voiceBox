# Created on 2024-05-08
# Author: ChatGPT
# Description: Digital signal processing helpers for voice synthesis.
"""Signal processing helpers for the GIAN vocal synthesis project (refactored)."""
from __future__ import annotations

import os
import wave
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from math import pi, sin, cos, exp, sqrt

# =======================
# Constants / Globals
# =======================

DTYPE = np.float32
PEAK_DEFAULT = 0.9
EPS = 1e-12
MAX_FORMANT_PERTURBATION = 0.65
DEFAULT_SPEED_OF_SOUND_CM_S = 34300.0
DEFAULT_LIP_RADIUS_CM = 0.9
LIP_END_CORRECTION_FACTOR = 0.6
DEFAULT_FORMANT_COUNT = 3
FORMANT_KEY = 'F'
BANDWIDTH_KEY = 'BW'
DEFAULT_DRIFT_CENTS = 12.0
DEFAULT_DRIFT_RETURN_RATE = 0.35
DEFAULT_VIBRATO_DEPTH_CENTS = 0.0
DEFAULT_VIBRATO_FREQUENCY_HZ = 5.5
DEFAULT_TREMOR_DEPTH_CENTS = 0.0
DEFAULT_TREMOR_FREQUENCY_HZ = 8.5


# =======================
# Vowel design helpers
# =======================


@dataclass(frozen=True)
class TubeDesign:
    """Quarter-wave tube model approximating the vocal tract."""

    length_cm: float
    lip_radius_cm: float = DEFAULT_LIP_RADIUS_CM
    speed_of_sound_cm_s: float = DEFAULT_SPEED_OF_SOUND_CM_S
    formant_count: int = DEFAULT_FORMANT_COUNT

    def compute_formants(self) -> List[float]:
        """Return the baseline 1/4 wave resonances in Hz.

        Returns:
            List[float]: Quarter-wave resonant frequencies ordered by mode.
        """
        effective_length = self.length_cm + LIP_END_CORRECTION_FACTOR * self.lip_radius_cm
        return [
            (2 * index - 1) * self.speed_of_sound_cm_s / (4.0 * effective_length)
            for index in range(1, self.formant_count + 1)
        ]


@dataclass(frozen=True)
class ConstrictionSpec:
    """Perturbation description for a localised constriction."""

    position_ratio: float
    intensity: float
    focus: Optional[Sequence[float]] = None

    def clamped_position(self) -> float:
        """Clamp the constriction position inside [0, 1]."""
        return float(max(0.0, min(1.0, self.position_ratio)))


@dataclass(frozen=True)
class VowelDesignSpec:
    """Declarative formant design based on tube length and constrictions."""

    vowel: str
    tube: TubeDesign
    constrictions: Sequence[ConstrictionSpec]
    bandwidths: Sequence[float]

    def compute_formants(self) -> List[float]:
        """Calculate perturbed formant frequencies for the vowel."""
        frequencies = list(self.tube.compute_formants())
        for constriction in self.constrictions:
            _apply_constriction_perturbation(frequencies, constriction)
        return [float(freq) for freq in frequencies]


def _apply_constriction_perturbation(
    frequencies: List[float],
    constriction: ConstrictionSpec,
) -> None:
    """Apply the perturbation contributed by a single constriction.

    Args:
        frequencies (List[float]): Formant frequency list to update in place.
        constriction (ConstrictionSpec): Constriction parameters applied to the tube.
    """

    position = constriction.clamped_position()
    focus = list(constriction.focus) if constriction.focus else None
    for index, base_freq in enumerate(frequencies):
        mode = index + 1
        angle = (2 * mode - 1) * pi * position / 2.0
        velocity = sin(angle)
        weight = 1.0 - 2.0 * velocity * velocity
        weight = max(-1.0, min(1.0, weight))
        if focus and index < len(focus):
            weight *= float(focus[index])
        delta = constriction.intensity * weight
        delta = max(-MAX_FORMANT_PERTURBATION, min(MAX_FORMANT_PERTURBATION, delta))
        frequencies[index] = base_freq * (1.0 + delta)


def compute_quarter_wave_formants(
    length_cm: float,
    *,
    lip_radius_cm: float = DEFAULT_LIP_RADIUS_CM,
    speed_of_sound_cm_s: float = DEFAULT_SPEED_OF_SOUND_CM_S,
    formant_count: int = DEFAULT_FORMANT_COUNT,
) -> List[float]:
    """Return quarter-wave formants for a simple tube configuration.

    Args:
        length_cm (float): Physical length of the vocal-tract tube in centimetres.
        lip_radius_cm (float): Lip radius used for end correction.
        speed_of_sound_cm_s (float): Propagation speed of sound in cm/s.
        formant_count (int): Number of formant frequencies to generate.

    Returns:
        List[float]: Computed baseline formants in Hz.
    """

    design = TubeDesign(
        length_cm=length_cm,
        lip_radius_cm=lip_radius_cm,
        speed_of_sound_cm_s=speed_of_sound_cm_s,
        formant_count=formant_count,
    )
    return design.compute_formants()


def build_vowel_table(
    specs: Dict[str, VowelDesignSpec],
) -> Dict[str, Dict[str, Sequence[float]]]:
    """Materialise a vowel table from design specifications.

    Args:
        specs (Dict[str, VowelDesignSpec]): Mapping from vowel symbols to designs.

    Returns:
        Dict[str, Dict[str, Sequence[float]]]: Runtime lookup table with
            frequency and bandwidth sequences.
    """

    table: Dict[str, Dict[str, Sequence[float]]] = {}
    for vowel, spec in specs.items():
        table[vowel] = {
            FORMANT_KEY: spec.compute_formants(),
            BANDWIDTH_KEY: list(spec.bandwidths),
        }
    return table


def register_vowel_design(spec: VowelDesignSpec) -> None:
    """Add or replace a vowel design and refresh the global formant table.

    Args:
        spec (VowelDesignSpec): Declarative vowel design definition to store.
    """

    DEFAULT_VOWEL_LIBRARY[spec.vowel] = spec
    VOWEL_TABLE[spec.vowel] = {
        FORMANT_KEY: spec.compute_formants(),
        BANDWIDTH_KEY: list(spec.bandwidths),
    }


def get_vowel_design(vowel: str) -> VowelDesignSpec:
    """Retrieve the declarative design spec for a vowel symbol.

    Args:
        vowel (str): Symbol key registered in the vowel library.

    Returns:
        VowelDesignSpec: Stored design specification.
    """

    if vowel not in DEFAULT_VOWEL_LIBRARY:
        raise KeyError(f'Unknown vowel design requested: {vowel}')
    return DEFAULT_VOWEL_LIBRARY[vowel]


def rebuild_vowel_table() -> None:
    """Recompute the global formant lookup from the registered designs."""

    VOWEL_TABLE.clear()
    VOWEL_TABLE.update(build_vowel_table(DEFAULT_VOWEL_LIBRARY))


def list_registered_vowels() -> Tuple[str, ...]:
    """Return the registered vowel symbols ordered alphabetically.

    Returns:
        Tuple[str, ...]: Sorted tuple of vowel identifiers available in the
            current vowel table.
    """

    return tuple(sorted(VOWEL_TABLE))


# ---- 母音プリセット（成人中性声の目安） ----
DEFAULT_VOWEL_LIBRARY: Dict[str, VowelDesignSpec] = {
    'a': VowelDesignSpec(
        vowel='a',
        tube=TubeDesign(length_cm=14.5),
        constrictions=(
            ConstrictionSpec(0.2, 0.42, (1.3, 0.6, 0.2)),
            ConstrictionSpec(0.45, 0.36, (0.0, 1.55, 0.45)),
        ),
        bandwidths=(90.0, 110.0, 150.0),
    ),
    'i': VowelDesignSpec(
        vowel='i',
        tube=TubeDesign(length_cm=17.5),
        constrictions=(
            ConstrictionSpec(0.71, 0.435, None),
            ConstrictionSpec(0.38, 0.22, (0.0, 0.0, 1.2)),
        ),
        bandwidths=(60.0, 100.0, 150.0),
    ),
    'u': VowelDesignSpec(
        vowel='u',
        tube=TubeDesign(length_cm=28.0),
        constrictions=(
            ConstrictionSpec(0.8, -0.6, (0.45, 1.1, 0.3)),
            ConstrictionSpec(0.15, -0.12, (1.0, 0.8, 0.6)),
            ConstrictionSpec(0.38, 0.5, (0.0, 0.0, 1.5)),
            ConstrictionSpec(0.32, 0.45, (0.0, 0.0, 1.4)),
        ),
        bandwidths=(60.0, 90.0, 140.0),
    ),
    'e': VowelDesignSpec(
        vowel='e',
        tube=TubeDesign(length_cm=19.0),
        constrictions=(
            ConstrictionSpec(0.6, 0.28, (0.5, 1.2, 0.45)),
            ConstrictionSpec(0.38, 0.26, (0.0, 0.0, 1.4)),
        ),
        bandwidths=(70.0, 100.0, 150.0),
    ),
    'o': VowelDesignSpec(
        vowel='o',
        tube=TubeDesign(length_cm=21.0),
        constrictions=(
            ConstrictionSpec(0.35, 0.18, (1.0, 0.6, 0.3)),
            ConstrictionSpec(0.74, -0.32, (0.45, 1.05, 0.3)),
            ConstrictionSpec(0.38, 0.22, (0.0, 0.0, 1.35)),
            ConstrictionSpec(0.42, 0.12, (0.0, 0.0, 1.1)),
        ),
        bandwidths=(80.0, 110.0, 150.0),
    ),
}

VOWEL_TABLE: Dict[str, Dict[str, Sequence[float]]] = {}
rebuild_vowel_table()

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

# =======================
# Text helpers
# =======================

def _normalize_to_hiragana(text: str) -> str:
    """NFKC normalize and convert katakana to hiragana."""
    normalized = unicodedata.normalize('NFKC', text)
    buf: List[str] = []
    for ch in normalized:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F3:           # カタカナ → ひらがな
            buf.append(chr(code - 0x60))
        elif ch == 'ヴ':
            buf.append('ゔ')
        elif ch == 'ヵ':
            buf.append('ゕ')
        elif ch == 'ヶ':
            buf.append('ゖ')
        else:
            buf.append(ch)
    return ''.join(buf)


def _append_pause(tokens: List[str]) -> None:
    """トークン列の末尾に休符 'pau' を付与する（既に末尾が休符なら何もしない）。
    
    Args:
        tokens (List[str]): CV/母音/鼻音/休符のトークン列（破壊的に更新）。
    """
    if tokens and tokens[-1] == PAUSE_TOKEN:
        return
    tokens.append(PAUSE_TOKEN)


def _extend_last_vowel(tokens: List[str]) -> None:
    """長音記号の処理：直前の母音を延長（なければ何もしない）"""
    for t in reversed(tokens):
        if t == PAUSE_TOKEN:
            continue
        if t in VOWEL_TABLE:
            tokens.append(t)
            return
        if t in CV_TOKEN_MAP:
            tokens.append(CV_TOKEN_MAP[t][1])
            return
    # fallback: do nothing


def _parse_romaji_sequence(seq: str) -> List[str]:
    """ASCII ローマ字を CV/母音/鼻音トークン列へ分割"""
    tokens: List[str] = []
    i, n = 0, len(seq)
    while i < n:
        matched = False
        max_len = min(_MAX_ROMAJI_TOKEN_LENGTH, n - i)
        for size in range(max_len, 0, -1):
            cand = seq[i:i + size]
            if cand in _VALID_TOKENS:
                tokens.append(cand)
                i += size
                matched = True
                break
        if matched:
            continue
        ch = seq[i]
        if ch in 'aiueo':
            tokens.append(ch)
        elif ch == 'n':
            tokens.append('n')
        i += 1
    return tokens


def text_to_tokens(text: str) -> List[str]:
    """Convert arbitrary text to the synthesizer token sequence."""
    if not text:
        return []

    tokens: List[str] = []
    roma_buf: List[str] = []

    def flush_buf() -> None:
        """ローマ字の一時バッファを解析してトークン列にフラッシュする内部ヘルパー。"""
        nonlocal roma_buf
        if roma_buf:
            tokens.extend(_parse_romaji_sequence(''.join(roma_buf)))
            roma_buf = []

    normalized = _normalize_to_hiragana(text)
    n, pos = len(normalized), 0

    while pos < n:
        ch = normalized[pos]

        # ASCII ローマ字をバッファリング
        if ch.isascii() and ch.isalpha():
            roma_buf.append(ch.lower())
            pos += 1
            continue

        flush_buf()

        # 空白・句読点 → ポーズ
        if ch.isspace() or ch in _PUNCTUATION_CHARS:
            _append_pause(tokens)
            pos += 1
            continue

        # 長音記号
        if ch == 'ー':
            _extend_last_vowel(tokens)
            pos += 1
            continue

        # 促音（簡易：無音ポーズ）
        if ch == 'っ':
            _append_pause(tokens)
            pos += 1
            continue

        # 拗音二文字
        nxt = normalized[pos + 1] if pos + 1 < n else ''
        pair = ch + nxt
        if pair in _KANA_DIGRAPH_MAP:
            tokens.extend(_KANA_DIGRAPH_MAP[pair])
            pos += 2
            continue

        # 単音
        if ch in _KANA_BASE_MAP:
            tokens.extend(_KANA_BASE_MAP[ch])
            pos += 1
            continue

        # 濁音・半濁音
        if ch in _VOICED_KANA_MAP:
            base = _VOICED_KANA_MAP[ch]
            tokens.extend(_KANA_BASE_MAP.get(base, []))
            pos += 1
            continue
        if ch in _HAND_DAKUTEN_MAP:
            base = _HAND_DAKUTEN_MAP[ch]
            tokens.extend(_KANA_BASE_MAP.get(base, []))
            pos += 1
            continue

        # 数字 → 区切り
        if ch.isdigit():
            _append_pause(tokens)
            pos += 1
            continue

        # 未対応はスキップ
        pos += 1

    flush_buf()

    # 圧縮: 連続ポーズ削除／先頭末尾ポーズ除去
    out: List[str] = []
    for t in tokens:
        if t == PAUSE_TOKEN and (not out or out[-1] == PAUSE_TOKEN):
            continue
        out.append(t)
    if out and out[0] == PAUSE_TOKEN:
        out = out[1:]
    if out and out[-1] == PAUSE_TOKEN:
        out = out[:-1]
    return out


def normalize_token_sequence(tokens: Optional[Any]) -> List[str]:
    """Normalize arbitrary token containers into a flat list of strings.

    Args:
        tokens (Optional[Any]): Token-like sequence returned from synthesis helpers.

    Returns:
        List[str]: A flat list representation that is safe for truth-value testing.
    """
    if tokens is None:
        return []
    if isinstance(tokens, list):
        return [str(token) for token in tokens]
    if isinstance(tokens, np.ndarray):
        flat_tokens = tokens.ravel().tolist()
        return [str(token) for token in flat_tokens]
    if isinstance(tokens, tuple):
        return [str(token) for token in tokens]
    if isinstance(tokens, str):
        return [tokens]

    try:
        iterable = list(tokens)
    except TypeError:
        return [str(tokens)]
    return [str(token) for token in iterable]

# =======================
# Utilities (numeric)
# =======================

def _ms_to_samples(ms: float, sr: int) -> int:
    """ミリ秒をサンプル数に変換（切り捨て・下限0）"""
    if ms <= 0:
        return 0
    return int(sr * (ms / 1000.0))


def _db_to_lin(db: float) -> float:
    """dB → 線形ゲイン"""
    return 10.0 ** (db / 20.0)


def _ensure_array(x: np.ndarray, *, dtype=DTYPE) -> np.ndarray:
    """型・連続性を整える"""
    if x.dtype != dtype:
        return x.astype(dtype, copy=False)
    return x


def _normalize_peak(sig: np.ndarray, target: float = PEAK_DEFAULT) -> np.ndarray:
    """ピーク正規化（無音ガード）"""
    sig = _ensure_array(sig)
    peak = float(np.max(np.abs(sig)) + EPS)
    scale = target / peak if peak > 0 else 1.0
    return (sig * scale).astype(DTYPE, copy=False)


def _apply_fade(sig: np.ndarray, sr: int, *, attack_ms: float = 5.0, release_ms: float = 8.0) -> np.ndarray:
    """線形アタック／リリース"""
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
    """息成分を加算（level_db<0で適用）"""
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

# =======================
# Filters
# =======================

def _bandpass_biquad_coeff(f0: float, Q: float, sr: int) -> Tuple[float, float, float, float, float]:
    """RBJ系BPF係数（ピークゲイン一定タイプ）"""
    w0 = 2.0 * pi * f0 / sr
    alpha = sin(w0) / (2.0 * max(Q, 0.5))
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos(w0)
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _biquad_process(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    """単一Biquadフィルタ"""
    x = _ensure_array(x)
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def _one_pole_lp(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """一次ローパス"""
    x = _ensure_array(x)
    if len(x) == 0:
        return x
    decay = exp(-2.0 * pi * cutoff / sr)
    y = np.empty_like(x)
    s = 0.0
    for i, xi in enumerate(x):
        s = (1 - decay) * xi + decay * s
        y[i] = s
    return y


def _lip_radiation(x: np.ndarray) -> np.ndarray:
    """唇放射の近似：微分"""
    x = _ensure_array(x)
    n = len(x)
    if n == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - x[:-1]
    return y

# =======================
# Sources / Noise
# =======================

def _glottal_source(
    f0: float,
    dur_s: float,
    sr: int,
    jitter_cents: float = 10.0,
    shimmer_db: float = 0.8,
    *,
    drift_cents: float = DEFAULT_DRIFT_CENTS,
    drift_return_rate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibrato_depth_cents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibrato_frequency_hz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremor_depth_cents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremor_frequency_hz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
) -> np.ndarray:
    """鋸波＋jitter/shimmer+OUドリフト+ビブラート合成。"""
    rng = np.random.default_rng()
    n = max(0, int(dur_s * sr))
    if n == 0:
        return np.zeros(0, dtype=DTYPE)

    # jitter（AR一次）
    jn = rng.standard_normal(n).astype(DTYPE)
    pole = 0.999
    js = np.empty_like(jn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * jn[i]
        js[i] = acc
    jitter_semitones = (jitter_cents / 100.0) * js

    drift = np.zeros(n, dtype=DTYPE)
    if drift_cents > 0.0 and drift_return_rate > 0.0:
        theta = float(drift_return_rate)
        dt = 1.0 / float(sr)
        sigma = float(drift_cents) * sqrt(2.0 * theta)
        sqrt_dt = sqrt(dt)
        state = 0.0
        for i in range(n):
            state += theta * (-state) * dt + sigma * sqrt_dt * rng.standard_normal()
            drift[i] = state
    drift_semitones = drift / 100.0

    t = np.arange(n, dtype=np.float64) / float(sr)
    vibrato_semitones = np.zeros(n, dtype=np.float64)
    if vibrato_depth_cents != 0.0 and vibrato_frequency_hz > 0.0:
        vib_phase = rng.uniform(0.0, 2.0 * pi)
        vibrato_semitones = (
            (vibrato_depth_cents / 100.0)
            * np.sin(2.0 * pi * vibrato_frequency_hz * t + vib_phase)
        )

    tremor_semitones = np.zeros(n, dtype=np.float64)
    if tremor_depth_cents != 0.0 and tremor_frequency_hz > 0.0:
        tremor_phase = rng.uniform(0.0, 2.0 * pi)
        tremor_semitones = (
            (tremor_depth_cents / 100.0)
            * np.sin(2.0 * pi * tremor_frequency_hz * t + tremor_phase)
        )

    composite_semitones = (
        jitter_semitones.astype(np.float64)
        + drift_semitones.astype(np.float64)
        + vibrato_semitones
        + tremor_semitones
    )
    inst_f = f0 * (2.0 ** (composite_semitones / 12.0))

    # 位相加算で鋸波
    phase = np.cumsum(2.0 * pi * inst_f / sr, dtype=np.float64)
    saw = (2.0 * ((phase / (2.0 * pi)) % 1.0) - 1.0).astype(DTYPE)

    # shimmer（AR一次）
    sn = rng.standard_normal(n).astype(DTYPE)
    ss = np.empty_like(sn)
    acc = 0.0
    for i in range(n):
        acc = pole * acc + (1.0 - pole) * sn[i]
        ss[i] = acc
    amp = (_db_to_lin(shimmer_db) ** ss).astype(DTYPE)  # 10**((dB/20)*noise) と等価形

    raw = saw * amp

    # 簡易 HP 抑制
    decay = exp(-2.0 * pi * 800.0 / sr)
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
    """帯域ノイズ生成"""
    rng = np.random.default_rng()
    n = max(0, int(dur_s * sr))
    if n == 0:
        return np.zeros(0, dtype=DTYPE)
    noise = rng.standard_normal(n).astype(DTYPE)
    b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(center, Q, sr)
    y = _biquad_process(noise, b0, b1, b2, a1, a2)
    return _lip_radiation(y) if use_lip_radiation else y

# =======================
# Core Synthesis
# =======================

def _apply_formant_filters(src: np.ndarray, formants: Sequence[float], bws: Sequence[float], sr: int) -> np.ndarray:
    """フォルマント設定に基づいて帯域通過を連結"""
    out = np.zeros_like(src, dtype=DTYPE)
    for f, bw in zip(formants, bws):
        Q = max(0.5, float(f) / float(bw))
        b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(float(f), Q, sr)
        out += _biquad_process(src, b0, b1, b2, a1, a2)
    return _lip_radiation(out)


def synth_vowel(
    vowel: str = 'a',
    f0: float = 120.0,
    durationSeconds: float = 1.0,
    sampleRate: int = 22050,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
    *,
    driftCents: float = DEFAULT_DRIFT_CENTS,
    driftReturnRate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibratoDepthCents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibratoFrequencyHz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremorDepthCents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremorFrequencyHz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
) -> np.ndarray:
    """母音合成"""
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
    y = _apply_formant_filters(src, spec[FORMANT_KEY], spec[BANDWIDTH_KEY], sampleRate)
    y = _add_breath_noise(y, breathLevelDb)
    return _normalize_peak(y, PEAK_DEFAULT)


def synth_fricative(
    consonant: str = 's',
    durationSeconds: float = 0.16,
    sampleRate: int = 22050,
    levelDb: float = -12.0,
) -> np.ndarray:
    """無声摩擦音の簡易実装。"""
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
    gain = _db_to_lin(levelDb)
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
    """無声破裂音（t / k）"""
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

    # 閉鎖
    closure_seg = np.zeros(_ms_to_samples(closure, sampleRate), dtype=DTYPE)

    # 破裂
    burst_len = _ms_to_samples(burst, sampleRate)
    burst_noise = _gen_band_noise(burst_len / sampleRate, sampleRate, b_center, b_Q, use_lip_radiation=False)
    tau = max(1.0, burst_len / 4.0)
    env = np.exp(-np.arange(burst_len, dtype=DTYPE) / tau)
    burst_noise = burst_noise[:burst_len] * env

    # 後続の息
    asp_len = _ms_to_samples(asp, sampleRate)
    asp_noise = _gen_band_noise(asp_len / sampleRate, sampleRate, a_center, a_Q, use_lip_radiation=False)
    asp_noise = _apply_fade(asp_noise, sampleRate, attack_ms=3, release_ms=18)
    asp_noise = _one_pole_lp(asp_noise, cutoff=4500.0, sr=sampleRate)

    y = np.concatenate([closure_seg, burst_noise, asp_noise]).astype(DTYPE)
    y *= _db_to_lin(levelDb)
    return _normalize_peak(y, 0.6)


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, *, overlap_ms: float = 30.0) -> np.ndarray:
    """a → b をオーバーラップ結合"""
    a = _ensure_array(a)
    b = _ensure_array(b)
    ov = min(_ms_to_samples(overlap_ms, sr), len(a), len(b))
    if ov <= 0:
        return np.concatenate([a, b]).astype(DTYPE)

    fade_out = np.linspace(1.0, 0.0, ov, dtype=DTYPE)
    fade_in = 1.0 - fade_out
    head = a[:-ov]
    tail = a[-ov:] * fade_out + b[:ov] * fade_in
    rest = b[ov:]
    return np.concatenate([head, tail, rest]).astype(DTYPE)


def _synth_vowel_fixed(
    formants: Sequence[float],
    bws: Sequence[float],
    f0: float,
    dur_s: float,
    sr: int,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
    driftCents: float = DEFAULT_DRIFT_CENTS,
    driftReturnRate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibratoDepthCents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibratoFrequencyHz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremorDepthCents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremorFrequencyHz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
) -> np.ndarray:
    """与えたフォルマントで固定合成（短区間）"""
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
    y = _apply_formant_filters(src, formants, bws, sr)
    y = _add_breath_noise(y, breathLevelDb)
    return _normalize_peak(y, PEAK_DEFAULT)


def _pre_emphasis(x: np.ndarray, coefficient: float = 0.85) -> np.ndarray:
    """y[n] = x[n] - a*x[n-1]（高域をわずかに強調）"""
    x = _ensure_array(x)
    n = len(x)
    if n == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coefficient * x[:-1]
    return y


def synth_vowel_with_onset(
    vowel: str,
    f0: float,
    sampleRate: int,
    totalMilliseconds: int = 240,
    onsetMilliseconds: int = 45,
    onsetFormants: Optional[Sequence[float]] = None,
    onsetBandwidthScale: float = 0.85,
    *,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
    driftCents: float = DEFAULT_DRIFT_CENTS,
    driftReturnRate: float = DEFAULT_DRIFT_RETURN_RATE,
    vibratoDepthCents: float = DEFAULT_VIBRATO_DEPTH_CENTS,
    vibratoFrequencyHz: float = DEFAULT_VIBRATO_FREQUENCY_HZ,
    tremorDepthCents: float = DEFAULT_TREMOR_DEPTH_CENTS,
    tremorFrequencyHz: float = DEFAULT_TREMOR_FREQUENCY_HZ,
) -> np.ndarray:
    """母音先頭だけフォルマント遷移を与える簡易版"""
    spec = VOWEL_TABLE[vowel]
    targetF, targetBW = spec[FORMANT_KEY], spec[BANDWIDTH_KEY]
    if not onsetFormants:
        return synth_vowel(
            vowel=vowel,
            f0=f0,
            durationSeconds=totalMilliseconds / 1000.0,
            sampleRate=sampleRate,
            jitterCents=jitterCents,
            shimmerDb=shimmerDb,
            breathLevelDb=breathLevelDb,
            driftCents=driftCents,
            driftReturnRate=driftReturnRate,
            vibratoDepthCents=vibratoDepthCents,
            vibratoFrequencyHz=vibratoFrequencyHz,
            tremorDepthCents=tremorDepthCents,
            tremorFrequencyHz=tremorFrequencyHz,
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
        jitterCents,
        shimmerDb,
        breathLevelDb,
        driftCents,
        driftReturnRate,
        vibratoDepthCents,
        vibratoFrequencyHz,
        tremorDepthCents,
        tremorFrequencyHz,
    )
    onset = _apply_fade(onset, sampleRate, attack_ms=6.0, release_ms=min(12.0, onset_ms * 0.5))

    if sustain_ms <= 0:
        return onset

    ov_ms = min(max(6.0, onset_ms * 0.45), 14.0, float(sustain_ms))
    rest_len_ms = sustain_ms + max(0.0, ov_ms)

    sustain = _synth_vowel_fixed(
        targetF,
        targetBW,
        f0,
        rest_len_ms / 1000.0,
        sampleRate,
        jitterCents,
        shimmerDb,
        breathLevelDb,
        driftCents,
        driftReturnRate,
        vibratoDepthCents,
        vibratoFrequencyHz,
        tremorDepthCents,
        tremorFrequencyHz,
    )
    sustain = _apply_fade(sustain, sampleRate, attack_ms=4.0, release_ms=12.0)

    return _crossfade(onset, sustain, sampleRate, overlap_ms=ov_ms)

# ===== Glide/Liquid onsets =====

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

# =======================
# Composition (CV/フレーズ)
# =======================

def synth_affricate(
    consonant: str = 'ch',
    sampleRate: int = 22050,
    closureMilliseconds: Optional[float] = None,
    fricativeMilliseconds: Optional[float] = None,
    levelDb: float = -11.0,
) -> np.ndarray:
    """破擦音 /ch/, /ts/ の簡易版"""
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


NASAL_PRESETS: Dict[str, Dict[str, Sequence[float]]] = {
    'n': {FORMANT_KEY: [250.0, 1900.0, 2800.0], BANDWIDTH_KEY: [80.0, 160.0, 220.0]},
    'm': {FORMANT_KEY: [220.0, 1600.0, 2500.0], BANDWIDTH_KEY: [70.0, 150.0, 210.0]},
}


def synth_nasal(
    consonant: str = 'n',
    f0: float = 120.0,
    durationMilliseconds: float = 90.0,
    sampleRate: int = 22050
) -> np.ndarray:
    """簡易的な鼻音 /n/, /m/"""
    c = consonant.lower()
    if c not in NASAL_PRESETS:
        raise ValueError("synth_nasal: supported nasals are 'n' or 'm'.")

    dur_s = max(20.0, float(durationMilliseconds)) / 1000.0
    spec = NASAL_PRESETS[c]
    y = _synth_vowel_fixed(spec[FORMANT_KEY], spec[BANDWIDTH_KEY], f0, dur_s, sampleRate,
                           jitterCents=4.0, shimmerDb=0.4, breathLevelDb=-38.0)
    y = _apply_fade(y, sampleRate,
                    attack_ms=8.0 if c == 'n' else 10.0,
                    release_ms=22.0 if c == 'n' else 28.0)
    gain = 0.6 if c == 'n' else 0.7
    return _normalize_peak(y * gain, 0.5)


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
) -> np.ndarray:
    """子音 + 母音（50音相当までカバー）"""
    c = cons.lower()
    v = vowel.lower()
    if v not in VOWEL_TABLE:
        raise ValueError("Unknown vowel for synth_cv")

    head = np.zeros(_ms_to_samples(preMilliseconds, sampleRate), dtype=DTYPE)

    if c in ('s', 'sh'):
        level = {'s': -14.0, 'sh': -15.0}[c]
        default_ms = {'s': 160.0, 'sh': 150.0}[c]
        fric_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else default_ms
        fric = synth_fricative(c, durationSeconds=fric_ms / 1000.0, sampleRate=sampleRate, levelDb=level)
        vow = synth_vowel(vowel=v, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        out = _crossfade(fric, vow, sampleRate, overlap_ms=max(24, overlapMilliseconds))

    elif c in ('h', 'f'):
        default_ms = {'h': 190.0, 'f': 170.0}[c]
        cons_len = float(consonantMilliseconds) if consonantMilliseconds is not None else default_ms
        cons_len = max(0.0, cons_len)
        sil = np.zeros(_ms_to_samples(cons_len, sampleRate), dtype=DTYPE)
        vow = synth_vowel(vowel=v, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)

        req_ov = float(overlapMilliseconds) if consonantMilliseconds is not None else 12.0
        eff_ov = min(12.0, cons_len, max(0.0, req_ov))
        out = _crossfade(sil, vow, sampleRate, overlap_ms=eff_ov) if eff_ov > 0 else np.concatenate([sil, vow]).astype(DTYPE)
        out = _apply_fade(out, sampleRate, attack_ms=8, release_ms=12)

    elif c in ('t', 'k'):
        plo = synth_plosive(c, sampleRate=sampleRate)
        if useOnsetTransition and v == 'a':
            onsetF = [800, 1800, 3000] if c == 't' else [800, 2200, 2400]
            vow = synth_vowel_with_onset('a', f0, sampleRate, totalMilliseconds=vowelMilliseconds,
                                         onsetMilliseconds=45, onsetFormants=onsetF)
        else:
            vow = synth_vowel(vowel=v, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)

        try:
            req = int(overlapMilliseconds)
            ov = max(6, min(req, 12))
        except Exception:
            ov = 8 if c == 't' else 10
        out = _crossfade(plo, vow, sampleRate, overlap_ms=ov)

    elif c in ('ch', 'ts'):
        aff = synth_affricate(c, sampleRate=sampleRate)
        vow = synth_vowel(vowel=v, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        out = _crossfade(aff, vow, sampleRate, overlap_ms=max(12, min(overlapMilliseconds, 18)))

    elif c == 'n':
        nasal_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 90.0
        nasal = synth_nasal('n', f0=f0, durationMilliseconds=nasal_ms, sampleRate=sampleRate)
        vow = synth_vowel(vowel=v, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        out = _crossfade(nasal, vow, sampleRate, overlap_ms=max(25, overlapMilliseconds))

    elif c == 'm':
        nasal_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 110.0
        nasal = synth_nasal('m', f0=f0, durationMilliseconds=nasal_ms, sampleRate=sampleRate)
        vow = synth_vowel(vowel=v, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        out = _crossfade(nasal, vow, sampleRate, overlap_ms=max(28, overlapMilliseconds))

    elif c == 'w':
        onset_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 44.0
        onset_ms = max(24.0, min(onset_ms, float(vowelMilliseconds) - 8.0))
        onsetF = GLIDE_ONSETS['w'].get(v, GLIDE_ONSETS['w']['a'])
        out = synth_vowel_with_onset(v, f0, sampleRate, totalMilliseconds=vowelMilliseconds,
                                     onsetMilliseconds=int(onset_ms), onsetFormants=onsetF, onsetBandwidthScale=1.18)
        out = _pre_emphasis(out, coefficient=0.86)
        out = _add_breath_noise(out, level_db=-36.0)
        out = _normalize_peak(out, PEAK_DEFAULT)

    elif c == 'y':
        onset_ms = float(consonantMilliseconds) if consonantMilliseconds is not None else 40.0
        onset_ms = max(24.0, min(onset_ms, float(vowelMilliseconds) - 10.0))
        onsetF = GLIDE_ONSETS['y'].get(v, GLIDE_ONSETS['y']['a'])
        out = synth_vowel_with_onset(v, f0, sampleRate, totalMilliseconds=vowelMilliseconds,
                                     onsetMilliseconds=int(onset_ms), onsetFormants=onsetF, onsetBandwidthScale=1.10)
        out = _pre_emphasis(out, coefficient=0.84)
        out = _add_breath_noise(out, level_db=-38.0)
        out = _normalize_peak(out, PEAK_DEFAULT)

    elif c == 'r':
        tap = synth_plosive('t', sampleRate=sampleRate, closureMilliseconds=12.0, burstMilliseconds=6.0,
                            aspirationMilliseconds=4.0, levelDb=-20.0)
        onsetF = LIQUID_ONSETS.get(v, LIQUID_ONSETS['a'])
        vow = synth_vowel_with_onset(v, f0, sampleRate, totalMilliseconds=vowelMilliseconds,
                                     onsetMilliseconds=36, onsetFormants=onsetF)
        out = _crossfade(tap, vow, sampleRate, overlap_ms=12)

    else:
        raise ValueError("synth_cv: unsupported consonant for synth_cv")

    combined = np.concatenate([head, out]).astype(DTYPE)
    return _normalize_peak(combined, PEAK_DEFAULT)

# =======================
# Higher-level helpers
# =======================

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
) -> str:
    """CV を合成して WAV 保存"""
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
    )
    return write_wav(outPath, waveform, sampleRate=sampleRate)


def synth_phrase_to_wav(
    vowels: Sequence[str],
    outPath: str,
    f0: float = 120.0,
    unitMilliseconds: int = 220,
    gapMilliseconds: int = 30,
    sampleRate: int = 22050,
) -> str:
    """MVP: 母音列 ['a','i',...] からフレーズ WAV を生成"""
    chunks: List[np.ndarray] = []
    for v in map(str.lower, vowels):
        if v in VOWEL_TABLE:
            seg = synth_vowel(v, f0=f0, durationSeconds=unitMilliseconds / 1000.0, sampleRate=sampleRate)
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
) -> np.ndarray:
    """ローマ字トークン列から波形を生成するヘルパー。"""
    segs: List[np.ndarray] = []
    gap = _ms_to_samples(max(0, int(gapMilliseconds)), sampleRate)
    vow_sec = max(0.12, float(vowelMilliseconds) / 1000.0)
    nasal_ms = max(80, int(vowelMilliseconds * 0.6))

    for tok in tokens:
        t = tok.strip().lower()
        if not t:
            continue
        if t == PAUSE_TOKEN:
            pause_ms = max(gapMilliseconds * 3, 120)
            segs.append(np.zeros(_ms_to_samples(pause_ms, sampleRate), dtype=DTYPE))
            continue
        if t in CV_TOKEN_MAP:
            ck, vk = CV_TOKEN_MAP[t]
            seg = synth_cv(ck, vk, f0=f0, sampleRate=sampleRate,
                           vowelMilliseconds=vowelMilliseconds,
                           overlapMilliseconds=overlapMilliseconds,
                           useOnsetTransition=useOnsetTransition)
        elif t in VOWEL_TABLE:
            seg = synth_vowel(t, f0=f0, durationSeconds=vow_sec, sampleRate=sampleRate)
        elif t in NASAL_TOKEN_MAP:
            seg = synth_nasal(NASAL_TOKEN_MAP[t], f0=f0, durationMilliseconds=nasal_ms, sampleRate=sampleRate)
        else:
            raise ValueError(f"Unsupported token '{tok}' for synthesis")

        if segs and gap > 0:
            segs.append(np.zeros(gap, dtype=DTYPE))
        segs.append(seg.astype(DTYPE, copy=False))

    if not segs:
        return np.zeros(_ms_to_samples(120, sampleRate), dtype=DTYPE)

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
) -> str:
    """トークン列合成 → WAV 保存。"""
    y = synth_token_sequence(
        tokens,
        f0=f0,
        sampleRate=sampleRate,
        vowelMilliseconds=vowelMilliseconds,
        overlapMilliseconds=overlapMilliseconds,
        gapMilliseconds=gapMilliseconds,
        useOnsetTransition=useOnsetTransition,
    )
    return write_wav(outPath, y, sampleRate=sampleRate)

# =======================
# I/O
# =======================

def write_wav(path: str, audio: np.ndarray, sampleRate: int = 22050) -> str:
    """16-bit PCM で WAV 保存"""
    audio = _ensure_array(audio)
    data16 = np.clip((audio * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampleRate)
        wf.writeframes(data16.tobytes())
    return os.path.abspath(path)