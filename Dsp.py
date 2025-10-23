# Date: 2024-06-09
# Author: ChatGPT
# Project: GIAN
# Description: DSP synthesis utilities for Japanese CV generation.
"""Signal processing helpers for the GIAN vocal synthesis project."""
from __future__ import annotations

import os
import wave
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from math import pi, sin, cos, exp
import unicodedata

# =======================
# Constants / Tables
# =======================

# ---- 母音プリセット（成人中性声の目安） ----
VOWEL_TABLE: Dict[str, Dict[str, Sequence[float]]] = {
    'a': {'F': [800, 1150, 2900], 'BW': [90, 110, 150]},
    'i': {'F': [350, 2000, 3000], 'BW': [60, 100, 150]},
    'u': {'F': [325, 700, 2530],  'BW': [60, 90, 140]},
    'e': {'F': [400, 1700, 2600], 'BW': [70, 100, 150]},
    'o': {'F': [450, 800, 2830],  'BW': [80, 110, 150]},
}

# ローマ字トークン → (子音, 母音) の変換テーブル
CV_TOKEN_MAP: Dict[str, Tuple[str, str]] = {
    'ka': ('k', 'a'),
    'ki': ('k', 'i'),
    'ku': ('k', 'u'),
    'ke': ('k', 'e'),
    'ko': ('k', 'o'),
    'sa': ('s', 'a'),
    'shi': ('sh', 'i'),
    'su': ('s', 'u'),
    'se': ('s', 'e'),
    'so': ('s', 'o'),
    'ta': ('t', 'a'),
    'chi': ('ch', 'i'),
    'tsu': ('ts', 'u'),
    'te': ('t', 'e'),
    'to': ('t', 'o'),
    'na': ('n', 'a'),
    'ni': ('n', 'i'),
    'nu': ('n', 'u'),
    'ne': ('n', 'e'),
    'no': ('n', 'o'),
    'ha': ('h', 'a'),
    'hi': ('h', 'i'),
    'fu': ('f', 'u'),
    'he': ('h', 'e'),
    'ho': ('h', 'o'),
    'ma': ('m', 'a'),
    'mi': ('m', 'i'),
    'mu': ('m', 'u'),
    'me': ('m', 'e'),
    'mo': ('m', 'o'),
    'ya': ('y', 'a'),
    'yu': ('y', 'u'),
    'yo': ('y', 'o'),
    'ra': ('r', 'a'),
    'ri': ('r', 'i'),
    'ru': ('r', 'u'),
    're': ('r', 'e'),
    'ro': ('r', 'o'),
    'wa': ('w', 'a'),
    'wo': ('w', 'o'),
}

NASAL_TOKEN_MAP: Dict[str, str] = {
    'n': 'n',
    'nn': 'n',
    'm': 'm',
}

PAUSE_TOKEN = 'pau'

DTYPE = np.float32
PEAK_DEFAULT = 0.9

# 乱数生成器を一元管理（毎回 new しない）
RNG = np.random.default_rng()


# =======================
# Text helpers
# =======================

_VALID_TOKENS = set(CV_TOKEN_MAP) | set(VOWEL_TABLE) | set(NASAL_TOKEN_MAP)
_MAX_ROMAJI_TOKEN_LENGTH = max(len(token) for token in _VALID_TOKENS)

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

_HAND_DAKUTEN_MAP: Dict[str, str] = {
    'ぱ': 'は', 'ぴ': 'ひ', 'ぷ': 'ふ', 'ぺ': 'へ', 'ぽ': 'ほ',
}

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


def _normalize_to_hiragana(text: str) -> str:
    """NFKC normalize and convert katakana to hiragana."""
    normalized = unicodedata.normalize('NFKC', text)
    buffer: List[str] = []
    for char in normalized:
        codePoint = ord(char)
        if 0x30A1 <= codePoint <= 0x30F3:
            buffer.append(chr(codePoint - 0x60))
        elif char == 'ヴ':
            buffer.append('ゔ')
        elif char == 'ヵ':
            buffer.append('ゕ')
        elif char == 'ヶ':
            buffer.append('ゖ')
        else:
            buffer.append(char)
    return ''.join(buffer)


def _append_pause(tokens: List[str]) -> None:
    if tokens and tokens[-1] == PAUSE_TOKEN:
        return
    tokens.append(PAUSE_TOKEN)


def _extend_last_vowel(tokens: List[str]) -> None:
    for token in reversed(tokens):
        if token == PAUSE_TOKEN:
            continue
        if token in VOWEL_TABLE:
            tokens.append(token)
            return
        if token in CV_TOKEN_MAP:
            tokens.append(CV_TOKEN_MAP[token][1])
            return
    # fallback: nothing to extend


def _parse_romaji_sequence(sequence: str) -> List[str]:
    tokens: List[str] = []
    index = 0
    length = len(sequence)
    while index < length:
        matched = False
        maxLength = min(_MAX_ROMAJI_TOKEN_LENGTH, length - index)
        for size in range(maxLength, 0, -1):
            candidate = sequence[index:index + size]
            if candidate in _VALID_TOKENS:
                tokens.append(candidate)
                index += size
                matched = True
                break
        if matched:
            continue
        char = sequence[index]
        if char in 'aiueo':
            tokens.append(char)
        elif char == 'n':
            tokens.append('n')
        index += 1
    return tokens


def text_to_tokens(text: str) -> List[str]:
    """Convert arbitrary text to the synthesizer token sequence."""
    if not text:
        return []

    tokens: List[str] = []
    romajiBuffer: List[str] = []

    normalized = _normalize_to_hiragana(text)
    length = len(normalized)
    position = 0

    def flush_buffer() -> None:
        nonlocal romajiBuffer
        if romajiBuffer:
            sequence = ''.join(romajiBuffer)
            tokens.extend(_parse_romaji_sequence(sequence))
            romajiBuffer = []

    while position < length:
        char = normalized[position]

        if char.isascii() and char.isalpha():
            romajiBuffer.append(char.lower())
            position += 1
            continue

        flush_buffer()

        if char.isspace() or char in _PUNCTUATION_CHARS:
            _append_pause(tokens)
            position += 1
            continue

        if char == 'ー':
            _extend_last_vowel(tokens)
            position += 1
            continue

        if char == 'っ':
            _append_pause(tokens)
            position += 1
            continue

        nextChar = normalized[position + 1] if position + 1 < length else ''
        pair = char + nextChar
        if pair in _KANA_DIGRAPH_MAP:
            tokens.extend(_KANA_DIGRAPH_MAP[pair])
            position += 2
            continue

        if char in _KANA_BASE_MAP:
            tokens.extend(_KANA_BASE_MAP[char])
            position += 1
            continue

        if char in _VOICED_KANA_MAP:
            base = _VOICED_KANA_MAP[char]
            tokens.extend(_KANA_BASE_MAP.get(base, []))
            position += 1
            continue

        if char in _HAND_DAKUTEN_MAP:
            base = _HAND_DAKUTEN_MAP[char]
            tokens.extend(_KANA_BASE_MAP.get(base, []))
            position += 1
            continue

        if char.isdigit():
            _append_pause(tokens)
            position += 1
            continue

        # unsupported symbol → skip
        position += 1

    flush_buffer()

    # 圧縮: 連続ポーズをまとめて削除し先頭末尾のポーズを除く
    compressed: List[str] = []
    for token in tokens:
        if token == PAUSE_TOKEN and (not compressed or compressed[-1] == PAUSE_TOKEN):
            continue
        compressed.append(token)

    if compressed and compressed[0] == PAUSE_TOKEN:
        compressed = compressed[1:]
    if compressed and compressed[-1] == PAUSE_TOKEN:
        compressed = compressed[:-1]

    return compressed


# =======================
# Utilities
# =======================

def _ms_to_samples(milliSeconds: float, sampleRate: int) -> int:
    """ミリ秒をサンプル数に変換（切り捨て）"""
    return int(sampleRate * (milliSeconds / 1000.0))


def _db_to_lin(decibelValue: float) -> float:
    """dB → 線形ゲイン"""
    return 10.0 ** (decibelValue / 20.0)


def _normalize_peak(signal: np.ndarray, targetPeak: float = PEAK_DEFAULT) -> np.ndarray:
    """ピーク正規化（無音ガード付き）"""
    currentPeak = float(np.max(np.abs(signal)) + 1e-12)
    if currentPeak == 0.0:
        return signal.astype(DTYPE, copy=False)
    return (targetPeak / currentPeak) * signal


def _apply_fade(signal: np.ndarray, sampleRate: int, attackMilliseconds: float = 5.0, releaseMilliseconds: float = 8.0) -> np.ndarray:
    """線形アタック／リリース"""
    length = len(signal)
    if length == 0:
        return signal
    fadedSignal = signal.astype(DTYPE).copy()
    attackSamples = _ms_to_samples(attackMilliseconds, sampleRate)
    releaseSamples = _ms_to_samples(releaseMilliseconds, sampleRate)

    if 0 < attackSamples < length:
        fadedSignal[:attackSamples] *= np.linspace(0.0, 1.0, attackSamples, dtype=DTYPE)
    if 0 < releaseSamples < length:
        fadedSignal[-releaseSamples:] *= np.linspace(1.0, 0.0, releaseSamples, dtype=DTYPE)
    return fadedSignal


def _apply_formant_filters(sourceSignal: np.ndarray, formants: Sequence[float], bandwidths: Sequence[float], sampleRate: int) -> np.ndarray:
    """フォルマント設定に基づいて帯域通過を連結"""
    filteredSignal = np.zeros_like(sourceSignal, dtype=DTYPE)
    for formantFrequency, bandwidth in zip(formants, bandwidths):
        resonanceQ = max(0.5, formantFrequency / float(bandwidth))
        coeffB0, coeffB1, coeffB2, coeffA1, coeffA2 = _bandpass_biquad_coeff(formantFrequency, resonanceQ, sampleRate)
        filteredSignal += _biquad_process(sourceSignal, coeffB0, coeffB1, coeffB2, coeffA1, coeffA2)
    return _lip_radiation(filteredSignal)


def _add_breath_noise(signal: np.ndarray, levelDb: float) -> np.ndarray:
    """息成分を加算（level_db<0で適用）"""
    signal = signal.astype(DTYPE, copy=False)
    if levelDb >= 0 or len(signal) == 0:
        return signal

    breathNoise = RNG.standard_normal(len(signal)).astype(DTYPE)
    rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
    targetLevel = rms * _db_to_lin(levelDb)
    currentLevel = float(np.sqrt(np.mean(breathNoise * breathNoise) + 1e-12))
    if currentLevel > 0:
        signal = signal + breathNoise * (targetLevel / currentLevel)
    return signal.astype(DTYPE, copy=False)


# =======================
# Filters
# =======================

def _bandpass_biquad_coeff(centerFrequency: float, resonanceQ: float, sampleRate: int) -> Tuple[float, float, float, float, float]:
    """RBJ系の簡易BPF係数（ピークゲイン一定タイプ）"""
    angularFrequency = 2.0 * pi * centerFrequency / sampleRate
    alpha = sin(angularFrequency) / (2.0 * resonanceQ)
    numeratorB0 = alpha
    numeratorB1 = 0.0
    numeratorB2 = -alpha
    denominatorA0 = 1.0 + alpha
    denominatorA1 = -2.0 * cos(angularFrequency)
    denominatorA2 = 1.0 - alpha
    return (
        numeratorB0 / denominatorA0,
        numeratorB1 / denominatorA0,
        numeratorB2 / denominatorA0,
        denominatorA1 / denominatorA0,
        denominatorA2 / denominatorA0,
    )


def _biquad_process(signal: np.ndarray, coeffB0: float, coeffB1: float, coeffB2: float, coeffA1: float, coeffA2: float) -> np.ndarray:
    """単一Biquadフィルタ処理（Direct Form I 相当）"""
    output = np.empty_like(signal, dtype=DTYPE)
    prevX1 = prevX2 = prevY1 = prevY2 = 0.0
    for index, sample in enumerate(signal):
        current = coeffB0 * sample + coeffB1 * prevX1 + coeffB2 * prevX2 - coeffA1 * prevY1 - coeffA2 * prevY2
        output[index] = current
        prevX2, prevX1 = prevX1, sample
        prevY2, prevY1 = prevY1, current
    return output


def _one_pole_lp(signal: np.ndarray, cutoffFrequency: float, sampleRate: int) -> np.ndarray:
    """一次ローパス"""
    decay = exp(-2.0 * pi * cutoffFrequency / sampleRate)
    output = np.empty_like(signal, dtype=DTYPE)
    state = 0.0
    for index, sample in enumerate(signal):
        state = (1 - decay) * sample + decay * state
        output[index] = state
    return output


def _lip_radiation(signal: np.ndarray) -> np.ndarray:
    """唇放射の近似：微分"""
    if len(signal) == 0:
        return signal
    output = np.empty_like(signal, dtype=DTYPE)
    output[0] = signal[0]
    output[1:] = signal[1:] - signal[:-1]
    return output


# =======================
# Sources / Noise
# =======================

def _glottal_source(
    fundamentalHz: float,
    durationSeconds: float,
    sampleRate: int,
    jitterCents: float = 10,
    shimmerDb: float = 0.8,
) -> np.ndarray:
    """鋸波＋jitter/shimmer。位相加算で生成（AI不使用）"""
    sampleCount = int(durationSeconds * sampleRate)
    jitterNoise = RNG.standard_normal(sampleCount).astype(DTYPE)
    pole = 0.999
    jitterSignal = np.empty_like(jitterNoise)
    accumulator = 0.0
    for index in range(sampleCount):
        accumulator = pole * accumulator + (1.0 - pole) * jitterNoise[index]
        jitterSignal[index] = accumulator
    cents = (jitterCents / 100.0) * jitterSignal
    instantFrequency = fundamentalHz * (2.0 ** (cents / 12.0))

    phase = np.cumsum(2.0 * pi * instantFrequency / sampleRate, dtype=np.float64)
    sawWave = 2.0 * ((phase / (2.0 * pi)) % 1.0) - 1.0

    shimmerNoise = RNG.standard_normal(sampleCount).astype(DTYPE)
    shimmerAccumulator = 0.0
    shimmerSignal = np.empty_like(shimmerNoise)
    for index in range(sampleCount):
        shimmerAccumulator = pole * shimmerAccumulator + (1.0 - pole) * shimmerNoise[index]
        shimmerSignal[index] = shimmerAccumulator
    amplitude = 10.0 ** ((shimmerDb / 20.0) * shimmerSignal)

    rawSource = sawWave.astype(DTYPE) * amplitude.astype(DTYPE)

    decay = exp(-2.0 * pi * 800.0 / sampleRate)
    filteredSource = np.empty_like(rawSource)
    state = 0.0
    for index in range(sampleCount):
        state = (1 - decay) * rawSource[index] + decay * state
        filteredSource[index] = state
    return filteredSource


def _gen_band_noise(
    durationSeconds: float,
    sampleRate: int,
    centerFrequency: float,
    resonanceQ: float,
    useLipRadiation: bool = True,
) -> np.ndarray:
    """帯域ノイズ生成（任意で唇放射を適用）"""
    sampleCount = int(durationSeconds * sampleRate)
    noise = RNG.standard_normal(sampleCount).astype(DTYPE)
    coeffB0, coeffB1, coeffB2, coeffA1, coeffA2 = _bandpass_biquad_coeff(centerFrequency, resonanceQ, sampleRate)
    filteredNoise = _biquad_process(noise, coeffB0, coeffB1, coeffB2, coeffA1, coeffA2)
    return _lip_radiation(filteredNoise) if useLipRadiation else filteredNoise


# =======================
# Core Synthesis
# =======================

def synth_vowel(
    vowel: str = 'a',
    f0: float = 120.0,
    durationSeconds: float = 1.0,
    sampleRate: int = 22050,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
) -> np.ndarray:
    """母音合成（3フォルマントの簡易ボコーダ）"""
    assert vowel in VOWEL_TABLE, f"unsupported vowel: {vowel}"
    glottalSource = _glottal_source(f0, durationSeconds, sampleRate, jitterCents, shimmerDb)

    spec = VOWEL_TABLE[vowel]
    formants = spec['F']
    bandwidths = spec['BW']

    filtered = _apply_formant_filters(glottalSource, formants, bandwidths, sampleRate)
    filtered = _add_breath_noise(filtered, breathLevelDb)

    return _normalize_peak(filtered, PEAK_DEFAULT).astype(DTYPE)


def synth_fricative(
    consonant: str = 's',
    durationSeconds: float = 0.16,
    sampleRate: int = 22050,
    levelDb: float = -12.0,
) -> np.ndarray:
    """無声摩擦音の簡易実装。"""
    consonantKey = consonant.lower()
    if consonantKey == 's':
        centerFrequency, resonanceQ, useLipRadiation, postLowPass = 6500.0, 3.0, True, None
    elif consonantKey == 'sh':
        centerFrequency, resonanceQ, useLipRadiation, postLowPass = 3800.0, 2.4, True, 4200.0
    elif consonantKey == 'h':
        centerFrequency, resonanceQ, useLipRadiation, postLowPass = 1800.0, 1.4, False, 2300.0
    elif consonantKey == 'f':
        centerFrequency, resonanceQ, useLipRadiation, postLowPass = 950.0, 1.3, False, 2100.0
    else:
        raise ValueError("synth_fricative: supported consonants are 's','sh','h','f'.")

    noise = _gen_band_noise(durationSeconds, sampleRate, centerFrequency, resonanceQ, useLipRadiation=useLipRadiation)
    if postLowPass is not None:
        noise = _one_pole_lp(noise, cutoffFrequency=postLowPass, sampleRate=sampleRate)
    if consonantKey == 'h':
        noise = _one_pole_lp(noise, cutoffFrequency=1700.0, sampleRate=sampleRate)
    attackMilliseconds = 6 if consonantKey in ('h', 'f') else 4
    releaseMilliseconds = 16 if consonantKey in ('h', 'f') else 12
    noise = _apply_fade(noise, sampleRate, attackMilliseconds=attackMilliseconds, releaseMilliseconds=releaseMilliseconds)

    peakTarget = 0.6 if consonantKey not in ('h', 'f') else (0.5 if consonantKey == 'h' else 0.55)
    noise = _normalize_peak(noise, peakTarget).astype(DTYPE, copy=False)
    gain = _db_to_lin(levelDb)
    if gain != 1.0:
        noise = noise * gain
    return noise.astype(DTYPE, copy=False)


def synth_plosive(
    consonant: str = 't',
    sampleRate: int = 22050,
    closureMilliseconds: Optional[float] = None,
    burstMilliseconds: Optional[float] = None,
    aspirationMilliseconds: Optional[float] = None,
    levelDb: float = -10.0,
) -> np.ndarray:
    """無声破裂音（t / k）"""
    consonantKey = consonant.lower()
    if consonantKey not in ('t', 'k'):
        raise ValueError("synth_plosive: supported only 't' or 'k'.")

    if consonantKey == 't':
        closureMilliseconds = 24 if closureMilliseconds is None else closureMilliseconds
        burstMilliseconds = 12 if burstMilliseconds is None else burstMilliseconds
        aspirationMilliseconds = 18 if aspirationMilliseconds is None else aspirationMilliseconds
        burstCenter, burstQ = 4500.0, 1.2
        aspirationCenter, aspirationQ = 6000.0, 0.9
    else:
        closureMilliseconds = 32 if closureMilliseconds is None else closureMilliseconds
        burstMilliseconds = 14 if burstMilliseconds is None else burstMilliseconds
        aspirationMilliseconds = 24 if aspirationMilliseconds is None else aspirationMilliseconds
        burstCenter, burstQ = 1700.0, 1.2
        aspirationCenter, aspirationQ = 2500.0, 0.9

    closureSegment = np.zeros(_ms_to_samples(closureMilliseconds, sampleRate), dtype=DTYPE)

    burstLength = _ms_to_samples(burstMilliseconds, sampleRate)
    burstNoise = _gen_band_noise(burstLength / sampleRate, sampleRate, burstCenter, burstQ, useLipRadiation=False)
    tau = max(1.0, burstLength / 4.0)
    envelope = np.exp(-np.arange(burstLength, dtype=DTYPE) / tau)
    burstNoise = burstNoise[:burstLength] * envelope

    aspirationLength = _ms_to_samples(aspirationMilliseconds, sampleRate)
    aspirationNoise = _gen_band_noise(aspirationLength / sampleRate, sampleRate, aspirationCenter, aspirationQ, useLipRadiation=False)
    aspirationNoise = _apply_fade(aspirationNoise, sampleRate, attackMilliseconds=3, releaseMilliseconds=18)
    aspirationNoise = _one_pole_lp(aspirationNoise, cutoffFrequency=4500.0, sampleRate=sampleRate)

    consonantSignal = np.concatenate([closureSegment, burstNoise, aspirationNoise]).astype(DTYPE)
    consonantSignal *= _db_to_lin(levelDb)
    return _normalize_peak(consonantSignal, 0.6)


def synth_affricate(
    consonant: str = 'ch',
    sampleRate: int = 22050,
    closureMilliseconds: Optional[float] = None,
    fricativeMilliseconds: Optional[float] = None,
    levelDb: float = -11.0,
) -> np.ndarray:
    """破擦音 /ch/, /ts/ の簡易版"""
    consonantKey = consonant.lower()
    if consonantKey not in ('ch', 'ts'):
        raise ValueError("synth_affricate: supported only 'ch' or 'ts'.")

    if consonantKey == 'ch':
        closureMilliseconds = 26 if closureMilliseconds is None else closureMilliseconds
        burstMilliseconds = 10
        fricativeMilliseconds = 95.0 if fricativeMilliseconds is None else fricativeMilliseconds
        fricativeNoise = synth_fricative('sh', durationSeconds=fricativeMilliseconds / 1000.0, sampleRate=sampleRate, levelDb=-16.0)
    else:
        closureMilliseconds = 24 if closureMilliseconds is None else closureMilliseconds
        burstMilliseconds = 9
        fricativeMilliseconds = 90.0 if fricativeMilliseconds is None else fricativeMilliseconds
        fricativeNoise = synth_fricative('s', durationSeconds=fricativeMilliseconds / 1000.0, sampleRate=sampleRate, levelDb=-16.0)

    plosiveBase = synth_plosive(
        't',
        sampleRate=sampleRate,
        closureMilliseconds=closureMilliseconds,
        burstMilliseconds=burstMilliseconds,
        aspirationMilliseconds=0.0,
        levelDb=levelDb,
    )
    affricate = _crossfade(plosiveBase, fricativeNoise, sampleRate, overlapMilliseconds=14)
    return _normalize_peak(affricate, 0.6)


NASAL_PRESETS: Dict[str, Dict[str, Sequence[float]]] = {
    'n': {'F': [250.0, 1900.0, 2800.0], 'BW': [80.0, 160.0, 220.0]},
    'm': {'F': [220.0, 1600.0, 2500.0], 'BW': [70.0, 150.0, 210.0]},
}


def synth_nasal(consonant: str = 'n', f0: float = 120.0, durationMilliseconds: float = 90.0, sampleRate: int = 22050) -> np.ndarray:
    """簡易的な鼻音 /n/, /m/"""
    consonantKey = consonant.lower()
    if consonantKey not in NASAL_PRESETS:
        raise ValueError("synth_nasal: supported nasals are 'n' or 'm'.")

    durationSeconds = max(20.0, float(durationMilliseconds)) / 1000.0
    spec = NASAL_PRESETS[consonantKey]
    formants = spec['F']
    bandwidths = spec['BW']
    nasal = _synth_vowel_fixed(
        formants,
        bandwidths,
        f0,
        durationSeconds,
        sampleRate,
        jitterCents=4.0,
        shimmerDb=0.4,
        breathLevelDb=-38.0,
    )
    nasal = _apply_fade(
        nasal,
        sampleRate,
        attackMilliseconds=8.0 if consonantKey == 'n' else 10.0,
        releaseMilliseconds=22.0 if consonantKey == 'n' else 28.0,
    )
    gain = 0.6 if consonantKey == 'n' else 0.7
    return _normalize_peak(nasal * gain, 0.5)


# =======================
# Building Blocks
# =======================

def _crossfade(first: np.ndarray, second: np.ndarray, sampleRate: int, overlapMilliseconds: float = 30.0) -> np.ndarray:
    """a → b をオーバーラップ結合（安全版）"""
    overlapSamples = _ms_to_samples(overlapMilliseconds, sampleRate)
    overlapSamples = min(overlapSamples, len(first), len(second))
    if overlapSamples <= 0:
        return np.concatenate([first, second]).astype(DTYPE)

    fadeOut = np.linspace(1.0, 0.0, overlapSamples, dtype=DTYPE)
    fadeIn = 1.0 - fadeOut
    head = first[:-overlapSamples]
    tail = first[-overlapSamples:] * fadeOut + second[:overlapSamples] * fadeIn
    rest = second[overlapSamples:]
    return np.concatenate([head, tail, rest]).astype(DTYPE)


# ===== onset helper =====

def _synth_vowel_fixed(
    formants: Sequence[float],
    bandwidths: Sequence[float],
    f0: float,
    durationSeconds: float,
    sampleRate: int,
    jitterCents: float = 6.0,
    shimmerDb: float = 0.6,
    breathLevelDb: float = -40.0,
) -> np.ndarray:
    """与えたフォルマントで固定合成（短区間）"""
    glottalSource = _glottal_source(f0, durationSeconds, sampleRate, jitterCents, shimmerDb)
    filtered = _apply_formant_filters(glottalSource, formants, bandwidths, sampleRate)
    filtered = _add_breath_noise(filtered, breathLevelDb)

    return _normalize_peak(filtered, PEAK_DEFAULT).astype(DTYPE)


def synth_vowel_with_onset(
    vowel: str,
    f0: float,
    sampleRate: int,
    totalMilliseconds: int = 240,
    onsetMilliseconds: int = 45,
    onsetFormants: Optional[Sequence[float]] = None,
    onsetBandwidthScale: float = 0.85,
) -> np.ndarray:
    """母音先頭だけフォルマント遷移を与える簡易版"""
    spec = VOWEL_TABLE[vowel]
    targetFormants, targetBandwidths = spec['F'], spec['BW']
    if not onsetFormants:
        return synth_vowel(vowel=vowel, f0=f0, durationSeconds=totalMilliseconds / 1000.0, sampleRate=sampleRate)

    totalMilliseconds = int(max(10, totalMilliseconds))
    onsetMilliseconds = int(max(1, min(onsetMilliseconds, totalMilliseconds - 1)))
    sustainMilliseconds = max(0, totalMilliseconds - onsetMilliseconds)

    onsetBandwidths = [bandwidth * float(onsetBandwidthScale) for bandwidth in targetBandwidths]
    onsetSegment = _synth_vowel_fixed(onsetFormants, onsetBandwidths, f0, onsetMilliseconds / 1000.0, sampleRate)
    onsetSegment = _apply_fade(onsetSegment, sampleRate, attackMilliseconds=6.0, releaseMilliseconds=min(12.0, onsetMilliseconds * 0.5))

    if sustainMilliseconds <= 0:
        return onsetSegment

    overlapMilliseconds = min(max(6.0, onsetMilliseconds * 0.45), 14.0, float(sustainMilliseconds))
    restLengthMilliseconds = sustainMilliseconds + max(0.0, overlapMilliseconds)

    sustainSegment = _synth_vowel_fixed(targetFormants, targetBandwidths, f0, restLengthMilliseconds / 1000.0, sampleRate)
    sustainSegment = _apply_fade(sustainSegment, sampleRate, attackMilliseconds=4.0, releaseMilliseconds=12.0)

    return _crossfade(onsetSegment, sustainSegment, sampleRate, overlapMilliseconds=overlapMilliseconds)


# =======================
# Composition (CV/フレーズ)
# =======================

# ===== 置き換え: GLIDE_ONSETS（/w/ のF1/F2を少し上げる、/y/はF1を少し上げる）=====
GLIDE_ONSETS: Dict[str, Dict[str, Sequence[float]]] = {
    'w': {
        # 旧: F1≈300台/F2≈600-700台 → 鼻音域と近くなるので少し離す
        'a': [420.0, 1000.0, 2300.0],
        'i': [400.0, 1200.0, 2500.0],
        'u': [380.0,  900.0, 2200.0],
        'e': [410.0, 1100.0, 2350.0],
        'o': [420.0,  950.0, 2300.0],
    },
    'y': {
        # /y/ は F2 は既に高いので F1 をやや上げて nasal 感を避ける
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
    consonantKey = cons.lower()
    vowelKey = vowel.lower()
    if vowelKey not in VOWEL_TABLE:
        raise ValueError("Unknown vowel for synth_cv")

    head = np.zeros(_ms_to_samples(preMilliseconds, sampleRate), dtype=DTYPE)

    if consonantKey in ('s', 'sh'):
        levelLookup = {'s': -14.0, 'sh': -15.0}
        defaultMilliseconds = {'s': 160.0, 'sh': 150.0}
        fricativeMilliseconds = float(consonantMilliseconds) if consonantMilliseconds is not None else defaultMilliseconds[consonantKey]
        fricativeNoise = synth_fricative(consonantKey, durationSeconds=fricativeMilliseconds / 1000.0, sampleRate=sampleRate, levelDb=levelLookup[consonantKey])
        vowelSegment = synth_vowel(vowel=vowelKey, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        output = _crossfade(fricativeNoise, vowelSegment, sampleRate, overlapMilliseconds=max(24, overlapMilliseconds))

    elif consonantKey in ('h', 'f'):
        defaultMilliseconds = {'h': 190.0, 'f': 170.0}
        consonantLength = float(consonantMilliseconds) if consonantMilliseconds is not None else defaultMilliseconds[consonantKey]
        consonantLength = max(0.0, consonantLength)
        silence = np.zeros(_ms_to_samples(consonantLength, sampleRate), dtype=DTYPE)

        vowelSegment = synth_vowel(vowel=vowelKey, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)

        if len(silence) > 0:
            try:
                requestedOverlap = float(overlapMilliseconds)
            except Exception:
                requestedOverlap = 12.0
            effectiveOverlap = min(12.0, consonantLength, max(0.0, requestedOverlap))
            if effectiveOverlap > 0.0:
                output = _crossfade(silence, vowelSegment, sampleRate, overlapMilliseconds=effectiveOverlap)
            else:
                output = np.concatenate([silence, vowelSegment]).astype(DTYPE)
        else:
            output = vowelSegment

        output = _apply_fade(output, sampleRate, attackMilliseconds=8, releaseMilliseconds=12)

    elif consonantKey in ('t', 'k'):
        plosive = synth_plosive(consonantKey, sampleRate=sampleRate)

        if useOnsetTransition and vowelKey == 'a':
            if consonantKey == 't':
                onsetFormants = [800, 1800, 3000]
            else:
                onsetFormants = [800, 2200, 2400]
            vowelSegment = synth_vowel_with_onset('a', f0, sampleRate, totalMilliseconds=vowelMilliseconds, onsetMilliseconds=45, onsetFormants=onsetFormants)
        else:
            vowelSegment = synth_vowel(vowel=vowelKey, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)

        overlapClamp = 8 if consonantKey == 't' else 10
        try:
            requestedOverlap = int(overlapMilliseconds)
            overlapClamp = max(6, min(requestedOverlap, 12))
        except Exception:
            pass
        output = _crossfade(plosive, vowelSegment, sampleRate, overlapMilliseconds=overlapClamp)

    elif consonantKey in ('ch', 'ts'):
        affricate = synth_affricate(consonantKey, sampleRate=sampleRate)
        vowelSegment = synth_vowel(vowel=vowelKey, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        output = _crossfade(affricate, vowelSegment, sampleRate, overlapMilliseconds=max(12, min(overlapMilliseconds, 18)))

    elif consonantKey == 'n':
        nasalMilliseconds = float(consonantMilliseconds) if consonantMilliseconds is not None else 90.0
        nasal = synth_nasal('n', f0=f0, durationMilliseconds=nasalMilliseconds, sampleRate=sampleRate)
        vowelSegment = synth_vowel(vowel=vowelKey, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        output = _crossfade(nasal, vowelSegment, sampleRate, overlapMilliseconds=max(25, overlapMilliseconds))

    elif consonantKey == 'm':
        nasalMilliseconds = float(consonantMilliseconds) if consonantMilliseconds is not None else 110.0
        nasal = synth_nasal('m', f0=f0, durationMilliseconds=nasalMilliseconds, sampleRate=sampleRate)
        vowelSegment = synth_vowel(vowel=vowelKey, f0=f0, durationSeconds=vowelMilliseconds / 1000.0, sampleRate=sampleRate)
        output = _crossfade(nasal, vowelSegment, sampleRate, overlapMilliseconds=max(28, overlapMilliseconds))

    elif consonantKey == 'w':
        onsetMilliseconds = float(consonantMilliseconds) if consonantMilliseconds is not None else 44.0
        onsetMilliseconds = max(24.0, min(onsetMilliseconds, float(vowelMilliseconds) - 8.0))
        onsetFormants = GLIDE_ONSETS['w'].get(vowelKey, GLIDE_ONSETS['w']['a'])
        output = synth_vowel_with_onset(
            vowelKey,
            f0,
            sampleRate,
            totalMilliseconds=vowelMilliseconds,
            onsetMilliseconds=int(onsetMilliseconds),
            onsetFormants=onsetFormants,
            onsetBandwidthScale=1.18,
        )
        output = _pre_emphasis(output, coefficient=0.86)
        output = _add_breath_noise(output, levelDb=-36.0)
        output = _normalize_peak(output, PEAK_DEFAULT)

    elif consonantKey == 'y':
        onsetMilliseconds = float(consonantMilliseconds) if consonantMilliseconds is not None else 40.0
        onsetMilliseconds = max(24.0, min(onsetMilliseconds, float(vowelMilliseconds) - 10.0))
        onsetFormants = GLIDE_ONSETS['y'].get(vowelKey, GLIDE_ONSETS['y']['a'])
        output = synth_vowel_with_onset(
            vowelKey,
            f0,
            sampleRate,
            totalMilliseconds=vowelMilliseconds,
            onsetMilliseconds=int(onsetMilliseconds),
            onsetFormants=onsetFormants,
            onsetBandwidthScale=1.10,
        )
        output = _pre_emphasis(output, coefficient=0.84)
        output = _add_breath_noise(output, levelDb=-38.0)
        output = _normalize_peak(output, PEAK_DEFAULT)

    elif consonantKey == 'r':
        tap = synth_plosive('t', sampleRate=sampleRate, closureMilliseconds=12.0, burstMilliseconds=6.0, aspirationMilliseconds=4.0, levelDb=-20.0)
        onsetFormants = LIQUID_ONSETS.get(vowelKey, LIQUID_ONSETS['a'])
        vowelSegment = synth_vowel_with_onset(vowelKey, f0, sampleRate, totalMilliseconds=vowelMilliseconds, onsetMilliseconds=36, onsetFormants=onsetFormants)
        output = _crossfade(tap, vowelSegment, sampleRate, overlapMilliseconds=12)

    else:
        raise ValueError("synth_cv: unsupported consonant for synth_cv")

    combined = np.concatenate([head, output]).astype(DTYPE)
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
    buffer = []
    for vowel in map(str.lower, vowels):
        if vowel in VOWEL_TABLE:
            segment = synth_vowel(vowel, f0=f0, durationSeconds=unitMilliseconds / 1000.0, sampleRate=sampleRate)
            buffer.append(segment)
            buffer.append(np.zeros(_ms_to_samples(gapMilliseconds, sampleRate), dtype=DTYPE))
    if not buffer:
        buffer = [np.zeros(int(sampleRate * 0.3), dtype=DTYPE)]
    concatenated = np.concatenate(buffer)
    return write_wav(outPath, concatenated, sampleRate=sampleRate)


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

    segments: list[np.ndarray] = []
    gapSamples = _ms_to_samples(max(0, int(gapMilliseconds)), sampleRate)
    vowelSeconds = max(0.12, float(vowelMilliseconds) / 1000.0)
    nasalDuration = max(80, int(vowelMilliseconds * 0.6))

    for token in tokens:
        normalized = token.strip().lower()
        if not normalized:
            continue

        if normalized == PAUSE_TOKEN:
            pauseMilliseconds = max(gapMilliseconds * 3, 120)
            pauseSamples = _ms_to_samples(pauseMilliseconds, sampleRate)
            segments.append(np.zeros(pauseSamples, dtype=DTYPE))
            continue

        if normalized in CV_TOKEN_MAP:
            consonantKey, vowelKey = CV_TOKEN_MAP[normalized]
            segment = synth_cv(
                consonantKey,
                vowelKey,
                f0=f0,
                sampleRate=sampleRate,
                vowelMilliseconds=vowelMilliseconds,
                overlapMilliseconds=overlapMilliseconds,
                useOnsetTransition=useOnsetTransition,
            )
        elif normalized in VOWEL_TABLE:
            segment = synth_vowel(
                normalized,
                f0=f0,
                durationSeconds=vowelSeconds,
                sampleRate=sampleRate,
            )
        elif normalized in NASAL_TOKEN_MAP:
            segment = synth_nasal(
                NASAL_TOKEN_MAP[normalized],
                f0=f0,
                durationMilliseconds=nasalDuration,
                sampleRate=sampleRate,
            )
        else:
            raise ValueError(f"Unsupported token '{token}' for synthesis")

        if segments and gapSamples > 0:
            segments.append(np.zeros(gapSamples, dtype=DTYPE))
        segments.append(segment.astype(DTYPE, copy=False))

    if not segments:
        return np.zeros(_ms_to_samples(120, sampleRate), dtype=DTYPE)

    combined = np.concatenate(segments).astype(DTYPE, copy=False)
    return _normalize_peak(combined, PEAK_DEFAULT)


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

    waveform = synth_token_sequence(
        tokens,
        f0=f0,
        sampleRate=sampleRate,
        vowelMilliseconds=vowelMilliseconds,
        overlapMilliseconds=overlapMilliseconds,
        gapMilliseconds=gapMilliseconds,
        useOnsetTransition=useOnsetTransition,
    )
    return write_wav(outPath, waveform, sampleRate=sampleRate)


# =======================
# I/O
# =======================

def write_wav(filePath: str, audioData: np.ndarray, sampleRate: int = 22050) -> str:
    """16-bit PCM で WAV 保存"""
    data16 = np.clip((audioData * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(filePath, 'wb') as waveFile:
        waveFile.setnchannels(1)
        waveFile.setsampwidth(2)
        waveFile.setframerate(sampleRate)
        waveFile.writeframes(data16.tobytes())
    return os.path.abspath(filePath)

# ===== 追加: 高域を少し持ち上げるプレエンファシス =====
def _pre_emphasis(signal: np.ndarray, coefficient: float = 0.85) -> np.ndarray:
    """y[n] = x[n] - a*x[n-1]（高域をわずかに強調）"""
    if len(signal) == 0:
        return signal
    emphasized = np.empty_like(signal, dtype=DTYPE)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coefficient * signal[:-1]
    return emphasized.astype(DTYPE, copy=False)
