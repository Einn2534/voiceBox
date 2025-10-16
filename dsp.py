# dsp.py (refactored)
from __future__ import annotations

import os
import wave
from typing import Sequence, Optional, Tuple, Dict

import numpy as np
from math import pi, sin, cos, exp

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

DTYPE = np.float32
PEAK_DEFAULT = 0.9

# 乱数生成器を一元管理（毎回 new しない）
RNG = np.random.default_rng()


# =======================
# Utilities
# =======================

def _ms_to_samples(ms: float, fs: int) -> int:
    """ミリ秒をサンプル数に変換（切り捨て）"""
    return int(fs * (ms / 1000.0))


def _db_to_lin(db: float) -> float:
    """dB → 線形ゲイン"""
    return 10.0 ** (db / 20.0)


def _normalize_peak(x: np.ndarray, peak: float = PEAK_DEFAULT) -> np.ndarray:
    """ピーク正規化（無音ガード付き）"""
    p = float(np.max(np.abs(x)) + 1e-12)
    if p == 0.0:
        return x.astype(DTYPE, copy=False)
    return (peak / p) * x


def _apply_fade(x: np.ndarray, fs: int, attack_ms: float = 5.0, release_ms: float = 8.0) -> np.ndarray:
    """線形アタック／リリース"""
    n = len(x)
    if n == 0:
        return x
    y = x.astype(DTYPE).copy()
    a = _ms_to_samples(attack_ms, fs)
    r = _ms_to_samples(release_ms, fs)

    if 0 < a < n:
        y[:a] *= np.linspace(0.0, 1.0, a, dtype=DTYPE)
    if 0 < r < n:
        y[-r:] *= np.linspace(1.0, 0.0, r, dtype=DTYPE)
    return y


def _apply_formant_filters(src: np.ndarray, formants: Sequence[float], bandwidths: Sequence[float], fs: int) -> np.ndarray:
    """フォルマント設定に基づいて帯域通過を連結"""
    y = np.zeros_like(src, dtype=DTYPE)
    for fi, bwi in zip(formants, bandwidths):
        Q = max(0.5, fi / float(bwi))
        b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(fi, Q, fs)
        y += _biquad_process(src, b0, b1, b2, a1, a2)
    return _lip_radiation(y)


def _add_breath_noise(signal: np.ndarray, level_db: float) -> np.ndarray:
    """息成分を加算（level_db<0で適用）"""
    signal = signal.astype(DTYPE, copy=False)
    if level_db >= 0 or len(signal) == 0:
        return signal

    noise = RNG.standard_normal(len(signal)).astype(DTYPE)
    rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
    target = rms * _db_to_lin(level_db)
    cur = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    if cur > 0:
        signal = signal + noise * (target / cur)
    return signal.astype(DTYPE, copy=False)


# =======================
# Filters
# =======================

def _bandpass_biquad_coeff(f0: float, Q: float, fs: int) -> Tuple[float, float, float, float, float]:
    """RBJ系の簡易BPF係数（ピークゲイン一定タイプ）"""
    w0 = 2.0 * pi * f0 / fs
    alpha = sin(w0) / (2.0 * Q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos(w0)
    a2 = 1.0 - alpha
    return (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)


def _biquad_process(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    """単一Biquadフィルタ処理（Direct Form I 相当）"""
    y = np.empty_like(x, dtype=DTYPE)
    x1 = x2 = y1 = y2 = 0.0
    for n, xn in enumerate(x):
        yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


def _one_pole_lp(x: np.ndarray, fc: float, fs: int) -> np.ndarray:
    """一次ローパス"""
    k = exp(-2.0*pi*fc/fs)
    y = np.empty_like(x, dtype=DTYPE)
    s = 0.0
    for i, xi in enumerate(x):
        s = (1 - k) * xi + k * s
        y[i] = s
    return y


def _lip_radiation(x: np.ndarray) -> np.ndarray:
    """唇放射の近似：微分"""
    if len(x) == 0:
        return x
    y = np.empty_like(x, dtype=DTYPE)
    y[0] = x[0]
    y[1:] = x[1:] - x[:-1]
    return y


# =======================
# Sources / Noise
# =======================

def _glottal_source(f0: float, dur_s: float, fs: int,
                    jitter_cents: float = 10, shimmer_db: float = 0.8) -> np.ndarray:
    """
    鋸波＋jitter/shimmer。位相加算で生成（AI不使用）
    """
    n = int(dur_s * fs)
    t = np.arange(n, dtype=DTYPE) / fs  # 使っていないが検証には便利

    # jitter（低周波揺らぎを一次LPで付与）
    noise = RNG.standard_normal(n).astype(DTYPE)
    alpha = 0.999
    jit = np.empty_like(noise)
    acc = 0.0
    for i in range(n):
        acc = alpha*acc + (1.0-alpha)*noise[i]
        jit[i] = acc
    cents = (jitter_cents/100.0) * jit
    f_inst = f0 * (2.0 ** (cents/12.0))

    # 位相加算 → 鋸波
    phase = np.cumsum(2.0*pi*f_inst/fs, dtype=np.float64)
    saw = 2.0 * ((phase/(2.0*pi)) % 1.0) - 1.0

    # shimmer（振幅揺らぎ）
    noise2 = RNG.standard_normal(n).astype(DTYPE)
    acc2 = 0.0
    shim = np.empty_like(noise2)
    for i in range(n):
        acc2 = alpha*acc2 + (1.0-alpha)*noise2[i]
        shim[i] = acc2
    amp = 10.0 ** ((shimmer_db/20.0) * shim)

    src = (saw.astype(DTYPE) * amp.astype(DTYPE))

    # 緩やかにローパス（傾き付与）
    k = exp(-2.0*pi*800.0/fs)
    y = np.empty_like(src)
    s = 0.0
    for i in range(n):
        s = (1-k)*src[i] + k*s
        y[i] = s
    return y


def _gen_band_noise(dur_s: float, fs: int, fc: float, Q: float, liprad: bool = True) -> np.ndarray:
    """帯域ノイズ生成（任意で唇放射を適用）"""
    n = int(dur_s * fs)
    noise = RNG.standard_normal(n).astype(DTYPE)
    b0, b1, b2, a1, a2 = _bandpass_biquad_coeff(fc, Q, fs)
    y = _biquad_process(noise, b0, b1, b2, a1, a2)
    return _lip_radiation(y) if liprad else y


# =======================
# Core Synthesis
# =======================

def synth_vowel(vowel: str = 'a', f0: float = 120.0, dur_s: float = 1.0, fs: int = 22050,
                jitter_cents: float = 6.0, shimmer_db: float = 0.6, breath_level_db: float = -40.0) -> np.ndarray:
    """母音合成（3フォルマントの簡易ボコーダ）"""
    assert vowel in VOWEL_TABLE, f"unsupported vowel: {vowel}"
    src = _glottal_source(f0, dur_s, fs, jitter_cents, shimmer_db)

    spec = VOWEL_TABLE[vowel]
    F = spec['F']
    BW = spec['BW']

    y = _apply_formant_filters(src, F, BW, fs)
    y = _add_breath_noise(y, breath_level_db)

    return _normalize_peak(y, PEAK_DEFAULT).astype(DTYPE)


def synth_fricative(consonant: str = 's', dur_s: float = 0.16, fs: int = 22050,
                    level_db: float = -12.0) -> np.ndarray:
    """
    無声摩擦音の簡易実装。

    Parameters
    ----------
    consonant:
        's', 'sh', 'h', 'f' に対応。
    dur_s:
        ノイズ区間の長さ（秒）。
    """
    c = consonant.lower()
    if c == 's':
        fc, Q, liprad, lp_post = 6500.0, 3.0, True, None
    elif c == 'sh':
        fc, Q, liprad, lp_post = 3800.0, 2.4, True, 4200.0
    elif c == 'h':
        fc, Q, liprad, lp_post = 1800.0, 1.4, False, 2600.0
    elif c == 'f':
        fc, Q, liprad, lp_post = 950.0, 1.3, False, 2100.0
    else:
        raise ValueError("synth_fricative: supported consonants are 's','sh','h','f'.")

    y = _gen_band_noise(dur_s, fs, fc, Q, liprad=liprad)
    if lp_post is not None:
        y = _one_pole_lp(y, fc=lp_post, fs=fs)
    y *= _db_to_lin(level_db)
    attack = 6 if c in ('h', 'f') else 4
    release = 16 if c in ('h', 'f') else 12
    y = _apply_fade(y, fs, attack_ms=attack, release_ms=release)
    return _normalize_peak(y, 0.6).astype(DTYPE)


def synth_plosive(consonant: str = 't', fs: int = 22050,
                  closure_ms: Optional[float] = None, burst_ms: Optional[float] = None,
                  aspiration_ms: Optional[float] = None, level_db: float = -10.0) -> np.ndarray:
    """
    無声破裂音（t / k）
    - t: 歯茎 → 高域寄りバースト
    - k: 軟口蓋 → 中域寄りバースト
    """
    c = consonant.lower()
    if c not in ('t', 'k'):
        raise ValueError("synth_plosive: supported only 't' or 'k'.")

    if c == 't':
        closure_ms = 24 if closure_ms is None else closure_ms
        burst_ms = 12 if burst_ms is None else burst_ms
        aspiration_ms = 18 if aspiration_ms is None else aspiration_ms
        fc_burst, Q_burst = 4500.0, 1.2
        fc_asp, Q_asp = 6000.0, 0.9
    else:
        closure_ms = 32 if closure_ms is None else closure_ms
        burst_ms = 14 if burst_ms is None else burst_ms
        aspiration_ms = 24 if aspiration_ms is None else aspiration_ms
        fc_burst, Q_burst = 1700.0, 1.2
        fc_asp, Q_asp = 2500.0, 0.9

    closure = np.zeros(_ms_to_samples(closure_ms, fs), dtype=DTYPE)

    # バースト（急峻すぎを避けるため唇放射なし＋指数減衰）
    burst_len = _ms_to_samples(burst_ms, fs)
    burst = _gen_band_noise(burst_len / fs, fs, fc_burst, Q_burst, liprad=False)
    tau = max(1.0, burst_len / 4.0)
    env = np.exp(-np.arange(burst_len, dtype=DTYPE) / tau)
    burst = burst[:burst_len] * env

    # アスピレーション（唇放射なし＋軽いLPで “s” 感を抑制）
    asp_len = _ms_to_samples(aspiration_ms, fs)
    aspiration = _gen_band_noise(asp_len / fs, fs, fc_asp, Q_asp, liprad=False)
    aspiration = _apply_fade(aspiration, fs, attack_ms=3, release_ms=18)
    aspiration = _one_pole_lp(aspiration, fc=4500.0, fs=fs)

    cons = np.concatenate([closure, burst, aspiration]).astype(DTYPE)
    cons *= _db_to_lin(level_db)
    return _normalize_peak(cons, 0.6)


def synth_affricate(consonant: str = 'ch', fs: int = 22050,
                    closure_ms: Optional[float] = None, fric_ms: Optional[float] = None,
                    level_db: float = -11.0) -> np.ndarray:
    """破擦音 /ch/, /ts/ の簡易版"""
    c = consonant.lower()
    if c not in ('ch', 'ts'):
        raise ValueError("synth_affricate: supported only 'ch' or 'ts'.")

    if c == 'ch':
        closure_ms = 26 if closure_ms is None else closure_ms
        burst_ms = 10
        fric_ms = 95.0 if fric_ms is None else fric_ms
        fric = synth_fricative('sh', dur_s=fric_ms/1000.0, fs=fs, level_db=-16.0)
    else:
        closure_ms = 24 if closure_ms is None else closure_ms
        burst_ms = 9
        fric_ms = 90.0 if fric_ms is None else fric_ms
        fric = synth_fricative('s', dur_s=fric_ms/1000.0, fs=fs, level_db=-16.0)

    base = synth_plosive('t', fs=fs, closure_ms=closure_ms,
                         burst_ms=burst_ms, aspiration_ms=0.0,
                         level_db=level_db)
    y = _crossfade(base, fric, fs, overlap_ms=14)
    return _normalize_peak(y, 0.6)


NASAL_PRESETS: Dict[str, Dict[str, Sequence[float]]] = {
    'n': {'F': [250.0, 1900.0, 2800.0], 'BW': [80.0, 160.0, 220.0]},
    'm': {'F': [220.0, 1600.0, 2500.0], 'BW': [70.0, 150.0, 210.0]},
}


def synth_nasal(consonant: str = 'n', f0: float = 120.0, dur_ms: float = 90.0,
                fs: int = 22050) -> np.ndarray:
    """簡易的な鼻音 /n/, /m/"""
    c = consonant.lower()
    if c not in NASAL_PRESETS:
        raise ValueError("synth_nasal: supported nasals are 'n' or 'm'.")

    dur_s = max(20.0, float(dur_ms)) / 1000.0
    spec = NASAL_PRESETS[c]
    F = spec['F']
    BW = spec['BW']
    y = _synth_vowel_fixed(F, BW, f0, dur_s, fs,
                           jitter_cents=4.0, shimmer_db=0.4, breath_level_db=-38.0)
    y = _apply_fade(y, fs, attack_ms=8.0 if c == 'n' else 10.0,
                    release_ms=22.0 if c == 'n' else 28.0)
    gain = 0.6 if c == 'n' else 0.7
    return _normalize_peak(y * gain, 0.5)


# =======================
# Building Blocks
# =======================

def _crossfade(a: np.ndarray, b: np.ndarray, fs: int, overlap_ms: float = 30.0) -> np.ndarray:
    """a → b をオーバーラップ結合（安全版）"""
    ov = _ms_to_samples(overlap_ms, fs)
    # ov が大きすぎる場合に安全に縮める
    ov = min(ov, len(a), len(b))
    if ov <= 0:
        return np.concatenate([a, b]).astype(DTYPE)

    fade_out = np.linspace(1.0, 0.0, ov, dtype=DTYPE)
    fade_in = 1.0 - fade_out
    head = a[:-ov]
    tail = a[-ov:] * fade_out + b[:ov] * fade_in
    rest = b[ov:]
    return np.concatenate([head, tail, rest]).astype(DTYPE)


# ===== onset helper =====

def _synth_vowel_fixed(F: Sequence[float], BW: Sequence[float], f0: float, dur_s: float, fs: int,
                       jitter_cents: float = 6.0, shimmer_db: float = 0.6, breath_level_db: float = -40.0) -> np.ndarray:
    """与えたフォルマントで固定合成（短区間）"""
    src = _glottal_source(f0, dur_s, fs, jitter_cents, shimmer_db)
    y = _apply_formant_filters(src, F, BW, fs)
    y = _add_breath_noise(y, breath_level_db)

    return _normalize_peak(y, PEAK_DEFAULT).astype(DTYPE)


def synth_vowel_with_onset(vowel: str, f0: float, fs: int,
                           total_ms: int = 240, onset_ms: int = 45,
                           F_onset: Optional[Sequence[float]] = None) -> np.ndarray:
    """母音先頭だけフォルマント遷移を与える簡易版"""
    spec = VOWEL_TABLE[vowel]
    F_t, BW_t = spec['F'], spec['BW']
    if not F_onset:
        return synth_vowel(vowel=vowel, f0=f0, dur_s=total_ms/1000.0, fs=fs)

    y_on = _synth_vowel_fixed(F_onset, BW_t, f0, onset_ms/1000.0, fs)
    y_rest = _synth_vowel_fixed(F_t, BW_t, f0, total_ms/1000.0, fs)
    return _crossfade(y_on, y_rest, fs, overlap_ms=10)


# =======================
# Composition (CV/フレーズ)
# =======================

GLIDE_ONSETS: Dict[str, Dict[str, Sequence[float]]] = {
    'w': {
        'a': [360.0, 950.0, 2550.0],
        'i': [340.0, 1200.0, 2850.0],
        'u': [330.0, 850.0, 2500.0],
        'e': [350.0, 1100.0, 2700.0],
        'o': [340.0, 950.0, 2550.0],
    },
    'y': {
        'a': [360.0, 2100.0, 2900.0],
        'i': [340.0, 2500.0, 3300.0],
        'u': [335.0, 2200.0, 3100.0],
        'e': [350.0, 2400.0, 3200.0],
        'o': [340.0, 2150.0, 3050.0],
    },
}


LIQUID_ONSETS: Dict[str, Sequence[float]] = {
    'a': [480.0, 1350.0, 2100.0],
    'i': [380.0, 1800.0, 2200.0],
    'u': [420.0, 1500.0, 2100.0],
    'e': [430.0, 1600.0, 2200.0],
    'o': [460.0, 1400.0, 2150.0],
}


def synth_cv(cons: str, vowel: str, f0: float = 120.0, fs: int = 22050,
             pre_ms: int = 0, cons_ms: Optional[int] = None, vowel_ms: int = 240,
             overlap_ms: int = 30, use_onset_transition: bool = False) -> np.ndarray:
    """子音 + 母音（50音相当までカバー）"""
    c = cons.lower()
    v = vowel.lower()
    if v not in VOWEL_TABLE:
        raise ValueError("Unknown vowel for synth_cv")

    head = np.zeros(_ms_to_samples(pre_ms, fs), dtype=DTYPE)

    if c in ('s', 'sh', 'h', 'f'):
        level_lookup = {'s': -14.0, 'sh': -15.0, 'h': -18.0, 'f': -18.0}
        default_ms = {'s': 160.0, 'sh': 150.0, 'h': 190.0, 'f': 170.0}
        fric_ms = float(cons_ms) if cons_ms is not None else default_ms[c]
        fric = synth_fricative(c, dur_s=fric_ms/1000.0, fs=fs,
                               level_db=level_lookup[c])
        vow = synth_vowel(vowel=v, f0=f0, dur_s=vowel_ms/1000.0, fs=fs)
        y = _crossfade(fric, vow, fs, overlap_ms=max(24, overlap_ms))

    elif c in ('t', 'k'):
        plosive = synth_plosive(c, fs=fs)

        # 母音（必要なら簡易遷移）
        if use_onset_transition and v == 'a':
            if c == 't':      # 歯茎：F2高めから
                F_on = [800, 1800, 3000]
            else:             # 軟口蓋：F2–F3 を接近させて開始（velar pinch）
                F_on = [800, 2200, 2400]
            vow = synth_vowel_with_onset('a', f0, fs, total_ms=vowel_ms, onset_ms=45, F_onset=F_on)
        else:
            vow = synth_vowel(vowel=v, f0=f0, dur_s=vowel_ms/1000.0, fs=fs)

        # ov を 6–12msにクランプ（母音に埋もれないよう短く）
        ov = 8 if c == 't' else 10
        try:
            req_ov = int(overlap_ms)
            ov = max(6, min(req_ov, 12))
        except Exception:
            pass
        y = _crossfade(plosive, vow, fs, overlap_ms=ov)

    elif c in ('ch', 'ts'):
        affric = synth_affricate(c, fs=fs)
        vow = synth_vowel(vowel=v, f0=f0, dur_s=vowel_ms/1000.0, fs=fs)
        y = _crossfade(affric, vow, fs, overlap_ms=max(12, min(overlap_ms, 18)))

    elif c == 'n':
        nasal_ms = float(cons_ms) if cons_ms is not None else 90.0
        nasal = synth_nasal('n', f0=f0, dur_ms=nasal_ms, fs=fs)
        vow = synth_vowel(vowel=v, f0=f0, dur_s=vowel_ms/1000.0, fs=fs)
        y = _crossfade(nasal, vow, fs, overlap_ms=max(25, overlap_ms))

    elif c == 'm':
        nasal_ms = float(cons_ms) if cons_ms is not None else 110.0
        nasal = synth_nasal('m', f0=f0, dur_ms=nasal_ms, fs=fs)
        vow = synth_vowel(vowel=v, f0=f0, dur_s=vowel_ms/1000.0, fs=fs)
        y = _crossfade(nasal, vow, fs, overlap_ms=max(28, overlap_ms))

    elif c == 'w':
        onset_ms = float(cons_ms) if cons_ms is not None else 60.0
        onset_ms = max(30.0, min(onset_ms, float(vowel_ms) - 10.0))
        F_on = GLIDE_ONSETS['w'].get(v, GLIDE_ONSETS['w']['a'])
        y = synth_vowel_with_onset(v, f0, fs, total_ms=vowel_ms,
                                   onset_ms=int(onset_ms), F_onset=F_on)

    elif c == 'y':
        onset_ms = float(cons_ms) if cons_ms is not None else 55.0
        onset_ms = max(28.0, min(onset_ms, float(vowel_ms) - 12.0))
        F_on = GLIDE_ONSETS['y'].get(v, GLIDE_ONSETS['y']['a'])
        y = synth_vowel_with_onset(v, f0, fs, total_ms=vowel_ms,
                                   onset_ms=int(onset_ms), F_onset=F_on)

    elif c == 'r':
        tap = synth_plosive('t', fs=fs, closure_ms=12.0, burst_ms=6.0,
                            aspiration_ms=4.0, level_db=-20.0)
        F_on = LIQUID_ONSETS.get(v, LIQUID_ONSETS['a'])
        vow = synth_vowel_with_onset(v, f0, fs, total_ms=vowel_ms,
                                     onset_ms=36, F_onset=F_on)
        y = _crossfade(tap, vow, fs, overlap_ms=12)

    else:
        raise ValueError("synth_cv: unsupported consonant for synth_cv")

    y = np.concatenate([head, y]).astype(DTYPE)
    return _normalize_peak(y, PEAK_DEFAULT)


def synth_cv_to_wav(cons: str, vowel: str, out_path: str, f0: float = 120.0, fs: int = 22050,
                    pre_ms: int = 0, cons_ms: Optional[int] = None, vowel_ms: int = 240,
                    overlap_ms: int = 30, use_onset_transition: bool = False) -> str:
    """CV を合成して WAV 保存"""
    y = synth_cv(cons, vowel, f0=f0, fs=fs,
                 pre_ms=pre_ms, cons_ms=cons_ms,
                 vowel_ms=vowel_ms, overlap_ms=overlap_ms,
                 use_onset_transition=use_onset_transition)
    return write_wav(out_path, y, fs=fs)


def synth_phrase_to_wav(vowels: Sequence[str], out_path: str, f0: float = 120.0,
                        unit_ms: int = 220, gap_ms: int = 30, fs: int = 22050) -> str:
    """
    MVP: 母音列 ['a','i',...] からフレーズ WAV を生成
    """
    buf = []
    for v in map(str.lower, vowels):
        if v in VOWEL_TABLE:
            y = synth_vowel(v, f0=f0, dur_s=unit_ms/1000.0, fs=fs)
            buf.append(y)
            buf.append(np.zeros(_ms_to_samples(gap_ms, fs), dtype=DTYPE))
    if not buf:
        buf = [np.zeros(int(fs*0.3), dtype=DTYPE)]
    yall = np.concatenate(buf)
    return write_wav(out_path, yall, fs=fs)


# =======================
# I/O
# =======================

def write_wav(path: str, data: np.ndarray, fs: int = 22050) -> str:
    """16-bit PCM で WAV 保存"""
    data16 = np.clip((data*32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data16.tobytes())
    return os.path.abspath(path)
