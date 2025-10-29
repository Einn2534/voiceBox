"""Text normalisation and tokenisation helpers."""
from __future__ import annotations

import unicodedata
from typing import Any, Iterable, List, Optional

import numpy as np

from .constants import (
    CV_TOKEN_MAP,
    NASAL_TOKEN_MAP,
    PAUSE_TOKEN,
    VOWEL_TABLE,
    _HAND_DAKUTEN_MAP,
    _KANA_BASE_MAP,
    _KANA_DIGRAPH_MAP,
    _MAX_ROMAJI_TOKEN_LENGTH,
    _PUNCTUATION_CHARS,
    _VALID_TOKENS,
    _VOICED_KANA_MAP,
)

__all__ = ["text_to_tokens", "normalize_token_sequence"]


def _normalize_to_hiragana(text: str) -> str:
    """NFKC normalisation followed by katakana→hiragana conversion."""
    normalized = unicodedata.normalize('NFKC', text)
    buf: List[str] = []
    for ch in normalized:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F3:  # カタカナ → ひらがな
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
    if tokens and tokens[-1] == PAUSE_TOKEN:
        return
    tokens.append(PAUSE_TOKEN)


def _extend_last_vowel(tokens: List[str]) -> None:
    for t in reversed(tokens):
        if t == PAUSE_TOKEN:
            continue
        if t in VOWEL_TABLE:
            tokens.append(t)
            return
        if t in CV_TOKEN_MAP:
            tokens.append(CV_TOKEN_MAP[t][1])
            return


def _parse_romaji_sequence(seq: str) -> List[str]:
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
        nonlocal roma_buf
        if roma_buf:
            tokens.extend(_parse_romaji_sequence(''.join(roma_buf)))
            roma_buf = []

    normalized = _normalize_to_hiragana(text)
    n, pos = len(normalized), 0

    while pos < n:
        ch = normalized[pos]

        if ch.isascii() and ch.isalpha():
            roma_buf.append(ch.lower())
            pos += 1
            continue

        flush_buf()

        if ch.isspace() or ch in _PUNCTUATION_CHARS:
            _append_pause(tokens)
            pos += 1
            continue

        if ch == 'ー':
            _extend_last_vowel(tokens)
            pos += 1
            continue

        if ch == 'っ':
            _append_pause(tokens)
            pos += 1
            continue

        nxt = normalized[pos + 1] if pos + 1 < n else ''
        pair = ch + nxt
        if pair in _KANA_DIGRAPH_MAP:
            tokens.extend(_KANA_DIGRAPH_MAP[pair])
            pos += 2
            continue

        if ch in _KANA_BASE_MAP:
            tokens.extend(_KANA_BASE_MAP[ch])
            pos += 1
            continue

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

        if ch.isdigit():
            _append_pause(tokens)
            pos += 1
            continue

        pos += 1

    flush_buf()

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
    """Normalise arbitrary token containers into a flat list of strings.

    Args:
        tokens (Optional[Any]): Token-like sequence returned from synthesis helpers.

    Returns:
        List[str]: Flat list of tokens suitable for downstream processing.
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
        iterable: Iterable[Any] = list(tokens)
    except TypeError:
        return [str(tokens)]
    return [str(token) for token in iterable]
