"""Audio I/O helpers."""
from __future__ import annotations

import os
import wave

import numpy as np

from .core import _ensure_array

__all__ = ["write_wav"]


def write_wav(path: str, audio: np.ndarray, sampleRate: int = 22050) -> str:
    """Write ``audio`` to ``path`` as 16-bit PCM mono WAV."""
    audio = _ensure_array(audio)
    data16 = np.clip((audio * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampleRate)
        wf.writeframes(data16.tobytes())
    return os.path.abspath(path)
