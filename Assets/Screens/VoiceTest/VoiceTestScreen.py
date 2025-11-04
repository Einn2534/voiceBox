# Created on 2024-05-08
# Author: ChatGPT
# Description: Voice test screen that reads entered text aloud.
"""Voice test screen that performs local speech synthesis."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import numpy as np
import pyaudio
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from Assets.Common.Audio.DigitalSignalProcessing import (
    normalize_token_sequence,
    synth_token_sequence,
    text_to_tokens,
)


logger = logging.getLogger(__name__)

LabelBase.register(
    name="DotGothic16",
    fn_regular="Assets/Common/Font/DotGothic16-Regular.ttf",
)

Builder.load_file("Assets/Screens/VoiceTest/VoiceTestScreen.kv")

PHRASE_INTERVAL_SECONDS = 0.12
MAX_PHRASE_LENGTH = 60


@dataclass(frozen=True)
class AudioSettings:
    """Collection of PyAudio-related constants."""

    format: int = pyaudio.paInt16
    channels: int = 1
    send_sample_rate: int = 16000
    receive_sample_rate: int = 24000
    dsp_sample_rate: int = 22050
    chunk_size: int = 1024


AUDIO_SETTINGS = AudioSettings()


class SpeechSynthesizer:
    """Queue-driven helper that turns text into spoken audio."""

    def __init__(self, audio_settings: AudioSettings) -> None:
        self.audio_settings = audio_settings
        self._lock = threading.Lock()
        self._queue: Optional[queue.Queue[Optional[str]]] = None
        self._stop_event: Optional[threading.Event] = None
        self._worker: Optional[threading.Thread] = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None

    def start(self) -> None:
        """Ensure that the worker thread is running."""
        if self._worker and self._worker.is_alive():
            return
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def enqueue(self, text: str) -> None:
        """Add a new phrase for synthesis.

        Args:
            text (str): Phrase that should be spoken.
        """
        if not text.strip():
            return
        if not self._queue:
            self.start()
        assert self._queue is not None
        self._queue.put(text)

    def stop(self) -> None:
        """Stop the worker thread without disposing the PyAudio backend."""
        if not self._worker:
            return
        if self._stop_event:
            self._stop_event.set()
        if self._queue:
            self._queue.put(None)
        self._worker.join(timeout=1.0)
        self._worker = None
        self._queue = None
        self._stop_event = None

    def shutdown(self) -> None:
        """Stop the worker thread and release the PyAudio backend."""
        self.stop()
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.exception("Failed to terminate PyAudio backend.")
            finally:
                self._pyaudio = None

    def _run(self) -> None:
        """Worker loop that processes queued phrases sequentially."""
        assert self._queue is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                continue

            for phrase in self._split_phrases(item):
                if self._stop_event.is_set():
                    break
                try:
                    self._play_phrase(phrase)
                except Exception:  # pragma: no cover - runtime safeguard
                    logger.exception("Failed to render phrase: %s", phrase)
                time.sleep(PHRASE_INTERVAL_SECONDS)

    def _split_phrases(self, text: str) -> List[str]:
        """Split long text into smaller phrases for natural playback.

        Args:
            text (str): Full response text that may contain multiple sentences.

        Returns:
            List[str]: Sequence of shorter phrases to synthesize.
        """
        normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not normalized:
            return []

        punctuation = set("。．.!?！？")
        parts = re.split(r"([。．.!?！？])", normalized)
        buffer = ""
        phrases: List[str] = []
        for part in parts:
            if not part:
                continue
            buffer += part
            if part in punctuation or len(buffer) >= MAX_PHRASE_LENGTH:
                phrase = buffer.strip()
                if phrase:
                    phrases.append(phrase)
                buffer = ""
        if buffer.strip():
            phrases.append(buffer.strip())
        return phrases

    def _ensure_backend(self) -> pyaudio.PyAudio:
        """Create the PyAudio backend on demand."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio

    def _play_phrase(self, text: str) -> None:
        """Perform synthesis for a single phrase and play it synchronously.

        Args:
            text (str): Phrase to convert to speech.
        """
        tokens = normalize_token_sequence(text_to_tokens(text))
        if not tokens:
            logger.debug("No tokens generated for phrase: %s", text)
            return

        waveform = synth_token_sequence(tokens, sampleRate=self.audio_settings.dsp_sample_rate)
        audio = np.clip(waveform, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)

        backend = self._ensure_backend()
        with self._lock:
            stream = backend.open(
                format=self.audio_settings.format,
                channels=self.audio_settings.channels,
                rate=self.audio_settings.dsp_sample_rate,
                output=True,
            )
            try:
                stream.write(audio_int16.tobytes())
            finally:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()


class VoiceTestScreen(Screen):
    """Screen that plays back typed text via speech synthesis."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_settings = AUDIO_SETTINGS
        self.speech_synthesizer = SpeechSynthesizer(self.audio_settings)

    def on_enter(self):
        """Start the speech synthesizer when the screen is entered."""
        logger.info("VoiceTest screen entered")
        self._schedule_label_update("Ready")
        self.speech_synthesizer.start()

    def on_leave(self):
        """Stop speech resources when the screen is left."""
        logger.info("VoiceTest screen leaving")
        self.speech_synthesizer.shutdown()

    def send_text(self) -> None:
        """Read the text from the input field aloud."""
        user_input = self.ids.apiRequest.text.strip()
        if not user_input:
            logger.debug("Empty input ignored")
            return

        self.speech_synthesizer.enqueue(user_input)
        self._schedule_label_update(user_input)
        self.ids.apiRequest.text = ""

    def update_label(self, text: str) -> None:
        """Update the response label text on the UI thread.

        Args:
            text (str): Text that should be displayed in the label.
        """
        if hasattr(self, "ids") and "apiResponse" in self.ids:
            self.ids.apiResponse.text = text
        else:
            logger.warning("apiResponse label not found in ids")

    def _schedule_label_update(self, text: str) -> None:
        """Schedule a label update on the Kivy UI thread.

        Args:
            text (str): Text to render when the scheduled callback fires.
        """
        Clock.schedule_once(partial(self._update_label_callback, text))

    def _update_label_callback(self, text: str, _dt: float) -> None:
        """Callback executed by Kivy's Clock to update the label.

        Args:
            text (str): Text that should be applied to the label.
            _dt (float): Delta time provided by Kivy's scheduler.
        """
        self.update_label(text)
