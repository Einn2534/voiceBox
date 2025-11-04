# Created on 2024-05-08
# Author: ChatGPT
# Description: Voice test screen that streams Gemini responses and synthesises audio.
"""Voice test screen that connects to Gemini Live and performs speech synthesis."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import numpy as np
import pyaudio
from google import genai
from google.genai import types
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

DEFAULT_SYSTEM_INSTRUCTION = "あなたは優秀な英語AIアシスタントです。"
CONFIG_FILE_NAME = "GeminiSettings.json"
CONFIG_FILE_ENCODING = "utf-8"
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), CONFIG_FILE_NAME)
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


@dataclass(frozen=True)
class GeminiSettings:
    """Configuration for connecting to the Gemini live API."""

    model: str
    system_instruction: str
    api_key: str
    api_version: str = "v1beta"

    def build_client(self) -> genai.Client:
        """Create a Gemini client using the configured API key."""
        if not self.api_key:
            raise RuntimeError(
                "Gemini設定ファイルにapiKeyが未設定です。"
            )
        return genai.Client(
            http_options={"api_version": self.api_version},
            api_key=self.api_key,
        )

    def build_live_config(self) -> types.LiveConnectConfig:
        """Create the live configuration shared by all sessions."""
        return types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_instruction)]
            ),
            response_modalities=["text"],
        )

    @classmethod
    def from_file(cls, path: str) -> "GeminiSettings":
        """Build settings from a JSON configuration file.

        Args:
            path (str): Absolute path to the configuration JSON file.

        Returns:
            GeminiSettings: Parsed settings instance built from the file content.
        """
        try:
            with open(path, "r", encoding=CONFIG_FILE_ENCODING) as config_file:
                data = json.load(config_file)
        except FileNotFoundError as error:
            raise RuntimeError(f"Gemini設定ファイルが見つかりません: {path}") from error
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Gemini設定ファイルの形式が不正です: {path}") from error

        model = str(data.get("model", "")).strip()
        api_key = str(data.get("apiKey", "")).strip()
        instruction = str(
            data.get("systemInstruction", DEFAULT_SYSTEM_INSTRUCTION)
        ).strip()

        if not model:
            raise RuntimeError("Gemini設定ファイルにmodelが定義されていません。")
        if not api_key:
            raise RuntimeError("Gemini設定ファイルにapiKeyが定義されていません。")
        if not instruction:
            instruction = DEFAULT_SYSTEM_INSTRUCTION

        return cls(model=model, system_instruction=instruction, api_key=api_key)


def load_gemini_settings() -> GeminiSettings:
    """Load Gemini configuration from the VoiceTest settings file.

    Returns:
        GeminiSettings: Settings loaded from the configuration JSON file.
    """
    return GeminiSettings.from_file(CONFIG_FILE_PATH)


AUDIO_SETTINGS = AudioSettings()
GEMINI_SETTINGS = load_gemini_settings()


class SpeechSynthesizer:
    """Queue-driven helper that turns Gemini text into spoken audio."""

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
    """Voice test screen class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.session = None
        self.running = False
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None
        self._session_future: Optional[asyncio.Future] = None
        self._gemini_client: Optional[genai.Client] = None
        self.gemini_settings = GEMINI_SETTINGS
        self.audio_settings = AUDIO_SETTINGS
        self.speech_synthesizer = SpeechSynthesizer(self.audio_settings)

    def on_enter(self):
        """Initialise the asynchronous loop and connect to Gemini."""
        logger.info("VoiceTest screen entered")
        self._schedule_label_update("Connecting...")
        self.speech_synthesizer.start()

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()

        self._session_future = asyncio.run_coroutine_threadsafe(self.run_audio_loop(), self.loop)

    def on_leave(self):
        """Tear down background resources when leaving the screen."""
        logger.info("VoiceTest screen leaving")
        self.running = False
        self.speech_synthesizer.shutdown()
        self._cancel_session_future()
        self._shutdown_async_loop()
        if hasattr(self, "audio_stream") and getattr(self, "audio_stream"):
            self.audio_stream.close()

    async def run_audio_loop(self) -> None:
        """Maintain the Gemini live session lifecycle."""
        try:
            client = self._get_or_create_client()
        except RuntimeError as error:
            logger.error("Gemini configuration error: %s", error)
            self._schedule_label_update(f"Error: {error}")
            return
        except Exception as error:  # pragma: no cover - defensive logging
            logger.exception("Failed to create Gemini client")
            self._schedule_label_update(f"Error: {error}")
            return

        try:
            async with (
                client.aio.live.connect(
                    model=self.gemini_settings.model,
                    config=self.gemini_settings.build_live_config(),
                ) as session,
                asyncio.TaskGroup() as task_group,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.running = True

                task_group.create_task(self.receive_text(session))

                self._schedule_label_update("Connected")

                while self.running:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Audio loop cancelled")
        except Exception as error:  # pragma: no cover - defensive logging
            logger.exception("Unexpected error in run_audio_loop")
            self._schedule_label_update(f"Error: {error}")
        finally:
            self.running = False
            self.session = None
            self.audio_in_queue = None
            self.out_queue = None
            self._schedule_label_update("Disconnect")

    def send_text(self) -> None:
        """Send the text from the UI input to Gemini and queue speech."""
        if not self.session or not self.loop or not self.loop.is_running():
            logger.warning("Cannot send text: session or loop is not ready.")
            return

        user_input = self.ids.apiRequest.text.strip()
        if not user_input:
            logger.debug("Empty input ignored")
            return

        self.speech_synthesizer.enqueue(user_input)
        asyncio.run_coroutine_threadsafe(self._async_send_text(user_input), self.loop)

    async def _async_send_text(self, text: str) -> None:
        """Send text to the Gemini session asynchronously.

        Args:
            text (str): Text content that will be forwarded to Gemini.
        """
        if not self.session:
            raise RuntimeError("Gemini session is not ready.")
        try:
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True,
            )
        except Exception as error:  # pragma: no cover - defensive logging
            logger.exception("Failed to send text to Gemini")
            self._schedule_label_update(f"送信エラー: {error}")

    async def receive_text(self, session) -> None:
        """Receive streaming responses from Gemini and update the UI.

        Args:
            session: Gemini live session that yields streaming responses.
        """
        try:
            while self.running:
                turn = session.receive()
                full_text = ""

                async for response in turn:
                    if response.text:
                        full_text += response.text
                        self._schedule_label_update(response.text)

                if full_text:
                    self._handle_final_response(full_text)
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
        except Exception as error:  # pragma: no cover - defensive logging
            logger.exception("Failed while receiving Gemini text")
            self._schedule_label_update(f"受信エラー: {error}")

    def _handle_final_response(self, text: str) -> None:
        """Handle the final response by updating the UI and queuing speech.

        Args:
            text (str): Final aggregated response text from Gemini.
        """
        self._schedule_label_update(text)
        self.speech_synthesizer.enqueue(text)

    def update_label(self, text: str) -> None:
        """Update the API response label text on the UI thread.

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

    def _cancel_session_future(self) -> None:
        """Cancel the background session future if it is still running."""
        if self._session_future and not self._session_future.done():
            self._session_future.cancel()
        self._session_future = None

    def _shutdown_async_loop(self) -> None:
        """Stop and close the background asyncio loop if it exists."""
        if not self.loop:
            return
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread:
            self.loop_thread.join(timeout=1.0)
        self.loop.close()
        self.loop = None
        self.loop_thread = None

    def _get_or_create_client(self) -> genai.Client:
        """Create or reuse a Gemini client instance."""
        if self._gemini_client is None:
            self._gemini_client = self.gemini_settings.build_client()
        return self._gemini_client
