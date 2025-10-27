# Created on: 2025-10-10
# Author: nullptr
# Description: Voice chat screen integrating Gemini Live API with audio playback.

import asyncio
import os
import queue
import re
import threading
import time
import traceback
from typing import List, Optional, Sequence

import numpy as np
import pyaudio
from google import genai
from google.genai import types
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from Assets.Common.Audio.DigitalSignalProcessing import (
    synth_token_sequence,
    text_to_tokens,
)

LabelBase.register(
    name="DotGothic16",
    fn_regular="Assets/Common/Font/DotGothic16-Regular.ttf",
)

Builder.load_file('Assets/Screens/VoiceTest/VoiceTestScreen.kv')

GEMINI_API_KEY_ENV = 'AIzaSyBC0-gE_aSsMXNL0fvFApzijUkEPRC8wSc' 
MODEL_NAME = "models/gemini-2.0-flash-live-001"
CONNECTION_POLL_INTERVAL = 0.1
SPEECH_QUEUE_MAXSIZE = 5
SPEECH_QUEUE_TIMEOUT = 0.1
PHRASE_MAX_LENGTH = 60
PHRASE_PAUSE_SECONDS = 0.12
DSP_SAMPLE_RATE = 22050
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
DEFAULT_SYSTEM_PROMPT = "あなたは優秀な英語AIアシスタントです。"


def load_gemini_api_key() -> str:
    """Read and validate the Gemini API key from the environment variable."""
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {GEMINI_API_KEY_ENV} must be set to use the voice test screen."
        )
    return api_key


def build_live_connect_config() -> types.LiveConnectConfig:
    """Create the configuration payload required when connecting to Gemini Live."""
    return types.LiveConnectConfig(
        system_instruction=types.Content(
            parts=[types.Part(text=DEFAULT_SYSTEM_PROMPT)]
        ),
        response_modalities=["text"],
    )


def create_genai_client(api_key: str) -> genai.Client:
    """Instantiate a Gemini client for the asynchronous live session APIs.

    Args:
        api_key: Gemini authentication token obtained from the environment.
    """
    return genai.Client(http_options={"api_version": "v1beta"}, api_key=api_key)


class VoiceTestScreen(Screen):
    """Provide a Gemini-powered interactive voice testing screen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.session: Optional[object] = None
        self.running = False
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None
        self._speech_lock = threading.Lock()
        self._speech_queue: Optional[queue.Queue] = None
        self._speech_worker: Optional[threading.Thread] = None
        self._speech_stop_event: Optional[threading.Event] = None
        self._audio_interface: Optional[pyaudio.PyAudio] = None
        self._client = create_genai_client(load_gemini_api_key())
        self._live_config = build_live_connect_config()

    def on_enter(self):
        """Start background workers and connect to Gemini when the screen appears."""
        print("VoiceTest Screen Entered!")
        self.ids.apiResponse.text = "Connecting..."
        self._start_speech_worker()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever,
            daemon=True,
        )
        self.loop_thread.start()
        asyncio.run_coroutine_threadsafe(self.run_audio_loop(), self.loop)

    def on_leave(self):
        """Tear down background workers and connections when leaving the screen."""
        self.running = False
        self._stop_speech_worker()
        self._shutdown_async_components()
        self._close_audio_interface()

    async def run_audio_loop(self):
        """Maintain the Gemini Live connection and coordinate message handling."""
        try:
            async with (
                self._client.aio.live.connect(model=MODEL_NAME, config=self._live_config) as session,
                asyncio.TaskGroup() as task_group,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=SPEECH_QUEUE_MAXSIZE)
                self.running = True
                task_group.create_task(self.receive_text())
                Clock.schedule_once(lambda dt: setattr(self.ids.apiResponse, "text", "Connected"))
                while self.running:
                    await asyncio.sleep(CONNECTION_POLL_INTERVAL)
        except Exception as error:
            print(traceback.format_exc())
            Clock.schedule_once(
                lambda dt, message=str(error): setattr(self.ids.apiResponse, "text", f"Error: {message}")
            )
        finally:
            self.running = False
            self.session = None
            Clock.schedule_once(lambda dt: setattr(self.ids.apiResponse, "text", "Disconnect"))

    def send_text(self):
        """Dispatch user input from the UI to Gemini asynchronously."""
        if not self.session or not self.loop:
            print("セッション未接続またはループ未初期化")
            return
        user_input = self.ids.apiRequest.text.strip()
        if not user_input:
            print("入力不備")
            return
        asyncio.run_coroutine_threadsafe(self._async_send_text(user_input), self.loop)

    async def _async_send_text(self, text: str):
        """Send user text to Gemini within the live session context.

        Args:
            text: Cleaned user utterance collected from the request text field.
        """
        if not self.session:
            return
        try:
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True,
            )
        except Exception as error:
            print("send_text error:", error)

    async def receive_text(self):
        """Stream Gemini responses and update the UI with intermediate and final text."""
        if not self.session:
            return
        try:
            while self.running and self.session:
                turn = await self.session.receive()
                full_text = ""
                async for response in turn:
                    if response.text:
                        full_text += response.text
                        Clock.schedule_once(
                            lambda dt, partial=response.text: self.update_label(partial)
                        )
                if full_text:
                    Clock.schedule_once(
                        lambda dt, message=full_text: self.on_final_response(message)
                    )
        except Exception as error:
            print("receive_text error:", error)
            Clock.schedule_once(
                lambda dt, message=str(error): setattr(self.ids.apiResponse, "text", f"受信エラー: {message}")
            )

    def update_label(self, text: str):
        """Update the response label with partial Gemini output on the UI thread.

        Args:
            text: Fragment of Gemini output used to refresh the UI label.
        """
        if hasattr(self, "ids") and "apiResponse" in self.ids:
            self.ids.apiResponse.text = text
        else:
            print("⚠️ Label(apiResponse) が見つかりません")

    def on_final_response(self, text: str):
        """Handle the final Gemini response by updating the UI and queueing speech.

        Args:
            text: Aggregated Gemini response that should be spoken aloud.
        """
        self.update_label(text)
        self.enqueue_speech(text)

    def enqueue_speech(self, text: str) -> None:
        """Place synthesized speech requests into the worker queue.

        Args:
            text: Sentence or paragraph queued for speech synthesis.
        """
        if not text or not self.running:
            return
        self._start_speech_worker()
        assert self._speech_queue is not None
        self._speech_queue.put(text)

    def _start_speech_worker(self) -> None:
        """Ensure that the speech worker thread and queue exist."""
        if self._speech_worker and self._speech_worker.is_alive():
            return
        self._speech_queue = queue.Queue(maxsize=SPEECH_QUEUE_MAXSIZE)
        self._speech_stop_event = threading.Event()
        self._speech_worker = threading.Thread(target=self._speech_loop, daemon=True)
        self._speech_worker.start()

    def _stop_speech_worker(self) -> None:
        """Stop the speech worker thread and release its resources."""
        if not self._speech_worker:
            return
        if self._speech_stop_event:
            self._speech_stop_event.set()
        if self._speech_queue is not None:
            try:
                self._speech_queue.put_nowait(None)
            except Exception:
                pass
        self._speech_worker.join(timeout=1.0)
        self._speech_worker = None
        self._speech_queue = None
        self._speech_stop_event = None

    def _speech_loop(self) -> None:
        """Consume queued text and synthesize audio sequentially."""
        assert self._speech_queue is not None
        assert self._speech_stop_event is not None
        while not self._speech_stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=SPEECH_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            if item is None:
                continue
            for phrase in self._iterate_phrases(item):
                if self._speech_stop_event.is_set() or not self.running:
                    break
                self._play_phrase(phrase)
                time.sleep(PHRASE_PAUSE_SECONDS)

    def _iterate_phrases(self, text: str) -> Sequence[str]:
        """Split the incoming text into manageable phrases for playback.

        Args:
            text: Gemini response awaiting segmentation into speech phrases.

        Returns:
            Sequence of phrases constrained by punctuation and length limits.
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
            if part in punctuation or len(buffer) >= PHRASE_MAX_LENGTH:
                phrase = buffer.strip()
                if phrase:
                    phrases.append(phrase)
                buffer = ""
        if buffer.strip():
            phrases.append(buffer.strip())
        return phrases

    def _play_phrase(self, text: str) -> None:
        """Convert a phrase into waveform audio and play it synchronously.

        Args:
            text: Phrase scheduled for audio synthesis and playback.
        """
        if not text or not self.running:
            return
        stream = None
        raw_tokens: Optional[Sequence] = None
        tokens: List[str] = []
        try:
            raw_tokens = text_to_tokens(text)
            tokens = self._normalize_tokens(raw_tokens)
            if not tokens:
                print("speak_text: no speakable tokens")
                return
            waveform = synth_token_sequence(tokens, sampleRate=DSP_SAMPLE_RATE)
            audio = np.clip(waveform, -1.0, 1.0)
            audio_int16 = (audio * 32767.0).astype(np.int16)
            if not self.running:
                return
            with self._speech_lock:
                stream = self._open_audio_stream()
                stream.write(audio_int16.tobytes())
        except Exception as speak_error:
            print("speak_text error:", speak_error)
            if raw_tokens is not None:
                try:
                    print(
                        "  raw_tokens type=",
                        type(raw_tokens),
                        "repr=",
                        repr(raw_tokens)[:200],
                    )
                except Exception:
                    pass
            if tokens:
                try:
                    preview = tokens if isinstance(tokens, list) else list(tokens)
                    print(
                        "  normalized tokens len=",
                        len(preview),
                        "sample=",
                        preview[:10],
                    )
                except Exception:
                    pass
            traceback.print_exc()
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                stream.close()

    def _normalize_tokens(self, raw_tokens: Optional[Sequence]) -> List[str]:
        """Normalize token collections into a plain list representation.

        Args:
            raw_tokens: Arbitrary token container yielded by the DSP helper.

        Returns:
            Flattened list of tokens suitable for synthesis.
        """
        if raw_tokens is None:
            return []
        if isinstance(raw_tokens, np.ndarray):
            return raw_tokens.ravel().tolist()
        if isinstance(raw_tokens, list):
            return raw_tokens
        if isinstance(raw_tokens, tuple):
            return list(raw_tokens)
        if isinstance(raw_tokens, str):
            return [raw_tokens]
        try:
            return list(raw_tokens)  # type: ignore[arg-type]
        except TypeError:
            return [str(raw_tokens)]

    def _open_audio_stream(self) -> pyaudio.Stream:
        """Obtain a PyAudio stream for playback, creating the interface on demand."""
        interface = self._ensure_audio_interface()
        return interface.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=DSP_SAMPLE_RATE,
            output=True,
        )

    def _ensure_audio_interface(self) -> pyaudio.PyAudio:
        """Create the PyAudio interface if it has not been initialized."""
        if self._audio_interface is None:
            self._audio_interface = pyaudio.PyAudio()
        return self._audio_interface

    def _close_audio_interface(self) -> None:
        """Dispose of the PyAudio interface if it exists."""
        if self._audio_interface is not None:
            try:
                self._audio_interface.terminate()
            except Exception:
                pass
            self._audio_interface = None

    def _shutdown_async_components(self) -> None:
        """Stop the background asyncio loop and associated thread."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread:
            self.loop_thread.join(timeout=1.0)
        if self.loop:
            try:
                self.loop.close()
            except Exception:
                pass
        self.loop = None
        self.loop_thread = None
