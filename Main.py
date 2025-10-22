# Date: 2024-06-09
# Author: ChatGPT
# Project: GIAN
# Description: Voice synthesis UI entry point built with Kivy.
"""Kivy UI for the GIAN vocal synthesis demonstrator."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Callable, Iterable, Sequence, Tuple

from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.slider import Slider

import numpy as np

from Dsp import synth_cv, synth_cv_to_wav, synth_nasal, synth_phrase_to_wav, synth_vowel, write_wav

FS = 22050  # サンプリング周波数（統一）

GOJUON_ROWS: Sequence[Sequence[Tuple[str, Tuple[str, str]] | None]] = [
    [
        ("KA", ("k", "a")),
        ("KI", ("k", "i")),
        ("KU", ("k", "u")),
        ("KE", ("k", "e")),
        ("KO", ("k", "o")),
    ],
    [
        ("SA", ("s", "a")),
        ("SHI", ("sh", "i")),
        ("SU", ("s", "u")),
        ("SE", ("s", "e")),
        ("SO", ("s", "o")),
    ],
    [
        ("TA", ("t", "a")),
        ("CHI", ("ch", "i")),
        ("TSU", ("ts", "u")),
        ("TE", ("t", "e")),
        ("TO", ("t", "o")),
    ],
    [
        ("NA", ("n", "a")),
        ("NI", ("n", "i")),
        ("NU", ("n", "u")),
        ("NE", ("n", "e")),
        ("NO", ("n", "o")),
    ],
    [
        ("HA", ("h", "a")),
        ("HI", ("h", "i")),
        ("FU", ("f", "u")),
        ("HE", ("h", "e")),
        ("HO", ("h", "o")),
    ],
    [
        ("MA", ("m", "a")),
        ("MI", ("m", "i")),
        ("MU", ("m", "u")),
        ("ME", ("m", "e")),
        ("MO", ("m", "o")),
    ],
    [
        ("YA", ("y", "a")),
        None,
        ("YU", ("y", "u")),
        None,
        ("YO", ("y", "o")),
    ],
    [
        ("RA", ("r", "a")),
        ("RI", ("r", "i")),
        ("RU", ("r", "u")),
        ("RE", ("r", "e")),
        ("RO", ("r", "o")),
    ],
    [
        ("WA", ("w", "a")),
        None,
        None,
        None,
        ("WO", ("w", "o")),
    ],
    [
        ("N", ("nasal", "n")),
        None,
        None,
        None,
        None,
    ],
]


class Root(BoxLayout):
    """Root layout that wires the sliders, buttons, and status label."""

    def __init__(self, **kwargs) -> None:
        """Configure the full UI layout and bind user actions."""
        super().__init__(orientation="vertical", spacing=8, padding=12, **kwargs)

        self.pitchSlider, _ = self._add_slider(
            labelTemplate="F0(Hz):",
            minValue=80,
            maxValue=260,
            defaultValue=120,
            width="80dp",
            showValue=False,
        )
        self.overlapSlider, self.overlapLabel = self._add_slider(
            labelTemplate="Overlap(ms): {value}",
            minValue=6,
            maxValue=30,
            defaultValue=10,
        )
        self.vowelDurationSlider, self.vowelDurationLabel = self._add_slider(
            labelTemplate="Vowel(ms): {value}",
            minValue=120,
            maxValue=300,
            defaultValue=200,
        )

        onsetRow = BoxLayout(size_hint_y=None, height="40dp", spacing=8)
        onsetRow.add_widget(Label(text="Onset (t/k 強調):", size_hint_x=None, width="140dp"))
        self.onsetCheckBox = CheckBox(active=True)
        onsetRow.add_widget(self.onsetCheckBox)
        self.add_widget(onsetRow)

        vowelButtonSpecs = [
            (vowel.upper(), lambda vowelValue=vowel: self.speak_vowel(vowelValue))
            for vowel in ["a", "i", "u", "e", "o"]
        ]
        self._add_button_row(vowelButtonSpecs)

        for rowEntries in GOJUON_ROWS:
            self._add_gojuon_row(rowEntries)

        statusRow = BoxLayout(size_hint_y=None, height="48dp", spacing=8)
        sequenceButton = Button(text="Play A-I-U-E-O")
        sequenceButton.bind(on_release=lambda *_: self.play_sequence())
        statusRow.add_widget(sequenceButton)

        loveButton = Button(text="Say あいしてる")
        loveButton.bind(on_release=lambda *_: self.speak_aishiteru())
        statusRow.add_widget(loveButton)

        self.statusLabel = Label(text="Ready")
        statusRow.add_widget(self.statusLabel)
        self.add_widget(statusRow)

        self.currentSound = None

    def _play_file(self, filePath: str) -> None:
        """Load the generated audio file and play it immediately."""
        if self.currentSound:
            try:
                self.currentSound.stop()
            except Exception:
                pass
        self.currentSound = SoundLoader.load(filePath)
        if self.currentSound:
            self.currentSound.play()
            self.statusLabel.text = f"Playing: {os.path.basename(filePath)}"
        else:
            self.statusLabel.text = "Failed to load sound"

    def speak_vowel(self, vowel: str) -> None:
        """Synthesize and play a standalone vowel."""
        vowel = vowel.lower()
        self._update_status(f"Synth {vowel}...")
        baseF0 = float(self.pitchSlider.value)
        durationSeconds = max(0.12, float(self.vowelDurationSlider.value) / 1000.0)

        def build(tempPath: str) -> str:
            waveform = synth_vowel(vowel=vowel, f0=baseF0, durationSeconds=durationSeconds, sampleRate=FS)
            return write_wav(tempPath, waveform, sampleRate=FS)

        self._play_async(f"vowel_{vowel}.wav", build, f"Played {vowel.upper()}")

    def speak_cv(self, consonant: str, vowel: str) -> None:
        """Synthesize and play a consonant-vowel pair."""
        consonant = consonant.lower()
        vowel = vowel.lower()
        label = f"{consonant.upper()}-{vowel.upper()}"
        self._update_status(f"Synth {label}...")
        baseF0 = float(self.pitchSlider.value)
        overlapMs = int(self.overlapSlider.value)
        vowelMs = int(self.vowelDurationSlider.value)
        useOnset = bool(self.onsetCheckBox.active)

        def build(tempPath: str) -> str:
            return synth_cv_to_wav(
                consonant,
                vowel,
                tempPath,
                f0=baseF0,
                sampleRate=FS,
                preMilliseconds=0,
                consonantMilliseconds=None,
                vowelMilliseconds=vowelMs,
                overlapMilliseconds=overlapMs,
                useOnsetTransition=useOnset,
            )

        self._play_async(f"{consonant}{vowel}.wav", build, f"Played {label}")

    def speak_nasal(self, consonant: str) -> None:
        """Synthesize and play a nasal consonant."""
        consonant = consonant.lower()
        label = "N" if consonant == "n" else consonant.upper()
        self._update_status(f"Synth {label}...")
        baseF0 = float(self.pitchSlider.value)
        durationMs = max(80, int(self.vowelDurationSlider.value * 0.6))

        def build(tempPath: str) -> str:
            waveform = synth_nasal(consonant, f0=baseF0, durationMilliseconds=durationMs, sampleRate=FS)
            return write_wav(tempPath, waveform, sampleRate=FS)

        self._play_async(f"nasal_{consonant}.wav", build, f"Played {label}")

    def speak_aishiteru(self) -> None:
        """Synthesize the fixed phrase 'あいしてる'."""
        self._update_status("Synth あいしてる...")
        baseF0 = float(self.pitchSlider.value)
        overlapMs = int(self.overlapSlider.value)
        vowelMs = int(self.vowelDurationSlider.value)
        useOnset = bool(self.onsetCheckBox.active)

        def build(tempPath: str) -> str:
            def pause(milliseconds: float) -> None:
                sampleCount = int(FS * (milliseconds / 1000.0))
                if sampleCount > 0:
                    segments.append(np.zeros(sampleCount, dtype=np.float32))

            vowelSeconds = max(0.12, vowelMs / 1000.0)
            segments = [synth_vowel("a", f0=baseF0, durationSeconds=vowelSeconds, sampleRate=FS)]
            pause(45.0)
            segments.append(synth_vowel("i", f0=baseF0, durationSeconds=max(0.1, vowelSeconds * 0.8), sampleRate=FS))
            pause(70.0)

            cvParameters = dict(
                f0=baseF0,
                sampleRate=FS,
                vowelMilliseconds=vowelMs,
                overlapMilliseconds=overlapMs,
                useOnsetTransition=useOnset,
            )
            segments.append(synth_cv("sh", "i", **cvParameters))
            pause(50.0)
            segments.append(synth_cv("t", "e", **cvParameters))
            pause(35.0)
            ruVowel = vowelMs + max(40, vowelMs // 3)
            segments.append(
                synth_cv(
                    "r",
                    "u",
                    f0=baseF0,
                    sampleRate=FS,
                    vowelMilliseconds=ruVowel,
                    overlapMilliseconds=overlapMs,
                    useOnsetTransition=useOnset,
                )
            )

            audio = np.concatenate(segments).astype(np.float32, copy=False)
            return write_wav(tempPath, audio, sampleRate=FS)

        self._play_async("phrase_aishiteru.wav", build, "Played あいしてる")

    def play_sequence(self) -> None:
        """Synthesize the vowel sequence A-I-U-E-O."""
        self._update_status("Synth sequence...")
        baseF0 = float(self.pitchSlider.value)

        def build(tempPath: str) -> str:
            vowels = ["a", "i", "u", "e", "o"]
            unitMs = max(120, int(self.vowelDurationSlider.value))
            gapMs = max(40, int(unitMs * 0.25))
            return synth_phrase_to_wav(
                vowels,
                tempPath,
                f0=baseF0,
                unitMilliseconds=unitMs,
                gapMilliseconds=gapMs,
                sampleRate=FS,
            )

        self._play_async("sequence_aiueo.wav", build, "Played A-I-U-E-O")

    def _add_slider(
        self,
        *,
        labelTemplate: str,
        minValue: float,
        maxValue: float,
        defaultValue: float,
        width: str = "120dp",
        showValue: bool = True,
    ) -> Tuple[Slider, Label]:
        """Create a slider with an optional value label and add it to the layout."""
        rowLayout = BoxLayout(size_hint_y=None, height="48dp", spacing=8)
        sliderLabel = Label(size_hint_x=None, width=width)

        slider = Slider(min=minValue, max=maxValue, value=defaultValue)

        if showValue and "{" in labelTemplate:
            sliderLabel.text = labelTemplate.format(value=int(defaultValue))

            def update_label(_, newValue) -> None:
                sliderLabel.text = labelTemplate.format(value=int(newValue))

            slider.bind(value=update_label)
        else:
            sliderLabel.text = labelTemplate

        rowLayout.add_widget(sliderLabel)
        rowLayout.add_widget(slider)
        self.add_widget(rowLayout)
        return slider, sliderLabel

    def _add_button_row(self, buttonSpecs: Iterable[Tuple[str, Callable[[], None]]], height: str = "48dp") -> None:
        """Create a row of buttons based on the given specifications."""
        rowLayout = BoxLayout(size_hint_y=None, height=height, spacing=8)
        for buttonText, callback in buttonSpecs:
            button = Button(text=buttonText)
            button.bind(on_release=self._wrap_button_callback(callback))
            rowLayout.add_widget(button)
        self.add_widget(rowLayout)

    def _update_status(self, message: str) -> None:
        """Display a status message in the footer label."""
        self.statusLabel.text = message

    def _play_async(self, fileName: str, build: Callable[[str], str], doneMessage: str) -> None:
        """Run synthesis on a worker thread and play the result once ready."""
        tempFilePath = os.path.join(tempfile.gettempdir(), fileName)

        def build_task() -> str:
            return build(tempFilePath)

        def on_success(path: str) -> None:
            self._play_file(path)
            self._update_status(doneMessage)

        self._run_async(build_task, on_success)

    def _run_async(self, work: Callable[[], str], onSuccessCallback: Callable[[str], None]) -> None:
        """Execute work on a background thread and call back on the main thread."""

        def task_runner() -> None:
            try:
                resultPath = work()
            except Exception as error:  # noqa: BLE001 - ユーザー通知のみ
                def handle_error(_dt, err=error):
                    self._update_status(f"Error: {err}")

                Clock.schedule_once(handle_error, 0)
                return

            Clock.schedule_once(lambda *_: onSuccessCallback(resultPath), 0)

        threading.Thread(target=task_runner, daemon=True).start()

    @staticmethod
    def _wrap_button_callback(callback: Callable[[], None]) -> Callable[..., None]:
        """Wrap callbacks so they ignore positional arguments from Kivy."""
        return lambda *_: callback()

    def _add_gojuon_row(self, entryList: Sequence[Tuple[str, Tuple[str, str]] | None]) -> None:
        """Render one row of the gojuon button table."""
        rowLayout = BoxLayout(size_hint_y=None, height="48dp", spacing=8)
        for cellEntry in entryList:
            if cellEntry is None:
                rowLayout.add_widget(Label(text=""))
                continue

            buttonText, cellPayload = cellEntry
            button = Button(text=buttonText)

            if cellPayload[0] == "nasal":
                button.bind(on_release=self._wrap_button_callback(lambda consonant=cellPayload[1]: self.speak_nasal(consonant)))
            else:
                button.bind(
                    on_release=self._wrap_button_callback(
                        lambda consonant=cellPayload[0], vowel=cellPayload[1]: self.speak_cv(consonant, vowel)
                    )
                )

            rowLayout.add_widget(button)

        self.add_widget(rowLayout)


class AppVocal(App):
    """Kivy application object for the voice synthesizer."""

    def build(self) -> Root:  # type: ignore[override]
        """Create the root widget for the application."""
        return Root()


if __name__ == "__main__":
    AppVocal().run()
