# main.py
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

from dsp import synth_cv_to_wav, synth_phrase_to_wav, synth_vowel, write_wav

FS = 22050  # サンプリング周波数（統一）


class Root(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=8, padding=12, **kwargs)

        # ---- Sliders ----
        self.pitch, _ = self._add_slider(
            label_template='F0(Hz):',
            min_value=80,
            max_value=260,
            value=120,
            width='80dp',
            show_value=False,
        )
        self.overlap, self.lbl_ov = self._add_slider(
            label_template='Overlap(ms): {value}',
            min_value=6,
            max_value=30,
            value=10,
        )
        self.vowel_ms, self.lbl_vm = self._add_slider(
            label_template='Vowel(ms): {value}',
            min_value=180,
            max_value=320,
            value=260,
        )

        # ---- Onset（t/k 強調） ----
        row_on = BoxLayout(size_hint_y=None, height='40dp', spacing=8)
        row_on.add_widget(Label(text='Onset (t/k 強調):', size_hint_x=None, width='140dp'))
        self.onset = CheckBox(active=True)  # デフォルトONで差が出やすい
        row_on.add_widget(self.onset)
        self.add_widget(row_on)

        # ---- 簡易ボタン（母音）----
        vowel_specs = [(v.upper(), lambda vv=v: self.speak_vowel(vv)) for v in ['a', 'i', 'u', 'e', 'o']]
        self._add_button_row(vowel_specs)

        # ---- 5母音並び 再生 + ステータス ----
        row3 = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        btn_seq = Button(text='Play A-I-U-E-O')
        btn_seq.bind(on_release=lambda *_: self.play_sequence())
        row3.add_widget(btn_seq)

        self.status = Label(text='Ready')
        row3.add_widget(self.status)
        self.add_widget(row3)

        # ---- 子音 + 母音（SA / TA / KA） ----
        cons_primary = [('SA', ('s', 'a')), ('TA', ('t', 'a')), ('KA', ('k', 'a'))]
        cons_extra = [('NA', ('n', 'a')), ('NI', ('n', 'i')), ('NU', ('n', 'u')),
                      ('NE', ('n', 'e')), ('NO', ('n', 'o')), ('WA', ('w', 'a'))]
        self._add_button_row(self._build_cv_specs(cons_primary))
        self._add_button_row(self._build_cv_specs(cons_extra))

        self.sound = None

    # ---------- audio I/O ----------
    def _play_file(self, path):
        if self.sound:
            try:
                self.sound.stop()
            except Exception:
                pass
        self.sound = SoundLoader.load(path)
        if self.sound:
            self.sound.play()
            self.status.text = f'Playing: {os.path.basename(path)}'
        else:
            self.status.text = 'Failed to load sound'

    # ---------- actions ----------
    def speak_vowel(self, v):
        v = v.lower()
        self._set_status(f'Synth {v}...')
        f0 = float(self.pitch.value)

        def build(path: str) -> str:
            y = synth_vowel(vowel=v, f0=f0, dur_s=0.6, fs=FS)
            return write_wav(path, y, fs=FS)

        self._play_async(f'vowel_{v}.wav', build, f'Played {v.upper()}')

    def speak_cv(self, c, v):
        c = c.lower()
        v = v.lower()
        label = f'{c.upper()}-{v.upper()}'
        self._set_status(f'Synth {label}...')
        f0 = float(self.pitch.value)
        ov = int(self.overlap.value)
        vms = int(self.vowel_ms.value)
        use_onset = bool(self.onset.active)

        def build(path: str) -> str:
            return synth_cv_to_wav(
                c,
                v,
                path,
                f0=f0,
                fs=FS,
                pre_ms=0,
                cons_ms=None,
                vowel_ms=vms,
                overlap_ms=ov,
                use_onset_transition=use_onset,
            )

        self._play_async(f'{c}{v}.wav', build, f'Played {label}')

    def play_sequence(self):
        self._set_status('Synth sequence...')
        f0 = float(self.pitch.value)

        def build(path: str) -> str:
            vowels: Sequence[str] = ['a', 'i', 'u', 'e', 'o']
            return synth_phrase_to_wav(vowels, path, f0=f0, unit_ms=250, gap_ms=60, fs=FS)

        self._play_async('sequence_aiueo.wav', build, 'Played A-I-U-E-O')

    # ---------- helpers ----------
    def _add_slider(
        self,
        *,
        label_template: str,
        min_value: float,
        max_value: float,
        value: float,
        width: str = '120dp',
        show_value: bool = True,
    ) -> Tuple[Slider, Label]:
        row = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        label = Label(size_hint_x=None, width=width)

        slider = Slider(min=min_value, max=max_value, value=value)

        if show_value and '{' in label_template:
            label.text = label_template.format(value=int(value))

            def update_label(_, val):
                label.text = label_template.format(value=int(val))

            slider.bind(value=update_label)
        else:
            label.text = label_template

        row.add_widget(label)
        row.add_widget(slider)
        self.add_widget(row)
        return slider, label

    def _add_button_row(self, specs: Iterable[Tuple[str, Callable[[], None]]], height: str = '48dp') -> None:
        row = BoxLayout(size_hint_y=None, height=height, spacing=8)
        for text, callback in specs:
            btn = Button(text=text)
            btn.bind(on_release=self._wrap_button_callback(callback))
            row.add_widget(btn)
        self.add_widget(row)

    def _set_status(self, message: str) -> None:
        self.status.text = message

    def _play_async(self, filename: str, build: Callable[[str], str], done_message: str) -> None:
        tmp = os.path.join(tempfile.gettempdir(), filename)

        def work() -> str:
            return build(tmp)

        def on_success(path: str) -> None:
            self._play_file(path)
            self._set_status(done_message)

        self._run_async(work, on_success)

    def _run_async(self, work: Callable[[], str], on_success: Callable[[str], None]) -> None:
        def task():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - ユーザー通知のみ
                Clock.schedule_once(lambda *_: self._set_status(f'Error: {exc}'), 0)
                return

            Clock.schedule_once(lambda *_: on_success(result), 0)

        threading.Thread(target=task, daemon=True).start()

    def _build_cv_specs(self, specs: Iterable[Tuple[str, Tuple[str, str]]]) -> Sequence[Tuple[str, Callable[[], None]]]:
        return [
            (label, lambda cc=c, vv=v: self.speak_cv(cc, vv))
            for label, (c, v) in specs
        ]

    @staticmethod
    def _wrap_button_callback(callback: Callable[[], None]) -> Callable[..., None]:
        return lambda *_: callback()


class AppVocal(App):
    def build(self):
        return Root()


if __name__ == '__main__':
    AppVocal().run()
