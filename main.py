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

import numpy as np

from dsp import synth_cv, synth_cv_to_wav, synth_nasal, synth_phrase_to_wav, synth_vowel, write_wav

FS = 22050  # サンプリング周波数（統一）

GOJUON_ROWS: Sequence[Sequence[Tuple[str, Tuple[str, str]] | None]] = [
    [
        ('KA', ('k', 'a')),
        ('KI', ('k', 'i')),
        ('KU', ('k', 'u')),
        ('KE', ('k', 'e')),
        ('KO', ('k', 'o')),
    ],
    [
        ('SA', ('s', 'a')),
        ('SHI', ('sh', 'i')),
        ('SU', ('s', 'u')),
        ('SE', ('s', 'e')),
        ('SO', ('s', 'o')),
    ],
    [
        ('TA', ('t', 'a')),
        ('CHI', ('ch', 'i')),
        ('TSU', ('ts', 'u')),
        ('TE', ('t', 'e')),
        ('TO', ('t', 'o')),
    ],
    [
        ('NA', ('n', 'a')),
        ('NI', ('n', 'i')),
        ('NU', ('n', 'u')),
        ('NE', ('n', 'e')),
        ('NO', ('n', 'o')),
    ],
    [
        ('HA', ('h', 'a')),
        ('HI', ('h', 'i')),
        ('FU', ('f', 'u')),
        ('HE', ('h', 'e')),
        ('HO', ('h', 'o')),
    ],
    [
        ('MA', ('m', 'a')),
        ('MI', ('m', 'i')),
        ('MU', ('m', 'u')),
        ('ME', ('m', 'e')),
        ('MO', ('m', 'o')),
    ],
    [
        ('YA', ('y', 'a')),
        None,
        ('YU', ('y', 'u')),
        None,
        ('YO', ('y', 'o')),
    ],
    [
        ('RA', ('r', 'a')),
        ('RI', ('r', 'i')),
        ('RU', ('r', 'u')),
        ('RE', ('r', 'e')),
        ('RO', ('r', 'o')),
    ],
    [
        ('WA', ('w', 'a')),
        None,
        None,
        None,
        ('WO', ('w', 'o')),
    ],
    [
        ('N', ('nasal', 'n')),
        None,
        None,
        None,
        None,
    ],
]


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
            min_value=120,
            max_value=300,
            value=200,
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
        
        # ---- 50音テーブル ----
        for entries in GOJUON_ROWS:
            self._add_gojuon_row(entries)

        # ---- 5母音並び 再生 + ステータス ----
        row3 = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        btn_seq = Button(text='Play A-I-U-E-O')
        btn_seq.bind(on_release=lambda *_: self.play_sequence())
        row3.add_widget(btn_seq)

        btn_love = Button(text='Say あいしてる')
        btn_love.bind(on_release=lambda *_: self.speak_aishiteru())
        row3.add_widget(btn_love)

        self.status = Label(text='Ready')
        row3.add_widget(self.status)
        self.add_widget(row3)



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
        dur_s = max(0.12, float(self.vowel_ms.value) / 1000.0)

        def build(path: str) -> str:
            y = synth_vowel(vowel=v, f0=f0, dur_s=dur_s, fs=FS)
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

    def speak_nasal(self, consonant: str):
        consonant = consonant.lower()
        label = 'N' if consonant == 'n' else consonant.upper()
        self._set_status(f'Synth {label}...')
        f0 = float(self.pitch.value)
        dur = max(80, int(self.vowel_ms.value * 0.6))

        def build(path: str) -> str:
            y = synth_nasal(consonant, f0=f0, dur_ms=dur, fs=FS)
            return write_wav(path, y, fs=FS)

        self._play_async(f'nasal_{consonant}.wav', build, f'Played {label}')

    def speak_aishiteru(self):
        self._set_status('Synth あいしてる...')
        f0 = float(self.pitch.value)
        ov = int(self.overlap.value)
        vms = int(self.vowel_ms.value)
        use_onset = bool(self.onset.active)

        def build(path: str) -> str:
            def pause(ms: float) -> None:
                samples = int(FS * (ms / 1000.0))
                if samples > 0:
                    segments.append(np.zeros(samples, dtype=np.float32))

            vowel_sec = max(0.12, vms / 1000.0)
            segments = [
                synth_vowel('a', f0=f0, dur_s=vowel_sec, fs=FS),
            ]
            pause(45.0)
            segments.append(synth_vowel('i', f0=f0, dur_s=max(0.1, vowel_sec * 0.8), fs=FS))
            pause(70.0)

            cv_kwargs = dict(f0=f0, fs=FS, vowel_ms=vms, overlap_ms=ov, use_onset_transition=use_onset)
            segments.append(synth_cv('sh', 'i', **cv_kwargs))
            pause(50.0)
            segments.append(synth_cv('t', 'e', **cv_kwargs))
            pause(35.0)
            ru_vowel = vms + max(40, vms // 3)
            segments.append(
                synth_cv('r', 'u', f0=f0, fs=FS, vowel_ms=ru_vowel, overlap_ms=ov, use_onset_transition=use_onset)
            )

            audio = np.concatenate(segments).astype(np.float32, copy=False)
            return write_wav(path, audio, fs=FS)

        self._play_async('phrase_aishiteru.wav', build, 'Played あいしてる')

    def play_sequence(self):
        self._set_status('Synth sequence...')
        f0 = float(self.pitch.value)

        def build(path: str) -> str:
            vowels: Sequence[str] = ['a', 'i', 'u', 'e', 'o']
            unit_ms = max(120, int(self.vowel_ms.value))
            gap_ms = max(40, int(unit_ms * 0.25))
            return synth_phrase_to_wav(vowels, path, f0=f0, unit_ms=unit_ms, gap_ms=gap_ms, fs=FS)

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
                def handle_error(_dt, err=exc):
                    self._set_status(f'Error: {err}')

                Clock.schedule_once(handle_error, 0)
                return

            Clock.schedule_once(lambda *_: on_success(result), 0)

        threading.Thread(target=task, daemon=True).start()

    @staticmethod
    def _wrap_button_callback(callback: Callable[[], None]) -> Callable[..., None]:
        return lambda *_: callback()

    def _add_gojuon_row(self, entries: Sequence[Tuple[str, Tuple[str, str]] | None]) -> None:
        row = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        for cell in entries:
            if cell is None:
                row.add_widget(Label(text=''))
                continue

            text, payload = cell
            btn = Button(text=text)

            if payload[0] == 'nasal':
                btn.bind(on_release=self._wrap_button_callback(
                    lambda cc=payload[1]: self.speak_nasal(cc)
                ))
            else:
                btn.bind(on_release=self._wrap_button_callback(
                    lambda cc=payload[0], vv=payload[1]: self.speak_cv(cc, vv)
                ))

            row.add_widget(btn)

        self.add_widget(row)


class AppVocal(App):
    def build(self):
        return Root()


if __name__ == '__main__':
    AppVocal().run()
