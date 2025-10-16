# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.core.audio import SoundLoader
from kivy.clock import Clock

import threading, tempfile, os

from dsp import synth_vowel, synth_phrase_to_wav, synth_cv_to_wav

FS = 22050  # サンプリング周波数（統一）


class Root(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=8, padding=12, **kwargs)

        # ---- F0 ----
        row = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        row.add_widget(Label(text='F0(Hz):', size_hint_x=None, width='80dp'))
        self.pitch = Slider(min=80, max=260, value=120)
        row.add_widget(self.pitch)
        self.add_widget(row)

        # ---- Overlap(ms) ----
        row_ov = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        self.lbl_ov = Label(text='Overlap(ms): 10', size_hint_x=None, width='120dp')
        self.overlap = Slider(min=6, max=30, value=10)
        self.overlap.bind(value=lambda _, v: setattr(self.lbl_ov, 'text', f'Overlap(ms): {int(v)}'))
        row_ov.add_widget(self.lbl_ov)
        row_ov.add_widget(self.overlap)
        self.add_widget(row_ov)

        # ---- Vowel(ms) ----
        row_vm = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        self.lbl_vm = Label(text='Vowel(ms): 260', size_hint_x=None, width='120dp')
        self.vowel_ms = Slider(min=180, max=320, value=260)
        self.vowel_ms.bind(value=lambda _, v: setattr(self.lbl_vm, 'text', f'Vowel(ms): {int(v)}'))
        row_vm.add_widget(self.lbl_vm)
        row_vm.add_widget(self.vowel_ms)
        self.add_widget(row_vm)

        # ---- Onset（t/k 強調） ----
        row_on = BoxLayout(size_hint_y=None, height='40dp', spacing=8)
        row_on.add_widget(Label(text='Onset (t/k 強調):', size_hint_x=None, width='140dp'))
        self.onset = CheckBox(active=True)  # デフォルトONで差が出やすい
        row_on.add_widget(self.onset)
        self.add_widget(row_on)

        # ---- 簡易ボタン（母音）----
        row2 = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        for v in ['a', 'i', 'u', 'e', 'o']:
            btn = Button(text=v.upper())
            btn.bind(on_release=lambda _, vv=v: self.speak_vowel(vv))
            row2.add_widget(btn)
        self.add_widget(row2)

        # ---- 5母音並び 再生 + ステータス ----
        row3 = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        btn_seq = Button(text='Play A-I-U-E-O')
        btn_seq.bind(on_release=lambda _: self.play_sequence())
        row3.add_widget(btn_seq)

        self.status = Label(text='Ready')
        row3.add_widget(self.status)
        self.add_widget(row3)

        # ---- 子音 + 母音（SA / TA / KA） ----
        row4 = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        for lab in [('SA', ('s', 'a')), ('TA', ('t', 'a')), ('KA', ('k', 'a'))]:
            btn = Button(text=lab[0])
            c, v = lab[1]
            btn.bind(on_release=lambda _, cc=c, vv=v: self.speak_cv(cc, vv))
            row4.add_widget(btn)
        self.add_widget(row4)

        row5 = BoxLayout(size_hint_y=None, height='48dp', spacing=8)
        for lab in [('NA', ('n','a')), ('NI', ('n','i')), ('NU', ('n','u')),
                    ('NE', ('n','e')), ('NO', ('n','o')), ('WA', ('w','a'))]:
            btn = Button(text=lab[0])
            c, v = lab[1]
            btn.bind(on_release=lambda _, cc=c, vv=v: self.speak_cv(cc, vv))
            row5.add_widget(btn)
        self.add_widget(row5)

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
        self.status.text = f'Synth {v}...'
        f0 = float(self.pitch.value)

        def task():
            y = synth_vowel(vowel=v, f0=f0, dur_s=0.6, fs=FS)
            tmp = os.path.join(tempfile.gettempdir(), f'vowel_{v}.wav')
            from dsp import write_wav
            path = write_wav(tmp, y, fs=FS)
            Clock.schedule_once(lambda *_: self._play_file(path), 0)
            Clock.schedule_once(lambda *_: setattr(self.status, 'text', f'Played {v}'), 0)

        threading.Thread(target=task, daemon=True).start()

    def speak_cv(self, c, v):
        self.status.text = f'Synth {c.upper()}-{v.upper()}...'
        f0 = float(self.pitch.value)
        ov = int(self.overlap.value)
        vms = int(self.vowel_ms.value)
        use_onset = bool(self.onset.active)

        def task():
            tmp = os.path.join(tempfile.gettempdir(), f'{c}{v}.wav')
            path = synth_cv_to_wav(
                c, v, tmp,
                f0=f0, fs=FS,
                pre_ms=0, cons_ms=None,
                vowel_ms=vms, overlap_ms=ov,
                use_onset_transition=use_onset  # ★ t/k の違いを強調
            )
            Clock.schedule_once(lambda *_: self._play_file(path), 0)
            Clock.schedule_once(lambda *_: setattr(self.status, 'text', f'Played {c.upper()}-{v.upper()}'), 0)

        threading.Thread(target=task, daemon=True).start()

    def play_sequence(self):
        self.status.text = 'Synth sequence...'
        f0 = float(self.pitch.value)

        def task():
            tmp = os.path.join(tempfile.gettempdir(), 'sequence_aiueo.wav')
            path = synth_phrase_to_wav(['a', 'i', 'u', 'e', 'o'], tmp, f0=f0, unit_ms=250, gap_ms=60, fs=FS)
            Clock.schedule_once(lambda *_: self._play_file(path), 0)
            Clock.schedule_once(lambda *_: setattr(self.status, 'text', 'Played A-I-U-E-O'), 0)

        threading.Thread(target=task, daemon=True).start()


class AppVocal(App):
    def build(self):
        return Root()


if __name__ == '__main__':
    AppVocal().run()
