# GIAN Voice Box

## 必要要件
- Python 3.10 以上
- `venv` などの仮想環境ツール
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/#downloads) が依存する PortAudio 等のシステムパッケージ

## セットアップ
1. プロジェクトルートで仮想環境を作成します。
   ```bash
   python -m venv .venv
   ```
2. 仮想環境を有効化します。
   - macOS / Linux
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell)
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows (コマンドプロンプト)
     ```bat
     .\.venv\Scripts\activate.bat
     ```
3. 依存パッケージをインストールします。
   ```bash
   pip install kivy google-genai numpy pyaudio
   ```

## 起動手順
1. 仮想環境を有効化します（未実行の場合）。
2. Kivy アプリケーションを起動します。
   ```bash
   python Main.py
   ```

Gemini API キーはプロジェクト内に直接定義されています。別途環境変数を設定する必要はありませんが、運用環境で使用する際はキーの保護方法を再検討してください。

起動後はホーム画面が表示されます。音声テスト画面へ遷移すると Gemini Live と音声合成による対話を開始できます。
