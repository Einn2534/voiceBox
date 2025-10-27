# GIAN Voice Box

## Requirements
- Python 3.10 or newer
- Virtual environment tool such as `venv`
- System packages required by [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/#downloads) (PortAudio)
- An environment variable named `GEMINI_API_KEY` that contains a valid Gemini Live API key

Install the Python dependencies before running the application:

```bash
python -m venv .venv
source .venv/bin/activate
pip install kivy google-genai numpy pyaudio
```

## Running the application
1. Activate the virtual environment (if not already active).
2. Export the Gemini API key so the voice test screen can authenticate:
   ```bash
   export GEMINI_API_KEY="<your_api_key>"
   ```
3. Start the Kivy application:
   ```bash
   python Main.py
   ```

The home screen loads first. Navigate to the voice test screen to begin an interactive Gemini Live session with synthesized audio playback.
