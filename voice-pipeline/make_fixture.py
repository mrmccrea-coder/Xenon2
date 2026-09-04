"""make_fixture.py

Generates the sample WAV fixtures used by cli_harness.py's --wav path, so the full voice
pipeline can be exercised end-to-end in this environment without a human available to
speak into a live microphone (see phase2_prompt.md's testing note):

- fixtures/sample_greeting.wav: uses piper-tts itself (already needed for the TTS stage)
  to synthesize a short greeting -- the same "hello, how are you?" phrase used throughout
  Phase 1's own acceptance testing -- as a stand-in for a recorded human voice sample.
- fixtures/silence.wav: 3 seconds of digital silence, for exercising the "no speech
  detected" graceful-recovery path (see README's "Graceful no-speech handling" section).

Run once (or whenever the fixtures need regenerating):
    python make_fixture.py
"""
import os
import wave

import numpy as np
import soundfile as sf
from piper import PiperVoice

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL = os.path.join(_THIS_DIR, "models", "en_GB-alan-medium.onnx")
VOICE_CONFIG = os.path.join(_THIS_DIR, "models", "en_GB-alan-medium.onnx.json")
FIXTURES_DIR = os.path.join(_THIS_DIR, "fixtures")

GREETING_TEXT = "Hello, how are you?"
SILENCE_SAMPLE_RATE = 16000
SILENCE_SECONDS = 3.0


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    greeting_path = os.path.join(FIXTURES_DIR, "sample_greeting.wav")
    voice = PiperVoice.load(VOICE_MODEL, config_path=VOICE_CONFIG)
    with wave.open(greeting_path, "wb") as wf:
        voice.synthesize_wav(GREETING_TEXT, wf)
    print(f"Wrote {greeting_path!r} synthesizing: {GREETING_TEXT!r}")

    silence_path = os.path.join(FIXTURES_DIR, "silence.wav")
    silence = np.zeros(int(SILENCE_SAMPLE_RATE * SILENCE_SECONDS), dtype=np.float32)
    sf.write(silence_path, silence, SILENCE_SAMPLE_RATE)
    print(f"Wrote {silence_path!r}: {SILENCE_SECONDS}s of digital silence")


if __name__ == "__main__":
    main()
