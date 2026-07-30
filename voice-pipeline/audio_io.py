"""audio_io.py

Audio input/output helpers: loading a WAV file (resampled to 16kHz mono, whatever its
source rate), live microphone capture gated by streaming VAD, and playback through
speakers. Kept separate from vad.py/stt.py/tts.py so pipeline.py can swap the input
source (mic vs. a pre-recorded fixture WAV) without touching the VAD/STT/TTS modules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as sps
import soundfile as sf
import sounddevice as sd

from vad import SAMPLE_RATE, VoiceActivityDetector
from silero_vad import VADIterator

FRAME_SAMPLES = 512  # silero-vad's recommended chunk size at 16kHz


def load_wav_16k_mono(path: str) -> np.ndarray:
    """Loads any WAV file and resamples/downmixes to float32 mono @ 16kHz, matching what
    a live mic capture in this pipeline produces -- so a pre-recorded fixture WAV and a
    live mic recording are interchangeable inputs to the rest of the pipeline."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n_samples = int(round(len(data) * SAMPLE_RATE / sr))
        data = sps.resample(data, n_samples).astype(np.float32)
    return data


def play_audio(audio: np.ndarray, sample_rate: int, block: bool = True) -> None:
    """Plays float32 mono audio through the default speaker output device."""
    if audio.size == 0:
        return
    sd.play(audio, sample_rate)
    if block:
        sd.wait()


@dataclass
class MicCaptureResult:
    audio: Optional[np.ndarray]  # None if no speech was ever detected
    speech_detected: bool
    listen_sec: float
    reason: str  # "speech_end" | "max_utterance" | "no_speech_timeout"


def record_from_mic(
    vad: VoiceActivityDetector,
    max_listen_sec: float = 6.0,
    max_utterance_sec: float = 10.0,
    silence_end_ms: int = 700,
    speech_threshold: float = 0.5,
    device: Optional[int] = None,
) -> MicCaptureResult:
    """Streams audio from the default microphone, frame-by-frame, through silero-vad's
    streaming VADIterator (CPU) to detect speech start/end in real time.

    - Waits up to `max_listen_sec` for speech to *start* at all. If nothing is detected
      in that window, returns immediately with speech_detected=False (reason=
      "no_speech_timeout") rather than blocking forever -- this is the mechanism that
      satisfies the "recovers gracefully if no speech is detected, does not hang"
      acceptance criterion for the live-mic path.
    - Once speech starts, keeps recording until `silence_end_ms` of continuous silence is
      observed (utterance end) or `max_utterance_sec` total speech duration is hit
      (safety cap against a stuck-open mic / non-stopping speaker).
    """
    vad_iterator = VADIterator(
        vad.model, threshold=speech_threshold, sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=silence_end_ms,
    )

    frames: list[np.ndarray] = []
    speech_started = False
    t_listen_start = time.perf_counter()

    frame_queue: list[np.ndarray] = []

    def _callback(indata, frame_count, time_info, status):
        frame_queue.append(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME_SAMPLES, device=device, callback=_callback,
        ):
            while True:
                now = time.perf_counter()
                elapsed = now - t_listen_start

                if not speech_started and elapsed > max_listen_sec:
                    return MicCaptureResult(
                        audio=None, speech_detected=False, listen_sec=elapsed,
                        reason="no_speech_timeout",
                    )
                if speech_started and (now - t_speech_start) > max_utterance_sec:
                    return MicCaptureResult(
                        audio=np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32),
                        speech_detected=True, listen_sec=elapsed, reason="max_utterance",
                    )

                if not frame_queue:
                    time.sleep(0.01)
                    continue

                chunk = frame_queue.pop(0)
                if speech_started:
                    frames.append(chunk)

                event = vad_iterator(chunk, return_seconds=True)
                if event is not None:
                    if "start" in event and not speech_started:
                        speech_started = True
                        t_speech_start = now
                        frames.append(chunk)  # include the frame that triggered onset
                    elif "end" in event and speech_started:
                        elapsed = time.perf_counter() - t_listen_start
                        return MicCaptureResult(
                            audio=np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32),
                            speech_detected=True, listen_sec=elapsed, reason="speech_end",
                        )
    finally:
        vad_iterator.reset_states()
