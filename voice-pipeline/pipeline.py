"""pipeline.py

Wires the full Phase 2 voice chain together:

    mic or WAV file
        -> VAD (CPU, silero-vad)         gates whether/where speech is
        -> STT (GPU, faster-whisper)      speech -> text
        -> XenonEngine.generate() (GPU)   text -> streamed response text (Phase 1)
        -> TTS (CPU, piper-tts)           text -> speech
        -> speakers

Each stage's wall-clock time and actual execution device are recorded into a
PipelineTiming, which cli_harness.py prints. No stage here is a UI concern -- this
module is meant to be called by a CLI test harness or, later, wired into Phase 3's
mic-toggle button; it does not know about any UI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from audio_io import load_wav_16k_mono, play_audio, record_from_mic, SAMPLE_RATE
from vad import VoiceActivityDetector
from stt import SpeechToText
from tts import TextToSpeech
from xenon_engine import XenonEngine
from speech_streamer import IncrementalSpeaker


@dataclass
class StageTiming:
    name: str
    device: str
    seconds: float
    detail: str = ""


@dataclass
class PipelineResult:
    ok: bool
    reason: str  # "ok" | "no_speech" | error description
    transcript: str = ""
    response_text: str = ""
    stages: List[StageTiming] = field(default_factory=list)
    total_sec: float = 0.0  # time to first spoken audio (the "round trip" latency metric)
    full_playback_sec: Optional[float] = None  # time until all response audio finished playing

    def add(self, name: str, device: str, seconds: float, detail: str = ""):
        self.stages.append(StageTiming(name, device, seconds, detail))


def run_once_from_audio(
    audio: np.ndarray,
    vad: VoiceActivityDetector,
    stt: SpeechToText,
    engine: XenonEngine,
    tts: TextToSpeech,
    speak: bool = True,
    max_response_tokens: int = 60,
) -> PipelineResult:
    """Runs one full round-trip starting from already-captured 16kHz mono audio (either
    a loaded WAV fixture or a live mic recording -- audio_io.py normalizes both to the
    same format so this function doesn't care which one it came from)."""
    result = PipelineResult(ok=False, reason="")
    t_pipeline_start = time.perf_counter()

    # --- VAD gate --------------------------------------------------------------
    vad_result = vad.detect(audio, sample_rate=SAMPLE_RATE)
    result.add("VAD", vad_result.device, vad_result.detect_sec,
               detail=f"segments={vad_result.segments}")

    if not vad_result.speech_detected:
        result.reason = "no_speech"
        result.total_sec = time.perf_counter() - t_pipeline_start
        return result

    # Trim to the detected speech span (with a little padding) before handing to STT --
    # avoids wasting GPU time transcribing leading/trailing silence.
    first_start = vad_result.segments[0]["start"]
    last_end = vad_result.segments[-1]["end"]
    pad = 0.2
    start_sample = max(0, int((first_start - pad) * SAMPLE_RATE))
    end_sample = min(len(audio), int((last_end + pad) * SAMPLE_RATE))
    speech_audio = audio[start_sample:end_sample]

    # --- STT (GPU) ---------------------------------------------------------------
    stt_result = stt.transcribe(speech_audio, sample_rate=SAMPLE_RATE)
    result.add("STT", stt_result.actual_device, stt_result.transcribe_sec,
               detail=f"requested={stt_result.device} lang={stt_result.language}"
                      f"({stt_result.language_probability:.2f})")
    result.transcript = stt_result.text

    if not stt_result.text.strip():
        result.reason = "no_speech"  # VAD found "speech" but STT decoded nothing usable
        result.total_sec = time.perf_counter() - t_pipeline_start
        return result

    # --- generate() (GPU, Phase 1) streamed straight into incremental TTS (CPU) --------
    # Per phase2_prompt.md: do not wait for a full batched response before starting TTS.
    # IncrementalSpeaker.feed() is called per streamed text fragment and synthesizes+
    # queues each completed sentence as soon as it's available, while generation for the
    # rest of the response continues -- see speech_streamer.py for the full rationale.
    speaker = IncrementalSpeaker(tts, speak=speak)
    gen_result = engine.generate(
        stt_result.text, max_tokens=max_response_tokens, on_partial=speaker.feed
    )
    speech_result = speaker.finish(wait_for_playback=speak)

    gpu_label = "cuda" if gen_result.gpu_layers > 0 and gen_result.has_gpu_support else "cpu"
    result.add("generate", gpu_label, gen_result.total_sec,
               detail=f"tokens={gen_result.tokens_emitted} ttft={gen_result.ttft_sec:.3f}s "
                      f"sentences={len(speech_result.chunks)}")
    result.response_text = gen_result.text

    ttfa = speech_result.time_to_first_audio_sec
    result.add("TTS (first sentence)", tts.device,
               speech_result.chunks[0].synth_sec if speech_result.chunks else 0.0,
               detail=f"time_to_first_audio={ttfa:.3f}s" if ttfa is not None else "no audio synthesized")
    if len(speech_result.chunks) > 1:
        result.add("TTS (remaining sentences)", tts.device,
                   speech_result.total_tts_synth_sec - (speech_result.chunks[0].synth_sec if speech_result.chunks else 0.0),
                   detail=f"{len(speech_result.chunks) - 1} more sentence(s), synthesized while "
                          f"earlier audio played / generation continued")

    result.ok = True
    result.reason = "ok"
    # "Round trip complete" = time to first spoken audio, not time until all speech has
    # finished playing out loud -- a multi-sentence response's total spoken duration is a
    # property of the response length, not of pipeline latency (see README for the
    # measured distinction and why the <2s acceptance criterion is evaluated against this).
    # first_audio_abs_time is recorded by the playback worker regardless of whether audio
    # was actually sent to the speakers (speak=False just skips the sd.play() call) -- so
    # this stays "time to first response chunk ready" even in --no-speak/debug mode.
    if speech_result.first_audio_abs_time is not None:
        result.total_sec = speech_result.first_audio_abs_time - t_pipeline_start
    else:
        result.total_sec = time.perf_counter() - t_pipeline_start
    if speech_result.all_playback_done_abs_time is not None:
        result.full_playback_sec = speech_result.all_playback_done_abs_time - t_pipeline_start
    return result


def run_once_from_wav(
    wav_path: str,
    vad: VoiceActivityDetector,
    stt: SpeechToText,
    engine: XenonEngine,
    tts: TextToSpeech,
    speak: bool = True,
    max_response_tokens: int = 60,
) -> PipelineResult:
    audio = load_wav_16k_mono(wav_path)
    return run_once_from_audio(audio, vad, stt, engine, tts, speak=speak,
                                max_response_tokens=max_response_tokens)


def run_once_from_mic(
    vad: VoiceActivityDetector,
    stt: SpeechToText,
    engine: XenonEngine,
    tts: TextToSpeech,
    speak: bool = True,
    max_response_tokens: int = 60,
    max_listen_sec: float = 6.0,
    max_utterance_sec: float = 10.0,
) -> PipelineResult:
    """Live mic path: mic -> streaming VAD gate (start/end detection) -> STT -> generate()
    -> TTS -> speakers. The VAD gating here happens twice by design and that's
    intentional, not redundant: record_from_mic() uses streaming VADIterator to decide
    *when to stop recording* (and bails out with speech_detected=False rather than
    hanging if nobody speaks within max_listen_sec); run_once_from_audio() then re-runs
    batch VAD (get_speech_timestamps) on the captured clip to trim silence before STT,
    the same as the WAV-fixture path -- keeping one shared code path for STT onward
    regardless of input source."""
    mic_result = record_from_mic(vad, max_listen_sec=max_listen_sec, max_utterance_sec=max_utterance_sec)

    result = PipelineResult(ok=False, reason="")
    result.add("VAD (mic listen)", "cpu", mic_result.listen_sec, detail=f"reason={mic_result.reason}")

    if not mic_result.speech_detected or mic_result.audio is None or mic_result.audio.size == 0:
        result.reason = "no_speech"
        result.total_sec = mic_result.listen_sec
        return result

    inner = run_once_from_audio(mic_result.audio, vad, stt, engine, tts, speak=speak,
                                 max_response_tokens=max_response_tokens)
    # Merge: keep the mic-listen stage first, then the rest of the pipeline's stages.
    result.stages.extend(inner.stages)
    result.ok = inner.ok
    result.reason = inner.reason
    result.transcript = inner.transcript
    result.response_text = inner.response_text
    result.total_sec = mic_result.listen_sec + inner.total_sec
    return result
