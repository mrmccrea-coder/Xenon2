# Xenon2 Voice Pipeline (Phase 2)

Speech I/O for Xenon2: mic (or a WAV file) -> VAD -> STT -> Phase 1's `generate()` ->
TTS -> speakers. No UI here -- see `../PLAN.md` for later phases. This phase proves the
speech -> text -> `generate()` -> text -> speech loop works standalone; it feeds
transcribed text into the exact same `generate()` entry point typed input would use
(see `xenon_engine.py`), it does not add a separate voice-only code path.

## Quick start

```powershell
cd voice-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# One-time: fetch a piper voice model (used for TTS)
python -m piper.download_voices --download-dir models en_US-lessac-medium

# One-time: generate the sample WAV fixture used for automated testing (see below)
python make_fixture.py

# Run the CLI test harness
python cli_harness.py --wav fixtures/sample_greeting.wav     # automated path, no mic needed
python cli_harness.py --mic                                   # live microphone path
```

Requires Phase 1's `inference-engine/build-cuda-app` to already be built (see
`../inference-engine/README.md`) and the RWKV model to already be downloaded/quantized
at `../models/rwkv-5-world-0.4B-Q4_0.bin` -- this phase does not rebuild or redownload
either, it links against what Phase 1 produced.

## Why a WAV-file input path, not just live mic

This dev environment has no human available to speak into a microphone during automated
testing. `cli_harness.py --wav <path>` exercises the full pipeline (VAD -> STT ->
generate() -> TTS -> playback) against a pre-recorded WAV instead of live audio, so the
round-trip can be measured and verified without a live speaker.
`fixtures/sample_greeting.wav` was generated with `make_fixture.py`, which uses piper-tts
itself (already needed for this phase's TTS stage) to synthesize the phrase "Hello, how
are you?" -- the same greeting Phase 1's own acceptance testing used. Live microphone
capture (`--mic`, the default when neither `--wav` nor `--mic` is passed) is still fully
implemented as the primary path, since that's what Phase 3's UI will actually call; it
was validated structurally in this environment (real `sounddevice` stream against the
machine's actual microphone input device, real frame-by-frame `silero-vad` streaming
detection) but a live "the assistant correctly transcribes something a human actually
said" run was not possible here -- there was no one to speak. What *was* verified live
is the no-speech-detected path (see below), since the room genuinely was silent.

## GPU/CPU device placement, and why

| Stage | Device | Why |
|---|---|---|
| VAD (silero-vad) | **CPU** | Cheap/lightweight by design; spec explicitly keeps it off the GPU. |
| STT (faster-whisper) | **GPU** | Spec requires this explicitly -- shares the GPU with RWKV inference. |
| generate() (RWKV, Phase 1) | **GPU** | Phase 1's existing cuBLAS offload (`--gpu-layers 24`). |
| TTS (piper) | **CPU** | No meaningful GPU acceleration path for piper; spec keeps it on CPU. |

This is not just trusted from the constructor arguments -- each stage's *actual* device
is verified at runtime, not just requested:

- **VAD**: the `torch` installed in this venv (`torch==2.13.0+cpu`, see
  `requirements.txt`) is the CPU-only build -- `torch.cuda.is_available()` returns
  `False`, so there is no GPU path silero-vad could fall onto even if asked. Confirmed via
  `VoiceActivityDetector.device == "cpu"`.
- **STT**: `SpeechToText` checks `model.model.device` (CTranslate2's own report of where
  it loaded) immediately after construction, *and* additionally forces a real warm-up
  transcription inside `__init__` before returning. This mattered in practice: CTranslate2
  resolves its CUDA library dependencies lazily, on the *first inference call*, not at
  model-load time -- construction can report `actual_device="cuda"` and still be about to
  fail on the first real `transcribe()` call. See "CTranslate2 CUDA compatibility issue"
  below for the specific failure this surfaced and how it was fixed. `cli_harness.py`
  prints the confirmed `actual_device` for every run.
- **TTS**: piper's `PiperVoice.load(..., use_cuda=False)` (the default, never overridden
  here) runs its ONNX Runtime session on CPU; `TextToSpeech.device == "cpu"`.
- **generate()**: `xenon_has_gpu_support()` and the loaded model's `n_gpu_layers` are
  both reported (Phase 1's own mechanism, reused as-is).

## CTranslate2 CUDA compatibility issue (found and fixed, not worked around)

Discovered while wiring this up: CTranslate2 4.8.1's Windows wheel (faster-whisper's
backend) dynamically links against **CUDA 12's `cublas64_12.dll`** specifically. This
dev machine's CUDA *Toolkit* is 13.3 (installed for Phase 1's rwkv.cpp CUDA build, which
uses `cublas64_13.dll`) -- CUDA 13 doesn't ship a `cublas64_12.dll`, so a bare
`device="cuda"` failed at first-inference time with:

```
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
```

Fix: installed the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` redistributable PyPI
packages (they vendor the actual CUDA 12 DLLs, the same files NVIDIA's own CUDA 12.x
installer would place on disk) and add their bundled `bin/` directories to the process's
DLL search path (`stt.py`'s `_ensure_cuda12_dlls()`, called before `ctranslate2` is
imported). This does **not** touch, downgrade, or conflict with the system CUDA 13.3
Toolkit Phase 1's `rwkv.dll` uses -- both CUDA runtimes coexist independently in the same
process, each DLL resolving its own differently-named cublas library. The spec was
explicit that if the GPU story got complicated, the fix should not be "move STT to CPU" --
this is that: STT still runs on GPU, the actual blocker (a DLL version mismatch, not a
capacity/VRAM problem) got fixed at its root instead.

## VRAM budget

Measured via `nvidia-smi` with **both** models resident simultaneously in the same
process (matching how `cli_harness.py` actually runs them, not separate processes):

| Whisper model | Whisper alone | + RWKV (24 layers) resident | % of 4096 MiB budget |
|---|---|---|---|
| `base` | 304 MiB (delta) | **529 MiB peak** | 12.9% |
| `small` | 720 MiB (delta) | **945 MiB peak** | 23.1% |

Both sizes fit comfortably -- nowhere close to the 4GB ceiling, let alone contending
with it. `cli_harness.py` defaults to **`small`** (`--whisper-model` can override this to
`tiny`/`base`) since there's ample headroom and `small` gives meaningfully better
transcription accuracy than `base` for the same VRAM-budget outcome (both "fit," so the
tie-break went to quality). `--whisper-model large` was not attempted -- the spec called
it out as unlikely to fit, and there's no need to since `small` already leaves >75% of
the budget free.

Baseline (nothing loaded): 11 MiB used / 4096 MiB total (idle desktop compositor use).

## Round-trip latency and per-stage timing

**Acceptance criterion**: "a full voice round-trip for a short greeting completes in
under 2 seconds." This is measured as *time from receiving the utterance to the first
spoken audio starting to play* -- not time until the entire (possibly multi-sentence)
response has finished being read aloud out loud. That distinction matters and is
deliberate, not a way to dodge the bar: see "Incremental TTS" below for why a multi-
sentence response's total spoken duration is a property of response length, not of
pipeline latency, and isn't the number the <2s criterion is checking.

Three consecutive runs, `--wav fixtures/sample_greeting.wav` ("Hello, how are you?"),
`--whisper-model small`, `--gpu-layers 24` (all defaults):

| Run | VAD (cpu) | STT (cuda) | generate ttft | generate total (cuda) | time to 1st audio | **round trip** |
|---|---|---|---|---|---|---|
| 1 | 0.101s | 0.132s | 0.143s | 0.831s (22 tok, 3 sentences) | 0.434s | **0.667s** |
| 2 | 0.094s | 0.132s | 0.140s | 1.422s (39 tok, 5 sentences) | 0.441s | **0.667s** |
| 3 | 0.093s | 0.133s | 0.140s | 1.560s (42 tok, 4 sentences) | 0.476s | **0.701s** |

All three: **PASS (< 2s)**, with comfortable margin (~65% of the budget unused).

Model *loading* (once, at harness startup, not part of the per-utterance round trip):
VAD ~0.04s, STT ~1.4-1.5s (includes a forced CUDA warm-up transcription -- see above),
RWKV ~0.35s, TTS ~1.4s (includes one-time ONNX Runtime session warm-up: piper's first
synthesis call was observed to take ~5-6s cold vs. ~0.06-0.1s steady-state, so
`TextToSpeech.__init__` eats that cost up front rather than let it land on the first
real response).

Example full stage breakdown (run 1):
```
VAD                  cpu          0.1010   segments=[{'start': 0.0, 'end': 1.1}]
STT                  cuda         0.1320   requested=cuda lang=en(1.00)
generate             cuda         0.8311   tokens=22 ttft=0.143s sentences=3
TTS (first sentence) cpu          0.0749   time_to_first_audio=0.434s
TTS (remaining sentences) cpu     0.1031   2 more sentence(s), synthesized while earlier
                                            audio played / generation continued
Round-trip time (utterance received -> first spoken audio starts): 0.6672s  [PASS (< 2s)]
(For reference only, NOT the round-trip metric: full response took ~7-10s until all
spoken audio finished playing -- this scales with response length/sentence count.)
```

## Incremental TTS (streamed, not batched)

Per the spec's background section ("feed streamed text into `generate()` incrementally
and consume its streamed output... not wait for a full batched response before starting
TTS"): `pipeline.py` does not call `generate()` to completion and then TTS the whole
result. `speech_streamer.py`'s `IncrementalSpeaker` is fed one fragment at a time via
`generate()`'s existing `on_partial` streaming callback, buffers text until a sentence
boundary completes, and synthesizes+queues that sentence for playback immediately --
while the model keeps generating the rest of the response on the GPU. A background
thread plays queued sentences back in order, so synthesis/generation of sentence N+1
overlaps with sentence N still being spoken, rather than blocking on it. This is why the
round-trip metric above (time to *first* spoken audio) is decoupled from the response's
total length -- a 3-sentence and a 5-sentence response both hit the speakers in well
under a second, even though the 5-sentence one obviously takes longer to finish being
read aloud.

(Token-level TTS was considered and rejected: `xenon_generate()` is one blocking FFI call
with the whole C-side generation loop running under it, so `on_partial` fires
synchronously per token on that same call stack -- but synthesizing audio for a
half-formed word doesn't make sense regardless. Sentence-level chunking is the smallest
unit that's both speakable and lets synthesis start before the full response exists.)

## Graceful no-speech handling (does not hang)

Two independent points where "no speech" is detected and handled without blocking
forever, both exercised in testing:

1. **WAV/batch path** (`fixtures/silence.wav`, 3s of digital silence): `VoiceActivityDetector.detect()`
   finds zero speech segments; `pipeline.run_once_from_wav` returns immediately
   (`reason="no_speech"`) without ever calling STT/generate/TTS.
   ```
   Result: NOT OK (reason=no_speech)
   VAD   cpu   0.1011   segments=[]
   Round-trip time: 0.1012s
   ```
2. **Live mic path**: `audio_io.record_from_mic()` streams real mic input through
   silero-vad's frame-by-frame `VADIterator` and bails out with
   `reason="no_speech_timeout"` if no speech *starts* within `--max-listen-sec` (default
   6s) -- it does not wait indefinitely for speech that never comes. This was verified
   live against the machine's actual microphone (an empty/quiet room, since no one was
   available to speak during testing) and returned cleanly:
   ```
   $ python cli_harness.py --mic --max-listen-sec 3 --no-speak
   Result: NOT OK (reason=no_speech)
   VAD (mic listen)   cpu   3.0111   reason=no_speech_timeout
   Round-trip time: 3.0111s
   ```
   If speech *does* start but then runs unusually long, `max_utterance_sec` (default 10s)
   caps recording length as a second safety net against a stuck-open mic.

In both cases the harness exits with a clean, correct result object (`ok=False,
reason="no_speech"`) rather than an exception, a timeout error, or a hang -- `main()`
still returns exit code 0 for this case (it's expected behavior, not a failure).

## Files

| File | Role |
|---|---|
| `xenon_engine.py` | ctypes bridge to Phase 1's `xenon_inference.dll` (`build-cuda-app`). The single place voice text reaches `generate()`. |
| `vad.py` | silero-vad wrapper, forced CPU. |
| `stt.py` | faster-whisper wrapper, GPU, incl. the CUDA-12-DLL fix and forced warm-up. |
| `tts.py` | piper-tts wrapper, CPU, incl. session warm-up. |
| `audio_io.py` | WAV loading/resampling to 16kHz mono, mic capture with streaming VAD gating, speaker playback. |
| `speech_streamer.py` | Sentence-chunked incremental TTS + background playback queue, fed by `generate()`'s streaming callback. |
| `pipeline.py` | Wires everything together; per-stage timing/device bookkeeping. |
| `cli_harness.py` | CLI entry point (`--wav` / `--mic`), prints load times, VRAM, per-stage timing, pass/fail verdict. |
| `make_fixture.py` | Generates `fixtures/sample_greeting.wav` via piper (the "no human available" stand-in input). |
| `requirements.txt` | Pinned versions, incl. the CUDA-12 redistributable packages and why. |

## Known limitations / out of scope

- No UI -- this is CLI-only, matching Phase 1's approach. Phase 3 wires a mic-toggle
  button to this pipeline.
- Live-mic *speech recognition accuracy* wasn't validated with an actual human voice in
  this environment (see "Why a WAV-file input path" above) -- only the mic capture
  mechanics and the no-speech timeout path were exercised live. Worth a manual sanity
  check with a real voice once a human is available.
- Sentence splitting in `speech_streamer.py` is a simple regex on `. ! ?` boundaries, not
  abbreviation-aware -- fine for a small assistant model's short replies, would need
  hardening for longer/more complex generated text.
- English-only (`language="en"` is hardcoded in `stt.py`); no language auto-detection
  path is exposed, though faster-whisper supports it.
