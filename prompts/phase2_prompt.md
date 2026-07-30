# Prompt: Xenon2 Phase 2 — Voice I/O Pipeline

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context on
all 6 phases, but you are only doing **Phase 2** right now. Do not start on
the chat UI, message editing, or file save/load — those are later phases
with their own separate prompts.

**Phase 1 must already be complete** before starting this phase — you need
a working `generate(prompt, max_tokens)` streaming function from the RWKV
inference engine at `inference-engine/`. If Phase 1 hasn't produced that
yet, stop and flag it rather than stubbing it out.

### Why this project is unusual

This project deliberately uses **RWKV**, a non-transformer recurrent
architecture, instead of a standard Transformer LLM. RWKV keeps a
fixed-size hidden state instead of a growing KV-cache, and streams tokens
one at a time. That part is Phase 1's concern, not yours — but it matters
here because your voice pipeline should feed text into `generate()`
incrementally and consume its streamed output, not wait for a full batched
response before starting TTS.

### What this phase is building

A pipeline that lets the user *speak* to the assistant instead of typing:

```
Microphone
   ↓
[VAD: silero-vad]        — detects when speech starts/stops
   ↓
[STT: faster-whisper]    — speech → text
   ↓
[generate() from Phase 1] — text → streamed response text
   ↓
[TTS: piper-tts]         — text → speech
   ↓
Speakers
```

No UI is involved yet — this is a CLI-testable pipeline, the same way
Phase 1 was. Phase 3 (desktop UI) will later call into whatever you build
here.

**Important scope note — voice is additive, not a replacement**: typed
text input already works today, because Phase 1's `generate()` takes
plain text as its input regardless of where that text came from. This
phase does not add typing — it adds a *second* way to produce input text
(speaking instead of typing), by turning speech into text via STT and
handing that text to the exact same `generate()` function typed input
would use. Do not build a voice-only code path that bypasses or
duplicates `generate()` — voice output should converge on the same text
input as typing does. The actual UI-level unification of "typed text and
voice transcripts both feed into the same send-message code path" is
Phase 3's job (see `MessageInput.vue`, Phase 3 task 4) — that's where a
human will actually choose between typing or speaking. This phase only
needs to prove the speech → text → `generate()` → text → speech loop
works standalone.

### Target hardware — and how to split work across it

Development and testing happens on:
- CPU: Intel i7-12850HX, 16 cores / 24 threads
- RAM: 32GB
- GPU: NVIDIA RTX A1000 Laptop GPU, ~4GB VRAM (CUDA-capable, driver
  installed). Integrated Intel UHD Graphics also present — ignore it.
- Note: as of 2026-07-29, the CUDA build toolchain (CMake, VS Build Tools,
  CUDA Toolkit) may not yet be installed on the dev machine — this was
  also true for Phase 1's GPU work. Verify it's present before assuming
  CUDA builds will succeed; install/document what's missing if not.

**Deliberate GPU/CPU split for this phase**: Phase 1's RWKV inference
already runs on GPU (cuBLAS offload). For Phase 2, also run **STT on GPU**
via faster-whisper's CUDA backend (faster-whisper is built on CTranslate2,
which supports CUDA execution — use `device="cuda"` when constructing the
Whisper model). That means:

- **GPU handles**: RWKV inference (Phase 1) + faster-whisper STT (this
  phase)
- **CPU handles**: silero-vad (VAD is intentionally lightweight and cheap
  — it does not need GPU) and piper-tts (piper is CPU-only; there's no
  meaningful GPU acceleration path for it)

This split matters because it means voice detection and speech synthesis
never compete with the LLM or STT for GPU time, and never bottleneck each
other on CPU either — VAD is near-instant and TTS for short responses is
cheap enough that 16 CPU cores can run it without stealing cycles the GPU
pipeline needs. Do not put STT on CPU "to keep it simple" — the point of
this phase is validating the GPU-handles-LLM+STT / CPU-handles-VAD+TTS
split works and is actually faster end-to-end than an all-CPU pipeline.

A 4GB VRAM budget note: a quantized RWKV model (430M or 1.6B INT4) plus a
faster-whisper model (use "base" or "small" — not "large", it likely won't
fit alongside the LLM in 4GB VRAM) should coexist in VRAM at the same
time. Verify this fits before assuming it does; if VRAM is tight, drop to
the smaller Whisper model size rather than moving STT back to CPU.

---

## Your task: Phase 2 — Voice I/O Pipeline

Working directory: `C:\Users\New user\Xenon2\`

### Tasks

1. Integrate **faster-whisper** for STT, configured to run on **GPU**
   (`device="cuda"`). Confirm which Whisper model size (base/small) fits
   alongside the Phase 1 RWKV model in the ~4GB VRAM budget.
2. Integrate **silero-vad** for voice activity detection, running on
   **CPU**. It should gate the STT step — only send audio to
   faster-whisper once speech has been detected, and mark when speech has
   stopped so STT knows the utterance is complete.
3. Integrate **piper-tts** for TTS, running on **CPU** — convert the
   generated response text into spoken audio through the speakers.
4. Wire the full chain together: mic → VAD (CPU) → STT (GPU) →
   Phase 1's `generate()` (GPU) → TTS (CPU) → speakers.
5. Build a CLI test harness that runs one full voice round-trip and logs
   timing for each stage separately (VAD detection time, STT time,
   inference time, TTS time), so it's clear where time is being spent and
   which device (CPU/GPU) each stage ran on.
6. Confirm VRAM usage stays within budget with both the RWKV model and
   the Whisper model loaded simultaneously — log peak VRAM usage in the
   test harness output.

### Acceptance criteria (verify before considering this phase done)

- A full voice round-trip for a short greeting ("hello, how are you?")
  completes in under 2 seconds on the dev machine.
- STT is confirmed running on GPU (not silently falling back to CPU —
  check and log the actual execution device faster-whisper reports).
- VAD and TTS are confirmed running on CPU.
- Both models (RWKV + Whisper) fit in VRAM simultaneously without OOM
  errors; if they don't fit together, document the conflict and the
  smaller Whisper model size that does fit.
- The pipeline recovers gracefully if no speech is detected — it does not
  hang waiting for STT input that will never come.

### When finished

Update the Phase 2 heading in `C:\Users\New user\Xenon2\PLAN.md` to mark
it complete, and note in a short `voice-pipeline/README.md` (or similar,
matching wherever this phase's code lives) how to run the CLI harness,
which device each stage runs on and why, the VRAM usage observed, and the
per-stage timing breakdown from testing.
