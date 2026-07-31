# Prompt: Xenon2 Phase 1 — Inference Engine Core

## Follow-up fix (read this first if Phase 1 was already built)

Phase 1 was already completed once and verified working, but a real gap
was found afterward: the CMake build does not copy
`inference-engine/data/world_vocab.bin` (the RWKV tokenizer vocab file)
into the compiled output directories (`build-cpu-app/bin/Release/` and
`build-cuda-app/bin/Release/`). As a result, `test_inference.exe` only
works if you `cd` to the project root and pass a relative path to the
vocab file — running the .exe directly from its own build folder (as a
truly portable, self-contained build should support) fails with
`WorldTokenizer: failed to open vocab file`. See task 8 and the added
acceptance criterion below for the fix — everything else in this prompt
describes Phase 1's original scope for context.

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context on
all 6 phases, but you are only doing **Phase 1** right now. Do not start
on UI, voice I/O, or file save/load — those are later phases with their
own separate prompts.

### Why this project is unusual

Most desktop AI assistants (and basically all commercial chat apps) run on
Transformer-based LLMs. This project deliberately does **not** use a
Transformer. Instead it uses **RWKV**, a non-transformer, recurrent-style
architecture. The reasons this matters for how you implement things:

- RWKV maintains a **fixed-size hidden state** between tokens, instead of a
  Transformer's KV-cache that grows with conversation length. This is why
  the project is viable on a portable/offline device with limited RAM —
  memory use must stay flat no matter how long the conversation gets.
- Because inference is recurrent, tokens can be streamed out **one at a
  time** as they're generated, with no need to wait for a full batch. Your
  implementation must expose token-by-token streaming, not just a
  batched "generate whole response" call — later phases (voice, chat UI)
  depend on this streaming behavior for responsiveness.
- The inference engine of choice is `rwkv.cpp` — a C/C++ implementation of
  RWKV inference with GGML-based quantization support (same quantization
  family used by llama.cpp). Do not substitute a Python-only RWKV
  implementation (e.g. HuggingFace transformers' RWKV support) — it will
  be too slow and heavy for the portable/offline goal of this project.

### Target hardware

Development and testing happens on:
- CPU: Intel i7-12850HX, 16 cores / 24 threads
- RAM: 32GB
- GPU: NVIDIA RTX A1000 Laptop GPU, ~4GB VRAM (CUDA-capable). There is also
  an integrated Intel UHD Graphics adapter present — ignore it, it is not
  relevant for inference offload.

Since a real CUDA-capable GPU is present, this phase should validate GPU
offload, not defer it. `rwkv.cpp` supports building with cuBLAS to offload
RWKV layers to GPU, the same way llama.cpp offloads Transformer layers.
A 430M (or even 1.6B) INT4 RWKV model fits comfortably inside 4GB VRAM,
which frees the CPU to run STT/TTS/VAD (added in Phase 2) in parallel
instead of competing with LLM inference for CPU time.

### What "done" looks like for the whole project (context only, not your job today)

Later phases will add: a voice pipeline (STT/TTS) piping through what you
build today, a Tauri+Vue desktop chat UI, message editing/regeneration,
project save/load to JSON, and export/import of models + conversations to
external USB/SSD and or internal storage as an option. None of that is in scope for Phase 1 — but keep
in mind that whatever `generate()` function or API you expose needs to be
callable from a different process/language later (Rust or Python), so
avoid designing it in a way that only makes sense as a one-off CLI tool.

---

## Your task: Phase 1 — Inference Engine Core

Working directory: `C:\Users\New user\Xenon2\`

Build a working, testable RWKV inference pipeline callable from the
command line. No UI, no voice, no file save/load.

### Tasks

1. Set up `inference-engine/` as a project wrapping `rwkv.cpp` (add it as
   a git submodule or vendored dependency — your choice, but document
   which in a README).
2. Implement model loading for a quantized `.ggml` RWKV model file.
3. Implement a `generate(prompt: str, max_tokens: int)` function that
   **streams tokens one at a time** via a callback (not a single batched
   return value).
4. Download a small RWKV model (start with the ~430M parameter size) and
   quantize it to INT4 using the GGML quantize tool that ships with
   `rwkv.cpp`. Store the resulting model file under `models/`.
5. Write a CLI test harness, e.g. `test_inference.exe "hello, how are
   you?"`, that prints streamed tokens to stdout as they're generated
   (not all at once at the end).
6. Additionally build `rwkv.cpp` with cuBLAS enabled (`GGML_CUBLAS=ON` or
   the equivalent build flag for the version in use) so RWKV layers can be
   offloaded to the RTX A1000. Expose a flag or config option on the CLI
   harness (e.g. `--gpu-layers N`) to control how many layers run on GPU
   vs CPU, defaulting to CPU-only if the flag is omitted.
7. Benchmark the same prompt/model both ways — CPU-only vs GPU-offloaded
   (all layers on GPU) — and record tokens/sec and time-to-first-token for
   each in the README.
8. Fix the build so `inference-engine/data/world_vocab.bin` is copied
   automatically into both `build-cpu-app/bin/Release/` and
   `build-cuda-app/bin/Release/` (alongside `test_inference.exe` and the
   other DLLs already copied there) as part of the normal CMake build —
   e.g. a `file(COPY ...)` or `add_custom_command(TARGET ... POST_BUILD
   ...)` step in `CMakeLists.txt`, not a manual copy step someone has to
   remember to run. The tokenizer should look for the vocab file next to
   the executable first (or via a path resolved relative to the
   executable's own location), not only relative to the current working
   directory or the project root.

### Acceptance criteria (verify before considering this phase done)

- The CLI harness loads the model and streams a coherent response to a
  basic greeting prompt in under 2 seconds on the dev machine described
  above, in **both** CPU-only and GPU-offloaded modes.
- Memory usage stays flat as generation proceeds — confirm the RWKV state
  is fixed-size and is not growing with the number of tokens generated
  (this is the core architectural property that makes the whole project
  work; verify it, don't just assume it).
- Tokens print to stdout incrementally as they're generated, visibly
  streaming rather than appearing all at once.
- GPU-offloaded mode is confirmed faster (or at least not slower) than
  CPU-only for the same model/prompt; if GPU offload is somehow slower or
  unstable, document that clearly rather than silently defaulting to CPU.
- Running `test_inference.exe` directly from inside its own build output
  folder (e.g. `cd build-cpu-app\bin\Release` then
  `.\test_inference.exe "hello, how are you?" --model <path-to-model>`,
  with no reliance on being run from the project root) successfully finds
  and loads `world_vocab.bin` without error. Verify this for both the
  CPU-only and CUDA build folders, not just one.

### When finished

Update the Phase 1 heading in `C:\Users\New user\Xenon2\PLAN.md` to mark
it complete, and note in a short `inference-engine/README.md` how to build
it (both CPU-only and cuBLAS builds), which model file it expects, how to
run the CLI test harness with and without GPU offload, and the CPU-vs-GPU
benchmark results. If you're doing the follow-up vocab-file fix on an
already-completed Phase 1, update that same README section to note the fix
and confirm both build folders were re-verified as self-contained.
