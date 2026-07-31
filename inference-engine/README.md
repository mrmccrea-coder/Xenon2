# Xenon2 Inference Engine (Phase 1)

A minimal, streaming RWKV inference engine wrapping [`rwkv.cpp`](https://github.com/RWKV/rwkv.cpp),
built as a small C API (`xenon_inference`) plus a CLI test harness (`test_inference.exe`).

No UI, no voice I/O, no persistence here -- see `../PLAN.md` for later phases.

## Why RWKV / rwkv.cpp

RWKV is a recurrent-style (non-Transformer) language model architecture. Unlike a Transformer's
KV-cache, which grows linearly with conversation length, RWKV carries a **fixed-size hidden
state** between tokens. That's the core property this project depends on for running on a
portable/offline device with limited RAM, and it's verified for real below (not just assumed).

`rwkv.cpp` is a C/C++ port of RWKV inference on top of `ggml` (the same quantization family used
by `llama.cpp`), supporting FP32/FP16 and INT4/5/8 quantized inference, with optional cuBLAS GPU
offload.

## Submodule choice

`rwkv.cpp` is vendored as a **git submodule** at `inference-engine/rwkv.cpp`, pinned to
`https://github.com/RWKV/rwkv.cpp` (the current/active org -- this project previously lived
under `saharNooby/rwkv.cpp`; verified via the repo's own release history before adding it).
`rwkv.cpp` itself vendors `ggml` (and `ggml` vendors `kompute`) as nested submodules, so clone
with:

```commandline
git submodule update --init --recursive
```

## Repo layout

```
inference-engine/
  rwkv.cpp/              git submodule: the C library + ggml, unmodified
  src/
    xenon_inference.h    plain-C API surface (stable FFI boundary -- see below)
    xenon_inference.cpp  model loading, prompt eval, streaming generate(), sampling
    world_tokenizer.h/.cpp  greedy-longest-match trie tokenizer for RWKV "World" models
    test_inference.cpp   CLI harness
  tools/
    convert_world_vocab.py  one-time conversion of rwkv.cpp's World vocab .txt -> a small
                             binary file the C++ tokenizer can parse without embedding a
                             Python-repr parser in C++
  data/
    world_vocab.bin      output of the above, checked in (small, ~0.9 MB)
  CMakeLists.txt          builds xenon_inference + test_inference against a *pre-built* rwkv.cpp
  build-cpu-app/          CMake build dir: app linked against rwkv.cpp/build-cpu (CPU-only)
  build-cuda-app/         CMake build dir: app linked against rwkv.cpp/build-cuda (cuBLAS)
```

`../models/` (outside `inference-engine/`) holds the downloaded/converted/quantized model
files; see "Getting a model" below. Large model files are git-ignored, not committed.

## The `generate()` API is a real FFI boundary, not a CLI-only hack

`xenon_inference.h` is plain C (opaque handle + POD types + a C function-pointer callback),
specifically so it's callable later from Rust (`bindgen` / manual `extern "C"` decls) or Python
(`ctypes`/`cffi`) without changes, per the project's later phases. `xenon_generate()` streams
one token at a time via a callback (`xenon_token_callback`) -- it does not buffer/batch the
response internally.

```c
xenon_engine * engine = xenon_load_model(model_path, vocab_path, n_threads, n_gpu_layers);
xenon_generate(engine, prompt, max_tokens, temperature, top_p, my_callback, my_user_data);
xenon_free_engine(engine);
```

## Building

`rwkv.cpp`'s C library and this project's wrapper/CLI are built as **two separate stages**, so
that the CPU-only and CUDA builds can live side by side as fully independent build directories
(per the Phase 1 spec -- both must be available, not just the last one built).

All commands assume PowerShell, run from `inference-engine/`. If `cmake`/`nvcc` aren't found,
your shell's PATH/env cache predates their install -- refresh it first:

```powershell
$machinePath = [System.Environment]::GetEnvironmentVariable("Path","Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path","User")
$env:Path = $machinePath + ";" + $userPath
$env:CUDA_PATH = [System.Environment]::GetEnvironmentVariable("CUDA_PATH","Machine")
```
(The last line is only needed for the CUDA build -- MSBuild's CUDA integration reads
`$env:CUDA_PATH` from the process environment, which a pre-existing shell won't have picked up
after a fresh CUDA Toolkit install.)

### 1a. Build rwkv.cpp, CPU-only

```powershell
cmake -B rwkv.cpp/build-cpu -S rwkv.cpp -G "Visual Studio 17 2022" -A x64 -DRWKV_CUBLAS=OFF
cmake --build rwkv.cpp/build-cpu --config Release --target rwkv
```
Produces `rwkv.cpp/build-cpu/bin/Release/rwkv.dll` + `rwkv.cpp/build-cpu/Release/rwkv.lib`.

### 1b. Build rwkv.cpp, CUDA/cuBLAS-enabled

```powershell
cmake -B rwkv.cpp/build-cuda -S rwkv.cpp -G "Visual Studio 17 2022" -A x64 -DRWKV_CUBLAS=ON `
  -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"
cmake --build rwkv.cpp/build-cuda --config Release --target rwkv
```
Produces the same two files under `rwkv.cpp/build-cuda/`.

`-DRWKV_CUBLAS=ON` is rwkv.cpp's own build flag (checked directly in the checked-out
`CMakeLists.txt` rather than assumed); internally it maps to ggml's current `GGML_CUDA` option
(older ggml history also used `GGML_CUBLAS` -- this checkout uses `GGML_CUDA`).

**Deviation from a "plain" build, discovered while getting this working on this machine:**
CUDA Toolkit 13.3's CCCL (`cuda/std/...`) headers hard-error under MSVC's default
("traditional") preprocessor with `fatal error C1189: ... switch to the standard conforming
preprocessor by passing /Zc:preprocessor to cl.exe`, triggered by ggml-cuda source files that
pull in CUB/Thrust-based reduction kernels (e.g. `sum.cu`) -- not all `.cu` files hit it, which
is why a first build attempt got ~70/146 object files in before failing on one of the affected
files. `-DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"` forwards the conforming-preprocessor
flag to the host compiler for every CUDA translation unit and resolves it. If you're on an
older CUDA Toolkit this flag is likely unnecessary (harmless either way).

### 2. Build the xenon_inference wrapper + test_inference CLI

Point `XENON_RWKV_BUILD_DIR` at whichever rwkv.cpp build you want to link against:

```powershell
# CPU-only app
cmake -B build-cpu-app -G "Visual Studio 17 2022" -A x64 `
  -DXENON_RWKV_BUILD_DIR="<repo>/inference-engine/rwkv.cpp/build-cpu" -DXENON_HAS_CUDA=OFF
cmake --build build-cpu-app --config Release

# CUDA-enabled app
cmake -B build-cuda-app -G "Visual Studio 17 2022" -A x64 `
  -DXENON_RWKV_BUILD_DIR="<repo>/inference-engine/rwkv.cpp/build-cuda" -DXENON_HAS_CUDA=ON
cmake --build build-cuda-app --config Release
```

Each produces `build-<variant>-app/bin/Release/test_inference.exe` (with `rwkv.dll` and
`world_vocab.bin` copied alongside it automatically by the build -- see "Follow-up fix" below).

## Getting a model

Phase 1 uses **RWKV-5 World 0.4B** (~430M params), matched with the **World tokenizer** (World
models use `n_vocab = 65536` and REQUIRE the World tokenizer; Pile/Raven models use the 20B BPE
tokenizer and `n_vocab = 50277` -- pairing the wrong one silently produces garbage output, so
`xenon_load_model` sanity-checks the loaded vocab size against the model's `n_vocab` and refuses
to load on a large mismatch).

```powershell
# 1. Download the PyTorch checkpoint (~924 MB)
#    https://huggingface.co/BlinkDL/rwkv-5-world/blob/main/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth
#    -> save under ../models/

# 2. Convert to rwkv.cpp's ggml format (FP16)
python rwkv.cpp/python/convert_pytorch_to_ggml.py `
  ..\models\RWKV-5-World-0.4B-v2-20231113-ctx4096.pth `
  ..\models\rwkv-5-world-0.4B-FP16.bin FP16

# 3. Quantize to INT4 (Q4_0) using rwkv.cpp's own quantize tool
python rwkv.cpp/python/quantize.py `
  ..\models\rwkv-5-world-0.4B-FP16.bin ..\models\rwkv-5-world-0.4B-Q4_0.bin Q4_0
```
(`quantize.py` needs `rwkv.dll` on its search path -- if it can't auto-find it, point it at
`rwkv.cpp/build-cpu/bin/Release/rwkv.dll` explicitly via
`rwkv_cpp_shared_library.RWKVSharedLibrary(path)`.)

Result: `models/rwkv-5-world-0.4B-Q4_0.bin`, 432.83 MB (from an 881.33 MB FP16 source --
2.04x compression, matching rwkv.cpp's own quantize.py output).

The World tokenizer vocab itself is already converted and checked in at
`inference-engine/data/world_vocab.bin` (see "Repo layout" above); regenerate it with
`python tools/convert_world_vocab.py rwkv.cpp/python/rwkv_cpp/rwkv_vocab_v20230424.txt data/world_vocab.bin`
if you ever need to.

## Follow-up fix: self-contained build output folders (2026-07-31)

Phase 1 was originally verified working, but only when run from the `Xenon2/` repo root, because
`test_inference.cpp`'s default `--vocab` was a path relative to the current working directory
(`inference-engine\data\world_vocab.bin`), and nothing copied that file into the build output
folders. Running `test_inference.exe` directly from inside `build-cpu-app\bin\Release\` or
`build-cuda-app\bin\Release\` (as a portable, self-contained build should support) failed with
`WorldTokenizer: failed to open vocab file`.

Fixed two places:
- `CMakeLists.txt`: a second `add_custom_command(TARGET test_inference POST_BUILD ...)` now
  copies `inference-engine/data/world_vocab.bin` next to `test_inference.exe` on every build,
  alongside the existing `rwkv.dll` copy step -- no manual copy step to remember.
- `test_inference.cpp`: the default vocab path is now resolved relative to the *executable's own
  directory* (via `GetModuleFileNameA`/`get_exe_dir()`), not the current working directory or
  the project root. `--vocab PATH` still overrides this explicitly if you want to point at a
  different vocab file.

Re-verified after the fix by running `test_inference.exe` directly from inside each build output
folder, with no `cd` to the repo root and no `--vocab` flag:

```powershell
cd build-cpu-app\bin\Release
.\test_inference.exe "hello, how are you?" --model "<repo>\models\rwkv-5-world-0.4B-Q4_0.bin" --max-tokens 30
# -> loads and streams normally, no vocab-file error

cd build-cuda-app\bin\Release
.\test_inference.exe "hello, how are you?" --model "<repo>\models\rwkv-5-world-0.4B-Q4_0.bin" --gpu-layers 24 --max-tokens 30
# -> loads and streams normally, no vocab-file error
```

Both build folders confirmed self-contained: `test_inference.exe`, `rwkv.dll`, and
`world_vocab.bin` (plus the runtime DLLs already covered) are all present together in
`build-cpu-app\bin\Release\` and `build-cuda-app\bin\Release\`, and the harness runs from either
folder without any reliance on the working directory or project root.

## Running the CLI harness

Can now be run either from the `Xenon2/` repo root, or directly from inside the build output
folder itself (see the follow-up fix above) -- the default `--vocab` resolves relative to
`test_inference.exe`'s own directory either way. From the repo root:

```powershell
# CPU-only
.\inference-engine\build-cpu-app\bin\Release\test_inference.exe "hello, how are you?"

# GPU-offloaded (all 24 layers on the RTX A1000)
.\inference-engine\build-cuda-app\bin\Release\test_inference.exe "hello, how are you?" --gpu-layers 24
```

Flags: `--model PATH` `--vocab PATH` `--threads N` (default 6) `--gpu-layers N` (default 0 =
CPU-only) `--max-tokens N` (default 100) `--temperature F` (default 0.8) `--top-p F` (default
0.5) `--measure-memory` (print periodic working-set samples to stderr) `--no-stop` (disable the
harness's own `\n\nUser:` stop-sequence heuristic, useful for long benchmark/memory runs).

`--gpu-layers` granularity: rwkv.cpp's C API (`rwkv_init_from_file(path, n_threads,
n_gpu_layers)`) takes a real per-layer offload count, the same granularity as llama.cpp -- this
is not an "all-or-nothing" flag being faked. `--gpu-layers` is a no-op (silently ignored, stays
CPU-only) if `test_inference.exe` was built from `build-cpu-app` (no CUDA support compiled in);
`xenon_has_gpu_support()` reports which build you're running.

The harness prompt-formats your input into a small "User: ... / Xenon: ..." chat prime (matching
rwkv.cpp's own `python/prompt/English-Chat.json` convention) so a small base-ish World model
gives a coherent turn instead of an open-ended continuation, and stops generation early once it
sees a new `\n\nUser:` turn begin.

## Acceptance criteria -- verified results

All measurements below were taken on the dev machine (i7-12850HX, 32GB RAM, RTX A1000 Laptop
GPU 4GB VRAM), prompt `"hello, how are you?"`, model `rwkv-5-world-0.4B-Q4_0.bin` (24 layers).

### 1. Loads + streams a coherent response to a basic greeting in under 2 seconds

CPU-only:
```
$ test_inference.exe "hello, how are you?" --max-tokens 60
xenon_inference: model loaded in 0.311 sec.
hello, how are you? I'm doing well, thanks for asking. I'm working on a project that
I'm excited about. I'm looking forward to getting to know you better.
time to first token: 0.262 sec
total generation time: 0.958 sec (35 tokens, stopped at natural turn end)
throughput: 36.53 tokens/sec
```
Model load + time-to-first-token = **0.31s + 0.26s ≈ 0.57s**, well under 2s. PASS.

GPU-offloaded (`--gpu-layers 24`, all layers on the RTX A1000):
```
$ test_inference.exe "hello, how are you?" --gpu-layers 24 --max-tokens 60
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A1000 Laptop GPU, compute capability 8.6, VMM: yes
xenon_inference: model loaded in 0.528 sec.
hello, how are you? I'm doing well, thanks for asking. I'm feeling a bit tired, but I'm
feeling more energetic today. I'm just trying to get some work done.
time to first token: 0.161 sec
total generation time: 1.052 sec (37 tokens, stopped at natural turn end)
throughput: 35.17 tokens/sec
```
Model load + time-to-first-token = **0.53s + 0.16s ≈ 0.69s**, well under 2s. PASS.

### 2. Memory stays flat as generation proceeds (fixed-size RWKV state, not a growing cache)

250-token CPU-only generation (`--max-tokens 250 --no-stop --measure-memory`), working-set
sampled every 10 tokens:

| tokens generated | working set |
|---|---|
| 10  | 595.50 MB |
| 50  | 595.42 MB |
| 100 | 595.42 MB |
| 150 | 595.42 MB |
| 200 | 595.42 MB |
| 250 | 595.42 MB |

Total drift across 240 additional tokens: **0.08 MB** (noise-level, not growth).

The same test in GPU-offloaded mode (`--gpu-layers 24 --max-tokens 250 --no-stop --measure-memory`)
is, if anything, flatter -- host-process working set held at exactly **710.47 MB** for every
sample from token 10 through token 250 (zero measurable drift; the RWKV state for GPU-offloaded
layers lives in VRAM, not host RAM, so host working set is even less token-count-sensitive).

The RWKV state
buffer itself is a fixed `state_len = 1,622,016` floats = 6.19 MB regardless of how many tokens
have been generated (`xenon_get_state_len()` returns the same value before and after
generation -- it's allocated once at model load and reused in place every step via
`rwkv_eval(ctx, token, state, state, logits)`, never reallocated or grown). This confirms RWKV's
fixed-size-state property directly, rather than assuming it. PASS.

### 3. Tokens print to stdout incrementally, not batched at the end

Confirmed by eye (visible per-token streaming during interactive runs) and by construction: the
CLI's `on_token` callback calls `fputs` + `fflush(stdout)` immediately for each token as
`xenon_generate`'s callback loop invokes it (see `test_inference.cpp`), inside the same
`rwkv_eval` step-by-step loop used for timing -- there is no buffering point where a full
response is assembled before any output happens. PASS.

### 4. GPU-offloaded mode is at least as fast as CPU-only

**Honest result: NOT faster for steady-state throughput at this model size -- documented here
rather than hidden, per the spec's explicit instruction to do so if this happened.**
Time-to-first-token IS meaningfully faster on GPU. See full numbers below.

## CPU vs GPU benchmark

Controlled benchmark: identical prompt (`"hello, how are you?"`), identical model
(`rwkv-5-world-0.4B-Q4_0.bin`, 24 layers, all offloaded in the GPU case), fixed 200-token
generation with the harness's stop-sequence heuristic disabled (`--max-tokens 200 --no-stop`)
so both runs generate the exact same amount of work. Two runs each, back to back:

| Mode | Model load | Time to first token | 200 tokens in | Throughput |
|---|---|---|---|---|
| CPU-only (6 threads) | 0.31s | 0.220s / 0.235s | 3.999s / 4.048s | **50.01 / 49.40 tok/s** |
| GPU-offloaded (24/24 layers, RTX A1000) | 0.53s | **0.133s / 0.135s** | 4.884s / 4.944s | 40.95 / 40.45 tok/s |

**Time-to-first-token is ~40% faster on GPU** (0.13s vs 0.22-0.24s) -- consistent with the
initial prompt-processing pass being a larger, more GPU-friendly batched matmul.

**Steady-state per-token throughput is ~18-20% SLOWER on GPU** than CPU-only (40.5-41.0 tok/s
vs 49.4-50.0 tok/s). This matches the Phase 1 spec's own caveat that this is "plausible for a
model this small": single-token RWKV decode issues many small CUDA kernel launches per layer
(24 layers x several ops each), and at batch size 1 with a 430M-parameter model, kernel-launch
and host<->device synchronization overhead outweighs the compute-throughput advantage the GPU
would have at larger batch sizes or larger models. rwkv.cpp's own README documents the same
pattern for its 169M-parameter cuBLAS benchmark (CPU threads competitive with or beating GPU at
low thread counts on a small model).

**Net verdict**: GPU offload is real, correctly wired (`n_gpu_layers` is a true rwkv.cpp C API
parameter, confirmed offloading via the `ggml_cuda_init` device log line and via VRAM usage),
and improves latency-to-first-response -- but for this specific ~430M model size, CPU-only is
the better choice for sustained generation throughput on this GPU. This isn't hidden or worked
around: `--gpu-layers` defaults to 0 (CPU-only) exactly as the spec requires, and a caller who
cares more about steady-state tok/s than initial latency should leave it at the default. A
larger RWKV model (1.5B+) would likely flip this result, since compute cost per kernel launch
would grow faster than the fixed launch overhead -- untested here, out of scope for Phase 1's
430M model target.
