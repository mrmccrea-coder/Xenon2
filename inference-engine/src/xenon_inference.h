// xenon_inference.h
//
// Small, stable C API wrapping rwkv.cpp for the Xenon2 project. This boundary is deliberately
// plain C (no C++ types in the public surface) so it can be called via FFI from Rust, Python
// (ctypes/cffi), or any other language later in the project, not just from the bundled CLI
// harness (test_inference.exe). Do not add C++-only types to this header.
#ifndef XENON_INFERENCE_H
#define XENON_INFERENCE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32)
#   if defined(XENON_INFERENCE_SHARED)
#       if defined(XENON_INFERENCE_BUILD)
#           define XENON_API __declspec(dllexport)
#       else
#           define XENON_API __declspec(dllimport)
#       endif
#   else
#       define XENON_API
#   endif
#else
#   define XENON_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded model + tokenizer + RWKV state, ready for generation.
typedef struct xenon_engine xenon_engine;

typedef enum xenon_status {
    XENON_OK = 0,
    XENON_ERROR_ARGS = 1,
    XENON_ERROR_MODEL_LOAD = 2,
    XENON_ERROR_VOCAB_LOAD = 3,
    XENON_ERROR_EVAL = 4,
} xenon_status;

// Invoked once per generated token, as soon as it is available (streaming, not batched).
// - text: pointer to a UTF-8, null-terminated buffer containing the newly available decoded
//   text for this step. May be empty ("") if the token only completed a partial multi-byte
//   UTF-8 sequence that is still being buffered internally (in that case a later callback
//   invocation will include the completed characters).
// - token_id: raw vocab id of the token that was generated, in case the caller wants it.
// - user_data: whatever pointer was passed into xenon_generate.
// Return `false` to stop generation early (e.g. caller wants to cancel); return `true` to keep
// going until max_tokens is reached.
typedef bool (*xenon_token_callback)(const char * text, uint32_t token_id, void * user_data);

// Loads a quantized rwkv.cpp ggml model file and the matching World-tokenizer vocab file
// (see inference-engine/tools/convert_world_vocab.py and inference-engine/data/world_vocab.bin).
// - model_path: path to a .bin ggml model file (FP16/FP32/quantized) produced by rwkv.cpp.
// - vocab_path: path to the binary World-tokenizer vocab file.
// - n_threads: CPU threads used for evaluation; must be >= 1.
// - n_gpu_layers: number of RWKV layers to offload to GPU. 0 = CPU-only. Only has effect if this
//   library was built and linked against a CUDA-enabled rwkv.cpp (see README for the two build
//   configurations); otherwise this value is accepted but has no effect (falls back to CPU).
// Returns NULL on failure -- call xenon_get_last_error() for details.
XENON_API xenon_engine * xenon_load_model(
    const char * model_path,
    const char * vocab_path,
    uint32_t n_threads,
    uint32_t n_gpu_layers
);

// Frees all resources associated with an engine. Safe to call with NULL.
XENON_API void xenon_free_engine(xenon_engine * engine);

// Resets the engine's RWKV state (equivalent to starting a brand new conversation with no
// prior context). Called automatically at the start of xenon_generate; exposed separately in
// case a future caller wants to manage state across multiple generate() calls itself.
XENON_API void xenon_reset_state(xenon_engine * engine);

// Streams a generated continuation of `prompt`, invoking `callback` once per generated token
// as it is produced (not batched). Stops after `max_tokens` tokens, when the callback returns
// false, or when the model emits its end-of-text token (id 0).
// - temperature / top_p: standard sampling parameters (temperature 0 = greedy/argmax).
// - repeat_penalty: discourages resampling tokens that recently appeared in the prompt's tail
//   (up to the last 256 prompt tokens) or earlier in this same generation, by dividing/scaling
//   their logits before sampling (the standard llama.cpp-style repetition penalty). 1.0 = no
//   penalty (identical to the old behavior); values around 1.1-1.3 are the usual useful range.
//   Added after real usage showed this small model imitating a canned phrase that was still
//   sitting in the conversation history from an earlier turn -- with no penalty at all, once a
//   phrase like "I'm sorry, I can't access my memories" entered the resent history, the model
//   would keep reproducing it for unrelated follow-up questions.
// Returns XENON_OK on success (including early stop via callback), otherwise an error code.
XENON_API xenon_status xenon_generate(
    xenon_engine * engine,
    const char * prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
);

// Number of float32 elements in this model's fixed-size RWKV state. This does NOT grow with
// conversation length -- exposed so callers/tests can verify that architectural property
// directly rather than just assuming it.
XENON_API size_t xenon_get_state_len(xenon_engine * engine);

// Number of transformer-analogous "layers" in the loaded model (used e.g. to validate
// --gpu-layers arguments against the actual model).
XENON_API size_t xenon_get_n_layer(xenon_engine * engine);

// Returns 1 if this build of the engine was compiled against a CUDA/cuBLAS-enabled rwkv.cpp,
// 0 otherwise. Lets callers know whether --gpu-layers will actually do anything.
XENON_API int xenon_has_gpu_support(void);

// Human-readable string for the last error that occurred on this thread. May be empty.
XENON_API const char * xenon_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // XENON_INFERENCE_H
