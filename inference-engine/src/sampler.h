// sampler.h
//
// Internal (not part of the public xenon_inference.h C API) sampling routine used by the
// generation loop in xenon_inference.cpp. Pulled into its own file (Phase 10) so it can be
// unit-tested in isolation from the model/tokenizer -- see test_sampler.cpp, which links this
// file directly and needs no model file, no rwkv.dll, nothing but this translation unit.
#ifndef XENON_SAMPLER_H
#define XENON_SAMPLER_H

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Softmax + repeat-penalty + temperature + top-p (nucleus) sampling. Picks one token index out of
// `n_vocab`. Mirrors rwkv.cpp's python/sampling.py reference implementation's semantics (see the
// long comment in sampler.cpp for the exact algorithm and why it's structured the way it is).
//
// `logits` is mutated in place (Phase 10: previously a `const float*`, copied internally when
// repeat-penalty was active -- see sampler.cpp's top comment for why in-place mutation is safe
// for every real caller). `repeat_penalty` == 1.0 disables the penalty; `top_p` outside (0, 1)
// disables nucleus filtering; `temperature <= 0` means greedy/argmax (no RNG consumed at all).
uint32_t xenon_sample_logits(
    float * logits,
    size_t n_vocab,
    float temperature,
    float top_p,
    float repeat_penalty,
    const std::vector<uint32_t> & recent_tokens,
    std::mt19937 & rng
);

#endif // XENON_SAMPLER_H
