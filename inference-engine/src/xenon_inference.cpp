#include "xenon_inference.h"
#include "world_tokenizer.h"

#include "rwkv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef XENON_HAS_CUDA
#define XENON_HAS_CUDA 0
#endif

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string & msg) {
    g_last_error = msg;
}

// How many trailing tokens (prompt tail + tokens generated so far this call) count toward the
// repetition penalty. Large enough to cover a canned phrase sitting a turn or two back in the
// resent conversation history (the actual failure mode this was added for), small enough that
// scanning it every step is negligible next to the O(n_vocab) softmax pass below.
constexpr size_t REPEAT_PENALTY_WINDOW = 256;

// Softmax + temperature + top-p (nucleus) sampling, mirroring rwkv.cpp's python/sampling.py
// reference implementation so behavior matches the rest of the rwkv.cpp ecosystem. `repeat_penalty`
// (1.0 = disabled) applies the standard llama.cpp-style penalty to any token present in
// `recent_tokens` before the softmax: divide positive logits, scale up negative ones, so recently
// -seen tokens become less likely without being hard-banned.
uint32_t sample_logits(
    const float * raw_logits,
    size_t n_vocab,
    float temperature,
    float top_p,
    float repeat_penalty,
    const std::vector<uint32_t> & recent_tokens,
    std::mt19937 & rng
) {
    std::vector<float> logits_buf;
    const float * logits = raw_logits;

    if (repeat_penalty != 1.0f && repeat_penalty > 0.0f && !recent_tokens.empty()) {
        logits_buf.assign(raw_logits, raw_logits + n_vocab);
        std::vector<bool> penalized(n_vocab, false);
        for (uint32_t t : recent_tokens) {
            if (t < n_vocab && !penalized[t]) {
                penalized[t] = true;
                float & l = logits_buf[t];
                l = (l > 0.0f) ? (l / repeat_penalty) : (l * repeat_penalty);
            }
        }
        logits = logits_buf.data();
    }

    std::vector<float> probs(n_vocab);

    float max_logit = *std::max_element(logits, logits + n_vocab);

    double sum = 0.0;
    for (size_t i = 0; i < n_vocab; i++) {
        probs[i] = std::exp(static_cast<double>(logits[i] - max_logit));
        sum += probs[i];
    }
    for (size_t i = 0; i < n_vocab; i++) {
        probs[i] = static_cast<float>(probs[i] / sum);
    }

    if (temperature <= 0.0f) {
        return static_cast<uint32_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
    }

    if (top_p < 1.0f && top_p > 0.0f) {
        std::vector<float> sorted_probs = probs;
        std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());

        double cumulative = 0.0;
        float cutoff = sorted_probs.back();
        for (float p : sorted_probs) {
            cumulative += p;
            if (cumulative > top_p) {
                cutoff = p;
                break;
            }
        }

        for (size_t i = 0; i < n_vocab; i++) {
            if (probs[i] < cutoff) probs[i] = 0.0f;
        }
    }

    if (temperature != 1.0f) {
        double s = 0.0;
        for (size_t i = 0; i < n_vocab; i++) {
            probs[i] = std::pow(probs[i], 1.0f / temperature);
            s += probs[i];
        }
        for (size_t i = 0; i < n_vocab; i++) {
            probs[i] = static_cast<float>(probs[i] / s);
        }
    } else {
        double s = std::accumulate(probs.begin(), probs.end(), 0.0);
        for (size_t i = 0; i < n_vocab; i++) probs[i] = static_cast<float>(probs[i] / s);
    }

    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Returns the length of the prefix of `buf` that consists of complete, valid UTF-8 sequences,
// holding back a possibly-incomplete trailing multi-byte sequence for the next call. This lets
// us stream World-tokenizer output (which is byte-level, so a single token can be half of a
// multi-byte UTF-8 character) as valid text without corrupting split characters.
size_t utf8_safe_prefix_len(const std::string & buf) {
    if (buf.empty()) return 0;

    size_t n = buf.size();
    // Look back up to 3 bytes from the end to find the start of a possibly-incomplete sequence.
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char b = static_cast<unsigned char>(buf[n - back]);

        size_t expected_len;
        if ((b & 0x80) == 0x00) expected_len = 1;       // ASCII
        else if ((b & 0xE0) == 0xC0) expected_len = 2;  // 110xxxxx
        else if ((b & 0xF0) == 0xE0) expected_len = 3;  // 1110xxxx
        else if ((b & 0xF8) == 0xF0) expected_len = 4;  // 11110xxx
        else continue; // continuation byte (10xxxxxx), keep looking back

        if (expected_len > back) {
            // Incomplete multi-byte sequence at the tail; hold it back.
            return n - back;
        }
        break;
    }

    return n;
}

} // namespace

struct xenon_engine {
    rwkv_context * ctx = nullptr;
    std::unique_ptr<WorldTokenizer> tokenizer;

    size_t n_vocab = 0;
    size_t n_layer = 0;
    size_t state_len = 0;
    size_t logits_len = 0;

    std::vector<float> state;
    std::vector<float> logits;

    std::mt19937 rng{std::random_device{}()};

    ~xenon_engine() {
        if (ctx) rwkv_free(ctx);
    }
};

extern "C" {

XENON_API xenon_engine * xenon_load_model(
    const char * model_path,
    const char * vocab_path,
    uint32_t n_threads,
    uint32_t n_gpu_layers
) {
    if (!model_path || !vocab_path || n_threads == 0) {
        set_last_error("xenon_load_model: invalid arguments (model_path/vocab_path must be non-null, n_threads >= 1)");
        return nullptr;
    }

    rwkv_set_print_errors(nullptr, true);

    rwkv_context * ctx = rwkv_init_from_file(model_path, n_threads, n_gpu_layers);
    if (!ctx) {
        set_last_error(std::string("xenon_load_model: rwkv_init_from_file failed for ") + model_path);
        return nullptr;
    }

    auto engine = std::make_unique<xenon_engine>();
    engine->ctx = ctx;

    try {
        engine->tokenizer = std::make_unique<WorldTokenizer>(vocab_path);
    } catch (const std::exception & e) {
        set_last_error(std::string("xenon_load_model: ") + e.what());
        return nullptr;
    }

    engine->n_vocab    = rwkv_get_n_vocab(ctx);
    engine->n_layer    = rwkv_get_n_layer(ctx);
    engine->state_len  = rwkv_get_state_len(ctx);
    engine->logits_len = rwkv_get_logits_len(ctx);

    // The World vocab file only assigns ids 1..65529 (id 0 and a handful of trailing ids in the
    // 65536-sized embedding table are reserved/unused), so a small gap between vocab_size() and
    // n_vocab is normal. A LARGE gap (e.g. World's 65536 vs. the 20B/Pile tokenizer's 50277)
    // means the wrong tokenizer was paired with this model -- reject that loudly instead of
    // silently producing garbage output.
    long long vocab_gap = static_cast<long long>(engine->n_vocab) - static_cast<long long>(engine->tokenizer->vocab_size());
    if (vocab_gap > 1000 || vocab_gap < -1000) {
        set_last_error(
            "xenon_load_model: vocab file entry count (" + std::to_string(engine->tokenizer->vocab_size()) +
            ") is wildly different from the model's n_vocab (" + std::to_string(engine->n_vocab) +
            "). This model almost certainly needs a different tokenizer (e.g. a Pile/20B-tokenizer model " +
            "loaded with the World tokenizer, or vice versa) -- output would be garbage, refusing to proceed."
        );
        return nullptr;
    }

    engine->state.resize(engine->state_len);
    engine->logits.resize(engine->logits_len);

    return engine.release();
}

XENON_API void xenon_free_engine(xenon_engine * engine) {
    delete engine;
}

XENON_API void xenon_reset_state(xenon_engine * engine) {
    if (!engine) return;
    rwkv_init_state(engine->ctx, engine->state.data());
}

XENON_API xenon_status xenon_generate(
    xenon_engine * engine,
    const char * prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
) {
    if (!engine || !prompt || !callback || max_tokens <= 0) {
        set_last_error("xenon_generate: invalid arguments");
        return XENON_ERROR_ARGS;
    }

    std::vector<uint32_t> prompt_tokens;
    try {
        prompt_tokens = engine->tokenizer->encode(prompt);
    } catch (const std::exception & e) {
        set_last_error(std::string("xenon_generate: tokenizer encode failed: ") + e.what());
        return XENON_ERROR_EVAL;
    }

    xenon_reset_state(engine);

    // Feed the prompt. Using rwkv_eval_sequence_in_chunks is the recommended way to process a
    // full prompt (chunked to stay within ggml's graph node limit and for better throughput
    // than evaluating one token at a time), per rwkv.h's own docs.
    if (!prompt_tokens.empty()) {
        bool ok = rwkv_eval_sequence_in_chunks(
            engine->ctx,
            prompt_tokens.data(),
            prompt_tokens.size(),
            /* chunk_size */ 16,
            /* state_in */ engine->state.data(),
            /* state_out */ engine->state.data(),
            /* logits_out */ engine->logits.data()
        );

        if (!ok) {
            set_last_error("xenon_generate: rwkv_eval_sequence_in_chunks failed while processing prompt");
            return XENON_ERROR_EVAL;
        }
    } else {
        // Empty prompt: initialize state only, no logits yet -- evaluate a no-op pass isn't
        // possible with zero tokens, so bail out with a clear error instead of guessing.
        set_last_error("xenon_generate: prompt encoded to zero tokens");
        return XENON_ERROR_ARGS;
    }

    // Seed the repetition-penalty window with the tail of the prompt itself (not just tokens
    // generated during this call) -- this is what actually catches a canned phrase sitting a
    // turn or two back in the resent conversation history, not only in-reply self-repetition.
    std::vector<uint32_t> recent_tokens;
    {
        size_t tail_start = prompt_tokens.size() > REPEAT_PENALTY_WINDOW
            ? prompt_tokens.size() - REPEAT_PENALTY_WINDOW
            : 0;
        recent_tokens.assign(prompt_tokens.begin() + tail_start, prompt_tokens.end());
    }

    std::string pending_utf8; // holds back incomplete multi-byte UTF-8 sequences

    for (int i = 0; i < max_tokens; i++) {
        uint32_t token = sample_logits(
            engine->logits.data(), engine->n_vocab, temperature, top_p,
            repeat_penalty, recent_tokens, engine->rng
        );

        recent_tokens.push_back(token);
        if (recent_tokens.size() > REPEAT_PENALTY_WINDOW) {
            recent_tokens.erase(recent_tokens.begin());
        }

        const std::string & token_bytes = engine->tokenizer->decode_token(token);
        pending_utf8 += token_bytes;

        size_t safe_len = utf8_safe_prefix_len(pending_utf8);
        std::string to_emit = pending_utf8.substr(0, safe_len);
        pending_utf8.erase(0, safe_len);

        bool keep_going = callback(to_emit.c_str(), token, user_data);
        if (!keep_going) {
            return XENON_OK;
        }

        if (i + 1 < max_tokens) {
            bool ok = rwkv_eval(
                engine->ctx,
                token,
                engine->state.data(),
                engine->state.data(),
                engine->logits.data()
            );

            if (!ok) {
                set_last_error("xenon_generate: rwkv_eval failed at token " + std::to_string(i));
                return XENON_ERROR_EVAL;
            }
        }
    }

    // Flush any leftover (possibly-invalid) tail bytes so callers don't silently lose them.
    if (!pending_utf8.empty()) {
        callback(pending_utf8.c_str(), 0, user_data);
    }

    return XENON_OK;
}

XENON_API size_t xenon_get_state_len(xenon_engine * engine) {
    return engine ? engine->state_len : 0;
}

XENON_API size_t xenon_get_n_layer(xenon_engine * engine) {
    return engine ? engine->n_layer : 0;
}

XENON_API int xenon_has_gpu_support(void) {
    return XENON_HAS_CUDA ? 1 : 0;
}

XENON_API const char * xenon_get_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
