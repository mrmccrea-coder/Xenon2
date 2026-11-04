#include "xenon_inference.h"
#include "sampler.h"
#include "world_tokenizer.h"

#include "rwkv.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
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

// How many trailing tokens xenon_state keeps for the repeat-penalty window -- must match
// REPEAT_PENALTY_WINDOW above (it's the same window, just persisted across calls instead of
// recomputed from a full prompt every time; see xenon_inference.h's xenon_state docs).
constexpr size_t STATE_RECENT_TOKENS_CAP = REPEAT_PENALTY_WINDOW;

// How much trailing *text* xenon_state keeps for the copy guard (see block_copy). Much larger than
// the repeat-penalty window on purpose: the penalty only needs to cover a canned phrase a turn or
// two back, but the copy guard has to reach back over whole previous replies -- with replies now
// allowed to run long, the start of the previous reply is far outside a 256-token window by the
// time the next one begins (that is how the model got stuck re-emitting its previous reply word
// for word, 2026-11-04). ~16k characters is several long replies. Trimmed in chunks via the slack.
constexpr size_t STATE_HISTORY_TEXT_CAP = 16384;
constexpr size_t STATE_HISTORY_TEXT_SLACK = 2048;

} // namespace

// Everything that used to live directly on xenon_engine (Phase 1-7) now lives here instead, so a
// caller can hold several independent conversations against one loaded engine (Phase 8). Defined
// before xenon_engine below because xenon_engine's legacy_state member needs a complete type.
struct xenon_state {
    std::vector<float> state;
    std::vector<float> logits;
    // Rolling window of the last STATE_RECENT_TOKENS_CAP tokens actually fed into or generated
    // from this state, in feed order. Kept persistently (rather than recomputed from a full
    // prompt like the old per-call `recent_tokens` local) so repeat-penalty behavior under
    // incremental prefill is identical to a full re-feed every time -- see xenon_prefill.
    std::vector<uint32_t> recent_tokens;
    // The decoded text of everything fed into or generated from this state (the last
    // STATE_HISTORY_TEXT_CAP bytes), used only by the copy guard. Text rather than token ids on
    // purpose: the same words can be split into tokens in different ways, and a token-level guard
    // let the model copy a 99-character run simply by splitting it differently. See block_copy.
    std::string history_text;
};

struct xenon_engine {
    rwkv_context * ctx = nullptr;
    std::unique_ptr<WorldTokenizer> tokenizer;

    // Copy guard setting, in characters: 0 = off (the default, so every other caller behaves
    // exactly as before). Set per use via xenon_set_copy_guard. See block_copy.
    uint32_t copy_guard_chars = 0;

    size_t n_vocab = 0;
    size_t n_layer = 0;
    size_t state_len = 0;
    size_t logits_len = 0;

    std::mt19937 rng{std::random_device{}()};

    // Lazily allocated on first use by the legacy xenon_reset_state(engine)/xenon_generate() path
    // only -- see those functions. Nothing in this codebase actually calls xenon_reset_state
    // directly any more (Phase 8 callers use xenon_state_reset on their own xenon_state), but it
    // stays part of the public C API for anyone binding against this header.
    std::unique_ptr<xenon_state> legacy_state;

    ~xenon_engine() {
        if (ctx) rwkv_free(ctx);
    }
};

namespace {

void push_recent_tokens(const xenon_engine * engine, xenon_state * state, const uint32_t * tokens, size_t count) {
    state->recent_tokens.insert(state->recent_tokens.end(), tokens, tokens + count);
    size_t n = state->recent_tokens.size();
    if (n > STATE_RECENT_TOKENS_CAP) {
        state->recent_tokens.erase(state->recent_tokens.begin(), state->recent_tokens.begin() + (n - STATE_RECENT_TOKENS_CAP));
    }

    for (size_t i = 0; i < count; i++) {
        state->history_text += engine->tokenizer->decode_token(tokens[i]);
    }
    size_t h = state->history_text.size();
    if (h > STATE_HISTORY_TEXT_CAP + STATE_HISTORY_TEXT_SLACK) {
        state->history_text.erase(0, h - STATE_HISTORY_TEXT_CAP);
    }
}

// Copy guard. If the last `min_chars` characters of the reply being generated already appear
// earlier in the conversation, every token that would carry on copying from there is pushed far
// down (not -inf, so the penalty arithmetic elsewhere can't produce NaN) -- every token whose text
// agrees with what came next in the earlier occurrence, so the copy can't continue however the
// words are split into tokens.
//
// Why: the repeat penalty is a soft nudge over a short window. A small model copying its own
// earlier reply has a far stronger pull than that; it re-emitted its previous reply word for word
// on follow-ups ("write another story", "tell me more") -- byte-identical replies in a row. A
// token-level version of this guard was beaten by the model copying the same text with a different
// token split (a 99-character verbatim run got through), and pushed it into misspellings
// ("goormet") by banning one spelling of a word but not the others. Matching text fixes both.
//
// Rules that keep it from getting in the way:
// - `generated_bytes`: the matched run must lie wholly inside the reply being written, never in the
//   prompt, so asking the same question twice can still get the same answer.
// - Runs made only of digits, punctuation and spaces may repeat (counting, dates, lists of numbers).
// - `min_chars` (~48, about ten tokens) is long enough that names and short phrases are never
//   affected; only real copying is.
// Cost: a substring search of the history text per token, plus -- only while a copy is actually
// being attempted -- one pass over the vocabulary comparing first bytes. Well under a millisecond.
// Returns true if anything was banned (a copy attempt was caught).
bool block_copy(const xenon_engine * engine, float * logits, const std::string & hist, size_t min_chars, size_t generated_bytes) {
    if (min_chars == 0 || generated_bytes < min_chars || hist.size() <= min_chars) return false;
    const size_t len = hist.size();
    const std::string_view h(hist);
    const std::string_view suffix = h.substr(len - min_chars);

    bool has_letters = false;
    for (unsigned char c : suffix) {
        if (std::isalpha(c) || c >= 0x80) { has_letters = true; break; }
    }
    if (!has_letters) return false;

    // Where the same run occurred before, and what followed it each time. The run's own position
    // at the very end (len - min_chars) is excluded by the loop bound.
    std::vector<std::string_view> continuations;
    for (size_t pos = h.find(suffix); pos != std::string_view::npos && pos < len - min_chars; pos = h.find(suffix, pos + 1)) {
        const size_t next = pos + min_chars;
        continuations.push_back(h.substr(next, std::min<size_t>(32, len - next)));
        if (continuations.size() >= 16) break;
    }
    if (continuations.empty()) return false;

    bool banned_any = false;
    for (size_t t = 0; t < engine->n_vocab; t++) {
        const std::string & text = engine->tokenizer->decode_token(static_cast<uint32_t>(t));
        if (text.empty()) continue;
        for (const std::string_view & c : continuations) {
            const size_t m = std::min(text.size(), c.size());
            if (m > 0 && text[0] == c[0] && text.compare(0, m, c.data(), m) == 0) {
                logits[t] = -1.0e9f;
                banned_any = true;
                break;
            }
        }
    }
    return banned_any;
}

// After this many caught attempts in one reply the model is stuck: end the reply cleanly rather
// than let it be forced into ever stranger text (seen: whole sentences written backwards). There
// is deliberately no "stricter after the first catch" mode any more: a 24-character version of it
// caught ordinary phrasing ("Without hesitation, Dragonfly") and pushed the model into junk text;
// the give-up limit already deals with a model that keeps trying to copy.
constexpr size_t COPY_GUARD_GIVE_UP_AFTER = 6;

// Shared by xenon_prefill and xenon_generate_with_state's initial-prompt step: encode `text` and
// evaluate it into `state` (state_in == state_out == state->state.data()), updating cached
// logits and the repeat-penalty window. Returns the encoded token count via `out_token_count`
// (0 if `text` is empty -- caller decides whether that's an error or a no-op).
bool eval_text_into_state(
    xenon_engine * engine,
    xenon_state * state,
    const char * text,
    size_t * out_token_count
) {
    *out_token_count = 0;
    if (!text || text[0] == '\0') {
        return true; // nothing to do; not an error at this layer
    }

    std::vector<uint32_t> tokens;
    try {
        tokens = engine->tokenizer->encode(text);
    } catch (const std::exception & e) {
        set_last_error(std::string("xenon_inference: tokenizer encode failed: ") + e.what());
        return false;
    }

    if (tokens.empty()) {
        return true;
    }

    bool ok = rwkv_eval_sequence_in_chunks(
        engine->ctx,
        tokens.data(),
        tokens.size(),
        /* chunk_size */ 16,
        /* state_in */ state->state.data(),
        /* state_out */ state->state.data(),
        /* logits_out */ state->logits.data()
    );
    if (!ok) {
        set_last_error("xenon_inference: rwkv_eval_sequence_in_chunks failed");
        return false;
    }

    push_recent_tokens(engine, state, tokens.data(), tokens.size());
    *out_token_count = tokens.size();

    // Phase 8 measurement instrumentation: prints the actual number of tokens evaluated on this
    // call when XENON_DEBUG_PREFILL_COUNT is set, so before/after prefill-cost claims can be
    // measured directly instead of estimated from prompt character counts. Off by default (zero
    // cost in the hot path otherwise) -- getenv is only called once per eval, not once per token.
    if (std::getenv("XENON_DEBUG_PREFILL_COUNT")) {
        fprintf(stderr, "[xenon_inference] evaluated %zu token(s)\n", tokens.size());
    }

    return true;
}

// Shared generation loop, used by both xenon_generate_with_state and (via a scratch state)
// xenon_generate. Assumes `state->logits` already holds valid logits for the next token to
// sample (i.e. something -- prefill or a previous generate call -- has evaluated at least one
// token into `state`).
xenon_status run_generation_loop(
    xenon_engine * engine,
    xenon_state * state,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
) {
    std::string pending_utf8; // holds back incomplete multi-byte UTF-8 sequences

    const size_t guard_chars = engine->copy_guard_chars;
    size_t catches = 0;
    size_t generated_bytes = 0;
    for (int i = 0; i < max_tokens; i++) {
        if (guard_chars > 0) {
            if (block_copy(engine, state->logits.data(), state->history_text, guard_chars, generated_bytes)) {
                catches++;
                if (catches >= COPY_GUARD_GIVE_UP_AFTER) {
                    break; // stuck on copying: end the reply here (pending bytes flushed below)
                }
            }
        }
        uint32_t token = xenon_sample_logits(
            state->logits.data(), engine->n_vocab, temperature, top_p,
            repeat_penalty, state->recent_tokens, engine->rng
        );

        push_recent_tokens(engine, state, &token, 1);

        const std::string & token_bytes = engine->tokenizer->decode_token(token);
        generated_bytes += token_bytes.size();
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
                state->state.data(),
                state->state.data(),
                state->logits.data()
            );

            if (!ok) {
                set_last_error("xenon_inference: rwkv_eval failed at token " + std::to_string(i));
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

} // namespace

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

    return engine.release();
}

XENON_API void xenon_free_engine(xenon_engine * engine) {
    delete engine;
}

// Phase 8 follow-up: xenon_engine no longer owns a state at all (see the struct above), so this
// legacy entry point now manages a single lazily-allocated scratch xenon_state internally --
// good enough for the two remaining direct callers of the old signature (voice-pipeline's
// ctypes bindings, test_inference.cpp), neither of which ever called this concurrently on the
// same engine from multiple threads (same discipline as before: this whole API isn't
// thread-safe per engine, see xenon_inference.h).
XENON_API void xenon_reset_state(xenon_engine * engine) {
    if (!engine) return;
    if (!engine->legacy_state) {
        engine->legacy_state.reset(xenon_state_new(engine));
    }
    xenon_state_reset(engine, engine->legacy_state.get());
}

XENON_API xenon_state * xenon_state_new(xenon_engine * engine) {
    if (!engine) return nullptr;
    auto state = std::make_unique<xenon_state>();
    state->state.resize(engine->state_len);
    state->logits.resize(engine->logits_len);
    return state.release();
}

XENON_API void xenon_state_free(xenon_state * state) {
    delete state;
}

XENON_API void xenon_state_copy(xenon_state * dst, const xenon_state * src) {
    if (!dst || !src) return;
    dst->state = src->state;
    dst->logits = src->logits;
    dst->recent_tokens = src->recent_tokens;
    dst->history_text = src->history_text;
}

XENON_API void xenon_state_reset(xenon_engine * engine, xenon_state * state) {
    if (!engine || !state) return;
    rwkv_init_state(engine->ctx, state->state.data());
    std::fill(state->logits.begin(), state->logits.end(), 0.0f);
    state->recent_tokens.clear();
    state->history_text.clear();
}

XENON_API void xenon_set_copy_guard(xenon_engine * engine, uint32_t min_chars) {
    if (!engine) return;
    engine->copy_guard_chars = min_chars;
}

XENON_API xenon_status xenon_prefill(
    xenon_engine * engine,
    xenon_state * state,
    const char * text
) {
    if (!engine || !state) {
        set_last_error("xenon_prefill: invalid arguments");
        return XENON_ERROR_ARGS;
    }
    size_t n_tokens = 0;
    if (!eval_text_into_state(engine, state, text, &n_tokens)) {
        return XENON_ERROR_EVAL;
    }
    return XENON_OK;
}

XENON_API xenon_status xenon_generate_with_state(
    xenon_engine * engine,
    xenon_state * state,
    const char * prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repeat_penalty,
    xenon_token_callback callback,
    void * user_data
) {
    if (!engine || !state || !prompt || !callback || max_tokens <= 0) {
        set_last_error("xenon_generate_with_state: invalid arguments");
        return XENON_ERROR_ARGS;
    }

    size_t n_tokens = 0;
    if (!eval_text_into_state(engine, state, prompt, &n_tokens)) {
        return XENON_ERROR_EVAL;
    }
    if (n_tokens == 0) {
        // Matches xenon_generate()'s long-standing behavior: an empty/zero-token prompt is
        // rejected rather than silently generating from whatever logits happen to already be in
        // `state` (which may be all-zero garbage if this is a freshly-reset state that was never
        // primed with xenon_prefill first).
        set_last_error("xenon_generate_with_state: prompt encoded to zero tokens");
        return XENON_ERROR_ARGS;
    }

    return run_generation_loop(engine, state, max_tokens, temperature, top_p, repeat_penalty, callback, user_data);
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
    if (!engine) {
        set_last_error("xenon_generate: invalid arguments");
        return XENON_ERROR_ARGS;
    }

    // Byte-identical to the pre-Phase-8 body of this function, just reimplemented on top of a
    // scratch xenon_state instead of fields that used to live directly on xenon_engine -- see
    // xenon_inference.h's docs on xenon_generate vs. xenon_generate_with_state.
    xenon_state * scratch = xenon_state_new(engine);
    if (!scratch) {
        set_last_error("xenon_generate: failed to allocate scratch state");
        return XENON_ERROR_ARGS;
    }
    xenon_state_reset(engine, scratch);

    xenon_status status = xenon_generate_with_state(
        engine, scratch, prompt, max_tokens, temperature, top_p, repeat_penalty, callback, user_data
    );

    xenon_state_free(scratch);
    return status;
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
