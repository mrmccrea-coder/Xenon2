// sampler.cpp
//
// Phase 10 rewrite of the token sampler. Before this phase, sample_logits() did six full passes
// over the entire n_vocab=65536-entry logit vector per token: heap-allocated four vocab-sized
// buffers, ran exp() in double 65536 times, fully sorted a 65536-element copy just to find a
// top-p cutoff, ran pow() 65536 times to apply temperature, and built a std::discrete_distribution
// over the whole vocabulary just to draw one sample. Measured cost: 4.904 ms/token at this app's
// production settings (temperature=0.8, top_p=0.5, repeat_penalty=1.3, n_vocab=65536).
//
// This version keeps the *same* one unavoidable O(n_vocab) pass (computing exp(logit - max) for
// the whole vocab -- there is no way to know what fraction of the total probability mass the
// top-p nucleus represents without the true softmax denominator, so that pass stays), but:
//   - mutates the caller's logits buffer in place for the repeat-penalty step instead of copying
//     it into a fresh vector (safe: see the doc comment on xenon_sample_logits in sampler.h --
//     every real caller either overwrites this buffer with the next rwkv_eval() before reading
//     it again, or discards the whole xenon_state it lives in without reading it again at all).
//   - replaces `vector<bool> penalized(n_vocab)` with a reusable stamped-epoch array, so repeated
///    calls don't need to re-zero anything.
//   - finds the top-p cutoff via std::nth_element (average O(n)) over a generously-sized
//     candidate set instead of a full O(n log n) sort, with a full-sort fallback for the (very
//     rare, for a real LM's peaked distribution) case where the candidate set doesn't cover the
//     true nucleus -- see TOP_P_CANDIDATE_K below.
//   - folds temperature into the exponent (mathematically identical to the old
//     pow(prob, 1/temperature), see the derivation below) and only re-weights the *surviving*
//     top-p candidates, not the full vocab.
//   - samples via a hand-rolled linear inverse-CDF walk over just the candidate set, instead of
///    building a std::discrete_distribution over all 65536 entries.
// Measured after this rewrite: 0.359 ms/token in isolation (13.7x), see
// inference-engine/README.md's "Fast sampling" section for the full before/after numbers,
// including the (much smaller, ~4%) effect on end-to-end tok/s -- rwkv_eval's own matmuls still
// dominate total per-token time on this model/hardware.
//
// Correctness note (read before touching this file): a subtly wrong version of this function
// produces plausible-but-wrong output that no smoke test catches -- every reply a user sees comes
// out of this. test_sampler.cpp is the correctness gate: it keeps a frozen copy of the pre-Phase-10
// algorithm and asserts this version matches it exactly at temperature=0 (fully deterministic --
// no RNG is consumed at all on that path) and matches its *distribution* (many draws, not one) at
// temperature>0, including a dedicated test for the top-p candidate-set fallback path.
#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace {

// How many candidates std::nth_element pulls out before falling back to a full sort. Chosen
// generously relative to n_vocab=65536: a real language model's softmax is normally extremely
// peaked (a handful of tokens capture the overwhelming majority of probability mass), so the
// nucleus fitting inside the top 2048 by raw probability is the expected case, not a near-miss.
// The fallback below guarantees correctness even if that assumption is ever wrong for some
// prompt/model combination -- this constant is a performance tuning knob, never a correctness
// one.
constexpr size_t TOP_P_CANDIDATE_K = 2048;

// Reusable scratch memory for xenon_sample_logits, so repeated calls don't heap-allocate fresh
// vocab-sized buffers every time (the "four vocab-sized buffers" problem this phase closes).
// `thread_local` rather than tied to a particular xenon_engine/xenon_state: sampling is called
// from inside the single mutex-serialized generation loop (see EngineInner's doc comment in
// app/src-tauri/src/inference.rs -- only one thread ever touches a given engine/state at a time),
// so there is no cross-conversation data race risk here; a thread_local just means each call
// reuses its own memory instead of a shared global, which would be the actually risky choice if
// this code path is ever used concurrently across engines in the future.
struct SamplerScratch {
    // Epoch-stamped "was this vocab entry already penalized this call" tracking, replacing
    // `vector<bool> penalized(n_vocab)`. `stamp[i] == epoch` means "already penalized in the
    // current call" -- bumping `epoch` for the next call is O(1) instead of re-zeroing the array.
    std::vector<int64_t> penalty_stamp;
    int64_t epoch = 0;

    // exp(logit - max) for the whole vocab, i.e. the softmax numerator before dividing by the
    // partition function -- computed once per call, reused for argmax (greedy), for finding the
    // top-p cutoff, and as the starting point for the tempered candidate weights below.
    std::vector<float> exp_vals;

    // Reused index buffer for std::nth_element/std::sort during top-p candidate selection.
    std::vector<uint32_t> candidate_idx;

    // Tempered weights for just the surviving top-p candidates (indices 0..n_candidates-1 of
    // candidate_idx address this by position, not by vocab id) -- a separate buffer from
    // exp_vals so the two stay independently readable during the reweighting step.
    std::vector<float> candidate_weight;

    void ensure_size(size_t n_vocab) {
        if (penalty_stamp.size() != n_vocab) {
            penalty_stamp.assign(n_vocab, 0);
            epoch = 0;
            exp_vals.resize(n_vocab);
            candidate_idx.resize(n_vocab);
            candidate_weight.resize(n_vocab);
        }
    }
};

thread_local SamplerScratch g_scratch;

// Sorts `idx[0..count)` descending by `exp_vals[idx[i]]` and walks the cumulative (normalized)
// sum until it exceeds `top_p`, returning the number of entries that belong in the nucleus (i.e.
// the first N such that their cumulative probability mass exceeds top_p) via `out_n_survivors`.
// Assumes `idx[0..count)` already contains exactly the candidate set to consider -- the caller
// decides whether that's a generous top-K subset or the entire vocab (the fallback path).
void sort_and_find_nucleus(
    std::vector<uint32_t> & idx,
    size_t count,
    const std::vector<float> & exp_vals,
    double inv_sum,
    float top_p,
    size_t & out_n_survivors
) {
    std::sort(idx.begin(), idx.begin() + count, [&](uint32_t a, uint32_t b) {
        return exp_vals[a] > exp_vals[b];
    });

    double cumulative = 0.0;
    for (size_t i = 0; i < count; i++) {
        cumulative += exp_vals[idx[i]] * inv_sum;
        if (cumulative > top_p) {
            out_n_survivors = i + 1;
            return;
        }
    }
    out_n_survivors = count; // top_p not exceeded even after all candidates -- keep everything considered
}

} // namespace

uint32_t xenon_sample_logits(
    float * logits,
    size_t n_vocab,
    float temperature,
    float top_p,
    float repeat_penalty,
    const std::vector<uint32_t> & recent_tokens,
    std::mt19937 & rng
) {
    SamplerScratch & s = g_scratch;
    s.ensure_size(n_vocab);

    // --- Repeat penalty: mutate `logits` in place using the stamped-epoch array instead of a
    // fresh vocab-sized copy + vector<bool> every call. ---
    if (repeat_penalty != 1.0f && repeat_penalty > 0.0f && !recent_tokens.empty()) {
        s.epoch++;
        for (uint32_t t : recent_tokens) {
            if (t < n_vocab && s.penalty_stamp[t] != s.epoch) {
                s.penalty_stamp[t] = s.epoch;
                float & l = logits[t];
                l = (l > 0.0f) ? (l / repeat_penalty) : (l * repeat_penalty);
            }
        }
    }

    // --- The one full-vocab pass that can't be avoided: exp(logit - max), needed both for
    // argmax (greedy) and as the true softmax numerator for everything below. ---
    float max_logit = logits[0];
    for (size_t i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Single-precision exp() here, not double -- the old code computed this same value as
    // std::exp(double), which the audit specifically calls out as wasteful (the input, the
    // stored result, and every consumer of it are float; only the running `sum` stays double, to
    // avoid accumulating rounding error over 65536 additions, same rationale a double
    // accumulator would have regardless of the per-term precision).
    double sum = 0.0;
    for (size_t i = 0; i < n_vocab; i++) {
        float e = std::exp(logits[i] - max_logit);
        s.exp_vals[i] = e;
        sum += e;
    }

    if (temperature <= 0.0f) {
        // Greedy: argmax(exp_vals) == argmax(softmax(logits)) exactly (dividing every entry by
        // the same positive `sum` can't change which is largest), so this matches the old code's
        // argmax(probs) bit-for-bit without needing to finish normalizing.
        size_t best = 0;
        for (size_t i = 1; i < n_vocab; i++) {
            if (s.exp_vals[i] > s.exp_vals[best]) best = i;
        }
        return static_cast<uint32_t>(best);
    }

    double inv_sum = 1.0 / sum;

    // --- Top-p: find the nucleus via a generous candidate set + nth_element instead of sorting
    // the entire vocab. ---
    size_t n_candidates;
    if (top_p < 1.0f && top_p > 0.0f) {
        size_t k = std::min(TOP_P_CANDIDATE_K, n_vocab);

        for (size_t i = 0; i < n_vocab; i++) s.candidate_idx[i] = static_cast<uint32_t>(i);
        std::nth_element(
            s.candidate_idx.begin(), s.candidate_idx.begin() + k, s.candidate_idx.begin() + n_vocab,
            [&](uint32_t a, uint32_t b) { return s.exp_vals[a] > s.exp_vals[b]; }
        );

        // Does the candidate set's own cumulative mass exceed top_p before running out? If so,
        // the true nucleus is fully contained in it (everything excluded is smaller than every
        // included candidate by construction of nth_element, so it can't be part of the
        // nucleus) and this result is identical to what a full sort would have found.
        double candidate_mass = 0.0;
        for (size_t i = 0; i < k; i++) candidate_mass += s.exp_vals[s.candidate_idx[i]] * inv_sum;

        if (candidate_mass > top_p) {
            sort_and_find_nucleus(s.candidate_idx, k, s.exp_vals, inv_sum, top_p, n_candidates);
        } else {
            // Rare fallback: the generous top-K candidate set still doesn't cover the nucleus
            // (e.g. an unusually flat/near-uniform distribution). Full sort over the whole
            // vocab, exactly matching the pre-Phase-10 code path -- correctness over speed here,
            // since this is not the common case this rewrite optimizes for.
            for (size_t i = 0; i < n_vocab; i++) s.candidate_idx[i] = static_cast<uint32_t>(i);
            sort_and_find_nucleus(s.candidate_idx, n_vocab, s.exp_vals, inv_sum, top_p, n_candidates);
        }
    } else {
        // top-p disabled: every token is a candidate, in original vocab-id order.
        for (size_t i = 0; i < n_vocab; i++) s.candidate_idx[i] = static_cast<uint32_t>(i);
        n_candidates = n_vocab;
    }

    // --- Temperature: fold into the exponent instead of a second full-vocab pow() pass, and
    // only over the surviving candidates. ---
    // Derivation: softmax(x)_i^(1/T) / sum_k softmax(x)_k^(1/T) == softmax(x/T)_i exactly (the
    // old code's approach -- pow() the already-softmaxed, possibly top-p-zeroed probabilities,
    // then renormalize -- is mathematically identical to computing exp((logit-max)/T) directly
    // and renormalizing over the same surviving set: raising exp_vals[i] = exp(logit_i - max) to
    // the power 1/temperature differs from raising the *normalized* probability by only a
    // constant factor common to every candidate, which cancels out on renormalization). Doing
    // this only over `n_candidates` (typically a few hundred, not 65536) is the actual win here.
    double t_sum = 0.0;
    for (size_t i = 0; i < n_candidates; i++) {
        uint32_t id = s.candidate_idx[i];
        float base = s.exp_vals[id];
        float tempered = (temperature == 1.0f)
            ? base
            : static_cast<float>(std::exp(std::log(static_cast<double>(base)) / temperature));
        s.candidate_weight[i] = tempered;
        t_sum += tempered;
    }

    // --- Sample via linear inverse-CDF over just the candidates, instead of building a
    // std::discrete_distribution over the whole vocab. ---
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng) * t_sum;
    double running = 0.0;
    for (size_t i = 0; i < n_candidates; i++) {
        running += s.candidate_weight[i];
        if (running >= r) return s.candidate_idx[i];
    }
    return s.candidate_idx[n_candidates - 1]; // floating-point rounding guard, same as discrete_distribution's own
}
