// test_sampler.cpp
//
// Phase 10 correctness/distribution gate for the sampler rewrite (sampler.h/.cpp). Needs no
// model, no tokenizer, no rwkv.dll -- everything here is synthetic logits vectors, so this runs
// in milliseconds and is safe to run on every change to sampler.cpp.
//
// `reference_sample_logits` below is a FROZEN, byte-for-byte copy of the pre-Phase-10
// implementation (the version that shipped through Phase 9) -- kept only so this test can prove
// the rewrite matches it, never used in production. Do not "clean up" or optimize this copy; its
// entire value is being an unmodified snapshot of the old behavior to compare against.
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

// --- Frozen copy of the pre-Phase-10 sampler (see xenon_inference.cpp's git history for the
// original) -- intentionally duplicated here rather than shared, so this file stays a
// self-contained, permanent record of "the old behavior" independent of what sampler.cpp
// becomes in the future. ---
uint32_t reference_sample_logits(
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

// --- Synthetic logits generators -----------------------------------------------------------
//
// No model is loaded in this test, so these stand in for "real RWKV logits" -- a mix of shapes
// meant to stress different code paths, not just the easy case.

constexpr size_t N_VOCAB = 65536;

// Typical peaked LM-like distribution: mostly small noise, a handful of standout high-logit
// tokens -- the common case the top-p candidate-set optimization is designed for. `spike_lo`/
// `spike_hi` control how dominant the spikes are: soft spikes (this is used for correctness/
// distribution tests, which want several tokens to stay genuinely competitive so a "top-N
// frequency" comparison is actually testing something) vs. sharp spikes (used for the benchmark,
// which wants a realistic RWKV-like near-deterministic peak, not an artificially flat one -- a
// too-flat distribution can push top_p's nucleus past TOP_P_CANDIDATE_K and silently measure the
// rare full-sort fallback path instead of the common case this phase optimizes for).
std::vector<float> make_peaked_logits(std::mt19937 & rng, int n_spikes, float spike_lo = 1.5f, float spike_hi = 4.0f) {
    std::vector<float> v(N_VOCAB);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    for (auto & x : v) x = noise(rng);
    std::uniform_int_distribution<size_t> pick(0, N_VOCAB - 1);
    std::uniform_real_distribution<float> spike_height(spike_lo, spike_hi);
    for (int i = 0; i < n_spikes; i++) v[pick(rng)] += spike_height(rng);
    return v;
}

// Near-uniform/flat distribution -- deliberately adversarial for the top-p candidate-set
// shortcut: with every logit nearly equal, the nucleus needs far more than TOP_P_CANDIDATE_K
// tokens, forcing the full-sort fallback path in sampler.cpp. This is the case trap #3 in
// prompts/phase10_prompt.md warns about.
std::vector<float> make_flat_logits(std::mt19937 & rng) {
    std::vector<float> v(N_VOCAB);
    std::uniform_real_distribution<float> noise(-0.01f, 0.01f); // tiny spread -> huge nucleus
    for (auto & x : v) x = noise(rng);
    return v;
}

std::vector<uint32_t> make_recent_tokens(std::mt19937 & rng, size_t count, size_t n_vocab) {
    std::vector<uint32_t> v(count);
    std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(n_vocab - 1));
    for (auto & t : v) t = pick(rng);
    return v;
}

// --- Test 1: exact match at temperature = 0 (fully deterministic, no RNG consumed at all on
// this path in either implementation) -------------------------------------------------------
bool test_exact_match_greedy(int & failures) {
    bool all_ok = true;
    std::mt19937 seed_rng(1);

    struct Case { const char * name; std::vector<float> logits; float repeat_penalty; size_t recent_count; };
    std::vector<Case> cases;
    cases.push_back({ "peaked, no penalty", make_peaked_logits(seed_rng, 3), 1.0f, 0 });
    cases.push_back({ "peaked, penalty on", make_peaked_logits(seed_rng, 3), 1.3f, 64 });
    cases.push_back({ "flat, no penalty", make_flat_logits(seed_rng), 1.0f, 0 });
    cases.push_back({ "flat, penalty on", make_flat_logits(seed_rng), 1.3f, 200 });

    for (auto & c : cases) {
        // Repeated calls on the SAME scratch, with varying recent_tokens each time, to stress
        // the epoch-stamp reuse logic across multiple sequential calls (not just a single one).
        for (int trial = 0; trial < 20; trial++) {
            std::vector<float> logits_a = c.logits;
            std::vector<float> logits_b = c.logits;
            auto recent = make_recent_tokens(seed_rng, c.recent_count, N_VOCAB);

            std::mt19937 rng_a(42), rng_b(42); // temperature<=0 consumes no RNG, but pass one anyway
            uint32_t ref = reference_sample_logits(logits_a.data(), N_VOCAB, 0.0f, 0.5f, c.repeat_penalty, recent, rng_a);
            uint32_t got = xenon_sample_logits(logits_b.data(), N_VOCAB, 0.0f, 0.5f, c.repeat_penalty, recent, rng_b);

            if (ref != got) {
                printf("[greedy exact-match] FAIL case=\"%s\" trial=%d: reference=%u new=%u\n", c.name, trial, ref, got);
                all_ok = false;
                failures++;
            }
        }
    }
    printf("[greedy exact-match] %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// --- Test 2: distributional match at temperature > 0. A single stochastic draw can't prove two
// samplers are equivalent (discrete_distribution and a hand-rolled inverse-CDF consume the RNG
// stream differently even given identical probabilities), so this draws many samples from each
// against the SAME fixed logits/recent_tokens and compares empirical frequencies. -------------
bool test_distribution_match(int & failures) {
    std::mt19937 seed_rng(7);
    auto logits = make_peaked_logits(seed_rng, 40);
    auto recent = make_recent_tokens(seed_rng, 100, N_VOCAB);

    constexpr int N_DRAWS = 20000;
    constexpr float TEMPERATURE = 0.8f; // matches app/src-tauri/src/inference.rs's production value
    constexpr float TOP_P = 0.5f;
    constexpr float REPEAT_PENALTY = 1.3f;

    std::map<uint32_t, int> ref_counts, new_counts;
    std::mt19937 rng_ref(123), rng_new(456); // different seeds are fine -- we compare distributions, not draws

    for (int i = 0; i < N_DRAWS; i++) {
        std::vector<float> a = logits, b = logits; // fresh copy each draw -- both mutate in place now
        ref_counts[reference_sample_logits(a.data(), N_VOCAB, TEMPERATURE, TOP_P, REPEAT_PENALTY, recent, rng_ref)]++;
        new_counts[xenon_sample_logits(b.data(), N_VOCAB, TEMPERATURE, TOP_P, REPEAT_PENALTY, recent, rng_new)]++;
    }

    // Compare the top-20 most-frequent tokens from the reference run against the new
    // implementation's count for those same tokens -- allow a generous statistical tolerance
    // (binomial std-dev-based) rather than expecting an exact count match between two
    // independent draws.
    std::vector<std::pair<uint32_t, int>> ref_sorted(ref_counts.begin(), ref_counts.end());
    std::sort(ref_sorted.begin(), ref_sorted.end(), [](auto & a, auto & b) { return a.second > b.second; });

    bool all_ok = true;
    size_t top_n = std::min<size_t>(20, ref_sorted.size());
    for (size_t i = 0; i < top_n; i++) {
        uint32_t token = ref_sorted[i].first;
        int ref_c = ref_sorted[i].second;
        int new_c = new_counts.count(token) ? new_counts[token] : 0;

        double p = static_cast<double>(ref_c) / N_DRAWS;
        double stddev = std::sqrt(p * (1.0 - p) * N_DRAWS);
        double diff = std::abs(ref_c - new_c);

        // 6 standard deviations is an extremely loose bound (astronomically unlikely to trip by
        // chance) -- this test is checking for a systematically wrong distribution, not doing
        // rigorous statistics.
        if (diff > 6.0 * stddev + 5) {
            printf("[distribution match] token=%u ref_count=%d new_count=%d (stddev~%.1f) -- suspiciously far apart\n",
                   token, ref_c, new_c, stddev);
            all_ok = false;
            failures++;
        }
    }
    printf("[distribution match] top-%zu tokens by reference frequency, %d draws each: %s\n",
           top_n, N_DRAWS, all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// --- Test 3: the top-p candidate-set fallback path, explicitly. Uses the near-uniform "flat"
// distribution where the nucleus needs far more than TOP_P_CANDIDATE_K entries, forcing
// sampler.cpp's full-sort fallback branch. Must use temperature > 0 -- at temperature <= 0 both
// implementations return before ever touching top-p at all, so a greedy-only comparison here
// would silently test nothing about the fallback path. Same distributional-comparison approach
// as test 2 (many draws, compare empirical frequencies), for the same reason (RNG consumption
// differs between std::discrete_distribution and a hand-rolled inverse-CDF walk even given
// identical probabilities). ---
bool test_large_nucleus_fallback(int & failures) {
    std::mt19937 seed_rng(99);
    auto logits = make_flat_logits(seed_rng);

    // Sanity-check the premise: with logits this flat, softmax is close enough to uniform that
    // the top 2048 entries alone cannot plausibly hold more than half the probability mass out
    // of 65536 near-equal entries (2048/65536 ~= 3%) -- confirming this distribution actually
    // forces sampler.cpp's fallback branch rather than silently taking the fast path.
    double max_l = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float l : logits) sum += std::exp(static_cast<double>(l - max_l));
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    double top_k_mass = 0.0;
    for (size_t i = 0; i < 2048; i++) top_k_mass += std::exp(static_cast<double>(sorted[i] - max_l)) / sum;
    bool premise_ok = top_k_mass < 0.5; // well under top_p=0.5, guaranteeing the fallback fires

    auto recent = make_recent_tokens(seed_rng, 50, N_VOCAB);
    // Fewer draws than test 2 -- the reference implementation's full sort over a near-uniform
    // 65536-entry array (the whole point of this test) is genuinely the slow path being fixed,
    // so this keeps the test suite's runtime reasonable while still giving each of the top-30
    // tokens enough samples for a meaningful comparison (this distribution is much flatter, so
    // even a smaller N_DRAWS spreads across many more distinct tokens than test 2's).
    constexpr int N_DRAWS = 6000;
    constexpr float TEMPERATURE = 0.8f;
    constexpr float TOP_P = 0.5f;
    constexpr float REPEAT_PENALTY = 1.3f;

    std::map<uint32_t, int> ref_counts, new_counts;
    std::mt19937 rng_ref(11), rng_new(22);
    for (int i = 0; i < N_DRAWS; i++) {
        std::vector<float> a = logits, b = logits;
        ref_counts[reference_sample_logits(a.data(), N_VOCAB, TEMPERATURE, TOP_P, REPEAT_PENALTY, recent, rng_ref)]++;
        new_counts[xenon_sample_logits(b.data(), N_VOCAB, TEMPERATURE, TOP_P, REPEAT_PENALTY, recent, rng_new)]++;
    }

    std::vector<std::pair<uint32_t, int>> ref_sorted(ref_counts.begin(), ref_counts.end());
    std::sort(ref_sorted.begin(), ref_sorted.end(), [](auto & a, auto & b) { return a.second > b.second; });

    bool all_ok = true;
    size_t top_n = std::min<size_t>(30, ref_sorted.size());
    for (size_t i = 0; i < top_n; i++) {
        uint32_t token = ref_sorted[i].first;
        int ref_c = ref_sorted[i].second;
        int new_c = new_counts.count(token) ? new_counts[token] : 0;
        double p = static_cast<double>(ref_c) / N_DRAWS;
        double stddev = std::sqrt(p * (1.0 - p) * N_DRAWS);
        if (std::abs(ref_c - new_c) > 6.0 * stddev + 5) {
            printf("[large-nucleus fallback] token=%u ref_count=%d new_count=%d (stddev~%.1f) -- suspiciously far apart\n",
                   token, ref_c, new_c, stddev);
            all_ok = false;
            failures++;
        }
    }

    printf("[large-nucleus fallback] premise (top-2048 mass=%.4f < 0.5, fallback genuinely exercised): %s\n",
           top_k_mass, premise_ok ? "confirmed" : "NOT CONFIRMED -- test may not be exercising the fallback");
    if (!premise_ok) { all_ok = false; failures++; }
    printf("[large-nucleus fallback] top-%zu tokens by reference frequency, %d draws each: %s\n",
           top_n, N_DRAWS, all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// --- Measurement: per-token timing at the app's production settings, old vs. new. -----------
void run_benchmark() {
    std::mt19937 seed_rng(2026);
    // Sharp spikes here (unlike the softer ones used for the distribution-comparison tests
    // above) -- meant to approximate a realistic, confident RWKV output where the top_p=0.5
    // nucleus is small, which is the common case this phase's candidate-set optimization
    // targets. See make_peaked_logits' doc comment.
    auto logits_template = make_peaked_logits(seed_rng, 3, 8.0f, 15.0f);
    auto recent = make_recent_tokens(seed_rng, 256, N_VOCAB);

    constexpr int N_ITERS = 2000;

    // Runs each labeled function twice (discarding the first) and reports the second timing --
    // a cheap guard against CPU frequency scaling / thermal-throttling making whichever
    // benchmark happens to run first look artificially fast (or a later one look artificially
    // slow), which otherwise silently confounds an old-vs-new comparison run back to back.
    auto bench = [&](const char * label, auto && fn) {
        double ms_per_token = 0.0;
        for (int pass = 0; pass < 2; pass++) {
            std::mt19937 rng(1);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N_ITERS; i++) {
                std::vector<float> logits = logits_template; // fresh copy each iter -- both mutate in place
                fn(logits.data(), rng);
            }
            auto t1 = std::chrono::steady_clock::now();
            ms_per_token = std::chrono::duration<double, std::milli>(t1 - t0).count() / N_ITERS;
        }
        printf("%-40s %.4f ms/token\n", label, ms_per_token);
    };

    printf("\n--- benchmark (temperature=0.8, top_p=0.5, repeat_penalty=1.3, n_vocab=%zu) ---\n", N_VOCAB);
    bench("old (reference_sample_logits)", [&](float * l, std::mt19937 & rng) {
        reference_sample_logits(l, N_VOCAB, 0.8f, 0.5f, 1.3f, recent, rng);
    });
    bench("new (xenon_sample_logits)", [&](float * l, std::mt19937 & rng) {
        xenon_sample_logits(l, N_VOCAB, 0.8f, 0.5f, 1.3f, recent, rng);
    });

    printf("\n--- benchmark (repeat_penalty=1.0 / top_p=1.0, both disabled) ---\n");
    bench("old, penalty+top_p disabled", [&](float * l, std::mt19937 & rng) {
        reference_sample_logits(l, N_VOCAB, 0.8f, 1.0f, 1.0f, recent, rng);
    });
    bench("new, penalty+top_p disabled", [&](float * l, std::mt19937 & rng) {
        xenon_sample_logits(l, N_VOCAB, 0.8f, 1.0f, 1.0f, recent, rng);
    });
}

} // namespace

int main(int argc, char ** argv) {
    bool bench_only = argc > 1 && std::string(argv[1]) == "--benchmark";

    if (bench_only) {
        run_benchmark();
        return 0;
    }

    int failures = 0;
    test_exact_match_greedy(failures);
    test_distribution_match(failures);
    test_large_nucleus_fallback(failures);

    run_benchmark();

    printf("\n%d failure(s).\n", failures);
    return failures > 0 ? 1 : 0;
}
