# Prompt: Xenon2 Phase 10 — Fast sampling (`sample_logits` hot-loop rewrite)

## Background

You're working on **Xenon2**, a portable, offline-first desktop AI assistant (Tauri 2 + Vue 3
front end, Rust backend, RWKV-7 2.9B via a C++ wrapper around `rwkv.cpp`). Read `PLAN.md` and
`app/README.md` first for full context — Phases 1–9 are complete. This is a **performance phase**,
not new user-facing scope, closing one specific finding from a performance audit. **Do not
re-derive it — it was already measured.** Phase 8's own scope notes named this exact function as
excluded ("the `sample_logits()` rewrite... a separate change") — this is that change.

---

## The problem

`sample_logits` in `inference-engine/src/xenon_inference.cpp` (currently lines 40–117) runs once
per generated token, over the full `n_vocab = 65536`-entry logit vector, and does six full passes
over it every single call:

1. Heap-allocates `logits_buf` (a `vector<float>` copy) when repeat-penalty is active, plus a
   `vector<bool> penalized(n_vocab)` alongside it.
2. Heap-allocates `probs` (another `vector<float>`).
3. A softmax pass: `exp()` **in `double`** for all 65536 entries.
4. If `top_p < 1.0`: copies `probs` into `sorted_probs` and **fully sorts** it (`O(n log n)` over
   65536 elements) just to find a cutoff percentile.
5. If `temperature != 1.0`: **`pow()` over all 65536 entries** — this is the redundant one.
   Applying temperature by raising already-softmaxed probabilities to `1/temperature` is
   mathematically equivalent to dividing the raw *logits* by `temperature` **before** the softmax
   (`softmax(x/T)_i = softmax(x)_i^(1/T) / Z'` up to renormalization) — doing it after, over the
   whole vocab, with `pow()`, is strictly extra work for the same result.
6. Builds a `std::discrete_distribution<uint32_t>` over the **entire** 65536-entry `probs` vector
   just to draw one sample.

Measured (`n_vocab = 65536`, `temperature = 0.8`, `top_p = 0.5`, `repeat_penalty = 1.3`, this
machine):

```
current implementation   4.904 ms/token
optimised implementation 0.359 ms/token   (13.7x faster, in isolation)
net effect on real generation throughput:  8.95 -> 9.33 tok/s   (+4.3%)
```

The net effect on end-to-end tok/s is small (4.3%) because `rwkv_eval`'s own matmuls still
dominate total per-token time on this model/hardware — **the audit's own numbers say this
explicitly, don't oversell it.** The win is real (13.7x on the sampling step itself, and it drops
per-token heap churn to near zero) but this is a "worth doing because it's basically free and
removes real waste," not a "this unblocks the app" change. Size the effort and the risk you take
accordingly — see "Explicitly NOT in scope" below.

---

## Non-obvious traps — read these before touching the code

1. **Correctness first: this changes the actual sampling distribution's numerics, not just its
   performance, if done carelessly.** `sample_logits` currently produces the model's real
   sampled output — every reply the user sees came out of this function. A subtly-wrong rewrite
   (an off-by-one in the top-p partial-sort boundary, a sign error folding temperature into the
   exponent, a biased inverse-CDF) produces **plausible-but-wrong output that no smoke test
   catches**, same class of risk as Phase 8's state-caching correctness gate. Prove the rewritten
   sampler matches the old one at `temperature = 0` (fully deterministic — the old code already
   short-circuits to `argmax` before touching top-p/RNG at `temperature <= 0`, so this is a free
   correctness check with zero new API surface), and separately verify the *distribution* (not
   just one sample) matches at `temperature > 0` — e.g. draw many samples from both
   implementations at a fixed logits snapshot and compare empirical distributions (chi-squared or
   just eyeball top-20 token frequencies), since a single stochastic draw can't prove the two are
   equivalent.

2. **Folding temperature into the exponent changes *where* `repeat_penalty` and `top_p` apply
   relative to it.** The current order is: repeat-penalty adjusts raw logits -> softmax -> top-p
   filters the resulting probabilities -> temperature reshapes via `pow`. If you fold temperature
   into the pre-softmax logits (`logits / temperature`), decide explicitly whether repeat-penalty
   is applied before or after that division (it changes the effective penalty strength relative to
   temperature) and whether top-p is computed on the temperature-adjusted distribution or the
   original one — then verify whichever you pick reproduces the *same* final distribution as the
   current code's order of operations, not just "a reasonable-looking one." If the orders can't be
   made equivalent, say so explicitly and pick a justified new order rather than silently changing
   behavior.

3. **`top_p` via partial top-k instead of a full sort changes what "the cutoff" means.**
   `top_p`/nucleus sampling's cutoff is defined by cumulative probability mass, not by a fixed
   candidate count — you don't know in advance how many tokens are in the nucleus. A `partial_sort`
   (or a bucket/radix approach, per llama.cpp) needs a candidate-set size large enough to guarantee
   it contains the true nucleus, or a fallback path when it doesn't. Get this wrong and you
   silently truncate the nucleus for a token that should be reachable, changing output for
   legitimate (if less common) completions. Verify against the full-sort result on real logits,
   not just synthetic/uniform ones — real RWKV logit distributions are typically dominated by a
   handful of tokens, but don't assume that shape, measure it.

4. **`vector<bool> penalized(n_vocab)` -> a stamped epoch array is a real allocation-avoidance
   win, but only if the epoch counter and the array are both owned by something that outlives one
   call.** `sample_logits` is currently a free function taking no persistent state; giving it (or
   its caller) a reusable scratch buffer means either passing a scratch struct in from
   `run_generation_loop`/`xenon_state` (Phase 8 already added a natural home for per-conversation
   scratch: consider whether this belongs on `xenon_state` itself, sized once at
   `xenon_state_new`, or on a separate thread-local/engine-level scratch buffer — the sampler
   itself has no notion of "which conversation," so a wrong choice here could reintroduce a
   cross-conversation data race if the same scratch buffer is naively shared and the engine mutex
   discipline is ever relaxed later). Whatever you pick, keep the existing "one mutex, no
   concurrent eval" discipline intact — this phase must not be the one that quietly introduces
   shared mutable scratch state that only happens to be safe *because* of a lock elsewhere.

5. **Reused scratch buffers must be correctly sized and re-zeroed/re-stamped between calls**, not
   just allocated once and left with stale data from the previous token's call — a heap-allocation
   win that silently reads garbage from a previous call is worse than the allocation it replaced.

6. **The C API surface (`xenon_prefill`, `xenon_generate_with_state`, `xenon_generate`) is
   untouched by this phase** — `sample_logits` is an internal implementation detail of
   `run_generation_loop`, not part of the public header. No signature changes needed anywhere, and
   none of Phase 8's callers (voice-pipeline, `test_inference.cpp`, the Rust side) should need to
   change at all.

---

## Suggested approach

Not mandated — if you have a cleaner design, take it and say why. Roughly, per the audit's own
suggestions (llama.cpp's `llama_sample_*` family is the reference implementation of this exact set
of tricks — read its actual source for the specific numerics, don't guess them):

- Fold `temperature` into the logits before the softmax (`logit / temperature`), removing the
  post-softmax `pow()` pass entirely.
- Replace the full sort for top-p with a `partial_sort`/`nth_element`-based top-k candidate set
  (or a bucketed/radix approach), sized generously enough to contain the true nucleus, with a
  documented fallback if it doesn't.
- Sample by linear inverse-CDF walk over just the candidate set, not `std::discrete_distribution`
  over the full vocab.
- Replace `vector<bool> penalized(n_vocab)` with a reusable stamped-epoch array (an `int` per
  vocab entry, a monotonically increasing epoch counter, "penalized" means "the entry's stamp
  equals the current epoch" — avoids re-zeroing the whole array every call).
- Reuse scratch buffers (the logits-adjustment buffer, the candidate-set buffer, the epoch array)
  across calls instead of allocating fresh `vector`s every token — see traps #4/#5 above for where
  that state should live.

---

## Correctness gate — this is the acceptance criterion that matters

Same standard as Phase 8's state-caching gate ("measure it, don't assume it," applied to
correctness): add a real test (extend `test_state_cache.cpp` or add a new small harness) that:

- At `temperature = 0`, asserts the rewritten sampler picks the **exact same token** as the old
  one, across a range of real logits snapshots (not just one greeting prompt) and a range of
  `repeat_penalty`/`top_p` settings (including `repeat_penalty = 1.0`/disabled and `top_p = 1.0`/
  disabled, to exercise every code path's "off" branch too).
- At `temperature > 0`, draws a large number of samples (e.g. 10,000) from both implementations
  against the same fixed logits snapshot and `recent_tokens` window, and compares the resulting
  empirical distributions (a simple top-N frequency comparison, or a chi-squared test) rather than
  comparing single draws.
- Explicitly tests the top-p boundary case from trap #3 above (a logits distribution where the
  nucleus is unusually large/flat, not just the typical sharply-peaked case) to catch a
  too-small candidate-set truncation bug.

## Measurement required

Report before/after, measured, not estimated:

- Per-token sampling time in isolation (µs/token), old vs. new, at the audit's own settings
  (`temperature = 0.8`, `top_p = 0.5`, `repeat_penalty = 1.3`, `n_vocab = 65536`) and at least one
  other setting combination (e.g. penalty disabled, top-p disabled) to confirm the win isn't
  specific to one code path.
- End-to-end steady-state tok/s, old vs. new, on a real multi-token generation (confirm the ~4.3%
  figure or explain what's different if your numbers land elsewhere).
- Peak per-token heap allocation count (or just "confirm zero heap allocations in the hot loop
  after the first call," if you don't have an allocation counter handy).

---

## Explicitly NOT in scope

- Finding 02 (making Sloth's fact-extraction fire-and-forget) — separate phase, separate file,
  separate part of the codebase (Rust `inference.rs`, not this C++ file).
- Quantization/GPU-offload decisions, thread/affinity tuning, `rwkv_eval_sequence_in_chunks` chunk
  size, the voice sidecar's IPC, Tauri Channels — all still out of scope per Phase 8's list, and
  none of them are touched by this change anyway.
- Any change to `xenon_prefill`/`xenon_generate_with_state`/`xenon_generate`'s public signatures
  or to Phase 8's state-caching logic.

---

## Deliverables

1. Rewritten `sample_logits` (or equivalent) with the redundant full-vocab passes removed, per the
   approach above or a justified alternative.
2. The correctness gate above, as a runnable test, passing.
3. Measured before/after numbers in the commit description.
4. Update `inference-engine/README.md` if it documents sampling behavior anywhere, and update the
   header/inline comments describing the old six-pass implementation.
5. One commit, scoped to this change only.
