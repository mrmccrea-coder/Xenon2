# Prompt: Xenon2 Phase 8 — Persistent RWKV State (stop re-prefilling the prompt)

## Background

You're working on **Xenon2**, a portable, offline-first desktop AI assistant
(Tauri 2 + Vue 3 front end, Rust backend, RWKV-7 2.9B via a C++ wrapper around
`rwkv.cpp`). Read `PLAN.md` and `app/README.md` first for full context —
Phases 1–7 are complete and this is a **performance phase**, not new
user-facing scope.

A performance audit found one dominant problem, and this phase closes only
that one. **Do not re-derive it — it was already measured.**

---

## The problem

`xenon_generate()` calls `xenon_reset_state(engine)` on entry
(`inference-engine/src/xenon_inference.cpp:257`). Because the model's state is
thrown away every call, `build_prompt()`
(`app/src-tauri/src/inference.rs:141`) has to re-serialise the *entire*
conversation as text on every single turn, and the engine re-evaluates all of
it before producing one new token.

Measured with the project's own `WorldTokenizer` against `world_vocab.bin`:

```
static preamble re-fed every turn : 153 tokens
turn 1   prefill 169 tokens     (153 already-seen)
turn 10  prefill 466 tokens     (450 already-seen)
Sloth extraction preamble        : 205 tokens, every Sloth turn
```

RWKV is a recurrent architecture whose entire advantage is a **fixed-size
state** (`rwkv_get_state_len()`) that makes prompt reprocessing unnecessary.
The app is currently paying transformer-shaped prefill costs on an
architecture chosen specifically to avoid them, and the cost grows the longer
someone talks to it. Time-to-first-token — the metric that matters most for
the voice path — scales directly with that growing number.

**Goal:** turn N should prefill only the genuinely new tokens (~16), not
153 + all history.

---

## Non-obvious traps — read all of these before designing

These were found during the audit. Each one will bite you if you miss it.

1. **The variable part of the prompt currently comes FIRST.** `build_prompt()`
   emits `[date/time] → [Sloth facts] → [static few-shot] → [time example
   with real clock] → [history] → "Xenon:"`. A prefix cache is impossible
   while a value that changes every turn sits at the front. You must decide
   and justify one of:
   - reorder so the truly static few-shot block leads, then the volatile
     header, then history; or
   - keep the order and cache at a later boundary, accepting a smaller win.

   If you reorder, you **must** re-verify reply quality — the current ordering
   and wording were arrived at empirically (see the comments in
   `build_prompt()` about the model imitating closing-question style, and the
   dedicated time example). Note that
   `try_answer_time_date_deterministically()` already intercepts time/date
   questions before the model runs, so the in-prompt time example matters less
   than it did when it was added. Weigh that; don't assume it.

2. **`run_fact_extraction()` would clobber the conversation state.** It calls
   `xenon_generate()` a second time under the same lock
   (`inference.rs:346–403`). Under a stateful engine that would destroy the
   conversation it just replied to. Extraction must run against a **separate
   scratch state**, never the conversation's.

3. **Two external consumers bind the existing `xenon_generate` signature.**
   `voice-pipeline/xenon_engine.py` declares it via explicit ctypes
   `argtypes`, and `inference-engine/src/test_inference.cpp` calls it
   directly. **Do not change that signature.** Add new entry points and
   re-implement the existing one as a thin wrapper (fresh state → generate)
   so both callers keep working untouched.

4. **Phase 4's edit/regenerate can modify non-last messages.** Editing message
   0 drops everything after it; regenerating can rewrite an assistant message
   in the middle. Any cached state for that conversation is invalid from that
   point on. The front end already sends the full history array to
   `generate_response`, so the backend can diff the incoming history against
   what its cached state has already consumed and decide *extend* vs
   *rebuild*. **Prefer that** — it means no front-end or IPC protocol change
   is needed. Keep this phase contained to C++ and Rust.

5. **The state must match the visible transcript exactly.** The generation
   loop leaves the state having consumed whatever the model emitted, but the
   Rust side then trims at the `\n\nUser:` stop sequence (`ends_with_stop`),
   and the loop also skips the final `rwkv_eval` on its last iteration. So
   "what the state has seen" and "what the transcript says" drift apart. The
   clean fix: snapshot the state *before* generating, and once the reply is
   trimmed to its canonical form, derive the turn's ending state by
   evaluating that canonical text — not by keeping whatever the loop left
   behind. Drift here produces subtly wrong context that is very hard to
   debug later.

6. **Measure `xenon_get_state_len()` for the 2.9B model before choosing a
   snapshot policy.** For the old 0.4B model it was 1,622,016 floats
   (6.19 MB). For RWKV-7 2.9B it will be substantially larger, and this is a
   USB-portable app. Print the real number first, then decide how many
   snapshots you can afford. Bound the cache (LRU over conversations, and a
   cap on per-message-boundary snapshots) rather than growing without limit.

7. **`rwkv.cpp` is not thread-safe per engine.** The existing discipline —
   one `Mutex<EnginePtr>` held across the whole blocking call
   (`EngineState`'s doc comment) — stays. State buffers are per-conversation
   data guarded alongside it; do not introduce concurrent eval.

8. **The C++ library must actually be rebuilt.** `app/src-tauri/build.rs`
   links `inference-engine/build-cpu-app/Release/xenon_inference.lib` and
   copies DLLs from `build-cpu-app/bin/Release`. Rebuild that CMake target
   after changing the C API, or the Rust side will link against a stale
   library and the change will silently do nothing.

---

## Suggested API shape

Not mandated — if you have a cleaner design, take it and say why. But the
shape should make state an explicit, caller-owned object rather than hidden
global mutable state on the engine:

```c
// Opaque, caller-owned. Sized from rwkv_get_state_len() internally.
xenon_state * xenon_state_new(xenon_engine * engine);
void          xenon_state_free(xenon_state * state);
void          xenon_state_copy(xenon_state * dst, const xenon_state * src);
void          xenon_state_reset(xenon_engine * engine, xenon_state * state);

// Evaluate text into `state` without generating anything (prefill only).
xenon_status  xenon_prefill(xenon_engine * engine, xenon_state * state,
                            const char * text);

// Generate starting from `state`; `state` is advanced in place.
xenon_status  xenon_generate_with_state(
    xenon_engine * engine, xenon_state * state, const char * prompt,
    int max_tokens, float temperature, float top_p, float repeat_penalty,
    xenon_token_callback callback, void * user_data);
```

`xenon_generate()` then becomes: make a scratch state, reset, call
`xenon_generate_with_state`, free. Byte-identical behaviour for existing
callers.

Keep `rwkv_eval_sequence_in_chunks` for multi-token prefill (it is
substantially faster than looping `rwkv_eval`), and single `rwkv_eval` for
the generation loop, exactly as now.

---

## Correctness gate — this is the acceptance criterion that matters

Prefix caching is a place where a subtle bug produces plausible-but-wrong
output that no smoke test catches. Prove the state math instead of eyeballing
replies:

- With the RNG seeded to a fixed value, generating a reply **with** the state
  cache must produce **byte-identical** output to generating the same logical
  conversation **without** it (the current reset-and-re-feed path). Build this
  as a real test in `test_inference.cpp` or an equivalent harness, run it
  across at least: turn 1, turn 5, a turn after an edit, and a turn after a
  regenerate.
- If they differ, the cache is wrong. Do not paper over it with "close
  enough" — find the divergence.

This is the project's existing standard ("measure it, don't assume it"), and
it applies to correctness here as much as to performance.

---

## Measurement required

Report before/after numbers, measured, not estimated:

- Time-to-first-token at turns 1, 5 and 10 of a real conversation.
- Prefill tokens actually evaluated per turn (instrument the count).
- `xenon_get_state_len()` in floats and MB for the 2.9B model.
- Peak process memory with the state cache at its bound.
- Confirm steady-state tok/s is unchanged (this phase should not affect
  per-token generation speed — if it moved, explain why).

---

## Explicitly NOT in scope

The audit found nine other issues. **Leave them alone** — they are separately
tracked and mixing them in will make this change impossible to review or
bisect:

- the `sample_logits()` rewrite (measured 13.7× available, but only 4.4% of a
  token — a separate change)
- Q4_0 / GPU-offload / quantisation decisions
- thread count and P-core affinity tuning
- `rwkv_eval_sequence_in_chunks` chunk-size tuning
- the voice sidecar's per-token IPC round-trip, and piper's WAV round trip
- moving token streaming to Tauri Channels

If you find yourself editing `voice.rs`, `ipc_server.py`, `tts.py` or the
sampler, stop — you've left the scope.

---

## Deliverables

1. Working incremental state in `inference-engine/` + `app/src-tauri/`, with
   no change to the existing `xenon_generate` signature.
2. The correctness gate above, as a runnable test, passing.
3. Measured before/after numbers in the PR/commit description.
4. `inference-engine/README.md` updated to document the new state API and the
   caching/invalidation rules; `app/README.md` updated where behaviour
   changed.
5. Update the stale comments that describe the old behaviour — in particular
   `xenon_inference.h`'s doc comment and `build_prompt()`'s "there is no
   cross-call incremental state to build on yet", which will no longer be
   true.
6. One commit, scoped to this change. Author/committer identity is
   `MrMcCrea_coder <MrMcCrea_coder@users.noreply.github.com>`, set via
   `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars — there is no global git
   identity on this machine.

Note: the repository currently has substantial uncommitted Phase 7 work in the
tree. Check `git status` before you start and agree with the user how to
handle it — do not silently commit unrelated changes alongside yours.
