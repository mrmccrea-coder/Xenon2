# Prompt: Xenon2 Phase 9 — Make Sloth fact-extraction fire-and-forget

## Background

You're working on **Xenon2**, a portable, offline-first desktop AI assistant (Tauri 2 + Vue 3
front end, Rust backend, RWKV-7 2.9B via a C++ wrapper around `rwkv.cpp`). Read `PLAN.md` and
`app/README.md` first for full context — Phases 1–8 are complete. This is a **latency phase**,
not new user-facing scope, closing one specific finding from a performance audit. **Do not
re-derive it — it was already measured.**

---

## The problem

`app/src-tauri/src/inference.rs`'s `generate_blocking` (around line 645) runs Sloth's
fact-extraction pass **synchronously**, before it emits `generation-done`:

```rust
if status == ffi::XenonStatus::OK {
    if is_sloth {
        let reply_text = ctx.full_text.borrow();
        run_fact_extraction(engine, &inner.extraction_prefix, &app, &new_user_turn.content, &reply_text);
    }
    let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id });
    ...
```

`run_fact_extraction` (around line 447) calls `xenon_generate_with_state` a second time, under
the *same already-held* `EngineInner` mutex — the one `generate_response`'s
`tauri::async_runtime::spawn_blocking` closure locked for "the entire blocking generate() call...
and, for Sloth, the extraction call right after it" (see that closure's own doc comment, line
~784). Nothing about the reply the user is looking at depends on this second generation — it only
persists a fact for *future* turns — yet it holds the lock and delays `generation-done`, and it
delays the *next* turn too, since that turn's `generate_response` call blocks on the same mutex
until extraction finishes.

Measured cost of the extraction call itself (separate from the blocking/ordering problem above):
up to 40 tokens generated, at whatever this model's steady-state tok/s is. Phase 8 already
eliminated the *preamble* cost that used to dominate this (`EngineInner::extraction_prefix` is
prefilled once at engine load — see `inference.rs:98` and `inference-engine/README.md`'s
"Persistent state across calls" section), so this phase is specifically about the ordering/
locking problem, not the extraction prompt's token cost.

**Goal:** `generation-done` fires as soon as the visible reply is ready, full stop. Fact
extraction happens afterward, on its own schedule, competing for the engine lock like any other
call rather than being smuggled in under a lock the *next* turn is already waiting on.

---

## Non-obvious traps — read these before touching the code

1. **`generate_blocking` currently borrows `&mut EngineInner` (already locked by its caller), not
   the `Arc<Mutex<EngineInner>>` itself.** To make extraction genuinely fire-and-forget (running
   under a lock acquired *later*, not the one already held), it can't just be moved to "after the
   emit" inside the same function call — it needs to happen in a **separate task that acquires its
   own lock**, after the current lock is released. That means either:
   - `generate_blocking` returns something like `Option<{ user_text, reply_text }>` instead of
     calling `run_fact_extraction` itself, and `generate_response`'s `spawn_blocking` closure
     drops its guard, then spawns a second `tauri::async_runtime::spawn_blocking` (with its own
     `engine_arc.clone()`) to run extraction; or
   - `generate_blocking` itself takes the `Arc<Mutex<EngineInner>>` (in addition to/instead of the
     already-locked guard) so it can spawn the follow-up task before returning.

   Pick one and be explicit about why. The first is probably simpler to reason about (no nested
   locking inside a function that's supposed to just be "the blocking part of one call").

2. **Extraction must still never touch `conversation_cache`.** Phase 8 already isolated this
   (`run_fact_extraction` only ever touches `extraction_prefix`, never a conversation's cached
   state), so making it run later, under a separately-acquired lock, doesn't introduce a new
   cache-correctness risk — but *verify* this stays true rather than assuming it, since the whole
   point of this phase is changing when/how the lock is held around it.

3. **The next turn can now genuinely race the still-pending extraction for the engine lock.**
   That's the intended fix (the lock is available for the next `generate_response` call
   immediately after `generation-done`, instead of only after extraction finishes too) — but
   confirm nothing about extraction's correctness depends on running before the next turn starts.
   It shouldn't: `run_fact_extraction` closes over its own `user_text`/`reply_text` (owned
   `String`s, not borrows into `history` or the conversation cache), so it has everything it needs
   independent of what the next turn does. Make sure whatever refactor you land on keeps that true
   (i.e. don't accidentally pass borrowed data into the background task that could be invalidated
   by the next turn running first).

4. **Mutex poisoning discipline.** The existing lock-acquisition pattern
   (`Err(poisoned) => poisoned.into_inner()`) needs to be duplicated for the new, second lock
   acquisition in the background task — a panic inside extraction shouldn't brick the engine for
   every subsequent turn.

5. **`run_fact_extraction`'s own error handling is already "never surface as `generation-error`"**
   (see its doc comment) — keep that. This phase changes *when* it runs, not what it does on
   failure. A failed/garbled extraction should still just mean no fact gets remembered, silently.

6. **`AppHandle` is cheap to clone** (already done once per call via `ctx.app`); clone it again
   for the background task rather than trying to thread a reference through.

---

## Suggested approach

Not mandated — if you find a cleaner shape, take it and say why. But roughly:

```rust
// generate_blocking no longer calls run_fact_extraction directly. On a successful Sloth reply,
// it returns the data extraction will need; the caller decides when/how to run it.
fn generate_blocking(...) -> Option<PendingExtraction> { ... }

struct PendingExtraction {
    user_text: String,
    reply_text: String,
}

// generate_response's spawn_blocking closure:
tauri::async_runtime::spawn_blocking(move || {
    let mut guard = engine_arc.lock()...;
    let pending = generate_blocking(&mut guard, app.clone(), conversation_id, history, agent);
    drop(guard); // release before scheduling extraction -- this is the actual fix
    if let Some(job) = pending {
        let engine_arc2 = engine_arc.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let mut guard2 = engine_arc2.lock()...;
            run_fact_extraction(guard2.engine.0, &guard2.extraction_prefix, &app, &job.user_text, &job.reply_text);
        });
    }
})
```

`generate_blocking` itself still emits `generation-done`/`generation-error` exactly where it does
today — the only change is that the Sloth-extraction call is replaced with building a
`PendingExtraction` and returning it, so `generation-done` is no longer preceded by a second
blocking generate call under the same lock.

---

## Measurement required

Report before/after, measured against the real dev build and/or a small harness, not estimated:

- Wall-clock time from the user's message being sent to `generation-done` firing, for a Sloth-mode
  turn, before and after this change (should drop by roughly the extraction call's own duration).
- Confirm a second `generate_response` call issued immediately after a Sloth turn's
  `generation-done` no longer waits on that turn's extraction to finish (e.g. instrument timestamps
  around the lock acquisition, or just observe token-stream start time for turn N+1).
- Confirm `sloth-facts-updated` still fires and the fact still lands in `sloth_memory.json`,
  just later (after `generation-done`, not before it) — i.e. this is a reordering/latency fix,
  not a change in what gets remembered.

---

## Explicitly NOT in scope

- Finding 03 (the `sample_logits` hot-loop rewrite) — separate phase, separate file.
- Any change to Phase 8's cache diff logic (`ensure_cache_extended`), the static/extraction
  preamble states, or the prompt-ordering decision — this phase only changes *when* the extraction
  call runs and how its lock is acquired, not what it evaluates or why.
- Any change to `run_fact_extraction`'s prompt text, few-shot examples, or echo/None-guard logic.
- Any UI change — `sloth-facts-updated` still carries the same payload to the same listener.

---

## Deliverables

1. `generate_blocking` no longer runs extraction under its caller's already-held lock;
   `generation-done` fires as soon as the visible reply is ready.
2. Extraction runs in a separate task that acquires the engine lock on its own, after the main
   call has released it.
3. Measured before/after numbers from the section above, in the commit description.
4. Update `app/README.md`'s Phase 7/8 sections (or add a short Phase 9 note, matching the existing
   per-phase write-up style) to describe the new ordering.
5. One commit, scoped to this change only.
