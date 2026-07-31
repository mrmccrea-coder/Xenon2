# Prompt: Xenon2 Phase 5 — Project File Save/Load

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context on
all 6 phases, but you are only doing **Phase 5** right now.

**Phases 1-4 are already complete and verified working** (not just marked
done in the plan — actually built, run, and tested):
- Phase 1: `inference-engine/` — RWKV inference core (`xenon_generate()`),
  both CPU-only and CUDA builds exist. The desktop app links the CPU-only
  build (see Phase 3/4's README for why).
- Phase 2: `voice-pipeline/` — standalone STT/VAD/TTS pipeline, not yet
  wired into the desktop app's mic button.
- Phase 3: `app/` — Tauri + Vue 3 desktop shell, sidebar + chat panel,
  typed-text-in / streamed-text-out working end to end against Phase 1.
- Phase 4: real edit-user-message and regenerate-assistant-message flows,
  built on a Pinia store (`app/src/stores/chat.ts`) that is the single
  source of truth for conversation state.

Read `app/README.md` before starting — it documents exactly what's real
vs. stubbed as of Phase 4, including the current conversation data model.

### Current data model (already built — do not redesign it)

`app/src/types.ts` currently defines:

```typescript
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  errored?: boolean;
  edited?: boolean;
  timestamp: number; // epoch ms
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number; // epoch ms
  messages: ChatMessage[];
}
```

Conversations currently live only in the Pinia store (`stores/chat.ts`),
in memory — closing the app loses everything. Your job is to make this
durable by writing it to disk and reading it back, not to change the
shape of the data itself. If you find a genuine reason the model needs to
change (e.g. a field that can't be serialized cleanly), extend it rather
than redesigning it, and document why in your README update.

### A known, deliberately unresolved gap — do not fix it here

The **delete-message** feature (icon exists on `ChatMessage.vue`, but is a
`console.log` stub) is not owned by any phase — this was surfaced during a
build-progress review and the decision was made to **leave it as a stub
for now**. Do not implement real delete logic as part of this phase; it's
out of scope and was explicitly deferred, not forgotten.

### Why this project is unusual (brief context, not directly relevant here)

This project uses RWKV, a non-transformer architecture, for inference —
relevant to Phases 1/2/3/4, but largely irrelevant to this phase. Phase 5
is standard file I/O; nothing about RWKV changes how you read/write JSON
to disk. The one place it's worth remembering: the saved file should
record which model the conversation was generated with (see task 5 below),
since a future non-RWKV backend (see "Future work" in PLAN.md) would need
to know not to naively reload an incompatible conversation against the
wrong engine.

---

## Your task: Phase 5 — Project File Save/Load

Working directory: `C:\Users\New user\Xenon2\`

### Tasks

1. Define the on-disk conversation JSON schema in a new `SCHEMA.md` file
   at the project root. Base it directly on the existing `Conversation`/
   `ChatMessage` TypeScript types shown above — don't invent a different
   shape. Include the model name/version field from task 5 below in the
   schema.
2. Implement **File > Save** and **File > Save As** in `MenuBar.vue`
   (currently stubs — see `app/README.md`'s stub table): open a Tauri
   save-file dialog, serialize the active conversation from the Pinia
   store to JSON matching the schema, and write it to the chosen path.
3. Implement **File > Open**: open a Tauri open-file dialog, read the
   chosen `.json` file, validate it against the schema (reject/report
   clearly on a malformed file rather than crashing), and load it into
   the Pinia store, populating the sidebar and chat panel.
4. Implement auto-save: after every completed message exchange (send,
   edit-triggered regeneration, or manual regenerate — anywhere the Pinia
   store's conversation content changes as a result of a completed
   generation), write to the last-used file path automatically, so a
   crash doesn't lose history. If no path has been established yet for a
   given conversation (never manually saved), auto-save should not
   silently invent a location — decide and document a sensible default
   (e.g. prompt on first auto-save, or save to a default app-data
   directory keyed by conversation id) rather than leaving conversations
   permanently unsaved until the user manually saves once.
5. Store the model name/version inside the saved file (e.g.
   `"model": "rwkv-5-world-0.4B-Q4_0"`) so a reopened project records
   which quantized model generated it. This phase does not need to
   implement model-switching or validation that the currently-loaded
   model matches — just record it faithfully.

### Acceptance criteria (verify before considering this phase done)

- Closing and reopening the app restores the most recently active
  conversation without the user needing to manually click Open.
- A saved `.json` file, opened in a plain text editor, is human-readable
  (sensible field names, not minified, reasonable indentation).
- File > Save As lets the user choose a new filename/location distinct
  from the auto-save path, and subsequent auto-saves for that
  conversation follow the new path.
- Opening a malformed or non-Xenon2 `.json` file fails with a clear error
  message rather than a silent crash or corrupted UI state.
- Verify this against the real running app and real generated
  conversations — not just unit tests against the schema in isolation.
  Follow the same verification standard Phases 1-4 used (see their
  READMEs): actually run the dev build, actually save/close/reopen, and
  document what you observed, not just what you implemented.

### When finished

Update the Phase 5 heading in `C:\Users\New user\Xenon2\PLAN.md` to mark
it complete, update `app/README.md`'s stub table (Open/Save/Save As move
from "Stub" to "Functional"), and document in `SCHEMA.md` and/or
`app/README.md` how auto-save decides where to write when no path exists
yet, and what was actually tested.
