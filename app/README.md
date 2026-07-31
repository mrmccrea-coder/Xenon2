# Xenon2 — Phase 3/4/5: Desktop Shell, Chat UI, Editing, and File Save/Load

A Tauri 2 + Vue 3 desktop chat app: sidebar + main chat panel, ChatGPT/Claude-Desktop-style
layout. Phase 3 (see `../PLAN.md` and `../prompts/phase3_prompt.md`) built the shell, layout, and
typed-text-in / streamed-text-out wiring against Phase 1's inference engine. Phase 4 (see
`../prompts/phase4_prompt.md`) migrated conversation state into a Pinia store and wired up real
edit-user-message and regenerate-assistant-message flows. Phase 5 (see
`../prompts/phase5_prompt.md` and `../SCHEMA.md`) made conversations durable: File > Save / Save
As / Open now do real file I/O, and every completed generation auto-saves to disk. No voice yet —
that's Phase 2 (already built standalone)'s integration, still pending.

## Running the dev build

Prerequisites: Phase 1 must already be built (`inference-engine/build-cpu-app`, see
`../inference-engine/README.md`) and the quantized model must exist at
`../models/rwkv-5-world-0.4B-Q4_0.bin`. Node.js, Rust/cargo, and the Tauri CLI must be on PATH.

From `app/`:

```powershell
npm install        # first time only
npm run tauri dev
```

The first build compiles the Rust/Tauri dependency tree from scratch and can take several
minutes; subsequent builds are incremental and fast. This launches a real desktop window (not a
browser tab) — Vite serves the Vue frontend on `http://localhost:1420` internally, but you
interact with the native WebView2 window `npm run tauri dev` opens.

To build a release bundle instead: `npm run tauri build`.

## What's fully functional vs. intentional stubs

Per `prompts/phase3_prompt.md`, `prompts/phase4_prompt.md`, and `prompts/phase5_prompt.md`'s
explicit scope boundaries, a couple of UI elements remain deliberate visual placeholders for later
phases — present and clickable, but not wired to real logic. Nothing below was skipped by
accident; each remaining stub logs to the browser devtools console (`console.log`) so it's easy to
confirm nothing silently does real work.

| Piece | Status | Notes |
|---|---|---|
| Typing a message + Send button | **Functional** | Full round trip through Rust into Phase 1's `xenon_generate()`, streamed back token-by-token. |
| "+ New Chat" (Sidebar) / File > New | **Functional** | Both call the same `newChat()` Pinia store action (`stores/chat.ts`) — clears the main panel and starts a fresh, empty conversation, tagged with the currently-loaded model's name. |
| Sidebar conversation list (Today/Yesterday/Older grouping, click to load) | **Functional** | Backed by durable storage as of Phase 5 (see below) — the sidebar is rebuilt from disk on every launch, not just held in memory. |
| **Edit icon on user messages** (`ChatMessage.vue`) | **Functional (Phase 4)** | Turns the bubble into an editable textarea (✓ confirm / ✕ cancel, or Enter/Escape). Confirming truncates the conversation after that message, updates its content, sets `edited: true`, and streams a fresh assistant reply from the edited text. See "Edit and regenerate flows" below. |
| **Regenerate icon on assistant messages** (`ChatMessage.vue`) | **Functional (Phase 4)** | Re-runs generation using history up to (not including) that message and replaces only that message's content in place, regardless of its position in the conversation. |
| **Delete icon on user messages** (`ChatMessage.vue`) | **Stub — unowned** | `console.log` only, exactly as Phase 3 left it. Explicitly out of scope for Phase 4 *and* Phase 5 — this is a real gap in `PLAN.md`'s task list, not an oversight. No phase currently owns implementing real delete. |
| Mic toggle button (`MessageInput.vue`) | **Stub** | Visually toggles on/off (turns red), but captures no audio. Phase 2's STT pipeline (already built standalone under `../voice-pipeline/`) gets wired into this button in a later integration step, not in this phase. |
| **File > Open / Save / Save As** | **Functional (Phase 5)** | Real native file dialogs (via `rfd`, called directly), real JSON read/write on disk, real schema validation with clear errors on malformed files. See "Phase 5: file persistence" below. |
| File > Export Memory | **Stub** | Present, `console.log` only. Real copy-to-external-drive logic is **Phase 6**'s job. |

## State management: the Pinia store (`src/stores/chat.ts`)

As of Phase 4, `conversations`/`activeId`/`generating` no longer live as local `ref`s in
`App.vue` — they live in `useChatStore()` (Pinia), which is the single source of truth for all
conversation/message state. `App.vue` reads/writes through the store instead of owning the data
itself; it still owns Tauri event-listener registration (`token-stream` / `generation-done` /
`generation-error`) and DOM-only concerns (scroll-to-bottom) that don't belong in a store.

`ChatMessage` (`src/types.ts`) gained two fields for this phase:
- `timestamp: number` — epoch ms set once when the message is created. Left alone on user edits
  (it records when the turn was first created); a regenerated assistant message's `timestamp` is
  updated, since regeneration produces an effectively new reply.
- `edited?: boolean` — set to `true` on a user message once it's been edited. Rendered as a small
  "(edited)" tag under the bubble.

The store does **not** assume "the currently-streaming message is always the last one in the
array" (true for a normal send, but false for a regenerate targeting an earlier message).
Instead it tracks `streamingMessageIds: Record<conversationId, messageId>` — set right before a
`generate_response` call, consulted by the `token-stream`/`generation-done`/`generation-error`
handlers to find the exact message to mutate by id. This is what makes regenerating message #2 of
5 update message #2 in place without touching #1, #3, #4, or #5.

## Edit and regenerate flows

- **Edit a user message** (`useChatStore().editUserMessage`): truncates that conversation's
  `messages` array to drop everything after the edited message (its old reply and anything past
  it), updates its `content`, sets `edited = true`, pushes a new empty streaming assistant
  message, and re-runs the same generation flow `sendMessage` uses — reconstructing `history` from
  the truncated array through the edited message and calling `generate_response`.
- **Regenerate an assistant message** (`useChatStore().regenerateAssistantMessage`): rebuilds
  `history` up to (not including) the target message, clears just that message's `content`
  (`streaming = true`), and streams a new reply into it in place — every other message, before or
  after it, is left untouched.
- Both flows (and `sendMessage`) guard on the store's `generating` flag before starting, and the
  send/edit-confirm/regenerate controls are all disabled while `generating` is true — the Rust
  backend serializes all `generate_response` calls through a single mutex anyway (see
  `inference.rs`), so one app-wide flag (rather than per-conversation) is sufficient to prevent
  overlapping calls. Verified empirically: rapid-firing the send button three times in a row while
  a generation was in flight produced exactly one new user/assistant message pair, not three.
- "Persist edits immediately" falls out of the Pinia store being the single source of truth —
  `ChatMessage.vue` has no local copy of message content that could go stale; it reads `message`
  directly from the array the store owns and mutates through emitted events, never a local clone.

## Phase 5: file persistence

Full schema documentation lives in `../SCHEMA.md` at the repo root — this section covers the
implementation, not the on-disk shape itself.

### Rust side (`app/src-tauri/src/persistence.rs`)

New module, registered as a set of `#[tauri::command]`s in `lib.rs`:
- `save_conversation_file(path, conversation)` / `open_conversation_file(path)` — plain
  `std::fs` + `serde_json` read/write to an exact, already-known path. No Tauri fs plugin needed:
  these are trusted backend commands operating on a path either chosen via the native dialog or
  computed by the app itself, not an arbitrary path from untrusted web content. Writes are
  pretty-printed (`serde_json::to_string_pretty`), not minified. Reads validate strictly: missing
  fields, wrong types (e.g. `role` outside `"user"`/`"assistant"`), or an unrecognized
  `schemaVersion` all produce a specific, human-readable `Err(String)` rather than a panic — see
  `SCHEMA.md`'s "Validation on Open" section for exact message shapes.
- `default_conversation_path(conversation_id)` — computes
  `<app-data-dir>/conversations/<id>.json` via Tauri's `app_data_dir()`, creating the directory if
  needed. This is the auto-save default (see below).
- `load_session_file` / `save_session_file` — read/write a small `session.json` in the app-data
  dir (NOT part of the conversation schema) that remembers `lastActiveConversationId` and every
  known conversation's current path, so a relaunch can rebuild the sidebar and restore the
  last-active conversation without the user clicking Open.
- `pick_save_dialog(default_file_name)` / `pick_open_dialog()` — call `rfd::FileDialog` **directly**,
  not through `tauri-plugin-dialog`'s `DialogExt`. See "A real bug found and fixed" below for why.

### Frontend side

- `app/src/persistence.ts` — the TS mirror of the on-disk schema (`ConversationFile`,
  `SessionFile`) plus pure conversion helpers (`toConversationFile` / `fromConversationFile`)
  between it and the in-memory `Conversation`/`ChatMessage` types. `streaming`/`errored` are
  dropped when serializing (see `SCHEMA.md`) since a persisted message is never mid-stream or a
  failed placeholder — auto-save only ever fires after a *successful* completed generation.
- `app/src/stores/chat.ts` gained: `filePaths` (conversationId → last-used path), `modelName`
  (fetched once at startup via a new `get_model_name` command, see `inference.rs`), `fileError`
  (surfaced in `App.vue` as a dismissable red banner, separate from the existing inference-error
  banner), and actions `saveConversation` / `saveConversationAs` / `openConversation` / `autoSave`
  / `restoreSession` / `persistSession`. `completeGeneration` now ends with `void this.autoSave(...)`
  — fire-and-forget so a disk write never blocks the UI thread on a just-finished reply.
- `app/src/types.ts`'s `Conversation` gained an optional `model?: string` field (task 5's
  requirement), populated from `modelName` on every `newChat()` and carried through save/load.
- `MenuBar.vue`: Open / Save / Save As are now real (emit `open` / `save` / `save-as`, wired in
  `App.vue` exactly like `New` already was); Save / Save As are disabled (visually + functionally)
  when there's no active conversation.

### Auto-save default path policy

See `SCHEMA.md`'s "Auto-save default path policy" section for the full writeup. Short version: a
conversation that's never been manually saved auto-saves to
`<app-data-dir>/conversations/<conversation-id>.json` (on this Windows dev machine, resolves under
`%APPDATA%\com.xenon2.app\`) on its first completed generation, rather than prompting — a save
dialog interrupting the user right as their first reply finishes would be disruptive and easy to
accidentally cancel, defeating the "never silently lose history" goal. File > Save As always
opens a dialog and, once a path is chosen, every later auto-save for that conversation follows the
new path instead.

### A real bug found and fixed: `tauri-plugin-dialog`'s dialog never appeared

Initial implementation used `@tauri-apps/plugin-dialog` / `tauri-plugin-dialog`'s `DialogExt`
(`app.dialog().file()...blocking_save_file()` / `.save_file(callback)`). On this dev machine, in
this app, **the native dialog never appeared at all** — not slow, not erroring, just silently
never showing a window — regardless of whether it was called from a sync command, an `async`
command, or wrapped in `spawn_blocking`. Confirmed via `EnumWindows` (no dialog HWND ever created)
and via Chrome DevTools Protocol (`invoke('pick_save_dialog', ...)` left permanently pending, no
resolution, no rejection). The fix was to drop `tauri-plugin-dialog` entirely and call `rfd`
directly (`Cargo.toml`: `rfd = "0.17.2"`, no more `tauri-plugin-dialog` dependency, no more
`dialog:default` capability) — `rfd::FileDialog::save_file()` on Windows just calls the Common
Item Dialog COM API directly on the calling thread with no main-thread requirement (see rfd's
`win_cid/file_dialog.rs`), unlike the plugin's wrapper which routes through
`AppHandle::run_on_main_thread`.

### A separate, machine-level issue: the Windows Common Item Dialog itself hangs on this dev machine

After fixing the above, `rfd`'s dialog call still never returned on this specific dev machine.
Independently reproduced with code that has nothing to do with Xenon2 or Tauri: a plain
`System.Windows.Forms.SaveFileDialog` invoked from an unrelated PowerShell process hung
indefinitely (still not responded after 15+ seconds, vs. a plain `MessageBox` on the same machine
appearing instantly). This points to a hung/slow Windows shell extension (a cloud-sync client,
a corrupted "Quick access" list, etc.) that the modern Explorer-style picker enumerates while
opening — a well-documented class of real-world Windows issue, unrelated to Xenon2's code.
Two things were done about it:
1. **Shipped**: `persistence.rs`'s `pick_save_dialog`/`pick_open_dialog` now wrap the dialog call
   in a 30-second timeout (`DIALOG_TIMEOUT`), so a hung shell extension produces a clear,
   dismissable error ("The save dialog did not respond within 30 seconds...") instead of an
   indefinitely frozen-looking UI. This is a real, permanent hardening, not a workaround.
2. **Test-only**: an opt-in escape hatch (`XENON2_TEST_SAVE_DIALOG_PATH` /
   `XENON2_TEST_OPEN_DIALOG_PATH` env vars, checked first in each command, `test_override_path` in
   `persistence.rs`) returns a fixed path immediately instead of invoking the OS dialog. Unset (the
   default) in normal use — this is what let the rest of the Save As / Open pipeline (path
   bookkeeping, writing, schema validation, auto-save switching to the new path) be exercised
   through the real running app and real UI clicks on this machine, with only the OS picker step
   substituted. Documented here rather than hidden so a future maintainer knows it exists and why.

### Phase 5 verification (what was actually run, not just code-reviewed)

Real dev build (`npm run tauri dev`), real compiled Phase 1 model, real generated replies — driven
via UI Automation (`InvokePattern`/`ValuePattern`, focus-independent) and, for the two checks that
needed to bypass the confirmed-broken OS dialog (see above), via the real Pinia store actions
called through Chrome DevTools Protocol against the live running app (`window.__TAURI_INTERNALS__`
/ a temporarily-exposed store reference, removed again afterward) — not simulated, not unit-tested
in isolation:

- **Auto-save to the default path**: sent a real message, got a real streamed reply, confirmed
  `%APPDATA%\com.xenon2.app\session.json` and `...\conversations\<id>.json` were written
  automatically with no user action, pretty-printed and human-readable, `model` field correctly set
  to `"rwkv-5-world-0.4B-Q4_0"`.
- **Close and reopen restores the last-active conversation automatically**: killed the app process,
  relaunched from scratch, confirmed the sidebar and chat panel repopulated with the exact same
  conversation and messages with zero manual clicks, before wiring any event listeners.
- **Save As chooses a new path, and subsequent auto-saves follow it**: ran `saveConversationAs`
  against a path outside the app-data directory, confirmed the file was written there
  (`filePaths` updated); sent a second real message; confirmed the *new* path grew to 4 messages
  while the original app-data-default file stayed frozen at 2 — auto-save correctly followed the
  relocated path, not the original default.
- **Opening a malformed / non-Xenon2 file fails clearly, not a crash**: tested both "not valid JSON
  at all" (syntax error) and "valid JSON but missing `schemaVersion`/wrong shape" — both produced
  specific error text (see `SCHEMA.md`'s "Validation on Open"), both left the app's existing
  conversation list completely untouched (`convCountBefore === convCountAfter`), both rendered as
  the dismissable red file-error banner in the real UI, confirmed via screenshot.
- **Opening a valid file loads it correctly**: created a fresh empty chat, then opened a
  previously-saved 4-message conversation file; confirmed it replaced the active conversation
  (correct id, correct message count, correct `model` field) and appeared in the sidebar alongside
  the still-present empty chat, not duplicated.

## How the Tauri IPC command connects to Phase 1

- `app/src-tauri/build.rs` links this crate directly against Phase 1's pre-built
  `inference-engine/build-cpu-app` (`xenon_inference.lib`/`.dll`) and copies `xenon_inference.dll`,
  `rwkv.dll`, and the MSVC/UCRT redistributable DLLs that build already collected next to
  `test_inference.exe`, so the app can load them at runtime without editing PATH. It deliberately
  links the **CPU-only** build, not `build-cuda-app` — Phase 1's own benchmarks found CPU-only
  faster for steady-state throughput at this model size, and it avoids requiring a CUDA toolkit on
  whatever machine runs this app.
- `app/src-tauri/src/ffi.rs` is a hand-written `extern "C"` binding mirroring
  `inference-engine/src/xenon_inference.h` exactly (opaque `xenon_engine*`, `xenon_load_model`,
  `xenon_generate` with its C callback, etc.) — chosen over `bindgen` since the header is small and
  intentionally stable.
- `app/src-tauri/src/inference.rs`:
  - Loads the model **once** at app startup (`tauri::Builder::setup`), from
    `../models/rwkv-5-world-0.4B-Q4_0.bin` and `../inference-engine/data/world_vocab.bin`
    (resolved relative to the repo root via `CARGO_MANIFEST_DIR`), and keeps the engine pointer in
    Tauri-managed state behind a `Mutex` (only one `xenon_generate` call may run at a time, since
    rwkv.cpp's per-engine state isn't designed for concurrent eval calls).
  - Exposes `#[tauri::command] generate_response(conversation_id, history)`, called from the
    Vue frontend via `invoke("generate_response", ...)` in `App.vue`'s `sendMessage()`. It builds
    the same "User: ... / Xenon: ..." chat-style prompt prime Phase 1's CLI harness uses (extended
    to include the full turn history, since `xenon_generate` resets RWKV state at the start of
    every call — there's no cross-call incremental state yet), then runs the actual blocking
    `xenon_generate()` call inside `tauri::async_runtime::spawn_blocking` so it doesn't block the
    Tauri event loop.
  - The token callback (`on_token`, a real `extern "C" fn`, not a stub) emits a Tauri event
    `token-stream` (payload `{ conversationId, text }`) for every non-empty decoded chunk as
    `xenon_generate` produces it — this is what drives the incremental, token-by-token rendering
    in the chat UI, not a single batched IPC response. `generation-done` / `generation-error`
    events signal completion.
  - Since Phase 5: also derives `ModelInfo` (the loaded model's name/version, e.g.
    `"rwkv-5-world-0.4B-Q4_0"`) from the same `model_path` used to load the engine, so it can never
    drift from what's actually running, managed as Tauri state alongside `EngineState` and exposed
    via `#[tauri::command] get_model_name`. The frontend fetches this once at startup
    (`stores/chat.ts`'s `initModelName`) and records it into every conversation's `model` field.
- `App.vue` listens for these three events (`@tauri-apps/api/event`'s `listen()`), appends
  incoming text directly onto the streaming assistant message's `content` as it arrives, and
  trims the trailing `\n\nUser:` stop-sequence text once generation completes (the model, like
  Phase 1's CLI harness, naturally continues into a fake next turn before the stop-sequence check
  fires — that already-streamed tail is stripped from the rendered bubble only, not from what was
  sent to the model).

This was verified against the real compiled Phase 1 library and the real quantized model, not
stubbed: launching the dev build and sending a message produces an actual RWKV-generated reply,
visibly appearing token-by-token.

Phase 4's edit/regenerate flows were verified the same way (real dev build, real model, not
code-reviewed only): a 6-message conversation had its first user message edited, which correctly
truncated the other 5 messages and streamed a fresh reply (2 messages remained, `edited: true`
shown); a 4-message conversation had its first (non-last) assistant message regenerated, and
comparing a JSON dump of all four messages before and after confirmed the other three were
byte-for-byte unchanged while only the targeted message's content changed.
