# Prompt: Xenon2 Phase 4 — Message Editing & Regeneration

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context on
all 6 phases, but you are only doing **Phase 4** right now.

**Phase 3 is already complete and committed** (git commits `0ab8a58`,
`3da3634`, `f6af1ee` for Phases 1-3). The Tauri + Vue 3 app under `app/`
already has a working chat shell: `Sidebar.vue`, `ChatMessage.vue`,
`MessageInput.vue`, `MenuBar.vue`, and an `App.vue` that owns in-memory
conversation state and streams tokens from Phase 1's inference engine via
Tauri events. Read these existing files in full before changing anything —
this phase extends them, it does not replace them:

- `app/src/App.vue` — currently owns `conversations`/`activeId` as plain
  Vue `ref`/`computed` state (no Pinia yet), plus `sendMessage(text)`
  which pushes a user message + an empty streaming assistant message, then
  calls the `generate_response` Tauri command with `{ conversationId,
  history }` where `history` is the full message array up to that point
  mapped to `{role, content}`. Token/done/error events are listened for
  and mutate the streaming assistant message in place.
- `app/src/components/ChatMessage.vue` — renders one bubble; currently has
  **stub-only** edit/delete (user messages) and regenerate (assistant
  messages) icon buttons that just `console.log` and do nothing else
  (search for `stubEdit`/`stubDelete`/`stubRegenerate`). This phase wires
  edit and regenerate up to real behavior (see explicit scope note below
  on delete).
- `app/src/types.ts` — current `ChatMessage` interface has `id, role,
  content, streaming?, errored?`; `Conversation` has `id, title,
  createdAt, messages`. PLAN.md's Phase 4 spec calls for messages to also
  carry `edited` and `timestamp` fields — add these.
- `app/package.json` — no state-management library is installed yet.
  PLAN.md's Phase 4 spec explicitly calls for the conversation data model
  to live in **Pinia** app state — add `pinia` as a dependency and
  actually migrate the conversation/message state out of `App.vue`'s
  local refs into a proper Pinia store (e.g. `app/src/stores/chat.ts`).
  This is a real refactor, not just a note — don't leave the state in
  `App.vue` and call it "Pinia-shaped."

### Why this project is unusual (recap, not new information)

RWKV (Phase 1) streams tokens one at a time and — this is the important
part for THIS phase — **`xenon_generate` resets the model's internal RWKV
state on every single call**. There is no persistent multi-turn state
carried between calls at the engine level. Phase 3 already solved this by
sending the *entire* conversation history as the prompt on every call (see
`app/src-tauri/src/inference.rs` and how `App.vue`'s `sendMessage` builds
`history`). **Phase 4 must preserve this exact mechanism**: when you
truncate history for an edit, or reconstruct history for a regenerate, you
are changing what array of `{role, content}` turns gets sent as the prompt
on the next `generate_response` call — you are not trying to manipulate
any persistent state inside the inference engine itself, because there
isn't any across calls.

### Target hardware (context only, not relevant to this phase's work)

CPU: Intel i7-12850HX (16c/24t), 32GB RAM. GPU: NVIDIA RTX A1000, ~4GB
VRAM. This phase is pure frontend/state-management work calling the
already-working `generate_response` IPC command — no new inference-engine
or hardware-specific work is needed.

---

## IMPORTANT — explicit scope boundary

- **Delete is NOT one of this phase's four tasks.** PLAN.md's Phase 4 task
  list is exactly: (1) data model, (2) edit user message, (3) regenerate
  assistant message, (4) persist edits in-memory. It does **not** include
  wiring up the delete icon, even though Phase 3 grouped edit/delete/
  regenerate together as "stubs for Phase 4" in its own scope note. This
  is a real gap in the plan, not an oversight you should silently fix —
  **leave the delete button exactly as the stub Phase 3 left it**
  (`stubDelete`, console.log only). Do not implement real delete logic.
  Flag this gap plainly in your final report so the user can decide which
  later phase (or a dedicated one) should own it.
- **Do not touch file save/load** (Phase 5) or **external export/import**
  (Phase 6). "Persist edits to the in-memory conversation state
  immediately" in task 4 means the Pinia store's in-memory state, not
  anything written to disk.
- **Do not touch voice/STT/TTS** (Phase 2's `voice-pipeline/`) or the mic
  button in `MessageInput.vue` (still Phase 2-integration's job, not
  this phase's).
- **Do not modify the `xenon_generate`/`xenon_inference` C API or the Rust
  FFI layer** (`app/src-tauri/src/ffi.rs`, `inference.rs`) unless you find
  an actual bug while wiring this up — the existing `generate_response`
  command taking `{conversationId, history}` already supports everything
  this phase needs (arbitrary history arrays), so this should be pure
  frontend work plus the Pinia migration.

---

## Your task: Phase 4 — Message Editing & Regeneration

Working directory: `C:\Users\New user\Xenon2\app\`

### Tasks

1. **Data model + Pinia store**: add `pinia` to `app/package.json`,
   install it in `app/src/main.ts`, and create a Pinia store that holds
   `conversations: Conversation[]` and `activeId: string | null` (move
   this out of `App.vue`). Extend `ChatMessage` in `types.ts` with
   `edited?: boolean` and `timestamp: number` (set on every message when
   created). Keep `App.vue`'s existing responsibilities (Tauri event
   listeners, scroll-to-bottom, etc.) but have it read/write through the
   store instead of local refs.
2. **Edit a user message**: clicking the edit icon on a user bubble turns
   it into an editable text field pre-filled with the current content
   (add a reasonable way to confirm vs. cancel — e.g. Enter/a checkmark to
   confirm, Escape/a cancel button to back out without changes — Phase 3
   didn't specify UI details here, use your judgment but keep it simple
   and consistent with the existing bubble styling). On confirm:
   - Truncate that conversation's `messages` array so nothing after that
     message's index remains (the edited message becomes the new last
     user message).
   - Update its `content` to the new text and set `edited = true`.
   - Push a new empty streaming assistant message and re-run the same
     generate flow `sendMessage` already uses (reconstruct `history` from
     the truncated array through the edited message, call
     `generate_response`), so a fresh response is produced from the
     edited text.
3. **Regenerate an assistant message**: clicking regenerate on an
   assistant bubble re-runs `generate_response` using the history up to
   (not including) that message, and **replaces only that specific
   message's content in place** — do not touch any other message before
   or after it, even if the regenerated message isn't the last one in the
   conversation (this is what the acceptance criteria below means by
   "nothing else" — implement it literally, don't infer that later
   messages should also be discarded).
4. **Persist edits to in-memory state immediately**: this should fall out
   naturally from steps 1-3 if the Pinia store is the single source of
   truth — verify there's no lag or stale-copy issue (e.g. a local
   component-level copy of message content that doesn't sync back to the
   store immediately on edit).

Guard both edit-submit and regenerate against firing while a generation is
already in flight for that conversation (reuse/extend whatever the
existing `generating` flag pattern does in `App.vue`) — don't allow two
concurrent `generate_response` calls to race against the same conversation
state.

### Acceptance criteria (verify before considering this phase done — actually launch the dev build and interact with it, don't just review code)

- Editing an earlier user message and confirming it removes all messages
  after that message (both the old assistant reply and anything past it)
  and produces a new, real streamed assistant response based on the
  edited text.
- Clicking regenerate on an assistant message produces a new streamed
  response that replaces only that message's content — every other
  message in the conversation (before and after it) is confirmed
  unchanged.
- Both flows still show token-by-token incremental streaming (this
  phase must not regress Phase 3's streaming behavior).
- The Pinia migration doesn't break anything Phase 3 already verified:
  app launch (empty sidebar/chat panel), typed send flow, "+ New Chat" /
  File > New reset behavior, and the still-stubbed mic button / non-New
  file menu items / delete icon.

### When finished

- Update the Phase 4 heading in `C:\Users\New user\Xenon2\PLAN.md` to mark
  it complete (`[x]`).
- Update `app/README.md` to document the Pinia store, the edit/regenerate
  flows, and explicitly note that delete remains an intentional stub with
  no phase currently owning it.
- Do NOT create any git commits — leave everything staged/unstaged for
  the user to review, same as Phases 1-3. If you do need to reference a
  commit author for any reason, it's `MrMcCrea_coder
  <MrMcCrea_coder@users.noreply.github.com>` (set via `GIT_AUTHOR_*`/
  `GIT_COMMITTER_*` env vars, never global git config — no global git
  identity exists on this machine).
- Stay in scope: no delete logic, no file save/load, no voice/STT/TTS
  wiring, no export/import.

## Report back

Concise final report: what was built, confirmation each acceptance
criterion was actually exercised (with specifics — e.g. "conversation had
5 messages, edited message 2, confirmed messages 3-5 removed and a new
message 3 streamed in") and passed, the delete-stub scope gap flagged
explicitly, and any deviations/blockers hit.
