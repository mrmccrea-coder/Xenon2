# Prompt: Xenon2 Phase 7 — Hardening & Verification Follow-Through

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context. All
6 phases in that plan are marked `[x]` complete. This is a **follow-up
hardening phase**, not in the original `PLAN.md` numbering — it exists to
close specific gaps a project review surfaced after Phase 6, not to build
new user-facing scope.

Read `app/README.md` in full before starting — it documents exactly what's
real vs. stubbed as of Phase 6, including the "What was actually tested" /
"Not verified in this pass" sections this phase directly follows up on.

**Do not re-derive the gaps below from scratch — they were already found and
documented by a prior review.** Your job is to close them, not rediscover
them.

---

## The gaps this phase closes

1. **Mic button is cosmetic** (`MessageInput.vue`) — toggles red visually but
   captures no audio. Phase 2's voice pipeline (`voice-pipeline/`: silero-vad
   → faster-whisper STT → sentence-chunked streaming piper-tts) is fully
   built and independently verified (see `voice-pipeline/README.md`), but was
   never wired into the desktop app's UI.
2. **Delete-message is an unowned stub** (`ChatMessage.vue`) — `console.log`
   only. `PLAN.md`'s own notes (see Phase 4's scope gap) confirm no phase
   ever claimed this; it's real missing functionality, not a deferred
   decision waiting on anything else.
3. **Saved `model` field isn't validated on load** — `SCHEMA.md` and
   `persistence.rs`'s `open_conversation_file` record which model generated
   a conversation but never compare it against the currently-loaded model.
   A conversation created under a different model silently loads and
   continues under the wrong one with no warning.
4. **Phase 6 verification gap**: per `app/README.md`'s own "Not verified in
   this pass" note, Export/Import/Data-Directory-Settings were verified at
   the Tauri-command/IPC level (via Chrome DevTools Protocol), not through
   an actual GUI click-through of the menu items and progress modal.
5. **Live human speech was never tested** — `voice-pipeline/README.md`
   documents STT accuracy verified only against a WAV fixture; live mic
   mechanics were verified with silence/timeout, never actual spoken words,
   because no working microphone was available in that environment.
6. **Uncommitted stray git state**: `git status` (run it yourself to
   confirm current state — it may have changed since this was written) has
   shown `prompts/phase1_prompt.md` modified and the `inference-engine/rwkv.cpp`
   submodule with uncommitted/untracked content. Reconcile this before
   treating the repo as clean.
7. **App icon**: replace the default Tauri icon with a provided image (a
   green alien in a UFO) so the app is visually identifiable when a user
   launches it to test the UI. The source image should be provided
   alongside this prompt — if it isn't present at
   `C:\Users\New user\Xenon2\app\src-tauri\icons\source-icon.png` when you
   start, stop and ask for it rather than substituting a placeholder.

**Explicitly out of scope, do not attempt:** the Windows Common Item Dialog
hang (`persistence.rs`/`memory.rs`'s 30-second timeout workaround). This has
already had a deep, dedicated root-cause investigation that hit a hard wall
(needs interactive UAC elevation for Process Monitor, which no automated
agent session can grant itself) — don't re-run that investigation. Leave the
existing timeout + `XENON2_TEST_*_DIALOG_PATH` escape hatch exactly as is.
Also out of scope: a real physical second-machine migration test for Phase
6 (task 4 below is about the *GUI*, not re-attempting cross-machine — that
substitution is an accepted, documented limitation of this dev environment).

---

## Your tasks

### Task 1 — Wire the mic button to the real voice pipeline

Working directory: `C:\Users\New user\Xenon2\app\`

`voice-pipeline/` is currently a standalone Python pipeline (`cli_harness.py`
+ modules for VAD/STT/TTS), not integrated into the Rust/Tauri app. Decide
and implement a real integration — the two most likely approaches, pick
whichever fits better once you've read both sides' code, and document the
choice:
- Rewrite/port the VAD→STT loop as a Rust Tauri command (calling into
  faster-whisper via a Rust binding, or shelling out to a bundled Python
  runtime), emitting a `transcription-done` event the same way
  `inference.rs` emits `token-stream`; or
- Run `voice-pipeline`'s existing Python code as a sidecar process Tauri
  spawns and communicates with over stdio/a local socket.

Whichever you choose, the mic button in `MessageInput.vue` should, on click:
start listening (real VAD-gated capture, not always-on), transcribe on
speech end, and feed the transcript into the exact same `sendMessage` path
typed text already uses — per Phase 3's original spec, "typed text and voice
transcripts both feed into the same send-message code path." Do not build a
parallel/separate code path for voice-originated messages.

TTS playback of the assistant's reply (using Phase 2's sentence-chunked
streaming synthesis, not a naive full-text-then-play) should also be wired
in as part of this task — Phase 2 built it standalone specifically so it
could be reused here.

### Task 2 — Implement real delete-message

Working directory: `app/`

Wire `ChatMessage.vue`'s delete icon (currently `stubDelete`) to a real
`useChatStore().deleteMessage(conversationId, messageId)` action. Decide and
document the semantics for deleting a non-last message (e.g., does deleting
a user message also delete its paired assistant reply? does it require
confirmation?) — there's no existing spec to follow here since no phase ever
owned this, so use your judgment, keep it simple, and write down the
behavior you chose in `app/README.md`. Persist the deletion the same way
edits persist (auto-save fires after the store mutation).

### Task 3 — Warn on model mismatch when opening a conversation

Working directory: `app/`

In `persistence.rs`'s `open_conversation_file` (or the frontend
`openConversation` action, whichever is the better layering call — your
judgment), compare the opened file's `model` field against the currently
loaded model (already available via `get_model_name`/`modelName` in the
store). If they differ, surface a clear, dismissable warning in the UI
(reuse the existing file-error-banner pattern in `App.vue`, or a distinct
banner if that reads better) rather than silently proceeding. Do not block
opening the file — RWKV models can usually still generate on a mismatched
conversation, just tell the user it's not guaranteed to make sense.

### Task 4 — Full GUI click-through of Phase 6 features

Working directory: `app/`

Using the same UI Automation approach Phases 3-5 used
(`InvokePattern`/`ValuePattern` at the COM level, `PrintWindow` +
`PW_RENDERFULLCONTENT` for screenshots — see `app/README.md`'s Phase 4 notes
for why plain simulated input/`ImageGrab` don't work reliably on this
machine), actually click through: File > Export Memory, the progress modal
appearing and updating, File > Import Memory, and File > Data Directory
Settings. Since the OS folder-picker dialog itself carries the known hang
risk, use the existing `XENON2_TEST_*_DIALOG_PATH` escape hatch (extend it
to cover the folder picker if `memory.rs`'s `pick_folder_dialog` doesn't
already respect it — check first) to get past just that one step, and drive
everything else for real. Document what you observed (screenshots or
described UI state, like Phase 4/5 did) in `app/README.md`, replacing the
"Not verified in this pass" GUI caveat with what was actually verified.

### Task 5 — Live human speech test

Depends on Task 1 being done first. Once the mic button is real, if a
working microphone is available in whatever environment runs this task,
speak a real greeting into it and confirm: VAD detects speech start/stop
correctly, STT produces a reasonable transcript, it appears in the chat as
a user message, and a real spoken reply comes back through TTS. If no
microphone is available (as was the case in earlier phases), say so
explicitly rather than silently skipping — don't let this task disappear
quietly the way it did before.

### Task 6 — Clean up git stray state

Run `git status` and `git diff` yourself (don't trust old output — repo
state may have moved on). For `prompts/phase1_prompt.md`'s modification:
diff it against the last commit, understand what changed and why, and
either commit it with an honest message or revert it if the change wasn't
intentional. For the `inference-engine/rwkv.cpp` submodule: check
`git submodule status` and `git -C inference-engine/rwkv.cpp status` to see
what's actually different (untracked build artifacts that should be
gitignored? a genuine pinned-commit bump that needs committing?) and
resolve it — don't just leave it stray. Do not force-discard anything
without understanding what it is first.

### Task 7 — Replace the app icon

Working directory: `app/src-tauri/`

A source image (green alien character in a flying-saucer/UFO, rounded
friendly cartoon style) should be available at
`app/src-tauri/icons/source-icon.png` — confirm it's there first; if not,
stop this task and flag it rather than inventing a placeholder icon. Use the
Tauri CLI's icon generator (`npm run tauri icon <path-to-source-icon>` from
`app/`, or `npx @tauri-apps/cli icon <path>`) to regenerate the full icon set
(`icon.ico`, `icon.icns`, and the various PNG sizes under
`src-tauri/icons/`) from that source image, replacing Tauri's default
placeholder icons. Confirm `tauri.conf.json`'s `bundle.icon` array already
points at the generated files (it should, by default, if you use the CLI
generator rather than hand-copying files) — update it if it doesn't.
Rebuild (`npm run tauri dev` or `npm run tauri build`) and confirm the new
icon actually appears on the app window / taskbar, not just that the icon
files changed on disk.

---

## Acceptance criteria (verify before considering this phase done — actually run the app, don't just review code)

- Clicking the mic button, speaking (or feeding a test WAV if no live mic is
  available), and seeing the transcript appear as a real user message that
  gets a real streamed reply — same code path as typed input.
- Deleting a message actually removes it from the store, persists across a
  save/reload, and the chosen semantics (documented in `app/README.md`) are
  applied consistently.
- Opening a conversation saved under a different model name shows a visible
  mismatch warning; opening one saved under the current model shows nothing
  extra.
- A full GUI-driven click-through of Export/Import/Data-Directory-Settings
  is documented with what was actually observed, not just "the underlying
  command works."
- `git status` is clean (or every remaining diff is deliberate and
  explained) by the end of this phase.
- The app window/taskbar shows the new alien/UFO icon, not Tauri's default.

## When finished

- Add a "Phase 7" section to `PLAN.md` documenting this hardening pass (it's
  outside the original 6-phase numbering, so don't renumber existing
  phases — append it after Phase 6).
- Update `app/README.md`'s stub table: mic toggle and delete icon move from
  "Stub" to "Functional," with the same level of implementation detail the
  existing table entries have.
- Do NOT create any git commits beyond what Task 6 explicitly calls for —
  leave new work staged/unstaged for the user to review, same convention as
  Phases 1-6. Commit author, if ever needed:
  `MrMcCrea_coder <MrMcCrea_coder@users.noreply.github.com>` via
  `GIT_AUTHOR_*`/`GIT_COMMITTER_*` env vars (no global git identity exists
  on this machine).

## Report back

Concise final report covering each of the 7 tasks: what was built, what was
actually exercised to verify it (with specifics, matching the evidentiary
style of prior phases' READMEs — exact numbers, byte counts, before/after
diffs where applicable), and any blockers or deviations, especially if the
source icon image wasn't available or if a working microphone wasn't
available for Task 5.
