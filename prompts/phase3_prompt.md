# Prompt: Xenon2 Phase 3 — Desktop Shell & Chat UI

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context on
all 6 phases, but you are only doing **Phase 3** right now.

**Phase 1 must already be complete** before starting this phase — you need
a working `generate(prompt, max_tokens)` streaming function from the RWKV
inference engine at `inference-engine/`. If Phase 1 hasn't produced that
yet, stop and flag it rather than stubbing it out.

**Phase 2 (voice pipeline) does NOT need to be done yet.** This phase only
wires up **typed** text input. Phase 2 and Phase 3 are designed to be
built in parallel by separate agent runs — do not block on Phase 2, and do
not attempt to build any part of Phase 2's STT/TTS/VAD pipeline yourself.

### Why this project is unusual

This project deliberately uses **RWKV**, a non-transformer recurrent
architecture, instead of a standard Transformer LLM. RWKV streams tokens
one at a time as they're generated rather than returning a full response
in one batch. That's Phase 1's concern, not yours — but it matters here
because the chat UI must render streamed tokens as they arrive (text
appearing incrementally in the assistant's message bubble), not wait for
a complete response before displaying anything.

### What this phase is building

The actual application window: a sidebar + main chat panel layout, similar
to ChatGPT or Claude Desktop.

```
┌─────────────┬──────────────────────────────┐
│  Sidebar    │  Chat Panel                   │
│ + New Chat  │  [message bubbles...]         │
│ Today       │                                │
│  • Chat 1   │  [type message...] [mic] [➤] │
│ Yesterday   │                                │
│  • Chat 2   │                                │
└─────────────┴──────────────────────────────┘
```

No voice, no message editing logic, no file save/load logic exist yet —
those are Phases 2, 4, and 5 respectively. This phase is strictly the
shell, the layout, and typed-text-in / streamed-text-out wiring.

---

## IMPORTANT — explicit scope boundary: build the UI elements, not their future logic

Several UI elements you're building in this phase are **visual
placeholders for features that don't exist yet**. This is intentional
layering, not an oversight — do not skip building these elements because
"the logic isn't ready" in a later phase, and do not try to implement that
later phase's logic early just because you're already touching the
component. Specifically:

- **Edit / delete / regenerate icons on `ChatMessage.vue`** (task 3): these
  icons must be visually present and clickable, but the actual behavior of
  editing a message, truncating conversation history, and regenerating a
  response is **Phase 4's job**. For this phase, clicking them can be a
  no-op or a stub that logs to the console — do not implement message
  truncation/re-generation logic here.
- **The mic toggle button on `MessageInput.vue`** (task 4): this button
  must exist and be visually togglable, but it does not need to actually
  capture audio or do anything functional — Phase 2's STT output will be
  wired into this button's behavior in a later integration step, not in
  this phase. Do not attempt to build any audio capture code here.
- **File menu items on `MenuBar.vue`** (task 5): New / Open / Save / Save
  As / Export Memory must all appear as menu items, but Open/Save/Save
  As's actual file I/O is **Phase 5's job**, and Export Memory's actual
  copy-to-external-drive logic is **Phase 6's job**. "New" is the one
  exception — that should actually work in this phase, since starting a
  new empty conversation doesn't depend on any later phase.

If you find yourself writing STT/TTS code, conversation-truncation logic,
or filesystem save/load code while working on this phase, stop — that
work belongs in a different phase's prompt.

---

## Target hardware (context only — not directly relevant to UI work)

- CPU: Intel i7-12850HX, 16 cores / 24 threads
- RAM: 32GB
- GPU: NVIDIA RTX A1000 Laptop GPU, ~4GB VRAM (CUDA-capable)

This phase is UI scaffolding and doesn't do inference or heavy compute
itself — it calls into Phase 1's `generate()` via IPC. Hardware specifics
matter far more to Phases 1 and 2 than to this one.

---

## Your task: Phase 3 — Desktop Shell & Chat UI

Working directory: `C:\Users\New user\Xenon2\`

### Tasks

1. Scaffold a Tauri + Vue 3 project under `app/`.
2. Build `Sidebar.vue`: list of past conversations grouped by date
   (Today / Yesterday / Older), with a "+ New Chat" button that actually
   works (starts a fresh, empty conversation in the main panel). Clicking
   an existing conversation in the list should load it into the main
   panel — for this phase, an in-memory list of conversations is fine
   since real persistence is Phase 5's job.
3. Build `ChatMessage.vue`: renders one message bubble (user or
   assistant). On hover, show edit/delete icons for user messages and a
   regenerate icon for assistant messages — these icons should be present
   and clickable but can be no-ops/console-log stubs (see scope boundary
   above; do not implement Phase 4's logic here).
4. Build `MessageInput.vue`: a text input box with a send button, plus a
   mic toggle button. The text box and send button must be fully
   functional. The mic button should be present and visually toggleable
   but does not need to capture or process audio (see scope boundary
   above; Phase 2 supplies that later).
5. Build `MenuBar.vue`: a File menu with New / Open / Save / Save As /
   Export Memory items. Only "New" needs to be functional in this phase
   (clears the chat panel and starts a fresh conversation); the rest can
   be present but stubbed (see scope boundary above).
6. Implement Tauri IPC commands connecting the UI to Phase 1's
   `generate()` function: when the user types a message and hits send,
   it should be sent to the Rust backend, which calls into the Phase 1
   inference engine and streams tokens back to the UI as they're
   generated, appearing incrementally in a new assistant message bubble.

### Acceptance criteria (verify before considering this phase done)

- The app launches and displays an empty sidebar and chat panel.
- Typing a message and clicking send displays it as a user message
  bubble, and a streamed assistant response appears below it, with text
  visibly appearing incrementally (not all at once).
- The layout visually resembles ChatGPT/Claude Desktop: a sidebar on one
  side, the main chat panel on the other.
- Edit/delete/regenerate icons, the mic button, and the non-"New" file
  menu items are all visually present but confirmed to be non-functional
  stubs — verify none of them silently attempt real logic that belongs to
  a later phase.
- "+ New Chat" and File > New both correctly clear/reset the main panel
  to a fresh empty conversation.

### When finished

Update the Phase 3 heading in `C:\Users\New user\Xenon2\PLAN.md` to mark
it complete, and note in a short `app/README.md` how to run the dev build,
which pieces are fully functional versus intentional stubs for later
phases, and how the Tauri IPC command(s) connect to the Phase 1 inference
engine.
