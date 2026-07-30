# Xenon2 — Phase 3: Desktop Shell & Chat UI

A Tauri 2 + Vue 3 desktop chat app: sidebar + main chat panel, ChatGPT/Claude-Desktop-style
layout. This is Phase 3 of the Xenon2 project (see `../PLAN.md` and
`../prompts/phase3_prompt.md`) — the shell, layout, and typed-text-in / streamed-text-out wiring
against Phase 1's inference engine. No voice, no message editing/regeneration, no file save/load
yet — those are Phases 2 (already built separately), 4, and 5.

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

Per `prompts/phase3_prompt.md`'s explicit scope boundary, several UI elements are deliberate
visual placeholders for later phases — present and clickable, but not wired to real logic yet.
Nothing below was skipped by accident; each stub logs to the browser devtools console
(`console.log`) so it's easy to confirm nothing silently does real work.

| Piece | Status | Notes |
|---|---|---|
| Typing a message + Send button | **Functional** | Full round trip through Rust into Phase 1's `xenon_generate()`, streamed back token-by-token. |
| "+ New Chat" (Sidebar) / File > New | **Functional** | Both call the same `newChat()` in `App.vue` — clears the main panel and starts a fresh, empty, in-memory conversation. The only File-menu item that's real this phase. |
| Sidebar conversation list (Today/Yesterday/Older grouping, click to load) | **Functional** | In-memory only, per spec — no persistence to disk yet (that's Phase 5). Closing the app loses history. |
| Edit / delete icons on user messages, regenerate icon on assistant messages (`ChatMessage.vue`) | **Stub** | Visible on hover, clickable, `console.log` only. Real message editing/truncation/regeneration is **Phase 4**'s job. |
| Mic toggle button (`MessageInput.vue`) | **Stub** | Visually toggles on/off (turns red), but captures no audio. Phase 2's STT pipeline (already built standalone under `../voice-pipeline/`) gets wired into this button in a later integration step, not in this phase. |
| File > Open / Save / Save As | **Stub** | Present in the dropdown, `console.log` only, no file dialog opens. Real file I/O is **Phase 5**'s job. |
| File > Export Memory | **Stub** | Present, `console.log` only. Real copy-to-external-drive logic is **Phase 6**'s job. |

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
- `App.vue` listens for these three events (`@tauri-apps/api/event`'s `listen()`), appends
  incoming text directly onto the streaming assistant message's `content` as it arrives, and
  trims the trailing `\n\nUser:` stop-sequence text once generation completes (the model, like
  Phase 1's CLI harness, naturally continues into a fake next turn before the stop-sequence check
  fires — that already-streamed tail is stripped from the rendered bubble only, not from what was
  sent to the model).

This was verified against the real compiled Phase 1 library and the real quantized model, not
stubbed: launching the dev build and sending a message produces an actual RWKV-generated reply,
visibly appearing token-by-token.
