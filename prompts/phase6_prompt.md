# Prompt: Xenon2 Phase 6 — External Memory Export/Import

## Background

You're building **Xenon2**, a portable, offline-first desktop AI assistant
with a chat interface similar to ChatGPT/Claude Desktop, plus voice
input/output. The full project plan lives at
`C:\Users\New user\Xenon2\PLAN.md` — read it first for complete context on
all 6 phases, but you are only doing **Phase 6** right now, the final
phase currently scoped.

**Phases 1-5 are already complete and verified working** (not just marked
done in the plan — actually built, run, and tested against real files on
disk):
- Phase 1: `inference-engine/` — RWKV inference core, CPU and CUDA builds.
- Phase 2: `voice-pipeline/` — standalone STT/VAD/TTS pipeline (not yet
  wired into the desktop app's mic button — that integration hasn't
  happened yet and is not part of this phase either).
- Phase 3/4: `app/` — Tauri + Vue 3 desktop shell with working chat,
  message editing, and regeneration.
- Phase 5: real file save/load. Conversations are written to and read from
  disk as JSON matching `SCHEMA.md`. Read `app/README.md`'s "Phase 5: file
  persistence" section before starting — it documents exactly where files
  actually live (see next section; the original Phase 6 plan text
  describing `<root>/xenon2-data/projects/` is **not** where things
  actually ended up, and you should build against the real paths, not that
  placeholder).

### Real on-disk layout you're exporting/importing (verified, not assumed)

This was checked directly against the actual filesystem before writing
this prompt — use these real paths, not the hypothetical ones in
`PLAN.md`'s original Phase 6 sketch:

**Conversations** (Phase 5's actual output):
- `<Tauri app-data-dir>/conversations/<conversation-id>.json` — one file
  per conversation, matching the schema in `SCHEMA.md`.
- `<Tauri app-data-dir>/session.json` — remembers last-active conversation
  and each conversation's file path. On Windows this resolves under
  `%APPDATA%\com.xenon2.app\`; use Tauri's `app_data_dir()` API to resolve
  it programmatically rather than hardcoding the Windows path, since this
  needs to make sense if the project ever targets other platforms.

**Models actually needed at runtime** (do not assume "everything in
`models/` needs exporting" — check what's actually loaded):
- `models/rwkv-5-world-0.4B-Q4_0.bin` — the quantized model the app
  actually loads (`inference.rs`, per `app/README.md`). **This is the one
  that must be exported.**
- `inference-engine/data/world_vocab.bin` — the tokenizer vocab file.
  Required for the model to load at all (see Phase 1's model loading
  code) — easy to forget since it's not inside `models/`, but exporting
  the model without it produces a non-functional export.
- `models/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth` and
  `models/rwkv-5-world-0.4B-FP16.bin` are conversion intermediates (the
  original downloaded checkpoint and an unquantized intermediate) — **not
  loaded by the app at runtime** and together are ~1.8GB. Do not include
  these in the export by default; they'd nearly triple export size for no
  functional benefit. If you want a "full/dev export" mode that includes
  them, make it explicitly opt-in and off by default.
- `voice-pipeline/models/en_US-lessac-medium.onnx` (+ its `.onnx.json`
  sidecar) — the piper TTS voice model used by the standalone voice
  pipeline. Include this in the export even though the voice pipeline
  isn't wired into the desktop app's UI yet — the goal of this phase is
  full portability of everything Xenon2 depends on, and the voice pipeline
  is a real, working part of the project already (see Phase 2's
  verification in its own README).

### Why this project is unusual (brief context, not directly relevant here)

This project uses RWKV instead of a Transformer for inference — relevant
to Phases 1/2/3/4, but not to this phase. Phase 6 is file/directory
copying and a settings option; nothing about RWKV changes how you
implement portable export/import. The one place it's worth remembering:
the exported conversation JSON already records which model generated it
(`"model"` field, see `SCHEMA.md`) — that's Phase 5's work, already done;
you don't need to add it again here.

---

## Your task: Phase 6 — External Memory Export/Import

Working directory: `C:\Users\New user\Xenon2\`

### Tasks

1. Define the on-disk layout for a portable export bundle, e.g.:
   ```
   <destination>/xenon2-backup/
     models/
       rwkv-5-world-0.4B-Q4_0.bin
       world_vocab.bin
     voice-models/
       en_US-lessac-medium.onnx
       en_US-lessac-medium.onnx.json
     conversations/
       <conversation-id>.json
       session.json
   ```
   Document this layout in a new `EXPORT_FORMAT.md` at the project root
   (the same way `SCHEMA.md` documents the conversation file format).
2. Implement **Export** (wire up the existing `File > Export Memory` stub
   in `MenuBar.vue` — see `app/README.md`'s stub table): open a Tauri
   folder-picker dialog for the destination (a USB drive, external SSD, or
   any local path), then copy the real runtime files listed above into
   the bundle layout from task 1. Show progress for large files (the
   quantized model alone is ~450MB) rather than appearing to hang.
3. Implement **Import**: a new menu item or dialog to pick a source folder
   (e.g. a previously-exported bundle on a plugged-in USB drive), and
   either (a) copy its `models/`, `voice-models/`, and `conversations/`
   contents into this machine's real local paths (Tauri app-data dir for
   conversations, `models/`/`voice-pipeline/models/` for the model files),
   or (b) run directly against the external path without copying — pick
   one as the default behavior and document why, and if you support both,
   make the non-default an explicit option rather than silently guessing.
4. Add a settings option for "data directory location" so the app can be
   pointed at an external drive as its primary storage location instead of
   the local app-data dir / local `models/` folder — i.e. the app's
   *ongoing* storage, not just one-time export/import. Persist this
   setting somewhere that survives restarts (e.g. a small settings file
   in the Tauri app-data dir, separate from `session.json`).

### Acceptance criteria (verify before considering this phase done)

- Export produces a self-contained folder that includes the quantized
  RWKV model, its tokenizer vocab file, the piper voice model, and all
  saved conversations — verify by actually running Export against a real
  destination folder and inspecting what landed there, not just by
  reading the code.
- That exported folder, copied to a path simulating "a different machine"
  (e.g. a fresh temp directory with no prior Xenon2 data) and then
  Imported, results in the app loading with full model + conversation
  history restored — verify this by actually doing it, the same
  verification standard Phases 1-5 used (real run, real files, documented
  observations, not just implementation claims).
- The unquantized `.pth` and FP16 intermediates are confirmed **excluded**
  from a default export (check the resulting folder size — it should be
  roughly 450MB (RWKV) + 63MB (voice model) + conversation JSON, not
  gigabytes).
- Changing the "data directory location" setting and restarting the app
  actually uses the new location for subsequent saves/loads.

### When finished

Update the Phase 6 heading in `C:\Users\New user\Xenon2\PLAN.md` to mark
it complete, update `app/README.md`'s stub table (Export Memory moves from
"Stub" to "Functional"), and document in `EXPORT_FORMAT.md` and/or
`app/README.md` what was actually tested, including the real export
folder size you observed and confirmation the import round-trip worked
end to end.
