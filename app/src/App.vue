<script setup lang="ts">
// App.vue -- top-level layout: MenuBar on top, Sidebar + chat panel below (Phase 3 shell).
//
// As of Phase 4, conversation/message state lives in the Pinia store (`stores/chat.ts`), not in
// local refs -- this component reads/writes through `useChatStore()`. It still owns Tauri
// event-listener registration and DOM-only concerns (scroll-to-bottom) that don't belong in a
// store. Token/done/error events from the `generate_response` Tauri command are forwarded
// straight into the matching store actions, which know how to find the right message (by id, not
// by array position) to update.

import { onMounted, onUnmounted, ref } from "vue";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import MenuBar from "./components/MenuBar.vue";
import Sidebar from "./components/Sidebar.vue";
import ChatMessageVue from "./components/ChatMessage.vue";
import MessageInput from "./components/MessageInput.vue";
import { useChatStore } from "./stores/chat";

const store = useChatStore();

const messagesContainer = ref<HTMLElement | null>(null);

function scrollToBottom() {
  requestAnimationFrame(() => {
    const el = messagesContainer.value;
    if (el) el.scrollTop = el.scrollHeight;
  });
}

function newChat() {
  store.newChat();
  scrollToBottom();
}

function selectConversation(id: string) {
  store.selectConversation(id);
  scrollToBottom();
}

// Phase 5: File > Open / Save / Save As. MenuBar just emits; the store owns the actual Tauri
// dialog + filesystem calls (see stores/chat.ts). Errors surface via store.fileError, rendered in
// the banner below rather than a native alert() so it matches the existing backendError banner.
async function openConversation() {
  await store.openConversation();
  scrollToBottom();
}

async function saveConversation() {
  const id = store.activeConversation?.id;
  if (!id) return;
  await store.saveConversation(id);
}

async function saveConversationAs() {
  const id = store.activeConversation?.id;
  if (!id) return;
  await store.saveConversationAs(id);
}

// Phase 6: Export / Import Memory + data directory settings. MenuBar just emits; the store owns
// the Tauri dialog + backend calls. Progress is rendered from store.memoryOp, populated by the
// export-progress/import-progress event listeners registered below.
async function exportMemory() {
  await store.exportMemory();
}

async function importMemory() {
  await store.importMemory();
}

const settingsOpen = ref(false);
const settingsInput = ref("");

function openSettings() {
  settingsInput.value = store.dataDir ?? "";
  settingsOpen.value = true;
}

async function pickDataDir() {
  try {
    const dir = await import("@tauri-apps/api/core").then((m) =>
      m.invoke<string | null>("pick_folder_dialog", { testEnvVar: "XENON2_TEST_DATADIR_PICK_PATH" })
    );
    if (dir) settingsInput.value = dir;
  } catch (err) {
    console.warn("[App] folder pick failed:", err);
  }
}

async function saveDataDirSetting() {
  await store.setDataDir(settingsInput.value.trim() || null);
  settingsOpen.value = false;
}

async function clearDataDirSetting() {
  await store.setDataDir(null);
  settingsInput.value = "";
  settingsOpen.value = false;
}

async function sendMessage(text: string) {
  scrollToBottom();
  await store.sendMessage(text);
  scrollToBottom();
}

async function editMessage(messageId: string, newText: string) {
  const conversationId = store.activeConversation?.id;
  if (!conversationId) return;
  scrollToBottom();
  await store.editUserMessage(conversationId, messageId, newText);
  scrollToBottom();
}

async function regenerateMessage(messageId: string) {
  const conversationId = store.activeConversation?.id;
  if (!conversationId) return;
  await store.regenerateAssistantMessage(conversationId, messageId);
  scrollToBottom();
}

let unlistenToken: UnlistenFn | null = null;
let unlistenDone: UnlistenFn | null = null;
let unlistenError: UnlistenFn | null = null;
let unlistenExportProgress: UnlistenFn | null = null;
let unlistenExportDone: UnlistenFn | null = null;
let unlistenExportError: UnlistenFn | null = null;
let unlistenImportProgress: UnlistenFn | null = null;
let unlistenImportDone: UnlistenFn | null = null;
let unlistenImportError: UnlistenFn | null = null;

type MemoryProgressPayload = {
  file: string;
  fileIndex: number;
  totalFiles: number;
  bytesDone: number;
  bytesTotal: number;
};
type MemoryDonePayload = { filesCopied: number; totalBytes: number };
type MemoryErrorPayload = { message: string };

onMounted(async () => {
  // Phase 5: fetch the backend's actually-loaded model name first (so any conversation created
  // or loaded afterwards has a real fallback value -- see store.newChat/autoSave), then restore
  // the last session (sidebar + last-active conversation) before wiring up generation events.
  await store.initModelName();
  await store.loadDataDirSetting();
  await store.restoreSession();
  scrollToBottom();

  // Phase 6: export/import progress events (see memory.rs's copy_with_progress).
  unlistenExportProgress = await listen<MemoryProgressPayload>("export-progress", (event) => {
    store.onMemoryProgress("export", event.payload);
  });
  unlistenExportDone = await listen<MemoryDonePayload>("export-done", (event) => {
    store.onMemoryDone("export", event.payload);
  });
  unlistenExportError = await listen<MemoryErrorPayload>("export-error", (event) => {
    store.onMemoryError("export", event.payload.message);
  });
  unlistenImportProgress = await listen<MemoryProgressPayload>("import-progress", (event) => {
    store.onMemoryProgress("import", event.payload);
  });
  unlistenImportDone = await listen<MemoryDonePayload>("import-done", (event) => {
    store.onMemoryDone("import", event.payload);
  });
  unlistenImportError = await listen<MemoryErrorPayload>("import-error", (event) => {
    store.onMemoryError("import", event.payload.message);
  });

  unlistenToken = await listen<{ conversationId: string; text: string }>("token-stream", (event) => {
    store.appendToken(event.payload.conversationId, event.payload.text);
    scrollToBottom();
  });

  unlistenDone = await listen<{ conversationId: string }>("generation-done", (event) => {
    store.completeGeneration(event.payload.conversationId);
  });

  unlistenError = await listen<{ conversationId: string; message: string }>(
    "generation-error",
    (event) => {
      store.failGeneration(event.payload.conversationId, event.payload.message);
    }
  );
});

onUnmounted(() => {
  unlistenToken?.();
  unlistenDone?.();
  unlistenError?.();
  unlistenExportProgress?.();
  unlistenExportDone?.();
  unlistenExportError?.();
  unlistenImportProgress?.();
  unlistenImportDone?.();
  unlistenImportError?.();
});
</script>

<template>
  <div class="app-shell">
    <MenuBar
      :has-active-conversation="!!store.activeConversation"
      @new="newChat"
      @open="openConversation"
      @save="saveConversation"
      @save-as="saveConversationAs"
      @export-memory="exportMemory"
      @import-memory="importMemory"
      @settings="openSettings"
    />

    <div class="body">
      <Sidebar
        :conversations="store.conversations"
        :active-id="store.activeId"
        @new-chat="newChat"
        @select="selectConversation"
      />

      <main class="chat-panel">
        <div v-if="!store.activeConversation" class="empty-state">
          <p>No conversation selected.</p>
          <p class="hint">Type a message below, or click "+ New Chat" to get started.</p>
        </div>

        <template v-else>
          <div ref="messagesContainer" class="messages">
            <ChatMessageVue
              v-for="m in store.activeConversation.messages"
              :key="m.id"
              :message="m"
              :disabled="store.generating"
              @edit="(newText) => editMessage(m.id, newText)"
              @regenerate="() => regenerateMessage(m.id)"
            />
          </div>
        </template>

        <div v-if="store.backendError" class="error-banner">{{ store.backendError }}</div>
        <div v-if="store.fileError" class="error-banner file-error">
          {{ store.fileError }}
          <button class="dismiss-btn" title="Dismiss" @click="store.fileError = null">✕</button>
        </div>

        <MessageInput :disabled="store.generating" @send="sendMessage" />
      </main>
    </div>

    <!-- Phase 6: export/import progress -->
    <div v-if="store.memoryOp" class="modal-backdrop">
      <div class="modal memory-modal">
        <h3>{{ store.memoryOp.kind === "export" ? "Exporting Memory..." : "Importing Memory..." }}</h3>
        <p class="memory-file">
          File {{ store.memoryOp.fileIndex + 1 }} / {{ store.memoryOp.totalFiles || "?" }}:
          {{ store.memoryOp.file }}
        </p>
        <div class="progress-bar">
          <div
            class="progress-fill"
            :style="{
              width:
                store.memoryOp.bytesTotal > 0
                  ? (store.memoryOp.bytesDone / store.memoryOp.bytesTotal) * 100 + '%'
                  : '0%',
            }"
          ></div>
        </div>
        <p class="memory-bytes">
          {{ (store.memoryOp.bytesDone / (1024 * 1024)).toFixed(1) }} MB /
          {{ (store.memoryOp.bytesTotal / (1024 * 1024)).toFixed(1) }} MB
        </p>
      </div>
    </div>

    <div v-if="store.memoryMessage && !store.memoryOp" class="toast">
      {{ store.memoryMessage }}
      <button class="dismiss-btn" @click="store.memoryMessage = null">✕</button>
    </div>
    <div v-if="store.memoryError" class="toast toast-error">
      {{ store.memoryError }}
      <button class="dismiss-btn" @click="store.memoryError = null">✕</button>
    </div>

    <!-- Phase 6: data directory location settings -->
    <div v-if="settingsOpen" class="modal-backdrop" @click.self="settingsOpen = false">
      <div class="modal">
        <h3>Data Directory Location</h3>
        <p class="modal-hint">
          When set, new conversations auto-save to and load from this folder's
          <code>conversations/</code> subfolder instead of the local app-data default. Leave blank
          to use the default location.
        </p>
        <div class="modal-row">
          <input v-model="settingsInput" class="modal-input" placeholder="(local app-data default)" />
          <button class="modal-btn" @click="pickDataDir">Browse...</button>
        </div>
        <p v-if="store.dataDir" class="modal-hint">Currently: {{ store.dataDir }}</p>
        <p v-else class="modal-hint">Currently: local app-data default.</p>
        <div class="modal-actions">
          <button class="modal-btn" @click="clearDataDirSetting">Use Default</button>
          <button class="modal-btn" @click="settingsOpen = false">Cancel</button>
          <button class="modal-btn primary" @click="saveDataDirSetting">Save</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
}

.body {
  display: flex;
  flex: 1;
  min-height: 0;
}

.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: #131314;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem 1.5rem 0.5rem;
  display: flex;
  flex-direction: column;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #888;
  text-align: center;
  gap: 0.3rem;
}

.empty-state .hint {
  font-size: 0.85rem;
  color: #666;
}

.error-banner {
  margin: 0 1rem;
  padding: 0.4rem 0.7rem;
  background: #5a2323;
  color: #ffd6d6;
  border-radius: 6px;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.error-banner.file-error {
  margin-top: 0.4rem;
}

.dismiss-btn {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  font-size: 0.8rem;
  flex-shrink: 0;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: #232325;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  min-width: 320px;
  max-width: 460px;
  color: #eee;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
}

.modal h3 {
  margin: 0 0 0.6rem;
  font-size: 1rem;
}

.memory-file {
  font-size: 0.8rem;
  color: #aaa;
  word-break: break-all;
  margin: 0 0 0.6rem;
}

.progress-bar {
  height: 8px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #5b8def;
  transition: width 0.1s linear;
}

.memory-bytes {
  font-size: 0.75rem;
  color: #888;
  margin: 0.5rem 0 0;
  text-align: right;
}

.modal-hint {
  font-size: 0.78rem;
  color: #999;
  margin: 0.3rem 0;
}

.modal-hint code {
  color: #ccc;
}

.modal-row {
  display: flex;
  gap: 0.5rem;
  margin: 0.6rem 0;
}

.modal-input {
  flex: 1;
  background: #1a1a1b;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 4px;
  color: #eee;
  padding: 0.4rem 0.5rem;
  font-size: 0.85rem;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1rem;
}

.modal-btn {
  background: #33333a;
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #eee;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
  font-size: 0.82rem;
  cursor: pointer;
}

.modal-btn:hover {
  background: #3d3d44;
}

.modal-btn.primary {
  background: #5b8def;
  border-color: #5b8def;
  color: #fff;
}

.toast {
  position: fixed;
  bottom: 1.25rem;
  right: 1.25rem;
  background: #234a2e;
  color: #d6ffe0;
  padding: 0.6rem 0.9rem;
  border-radius: 6px;
  font-size: 0.82rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  z-index: 110;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.toast-error {
  background: #5a2323;
  color: #ffd6d6;
}
</style>

<style>
:root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Avenir, Helvetica, Arial,
    sans-serif;
  font-size: 16px;
  color: #eee;
  background-color: #131314;
}

* {
  box-sizing: border-box;
}

html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
}
</style>
