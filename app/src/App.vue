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

onMounted(async () => {
  // Phase 5: fetch the backend's actually-loaded model name first (so any conversation created
  // or loaded afterwards has a real fallback value -- see store.newChat/autoSave), then restore
  // the last session (sidebar + last-active conversation) before wiring up generation events.
  await store.initModelName();
  await store.restoreSession();
  scrollToBottom();

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
