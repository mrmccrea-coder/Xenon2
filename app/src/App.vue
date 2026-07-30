<script setup lang="ts">
// App.vue -- top-level layout: MenuBar on top, Sidebar + chat panel below (Phase 3 shell).
//
// Owns the in-memory conversation state (real persistence is Phase 5's job). Wires
// MessageInput's "send" event through the `generate_response` Tauri command and listens for the
// `token-stream` / `generation-done` / `generation-error` events the Rust backend emits while it
// streams Phase 1's inference engine output back token-by-token.

import { onMounted, onUnmounted, ref, computed } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import MenuBar from "./components/MenuBar.vue";
import Sidebar from "./components/Sidebar.vue";
import ChatMessageVue from "./components/ChatMessage.vue";
import MessageInput from "./components/MessageInput.vue";
import type { Conversation, ChatMessage } from "./types";
import { newId } from "./types";

const conversations = ref<Conversation[]>([]);
const activeId = ref<string | null>(null);
const generating = ref(false);
const backendError = ref<string | null>(null);

const activeConversation = computed<Conversation | null>(
  () => conversations.value.find((c) => c.id === activeId.value) ?? null
);

const messagesContainer = ref<HTMLElement | null>(null);

function scrollToBottom() {
  requestAnimationFrame(() => {
    const el = messagesContainer.value;
    if (el) el.scrollTop = el.scrollHeight;
  });
}

/** Starts a fresh, empty conversation and makes it active. Used by both "+ New Chat" (Sidebar)
 * and File > New (MenuBar) -- the one file-menu item that's actually functional this phase. */
function newChat() {
  const convo: Conversation = {
    id: newId(),
    title: "New Chat",
    createdAt: Date.now(),
    messages: [],
  };
  conversations.value.unshift(convo);
  activeId.value = convo.id;
  backendError.value = null;
}

function selectConversation(id: string) {
  activeId.value = id;
  backendError.value = null;
}

async function sendMessage(text: string) {
  if (!activeConversation.value) {
    newChat();
  }
  const convo = activeConversation.value!;
  if (convo.title === "New Chat" || !convo.title) {
    convo.title = text.length > 40 ? text.slice(0, 40) + "..." : text;
  }

  const userMsg: ChatMessage = { id: newId(), role: "user", content: text };
  const assistantMsg: ChatMessage = { id: newId(), role: "assistant", content: "", streaming: true };
  convo.messages.push(userMsg);
  convo.messages.push(assistantMsg);
  scrollToBottom();

  const conversationId = convo.id;
  const history = convo.messages
    .filter((m) => m.id !== assistantMsg.id)
    .map((m) => ({ role: m.role, content: m.content }));

  generating.value = true;
  backendError.value = null;

  try {
    await invoke("generate_response", { conversationId, history });
  } catch (err) {
    generating.value = false;
    assistantMsg.streaming = false;
    assistantMsg.errored = true;
    assistantMsg.content = `(failed to reach inference engine: ${String(err)})`;
    backendError.value = String(err);
  }
}

let unlistenToken: UnlistenFn | null = null;
let unlistenDone: UnlistenFn | null = null;
let unlistenError: UnlistenFn | null = null;

function findStreamingAssistantMessage(conversationId: string): ChatMessage | null {
  const convo = conversations.value.find((c) => c.id === conversationId);
  if (!convo) return null;
  // Streaming assistant message is always the last one while generation is in flight.
  const last = convo.messages[convo.messages.length - 1];
  return last && last.role === "assistant" && last.streaming ? last : null;
}

onMounted(async () => {
  unlistenToken = await listen<{ conversationId: string; text: string }>("token-stream", (event) => {
    const msg = findStreamingAssistantMessage(event.payload.conversationId);
    if (msg) {
      msg.content += event.payload.text;
      scrollToBottom();
    }
  });

  unlistenDone = await listen<{ conversationId: string }>("generation-done", (event) => {
    const msg = findStreamingAssistantMessage(event.payload.conversationId);
    if (msg) {
      msg.streaming = false;
      // The Rust backend's stop-sequence check (mirroring Phase 1's CLI harness) stops
      // generation as soon as the model starts a new "User:" turn, but by then that
      // stop-sequence text has already been streamed and appended -- strip it here so the
      // bubble shows only Xenon's actual reply, not the start of a fake next turn.
      msg.content = msg.content.replace(/\n+\s*[Uu]ser:\s*$/, "").trimEnd();
    }
    generating.value = false;
  });

  unlistenError = await listen<{ conversationId: string; message: string }>(
    "generation-error",
    (event) => {
      const msg = findStreamingAssistantMessage(event.payload.conversationId);
      if (msg) {
        msg.streaming = false;
        msg.errored = true;
        if (!msg.content) msg.content = `(inference error: ${event.payload.message})`;
      }
      backendError.value = event.payload.message;
      generating.value = false;
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
    <MenuBar @new="newChat" />

    <div class="body">
      <Sidebar
        :conversations="conversations"
        :active-id="activeId"
        @new-chat="newChat"
        @select="selectConversation"
      />

      <main class="chat-panel">
        <div v-if="!activeConversation" class="empty-state">
          <p>No conversation selected.</p>
          <p class="hint">Type a message below, or click "+ New Chat" to get started.</p>
        </div>

        <template v-else>
          <div ref="messagesContainer" class="messages">
            <ChatMessageVue
              v-for="m in activeConversation.messages"
              :key="m.id"
              :message="m"
            />
          </div>
        </template>

        <div v-if="backendError" class="error-banner">{{ backendError }}</div>

        <MessageInput :disabled="generating" @send="sendMessage" />
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
