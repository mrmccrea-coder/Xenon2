<script setup lang="ts">
// ChatMessage.vue -- one message bubble (user or assistant).
//
// Scope boundary (see prompts/phase3_prompt.md): edit/delete (user) and regenerate (assistant)
// icons must be visually present and clickable, but their real behavior -- editing a message,
// truncating conversation history, re-running generate() -- is Phase 4's job. Here they are
// no-op console.log stubs on purpose; do not wire up real logic.

import type { ChatMessage } from "../types";

const props = defineProps<{
  message: ChatMessage;
}>();

function stubEdit() {
  console.log(
    `[ChatMessage] Edit clicked for message ${props.message.id} -- stub only, real editing/` +
      `truncation logic is Phase 4's job.`
  );
}

function stubDelete() {
  console.log(
    `[ChatMessage] Delete clicked for message ${props.message.id} -- stub only, real deletion ` +
      `logic is Phase 4's job.`
  );
}

function stubRegenerate() {
  console.log(
    `[ChatMessage] Regenerate clicked for message ${props.message.id} -- stub only, real ` +
      `regenerate logic is Phase 4's job.`
  );
}
</script>

<template>
  <div class="row" :class="message.role">
    <div class="bubble" :class="{ errored: message.errored }">
      <div class="content">{{ message.content }}<span v-if="message.streaming" class="cursor">▍</span></div>

      <div class="icons">
        <template v-if="message.role === 'user'">
          <button class="icon-btn" title="Edit (stub)" @click="stubEdit">✎</button>
          <button class="icon-btn" title="Delete (stub)" @click="stubDelete">🗑</button>
        </template>
        <template v-else>
          <button class="icon-btn" title="Regenerate (stub)" @click="stubRegenerate">⟳</button>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.row {
  display: flex;
  width: 100%;
  margin: 0.35rem 0;
}

.row.user {
  justify-content: flex-end;
}

.row.assistant {
  justify-content: flex-start;
}

.bubble {
  position: relative;
  max-width: 70%;
  padding: 0.6rem 0.85rem;
  border-radius: 12px;
  font-size: 0.92rem;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}

.row.user .bubble {
  background: #2f6feb;
  color: #fff;
  border-bottom-right-radius: 3px;
}

.row.assistant .bubble {
  background: #2a2a2c;
  color: #eee;
  border-bottom-left-radius: 3px;
}

.bubble.errored {
  background: #5a2323;
  color: #ffd6d6;
}

.cursor {
  display: inline-block;
  animation: blink 1s steps(1) infinite;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.icons {
  position: absolute;
  top: -1.3rem;
  display: flex;
  gap: 0.25rem;
  opacity: 0;
  transition: opacity 0.12s ease;
}

.row.user .icons {
  right: 0;
}

.row.assistant .icons {
  left: 0;
}

.bubble:hover .icons {
  opacity: 1;
}

.icon-btn {
  border: none;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  border-radius: 5px;
  width: 1.5rem;
  height: 1.5rem;
  font-size: 0.75rem;
  cursor: pointer;
  line-height: 1;
}

.icon-btn:hover {
  background: rgba(0, 0, 0, 0.8);
}
</style>
