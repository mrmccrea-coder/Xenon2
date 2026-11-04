<script setup lang="ts">
// Sidebar.vue -- conversation list grouped by date (Today / Yesterday / Older), "+ New Chat",
// click-to-load. Conversations are an in-memory prop from App.vue; real persistence is Phase 5's
// job (see prompts/phase3_prompt.md).

import { computed, ref } from "vue";
import type { Conversation } from "../types";

const props = defineProps<{
  conversations: Conversation[];
  activeId: string | null;
}>();

const emit = defineEmits<{
  (e: "new-chat"): void;
  (e: "select", id: string): void;
  (e: "delete", id: string): void;
}>();

// Search: matches chat titles and message text, case-insensitively.
const query = ref("");
// Two-step delete: the first click on the bin asks "Delete?" in place, the second confirms.
const confirmId = ref<string | null>(null);

function matches(c: Conversation, q: string): boolean {
  if (!q) return true;
  if ((c.title || "").toLowerCase().includes(q)) return true;
  return c.messages.some((m) => m.content.toLowerCase().includes(q));
}

function confirmDelete(id: string) {
  emit("delete", id);
  confirmId.value = null;
}

function startOfDay(ts: number): number {
  const d = new Date(ts);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

const groups = computed(() => {
  const today = startOfDay(Date.now());
  const yesterday = today - 24 * 60 * 60 * 1000;

  const todayList: Conversation[] = [];
  const yesterdayList: Conversation[] = [];
  const olderList: Conversation[] = [];

  // Newest first within each group.
  const q = query.value.trim().toLowerCase();
  const sorted = [...props.conversations].filter((c) => matches(c, q)).sort((a, b) => b.createdAt - a.createdAt);

  for (const c of sorted) {
    const day = startOfDay(c.createdAt);
    if (day === today) todayList.push(c);
    else if (day === yesterday) yesterdayList.push(c);
    else olderList.push(c);
  }

  return [
    { label: "Today", items: todayList },
    { label: "Yesterday", items: yesterdayList },
    { label: "Older", items: olderList },
  ].filter((g) => g.items.length > 0);
});
</script>

<template>
  <aside class="sidebar">
    <button class="new-chat-btn" @click="emit('new-chat')">+ New Chat</button>

    <input
      v-if="conversations.length > 0"
      v-model="query"
      class="search-box"
      type="search"
      placeholder="Search chats..."
      aria-label="Search chats"
    />

    <div class="conversation-list">
      <div v-if="conversations.length === 0" class="empty-hint">No conversations yet</div>
      <div v-else-if="groups.length === 0" class="empty-hint">No chats match "{{ query }}"</div>

      <div v-for="group in groups" :key="group.label" class="group">
        <div class="group-label">{{ group.label }}</div>
        <div
          v-for="c in group.items"
          :key="c.id"
          class="conversation-row"
          :class="{ active: c.id === activeId, confirming: confirmId === c.id }"
        >
          <template v-if="confirmId === c.id">
            <span class="confirm-text">Delete this chat?</span>
            <button class="confirm-btn yes" title="Delete it" @click="confirmDelete(c.id)">Delete</button>
            <button class="confirm-btn" title="Keep it" @click="confirmId = null">Keep</button>
          </template>
          <template v-else>
            <button class="conversation-item" :title="c.title || 'New Chat'" @click="emit('select', c.id)">
              {{ c.title || "New Chat" }}
            </button>
            <button class="row-delete" title="Delete chat" aria-label="Delete chat" @click="confirmId = c.id">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
            </button>
          </template>
        </div>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  width: clamp(168px, 22vw, 240px);
  min-width: 0;
  flex: 0 1 240px;
  background: var(--sidebar-bg, #191919);
  color: #e6e6e6;
  height: 100%;
  box-sizing: border-box;
  padding: 0.6rem;
  gap: 0.6rem;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

@media (max-width: 859px) {
  .sidebar {
    position: absolute;
    z-index: 30;
    top: 0;
    bottom: 0;
    left: 0;
    width: min(260px, calc(100vw - 3rem));
    flex: none;
    box-shadow: 8px 0 24px rgba(0, 0, 0, 0.45);
  }
}

.new-chat-btn {
  width: 100%;
  padding: 0.55rem 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: rgba(255, 255, 255, 0.05);
  color: #fff;
  font-size: 0.9rem;
  cursor: pointer;
  text-align: left;
}

.new-chat-btn:hover {
  background: rgba(255, 255, 255, 0.12);
}

.conversation-list {
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.empty-hint {
  color: #777;
  font-size: 0.82rem;
  padding: 0.4rem 0.2rem;
}

.group-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #888;
  padding: 0.2rem 0.3rem;
}

.search-box {
  width: 100%;
  box-sizing: border-box;
  padding: 0.4rem 0.6rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
  color: #e6e6e6;
  font-size: 0.84rem;
}

.search-box:focus {
  outline: none;
  border-color: rgba(120, 140, 255, 0.6);
}

.conversation-row {
  display: flex;
  align-items: center;
  border-radius: 6px;
  min-height: 2rem;
}

.conversation-row:hover {
  background: rgba(255, 255, 255, 0.08);
}

.conversation-row.active {
  background: rgba(255, 255, 255, 0.14);
}

.conversation-row.confirming {
  background: rgba(200, 60, 60, 0.18);
  padding: 0.2rem 0.3rem;
  gap: 0.3rem;
}

.conversation-item {
  display: block;
  flex: 1;
  min-width: 0;
  text-align: left;
  background: none;
  border: none;
  color: #ddd;
  padding: 0.45rem 0.5rem;
  font-size: 0.85rem;
  border-radius: 6px;
  cursor: pointer;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conversation-row.active .conversation-item {
  color: #fff;
}

.row-delete {
  flex-shrink: 0;
  background: none;
  border: none;
  color: #888;
  padding: 0.3rem 0.45rem;
  cursor: pointer;
  border-radius: 5px;
  /* Takes no room until shown, so chat titles get the full row width. */
  display: none;
}

.conversation-row:hover .row-delete,
.conversation-row.active .row-delete,
.row-delete:focus-visible {
  display: block;
}

.row-delete:hover {
  color: #ff8a8a;
  background: rgba(255, 255, 255, 0.08);
}

.confirm-text {
  flex: 1;
  font-size: 0.8rem;
  color: #f2c2c2;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.confirm-btn {
  flex-shrink: 0;
  font-size: 0.75rem;
  padding: 0.2rem 0.45rem;
  border-radius: 5px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.06);
  color: #ddd;
  cursor: pointer;
}

.confirm-btn.yes {
  background: #a33;
  border-color: #c44;
  color: #fff;
}
</style>
