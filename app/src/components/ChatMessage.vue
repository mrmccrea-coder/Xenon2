<script setup lang="ts">
// ChatMessage.vue -- one message bubble (user or assistant).
//
// Phase 4: edit (user messages) and regenerate (assistant messages) are now real -- clicking edit
// swaps the bubble into an editable textarea; confirming emits `edit` with the new text, and
// App.vue/the Pinia store handle the truncate + re-generate flow. Regenerate just emits
// `regenerate` and the store replaces that one message's content in place.
//
// Phase 7: delete is now real too (see stores/chat.ts's deleteMessage). Deleting a user message
// also removes its immediately-following assistant reply, if any -- a user turn and its reply are
// treated as one deletable unit rather than leaving an orphaned reply with no prompt above it.
// Confirmed via a native confirm() before emitting, since this is destructive and there's no undo.

import { computed, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import type { ChatMessage } from "../types";

const props = defineProps<{
  message: ChatMessage;
  /** True while any generation is in flight app-wide -- disables edit-confirm/regenerate so a
   * second generate_response call can't race the one already running. */
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: "edit", newText: string): void;
  (e: "regenerate"): void;
  (e: "delete"): void;
}>();

const isEditing = ref(false);
const draft = ref("");

// Generated pictures live on disk; load the PNG (as a data URL) whenever this message's image
// path changes, e.g. after Regenerate redraws it.
const imgSrc = ref("");
const imgError = ref("");
watch(
  () => props.message.image,
  async (path) => {
    imgSrc.value = "";
    imgError.value = "";
    if (!path) return;
    try {
      imgSrc.value = await invoke<string>("read_image", { path });
    } catch (err) {
      imgError.value = `(image unavailable: ${String(err)})`;
    }
  },
  { immediate: true }
);

function startEdit() {
  if (props.disabled) return;
  draft.value = props.message.content;
  isEditing.value = true;
}

function confirmEdit() {
  const trimmed = draft.value.trim();
  if (!trimmed || props.disabled) return;
  emit("edit", trimmed);
  isEditing.value = false;
}

function cancelEdit() {
  isEditing.value = false;
  draft.value = props.message.content;
}

function onEditKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    confirmEdit();
  } else if (e.key === "Escape") {
    e.preventDefault();
    cancelEdit();
  }
}

function onRegenerate() {
  if (props.disabled) return;
  emit("regenerate");
}

// Picture requests are stored as "/image <prompt>" (so the chat model sees what was asked for in
// later turns); show them as a labelled request instead of a raw command.
const pictureRequest = computed(() =>
  props.message.role === "user" && props.message.content.startsWith("/image ")
    ? props.message.content.slice("/image ".length)
    : null
);

// Download button on a picture: opens a Save dialog and copies the PNG to the chosen place.
const saveState = ref<"idle" | "saving" | "saved" | "error">("idle");
const savedTo = ref("");
const saveMessage = ref("");
async function downloadImage() {
  if (!props.message.image) return;
  saveState.value = "saving";
  saveMessage.value = "";
  try {
    const dest = await invoke<string | null>("save_image_as", {
      path: props.message.image,
      suggestedName: props.message.imagePrompt ?? null,
    });
    if (!dest) {
      saveState.value = "idle"; // cancelled
      return;
    }
    savedTo.value = dest;
    saveState.value = "saved";
    saveMessage.value = `Saved to ${dest}`;
    window.setTimeout(() => {
      if (saveState.value === "saved") {
        saveState.value = "idle";
        saveMessage.value = "";
      }
    }, 4000);
  } catch (err) {
    saveState.value = "error";
    saveMessage.value = `Could not save: ${String(err)}`;
  }
}

// Copy a reply to the clipboard; the icon turns into a tick for a moment as confirmation.
const copied = ref(false);
async function copyReply() {
  try {
    await navigator.clipboard.writeText(props.message.content);
    copied.value = true;
    window.setTimeout(() => (copied.value = false), 1500);
  } catch (err) {
    console.warn("[ChatMessage] copy failed:", err);
  }
}

function onDelete() {
  if (props.disabled) return;
  const msg =
    "Delete this message" +
    (props.message.role === "user" ? " and its reply" : "") +
    "? This can't be undone.";
  if (!window.confirm(msg)) return;
  emit("delete");
}
</script>

<template>
  <div class="row" :class="message.role">
    <div class="bubble" :class="{ errored: message.errored, editing: isEditing }">
      <template v-if="isEditing">
        <textarea
          v-model="draft"
          class="edit-box"
          rows="2"
          autofocus
          @keydown="onEditKeydown"
        ></textarea>
        <div class="edit-actions">
          <button class="edit-btn confirm" title="Confirm (Enter)" @click="confirmEdit">✓</button>
          <button class="edit-btn cancel" title="Cancel (Esc)" @click="cancelEdit">✕</button>
        </div>
      </template>

      <template v-else>
        <template v-if="message.image">
          <div v-if="imgSrc" class="image-wrap">
            <img class="gen-image" :src="imgSrc" :alt="message.imagePrompt || 'Generated image'" />
            <button
              class="download-btn"
              :class="{ done: saveState === 'saved' }"
              :title="saveState === 'saved' ? `Saved to ${savedTo}` : 'Download this picture'"
              aria-label="Download this picture"
              :disabled="saveState === 'saving'"
              @click="downloadImage"
            >
              <svg v-if="saveState !== 'saved'" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              <svg v-else viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </button>
          </div>
          <div v-else class="content">{{ imgError || "Loading image..." }}</div>
          <div v-if="saveMessage" class="save-message" :class="{ error: saveState === 'error' }">{{ saveMessage }}</div>
          <div class="image-caption">{{ message.imagePrompt }}</div>
        </template>
        <template v-else-if="pictureRequest !== null">
          <div class="picture-tag">Picture request</div>
          <div class="content">{{ pictureRequest }}</div>
        </template>
        <div v-else class="content">
          {{ message.content }}<span v-if="message.streaming" class="cursor">▍</span>
        </div>
        <div v-if="message.edited" class="edited-tag">(edited)</div>
        <div v-if="message.agent === 'sloth'" class="agent-badge sloth">Sloth</div>

        <div class="icons">
          <template v-if="message.role === 'user'">
            <button class="icon-btn" title="Edit" @click="startEdit">✎</button>
            <button class="icon-btn" title="Delete" @click="onDelete">
              <!-- Inline SVG -- 🗑 was found to render as a garbled fallback glyph in this
                   environment's WebView2 (missing color-emoji font support). -->
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <line x1="10" y1="11" x2="10" y2="17" />
                <line x1="14" y1="11" x2="14" y2="17" />
              </svg>
            </button>
          </template>
          <template v-else>
            <button
              v-if="!message.image && !message.streaming && message.content"
              class="icon-btn"
              :title="copied ? 'Copied!' : 'Copy reply'"
              @click="copyReply"
            >
              <svg v-if="!copied" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <svg v-else viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </button>
            <button class="icon-btn" title="Regenerate" @click="onRegenerate">⟳</button>
            <button class="icon-btn" title="Delete" @click="onDelete">
              <!-- Inline SVG -- 🗑 was found to render as a garbled fallback glyph in this
                   environment's WebView2 (missing color-emoji font support). -->
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <line x1="10" y1="11" x2="10" y2="17" />
                <line x1="14" y1="11" x2="14" y2="17" />
              </svg>
            </button>
          </template>
        </div>
      </template>
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

.picture-tag {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.8;
  margin-bottom: 0.15rem;
}

.image-wrap {
  position: relative;
  display: inline-block;
  max-width: 100%;
}

.gen-image {
  display: block;
  max-width: 100%;
  max-height: 60vh;
  border-radius: 8px;
}

/* Download button, top-right corner of the picture. Always visible (discoverable), stronger on hover. */
.download-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.1rem;
  height: 2.1rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  background: rgba(0, 0, 0, 0.45);
  color: #fff;
  cursor: pointer;
  opacity: 0.75;
  transition: opacity 0.15s, background 0.15s;
}

.image-wrap:hover .download-btn,
.download-btn:focus-visible {
  opacity: 1;
}

.download-btn:hover {
  background: rgba(47, 111, 235, 0.85);
}

.download-btn.done {
  background: rgba(40, 160, 90, 0.9);
  opacity: 1;
}

.download-btn:disabled {
  cursor: wait;
}

.save-message {
  margin-top: 0.3rem;
  font-size: 0.75rem;
  color: #8fd19e;
  word-break: break-all;
}

.save-message.error {
  color: #ff9a9a;
}

.image-caption {
  margin-top: 0.35rem;
  font-size: 0.75rem;
  opacity: 0.65;
  font-style: italic;
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

.bubble.editing {
  max-width: 85%;
  width: 70%;
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

.edited-tag {
  margin-top: 0.15rem;
  font-size: 0.7rem;
  opacity: 0.7;
  font-style: italic;
}

.agent-badge {
  display: inline-block;
  margin-top: 0.3rem;
  font-size: 0.65rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
}

.agent-badge.sloth {
  background: #3a3a6b;
  color: #cfcfff;
}

.edit-box {
  width: 100%;
  resize: vertical;
  min-height: 2.6rem;
  padding: 0.4rem 0.5rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  background: rgba(0, 0, 0, 0.2);
  color: inherit;
  font: inherit;
  box-sizing: border-box;
}

.edit-box:focus {
  outline: none;
  border-color: rgba(255, 255, 255, 0.55);
}

.edit-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.35rem;
  margin-top: 0.35rem;
}

.edit-btn {
  border: none;
  border-radius: 5px;
  width: 1.6rem;
  height: 1.6rem;
  font-size: 0.8rem;
  cursor: pointer;
  line-height: 1;
  background: rgba(0, 0, 0, 0.35);
  color: #fff;
}

.edit-btn.confirm:hover {
  background: #2f9e57;
}

.edit-btn.cancel:hover {
  background: #b8402f;
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
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: rgba(0, 0, 0, 0.8);
}
</style>
