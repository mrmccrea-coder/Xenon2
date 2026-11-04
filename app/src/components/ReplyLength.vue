<script setup lang="ts">
// ReplyLength.vue -- the "Reply length" control that sits right above the chat.
//
// Auto (the default) lets the backend size each reply to the request ("a 400 word story" gets a
// long budget, "what is 6 times 7" a small one). Picking a fixed length, or typing a custom
// number of tokens, overrides that for every reply until changed. A token is roughly three
// quarters of a word, and generation runs about 15-20 tokens a second on the GPU, so higher
// numbers mean longer waits only for replies that actually use them -- short answers still stop
// by themselves.

import { computed } from "vue";
import { useChatStore } from "../stores/chat";

defineProps<{ disabled?: boolean }>();

const store = useChatStore();

const PRESETS = [
  { label: "Short", tokens: 150 },
  { label: "Medium", tokens: 300 },
  { label: "Long", tokens: 600 },
  { label: "Very long", tokens: 1200 },
  { label: "Max", tokens: 3000 },
];

// "auto" | "custom" | a preset's token count as a string.
const mode = computed({
  get(): string {
    if (store.replyTokens === 0) return "auto";
    return PRESETS.some((p) => p.tokens === store.replyTokens) ? String(store.replyTokens) : "custom";
  },
  set(v: string) {
    if (v === "auto") store.setReplyTokens(0);
    else if (v === "custom") store.setReplyTokens(store.replyTokens > 0 ? store.replyTokens : 450);
    else store.setReplyTokens(Number(v));
  },
});

const approxWords = computed(() => (store.replyTokens > 0 ? Math.round(store.replyTokens * 0.75) : 0));

function onCustom(e: Event) {
  const n = Number((e.target as HTMLInputElement).value);
  if (Number.isFinite(n) && n > 0) store.setReplyTokens(n);
}
</script>

<template>
  <div
    class="reply-length"
    title="How long a reply can get. Auto sizes it to your request (e.g. '400 word story'). Higher = longer replies; short answers still stop on their own."
  >
    <label for="reply-length-select">Reply length</label>
    <select id="reply-length-select" v-model="mode" :disabled="disabled">
      <option value="auto">Auto</option>
      <option v-for="p in PRESETS" :key="p.tokens" :value="String(p.tokens)">
        {{ p.label }} ({{ p.tokens }})
      </option>
      <option value="custom">Custom...</option>
    </select>
    <input
      v-if="mode === 'custom'"
      type="number"
      min="16"
      max="3000"
      step="50"
      :value="store.replyTokens"
      :disabled="disabled"
      aria-label="Custom reply length in tokens"
      @change="onCustom"
    />
    <span v-if="store.replyTokens > 0" class="hint">tokens, about {{ approxWords }} words</span>
    <span v-else class="hint">sized to each request</span>
  </div>
</template>

<style scoped>
.reply-length {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.3rem 0.75rem;
  font-size: 0.78rem;
  color: #aaa;
  flex-shrink: 0;
}
.reply-length label {
  color: #bbb;
}
.reply-length select,
.reply-length input {
  background: #232324;
  color: #ddd;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 6px;
  padding: 0.2rem 0.4rem;
  font-size: 0.78rem;
}
.reply-length input {
  width: 5.5rem;
}
.hint {
  opacity: 0.6;
}
</style>
