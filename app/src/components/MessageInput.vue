<script setup lang="ts">
// MessageInput.vue -- text box + send button (fully functional) + mic toggle.
//
// Scope boundary (see prompts/phase3_prompt.md): the mic button must exist and be visually
// togglable, but it does NOT capture or process audio here -- Phase 2's STT output gets wired
// into this button in a later integration step, not in this phase. Toggling it only flips a
// local boolean for visual feedback and logs to the console.

import { ref } from "vue";

const props = defineProps<{
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: "send", text: string): void;
}>();

const text = ref("");
const micOn = ref(false);

function submit() {
  const trimmed = text.value.trim();
  if (!trimmed || props.disabled) return;
  emit("send", trimmed);
  text.value = "";
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
}

function toggleMic() {
  micOn.value = !micOn.value;
  console.log(
    `[MessageInput] Mic toggled ${micOn.value ? "on" : "off"} -- visual only, no audio capture ` +
      `in this phase. Phase 2's STT pipeline gets wired into this button later.`
  );
}
</script>

<template>
  <div class="input-bar">
    <button
      class="mic-btn"
      :class="{ on: micOn }"
      title="Voice input (stub -- not wired up yet)"
      @click="toggleMic"
    >
      🎤
    </button>

    <textarea
      v-model="text"
      class="text-box"
      rows="1"
      placeholder="Type a message..."
      :disabled="disabled"
      @keydown="onKeydown"
    ></textarea>

    <button
      class="send-btn"
      :disabled="disabled || !text.trim()"
      title="Send"
      @click="submit"
    >
      ➤
    </button>
  </div>
</template>

<style scoped>
.input-bar {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
  padding: 0.6rem 0.8rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  background: var(--input-bg, #191919);
}

.text-box {
  flex: 1;
  resize: none;
  max-height: 8rem;
  padding: 0.55rem 0.7rem;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: #232324;
  color: #eee;
  font-size: 0.92rem;
  font-family: inherit;
}

.text-box:focus {
  outline: none;
  border-color: #2f6feb;
}

.mic-btn,
.send-btn {
  flex-shrink: 0;
  width: 2.3rem;
  height: 2.3rem;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.mic-btn {
  background: #2a2a2c;
  color: #ccc;
}

.mic-btn.on {
  background: #b8402f;
  color: #fff;
}

.send-btn {
  background: #2f6feb;
  color: #fff;
}

.send-btn:disabled {
  background: #2a2a2c;
  color: #666;
  cursor: default;
}
</style>
