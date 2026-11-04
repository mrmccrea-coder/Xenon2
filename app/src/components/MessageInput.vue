<script setup lang="ts">
// MessageInput.vue -- text box + send button + mic button, all fully functional (Phase 7).
//
// Clicking the mic starts real VAD-gated mic capture via the voice-pipeline sidecar
// (`voice_listen`, see app/src-tauri/src/voice.rs): it blocks (from this component's
// perspective, as an awaited invoke() call) until speech is detected and transcribed, or a
// listen/utterance timeout is hit. A successful transcript is emitted as `voice-send`, which
// App.vue feeds into the exact same store.sendMessage path typed text uses -- not a separate
// voice-only code path (per phase7_prompt.md task 1).

import { computed, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { useChatStore } from "../stores/chat";
import ModelPicker from "./ModelPicker.vue";

// Phase 7 follow-up: Goldfish/Sloth agent toggle. (Shown as "Goldfish" since 2026-11-04; it was
// labelled "Dementia", which could offend. The stored agent id is still "dementia" so saved
// conversations keep loading -- this is a display-name change only.) Reads/writes the store directly (like the mic
// button's own invoke() calls above) rather than round-tripping through App.vue -- this
// component already isn't purely presentational.
const store = useChatStore();

const props = defineProps<{
  disabled?: boolean;
  /** Whether the voice sidecar has finished loading its models (VAD/STT/TTS) and can actually
   * service a `voice_listen` call yet -- see stores/chat.ts's voiceReady/refreshVoiceReady. The
   * mic button stays clickable either way (a call made before this is true just queues behind
   * the sidecar's own startup instead of failing), but is visually dimmed with an explanatory
   * title so it doesn't look broken during the few seconds of startup. */
  voiceReady?: boolean;
  /** True while the app's own spoken reply is queued or still audible through the speakers.
   * The mic button stays disabled for this whole window -- see App.vue/stores/chat.ts's
   * `speaking` and `voice-speak-done` handling, and ipc_server.py's `_wait_out_speaker_feedback`
   * for why opening the mic during/right after TTS playback is a real feedback bug, not just a
   * theoretical one. */
  speaking?: boolean;
}>();

const emit = defineEmits<{
  (e: "send", text: string): void;
  (e: "voice-send", text: string): void;
  (e: "setup"): void;
}>();

const text = ref("");

// A chat reply is being written (not a picture: drawing can't be interrupted).
const canStop = computed(() => store.generating && Object.keys(store.streamingMessageIds).length > 0);

// One plain-language line under the input saying what will happen with the next message.
const modeHint = computed(() => {
  if (canStop.value) return "Writing a reply... press the square to stop it early.";
  if (store.generating && store.imageMode) return "Drawing your picture. Chat is paused until it finishes (about half a minute).";
  if (store.imageMode) return "Image mode: describe a picture and Xenon will draw it. Click Image again to go back to chatting.";
  if (store.activeAgent === "sloth") return "Sloth remembers facts about you across all your chats.";
  return "Goldfish only remembers this chat. Switch to Sloth to have Xenon remember you across chats.";
});
const listening = ref(false);
const micStatus = ref("");

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

async function toggleMic() {
  if (props.disabled || props.speaking || listening.value) return;
  listening.value = true;
  micStatus.value = "Listening...";
  try {
    const result = await invoke<{ ok: boolean; text?: string; reason?: string }>("voice_listen", {
      maxListenSec: 6,
      maxUtteranceSec: 10,
    });
    if (result.ok && result.text) {
      emit("voice-send", result.text);
      micStatus.value = "";
    } else {
      micStatus.value = result.reason === "no_speech_timeout" ? "No speech detected" : "Didn't catch that";
      window.setTimeout(() => {
        if (!listening.value) micStatus.value = "";
      }, 2000);
    }
  } catch (err) {
    console.warn("[MessageInput] voice_listen failed:", err);
    micStatus.value = "Voice input unavailable";
    window.setTimeout(() => {
      if (!listening.value) micStatus.value = "";
    }, 3000);
  } finally {
    listening.value = false;
  }
}
</script>

<template>
  <div class="input-bar">
    <ModelPicker @setup="emit('setup')" />

    <div class="agent-toggle" title="Which agent handles the next message">
      <button
        class="agent-btn"
        :class="{ active: store.activeAgent === 'dementia' }"
        :disabled="disabled"
        title="Goldfish: remembers only this chat, nothing carries over to other chats"
        @click="store.setActiveAgent('dementia')"
      >
        Goldfish
      </button>
      <button
        class="agent-btn"
        :class="{ active: store.activeAgent === 'sloth' }"
        :disabled="disabled"
        title="Sloth: remembers facts across every conversation"
        @click="store.setActiveAgent('sloth')"
      >
        Sloth
      </button>
    </div>

    <div v-if="store.imageAvailable" class="agent-toggle image-toggle">
      <button
        class="agent-btn"
        :class="{ active: store.imageMode }"
        :disabled="disabled"
        :title="
          store.imageMode
            ? 'Image mode ON: your next message is drawn as a picture (chat model is paused while it draws)'
            : 'Turn on image mode to draw pictures instead of chatting'
        "
        @click="store.setImageMode(!store.imageMode)"
      >
        Image
      </button>
      <select
        v-if="store.imageMode"
        v-model.number="store.imageSize"
        class="image-select"
        :disabled="disabled"
        title="Picture size (larger takes longer)"
      >
        <option :value="512">512</option>
        <option :value="768">768</option>
        <option :value="1024">1024</option>
      </select>
      <select
        v-if="store.imageMode && store.imageModels.length > 1"
        v-model="store.imageModel"
        class="image-select"
        :disabled="disabled"
        title="Image model"
      >
        <option v-for="m in store.imageModels" :key="m.file" :value="m.file">{{ m.label }}</option>
      </select>
    </div>

    <div class="mic-wrap">
      <button
        class="mic-btn"
        :class="{ on: listening }"
        :disabled="speaking"
        :title="
          speaking
            ? 'Voice input is disabled while Xenon is speaking'
            : voiceReady === false
              ? 'Voice input is still starting up...'
              : 'Voice input'
        "
        @click="toggleMic"
      >
        <!-- Inline SVG, not an emoji glyph -- the 🎤 emoji was found to render as an unrecognizable
             garbled fallback glyph on this environment's WebView2 (missing color-emoji font
             support), making the mic button impossible to identify visually despite being fully
             wired up. SVG renders identically regardless of what emoji fonts are installed. -->
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      </button>
      <div v-if="micStatus" class="mic-status">{{ micStatus }}</div>
    </div>

    <textarea
      v-model="text"
      class="text-box"
      rows="1"
      :placeholder="store.imageMode ? 'Describe the picture you want...' : 'Type a message...'"
      :disabled="disabled"
      @keydown="onKeydown"
    ></textarea>

    <button
      v-if="canStop"
      class="send-btn stop-btn"
      title="Stop writing this reply (keeps what's written so far)"
      aria-label="Stop"
      @click="store.stopGeneration()"
    >
      <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2" /></svg>
    </button>
    <button
      v-else
      class="send-btn"
      :disabled="disabled || !text.trim()"
      title="Send (Enter). Shift+Enter for a new line"
      @click="submit"
    >
      ➤
    </button>

    <div class="mode-hint">{{ modeHint }}</div>
  </div>
</template>

<style scoped>
.input-bar {
  display: flex;
  align-items: flex-end;
  flex-wrap: wrap;
  flex-shrink: 0;
  gap: 0.5rem;
  padding: 0.6rem 0.8rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  background: var(--input-bg, #191919);
  max-width: 100%;
}

.text-box {
  flex: 1 1 140px;
  min-width: 0;
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

.agent-toggle {
  display: flex;
  flex-shrink: 0;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.15);
}

.agent-btn {
  background: #232324;
  color: #999;
  border: none;
  padding: 0 0.6rem;
  height: 2.3rem;
  font-size: 0.75rem;
  cursor: pointer;
}

.image-select {
  background: #232324;
  color: #ccc;
  border: none;
  border-left: 1px solid rgba(255, 255, 255, 0.15);
  height: 2.3rem;
  font-size: 0.75rem;
  max-width: 9rem;
}
.agent-btn.active {
  background: #3a3a6b;
  color: #fff;
}

.agent-btn:disabled {
  cursor: default;
  opacity: 0.6;
}

.mic-wrap {
  position: relative;
  flex-shrink: 0;
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
  animation: mic-pulse 1s ease-in-out infinite;
}

.mic-btn:disabled {
  opacity: 0.45;
  cursor: default;
}

@keyframes mic-pulse {
  50% {
    box-shadow: 0 0 0 6px rgba(184, 64, 47, 0.25);
  }
}

.mic-status {
  position: absolute;
  bottom: 2.7rem;
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
  background: #232325;
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #ccc;
  font-size: 0.72rem;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.send-btn {
  background: #2f6feb;
  color: #fff;
}
.stop-btn {
  background: #c0392b;
}
.stop-btn:hover {
  background: #d9473a;
}
.mode-hint {
  flex-basis: 100%;
  font-size: 0.74rem;
  color: #8a8a8a;
  padding: 0 0.2rem;
  margin-top: -0.15rem;
}

.send-btn:disabled {
  background: #2a2a2c;
  color: #666;
  cursor: default;
}
</style>
