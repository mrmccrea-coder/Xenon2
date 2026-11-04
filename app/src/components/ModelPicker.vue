<script setup lang="ts">
// ModelPicker.vue -- one compact "model" button that lives in the input bar, next to where the
// user types. Clicking it opens a small menu (upward, above the input) listing Fast / Balanced /
// Quality with their measured speed, a tick on the active one, the recommended star, and a
// "Set up models..." entry. Replaces the three large cards that used to sit across the top of
// the chat (2026-11-04).
//
// Not tied to the input's own `disabled` state on purpose: when no model is loaded the input is
// disabled, and this button is exactly how the user gets out of that state.

import { computed, onMounted, onUnmounted, ref } from "vue";
import { useChatStore, type ModelOption } from "../stores/chat";

const store = useChatStore();
const emit = defineEmits<{ (e: "setup"): void }>();

const open = ref(false);
const root = ref<HTMLElement | null>(null);

const active = computed(() => store.models.find((m) => m.active) ?? null);
const busy = computed(() => store.generating || store.switchingModel);
const buttonLabel = computed(() => {
  if (store.switchingModel) return "Loading...";
  if (active.value) return active.value.label;
  return store.models.length ? "Choose model" : "Set up model";
});

function speed(m: ModelOption): string {
  if (!m.predictedTps) return "";
  const t = m.predictedTps >= 10 ? m.predictedTps.toFixed(0) : m.predictedTps.toFixed(1);
  return `~${t} words/sec`;
}

function subtitle(m: ModelOption): string {
  if (!m.installed) return "Not installed: click to download";
  const s = speed(m);
  return s ? `${m.blurb} · ${s}` : m.blurb;
}

function toggle() {
  if (!store.models.length) {
    emit("setup");
    return;
  }
  open.value = !open.value;
}

function choose(m: ModelOption) {
  open.value = false;
  if (!m.installed) {
    emit("setup");
    return;
  }
  if (!m.active) void store.switchModel(m.id);
}

function openSetup() {
  open.value = false;
  emit("setup");
}

function onDocMouseDown(e: MouseEvent) {
  if (open.value && root.value && !root.value.contains(e.target as Node)) open.value = false;
}
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape") open.value = false;
}
onMounted(() => {
  document.addEventListener("mousedown", onDocMouseDown);
  document.addEventListener("keydown", onKey);
});
onUnmounted(() => {
  document.removeEventListener("mousedown", onDocMouseDown);
  document.removeEventListener("keydown", onKey);
});
</script>

<template>
  <div ref="root" class="model-menu">
    <button
      class="model-btn"
      :disabled="busy"
      :title="active ? `Model: ${active.label} (${active.blurb}). Click to change.` : 'Choose or set up a model'"
      aria-haspopup="menu"
      :aria-expanded="open"
      @click="toggle"
    >
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="5" y="5" width="14" height="14" rx="2" />
        <rect x="9" y="9" width="6" height="6" />
        <path d="M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4" />
      </svg>
      <span class="model-name">{{ buttonLabel }}</span>
      <span v-if="active?.recommended" class="star" title="Recommended for this computer">★</span>
      <span class="caret" aria-hidden="true">▴</span>
    </button>

    <div v-if="open" class="menu" role="menu" aria-label="Model">
      <div class="menu-title">Model</div>
      <button
        v-for="m in store.models"
        :key="m.id"
        class="menu-item"
        :class="{ active: m.active, missing: !m.installed }"
        role="menuitemradio"
        :aria-checked="m.active"
        @click="choose(m)"
      >
        <span class="check" aria-hidden="true">{{ m.active ? "✓" : "" }}</span>
        <span class="item-main">
          <span class="item-label">
            {{ m.label }}<span v-if="m.recommended" class="star" title="Recommended for this computer"> ★</span>
          </span>
          <span class="item-sub">{{ subtitle(m) }}</span>
        </span>
      </button>
      <div class="sep"></div>
      <button class="menu-item setup" role="menuitem" @click="openSetup">
        <span class="check" aria-hidden="true"></span>
        <span class="item-main"><span class="item-label">Set up models...</span></span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.model-menu {
  position: relative;
  flex-shrink: 0;
}

.model-btn {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  height: 2.3rem;
  padding: 0 0.65rem;
  background: #232324;
  color: #ccc;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  font: inherit;
  font-size: 0.78rem;
  cursor: pointer;
}

.model-btn:hover:not(:disabled) {
  background: #2c2c2e;
  color: #fff;
}

.model-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.model-name {
  white-space: nowrap;
}

.caret {
  font-size: 0.65rem;
  opacity: 0.7;
}

.star {
  color: #f2c14e;
}

.menu {
  position: absolute;
  bottom: calc(100% + 6px);
  left: 0;
  z-index: 40;
  min-width: 270px;
  padding: 0.35rem;
  background: #1f1f22;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.menu-title {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #888;
  padding: 0.3rem 0.5rem 0.4rem;
}

.menu-item {
  display: flex;
  align-items: flex-start;
  gap: 0.45rem;
  width: 100%;
  text-align: left;
  padding: 0.45rem 0.5rem;
  background: none;
  border: none;
  border-radius: 7px;
  color: #ddd;
  font: inherit;
  cursor: pointer;
}

.menu-item:hover {
  background: rgba(255, 255, 255, 0.08);
}

.menu-item.active {
  background: rgba(90, 150, 255, 0.16);
  color: #fff;
}

.menu-item.missing {
  opacity: 0.7;
}

.check {
  width: 0.9rem;
  flex-shrink: 0;
  color: #8fb4ff;
  font-size: 0.85rem;
  line-height: 1.3;
}

.item-main {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.item-label {
  font-size: 0.86rem;
  font-weight: 600;
}

.item-sub {
  font-size: 0.74rem;
  color: #9a9a9a;
}

.sep {
  height: 1px;
  margin: 0.3rem 0.4rem;
  background: rgba(255, 255, 255, 0.1);
}

.menu-item.setup .item-label {
  font-weight: 400;
  color: #bbb;
}
</style>
