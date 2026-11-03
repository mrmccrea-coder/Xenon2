<script setup lang="ts">
// ModelPicker.vue -- always-visible Fast / Balanced / Quality switch at the top of the chat.

import type { ModelOption } from "../stores/chat";

defineProps<{
  models: ModelOption[];
  /** True while a reply is generating or a model is loading. */
  busy: boolean;
  loading: boolean;
}>();

defineEmits<{ (e: "select", id: string): void; (e: "setup"): void }>();

function speed(m: ModelOption): string {
  if (!m.installed) return "not installed";
  if (!m.predictedTps) return m.blurb;
  const t = m.predictedTps >= 10 ? m.predictedTps.toFixed(0) : m.predictedTps.toFixed(1);
  return `~${t} words/sec`;
}
</script>

<template>
  <div v-if="models.length" class="model-picker" role="radiogroup" aria-label="Model">
    <button
      v-for="m in models"
      :key="m.id"
      class="pick"
      :class="{ active: m.active, missing: !m.installed }"
      role="radio"
      :aria-checked="m.active"
      :disabled="m.installed && (busy || m.active)"
      :title="m.installed ? m.blurb : 'Not installed yet: click to download'"
      @click="m.installed ? $emit('select', m.id) : $emit('setup')"
    >
      <span class="pick-label">
        {{ m.label }}<span v-if="m.recommended" class="star" title="Recommended for this computer"> ★</span>
      </span>
      <span class="pick-blurb">{{ speed(m) }}</span>
    </button>
    <span v-if="loading" class="pick-status">Loading model...</span>
    <button class="gear" title="Set up models and see what suits this computer" @click="$emit('setup')">
      Set up
    </button>
  </div>
</template>

<style scoped>
.model-picker {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 0 0.2rem;
}

.pick {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.1rem;
  padding: 0.5rem 0.6rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  color: #ddd;
  font: inherit;
  cursor: pointer;
}

.pick:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.1);
}

.pick.active {
  background: rgba(90, 150, 255, 0.22);
  border-color: rgba(90, 150, 255, 0.8);
  color: #fff;
  cursor: default;
}

.pick.missing {
  opacity: 0.65;
  border-style: dashed;
}

.pick:disabled:not(.active) {
  opacity: 0.5;
  cursor: not-allowed;
}

.pick-label {
  font-weight: 600;
}

.star {
  color: #f2c14e;
}

.pick-blurb {
  font-size: 0.75rem;
  color: #aaa;
}

.pick-status {
  font-size: 0.8rem;
  color: #aaa;
  white-space: nowrap;
}

.gear {
  align-self: stretch;
  padding: 0 0.7rem;
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  color: #bbb;
  font: inherit;
  font-size: 0.8rem;
  cursor: pointer;
}

.gear:hover {
  background: rgba(255, 255, 255, 0.08);
  color: #fff;
}
</style>
