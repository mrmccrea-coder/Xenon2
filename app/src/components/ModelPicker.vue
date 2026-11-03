<script setup lang="ts">
// ModelPicker.vue -- always-visible Fast / Balanced / Quality switch at the top of the chat.

defineProps<{
  models: { id: string; label: string; blurb: string; installed: boolean; active: boolean }[];
  /** True while a reply is generating or a model is loading. */
  busy: boolean;
  loading: boolean;
}>();

defineEmits<{ (e: "select", id: string): void }>();
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
      :disabled="busy || !m.installed || m.active"
      :title="m.installed ? m.blurb : 'Model file not installed'"
      @click="$emit('select', m.id)"
    >
      <span class="pick-label">{{ m.label }}</span>
      <span class="pick-blurb">{{ m.installed ? m.blurb : "not installed" }}</span>
    </button>
    <span v-if="loading" class="pick-status">Loading model...</span>
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

.pick:disabled:not(.active) {
  opacity: 0.5;
  cursor: not-allowed;
}

.pick-label {
  font-weight: 600;
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
</style>
