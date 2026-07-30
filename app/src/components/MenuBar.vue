<script setup lang="ts">
// MenuBar.vue -- File menu.
//
// Scope boundary (see prompts/phase3_prompt.md): only "New" is functional this phase. Open /
// Save / Save As / Export Memory are intentional visual stubs -- their real logic belongs to
// Phase 5 (file save/load) and Phase 6 (external memory export), not here. Clicking them just
// logs to the console so it's easy to confirm in devtools that nothing silently "works".

import { ref } from "vue";

const emit = defineEmits<{
  (e: "new"): void;
}>();

const fileMenuOpen = ref(false);

function toggleFileMenu() {
  fileMenuOpen.value = !fileMenuOpen.value;
}

function closeFileMenu() {
  fileMenuOpen.value = false;
}

function onNew() {
  emit("new");
  closeFileMenu();
}

// Intentional stub -- see file header. `feature` names the later phase that owns the real
// behavior so it's obvious in devtools this isn't a bug, it's scope.
function stub(item: string, phase: string) {
  console.log(`[MenuBar] "${item}" is a Phase 3 visual stub -- real logic lands in ${phase}.`);
  closeFileMenu();
}
</script>

<template>
  <div class="menu-bar" @keydown.escape="closeFileMenu">
    <div class="menu-item" tabindex="0" @click="toggleFileMenu" @blur="closeFileMenu">
      File
      <div v-if="fileMenuOpen" class="dropdown" @mousedown.prevent>
        <button class="dropdown-item" @click="onNew">New</button>
        <button class="dropdown-item stub" @click="stub('Open', 'Phase 5')">Open...</button>
        <button class="dropdown-item stub" @click="stub('Save', 'Phase 5')">Save</button>
        <button class="dropdown-item stub" @click="stub('Save As', 'Phase 5')">Save As...</button>
        <div class="dropdown-sep"></div>
        <button class="dropdown-item stub" @click="stub('Export Memory', 'Phase 6')">
          Export Memory...
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.menu-bar {
  display: flex;
  align-items: stretch;
  background: var(--menu-bg, #1e1e1f);
  color: var(--menu-fg, #ddd);
  font-size: 0.85rem;
  user-select: none;
  -webkit-app-region: drag;
}

.menu-item {
  position: relative;
  padding: 0.4rem 0.75rem;
  cursor: default;
  -webkit-app-region: no-drag;
  outline: none;
}

.menu-item:hover,
.menu-item:focus {
  background: rgba(255, 255, 255, 0.08);
}

.dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 170px;
  background: #2a2a2c;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  padding: 4px;
  z-index: 50;
}

.dropdown-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  color: #eee;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;
  border-radius: 4px;
  cursor: pointer;
}

.dropdown-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.dropdown-item.stub {
  color: #aaa;
}

.dropdown-sep {
  height: 1px;
  margin: 4px 2px;
  background: rgba(255, 255, 255, 0.1);
}
</style>
