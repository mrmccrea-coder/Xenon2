<script setup lang="ts">
// MenuBar.vue -- File menu.
//
// Phase 5: Open / Save / Save As are now real (see stores/chat.ts's openConversation /
// saveConversation / saveConversationAs), wired the same way "New" already was -- this component
// just emits, App.vue calls the store. Save / Save As are disabled when there's no active
// conversation to save. Export Memory remains an intentional visual stub -- its real logic is
// Phase 6's job, not this phase's.

import { ref } from "vue";

const props = defineProps<{
  /** Disables Save / Save As when there's no active conversation. */
  hasActiveConversation?: boolean;
}>();

const emit = defineEmits<{
  (e: "new"): void;
  (e: "open"): void;
  (e: "save"): void;
  (e: "save-as"): void;
  (e: "export-memory"): void;
  (e: "import-memory"): void;
  (e: "settings"): void;
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

function onOpen() {
  emit("open");
  closeFileMenu();
}

function onSave() {
  if (!props.hasActiveConversation) return;
  emit("save");
  closeFileMenu();
}

function onSaveAs() {
  if (!props.hasActiveConversation) return;
  emit("save-as");
  closeFileMenu();
}

function onExportMemory() {
  emit("export-memory");
  closeFileMenu();
}

function onImportMemory() {
  emit("import-memory");
  closeFileMenu();
}

function onSettings() {
  emit("settings");
  closeFileMenu();
}
</script>

<template>
  <div class="menu-bar" @keydown.escape="closeFileMenu">
    <div class="menu-item" tabindex="0" @click="toggleFileMenu" @blur="closeFileMenu">
      File
      <div v-if="fileMenuOpen" class="dropdown" @mousedown.prevent>
        <button class="dropdown-item" @click="onNew">New</button>
        <button class="dropdown-item" @click="onOpen">Open...</button>
        <button
          class="dropdown-item"
          :class="{ disabled: !hasActiveConversation }"
          @click="onSave"
        >
          Save
        </button>
        <button
          class="dropdown-item"
          :class="{ disabled: !hasActiveConversation }"
          @click="onSaveAs"
        >
          Save As...
        </button>
        <div class="dropdown-sep"></div>
        <button class="dropdown-item" @click="onExportMemory">Export Memory...</button>
        <button class="dropdown-item" @click="onImportMemory">Import Memory...</button>
        <div class="dropdown-sep"></div>
        <button class="dropdown-item" @click="onSettings">Data Directory Settings...</button>
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

.dropdown-item.disabled {
  color: #666;
  cursor: default;
}

.dropdown-item.disabled:hover {
  background: none;
}

.dropdown-sep {
  height: 1px;
  margin: 4px 2px;
  background: rgba(255, 255, 255, 0.1);
}
</style>
