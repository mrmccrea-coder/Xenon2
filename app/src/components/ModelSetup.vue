<script setup lang="ts">
// ModelSetup.vue -- "Set up models": what this computer is, which model suits it, and
// one-click download / switch for each tier. Opens automatically when no model is installed.

import { computed } from "vue";
import { useChatStore, type ModelOption } from "../stores/chat";

const store = useChatStore();

const hw = computed(() => store.hardware?.hardware ?? null);
const rec = computed(() => store.hardware?.recommendation ?? null);
const recommended = computed(() => store.models.find((m) => m.recommended) ?? null);
const anyDownloading = computed(() => Object.values(store.downloads).some((d) => d.active));
const noModel = computed(() => store.modelStatus != null && !store.modelStatus.ready);

function gb(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

function pct(id: string): number {
  const d = store.downloads[id];
  return d && d.total > 0 ? Math.min(100, (d.done / d.total) * 100) : 0;
}

function speedText(m: ModelOption): string {
  if (!m.predictedTps) return "";
  const t = m.predictedTps >= 10 ? m.predictedTps.toFixed(0) : m.predictedTps.toFixed(1);
  const word: Record<string, string> = {
    comfortable: "fast enough",
    usable: "usable, a bit slow",
    slow: "slow",
    "too big": "too big for this computer's memory",
  };
  return m.verdict === "too big" ? word[m.verdict] : `~${t} words/sec, ${word[m.verdict] ?? ""}`;
}

function gpuText(): string {
  if (!hw.value) return "";
  if (!hw.value.gpus.length) return "No graphics card detected";
  return hw.value.gpus
    .map((g) => (g.vramMb ? `${g.name} (${gb(g.vramMb)})` : g.name))
    .join(", ");
}

function gpuNote(): string {
  if (!hw.value || !hw.value.gpus.length) return "";
  const usable = hw.value.engineGpuSupport && hw.value.gpus.some((g) => g.vendor === "NVIDIA");
  return usable ? "Used to speed up the larger models." : "This build can only use NVIDIA graphics cards, so the CPU does the work.";
}
</script>

<template>
  <div class="backdrop" @click.self="store.setupOpen = false">
    <div class="panel" role="dialog" aria-label="Set up models">
      <h3>Set up models</h3>

      <div v-if="store.modelStatus?.note" class="banner" :class="store.modelStatus.noteKind">
        {{ store.modelStatus.note }}
      </div>

      <section class="machine">
        <div class="machine-head">
          <strong>This computer</strong>
          <button class="btn small" :disabled="store.hardwareLoading" @click="store.loadHardware(true)">
            {{ store.hardwareLoading ? "Checking..." : "Re-check" }}
          </button>
        </div>
        <template v-if="hw">
          <div class="row"><span>Processor</span><span>{{ hw.cpuName }} ({{ hw.physicalCores }} cores)</span></div>
          <div class="row"><span>Memory</span><span>{{ gb(hw.totalRamMb) }} ({{ gb(hw.availableRamMb) }} free)</span></div>
          <div class="row">
            <span>Graphics</span>
            <span>{{ gpuText() }}<em v-if="gpuNote()"> &mdash; {{ gpuNote() }}</em></span>
          </div>
          <div v-if="rec" class="row">
            <span>Memory speed</span>
            <span>{{ rec.bandwidthGbps.toFixed(1) }} GB/s {{ rec.measured ? "(measured)" : "(estimated; press Re-check to measure)" }}</span>
          </div>
        </template>
        <div v-else class="hint">Reading hardware...</div>
      </section>

      <p v-if="recommended && rec" class="recommend">
        Recommended: <strong>{{ recommended.label }}</strong> &mdash; {{ rec.reason }}.
      </p>

      <section class="tiers">
        <div v-for="m in store.models" :key="m.id" class="tier" :class="{ active: m.active, rec: m.recommended }">
          <div class="tier-top">
            <div>
              <strong>{{ m.label }}</strong>
              <span v-if="m.recommended" class="chip">Recommended</span>
              <span v-if="m.active" class="chip on">In use</span>
              <div class="hint">{{ m.blurb }} &middot; {{ gb(m.sizeMb) }}</div>
              <div v-if="speedText(m)" class="hint" :class="'v-' + m.verdict.replace(' ', '-')">{{ speedText(m) }}</div>
            </div>
            <div class="actions">
              <template v-if="m.active"><span class="hint">Running</span></template>
              <button v-else-if="m.installed" class="btn primary" :disabled="store.generating || store.switchingModel" @click="store.switchModel(m.id)">
                Use this
              </button>
              <template v-else-if="store.downloads[m.id]?.active">
                <button class="btn" @click="store.cancelDownload()">Cancel</button>
              </template>
              <button v-else class="btn primary" :disabled="anyDownloading" @click="store.downloadModel(m.id)">
                {{ store.downloads[m.id]?.error ? "Retry download" : "Download" }}
              </button>
            </div>
          </div>
          <div v-if="store.downloads[m.id]?.active" class="progress">
            <div class="bar"><div class="fill" :style="{ width: pct(m.id) + '%' }"></div></div>
            <div class="hint">{{ gb(Math.round(store.downloads[m.id].done / 1048576)) }} of {{ gb(m.sizeMb) }}</div>
          </div>
          <div v-if="store.downloads[m.id]?.error" class="err">{{ store.downloads[m.id].error }}</div>
        </div>
      </section>

      <p class="hint folder">
        Model files live in: {{ store.modelStatus?.modelsDir }}. You can also copy a model file there
        yourself.
      </p>

      <div class="footer">
        <button class="btn" :disabled="noModel" :title="noModel ? 'Install a model first' : ''" @click="store.setupOpen = false">
          Close
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 120;
}

.panel {
  background: #232325;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  width: min(560px, calc(100vw - 1.5rem));
  max-height: calc(100vh - 1.5rem);
  max-height: calc(100dvh - 1.5rem);
  overflow: auto;
  color: #eee;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
  font-size: 0.85rem;
}

h3 {
  margin: 0 0 0.6rem;
  font-size: 1rem;
}

.banner {
  padding: 0.55rem 0.7rem;
  border-radius: 6px;
  margin-bottom: 0.8rem;
  background: rgba(90, 150, 255, 0.15);
}

.banner.error {
  background: rgba(220, 80, 80, 0.2);
}

.machine {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 0.6rem 0.75rem;
  margin-bottom: 0.8rem;
}

.machine-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.3rem;
}

.row {
  display: flex;
  gap: 0.75rem;
  padding: 0.15rem 0;
}

.row > span:first-child {
  flex: 0 0 6.5rem;
  color: #aaa;
}

.row em {
  color: #aaa;
  font-style: normal;
}

.recommend {
  margin: 0 0 0.7rem;
}

.tier {
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  padding: 0.6rem 0.75rem;
  margin-bottom: 0.5rem;
}

.tier.rec {
  border-color: rgba(90, 150, 255, 0.55);
}

.tier.active {
  background: rgba(90, 150, 255, 0.12);
}

.tier-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
}

.chip {
  margin-left: 0.5rem;
  padding: 0.05rem 0.4rem;
  border-radius: 999px;
  font-size: 0.7rem;
  background: rgba(255, 255, 255, 0.12);
}

.chip.on {
  background: #2f6b3f;
}

.hint {
  color: #aaa;
  font-size: 0.78rem;
}

.v-slow,
.v-too-big {
  color: #e0a458;
}

.progress {
  margin-top: 0.5rem;
}

.bar {
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.12);
  overflow: hidden;
  margin-bottom: 0.2rem;
}

.fill {
  height: 100%;
  background: #5b8def;
  transition: width 0.15s;
}

.err {
  margin-top: 0.4rem;
  color: #ffb4b4;
  font-size: 0.78rem;
}

.folder {
  margin: 0.6rem 0 0;
  word-break: break-all;
}

.footer {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.8rem;
}

.btn {
  background: #33333a;
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #eee;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
  font-size: 0.82rem;
  cursor: pointer;
  white-space: nowrap;
}

.btn.small {
  padding: 0.2rem 0.6rem;
  font-size: 0.75rem;
}

.btn:hover:not(:disabled) {
  background: #3d3d44;
}

.btn.primary {
  background: #5b8def;
  border-color: #5b8def;
  color: #fff;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
