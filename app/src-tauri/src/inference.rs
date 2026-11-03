//! Safe(r) wrapper around the Phase 1 `xenon_inference` C API, plus the Tauri IPC command that
//! drives token-by-token streaming into the chat UI.
//!
//! This is the ONLY place in the app that talks to Phase 1's compiled library. It does not
//! reimplement or stub inference -- `generate_reply` calls the real `xenon_generate()` from
//! `inference-engine`, loaded once at app startup via `xenon_load_model()`.

use std::cell::RefCell;
use std::ffi::{c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::ffi;
use crate::hardware;
use crate::models_dl;
use crate::sloth_memory;

/// One turn of conversation, as sent from the Vue frontend. Mirrors the shape of a chat message
/// but deliberately doesn't carry UI-only fields (ids, timestamps, streaming flags, etc) -- this
/// is a minimal payload just for prompt construction.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String, // "user" | "assistant"
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
struct TokenEvent {
    #[serde(rename = "conversationId")]
    conversation_id: String,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationDoneEvent {
    #[serde(rename = "conversationId")]
    conversation_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationErrorEvent {
    #[serde(rename = "conversationId")]
    conversation_id: String,
    message: String,
}

/// Raw engine pointer. `xenon_engine*` is safe to hand off across threads as long as access is
/// serialized, which the Mutex around `EngineInner` guarantees.
pub(crate) struct EnginePtr(*mut ffi::XenonEngine);
unsafe impl Send for EnginePtr {}

/// Raw incremental-state pointer (Phase 8, see `xenon_inference.h`'s `xenon_state` docs). Frees
/// itself via `xenon_state_free` on drop so every place that allocates one (a cache entry, a
/// per-turn scratch copy) can just let it go out of scope instead of remembering to free it.
pub(crate) struct XenonStatePtr(*mut ffi::XenonState);
unsafe impl Send for XenonStatePtr {}

impl Drop for XenonStatePtr {
    fn drop(&mut self) {
        unsafe { ffi::xenon_state_free(self.0) };
    }
}

/// One conversation's cached incremental state, plus the exact turns it has already consumed --
/// used to diff against an incoming `history` array and decide "extend" vs. "rebuild" (see
/// `ensure_cache_extended`). `consumed` never includes the newest, not-yet-answered user turn,
/// and is only ever advanced from caller-confirmed history text, never from this turn's own
/// generation output -- see `generate_blocking`'s doc comment for why that matters.
struct CachedConversation {
    consumed: Vec<ChatTurn>,
    state: XenonStatePtr,
}

/// Phase 9: a completed Sloth turn's data, handed back from `generate_blocking` instead of being
/// acted on immediately, so `generate_response` can run `run_fact_extraction` in a background
/// task *after* releasing the engine lock instead of holding it across a second generate call
/// the visible reply doesn't depend on. Owned `String`s, not borrows -- this must be able to
/// outlive `history` and run correctly no matter how much later (and after how many other turns)
/// it actually executes, see `run_fact_extraction`'s doc comment.
struct PendingExtraction {
    user_text: String,
    reply_text: String,
}

/// Bound on how many conversations' incremental state is kept alive at once. Each entry is one
/// `xenon_get_state_len()`-sized buffer (measured at ~20.6 MB for the RWKV-7 2.9B model -- see
/// `inference-engine/README.md`), so this caps the feature's steady-state memory cost at
/// roughly 4 x that, plus the two engine-level static states -- trivial next to the multi-GB
/// model weights already resident. Least-recently-used entries are evicted first (see
/// `ensure_cache_extended`).
const MAX_CACHED_CONVERSATIONS: usize = 4;

/// Everything needed to run generation, guarded by one mutex so only one generation (or cache
/// mutation) runs at a time -- rwkv.cpp's per-engine state is not designed for concurrent calls,
/// and Phase 8's incremental states are per-conversation data guarded alongside it, not a
/// separate lock (see this struct's fields). When no model is installed (or loading failed) the
/// engine pointer is null; see `EngineInner::empty` and `is_loaded`.
pub(crate) struct EngineInner {
    engine: EnginePtr,
    /// The conversation preamble (instruction + demo turns) -- byte-identical on every call ever
    /// made, so it's prefilled once here instead of every turn. See `inference.rs`'s
    /// `STATIC_PREFIX` and `build_prompt`'s Phase 8 doc comment for why this text can be first
    /// and still cacheable (unlike the volatile header, which can't).
    static_prefix: XenonStatePtr,
    /// The Sloth fact-extraction few-shot preamble, prefilled once the same way. Entirely
    /// separate from `conversation_cache` below and never touched by conversation generation --
    /// extraction must never be able to corrupt a conversation's cached state (see
    /// `run_fact_extraction`).
    extraction_prefix: XenonStatePtr,
    /// LRU cache of per-conversation incremental state, keyed by conversation id. A `Vec` (not a
    /// `HashMap`) because eviction order is exactly insertion/access order and there are at most
    /// `MAX_CACHED_CONVERSATIONS` entries -- a linear scan over a handful of entries is simpler
    /// and just as fast as a HashMap + separate LRU list here.
    conversation_cache: Vec<(String, CachedConversation)>,
    /// File this engine was loaded from; lets a failed low-memory switch reload the old model.
    model_path: PathBuf,
}

impl EngineInner {
    /// A placeholder with nothing loaded. Freeing null pointers is a documented no-op in
    /// `xenon_inference.h`, so dropping this is safe.
    fn empty() -> Self {
        EngineInner {
            engine: EnginePtr(std::ptr::null_mut()),
            static_prefix: XenonStatePtr(std::ptr::null_mut()),
            extraction_prefix: XenonStatePtr(std::ptr::null_mut()),
            conversation_cache: Vec::new(),
            model_path: PathBuf::new(),
        }
    }

    fn is_loaded(&self) -> bool {
        !self.engine.0.is_null()
    }
}

impl Drop for EngineInner {
    /// Per-conversation states are plain buffers that do not reference the engine, so freeing
    /// the engine here and the states as fields drop afterwards is safe in either order.
    fn drop(&mut self) {
        unsafe { ffi::xenon_free_engine(self.engine.0) };
    }
}

/// `Arc` (not a bare `Mutex`) because the IPC command below needs to clone a handle to it into a
/// `spawn_blocking` closure and hold the lock for the *entire* generate() call, not just long
/// enough to read the pointer out -- rwkv.cpp's per-engine state isn't safe for concurrent eval
/// calls, so the lock must stay held across the whole blocking generation, not be dropped early.
#[derive(Clone)]
pub struct EngineState(pub Arc<Mutex<EngineInner>>);

/// The loaded model's name/version (its filename, minus extension, e.g.
/// `"rwkv-5-world-0.4B-Q4_0"`), recorded so Phase 5's saved conversation files can note which
/// model generated them (see `persistence::ConversationFile::model`). Computed once from the
/// actual path `load_engine` loads, so it can never drift from what's really running. Empty when
/// no model is loaded.
pub struct ModelInfo(pub Mutex<String>);

/// Short label for the menu bar. "CUDA — all layers on GPU" when offload succeeded.
pub struct DeviceInfo(pub Mutex<String>);

#[tauri::command]
pub fn get_model_name(model_info: tauri::State<'_, ModelInfo>) -> String {
    model_info.0.lock().map(|g| g.clone()).unwrap_or_default()
}

#[tauri::command]
pub fn get_inference_device(device_info: tauri::State<'_, DeviceInfo>) -> String {
    device_info.0.lock().map(|g| g.clone()).unwrap_or_default()
}

/// One selectable model tier. `candidates` are filenames under `models/`, tried in order -- the
/// first one that exists is the tier's model, so a tier can fall back to an older file until its
/// preferred one has been downloaded/converted.
struct ModelTier {
    id: &'static str,
    label: &'static str,
    blurb: &'static str,
    candidates: &'static [&'static str],
}

const MODEL_TIERS: [ModelTier; 3] = [
    ModelTier {
        id: "fast",
        label: "Fast",
        blurb: "~0.4B, quick but error-prone",
        candidates: &["rwkv7-g1-0.4B-Q5_1.bin", "rwkv-5-world-0.4B-Q4_0.bin"],
    },
    ModelTier {
        id: "balanced",
        label: "Balanced",
        blurb: "~1.5B, good all-rounder",
        candidates: &["rwkv7-g1-1.5B-Q5_1.bin"],
    },
    ModelTier {
        id: "quality",
        label: "Quality",
        blurb: "~2.9B, best replies",
        candidates: &["rwkv7-g1-2.9B-Q5_1.bin", "rwkv-7-world-2.9B-Q5_1.bin"],
    },
];

fn tier_by_id(id: &str) -> Option<&'static ModelTier> {
    MODEL_TIERS.iter().find(|t| t.id == id)
}

fn tier_model_path(repo_root: &PathBuf, tier: &ModelTier) -> Option<PathBuf> {
    let dir = repo_root.join("models");
    tier.candidates.iter().map(|f| dir.join(f)).find(|p| p.exists())
}

fn vocab_path_for(repo_root: &PathBuf) -> PathBuf {
    crate::paths::resolve_vocab(repo_root)
}

fn choice_file(app: &AppHandle) -> Option<PathBuf> {
    use tauri::Manager;
    app.path().app_data_dir().ok().map(|d| d.join("model_choice.json"))
}

fn parse_saved_tier(raw: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    v.get("tier")?.as_str().map(|s| s.to_string())
}

/// The tier saved by the last `switch_model` (or first-run auto-selection), if any. Kept in its
/// own file (not `Settings`) because the frontend rewrites the whole settings struct whenever the
/// data directory changes.
fn saved_tier_id(app: &AppHandle) -> Option<String> {
    parse_saved_tier(&std::fs::read_to_string(choice_file(app)?).ok()?)
}

fn save_tier_id(app: &AppHandle, id: &str, auto: bool) -> Result<(), String> {
    let path = choice_file(app).ok_or("Could not resolve the app data directory")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Could not create '{}': {e}", dir.display()))?;
    }
    std::fs::write(&path, serde_json::json!({ "tier": id, "auto": auto }).to_string())
        .map_err(|e| format!("Could not write '{}': {e}", path.display()))
}

/// Switching loads the new model while the old one is still resident, which needs both in RAM at
/// once. When that would not leave comfortable headroom, release the old model first instead.
fn should_unload_first(available_mb: u64, new_file_mb: u64) -> bool {
    available_mb < new_file_mb * 13 / 10 + 1024
}

/// Loads `model_path`, prefills the two static preambles, and returns the ready-to-use engine
/// bundle plus the model's name (its filename minus extension). Returns `Err` instead of
/// panicking so a failed runtime switch leaves the currently loaded model untouched.
fn build_engine_inner(model_path: &PathBuf, vocab_path: &PathBuf) -> Result<(EngineInner, String, String), String> {
    let model_name = model_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown-model".to_string());

    let model_path_c = CString::new(model_path.to_string_lossy().as_bytes())
        .map_err(|_| "model path contains NUL byte".to_string())?;
    let vocab_path_c = CString::new(vocab_path.to_string_lossy().as_bytes())
        .map_err(|_| "vocab path contains NUL byte".to_string())?;

    // GPU offload is preferred when this binary was built with CUDA (see build.rs, which links
    // build-cuda-app ahead of build-cpu-app). An RTX 2060 has 6GB; this Q5_1 file is ~2.75GB
    // and the RWKV state is ~21MB, so every layer fits. If CUDA init fails, load retries on CPU
    // so a CUDA build still starts on a machine with no NVIDIA GPU.
    // 6 threads matches the CLI harness default. rwkv.cpp clamps n_gpu_layers to n_layer+1,
    // so 999 means "every block plus the output head", which is what a 6GB 2060 should do
    // with this model. A CPU-only library ignores the count (no CUDA backend is compiled in).
    let gpu_compiled = unsafe { ffi::xenon_has_gpu_support() } != 0;
    let gpu_layers: u32 = if gpu_compiled { 999 } else { 0 };
    let mut on_gpu = false;
    let mut engine = unsafe {
        ffi::xenon_load_model(
            model_path_c.as_ptr(),
            vocab_path_c.as_ptr(),
            6,
            gpu_layers,
        )
    };

    if engine.is_null() && gpu_layers > 0 {
        let err = last_error();
        eprintln!(
            "xenon2: CUDA offload failed ({err}). Loading the model on CPU instead."
        );
        engine = unsafe {
            ffi::xenon_load_model(model_path_c.as_ptr(), vocab_path_c.as_ptr(), 6, 0)
        };
    } else if gpu_layers > 0 {
        on_gpu = true;
    }

    if engine.is_null() {
        return Err(format!(
            "Failed to load model '{}' (vocab '{}'): {}",
            model_path.display(),
            vocab_path.display(),
            last_error()
        ));
    }

    let device_label = if on_gpu {
        "CUDA — all layers on GPU".to_string()
    } else if gpu_compiled {
        "CPU (GPU offload failed)".to_string()
    } else {
        "CPU".to_string()
    };
    eprintln!("xenon2: inference device: {device_label}");
    // Phase 8: prefill the two engine-level static preambles once per loaded model instead of on
    // every generate/extraction call -- see `EngineInner`'s doc comment.
    let prefill_static = |text: &str, what: &str| -> Result<XenonStatePtr, String> {
        unsafe {
            let state = ffi::xenon_state_new(engine);
            ffi::xenon_state_reset(engine, state);
            let c = CString::new(text).expect("static prefix contains no NUL bytes");
            if ffi::xenon_prefill(engine, state, c.as_ptr()) != ffi::XenonStatus::OK {
                let err = last_error();
                ffi::xenon_state_free(state);
                return Err(format!("Failed to prefill {what}: {err}"));
            }
            Ok(XenonStatePtr(state))
        }
    };

    let built = prefill_static(STATIC_PREFIX, "static prefix")
        .and_then(|s| prefill_static(EXTRACTION_STATIC_PREFIX, "extraction prefix").map(|e| (s, e)));
    match built {
        Ok((static_prefix, extraction_prefix)) => Ok((
            EngineInner {
                engine: EnginePtr(engine),
                static_prefix,
                extraction_prefix,
                conversation_cache: Vec::new(),
                model_path: model_path.clone(),
            },
            model_name,
            device_label,
        )),
        Err(e) => {
            unsafe { ffi::xenon_free_engine(engine) };
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Hardware profile, first-run model choice, and load status
// ---------------------------------------------------------------------------------------------

/// (kind, message) describing the most recent startup outcome: kind "error" means no model is
/// running (nothing installed / failed to load); "info" explains an automatic choice. Module-level
/// so the setup code in lib.rs does not need another managed-state entry.
static LOAD_NOTE: Mutex<Option<(String, String)>> = Mutex::new(None);
static HW_CACHE: Mutex<Option<hardware::HardwareReport>> = Mutex::new(None);
/// (effective GB/s, measured on this machine)
static BW_CACHE: Mutex<Option<(f64, bool)>> = Mutex::new(None);

fn set_note(kind: &str, message: impl Into<String>) {
    if let Ok(mut g) = LOAD_NOTE.lock() {
        *g = Some((kind.to_string(), message.into()));
    }
}

fn clear_note() {
    if let Ok(mut g) = LOAD_NOTE.lock() {
        *g = None;
    }
}

fn hardware_report(refresh: bool) -> hardware::HardwareReport {
    if !refresh {
        if let Some(h) = HW_CACHE.lock().ok().and_then(|g| g.clone()) {
            return h;
        }
    }
    let hw = hardware::detect();
    if let Ok(mut g) = HW_CACHE.lock() {
        *g = Some(hw.clone());
    }
    hw
}

#[derive(Clone, Copy, PartialEq)]
enum Calibrate {
    Never,
    IfMissing,
    Force,
}

fn profile_path(app: &AppHandle) -> Option<PathBuf> {
    use tauri::Manager;
    app.path().app_data_dir().ok().map(|d| d.join("hardware_profile.json"))
}

fn tier_size_mb(repo_root: &PathBuf, tier: &ModelTier) -> u64 {
    if let Some(p) = tier_model_path(repo_root, tier) {
        if let Ok(m) = std::fs::metadata(&p) {
            return m.len() / (1024 * 1024);
        }
    }
    models_dl::entry_for_tier(tier.id).map(|e| e.size_bytes / (1024 * 1024)).unwrap_or(0)
}

/// Effective memory bandwidth for this machine: the cached measurement, a fresh measurement with
/// the smallest installed model, or a conservative default when neither is available.
fn bandwidth(app: &AppHandle, repo_root: &PathBuf, hw: &hardware::HardwareReport, mode: Calibrate) -> (f64, bool) {
    let path = profile_path(app);
    if mode != Calibrate::Force {
        if let Some(cached) = BW_CACHE.lock().ok().and_then(|g| *g) {
            return cached;
        }
        if let Some(p) = path.as_ref().and_then(|p| hardware::load_profile(p, &hw.cpu_name)) {
            let v = (p.bandwidth_gbps, true);
            if let Ok(mut g) = BW_CACHE.lock() {
                *g = Some(v);
            }
            return v;
        }
    }
    if mode != Calibrate::Never {
        let smallest = MODEL_TIERS
            .iter()
            .filter_map(|t| tier_model_path(repo_root, t))
            .min_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX));
        if let Some(model) = smallest {
            let threads = hardware::calibration_threads(hw);
            if let Some(bw) = hardware::calibrate_cpu_bandwidth(&model, &vocab_path_for(repo_root), threads) {
                if let Some(p) = &path {
                    hardware::save_profile(p, &hardware::HardwareProfile { cpu_name: hw.cpu_name.clone(), bandwidth_gbps: bw });
                }
                if let Ok(mut g) = BW_CACHE.lock() {
                    *g = Some((bw, true));
                }
                return (bw, true);
            }
        }
    }
    (hardware::DEFAULT_BW_GBPS, false)
}

fn recommendation(app: &AppHandle, repo_root: &PathBuf, hw: &hardware::HardwareReport, mode: Calibrate) -> hardware::Recommendation {
    let (bw, measured) = bandwidth(app, repo_root, hw, mode);
    let rows: Vec<(String, u64, bool)> = MODEL_TIERS
        .iter()
        .map(|t| (t.id.to_string(), tier_size_mb(repo_root, t), tier_model_path(repo_root, t).is_some()))
        .collect();
    hardware::recommend(hardware::estimate_tiers(hw, &rows, bw), bw, measured)
}

fn no_model_message(repo_root: &PathBuf) -> String {
    format!(
        "No model is installed yet. Use \"Set up models\" to download one, or copy a model file \
         (for example rwkv7-g1-1.5B-Q5_1.bin) into {}.",
        repo_root.join("models").display()
    )
}

/// Startup choice: the saved tier if its file is still there, otherwise the best installed tier
/// for this machine (measuring it the first time). Returns (path, tier, reason if automatic).
fn choose_startup_model(
    app: &AppHandle,
    repo_root: &PathBuf,
    hw: &hardware::HardwareReport,
) -> Result<(PathBuf, &'static ModelTier, Option<String>), String> {
    if let Some(tier) = saved_tier_id(app).as_deref().and_then(tier_by_id) {
        if let Some(path) = tier_model_path(repo_root, tier) {
            return Ok((path, tier, None));
        }
    }
    let rec = recommendation(app, repo_root, hw, Calibrate::IfMissing);
    let tier = rec
        .best_installed
        .as_deref()
        .and_then(tier_by_id)
        .ok_or_else(|| no_model_message(repo_root))?;
    let path = tier_model_path(repo_root, tier).ok_or_else(|| no_model_message(repo_root))?;
    Ok((path, tier, Some(rec.reason)))
}

/// Loads the model once at app startup. Never panics: with nothing installed, or a model that
/// fails to load, the app still opens (with an empty engine and an explanatory `LOAD_NOTE`) so the
/// user can download or pick a model instead of seeing a crash. Also returns a `ModelInfo` derived
/// from the same path, so the name recorded in saved conversation files (Phase 5) can never drift
/// from what's actually loaded.
pub fn load_engine(app: &AppHandle, repo_root: &PathBuf) -> (EngineState, ModelInfo, DeviceInfo) {
    clear_note();
    let hw = hardware_report(true);
    let vocab = vocab_path_for(repo_root);

    let loaded = choose_startup_model(app, repo_root, &hw).and_then(|(path, tier, auto_reason)| {
        let r = build_engine_inner(&path, &vocab)?;
        if let Some(reason) = auto_reason {
            if let Err(e) = save_tier_id(app, tier.id, true) {
                eprintln!("[model] could not remember automatic choice: {e}");
            }
            set_note("info", format!("Chose {} automatically for this computer: {reason}.", tier.label));
        }
        Ok(r)
    });

    match loaded {
        Ok((inner, model_name, device_label)) => (
            EngineState(Arc::new(Mutex::new(inner))),
            ModelInfo(Mutex::new(model_name)),
            DeviceInfo(Mutex::new(device_label)),
        ),
        Err(message) => {
            eprintln!("xenon2: no model loaded: {message}");
            set_note("error", message);
            (
                EngineState(Arc::new(Mutex::new(EngineInner::empty()))),
                ModelInfo(Mutex::new(String::new())),
                DeviceInfo(Mutex::new("No model loaded".to_string())),
            )
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    ready: bool,
    name: String,
    #[serde(rename = "noteKind")]
    note_kind: Option<String>,
    note: Option<String>,
    #[serde(rename = "modelsDir")]
    models_dir: String,
}

#[tauri::command]
pub fn get_model_status(
    root: tauri::State<'_, crate::memory::RepoRoot>,
    model_info: tauri::State<'_, ModelInfo>,
) -> ModelStatus {
    let name = model_info.0.lock().map(|g| g.clone()).unwrap_or_default();
    let note = LOAD_NOTE.lock().ok().and_then(|g| g.clone());
    ModelStatus {
        ready: !name.is_empty(),
        name,
        note_kind: note.as_ref().map(|n| n.0.clone()),
        note: note.map(|n| n.1),
        models_dir: root.0.join("models").display().to_string(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelOption {
    id: &'static str,
    label: &'static str,
    blurb: &'static str,
    installed: bool,
    #[serde(rename = "fileName")]
    file_name: Option<String>,
    active: bool,
    #[serde(rename = "sizeMb")]
    size_mb: u64,
    recommended: bool,
    #[serde(rename = "predictedTps")]
    predicted_tps: f64,
    verdict: String,
    #[serde(rename = "canDownload")]
    can_download: bool,
}

#[tauri::command]
pub fn list_models(
    app: AppHandle,
    root: tauri::State<'_, crate::memory::RepoRoot>,
    model_info: tauri::State<'_, ModelInfo>,
) -> Vec<ModelOption> {
    let current = model_info.0.lock().map(|g| g.clone()).unwrap_or_default();
    let hw = hardware_report(false);
    let rec = recommendation(&app, &root.0, &hw, Calibrate::Never);
    MODEL_TIERS
        .iter()
        .map(|t| {
            let file_name = tier_model_path(&root.0, t)
                .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()));
            let est = rec.estimates.iter().find(|e| e.id == t.id);
            ModelOption {
                id: t.id,
                label: t.label,
                blurb: t.blurb,
                installed: file_name.is_some(),
                active: !current.is_empty() && file_name.as_deref() == Some(current.as_str()),
                file_name,
                size_mb: tier_size_mb(&root.0, t),
                recommended: rec.recommended == t.id,
                predicted_tps: est.map(|e| e.predicted_tps).unwrap_or(0.0),
                verdict: est.map(|e| e.verdict.clone()).unwrap_or_default(),
                can_download: models_dl::entry_for_tier(t.id).is_some(),
            }
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareView {
    hardware: hardware::HardwareReport,
    recommendation: hardware::Recommendation,
}

/// Hardware summary plus per-tier speed predictions. `refresh` re-detects the hardware and
/// re-measures memory bandwidth (about two seconds); otherwise cached values are used.
#[tauri::command]
pub async fn get_hardware_report(
    app: AppHandle,
    root: tauri::State<'_, crate::memory::RepoRoot>,
    refresh: bool,
) -> Result<HardwareView, String> {
    let repo_root = root.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let hw = hardware_report(refresh);
        let mode = if refresh { Calibrate::Force } else { Calibrate::Never };
        let recommendation = recommendation(&app, &repo_root, &hw, mode);
        HardwareView { hardware: hw, recommendation }
    })
    .await
    .map_err(|e| e.to_string())
}

/// Hot-swaps the loaded model. Normally the new model is fully loaded and primed *before* the old
/// one is touched, so a failure (missing file, unsupported weights) leaves the current model
/// running. When free RAM cannot hold both at once (see `should_unload_first`) the old model is
/// released first, and reloaded if the new one fails to load. Waits for any in-flight
/// generation/extraction to finish (they hold the engine lock). All cached conversation state
/// belongs to the old model and is dropped with it; conversations rebuild their state from
/// history on their next turn.
#[tauri::command]
pub async fn switch_model(
    app: AppHandle,
    engine: tauri::State<'_, EngineState>,
    model_info: tauri::State<'_, ModelInfo>,
    device_info: tauri::State<'_, DeviceInfo>,
    root: tauri::State<'_, crate::memory::RepoRoot>,
    tier_id: String,
) -> Result<String, String> {
    let tier = tier_by_id(&tier_id).ok_or_else(|| format!("Unknown model tier '{tier_id}'"))?;
    let repo_root = root.0.clone();
    let model_path = tier_model_path(&repo_root, tier).ok_or_else(|| {
        format!("The {} model isn't installed (looked for: {})", tier.label, tier.candidates.join(", "))
    })?;

    let engine_arc = engine.0.clone();
    let (name, device) = tauri::async_runtime::spawn_blocking(move || -> Result<(String, String), String> {
        let vocab = vocab_path_for(&repo_root);
        let new_mb = std::fs::metadata(&model_path).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);

        if should_unload_first(hardware::available_ram_mb(), new_mb) {
            let mut guard = match engine_arc.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            let old_path = guard.model_path.clone();
            *guard = EngineInner::empty(); // drops (frees) the old model
            return match build_engine_inner(&model_path, &vocab) {
                Ok((inner, name, device)) => {
                    *guard = inner;
                    Ok((name, device))
                }
                Err(e) => {
                    if old_path.as_os_str().is_empty() {
                        Err(e)
                    } else if let Ok((inner, _, _)) = build_engine_inner(&old_path, &vocab) {
                        *guard = inner;
                        Err(format!("{e} (the previous model was restored)"))
                    } else {
                        Err(format!("{e} (and the previous model could not be reloaded)"))
                    }
                }
            };
        }

        let (new_inner, name, device) = build_engine_inner(&model_path, &vocab)?;
        let mut guard = match engine_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = new_inner; // drops (frees) the old model
        Ok((name, device))
    })
    .await
    .map_err(|e| e.to_string())??;

    if let Ok(mut g) = model_info.0.lock() {
        *g = name.clone();
    }
    if let Ok(mut g) = device_info.0.lock() {
        *g = device;
    }
    clear_note();
    if let Err(e) = save_tier_id(&app, tier.id, false) {
        eprintln!("[model] could not remember model choice: {e}");
    }
    Ok(name)
}

fn last_error() -> String {
    unsafe {
        let ptr = ffi::xenon_get_last_error();
        if ptr.is_null() {
            "(no error message)".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

// Phase 8 follow-up: this used to be one `build_prompt()` function producing one string in the
// order [volatile date/time header] -> [static few-shot] -> [history] -> "Xenon:". That ordering
// made incremental state caching *impossible*: RWKV state is strictly sequential, so a value that
// changes every call (the header) sitting at position 0 invalidates everything fed after it on
// every single call, no matter how it's cached. Reordered to [static prefix] -> [history] ->
// [volatile header] -> [new turn] -> "Xenon:" -- same wording, same facts injection, same time
// example, just moved so the two truly-cacheable pieces (the static prefix, prefilled once ever
// at engine load; the growing history, prefilled incrementally per conversation) both come before
// the one piece that has to be re-fed every turn regardless (the header). See `EngineInner`'s doc
// comment and `ensure_cache_extended` for how the three pieces below are actually assembled and
// cached. Moving the header to sit immediately before the new turn is plausibly *better* for
// grounding (closer to the generation point = more salient to an autoregressive model) rather
// than worse, and was sanity-checked against the old ordering on real prompts (see Phase 8 commit
// notes) rather than assumed safe.

/// Byte-identical on every single call this app will ever make -- prefilled once into
/// `EngineInner::static_prefix` at `load_engine` time instead of being re-evaluated per turn.
/// Phase 7 follow-up note preserved: the seed example used to end with "How can I help you
/// today?" -- real usage showed the model imitating that closing-question *style* on nearly every
/// reply regardless of content, because a small base model leans heavily on the literal style of
/// whatever example it's shown, not just the instruction-like framing around it. These two
/// example turns both end declaratively, no trailing question, to actually change the imitated
/// style rather than just ask for it in prose (which this model has already shown it doesn't
/// reliably follow -- see sloth_memory's extraction prompt fix for the same lesson).
const STATIC_PREFIX: &str =
    "The following is a coherent, friendly conversation between a user and Xenon, a \
     helpful voice assistant. Xenon answers naturally and doesn't end every reply by asking \
     what else it can help with.\n\n\
     User: Hello Xenon, how are you doing?\n\n\
     Xenon: Hi! I'm doing well, thanks for asking.\n\n\
     User: What's a fun fact about space?\n\n\
     Xenon: A day on Venus is longer than its year -- it rotates so slowly that one spin \
     takes longer than one full trip around the sun.\n\n";

/// Formats a slice of turns exactly as the old `build_prompt`'s history loop did: this is the
/// piece that gets prefilled incrementally into a conversation's cached state (see
/// `ensure_cache_extended`), so its exact text must stay stable across calls -- changing this
/// function's output for turns already in a cache would silently desync the cache from what a
/// fresh rebuild would produce.
fn history_text(turns: &[ChatTurn]) -> String {
    let mut out = String::new();
    for turn in turns {
        if turn.role == "user" {
            out.push_str("User: ");
        } else {
            out.push_str("Xenon: ");
        }
        out.push_str(turn.content.trim());
        out.push_str("\n\n");
    }
    out
}

/// The one part of the prompt that must be re-fed every single turn regardless of caching --
/// see this section's Phase 8 doc comment above for why it can't sit at the front any more.
/// Preserves both Phase 7 follow-up fixes verbatim, just relocated: real-clock grounding (the
/// model was found to confidently make up a specific time with no actual clock access at all)
/// and Sloth's persistent facts injection (skipped entirely for Dementia turns, which always
/// pass an empty `facts` slice, so Dementia's volatile header is byte-for-byte what it always
/// was pre-Phase-8, just at a different position in the overall prompt).
fn volatile_header_text(facts: &[sloth_memory::SlothFact]) -> String {
    let mut out = String::new();

    let now = chrono::Local::now();
    let now_full = now.format("%A, %B %d, %Y, %I:%M %p").to_string();
    let now_time_only = now.format("%I:%M %p").to_string();
    out.push_str("The current date and time is ");
    out.push_str(&now_full);
    out.push_str(".\n\n");

    if !facts.is_empty() {
        out.push_str("Known facts about the user, remembered from past conversations:\n");
        for fact in facts {
            out.push_str("- ");
            out.push_str(&fact.text);
            out.push('\n');
        }
        out.push('\n');
    }

    // A dedicated example for time questions, using the *real* current time computed above (not
    // a fixed fake value) -- a plain instruction to "use the date/time given above" wasn't
    // reliably followed for short phrasings like "What time is it?" (it worked for a more
    // elaborate phrasing but not this one, tested side by side), so this demonstrates the exact
    // pattern instead. Using the real time here means the example is never factually wrong
    // regardless of when it's generated, so there's no risk of the model anchoring on a stale or
    // made-up value the way a fixed example would.
    out.push_str(&format!(
        "User: What time is it?\n\n\
         Xenon: It's currently {now_time_only}.\n\n"
    ));

    out
}

fn ends_with_stop(tail: &str) -> bool {
    tail.ends_with("\n\nUser:") || tail.ends_with("\n\nuser:")
}

/// Context handed to the C callback via the `user_data` void*. The callback runs synchronously
/// on the same thread that calls `xenon_generate`, so `RefCell` (not a Mutex) is fine here.
struct CallbackCtx {
    app: AppHandle,
    conversation_id: String,
    tail: RefCell<String>,
    /// Full generated text, untruncated (unlike `tail`, which is capped for stop-sequence
    /// checking) -- used by the Sloth fact-extraction step after generation completes. `None`
    /// for the extraction call itself (see `run_fact_extraction`), which uses a separate,
    /// simpler callback that doesn't emit `token-stream` events.
    full_text: RefCell<String>,
}

extern "C" fn on_token(
    text: *const std::os::raw::c_char,
    _token_id: u32,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return true;
    }
    let ctx = unsafe { &*(user_data as *const CallbackCtx) };

    let text_str = if text.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("")
    };

    if !text_str.is_empty() {
        let _ = ctx.app.emit(
            "token-stream",
            TokenEvent {
                conversation_id: ctx.conversation_id.clone(),
                text: text_str.to_string(),
            },
        );

        ctx.full_text.borrow_mut().push_str(text_str);

        let mut tail = ctx.tail.borrow_mut();
        tail.push_str(text_str);
        let len = tail.len();
        if len > 64 {
            let mut cut = len - 64;
            while !tail.is_char_boundary(cut) {
                cut += 1;
            }
            tail.drain(0..cut);
        }

        if ends_with_stop(&tail) {
            return false; // natural end of this turn, matches the CLI harness's stop heuristic
        }
    }

    true
}

/// Minimal callback for the Sloth fact-extraction call -- just accumulates text into `user_data`
/// (a bare `RefCell<String>`, not a full `CallbackCtx`), since the extraction reply is never
/// shown in the UI and doesn't need `token-stream` events or stop-sequence checking beyond a
/// simple length cap (extraction prompts ask for one short sentence; a runaway generation would
/// mean something went wrong, not that there's more useful fact text coming).
extern "C" fn on_extraction_token(
    text: *const std::os::raw::c_char,
    _token_id: u32,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return true;
    }
    let buf = unsafe { &*(user_data as *const RefCell<String>) };
    let text_str = if text.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("")
    };
    buf.borrow_mut().push_str(text_str);
    buf.borrow().len() < 300 // safety cap, well beyond one short sentence
}

/// The static half of the extraction prompt -- everything except the just-completed exchange
/// itself. Byte-identical on every extraction call, so it's prefilled once into
/// `EngineInner::extraction_prefix` at `load_engine` time (see that field's doc comment) instead
/// of being re-evaluated on every Sloth turn (measured before Phase 8: 205 tokens, every time).
/// Kept deliberately strict ("reply with exactly: NONE") since this is a small,
/// non-instruction-tuned base model -- without a strong steer it tends to invent something
/// rather than correctly recognize "nothing new here". Few-shot, not just an abstract
/// instruction -- a bare "extract a fact or say NONE" instruction was found to make this small
/// base model *invent* a plausible-sounding fact (e.g. "named John, loves pizza") even when the
/// user never said any such thing, rather than correctly recognizing nothing new was revealed.
/// Real bug found in usage (2026-08-02): the model once misattributed Xenon's *own* stated name
/// as a fact about the user ("The user's name is Xenon") after being asked "What is your name?"
/// -- added a dedicated example so it's explicit that facts about Xenon itself never count as
/// facts about the user, the same conflation risk as the NONE examples below.
const EXTRACTION_STATIC_PREFIX: &str =
    "Xenon only records a fact if the user directly stated it about *themselves*. Facts \
     Xenon states about itself (its own name, its own capabilities) are never facts about \
     the user. If the user did not clearly state something new and durable about \
     themselves, Xenon replies with exactly: NONE\n\n\
     User: My favorite color is blue and I have a dog named Rex.\n\n\
     Xenon: That's lovely! Rex sounds like a great dog.\n\n\
     Fact: The user's favorite color is blue and they have a dog named Rex.\n\n\
     User: What's the weather like today?\n\n\
     Xenon: I don't have access to real-time weather data.\n\n\
     Fact: NONE\n\n\
     User: What is my name?\n\n\
     Xenon: I'm sorry, I don't have that information. Can you tell me your name?\n\n\
     Fact: NONE\n\n\
     User: What is your name?\n\n\
     Xenon: My name is Xenon.\n\n\
     Fact: NONE\n\n";

/// The volatile (per-call) half of the extraction prompt: just this turn's exchange.
fn build_extraction_suffix(user_text: &str, reply_text: &str) -> String {
    format!("User: {user_text}\n\nXenon: {reply_text}\n\nFact:")
}

/// Runs the Sloth fact-extraction step: a second, small, low-temperature generate call using the
/// just-completed turn. As of Phase 9 this is called from a background task, on a lock acquired
/// separately and later than the main reply's -- see `generate_response`'s doc comment for why
/// (nothing about the visible reply depends on this, so it must not delay `generation-done` or
/// the next turn's own lock acquisition). Still fully sequential with respect to *other* engine
/// calls once it does run (rwkv.cpp's per-engine state isn't safe for concurrent eval calls, see
/// `EngineInner`'s doc comment) -- "fire-and-forget" here means "don't block on it," not
/// "run it concurrently with other engine access." Always starts from a fresh copy of
/// `extraction_prefix` and discards it afterwards -- this call must never be able to read or
/// write a conversation's cached state in `EngineInner::conversation_cache` (a Phase 8
/// requirement that Phase 9 leans on: extraction closes over its own owned `user_text`/
/// `reply_text`, not a borrow into `history` or the cache, so it stays correct no matter how much
/// later it actually runs, even after the next turn has already started). On success, persists
/// any real fact via `sloth_memory::append_fact` and emits `sloth-facts-updated` so an open
/// Memory panel (see App.vue) can refresh live. Never surfaces its own errors as a
/// `generation-error` -- a failed/garbled extraction just means no new fact gets remembered this
/// turn, which shouldn't interrupt or error out the visible reply that already completed
/// successfully (which, as of Phase 9, the user has already seen by the time this even starts).
fn run_fact_extraction(
    engine: *mut ffi::XenonEngine,
    extraction_prefix: &XenonStatePtr,
    app: &AppHandle,
    user_text: &str,
    reply_text: &str,
) {
    let suffix = match CString::new(build_extraction_suffix(user_text, reply_text)) {
        Ok(c) => c,
        Err(_) => return,
    };

    let scratch = unsafe { ffi::xenon_state_new(engine) };
    unsafe { ffi::xenon_state_copy(scratch, extraction_prefix.0) };

    let buf = RefCell::new(String::new());
    let buf_ptr = &buf as *const RefCell<String> as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate_with_state(
            engine,
            scratch,
            suffix.as_ptr(),
            40,   // max_tokens -- one short sentence, not a full reply
            0.2,  // temperature -- extraction should be far more deterministic than a chat reply
            0.2,  // top_p
            1.1,  // repeat_penalty -- mild; extraction is already short/constrained by the few-shot prompt
            on_extraction_token,
            buf_ptr,
        )
    };
    unsafe { ffi::xenon_state_free(scratch) };
    if status != ffi::XenonStatus::OK {
        return;
    }

    let extracted = buf.into_inner();
    let trimmed = extracted
        .split("\n\n") // guard against the model drifting into a fake next turn, same as chat replies do
        .next()
        .unwrap_or("")
        .trim();

    // Safety net beyond the prompt's own few-shot guidance: an extraction that just echoes the
    // assistant's own reply (observed failure mode -- e.g. "I don't know your name" verbatim as
    // the "fact") is never a real fact about the user, regardless of what the model said.
    let looks_like_echo = trimmed.eq_ignore_ascii_case(reply_text.trim())
        || reply_text.trim().contains(trimmed);

    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("none")
        || trimmed.len() > 300
        || looks_like_echo
    {
        return;
    }

    match sloth_memory::append_fact(app, trimmed.to_string(), now_millis()) {
        Ok(facts) => {
            let _ = app.emit("sloth-facts-updated", facts);
        }
        Err(e) => eprintln!("[sloth_memory] could not persist extracted fact: {e}"),
    }
}

/// Deterministic answer for time/date questions, bypassing the model entirely. Added
/// 2026-08-04 after prompt-grounding (real clock text injected into the prompt, plus a
/// demonstrated example) still wasn't reliable enough: the model would get *close* to the real
/// time but not exact (e.g. reporting "8:05 PM" when it was actually 8:10), and sometimes dropped
/// AM/PM -- a known small-model weakness (precisely reproducing a specific number from context
/// isn't guaranteed even when the right value is right there in the prompt). Time/date has a
/// single objectively correct answer, so there's no reason to leave it up to a small model's
/// approximation once heuristic detection is reliable enough to catch the question.
/// Returns `None` for anything that doesn't clearly ask for the time/date, falling through to
/// real generation as normal.
fn try_answer_time_date_deterministically(user_text: &str) -> Option<String> {
    let lower = user_text.to_lowercase();

    let asks_time = ["what time", "current time", "time is it", "know the time", "tell me the time"]
        .iter()
        .any(|p| lower.contains(p));
    let asks_date = ["what day", "today's date", "what's the date", "what is the date", "what date", "current date"]
        .iter()
        .any(|p| lower.contains(p));

    if !asks_time && !asks_date {
        return None;
    }

    let now = chrono::Local::now();
    // "%-I"/"%-d" (non-padded) so TTS reads "8:10 PM" naturally instead of piper sounding out
    // "zero eight ten" for a zero-padded "08:10".
    let time_str = now.format("%-I:%M %p").to_string();
    let date_str = now.format("%A, %B %-d, %Y").to_string();

    Some(match (asks_time, asks_date) {
        (true, true) => format!("It's {time_str} on {date_str}."),
        (true, false) => format!("It's currently {time_str}."),
        (false, true) => format!("Today is {date_str}."),
        (false, false) => unreachable!(),
    })
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Phase 9 measurement instrumentation: prints a labeled timestamp to stderr when
/// `XENON_DEBUG_TIMING` is set, so the generation-done/extraction ordering and latency claims
/// this phase makes can be measured directly against the real dev build instead of assumed. Off
/// by default -- `var_os` is only called once per log point, not in any hot loop.
fn debug_timing_log(label: &str) {
    if std::env::var_os("XENON_DEBUG_TIMING").is_some() {
        eprintln!("[timing] {} t={}", label, now_millis());
    }
}

/// Feeds `text` into `state` via `xenon_prefill`, mapping the C error into a `Result` instead of
/// silently ignoring it. `text` == "" is a cheap no-op (see `xenon_prefill`'s doc comment).
fn prefill_text(engine: *mut ffi::XenonEngine, state: *mut ffi::XenonState, text: &str) -> Result<(), String> {
    let c_text = CString::new(text).map_err(|e| format!("prefill text contained an embedded NUL byte: {e}"))?;
    let status = unsafe { ffi::xenon_prefill(engine, state, c_text.as_ptr()) };
    if status == ffi::XenonStatus::OK {
        Ok(())
    } else {
        Err(last_error())
    }
}

/// The heart of Phase 8: given the turns already known to have happened (`prior_turns` --
/// everything in this call's `history` except the newest, not-yet-answered user turn), returns a
/// pointer to a state that has consumed exactly `STATIC_PREFIX` + `history_text(prior_turns)`,
/// reusing as much of `inner.conversation_cache` as possible instead of rebuilding from scratch.
///
/// Diff logic: if this conversation has a cache entry whose `consumed` turns are a byte-equal
/// prefix of `prior_turns`, this is a plain continuation (or a regenerate of the last reply, which
/// sends the identical `prior_turns` as before) -- "extend" by prefilling only the new tail turns.
/// Otherwise (no entry yet, an evicted entry, or `prior_turns` diverges anywhere -- an edit to an
/// earlier message, a delete, or a shorter history) -- "rebuild" from a fresh copy of
/// `static_prefix`, prefilling all of `prior_turns`. Either way, `consumed` is set to
/// `prior_turns` *before* this turn generates anything, from `history` text the frontend already
/// considers canonical -- this cache is never advanced from this turn's own (possibly still to be
/// trimmed, see `chat.ts`'s `completeGeneration`) generation output. See `EngineInner`'s doc
/// comment and this phase's design notes for why that ordering is what makes the cache safe
/// against edits/regenerates/the frontend's post-hoc trimming without any special-casing here.
fn ensure_cache_extended(
    inner: &mut EngineInner,
    conversation_id: &str,
    prior_turns: &[ChatTurn],
) -> Result<*mut ffi::XenonState, String> {
    let engine = inner.engine.0;

    if let Some(idx) = inner
        .conversation_cache
        .iter()
        .position(|(id, _)| id == conversation_id)
    {
        let is_extendable = {
            let cached = &inner.conversation_cache[idx].1;
            cached.consumed.len() <= prior_turns.len() && cached.consumed == prior_turns[..cached.consumed.len()]
        };

        if is_extendable {
            let cached = &mut inner.conversation_cache[idx].1;
            let delta = &prior_turns[cached.consumed.len()..];
            if !delta.is_empty() {
                prefill_text(engine, cached.state.0, &history_text(delta))?;
            }
            cached.consumed = prior_turns.to_vec();

            // Move to the back (most-recently-used) so LRU eviction below doesn't pick this one.
            let entry = inner.conversation_cache.remove(idx);
            inner.conversation_cache.push(entry);
            return Ok(inner.conversation_cache.last().unwrap().1.state.0);
        }

        // Diverges somewhere -- stale relative to what the frontend just sent (an edit, a
        // delete, or otherwise). Drop it and rebuild below.
        inner.conversation_cache.remove(idx);
    }

    let new_state = unsafe { ffi::xenon_state_new(engine) };
    unsafe { ffi::xenon_state_copy(new_state, inner.static_prefix.0) };
    if let Err(e) = prefill_text(engine, new_state, &history_text(prior_turns)) {
        unsafe { ffi::xenon_state_free(new_state) };
        return Err(e);
    }

    if inner.conversation_cache.len() >= MAX_CACHED_CONVERSATIONS {
        inner.conversation_cache.remove(0); // least-recently-used is always at the front
    }
    inner.conversation_cache.push((
        conversation_id.to_string(),
        CachedConversation { consumed: prior_turns.to_vec(), state: XenonStatePtr(new_state) },
    ));
    Ok(inner.conversation_cache.last().unwrap().1.state.0)
}

/// Runs generation to completion on the calling thread (blocking) and emits Tauri events as
/// tokens stream in. Meant to be called from inside `spawn_blocking` by the IPC command below,
/// never directly on the async/event-loop thread. Takes the already-locked `EngineInner` so the
/// mutex stays held for the full duration of the call (see `EngineInner` doc comment). `agent`
/// ("dementia" | "sloth") controls whether Sloth's persistent facts are injected into the prompt.
/// Returns `Some(PendingExtraction)` when a Sloth turn completed successfully and fact-extraction
/// should run -- as of Phase 9, this function no longer runs that extraction itself (see
/// `generate_response`'s doc comment for why: nothing about the visible reply depends on it, so
/// it must not delay `generation-done`, which this function always emits itself before
/// returning, on every path).
fn generate_blocking(
    inner: &mut EngineInner,
    app: AppHandle,
    conversation_id: String,
    history: Vec<ChatTurn>,
    agent: String,
) -> Option<PendingExtraction> {
    if !inner.is_loaded() {
        let _ = app.emit(
            "generation-error",
            GenerationErrorEvent {
                conversation_id,
                message: "No model is loaded. Open \"Set up models\" to download or choose one.".to_string(),
            },
        );
        return None;
    }
    let engine = inner.engine.0;

    // `prior_turns` is everything already-happened; the newest user turn is what this call
    // answers. Both the deterministic short-circuit below and the real generation path share
    // the same cache-extend step so the cache never falls behind regardless of which path a
    // given turn takes.
    let (prior_turns, new_user_turn) = match history.split_last() {
        Some((last, rest)) => (rest, last),
        None => {
            let _ = app.emit(
                "generation-error",
                GenerationErrorEvent { conversation_id, message: "generate_response called with empty history".to_string() },
            );
            return None;
        }
    };

    let base_state = match ensure_cache_extended(inner, &conversation_id, prior_turns) {
        Ok(s) => s,
        Err(message) => {
            let _ = app.emit("generation-error", GenerationErrorEvent { conversation_id, message });
            return None;
        }
    };

    if let Some(answer) = try_answer_time_date_deterministically(&new_user_turn.content) {
        let _ = app.emit(
            "token-stream",
            TokenEvent { conversation_id: conversation_id.clone(), text: answer },
        );
        let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id });
        return None;
    }

    let is_sloth = agent == "sloth";
    let facts = if is_sloth {
        sloth_memory::load_memory(&app).map(|m| m.facts).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Never generate directly into the authoritative cached state (`base_state`) -- see
    // `ensure_cache_extended`'s doc comment on why the cache only ever advances from
    // caller-confirmed history text, never from this turn's own raw generation output. Copy it
    // into a scratch state, feed the volatile header + new turn, generate there, then discard.
    let turn_state = unsafe { ffi::xenon_state_new(engine) };
    unsafe { ffi::xenon_state_copy(turn_state, base_state) };

    let suffix = format!(
        "{}User: {}\n\nXenon:",
        volatile_header_text(&facts),
        new_user_turn.content.trim()
    );
    let suffix_c = match CString::new(suffix) {
        Ok(c) => c,
        Err(e) => {
            unsafe { ffi::xenon_state_free(turn_state) };
            let _ = app.emit(
                "generation-error",
                GenerationErrorEvent {
                    conversation_id,
                    message: format!("prompt contained an embedded NUL byte: {e}"),
                },
            );
            return None;
        }
    };

    let ctx = CallbackCtx {
        app: app.clone(),
        conversation_id: conversation_id.clone(),
        tail: RefCell::new(String::new()),
        full_text: RefCell::new(String::new()),
    };
    let ctx_ptr = &ctx as *const CallbackCtx as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate_with_state(
            engine,
            turn_state,
            suffix_c.as_ptr(),
            200,  // max_tokens
            0.8,  // temperature (matches Phase 1 CLI harness default)
            0.5,  // top_p (matches Phase 1 CLI harness default)
            // repeat_penalty (Phase 7 follow-up, 2026-08-02): added after real usage showed this
            // model imitating its own earlier canned reply for unrelated follow-up questions once
            // that phrase was sitting in the resent conversation history -- see xenon_inference.h's
            // doc comment. Empirically tuned, not guessed: replaying the actual failure
            // conversation (a phrase repeated 3x already in history) showed 1.15 (llama.cpp's
            // typical low end) was NOT enough to break the loop -- the model repeated the trap
            // phrase a 4th time regardless. 1.3 (llama.cpp's typical high end) did break it,
            // producing a real relevant answer, and re-verified to cause no quality loss on normal
            // short exchanges (correct arithmetic, coherent fun facts, normal greetings).
            1.3,
            on_token,
            ctx_ptr,
        )
    };

    unsafe { ffi::xenon_state_free(turn_state) };

    if status == ffi::XenonStatus::OK {
        // Phase 9: emit generation-done FIRST -- the user sees the reply now, full stop. Building
        // a PendingExtraction (just cloning two owned Strings) and handing it back to the caller
        // is essentially free; the expensive part (the actual second generate call) happens in
        // `generate_response`'s background task, after this function has already returned and
        // its caller has released the engine lock.
        let pending = if is_sloth {
            Some(PendingExtraction {
                user_text: new_user_turn.content.clone(),
                reply_text: ctx.full_text.borrow().clone(),
            })
        } else {
            None
        };
        debug_timing_log("generation-done emitted");
        let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id });
        pending
    } else {
        let _ = app.emit(
            "generation-error",
            GenerationErrorEvent {
                conversation_id,
                message: last_error(),
            },
        );
        None
    }
}

/// Tauri IPC command: frontend sends the full turn history (ending with the new user message)
/// plus which agent this turn should use, backend streams the assistant's reply back via
/// `token-stream` / `generation-done` / `generation-error` events tagged with `conversation_id`.
///
/// Phase 9: Sloth fact-extraction no longer runs under the same lock acquisition as the visible
/// reply. `generate_blocking` emits `generation-done` and returns as soon as the reply is ready;
/// only *then* is its lock guard dropped, and if it handed back a `PendingExtraction`, a second,
/// independent `spawn_blocking` task acquires the engine lock on its own (whenever it's next
/// free -- possibly after the very next turn, if the user replies fast) and runs extraction
/// there. Nothing about the visible reply depends on extraction, so it must never be able to
/// delay `generation-done` or make the *next* turn's `generate_response` call wait on it.
#[tauri::command]
pub async fn generate_response(
    app: AppHandle,
    engine: tauri::State<'_, EngineState>,
    conversation_id: String,
    history: Vec<ChatTurn>,
    agent: String,
) -> Result<(), String> {
    let engine_arc = engine.0.clone();
    let extraction_app = app.clone();
    let extraction_engine_arc = engine_arc.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Held for the blocking generate() call and the cache lookups/extends around it, but no
        // longer across a Sloth extraction call too -- see this function's doc comment.
        debug_timing_log("generate_response task started, awaiting lock (main)");
        let mut guard = match engine_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        debug_timing_log("engine lock acquired (main)");
        let pending = generate_blocking(&mut guard, app, conversation_id, history, agent);
        drop(guard); // release before scheduling extraction -- this is the actual fix
        debug_timing_log("engine lock released (main)");

        if let Some(job) = pending {
            tauri::async_runtime::spawn_blocking(move || {
                debug_timing_log("extraction task spawned, waiting for lock");
                let guard = match extraction_engine_arc.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                debug_timing_log("engine lock acquired (extraction)");
                if !guard.is_loaded() {
                    return;
                }
                run_fact_extraction(
                    guard.engine.0,
                    &guard.extraction_prefix,
                    &extraction_app,
                    &job.user_text,
                    &job.reply_text,
                );
                debug_timing_log("extraction finished, engine lock released");
            });
        }
    })
    .await
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod model_tests {
    use super::*;

    fn temp_root(name: &str, files: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("xenon_inf_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("models")).unwrap();
        for f in files {
            std::fs::write(root.join("models").join(f), b"x").unwrap();
        }
        root
    }

    #[test]
    fn tier_prefers_the_new_file_over_the_legacy_one() {
        let root = temp_root("prefer", &["rwkv7-g1-2.9B-Q5_1.bin", "rwkv-7-world-2.9B-Q5_1.bin"]);
        let p = tier_model_path(&root, tier_by_id("quality").unwrap()).unwrap();
        assert!(p.ends_with("rwkv7-g1-2.9B-Q5_1.bin"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn tier_falls_back_to_the_legacy_file() {
        let root = temp_root("fallback", &["rwkv-7-world-2.9B-Q5_1.bin", "rwkv-5-world-0.4B-Q4_0.bin"]);
        assert!(tier_model_path(&root, tier_by_id("quality").unwrap()).unwrap().ends_with("rwkv-7-world-2.9B-Q5_1.bin"));
        assert!(tier_model_path(&root, tier_by_id("fast").unwrap()).unwrap().ends_with("rwkv-5-world-0.4B-Q4_0.bin"));
        assert!(tier_model_path(&root, tier_by_id("balanced").unwrap()).is_none(), "no balanced file and no legacy substitute");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unknown_tier_is_rejected() {
        assert!(tier_by_id("ultra").is_none());
    }

    #[test]
    fn low_ram_switch_unloads_first() {
        // 2.9B (2625 MB) needs 2625*1.3 + 1024 = 4436 MB free to coexist with the old model.
        assert!(should_unload_first(4000, 2625));
        assert!(!should_unload_first(16000, 2625));
        assert!(should_unload_first(1500, 525), "even the small model needs headroom on a tight machine");
        assert!(!should_unload_first(2500, 525));
    }

    #[test]
    fn saved_choice_parses_old_and_new_formats() {
        assert_eq!(parse_saved_tier(r#"{"tier":"balanced"}"#).as_deref(), Some("balanced"));
        assert_eq!(parse_saved_tier(r#"{"tier":"fast","auto":true}"#).as_deref(), Some("fast"));
        assert_eq!(parse_saved_tier("not json"), None);
        assert_eq!(parse_saved_tier("{}"), None);
    }

    #[test]
    fn empty_engine_is_not_loaded_and_drops_safely() {
        let e = EngineInner::empty();
        assert!(!e.is_loaded());
        drop(e); // frees a null engine: documented no-op
    }

    /// Needs the real model files and the engine DLL: `cargo test -- --ignored`.
    /// Covers hot-swap, failed-switch recovery, and the cache being dropped with the old model.
    #[test]
    #[ignore]
    fn hot_swap_and_failed_switch_keep_a_working_engine() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
        let vocab = vocab_path_for(&root);
        let small = tier_model_path(&root, tier_by_id("fast").unwrap()).expect("fast model installed");
        let mid = tier_model_path(&root, tier_by_id("balanced").unwrap()).expect("balanced model installed");

        let (first, name1, _) = build_engine_inner(&small, &vocab).unwrap();
        let holder = Mutex::new(first);
        assert!(holder.lock().unwrap().is_loaded());

        let (second, name2, _) = build_engine_inner(&mid, &vocab).unwrap();
        assert_ne!(name1, name2);
        *holder.lock().unwrap() = second; // old engine freed by Drop
        assert_eq!(holder.lock().unwrap().model_path, mid);

        // A bad file must fail cleanly and leave the loaded engine untouched.
        let bad = std::env::temp_dir().join(format!("xenon_bad_{}.bin", std::process::id()));
        std::fs::write(&bad, b"not a model").unwrap();
        assert!(build_engine_inner(&bad, &vocab).is_err());
        assert!(holder.lock().unwrap().is_loaded());
        let _ = std::fs::remove_file(bad);
    }
}
