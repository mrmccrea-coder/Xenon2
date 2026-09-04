//! Phase 7 follow-up: persistent cross-conversation memory for the "Sloth" agent.
//!
//! Dementia (the default agent) behaves exactly like the app always has -- each conversation's
//! own history is used, nothing survives outside that chat window. Sloth additionally reads and
//! writes a small store of short, distilled "facts" (`<app-data-dir>/sloth_memory.json`) that
//! persists across every conversation and every app restart -- see `inference.rs`'s
//! `generate_blocking` for where facts get injected into the prompt (every Sloth turn) and
//! extracted from it (after every Sloth turn completes).
//!
//! Facts are automatically extracted by asking the model itself (a second, small, low-
//! temperature generate() call after the main reply) whether the exchange revealed anything new
//! and durable worth remembering -- not user-curated. Given the underlying model is small, this
//! extraction step will sometimes get it wrong (miss something, or invent something) -- the
//! "Sloth Memory..." UI (see App.vue) lets facts be deleted individually or all at once so a bad
//! extraction doesn't have to live forever.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// Hard cap on stored facts -- keeps the "Known facts" prompt preamble bounded so it can never
/// alone blow past the model's 4096-token context window as Sloth is used over a long time.
/// Oldest facts are dropped first once the cap is hit (see `append_fact`).
const MAX_FACTS: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlothFact {
    pub id: String,
    pub text: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlothMemory {
    #[serde(default)]
    pub facts: Vec<SlothFact>,
}

fn memory_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Could not resolve the app data directory: {e}"))?;
    Ok(dir.join("sloth_memory.json"))
}

/// Loads the fact store. A missing file (Sloth never used yet) is normal and returns an empty
/// store, not an error.
pub fn load_memory(app: &AppHandle) -> Result<SlothMemory, String> {
    let path = memory_path(app)?;
    if !path.exists() {
        return Ok(SlothMemory::default());
    }
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("Could not read '{}': {}", path.display(), e))?;
    serde_json::from_str(&raw).map_err(|e| format!("Could not parse sloth_memory.json: {e}"))
}

fn save_memory(app: &AppHandle, memory: &SlothMemory) -> Result<(), String> {
    let path = memory_path(app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Could not create '{}': {}", parent.display(), e))?;
    }
    let json = serde_json::to_string_pretty(memory)
        .map_err(|e| format!("Could not serialize sloth_memory.json: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Could not write '{}': {}", path.display(), e))
}

/// Appends one newly-extracted fact, dropping the oldest fact first if already at `MAX_FACTS`.
/// Called from `inference.rs` after a Sloth turn's extraction step produces something worth
/// keeping. Returns the updated fact list (so the caller can emit it to the frontend without a
/// second read).
pub fn append_fact(app: &AppHandle, text: String, created_at: i64) -> Result<Vec<SlothFact>, String> {
    let mut memory = load_memory(app)?;
    memory.facts.push(SlothFact {
        id: created_at.to_string(),
        text,
        created_at,
    });
    if memory.facts.len() > MAX_FACTS {
        let excess = memory.facts.len() - MAX_FACTS;
        memory.facts.drain(0..excess);
    }
    save_memory(app, &memory)?;
    Ok(memory.facts)
}

#[tauri::command]
pub fn list_sloth_facts(app: AppHandle) -> Result<Vec<SlothFact>, String> {
    Ok(load_memory(&app)?.facts)
}

#[tauri::command]
pub fn delete_sloth_fact(app: AppHandle, id: String) -> Result<Vec<SlothFact>, String> {
    let mut memory = load_memory(&app)?;
    memory.facts.retain(|f| f.id != id);
    save_memory(&app, &memory)?;
    Ok(memory.facts)
}

#[tauri::command]
pub fn clear_sloth_facts(app: AppHandle) -> Result<(), String> {
    save_memory(&app, &SlothMemory::default())
}
