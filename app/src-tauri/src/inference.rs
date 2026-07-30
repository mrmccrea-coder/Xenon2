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

/// One turn of conversation, as sent from the Vue frontend. Mirrors the shape of a chat message
/// but deliberately doesn't carry UI-only fields (ids, timestamps, streaming flags, etc) -- this
/// is a minimal payload just for prompt construction.
#[derive(Debug, Clone, Deserialize)]
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

/// Raw engine pointer, guarded by a mutex so only one generation runs at a time (rwkv.cpp's
/// per-engine state is not designed for concurrent calls). `xenon_engine*` is safe to hand off
/// across threads as long as access is serialized, which the Mutex guarantees.
pub(crate) struct EnginePtr(*mut ffi::XenonEngine);
unsafe impl Send for EnginePtr {}

/// `Arc` (not a bare `Mutex`) because the IPC command below needs to clone a handle to it into a
/// `spawn_blocking` closure and hold the lock for the *entire* generate() call, not just long
/// enough to read the pointer out -- rwkv.cpp's per-engine state isn't safe for concurrent eval
/// calls, so the lock must stay held across the whole blocking generation, not be dropped early.
#[derive(Clone)]
pub struct EngineState(pub Arc<Mutex<EnginePtr>>);

/// Loads the Phase 1 model once at app startup. Panics (fails app startup) rather than silently
/// falling back to a stub -- per the phase spec, this app must call the real inference engine,
/// not reimplement or fake it.
pub fn load_engine(repo_root: &PathBuf) -> EngineState {
    let model_path = repo_root.join("models").join("rwkv-5-world-0.4B-Q4_0.bin");
    let vocab_path = repo_root
        .join("inference-engine")
        .join("data")
        .join("world_vocab.bin");

    let model_path_c = CString::new(model_path.to_string_lossy().as_bytes())
        .expect("model path contains NUL byte");
    let vocab_path_c = CString::new(vocab_path.to_string_lossy().as_bytes())
        .expect("vocab path contains NUL byte");

    // 6 threads to match Phase 1's CLI harness default (a reasonable middle ground on this
    // 16c/24t dev machine); 0 GPU layers -- see build.rs for why this app links the CPU-only
    // build of the inference engine.
    let engine = unsafe { ffi::xenon_load_model(model_path_c.as_ptr(), vocab_path_c.as_ptr(), 6, 0) };

    if engine.is_null() {
        let err = last_error();
        panic!(
            "Failed to load Phase 1 model from '{}' (vocab '{}'): {}",
            model_path.display(),
            vocab_path.display(),
            err
        );
    }

    EngineState(Arc::new(Mutex::new(EnginePtr(engine))))
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

/// Builds the same "User: ... / Xenon: ..." chat-style prime used by Phase 1's CLI harness
/// (`test_inference.cpp`'s `build_chat_prompt`), extended to include the full turn history so a
/// small base-ish World model stays on-topic across a multi-turn conversation. `xenon_generate`
/// resets the model's RWKV state at the start of every call (see xenon_inference.h), so the
/// entire conversation-so-far has to be re-fed as text each time -- there is no cross-call
/// incremental state to build on yet.
fn build_prompt(history: &[ChatTurn]) -> String {
    let mut prompt = String::from(
        "The following is a coherent, friendly conversation between a user and Xenon, a \
         helpful voice assistant.\n\n\
         User: Hello Xenon, how are you doing?\n\n\
         Xenon: Hi! I'm doing well, thanks for asking. How can I help you today?\n\n",
    );

    for turn in history {
        if turn.role == "user" {
            prompt.push_str("User: ");
        } else {
            prompt.push_str("Xenon: ");
        }
        prompt.push_str(turn.content.trim());
        prompt.push_str("\n\n");
    }

    prompt.push_str("Xenon:");
    prompt
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

        let mut tail = ctx.tail.borrow_mut();
        tail.push_str(text_str);
        let len = tail.len();
        if len > 64 {
            tail.drain(0..len - 64);
        }

        if ends_with_stop(&tail) {
            return false; // natural end of this turn, matches the CLI harness's stop heuristic
        }
    }

    true
}

/// Runs generation to completion on the calling thread (blocking) and emits Tauri events as
/// tokens stream in. Meant to be called from inside `spawn_blocking` by the IPC command below,
/// never directly on the async/event-loop thread. Takes the already-locked engine pointer so the
/// mutex stays held for the full duration of the call (see `EngineState` doc comment).
fn generate_blocking(engine: *mut ffi::XenonEngine, app: AppHandle, conversation_id: String, history: Vec<ChatTurn>) {
    let prompt = build_prompt(&history);
    let prompt_c = match CString::new(prompt) {
        Ok(c) => c,
        Err(e) => {
            let _ = app.emit(
                "generation-error",
                GenerationErrorEvent {
                    conversation_id,
                    message: format!("prompt contained an embedded NUL byte: {e}"),
                },
            );
            return;
        }
    };

    let ctx = CallbackCtx {
        app: app.clone(),
        conversation_id: conversation_id.clone(),
        tail: RefCell::new(String::new()),
    };
    let ctx_ptr = &ctx as *const CallbackCtx as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate(
            engine,
            prompt_c.as_ptr(),
            200,  // max_tokens
            0.8,  // temperature (matches Phase 1 CLI harness default)
            0.5,  // top_p (matches Phase 1 CLI harness default)
            on_token,
            ctx_ptr,
        )
    };

    if status == ffi::XenonStatus::OK {
        let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id });
    } else {
        let _ = app.emit(
            "generation-error",
            GenerationErrorEvent {
                conversation_id,
                message: last_error(),
            },
        );
    }
}

/// Tauri IPC command: frontend sends the full turn history (ending with the new user message),
/// backend streams the assistant's reply back via `token-stream` / `generation-done` /
/// `generation-error` events tagged with `conversation_id`.
#[tauri::command]
pub async fn generate_response(
    app: AppHandle,
    engine: tauri::State<'_, EngineState>,
    conversation_id: String,
    history: Vec<ChatTurn>,
) -> Result<(), String> {
    let engine_arc = engine.0.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Held for the entire blocking generate() call -- see EngineState's doc comment.
        let guard = match engine_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        generate_blocking(guard.0, app, conversation_id, history);
    })
    .await
    .map_err(|e| e.to_string())
}
