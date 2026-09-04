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
use crate::sloth_memory;

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

/// The loaded model's name/version (its filename, minus extension, e.g.
/// `"rwkv-5-world-0.4B-Q4_0"`), recorded so Phase 5's saved conversation files can note which
/// model generated them (see `persistence::ConversationFile::model`). Computed once from the
/// actual path `load_engine` loads, so it can never drift from what's really running.
pub struct ModelInfo(pub String);

#[tauri::command]
pub fn get_model_name(model_info: tauri::State<'_, ModelInfo>) -> String {
    model_info.0.clone()
}

/// Loads the Phase 1 model once at app startup. Panics (fails app startup) rather than silently
/// falling back to a stub -- per the phase spec, this app must call the real inference engine,
/// not reimplement or fake it. Also returns a `ModelInfo` derived from the same path, so the name
/// recorded in saved conversation files (Phase 5) can never drift from what's actually loaded.
pub fn load_engine(repo_root: &PathBuf) -> (EngineState, ModelInfo) {
    // Phase 7 follow-up (model upgrade): swapped from RWKV-5 World 0.4B to RWKV-7 World 2.9B v3
    // -- the 0.4B base model was found to collapse onto generic canned replies for short/
    // greeting-style prompts and got basic arithmetic wrong (see conversation history around
    // 2026-08-01). RWKV-7 ("Goose") is a materially more capable architecture per-parameter --
    // the paper reports RWKV7-World3-2.9B nearly matching Qwen2.5-3B on English benchmarks
    // despite far less training data. Same World tokenizer, same rwkv.cpp conversion pipeline,
    // still CPU-only (measured: CPU 8.95 tok/s vs GPU-offloaded 11.18 tok/s for this model on
    // this machine -- GPU is faster now, unlike the 0.4B result, but CPU-only was kept to avoid
    // making the app hard-depend on a CUDA-capable GPU, matching the project's portable/
    // USB-first goal; revisit if that tradeoff should flip).
    let model_path = repo_root.join("models").join("rwkv-7-world-2.9B-Q5_1.bin");
    let vocab_path = repo_root
        .join("inference-engine")
        .join("data")
        .join("world_vocab.bin");

    let model_name = model_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown-model".to_string());

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

    (
        EngineState(Arc::new(Mutex::new(EnginePtr(engine)))),
        ModelInfo(model_name),
    )
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
fn build_prompt(history: &[ChatTurn], facts: &[sloth_memory::SlothFact]) -> String {
    let mut prompt = String::new();

    // Phase 7 follow-up (2026-08-04): real usage showed the model confidently making up a
    // specific time ("The time is 12:30 PM.") when asked, with no actual clock access at all --
    // then falling back to a canned non-answer when asked how it knew that. Ground every prompt
    // (both agents -- this is basic environmental grounding, not memory, so it's not gated behind
    // Sloth) in the real system clock so time/date questions get a real answer instead of a
    // plausible-sounding guess.
    let now = chrono::Local::now();
    let now_full = now.format("%A, %B %d, %Y, %I:%M %p").to_string();
    let now_time_only = now.format("%I:%M %p").to_string();
    prompt.push_str("The current date and time is ");
    prompt.push_str(&now_full);
    prompt.push_str(".\n\n");

    // Phase 7 follow-up: Sloth's persistent cross-conversation memory. Dementia turns always
    // call this with an empty `facts` slice, so this block is skipped entirely and Dementia's
    // prompt is byte-for-byte what it always was -- Sloth is strictly additive, not a change to
    // the default/no-memory behavior.
    if !facts.is_empty() {
        prompt.push_str(
            "Known facts about the user, remembered from past conversations:\n",
        );
        for fact in facts {
            prompt.push_str("- ");
            prompt.push_str(&fact.text);
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    // Phase 7 follow-up (2026-08-02): the seed example used to end with "How can I help you
    // today?" -- a real usage session showed the model imitating that closing-question *style*
    // on nearly every reply regardless of content ("Is there anything else I can help you
    // with?"), because a small base model leans heavily on the literal style of whatever example
    // it's shown, not just the instruction-like framing around it. Replaced with two example
    // turns that both end declaratively, no trailing question, to actually change the imitated
    // style rather than just ask for it in prose (which this model has already shown it doesn't
    // reliably follow -- see sloth_memory's extraction prompt fix for the same lesson).
    prompt.push_str(
        "The following is a coherent, friendly conversation between a user and Xenon, a \
         helpful voice assistant. Xenon answers naturally and doesn't end every reply by asking \
         what else it can help with.\n\n\
         User: Hello Xenon, how are you doing?\n\n\
         Xenon: Hi! I'm doing well, thanks for asking.\n\n\
         User: What's a fun fact about space?\n\n\
         Xenon: A day on Venus is longer than its year -- it rotates so slowly that one spin \
         takes longer than one full trip around the sun.\n\n",
    );
    // A dedicated example for time questions, using the *real* current time computed above (not
    // a fixed fake value) -- a plain instruction to "use the date/time given above" wasn't
    // reliably followed for short phrasings like "What time is it?" (it worked for a more
    // elaborate phrasing but not this one, tested side by side), so this demonstrates the exact
    // pattern instead. Using the real time here means the example is never factually wrong
    // regardless of when it's generated, so there's no risk of the model anchoring on a stale or
    // made-up value the way a fixed example would.
    prompt.push_str(&format!(
        "User: What time is it?\n\n\
         Xenon: It's currently {now_time_only}.\n\n"
    ));

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
            tail.drain(0..len - 64);
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

/// Builds the extraction prompt asking the model whether the just-completed exchange revealed
/// anything new and durable worth remembering long-term. Kept deliberately strict ("reply with
/// exactly: NONE") since this is a small, non-instruction-tuned base model -- without a strong
/// steer it tends to invent something rather than correctly recognize "nothing new here".
fn build_extraction_prompt(user_text: &str, reply_text: &str) -> String {
    // Few-shot, not just an abstract instruction -- a bare "extract a fact or say NONE"
    // instruction was found to make this small base model *invent* a plausible-sounding fact
    // (e.g. "named John, loves pizza") even when the user never said any such thing, rather than
    // correctly recognizing nothing new was revealed. Demonstrating both the extract case and
    // the NONE case anchors the behavior far better, matching how the main chat prompt already
    // relies on a demonstrated example turn rather than an instruction alone.
    // Real bug found in usage (2026-08-02): the model once misattributed Xenon's *own* stated
    // name as a fact about the user ("The user's name is Xenon") after being asked "What is your
    // name?" -- added a dedicated example so it's explicit that facts about Xenon itself never
    // count as facts about the user, the same conflation risk as the NONE examples below.
    format!(
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
         Fact: NONE\n\n\
         User: {user_text}\n\n\
         Xenon: {reply_text}\n\n\
         Fact:"
    )
}

/// Runs the Sloth fact-extraction step: a second, small, low-temperature generate() call using
/// the just-completed turn, under the *same* already-held engine lock as the main reply (kept
/// sequential rather than concurrent -- rwkv.cpp's per-engine state isn't safe for concurrent
/// eval calls, see `EngineState`'s doc comment, and this call is short at `max_tokens=40`). On
/// success, persists any real fact via `sloth_memory::append_fact` and emits
/// `sloth-facts-updated` so an open Memory panel (see App.vue) can refresh live. Never surfaces
/// its own errors as a `generation-error` -- a failed/garbled extraction just means no new fact
/// gets remembered this turn, which shouldn't interrupt or error out the visible reply that
/// already completed successfully.
fn run_fact_extraction(
    engine: *mut ffi::XenonEngine,
    app: &AppHandle,
    user_text: &str,
    reply_text: &str,
) {
    let prompt = match CString::new(build_extraction_prompt(user_text, reply_text)) {
        Ok(c) => c,
        Err(_) => return,
    };

    let buf = RefCell::new(String::new());
    let buf_ptr = &buf as *const RefCell<String> as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate(
            engine,
            prompt.as_ptr(),
            40,   // max_tokens -- one short sentence, not a full reply
            0.2,  // temperature -- extraction should be far more deterministic than a chat reply
            0.2,  // top_p
            1.1,  // repeat_penalty -- mild; extraction is already short/constrained by the few-shot prompt
            on_extraction_token,
            buf_ptr,
        )
    };
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

/// Runs generation to completion on the calling thread (blocking) and emits Tauri events as
/// tokens stream in. Meant to be called from inside `spawn_blocking` by the IPC command below,
/// never directly on the async/event-loop thread. Takes the already-locked engine pointer so the
/// mutex stays held for the full duration of the call (see `EngineState` doc comment). `agent`
/// ("dementia" | "sloth") controls whether Sloth's persistent facts are injected into the prompt
/// and whether a fact-extraction pass runs after a successful reply -- Dementia leaves both
/// skipped, matching the app's pre-Sloth behavior exactly.
fn generate_blocking(
    engine: *mut ffi::XenonEngine,
    app: AppHandle,
    conversation_id: String,
    history: Vec<ChatTurn>,
    agent: String,
) {
    if let Some(last_user_text) = history.iter().rev().find(|t| t.role == "user").map(|t| t.content.as_str()) {
        if let Some(answer) = try_answer_time_date_deterministically(last_user_text) {
            let _ = app.emit(
                "token-stream",
                TokenEvent { conversation_id: conversation_id.clone(), text: answer },
            );
            let _ = app.emit("generation-done", GenerationDoneEvent { conversation_id });
            return;
        }
    }

    let is_sloth = agent == "sloth";
    let facts = if is_sloth {
        sloth_memory::load_memory(&app).map(|m| m.facts).unwrap_or_default()
    } else {
        Vec::new()
    };

    let prompt = build_prompt(&history, &facts);
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
        full_text: RefCell::new(String::new()),
    };
    let ctx_ptr = &ctx as *const CallbackCtx as *mut c_void;

    let status = unsafe {
        ffi::xenon_generate(
            engine,
            prompt_c.as_ptr(),
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

    if status == ffi::XenonStatus::OK {
        if is_sloth {
            let user_text = history
                .iter()
                .rev()
                .find(|t| t.role == "user")
                .map(|t| t.content.as_str())
                .unwrap_or("");
            let reply_text = ctx.full_text.borrow();
            run_fact_extraction(engine, &app, user_text, &reply_text);
        }
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

/// Tauri IPC command: frontend sends the full turn history (ending with the new user message)
/// plus which agent this turn should use, backend streams the assistant's reply back via
/// `token-stream` / `generation-done` / `generation-error` events tagged with `conversation_id`.
#[tauri::command]
pub async fn generate_response(
    app: AppHandle,
    engine: tauri::State<'_, EngineState>,
    conversation_id: String,
    history: Vec<ChatTurn>,
    agent: String,
) -> Result<(), String> {
    let engine_arc = engine.0.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Held for the entire blocking generate() call (and, for Sloth, the extraction call
        // right after it) -- see EngineState's doc comment.
        let guard = match engine_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        generate_blocking(guard.0, app, conversation_id, history, agent);
    })
    .await
    .map_err(|e| e.to_string())
}
