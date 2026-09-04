//! Phase 7 -- wires the desktop app's mic button to Phase 2's standalone voice pipeline
//! (`voice-pipeline/`: silero-vad -> faster-whisper STT -> ... -> piper TTS).
//!
//! Chosen integration: run `voice-pipeline/ipc_server.py` as a long-lived sidecar process,
//! communicating over newline-delimited JSON on stdin/stdout (see that file's module doc for
//! the protocol). The alternative considered -- porting VAD/STT/TTS to Rust bindings -- would
//! mean re-verifying faster-whisper/piper/silero-vad's GPU/CPU device placement and CUDA-DLL
//! workarounds (see voice-pipeline/README.md) all over again in a different binding layer, for
//! libraries that only ship first-class Python packages. The sidecar reuses Phase 2's exact,
//! already-verified code (including `IncrementalSpeaker`'s sentence-chunked streaming TTS)
//! unchanged.
//!
//! Deliberately does NOT call `xenon_generate()` from Python -- `inference.rs` owns the one
//! loaded RWKV engine, and per phase7_prompt.md task 1 ("do not build a parallel/separate code
//! path for voice-originated messages"), a voice transcript is fed into the frontend's existing
//! `sendMessage` -> `generate_response` path exactly like typed text. This module only bridges
//! the two edges: mic audio -> transcript (`voice_listen`), and streamed reply text -> speech
//! (`voice_speak_start`/`voice_speak_feed`/`voice_speak_finish`, fed fragment-by-fragment from
//! the frontend's `token-stream` handler -- see App.vue).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// How long a voice-pipeline command may take before giving up and surfacing an error. Listen
/// has its own, larger, caller-supplied budget (see `voice_listen`); this is used for the quick
/// speak_start/speak_feed/speak_finish acks.
const QUICK_TIMEOUT: Duration = Duration::from_secs(10);

/// Guards the sidecar's stdin/response channel so only one command is in flight at a time --
/// mirrors `inference::EngineState`'s mutex-around-the-whole-call approach, for the same reason
/// (the sidecar's command loop is single-threaded and processes one line at a time).
pub struct VoiceProcess {
    /// `None` if the sidecar failed to spawn at all (e.g. the venv/ipc_server.py aren't present
    /// in this checkout) -- commands then fail fast with `spawn_error` instead of hanging.
    stdin: Mutex<Option<std::process::ChildStdin>>,
    responses: Mutex<Option<Receiver<serde_json::Value>>>,
    ready: Arc<AtomicBool>,
    spawn_error: Option<String>,
    request_lock: Mutex<()>,
    _child: Mutex<Option<Child>>,
}

#[derive(Clone)]
pub struct VoiceState(pub Arc<VoiceProcess>);

#[derive(Debug, Clone, Serialize)]
pub struct VoiceListenResult {
    pub ok: bool,
    pub text: Option<String>,
    pub reason: Option<String>,
}

fn venv_python(voice_pipeline_dir: &PathBuf) -> PathBuf {
    let win = voice_pipeline_dir.join(".venv").join("Scripts").join("python.exe");
    if win.exists() {
        return win;
    }
    voice_pipeline_dir.join(".venv").join("bin").join("python3")
}

/// Spawns `ipc_server.py` and starts a background reader thread that demultiplexes its stdout:
/// `speak_done` events are emitted directly as a `voice-speak-done` Tauri event (no command is
/// waiting on them specifically -- `voice_speak_finish` returns as soon as its `ack` arrives, not
/// when playback finishes), everything else is forwarded to `responses` for whichever command is
/// currently waiting. Never panics -- a spawn failure is recorded in `spawn_error` so the app
/// still starts without voice support (per phase7_prompt.md, voice is new functionality being
/// added, not something the rest of the app should become unusable without).
pub fn spawn_voice_process(repo_root: &PathBuf, app: AppHandle) -> Arc<VoiceProcess> {
    let voice_pipeline_dir = repo_root.join("voice-pipeline");
    let python = venv_python(&voice_pipeline_dir);
    let script = voice_pipeline_dir.join("ipc_server.py");

    let spawn_result = Command::new(&python)
        .arg(&script)
        .current_dir(&voice_pipeline_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // sidecar's own diagnostics -- visible in the dev console, never mixed into the stdout JSON protocol
        .spawn();

    let mut child = match spawn_result {
        Ok(c) => c,
        Err(e) => {
            let msg = format!(
                "Could not start voice pipeline sidecar ('{}' '{}'): {}. Voice input/output is \
                 unavailable this session -- typed chat still works normally.",
                python.display(),
                script.display(),
                e
            );
            eprintln!("[voice] {msg}");
            return Arc::new(VoiceProcess {
                stdin: Mutex::new(None),
                responses: Mutex::new(None),
                ready: Arc::new(AtomicBool::new(false)),
                spawn_error: Some(msg),
                request_lock: Mutex::new(()),
                _child: Mutex::new(None),
            });
        }
    };

    let stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");

    let (tx, rx) = mpsc::channel::<serde_json::Value>();
    let ready = Arc::new(AtomicBool::new(false));
    let ready_writer = ready.clone();

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break, // pipe closed -- sidecar exited
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[voice] sidecar emitted non-JSON line, ignoring: {trimmed} ({e})");
                    continue;
                }
            };

            match value.get("type").and_then(|t| t.as_str()) {
                Some("ready") => {
                    ready_writer.store(true, Ordering::SeqCst);
                }
                Some("speak_done") => {
                    let _ = app.emit("voice-speak-done", ());
                }
                _ => {
                    let _ = tx.send(value);
                }
            }
        }
        ready_writer.store(false, Ordering::SeqCst);
    });

    Arc::new(VoiceProcess {
        stdin: Mutex::new(Some(stdin)),
        responses: Mutex::new(Some(rx)),
        ready,
        spawn_error: None,
        request_lock: Mutex::new(()),
        _child: Mutex::new(Some(child)),
    })
}

/// Writes one JSON command line to the sidecar's stdin and waits (up to `timeout`) for the next
/// response line. Serialized via `request_lock` since the sidecar processes one command at a
/// time and `speak_done` events are drained out-of-band by the reader thread, so there is exactly
/// one "next relevant line" per outstanding request.
fn send_command(
    process: &VoiceProcess,
    cmd: serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    if let Some(err) = &process.spawn_error {
        return Err(err.clone());
    }
    let _guard = process.request_lock.lock().map_err(|_| "voice process lock poisoned".to_string())?;

    {
        let mut stdin_guard = process.stdin.lock().map_err(|_| "voice stdin lock poisoned".to_string())?;
        let stdin = stdin_guard.as_mut().ok_or("voice sidecar is not running")?;
        let mut line = serde_json::to_string(&cmd).map_err(|e| e.to_string())?;
        line.push('\n');
        stdin.write_all(line.as_bytes()).map_err(|e| format!("could not write to voice sidecar: {e}"))?;
        stdin.flush().map_err(|e| format!("could not flush voice sidecar stdin: {e}"))?;
    }

    let responses_guard = process.responses.lock().map_err(|_| "voice responses lock poisoned".to_string())?;
    let rx = responses_guard.as_ref().ok_or("voice sidecar is not running")?;
    let value = rx
        .recv_timeout(timeout)
        .map_err(|_| "voice sidecar did not respond in time".to_string())?;

    if value.get("type").and_then(|t| t.as_str()) == Some("error") {
        let msg = value
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown voice pipeline error")
            .to_string();
        return Err(msg);
    }

    Ok(value)
}

#[tauri::command]
pub fn voice_ready(voice: tauri::State<'_, VoiceState>) -> bool {
    voice.0.spawn_error.is_none() && voice.0.ready.load(Ordering::SeqCst)
}

/// Records from the mic, VAD-gated, and transcribes the resulting utterance. Blocks (on a
/// background thread, not the async runtime) for up to `max_listen_sec` waiting for speech to
/// start, then up to `max_utterance_sec` more while it's captured, plus STT time -- the timeout
/// passed to the sidecar covers all of that with a margin for transcription itself.
#[tauri::command]
pub async fn voice_listen(
    voice: tauri::State<'_, VoiceState>,
    max_listen_sec: Option<f64>,
    max_utterance_sec: Option<f64>,
) -> Result<VoiceListenResult, String> {
    let process = voice.0.clone();
    let max_listen = max_listen_sec.unwrap_or(6.0);
    let max_utterance = max_utterance_sec.unwrap_or(10.0);
    let timeout = Duration::from_secs_f64(max_listen + max_utterance + 15.0);

    tauri::async_runtime::spawn_blocking(move || {
        let cmd = serde_json::json!({
            "cmd": "listen",
            "maxListenSec": max_listen,
            "maxUtteranceSec": max_utterance,
        });
        let value = send_command(&process, cmd, timeout)?;
        Ok(VoiceListenResult {
            ok: value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
            text: value.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()),
            reason: value.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string()),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Starts a new incremental-speech turn. Call once right before streaming an assistant reply
/// that should be spoken aloud (i.e. one that was triggered by a voice `voice_listen` transcript
/// -- see App.vue's `speakReplies` flag), before any `voice_speak_feed` calls for that turn.
#[tauri::command]
pub async fn voice_speak_start(voice: tauri::State<'_, VoiceState>) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "speak_start" }), QUICK_TIMEOUT).map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Feeds one streamed text fragment (the same fragment forwarded by a `token-stream` event) into
/// the sidecar's `IncrementalSpeaker`, which buffers it and synthesizes+plays each completed
/// sentence as soon as it's available -- see `speech_streamer.py`.
#[tauri::command]
pub async fn voice_speak_feed(voice: tauri::State<'_, VoiceState>, text: String) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "speak_feed", "text": text }), QUICK_TIMEOUT)
            .map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Flushes any remaining buffered text as a final sentence. Returns as soon as the sidecar
/// acknowledges the command (not once playback finishes) -- listen for the `voice-speak-done`
/// Tauri event if the UI needs to know when speech audio has actually finished playing.
#[tauri::command]
pub async fn voice_speak_finish(voice: tauri::State<'_, VoiceState>) -> Result<(), String> {
    let process = voice.0.clone();
    tauri::async_runtime::spawn_blocking(move || {
        send_command(&process, serde_json::json!({ "cmd": "speak_finish" }), QUICK_TIMEOUT).map(|_| ())
    })
    .await
    .map_err(|e| e.to_string())?
}
