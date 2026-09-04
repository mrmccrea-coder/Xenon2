mod ffi;
mod inference;
mod memory;
mod persistence;
mod sloth_memory;
mod voice;

use std::path::PathBuf;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Repo root = two levels up from this crate (app/src-tauri -> app -> <repo root>),
            // so the app finds Phase 1's model/vocab files without needing them copied into
            // app/ or bundled -- fine for this dev-focused phase; portable/USB-relative paths
            // are Phase 6's job (see PLAN.md).
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let repo_root = manifest_dir
                .parent()
                .expect("app/ dir")
                .parent()
                .expect("repo root")
                .to_path_buf();

            let (engine_state, model_info) = inference::load_engine(&repo_root);
            app.manage(engine_state);
            app.manage(model_info);
            // Phase 6: export/import need to find models/, inference-engine/data/, and
            // voice-pipeline/models/ relative to the repo root too, so it's kept as managed state
            // instead of being recomputed from CARGO_MANIFEST_DIR a second time.
            app.manage(memory::RepoRoot(repo_root.clone()));

            // Phase 7: spawn the voice-pipeline sidecar (see voice.rs's module doc). Model
            // loading happens inside the sidecar process itself (a few seconds for STT/TTS), so
            // this doesn't block the app's own startup -- voice_ready() lets the frontend poll
            // for when it's actually usable, and a spawn failure (e.g. voice-pipeline's venv
            // isn't set up in this checkout) degrades to "voice unavailable" rather than
            // panicking, since typed chat must keep working regardless.
            let voice_process = voice::spawn_voice_process(&repo_root, app.handle().clone());
            app.manage(voice::VoiceState(voice_process));

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            inference::generate_response,
            inference::get_model_name,
            persistence::save_conversation_file,
            persistence::open_conversation_file,
            persistence::pick_save_dialog,
            persistence::pick_open_dialog,
            persistence::default_conversation_path,
            persistence::load_session_file,
            persistence::save_session_file,
            memory::load_settings,
            memory::save_settings,
            memory::effective_conversations_dir,
            memory::pick_folder_dialog,
            memory::export_memory,
            memory::import_memory,
            sloth_memory::list_sloth_facts,
            sloth_memory::delete_sloth_fact,
            sloth_memory::clear_sloth_facts,
            voice::voice_ready,
            voice::voice_listen,
            voice::voice_speak_start,
            voice::voice_speak_feed,
            voice::voice_speak_finish,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
