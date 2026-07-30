mod ffi;
mod inference;

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

            let engine_state = inference::load_engine(&repo_root);
            app.manage(engine_state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![inference::generate_response])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
