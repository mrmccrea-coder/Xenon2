//! Where the app keeps models and finds its vocabulary file.
//!
//! In a development checkout everything lives under the repo (`models/`,
//! `inference-engine/data/`), found through the path baked in at compile time. An installed copy
//! has no checkout, so it falls back to a portable layout next to the executable, then to the
//! per-user app data folder, which is where downloaded models go.

use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager};

const VOCAB: &str = "world_vocab.bin";

/// Pure so the precedence is testable: explicit override, then the development checkout, then a
/// portable folder beside the executable (anything with `models/` or the vocab file), then the
/// per-user data folder.
pub fn choose_root(
    env_override: Option<PathBuf>,
    dev_checkout: PathBuf,
    exe_dir: Option<PathBuf>,
    data_dir: PathBuf,
    exists: impl Fn(&Path) -> bool,
) -> PathBuf {
    if let Some(p) = env_override.filter(|p| exists(p)) {
        return p;
    }
    if exists(&dev_checkout.join("inference-engine").join("data").join(VOCAB)) {
        return dev_checkout;
    }
    if let Some(dir) = exe_dir {
        if exists(&dir.join("models")) || exists(&dir.join(VOCAB)) {
            return dir;
        }
    }
    data_dir
}

pub fn resolve_repo_root(app: &AppHandle) -> PathBuf {
    let dev_checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let data_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let root = choose_root(
        std::env::var_os("XENON_HOME").map(PathBuf::from),
        dev_checkout,
        exe_dir,
        data_dir,
        |p| p.exists(),
    );
    let _ = std::fs::create_dir_all(root.join("models"));
    root
}

/// The tokenizer vocabulary: the repo copy in a checkout, otherwise the one shipped beside the
/// executable (or in a `resources` folder there), otherwise inside `models/`.
pub fn resolve_vocab(root: &Path) -> PathBuf {
    let mut candidates = vec![root.join("inference-engine").join("data").join(VOCAB)];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(VOCAB));
            candidates.push(dir.join("resources").join(VOCAB));
        }
    }
    candidates.push(root.join("models").join(VOCAB));
    candidates.iter().find(|p| p.exists()).cloned().unwrap_or_else(|| candidates[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn fs(paths: &[&str]) -> impl Fn(&Path) -> bool {
        let set: HashSet<PathBuf> = paths.iter().map(PathBuf::from).collect();
        move |p| set.contains(p)
    }

    #[test]
    fn env_override_wins_when_it_exists() {
        let r = choose_root(
            Some("/custom".into()),
            "/repo".into(),
            Some("/app".into()),
            "/data".into(),
            fs(&["/custom", "/repo/inference-engine/data/world_vocab.bin"]),
        );
        assert_eq!(r, PathBuf::from("/custom"));
    }

    #[test]
    fn missing_override_is_ignored() {
        let r = choose_root(Some("/nope".into()), "/repo".into(), None, "/data".into(), fs(&["/repo/inference-engine/data/world_vocab.bin"]));
        assert_eq!(r, PathBuf::from("/repo"));
    }

    #[test]
    fn dev_checkout_beats_exe_dir() {
        let r = choose_root(None, "/repo".into(), Some("/app".into()), "/data".into(), fs(&["/repo/inference-engine/data/world_vocab.bin", "/app/models"]));
        assert_eq!(r, PathBuf::from("/repo"));
    }

    #[test]
    fn installed_copy_uses_portable_folder_then_data_dir() {
        let portable = choose_root(None, "/gone".into(), Some("/usb/xenon".into()), "/data".into(), fs(&["/usb/xenon/models"]));
        assert_eq!(portable, PathBuf::from("/usb/xenon"));
        let installed = choose_root(None, "/gone".into(), Some("/Program Files/Xenon2".into()), "/data".into(), fs(&[]));
        assert_eq!(installed, PathBuf::from("/data"), "no checkout, nothing beside the exe: per-user data folder");
    }
}
