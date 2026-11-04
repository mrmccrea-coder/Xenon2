//! Local image generation through stable-diffusion.cpp's `sd-cli` (SDXL-class GGUF models).
//!
//! Layout (relative to the data root, see paths.rs):
//!   `image-engine/bin/sd-cli.exe` (+ its DLLs)   the engine, run as a child process
//!   `models/image/*.gguf`                        any number of image models
//! Generated PNGs are written to `<app data>/images/`.
//!
//! A 4 GB GPU cannot hold the chat model and an SDXL model together, so `generate_image` unloads
//! the chat model, runs the engine, then reloads the chat model (see
//! `inference::with_engine_unloaded`). Chat is unavailable for the duration; the UI shows stages.
//!
//! The engine is never run through a shell: the prompt is passed as a single argv entry.

use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::inference::{with_engine_unloaded, EngineState};
use crate::memory::RepoRoot;

const MAX_PROMPT_CHARS: usize = 1000;
const TIMEOUT: Duration = Duration::from_secs(900);
const ALLOWED_SIZES: [u32; 3] = [512, 768, 1024];

fn engine_exe(root: &Path) -> PathBuf {
    let name = if cfg!(windows) { "sd-cli.exe" } else { "sd-cli" };
    root.join("image-engine").join("bin").join(name)
}

fn models_dir(root: &Path) -> PathBuf {
    root.join("models").join("image")
}

fn images_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data folder: {e}"))?
        .join("images");
    std::fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    Ok(dir)
}

#[derive(Serialize, Clone)]
pub struct ImageModel {
    pub file: String,
    pub label: String,
    #[serde(rename = "sizeMb")]
    pub size_mb: u64,
}

#[derive(Serialize)]
pub struct ImageStatus {
    pub available: bool,
    pub reason: Option<String>,
    pub models: Vec<ImageModel>,
    pub sizes: Vec<u32>,
}

/// "sdxl_lightning_4step.q4_0.gguf" -> "sdxl lightning 4step q4 0"
fn label_for(file: &str) -> String {
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    stem.replace(['_', '-', '.'], " ")
}

fn list_models(root: &Path) -> Vec<ImageModel> {
    let mut out: Vec<ImageModel> = std::fs::read_dir(models_dir(root))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.to_lowercase().ends_with(".gguf") {
                return None;
            }
            let size_mb = e.metadata().map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
            Some(ImageModel { label: label_for(&name), file: name, size_mb })
        })
        .collect();
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

#[tauri::command]
pub fn image_status(root: tauri::State<'_, RepoRoot>) -> ImageStatus {
    let models = list_models(&root.0);
    let reason = if !engine_exe(&root.0).exists() {
        Some(format!("Image engine not found (expected {}).", engine_exe(&root.0).display()))
    } else if models.is_empty() {
        Some(format!("No image model found. Put an SDXL .gguf file in {}.", models_dir(&root.0).display()))
    } else {
        None
    };
    ImageStatus { available: reason.is_none(), reason, models, sizes: ALLOWED_SIZES.to_vec() }
}

/// Sampler settings implied by the file name. Distilled "Lightning"/"Turbo" models want a handful
/// of steps and no classifier-free guidance; full SDXL checkpoints want the usual ~20 steps.
fn settings_for(file: &str) -> (u32, &'static str) {
    let f = file.to_lowercase();
    for (tag, steps) in [("1step", 1u32), ("2step", 2), ("4step", 4), ("8step", 8)] {
        if f.contains(tag) {
            return (steps, "1.0");
        }
    }
    if f.contains("lightning") || f.contains("turbo") || f.contains("hyper") {
        return (6, "1.0");
    }
    (20, "6.0")
}

fn random_seed() -> u64 {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(1);
    (n as u64) % 2_000_000_000
}

#[derive(Serialize)]
pub struct ImageResult {
    pub path: String,
    pub seed: u64,
    pub secs: f32,
    /// Set when the image was made but the chat model failed to come back.
    #[serde(rename = "warning")]
    pub warning: Option<String>,
}

fn tail_of(path: &Path, max: usize) -> String {
    let mut s = String::new();
    if let Ok(mut f) = File::open(path) {
        let _ = f.read_to_string(&mut s);
    }
    // Progress bars use carriage returns; keep only lines that are errors/warnings if any exist.
    let errs: Vec<&str> = s.lines().filter(|l| l.contains("[E]") || l.contains("[W]")).collect();
    let pick: Vec<&str> = if errs.is_empty() { s.lines().collect() } else { errs };
    let joined = pick.join("\n");
    let start = joined.len().saturating_sub(max);
    joined[joined.char_indices().map(|(i, _)| i).find(|&i| i >= start).unwrap_or(0)..].to_string()
}

/// Blocking; runs on a worker thread. Waits for `child` with a deadline, killing it on timeout.
fn run_with_timeout(mut cmd: Command, log: &Path) -> Result<(), String> {
    let log_file = File::create(log).map_err(|e| format!("could not create log: {e}"))?;
    let log_err = log_file.try_clone().map_err(|e| e.to_string())?;
    cmd.stdout(Stdio::from(log_file)).stderr(Stdio::from(log_err)).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start the image engine: {e}"))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err(format!("image engine failed ({status}): {}", tail_of(log, 600)))
                };
            }
            Ok(None) => {
                if start.elapsed() > TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("image generation timed out and was stopped".to_string());
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(format!("lost track of the image engine: {e}")),
        }
    }
}

#[tauri::command]
pub async fn generate_image(
    app: AppHandle,
    engine: tauri::State<'_, EngineState>,
    root: tauri::State<'_, RepoRoot>,
    prompt: String,
    model: Option<String>,
    size: Option<u32>,
) -> Result<ImageResult, String> {
    let prompt: String = prompt.trim().trim_start_matches('-').trim().chars().take(MAX_PROMPT_CHARS).collect();
    if prompt.is_empty() {
        return Err("Describe the image you want first.".to_string());
    }
    if prompt.contains('\0') {
        return Err("The prompt contained an invalid character.".to_string());
    }
    let size = size.unwrap_or(768);
    if !ALLOWED_SIZES.contains(&size) {
        return Err(format!("Unsupported image size {size} (use 512, 768 or 1024)."));
    }

    let root_path = root.0.clone();
    let models = list_models(&root_path);
    let chosen = match model {
        // Only ever a bare file name from our own listing -- never a path from the UI.
        Some(m) => models.iter().find(|x| x.file == m).cloned().ok_or_else(|| format!("Unknown image model '{m}'."))?,
        None => models.first().cloned().ok_or_else(|| "No image model installed.".to_string())?,
    };
    let exe = engine_exe(&root_path);
    if !exe.exists() {
        return Err(format!("Image engine not found at {}.", exe.display()));
    }
    let out_dir = images_dir(&app)?;
    let seed = random_seed();
    let out_path = out_dir.join(format!("img_{}_{seed}.png", chrono::Local::now().format("%Y%m%d_%H%M%S")));
    let log_path = out_dir.join("last_run.log");
    let model_path = models_dir(&root_path).join(&chosen.file);
    let (steps, cfg) = settings_for(&chosen.file);

    let engine_state = engine.inner().clone();
    let app2 = app.clone();
    let out2 = out_path.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let t0 = Instant::now();
        let _ = app2.emit("image-stage", "Freeing the chat model...");
        let app3 = app2.clone();
        let (run, reload_err) = with_engine_unloaded(&engine_state, &root_path, move || {
            let _ = app3.emit("image-stage", "Drawing... (the chat model reloads when this finishes)");
            let mut cmd = Command::new(&exe);
            cmd.current_dir(exe.parent().unwrap_or(Path::new(".")))
                .arg("-m").arg(&model_path)
                .arg("-p").arg(&prompt)
                .arg("--cfg-scale").arg(cfg)
                .arg("--steps").arg(steps.to_string())
                .arg("--sampling-method").arg("euler")
                .arg("-W").arg(size.to_string())
                .arg("-H").arg(size.to_string())
                .arg("-s").arg(seed.to_string())
                .arg("--vae-tiling")
                .arg("--offload-to-cpu")
                // Cap the engine's GPU use at ~1 GiB (weights stream from RAM). Measured: peak
                // +1.05 GiB even at 1024 px, which fits beside Xenon2's own ~0.8 GiB CUDA context
                // and desktop apps on a 4 GB card. Without it the engine grabs everything free.
                .arg("--max-vram").arg("1.0")
                .arg("-o").arg(&out2);
            run_with_timeout(cmd, &log_path)
        });
        (run, reload_err, t0.elapsed().as_secs_f32())
    })
    .await
    .map_err(|e| e.to_string())?;

    let (run, reload_err, secs) = result;
    run?;
    if !out_path.exists() {
        return Err("The image engine finished but produced no image.".to_string());
    }
    Ok(ImageResult { path: out_path.to_string_lossy().into_owned(), seed, secs, warning: reload_err })
}

/// Returns a generated image as a data URL. Only files inside the app's own images folder can be
/// read, whatever path the UI passes.
#[tauri::command]
pub fn read_image(app: AppHandle, path: String) -> Result<String, String> {
    let dir = images_dir(&app)?.canonicalize().map_err(|e| e.to_string())?;
    let file = PathBuf::from(&path).canonicalize().map_err(|_| "That image no longer exists.".to_string())?;
    if !file.starts_with(&dir) || file.extension().and_then(|e| e.to_str()) != Some("png") {
        return Err("Not an image generated by Xenon2.".to_string());
    }
    let bytes = std::fs::read(&file).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lightning_files_use_few_steps_and_no_cfg() {
        assert_eq!(settings_for("sdxl_lightning_4step.q4_0.gguf"), (4, "1.0"));
        assert_eq!(settings_for("sdxl_lightning_2step.q5_0.gguf"), (2, "1.0"));
        assert_eq!(settings_for("juggernaut-lightning.gguf"), (6, "1.0"));
        assert_eq!(settings_for("juggernaut-xl-v9-Q4_K.gguf"), (20, "6.0"));
    }

    #[test]
    fn labels_are_readable() {
        assert_eq!(label_for("sdxl_lightning_4step.q4_0.gguf"), "sdxl lightning 4step q4 0");
    }

    #[test]
    fn only_gguf_files_are_listed() {
        let dir = std::env::temp_dir().join(format!("xenon_img_test_{}", std::process::id()));
        let m = dir.join("models").join("image");
        std::fs::create_dir_all(&m).unwrap();
        std::fs::write(m.join("a.gguf"), b"x").unwrap();
        std::fs::write(m.join("b.txt"), b"x").unwrap();
        let list = list_models(&dir);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].file, "a.gguf");
        std::fs::remove_dir_all(&dir).ok();
    }
}
