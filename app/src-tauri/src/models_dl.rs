//! Model catalog and resumable, checksum-verified downloader.
//!
//! The files are rwkv.cpp-format (ggml) models converted from BlinkDL's `.pth` checkpoints. That
//! conversion needs Python + torch, so end users cannot do it: the files must be hosted somewhere
//! and fetched by URL. `model_catalog.json` carries each file's exact size and SHA-256 (taken from
//! the files actually tested); where they are hosted is `baseUrl`, left empty until a host exists.
//! Override it without rebuilding via the `XENON_MODEL_BASE_URL` environment variable or a
//! `model_source.txt` file in the app data directory.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogEntry {
    pub tier: String,
    pub file: String,
    pub size_bytes: u64,
    pub sha256: String,
    /// When set, the file is hosted as consecutive parts `<file>.001`, `<file>.002`, ... of this
    /// many bytes each (the last one shorter): GitHub release files are limited to 2 GB and the
    /// Quality model is 2.75 GB. The parts are downloaded into one file and checked as a whole.
    #[serde(default)]
    pub part_bytes: Option<u64>,
}

impl CatalogEntry {
    /// (url, length) of each piece to download, in order. A single piece for unsplit files.
    pub fn pieces(&self, base: &str) -> Vec<(String, u64)> {
        let base = base.trim_end_matches('/');
        match self.part_bytes.filter(|p| *p > 0 && *p < self.size_bytes) {
            None => vec![(format!("{base}/{}", self.file), self.size_bytes)],
            Some(p) => {
                let count = self.size_bytes.div_ceil(p);
                (0..count)
                    .map(|i| {
                        let len = if i + 1 == count { self.size_bytes - p * i } else { p };
                        (format!("{base}/{}.{:03}", self.file, i + 1), len)
                    })
                    .collect()
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Catalog {
    pub base_url: String,
    pub models: Vec<CatalogEntry>,
}

pub fn catalog() -> Catalog {
    serde_json::from_str(include_str!("../model_catalog.json")).expect("model_catalog.json is valid")
}

pub fn entry_for_tier(tier: &str) -> Option<CatalogEntry> {
    catalog().models.into_iter().find(|m| m.tier == tier)
}

/// Where to download from, if anywhere is configured.
pub fn base_url(app: &AppHandle) -> Option<String> {
    let from_env = std::env::var("XENON_MODEL_BASE_URL").ok();
    let from_file = app
        .path()
        .app_data_dir()
        .ok()
        .and_then(|d| fs::read_to_string(d.join("model_source.txt")).ok());
    [from_env, from_file, Some(catalog().base_url)]
        .into_iter()
        .flatten()
        .map(|s| s.trim().to_string())
        .find(|s| !s.is_empty())
}

pub fn free_space_bytes(path: &Path) -> Option<u64> {
    let disks = sysinfo::Disks::new_with_refreshed_list().ok()?;
    let target = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    (&disks)
        .into_iter()
        .filter(|d| target.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

#[derive(Debug, PartialEq)]
pub enum DownloadError {
    Cancelled,
    Failed(String),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::Cancelled => write!(f, "Download cancelled"),
            DownloadError::Failed(m) => write!(f, "{m}"),
        }
    }
}

fn fail<T>(msg: impl Into<String>) -> Result<T, DownloadError> {
    Err(DownloadError::Failed(msg.into()))
}

pub fn sha256_file(path: &Path, cancel: &AtomicBool) -> Result<String, DownloadError> {
    let mut f = File::open(path).map_err(|e| DownloadError::Failed(format!("Could not open '{}': {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        let n = f.read(&mut buf).map_err(|e| DownloadError::Failed(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Downloads `url` to `dest` via `dest.part`, resuming a partial file when the server supports
/// ranges, then verifies size and SHA-256 before renaming into place. A failed checksum deletes
/// the partial file so the next attempt starts clean instead of resuming corrupt data.
/// `progress(done, total)` is called as bytes arrive.
pub fn download_file(
    url: &str,
    dest: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancel: &AtomicBool,
    progress: impl FnMut(u64, u64),
) -> Result<(), DownloadError> {
    download_pieces(&[(url.to_string(), expected_size)], dest, expected_size, expected_sha256, cancel, progress)
}

/// Downloads `pieces` (url, length) one after another into a single `<dest>.part` file, resuming
/// from however much of it already exists -- including from the middle of any piece -- then checks
/// the whole file's SHA-256 and moves it to `dest`.
pub fn download_pieces(
    pieces: &[(String, u64)],
    dest: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<(), DownloadError> {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).map_err(|e| DownloadError::Failed(format!("Could not create '{}': {e}", dir.display())))?;
    }
    if pieces.iter().map(|(_, len)| len).sum::<u64>() != expected_size {
        return fail("The model catalog's piece sizes don't add up to the file size.");
    }

    let mut have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > expected_size {
        let _ = fs::remove_file(&part);
        have = 0;
    }

    let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
    let mut piece_start = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    progress(have, expected_size);
    for (url, len) in pieces {
        let piece_end = piece_start + len;
        if have >= piece_end {
            piece_start = piece_end; // already have all of this piece
            continue;
        }
        // Resume inside this piece if part of it is already on disk.
        let mut offset = have - piece_start;
        let mut req = agent.get(url);
        if offset > 0 {
            req = req.header("Range", format!("bytes={offset}-"));
        }
        let resp = req.call().map_err(|e| DownloadError::Failed(format!("Could not reach the download server: {e}")))?;
        match resp.status().as_u16() {
            206 => {}
            200 => offset = 0, // server ignored the range: redo this piece from its start
            404 => return fail("The model file was not found on the download server."),
            s => return fail(format!("The download server answered with HTTP {s}.")),
        }
        if offset == 0 && have > piece_start {
            // Throw away the partial piece so the full one can be appended from its start.
            let f = OpenOptions::new()
                .write(true)
                .open(&part)
                .map_err(|e| DownloadError::Failed(format!("Could not write '{}': {e}", part.display())))?;
            f.set_len(piece_start).map_err(|e| DownloadError::Failed(format!("Could not write to disk: {e}")))?;
            have = piece_start;
        }

        let mut out = if have == 0 {
            File::create(&part)
        } else {
            OpenOptions::new().append(true).open(&part)
        }
        .map_err(|e| DownloadError::Failed(format!("Could not write '{}': {e}", part.display())))?;

        let mut reader = resp.into_body().into_reader();
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(DownloadError::Cancelled);
            }
            let n = reader
                .read(&mut buf)
                .map_err(|e| DownloadError::Failed(format!("The connection dropped: {e}. Run the download again to resume.")))?;
            if n == 0 {
                break;
            }
            // Never write past this piece (a misbehaving server can't corrupt the next one).
            let room = (piece_end - have) as usize;
            let n = n.min(room);
            out.write_all(&buf[..n]).map_err(|e| DownloadError::Failed(format!("Could not write to disk: {e}")))?;
            have += n as u64;
            progress(have, expected_size);
            if have >= piece_end {
                break;
            }
        }
        out.flush().ok();
        if have < piece_end {
            return fail(format!(
                "The download ended early ({have} of {expected_size} bytes). Run it again to resume."
            ));
        }
        piece_start = piece_end;
    }

    let size = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if size != expected_size {
        return fail(format!("The download ended early ({size} of {expected_size} bytes). Run it again to resume."));
    }
    let actual = sha256_file(&part, cancel)?;
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        let _ = fs::remove_file(&part);
        return fail("The downloaded file failed its checksum and was discarded. Please try again.");
    }
    fs::rename(&part, dest).map_err(|e| DownloadError::Failed(format!("Could not finalize the download: {e}")))
}

// ---------------------------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------------------------

/// The cancel flag of the download in flight, if any; only one download runs at a time.
#[derive(Default)]
pub struct DownloadState(pub Mutex<Option<Arc<AtomicBool>>>);

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgressEvent {
    tier_id: String,
    done: u64,
    total: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DoneEvent {
    tier_id: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorEvent {
    tier_id: String,
    message: String,
    cancelled: bool,
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: tauri::State<'_, DownloadState>,
    root: tauri::State<'_, crate::memory::RepoRoot>,
    tier_id: String,
) -> Result<(), String> {
    let entry = entry_for_tier(&tier_id).ok_or_else(|| format!("Unknown model tier '{tier_id}'"))?;
    let base = base_url(&app).ok_or_else(|| {
        "No download source is configured yet. Set the XENON_MODEL_BASE_URL environment variable, or \
         put the address in model_source.txt in the app data folder, or copy the model file into the \
         models folder by hand."
            .to_string()
    })?;
    let models_dir = root.0.join("models");
    let dest = models_dir.join(&entry.file);
    if dest.exists() {
        return Ok(());
    }

    if let Some(free) = free_space_bytes(&models_dir) {
        let need = entry.size_bytes + 300 * 1024 * 1024;
        if free < need {
            return Err(format!(
                "Not enough disk space: this model needs about {} MB free and only {} MB is available.",
                need / (1024 * 1024),
                free / (1024 * 1024)
            ));
        }
    }

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut slot = state.0.lock().map_err(|_| "download state poisoned".to_string())?;
        if slot.is_some() {
            return Err("Another download is already running.".to_string());
        }
        *slot = Some(cancel.clone());
    }

    let pieces = entry.pieces(&base);
    let app2 = app.clone();
    let tier2 = tier_id.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut last_emit = std::time::Instant::now() - std::time::Duration::from_secs(1);
        download_pieces(&pieces, &dest, entry.size_bytes, &entry.sha256, &cancel, |done, total| {
            if done >= total || last_emit.elapsed() >= std::time::Duration::from_millis(150) {
                last_emit = std::time::Instant::now();
                let _ = app2.emit("model-download-progress", ProgressEvent { tier_id: tier2.clone(), done, total });
            }
        })
    })
    .await
    .map_err(|e| e.to_string());

    if let Ok(mut slot) = state.0.lock() {
        *slot = None;
    }

    match result? {
        Ok(()) => {
            let _ = app.emit("model-download-done", DoneEvent { tier_id });
            Ok(())
        }
        Err(e) => {
            let cancelled = e == DownloadError::Cancelled;
            let _ = app.emit(
                "model-download-error",
                ErrorEvent { tier_id, message: e.to_string(), cancelled },
            );
            Err(e.to_string())
        }
    }
}

#[tauri::command]
pub fn cancel_model_download(state: tauri::State<'_, DownloadState>) {
    if let Ok(slot) = state.0.lock() {
        if let Some(flag) = slot.as_ref() {
            flag.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// Minimal HTTP server: serves `body` for any path, honoring `Range: bytes=N-` unless
    /// `ignore_range`. Handles `connections` requests then exits.
    fn serve(body: Vec<u8>, ignore_range: bool, connections: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..connections {
                let Ok((mut stream, _)) = listener.accept() else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut start = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("range: bytes=") {
                        start = v.trim().trim_end_matches('-').parse().unwrap_or(0);
                    }
                }
                let (head, slice) = if start > 0 && !ignore_range {
                    (
                        format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                            body.len() - start, start, body.len() - 1, body.len()
                        ),
                        &body[start..],
                    )
                } else {
                    (format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()), &body[..])
                };
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(slice);
            }
        });
        format!("http://{addr}/model.bin")
    }

    /// Serves `parts[i]` at any path ending in `.00{i+1}`, honoring `Range: bytes=N-`.
    fn serve_parts(parts: Vec<Vec<u8>>, connections: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..connections {
                let Ok((mut stream, _)) = listener.accept() else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let (mut start, mut index) = (0usize, 0usize);
                let mut first = true;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if first {
                        // "GET /model.bin.002 HTTP/1.1"
                        let path = line.split_whitespace().nth(1).unwrap_or("");
                        index = path.rsplit('.').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(1) - 1;
                        first = false;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("range: bytes=") {
                        start = v.trim().trim_end_matches('-').parse().unwrap_or(0);
                    }
                }
                let body = &parts[index];
                let head = if start > 0 {
                    format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                        body.len() - start, start, body.len() - 1, body.len()
                    )
                } else {
                    format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())
                };
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body[start..]);
            }
        });
        format!("http://{addr}")
    }

    fn split_entry(size: u64, sha: &str, part_bytes: u64) -> CatalogEntry {
        CatalogEntry { tier: "t".into(), file: "model.bin".into(), size_bytes: size, sha256: sha.into(), part_bytes: Some(part_bytes) }
    }

    #[test]
    fn quality_model_is_split_into_two_parts_under_githubs_limit() {
        let q = entry_for_tier("quality").unwrap();
        let pieces = q.pieces("https://example.com/rel/");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].0, "https://example.com/rel/rwkv7-g1-2.9B-Q5_1.bin.001");
        assert_eq!(pieces[1].0, "https://example.com/rel/rwkv7-g1-2.9B-Q5_1.bin.002");
        assert_eq!(pieces.iter().map(|p| p.1).sum::<u64>(), q.size_bytes);
        assert!(pieces.iter().all(|p| p.1 < 2 * 1024 * 1024 * 1024), "each part under GitHub's 2 GiB limit");
    }

    #[test]
    fn downloads_a_split_file_and_verifies_the_whole() {
        let (body, sha) = sample();
        let entry = split_entry(body.len() as u64, &sha, 1_200_000);
        let parts: Vec<Vec<u8>> = body.chunks(1_200_000).map(|c| c.to_vec()).collect();
        assert_eq!(parts.len(), 3);
        let base = serve_parts(parts, 3);
        let dir = tmp("split");
        let dest = dir.join("m.bin");
        download_pieces(&entry.pieces(&base), &dest, body.len() as u64, &sha, &AtomicBool::new(false), |_, _| {}).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn resumes_a_split_file_from_inside_either_part() {
        let (body, sha) = sample();
        for already in [500_000usize, 1_700_000] {
            // 500k: inside part 1; 1.7M: inside part 2 (part 1 complete)
            let entry = split_entry(body.len() as u64, &sha, 1_200_000);
            let parts: Vec<Vec<u8>> = body.chunks(1_200_000).map(|c| c.to_vec()).collect();
            let remaining_requests = parts.len() - already / 1_200_000;
            let base = serve_parts(parts, remaining_requests);
            let dir = tmp(&format!("split_resume_{already}"));
            fs::write(dir.join("m.bin.part"), &body[..already]).unwrap();
            let mut first_progress = None;
            download_pieces(&entry.pieces(&base), &dir.join("m.bin"), body.len() as u64, &sha, &AtomicBool::new(false), |d, _| {
                first_progress.get_or_insert(d);
            })
            .unwrap();
            assert_eq!(first_progress, Some(already as u64), "picked up where it left off");
            assert_eq!(fs::read(dir.join("m.bin")).unwrap(), body);
        }
    }

    fn sample() -> (Vec<u8>, String) {
        let body: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        let sha = Sha256::digest(&body).iter().map(|b| format!("{b:02x}")).collect();
        (body, sha)
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xenon_dl_{}_{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn downloads_and_verifies() {
        let (body, sha) = sample();
        let url = serve(body.clone(), false, 1);
        let dir = tmp("full");
        let dest = dir.join("m.bin");
        let mut last = (0, 0);
        download_file(&url, &dest, body.len() as u64, &sha, &AtomicBool::new(false), |d, t| last = (d, t)).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        assert_eq!(last, (body.len() as u64, body.len() as u64));
        assert!(!dir.join("m.bin.part").exists());
    }

    #[test]
    fn resumes_a_partial_file() {
        let (body, sha) = sample();
        let url = serve(body.clone(), false, 1);
        let dir = tmp("resume");
        fs::write(dir.join("m.bin.part"), &body[..1_200_000]).unwrap();
        let mut first = None;
        download_file(&url, &dir.join("m.bin"), body.len() as u64, &sha, &AtomicBool::new(false), |d, _| {
            first.get_or_insert(d);
        })
        .unwrap();
        assert_eq!(first, Some(1_200_000), "must continue from the partial size, not restart");
        assert_eq!(fs::read(dir.join("m.bin")).unwrap(), body);
    }

    #[test]
    fn restarts_when_server_ignores_range() {
        let (body, sha) = sample();
        let url = serve(body.clone(), true, 1);
        let dir = tmp("norange");
        fs::write(dir.join("m.bin.part"), &body[..500_000]).unwrap();
        download_file(&url, &dir.join("m.bin"), body.len() as u64, &sha, &AtomicBool::new(false), |_, _| {}).unwrap();
        assert_eq!(fs::read(dir.join("m.bin")).unwrap(), body, "a 200 reply must overwrite, not append");
    }

    #[test]
    fn rejects_and_discards_corrupt_download() {
        let (body, _) = sample();
        let url = serve(body.clone(), false, 1);
        let dir = tmp("badsha");
        let err = download_file(&url, &dir.join("m.bin"), body.len() as u64, &"0".repeat(64), &AtomicBool::new(false), |_, _| {}).unwrap_err();
        assert!(matches!(err, DownloadError::Failed(m) if m.contains("checksum")));
        assert!(!dir.join("m.bin").exists());
        assert!(!dir.join("m.bin.part").exists(), "corrupt data must not be resumed next time");
    }

    #[test]
    fn cancel_keeps_partial_for_resume() {
        let (body, sha) = sample();
        let url = serve(body.clone(), false, 1);
        let dir = tmp("cancel");
        let cancel = AtomicBool::new(false);
        let err = download_file(&url, &dir.join("m.bin"), body.len() as u64, &sha, &cancel, |d, _| {
            if d > 1_000_000 {
                cancel.store(true, Ordering::Relaxed);
            }
        })
        .unwrap_err();
        assert_eq!(err, DownloadError::Cancelled);
        assert!(dir.join("m.bin.part").exists());
        assert!(!dir.join("m.bin").exists());
    }

    #[test]
    fn reports_missing_file() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut l = String::new();
            while r.read_line(&mut l).unwrap_or(0) > 0 && l != "\r\n" {
                l.clear();
            }
            let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        });
        let dir = tmp("404");
        let err = download_file(&format!("http://{addr}/x"), &dir.join("m.bin"), 10, "00", &AtomicBool::new(false), |_, _| {}).unwrap_err();
        assert!(matches!(err, DownloadError::Failed(m) if m.contains("not found")));
    }

    #[test]
    fn catalog_is_well_formed() {
        let c = catalog();
        for tier in ["quality"] {
            let e = c.models.iter().find(|m| m.tier == tier).unwrap_or_else(|| panic!("missing {tier}"));
            assert_eq!(e.sha256.len(), 64);
            assert!(e.size_bytes > 100_000_000);
        }
    }
}
