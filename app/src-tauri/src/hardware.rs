//! Hardware detection and model-tier recommendation.
//!
//! Token generation on these models is memory-bandwidth bound (measured: 4 -> 16 threads moves the
//! 2.9B from 5.2 to 6.0 tok/s), so predicted speed is `effective bandwidth / model size`. The
//! bandwidth is measured once per machine with a ~2 second run of the smallest model and cached,
//! because core counts and RAM sizes do not tell a fast laptop from a slow one.

use std::ffi::{c_void, CString};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::ffi;

/// Voice replies need roughly this to feel live; below `USABLE_TPS` chat is painful.
pub const TARGET_TPS: f64 = 5.0;
pub const USABLE_TPS: f64 = 2.5;
/// Assumed when no measurement exists: a single-channel laptop.
pub const DEFAULT_BW_GBPS: f64 = 8.0;
/// A small model's tok/s overstates what a large one gets (cache effects). Fitted on the dev
/// machine only: 0.4B implied 23 GB/s while the 1.5B and 2.9B measured 11 and 6 tok/s, i.e.
/// ~16 GB/s effective, a ratio of 0.70. Re-fit when more machines have been measured.
const BW_DERATE: f64 = 0.70;
/// GPU offload roughly doubled 1.5B and 2.9B speed on the RTX A1000 (11->20, 6->12.5 tok/s).
const GPU_SPEEDUP: f64 = 1.8;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    /// Known only for NVIDIA (via nvidia-smi); other vendors report `None` rather than a guess.
    pub vram_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareReport {
    pub cpu_name: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub gpus: Vec<GpuInfo>,
    /// True when the linked inference library was built with CUDA, i.e. GPU offload is possible
    /// at all. A GPU in the machine is useless to this build otherwise.
    pub engine_gpu_support: bool,
    pub os: String,
}

pub fn detect() -> HardwareReport {
    use sysinfo::System;
    let mut cpu_name = "Unknown CPU".to_string();
    let (mut logical, mut total_mb, mut avail_mb) = (1usize, 0u64, 0u64);
    if let Ok(mut sys) = System::new() {
        sys.refresh_memory();
        sys.refresh_cpu_all();
        if let Some(c) = sys.cpus().first() {
            let brand = c.brand().trim();
            if !brand.is_empty() {
                cpu_name = brand.to_string();
            }
        }
        logical = sys.cpus().len().max(1);
        total_mb = sys.total_memory() / (1024 * 1024);
        avail_mb = sys.available_memory() / (1024 * 1024);
    }
    HardwareReport {
        cpu_name,
        physical_cores: System::physical_core_count().unwrap_or(logical).max(1),
        logical_cores: logical,
        total_ram_mb: total_mb,
        available_ram_mb: avail_mb,
        gpus: detect_gpus(),
        engine_gpu_support: unsafe { ffi::xenon_has_gpu_support() } != 0,
        os: std::env::consts::OS.to_string(),
    }
}

/// Free RAM right now. Returns a large value if it cannot be read, so callers fall back to the
/// normal (fast, load-new-first) path instead of the cautious one.
pub fn available_ram_mb() -> u64 {
    match sysinfo::System::new() {
        Ok(mut sys) => {
            sys.refresh_memory();
            sys.available_memory() / (1024 * 1024)
        }
        Err(_) => u64::MAX / 2,
    }
}

// ---------------------------------------------------------------------------------------------
// GPU detection (best effort; every probe is optional and time-limited)
// ---------------------------------------------------------------------------------------------

fn run_with_timeout(program: &str, args: &[&str], secs: u64) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flash from a GUI app
    }
    let mut child = cmd.spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(_)) => return None,
            Ok(None) if start.elapsed() > Duration::from_secs(secs) => {
                let _ = child.kill();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
    let mut out = String::new();
    std::io::Read::read_to_string(&mut child.stdout.take()?, &mut out).ok()?;
    Some(out)
}

fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = run_with_timeout(
        "nvidia-smi",
        &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        4,
    )
    .map(|o| parse_nvidia_smi(&o))
    .unwrap_or_default();

    let others = if cfg!(windows) {
        run_with_timeout(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterCompatibility }",
            ],
            6,
        )
        .map(|o| parse_windows_video(&o))
    } else if cfg!(target_os = "macos") {
        run_with_timeout("system_profiler", &["SPDisplaysDataType"], 6).map(|o| parse_macos_displays(&o))
    } else {
        run_with_timeout("lspci", &["-mm"], 4).map(|o| parse_lspci(&o))
    }
    .unwrap_or_default();

    for g in others {
        let already = gpus.iter().any(|k| {
            k.vendor == g.vendor && (k.vendor == "NVIDIA" || k.name.eq_ignore_ascii_case(&g.name))
        });
        if !already {
            gpus.push(g);
        }
    }
    gpus
}

pub fn vendor_of(name: &str) -> String {
    let n = name.to_lowercase();
    if n.contains("nvidia") || n.contains("geforce") || n.contains("rtx") || n.contains("quadro") {
        "NVIDIA".into()
    } else if n.contains("amd") || n.contains("radeon") || n.contains("advanced micro") {
        "AMD".into()
    } else if n.contains("intel") {
        "Intel".into()
    } else if n.contains("apple") {
        "Apple".into()
    } else {
        "Other".into()
    }
}

pub fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|line| {
            let (name, mem) = line.rsplit_once(',')?;
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            Some(GpuInfo {
                name: name.to_string(),
                vendor: "NVIDIA".into(),
                vram_mb: mem.trim().parse::<u64>().ok(),
            })
        })
        .collect()
}

pub fn parse_windows_video(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|line| {
            let (name, vendor) = line.trim().split_once('|')?;
            let name = name.trim();
            // Remote-desktop and virtual adapters are not real GPUs.
            if name.is_empty() || name.contains("Basic") || name.contains("Remote") || name.contains("Virtual") {
                return None;
            }
            let mut v = vendor_of(vendor);
            if v == "Other" {
                v = vendor_of(name);
            }
            Some(GpuInfo { name: name.to_string(), vendor: v, vram_mb: None })
        })
        .collect()
}

pub fn parse_lspci(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|line| {
            // `lspci -mm` prints quoted fields: slot "class" "vendor" "device" ...
            let fields: Vec<&str> = line.split('"').skip(1).step_by(2).collect();
            let class = fields.first()?;
            if !(class.contains("VGA") || class.contains("3D controller") || class.contains("Display")) {
                return None;
            }
            let vendor_field = fields.get(1).copied().unwrap_or("");
            let device = fields.get(2).copied().unwrap_or("");
            Some(GpuInfo {
                name: format!("{} {}", vendor_field, device).trim().to_string(),
                vendor: vendor_of(vendor_field),
                vram_mb: None,
            })
        })
        .collect()
}

pub fn parse_macos_displays(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter_map(|l| {
            let name = l.trim().strip_prefix("Chipset Model:")?.trim();
            Some(GpuInfo { name: name.to_string(), vendor: vendor_of(name), vram_mb: None })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Speed model and recommendation
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TierEstimate {
    pub id: String,
    pub size_mb: u64,
    pub installed: bool,
    pub fits_ram: bool,
    pub gpu_capable: bool,
    pub predicted_tps: f64,
    /// "comfortable" | "usable" | "slow" | "too big"
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Recommendation {
    /// Best tier for this machine whether or not its file is installed (drives "download X").
    pub recommended: String,
    /// Best tier among the installed ones, or `None` when nothing is installed.
    pub best_installed: Option<String>,
    pub reason: String,
    pub bandwidth_gbps: f64,
    pub measured: bool,
    pub estimates: Vec<TierEstimate>,
}

fn usable_gpu(hw: &HardwareReport, size_mb: u64) -> bool {
    hw.engine_gpu_support
        && hw
            .gpus
            .iter()
            .any(|g| g.vendor == "NVIDIA" && g.vram_mb.is_some_and(|v| v >= size_mb + 800))
}

fn fits_ram(hw: &HardwareReport, size_mb: u64) -> bool {
    let size = size_mb as f64;
    hw.available_ram_mb as f64 >= size * 1.25 + 768.0 && hw.total_ram_mb as f64 >= size * 1.8 + 1500.0
}

fn verdict(tps: f64, fits: bool) -> &'static str {
    if !fits {
        "too big"
    } else if tps >= TARGET_TPS {
        "comfortable"
    } else if tps >= USABLE_TPS {
        "usable"
    } else {
        "slow"
    }
}

/// `tiers` is `(id, file size in MB, installed)`; order does not matter.
pub fn estimate_tiers(hw: &HardwareReport, tiers: &[(String, u64, bool)], bandwidth_gbps: f64) -> Vec<TierEstimate> {
    let mut out: Vec<TierEstimate> = tiers
        .iter()
        .map(|(id, size_mb, installed)| {
            let cpu_tps = bandwidth_gbps * BW_DERATE / (*size_mb as f64 / 1024.0);
            // The GPU lost to the CPU on the 0.4B (launch overhead dominates a tiny model).
            let gpu_capable = usable_gpu(hw, *size_mb);
            let tps = if gpu_capable && *size_mb >= 1000 { cpu_tps * GPU_SPEEDUP } else { cpu_tps };
            let fits = fits_ram(hw, *size_mb);
            TierEstimate {
                id: id.clone(),
                size_mb: *size_mb,
                installed: *installed,
                fits_ram: fits,
                gpu_capable,
                predicted_tps: (tps * 10.0).round() / 10.0,
                verdict: verdict(tps, fits).to_string(),
            }
        })
        .collect();
    out.sort_by_key(|e| e.size_mb);
    out
}

fn pick<'a>(candidates: impl Iterator<Item = &'a TierEstimate> + Clone) -> Option<&'a TierEstimate> {
    // Largest model that is still comfortable, else the largest usable one, else the smallest.
    let comfortable = candidates.clone().filter(|e| e.fits_ram && e.predicted_tps >= TARGET_TPS).last();
    let usable = candidates.clone().filter(|e| e.fits_ram && e.predicted_tps >= USABLE_TPS).last();
    comfortable.or(usable).or_else(|| candidates.min_by_key(|e| e.size_mb))
}

pub fn recommend(estimates: Vec<TierEstimate>, bandwidth_gbps: f64, measured: bool) -> Recommendation {
    let best = pick(estimates.iter()).cloned();
    let best_installed = pick(estimates.iter().filter(|e| e.installed)).map(|e| e.id.clone());
    let (recommended, reason) = match &best {
        Some(b) => {
            let how = if b.gpu_capable && b.size_mb >= 1000 { "GPU" } else { "CPU" };
            let reason = match b.verdict.as_str() {
                "comfortable" => format!("about {:.0} tokens/sec on this computer's {how}, fast enough for chat and voice", b.predicted_tps),
                "usable" => format!("about {:.1} tokens/sec on this computer's {how}: usable, but replies will be slow", b.predicted_tps),
                _ => format!("the smallest model; this computer manages only about {:.1} tokens/sec, so expect slow replies", b.predicted_tps),
            };
            (b.id.clone(), reason)
        }
        None => (String::new(), "no models are defined".to_string()),
    };
    Recommendation { recommended, best_installed, reason, bandwidth_gbps, measured, estimates }
}

// ---------------------------------------------------------------------------------------------
// Bandwidth calibration, cached per CPU
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    pub cpu_name: String,
    pub bandwidth_gbps: f64,
}

pub fn load_profile(path: &Path, cpu_name: &str) -> Option<HardwareProfile> {
    let p: HardwareProfile = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    (p.cpu_name == cpu_name && p.bandwidth_gbps > 0.0).then_some(p)
}

pub fn save_profile(path: &Path, p: &HardwareProfile) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_string_pretty(p) {
        let _ = std::fs::write(path, json);
    }
}

extern "C" fn count_token(_text: *const std::os::raw::c_char, _id: u32, user_data: *mut c_void) -> bool {
    unsafe { *(user_data as *mut u32) += 1 };
    true
}

/// Measures effective memory bandwidth in GB/s by timing CPU-only decode of `model_path` (the
/// smallest model): `tokens/sec * file size`. One warm-up pass first so cold disk reads do not
/// masquerade as a slow machine. Returns `None` if the model cannot be loaded or ran too briefly.
pub fn calibrate_cpu_bandwidth(model_path: &Path, vocab_path: &Path, threads: u32) -> Option<f64> {
    let size_gb = std::fs::metadata(model_path).ok()?.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let model_c = CString::new(model_path.to_string_lossy().as_bytes()).ok()?;
    let vocab_c = CString::new(vocab_path.to_string_lossy().as_bytes()).ok()?;
    let prompt = CString::new("User: Count from one to twenty.\n\nXenon:").ok()?;

    let engine = unsafe { ffi::xenon_load_model(model_c.as_ptr(), vocab_c.as_ptr(), threads.max(1), 0) };
    if engine.is_null() {
        return None;
    }
    let mut result = None;
    for pass in 0..2 {
        let mut tokens: u32 = 0;
        let start = Instant::now();
        let status = unsafe {
            ffi::xenon_generate(
                engine,
                prompt.as_ptr(),
                40,
                0.0,
                1.0,
                1.0,
                count_token,
                &mut tokens as *mut u32 as *mut c_void,
            )
        };
        let secs = start.elapsed().as_secs_f64();
        if status != ffi::XenonStatus::OK || tokens < 8 || secs <= 0.0 {
            break;
        }
        if pass == 1 {
            result = Some(tokens as f64 / secs * size_gb);
        }
    }
    unsafe { ffi::xenon_free_engine(engine) };
    result
}

/// CPU threads for calibration; decode is bandwidth bound so more than the physical core count
/// only adds contention.
pub fn calibration_threads(hw: &HardwareReport) -> u32 {
    hw.physical_cores.clamp(2, 8) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(avail_mb: u64, total_mb: u64, gpus: Vec<GpuInfo>, cuda: bool) -> HardwareReport {
        HardwareReport {
            cpu_name: "test".into(),
            physical_cores: 8,
            logical_cores: 16,
            total_ram_mb: total_mb,
            available_ram_mb: avail_mb,
            gpus,
            engine_gpu_support: cuda,
            os: "windows".into(),
        }
    }

    fn tiers(installed: [bool; 3]) -> Vec<(String, u64, bool)> {
        vec![
            ("fast".into(), 500, installed[0]),
            ("balanced".into(), 1475, installed[1]),
            ("quality".into(), 2625, installed[2]),
        ]
    }

    fn rec(hw: &HardwareReport, bw: f64, installed: [bool; 3]) -> Recommendation {
        recommend(estimate_tiers(hw, &tiers(installed), bw), bw, true)
    }

    #[test]
    fn dev_machine_cpu_only_gets_quality() {
        // 23.1 GB/s measured with the 0.4B on the dev machine -> ~6 tok/s for the 2.9B on CPU,
        // which is what a real CPU-only run measured.
        let r = rec(&hw(16000, 32000, vec![], false), 23.1, [true; 3]);
        assert_eq!(r.recommended, "quality");
        let q = r.estimates.iter().find(|e| e.id == "quality").unwrap();
        assert!((q.predicted_tps - 6.3).abs() < 0.5, "got {}", q.predicted_tps);
        let b = r.estimates.iter().find(|e| e.id == "balanced").unwrap();
        assert!((b.predicted_tps - 11.0).abs() < 1.5, "1.5B measured ~11 tok/s, got {}", b.predicted_tps);
    }

    #[test]
    fn slow_laptop_gets_a_smaller_tier() {
        // 8 GB/s -> 2.9B ~2.2 tok/s (slow), 1.5B ~3.9 (usable), 0.4B ~11 (comfortable).
        let r = rec(&hw(9000, 16000, vec![], false), 8.0, [true; 3]);
        assert_eq!(r.recommended, "fast", "balanced is 3.9 tok/s, below the 5.0 target");
        let r = rec(&hw(9000, 16000, vec![], false), 12.0, [true; 3]);
        assert_eq!(r.recommended, "balanced", "12 GB/s -> balanced ~5.8 tok/s");
    }

    #[test]
    fn very_slow_machine_still_falls_back_to_smallest() {
        let r = rec(&hw(4000, 8000, vec![], false), 2.0, [true; 3]);
        assert_eq!(r.recommended, "fast");
        assert!(r.reason.contains("slow"));
    }

    #[test]
    fn low_ram_excludes_big_models_even_if_fast() {
        let r = rec(&hw(2500, 4000, vec![], false), 40.0, [true; 3]);
        let q = r.estimates.iter().find(|e| e.id == "quality").unwrap();
        assert!(!q.fits_ram && q.verdict == "too big");
        assert_ne!(r.recommended, "quality");
    }

    #[test]
    fn nvidia_gpu_needs_cuda_build_and_enough_vram() {
        let gpu = vec![GpuInfo { name: "RTX A1000".into(), vendor: "NVIDIA".into(), vram_mb: Some(4096) }];
        let with_cuda = rec(&hw(16000, 32000, gpu.clone(), true), 8.0, [true; 3]);
        let q = with_cuda.estimates.iter().find(|e| e.id == "quality").unwrap();
        assert!(q.gpu_capable, "2625MB + 800 fits in 4096MB");
        let cpu_only = rec(&hw(16000, 32000, vec![], false), 8.0, [true; 3]);
        let q_cpu = cpu_only.estimates.iter().find(|e| e.id == "quality").unwrap();
        assert!(q.predicted_tps > q_cpu.predicted_tps * 1.5, "GPU speedup applied: {} vs {}", q.predicted_tps, q_cpu.predicted_tps);

        let cpu_build = rec(&hw(16000, 32000, gpu, false), 8.0, [true; 3]);
        assert!(!cpu_build.estimates.iter().any(|e| e.gpu_capable), "CPU-only build cannot use the GPU");

        let small = vec![GpuInfo { name: "GTX 1050".into(), vendor: "NVIDIA".into(), vram_mb: Some(2048) }];
        let r = rec(&hw(16000, 32000, small, true), 8.0, [true; 3]);
        assert!(!r.estimates.iter().find(|e| e.id == "quality").unwrap().gpu_capable);
    }

    #[test]
    fn best_installed_ignores_missing_files() {
        let r = rec(&hw(16000, 32000, vec![], false), 19.4, [true, true, false]);
        assert_eq!(r.recommended, "quality");
        assert_eq!(r.best_installed.as_deref(), Some("balanced"));
        let none = rec(&hw(16000, 32000, vec![], false), 19.4, [false; 3]);
        assert_eq!(none.best_installed, None);
        assert_eq!(none.recommended, "quality");
    }

    #[test]
    fn parses_nvidia_smi() {
        let g = parse_nvidia_smi("NVIDIA RTX A1000 Laptop GPU, 4096\nNVIDIA GeForce RTX 2060, 6144\n");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].vram_mb, Some(4096));
        assert_eq!(g[1].name, "NVIDIA GeForce RTX 2060");
        assert!(parse_nvidia_smi("").is_empty());
    }

    #[test]
    fn parses_windows_video_and_skips_virtual() {
        let g = parse_windows_video("Intel(R) UHD Graphics|Intel Corporation\nAMD Radeon RX 6600|Advanced Micro Devices, Inc.\nMicrosoft Basic Display Adapter|Microsoft\n");
        assert_eq!(g.iter().map(|x| x.vendor.as_str()).collect::<Vec<_>>(), ["Intel", "AMD"]);
    }

    #[test]
    fn parses_lspci() {
        let out = "00:02.0 \"VGA compatible controller\" \"Intel Corporation\" \"Alder Lake-UP3 GT2 [Iris Xe Graphics]\" -r0c\n00:1f.3 \"Audio device\" \"Intel Corporation\" \"Alder Lake PCH\" -r01\n";
        let g = parse_lspci(out);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].vendor, "Intel");
        assert!(g[0].name.contains("Iris Xe"));
    }

    #[test]
    fn profile_is_per_cpu() {
        let dir = std::env::temp_dir().join(format!("xenon_hw_test_{}", std::process::id()));
        let path = dir.join("p.json");
        save_profile(&path, &HardwareProfile { cpu_name: "A".into(), bandwidth_gbps: 12.0 });
        assert!(load_profile(&path, "A").is_some());
        assert!(load_profile(&path, "B").is_none(), "a different CPU must trigger re-measurement");
        let _ = std::fs::remove_dir_all(dir);
    }
}
