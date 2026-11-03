# Builds the universal Windows installer (MSI + NSIS setup).
#
# "Universal" = the CPU inference engine, which runs on any PC. The CUDA engine needs an NVIDIA
# driver and ships ~500 MB of cuBLAS, so it is not what a general installer should carry. The
# Tauri bundler packs every DLL sitting beside the release exe, so this script puts the CPU
# engine DLLs there first (build.rs does not copy DLLs on Windows) and removes any CUDA ones.
#
# Prerequisite: the CPU engine, built per inference-engine/README.md ("1a" and "2", CPU variant):
#   inference-engine/build-cpu-app/bin/Release/{rwkv.dll,xenon_inference.dll}
#
# Models are NOT in the installer: first launch opens "Set up models", which downloads them.

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cpu = Join-Path $root "inference-engine\build-cpu-app\bin\Release"
$release = Join-Path $root "app\src-tauri\target\release"

foreach ($f in "rwkv.dll", "xenon_inference.dll") {
    if (-not (Test-Path (Join-Path $cpu $f))) {
        throw "Missing $cpu\$f. Build the CPU engine first (inference-engine/README.md)."
    }
}

New-Item -ItemType Directory -Force $release | Out-Null
Get-ChildItem $release -Filter "cu*.dll" -ErrorAction SilentlyContinue | Remove-Item -Force
Copy-Item (Join-Path $cpu "*.dll") $release -Force
Write-Host "Engine DLLs for the installer: CPU variant ($((Get-Item (Join-Path $release 'rwkv.dll')).Length / 1MB) MB rwkv.dll)"

Push-Location (Join-Path $root "app")
try {
    npm run tauri build
} finally {
    Pop-Location
}

Get-ChildItem (Join-Path $release "bundle") -Recurse -File -Include *.msi, *setup.exe |
    Select-Object FullName, @{N = "MB"; E = { [math]::Round($_.Length / 1MB, 2) } }
