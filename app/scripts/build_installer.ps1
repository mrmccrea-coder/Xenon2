# Builds the Windows installer (NSIS setup + MSI) for public release.
#
# One installer for every PC: it carries the CUDA inference engine, which runs on the GPU when an
# NVIDIA card and driver are present and falls back to the CPU otherwise (verified 2026-11-04 by
# hiding the GPU with CUDA_VISIBLE_DEVICES=-1: the app reports "CPU (GPU offload failed)" and
# chats normally). The CUDA engine does not load NVIDIA's driver at startup, so it launches on
# machines without one. Cost: ~370 MB installer, mostly cuBLAS.
#
# The engine DLLs are bundled through tauri.conf.json `bundle.resources`, straight from
# inference-engine/build-cuda-app/bin/Release. This script makes sure everything they need is
# there first -- in particular the OpenMP runtime vcomp140.dll that rwkv.dll imports. It is part
# of the Visual C++ redistributable, so it exists on developer machines (and was silently found
# in System32 here) but not on a clean PC, where the app would fail to start without it.
#
# Prerequisite: the CUDA engine, built per inference-engine/README.md:
#   inference-engine/build-cuda-app/bin/Release/{rwkv.dll,xenon_inference.dll,cublas*.dll,...}
#
# Models are NOT in the installer: first launch opens "Set up models", which downloads them from
# the GitHub release named in src-tauri/model_catalog.json (`baseUrl`).

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$engine = Join-Path $root "inference-engine\build-cuda-app\bin\Release"
$release = Join-Path $root "app\src-tauri\target\release"

foreach ($f in "rwkv.dll", "xenon_inference.dll", "cublas64_13.dll", "cublasLt64_13.dll") {
    if (-not (Test-Path (Join-Path $engine $f))) {
        throw "Missing $engine\$f. Build the CUDA engine first (inference-engine/README.md)."
    }
}

# vcomp140.dll (OpenMP), from the Visual Studio redistributable folder (redistribution is allowed).
if (-not (Test-Path (Join-Path $engine "vcomp140.dll"))) {
    $vs = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise"
    ) | Where-Object { Test-Path $_ }
    $omp = $vs | ForEach-Object { Get-ChildItem (Join-Path $_ "VC\Redist\MSVC\*\x64\Microsoft.VC143.OpenMP\vcomp140.dll") -ErrorAction SilentlyContinue } |
        Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $omp) { throw "vcomp140.dll not found in a Visual Studio redist folder; install the VS 2022 C++ build tools." }
    Copy-Item $omp.FullName $engine -Force
    Write-Host "Copied $($omp.FullName) -> $engine"
}

Push-Location (Join-Path $root "app")
try {
    npm run tauri build
} finally {
    Pop-Location
}

Get-ChildItem (Join-Path $release "bundle") -Recurse -File -Include *.msi, *setup.exe |
    Select-Object FullName, @{N = "MB"; E = { [math]::Round($_.Length / 1MB, 2) } }
