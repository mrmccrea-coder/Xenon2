# Builds the Microsoft Store test package for Xenon2 Test.
#   .\store\build-msix.ps1 -Version 0.2.0.0 [-IdentityName ... -Publisher "CN=..." -PublisherDisplayName ...]
# Without Partner Center values it uses local placeholders: fine for a sideload test, NOT for upload.
# Read COMPLIANCE.md before submitting.
param(
  [string]$Version = "0.2.0.0",
  [string]$IdentityName = "LOCAL.Xenon2Test",
  [string]$Publisher = "CN=Xenon2LocalTest",
  [string]$PublisherDisplayName = "Xenon2 (local test)"
)
$ErrorActionPreference = "Stop"
$root  = Split-Path $PSScriptRoot -Parent
$app   = Join-Path $root "app"
$tauri = Join-Path $app "src-tauri"
$out   = Join-Path $PSScriptRoot "out"
$stage = Join-Path $out "stage"

if ($Version -notmatch '^\d+\.\d+\.\d+\.0$') { throw "Store versions must be a.b.c.0 (last part must be 0)." }

# 1. Build the exe with the store feature (no installer bundle; MSIX replaces it).
Push-Location $app
# Through cmd so Windows PowerShell 5.1 doesn't treat Tauri's stderr info lines as fatal errors.
cmd /c "npx tauri build --features store --config ..\store\tauri.store.conf.json --no-bundle 2>&1"
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "tauri build failed" }
Pop-Location

# 2. Stage files: exe + the bundle.resources from tauri.conf.json, flattened beside it.
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force (Join-Path $stage "Assets") | Out-Null
$exe = Get-ChildItem (Join-Path $tauri "target\release") -Filter *.exe | Where-Object { $_.Name -notmatch 'build-script' } | Select-Object -First 1
Copy-Item $exe.FullName (Join-Path $stage "xenon2.exe")
$conf = Get-Content (Join-Path $tauri "tauri.conf.json") -Raw | ConvertFrom-Json
foreach ($p in $conf.bundle.resources.PSObject.Properties) {
  Copy-Item (Join-Path $tauri $p.Name) (Join-Path $stage $p.Value)
}
# Bundled models: all three tiers, so the app works straight after install with no setup screen.
New-Item -ItemType Directory -Force (Join-Path $stage "models") | Out-Null
foreach ($m in "rwkv7-g1-0.4B-Q5_1.bin","rwkv7-g1-1.5B-Q5_1.bin","rwkv7-g1-2.9B-Q5_1.bin") {
  Copy-Item (Join-Path (Join-Path $root "models") $m) (Join-Path $stage "models")
}
# Image generator: engine (minus the unused server + scripts) and the picture model + VAE.
# The 6.9 GB inpainting model is left out: brush-to-edit is not in the UI yet.
$imgBin = Join-Path $stage "image-engine\bin"
New-Item -ItemType Directory -Force $imgBin, (Join-Path $stage "models\image") | Out-Null
Get-ChildItem (Join-Path $root "image-engine\bin") | Where-Object { $_.Name -notmatch '^(sd-server\.exe|start-server\..*)$' } |
  ForEach-Object { Copy-Item $_.FullName $imgBin }
foreach ($m in "sdxl_lightning_4step.q4_0.gguf","sdxl_vae_fp16_fix.safetensors") {
  Copy-Item (Join-Path $root "models\image\$m") (Join-Path $stage "models\image")
}
foreach ($i in "StoreLogo.png","Square150x150Logo.png","Square44x44Logo.png") {
  Copy-Item (Join-Path $tauri "icons\$i") (Join-Path $stage "Assets\$i")
}
Copy-Item (Join-Path $root "LICENSE*") $stage -ErrorAction SilentlyContinue

# 3. Manifest.
(Get-Content (Join-Path $PSScriptRoot "AppxManifest.template.xml") -Raw) `
  -replace '{{IdentityName}}',$IdentityName -replace '{{Publisher}}',$Publisher `
  -replace '{{PublisherDisplayName}}',$PublisherDisplayName -replace '{{Version}}',$Version |
  Set-Content (Join-Path $stage "AppxManifest.xml") -Encoding utf8

# 4. Pack.
$makeappx = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\makeappx.exe" | Sort-Object FullName | Select-Object -Last 1
$msix = Join-Path $out "Xenon2Test_$Version`_x64.msix"
& $makeappx.FullName pack /o /d $stage /p $msix
if ($LASTEXITCODE -ne 0) { throw "makeappx failed" }
Write-Host "Built $msix"
Write-Host "Local test: sign it with a self-signed cert matching Publisher, or upload unsigned to Partner Center (the Store signs it)."
