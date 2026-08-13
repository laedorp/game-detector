param(
    [ValidateSet("cuda", "directml")]
    [string]$RuntimeVariant = "cuda"
)

$ErrorActionPreference = "Stop"
$env:PROAIM_RUNTIME_VARIANT = $RuntimeVariant

$ProjectDir = Split-Path -Parent $PSScriptRoot
$ProjectPython = if ($env:GAME_DETECTOR_PYTHON) {
    $env:GAME_DETECTOR_PYTHON
} else {
    Join-Path $ProjectDir ".venv\Scripts\python.exe"
}

if (-not (Test-Path (Join-Path $ProjectDir "app.py") -PathType Leaf)) {
    throw "Desktop entry point is missing: app.py"
}

if (-not (Test-Path $ProjectPython -PathType Leaf)) {
    throw "Python environment not found at $ProjectPython. Create .venv and install requirements-build.txt first."
}

$ReleasePreflight = Join-Path $ProjectDir "scripts\validate_release_assets.py"
& $ProjectPython $ReleasePreflight --project-root $ProjectDir
if ($LASTEXITCODE -ne 0) {
    throw "Release asset preflight failed with exit code $LASTEXITCODE. PyInstaller was not run."
}

& $ProjectPython -c "import PyInstaller"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is missing. Run: `"$ProjectPython`" -m pip install -r requirements-build.txt"
}

& $ProjectPython -c "import serial, serial.tools.list_ports"
if ($LASTEXITCODE -ne 0) {
    throw "MAKCU support requires pyserial. Install requirements.txt before building."
}

& $ProjectPython -c "import onnxruntime"
if ($LASTEXITCODE -ne 0) {
    throw "Windows release requires ONNX Runtime. Install requirements-build.txt before building."
}

if ($RuntimeVariant -eq "cuda") {
    & $ProjectPython -c "from importlib import metadata; metadata.version('onnxruntime-gpu')"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA variant requires onnxruntime-gpu. Install it in the build environment before building."
    }
} else {
    & $ProjectPython -c "from importlib import metadata; metadata.version('onnxruntime-directml')"
    if ($LASTEXITCODE -ne 0) {
        throw "DirectML variant requires onnxruntime-directml. Install it in the build environment before building."
    }
}

Push-Location $ProjectDir
try {
    & $ProjectPython -m PyInstaller --noconfirm --clean "packaging\game_detector.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

$BundleDir = Join-Path $ProjectDir "dist\ProAim"
$LegacyBundleDir = Join-Path $ProjectDir "dist\GameDetector"
$BundleDisplayName = "ProAim"
if (-not (Test-Path $BundleDir -PathType Container)) {
    if (Test-Path $LegacyBundleDir -PathType Container) {
        $BundleDir = $LegacyBundleDir
        $BundleDisplayName = "GameDetector"
    } else {
        throw "Expected bundle directory not found: $BundleDir"
    }
}
$TesterGuide = Join-Path $ProjectDir "packaging\windows\README-Windows.txt"
$ZipSuffix = if ($RuntimeVariant -eq "cuda") { "NVIDIA-CUDA" } else { "DirectML" }
$ZipPath = Join-Path $ProjectDir ("dist\ProAim-Windows-x64-" + $ZipSuffix + ".zip")
Copy-Item $TesterGuide (Join-Path $BundleDir "README-Windows.txt") -Force
# PyInstaller datas live under _internal. Copy the user-facing legal/help
# documents beside the executables as well so archive recipients can find them
# without knowing the frozen runtime layout.
Copy-Item (Join-Path $ProjectDir "LICENSE") (Join-Path $BundleDir "LICENSE") -Force
Copy-Item (Join-Path $ProjectDir "README.md") (Join-Path $BundleDir "README.md") -Force
Copy-Item (Join-Path $ProjectDir "THIRD_PARTY_NOTICES.md") (Join-Path $BundleDir "THIRD_PARTY_NOTICES.md") -Force
$DocsDir = Join-Path $BundleDir "docs"
New-Item -ItemType Directory -Path $DocsDir -Force | Out-Null
Copy-Item (Join-Path $ProjectDir "docs\MODEL_BENCHMARKS.md") (Join-Path $DocsDir "MODEL_BENCHMARKS.md") -Force
Copy-Item (Join-Path $ProjectDir "docs\RELEASE_CHECKLIST.md") (Join-Path $DocsDir "RELEASE_CHECKLIST.md") -Force
$LicenseDir = Join-Path $BundleDir "licenses"
New-Item -ItemType Directory -Path $LicenseDir -Force | Out-Null
$QtLicenseSource = Join-Path $ProjectDir "packaging\licenses\LGPL-3.0-only.txt"
if (-not (Test-Path $QtLicenseSource -PathType Leaf)) {
    throw "Qt LGPL license text is missing: $QtLicenseSource"
}
Copy-Item $QtLicenseSource (Join-Path $LicenseDir "LGPL-3.0-only.txt") -Force
$QtGplLicenseSource = Join-Path $ProjectDir "packaging\licenses\GPL-3.0-only.txt"
if (-not (Test-Path $QtGplLicenseSource -PathType Leaf)) {
    throw "Qt GPL license text is missing: $QtGplLicenseSource"
}
Copy-Item $QtGplLicenseSource (Join-Path $LicenseDir "GPL-3.0-only.txt") -Force
$BuildInfo = Join-Path $ProjectDir "scripts\write_build_info.py"
& $ProjectPython $BuildInfo --bundle $BundleDir --runtime-variant $RuntimeVariant
if ($LASTEXITCODE -ne 0) {
    throw "Writing BUILD-INFO.json failed with exit code $LASTEXITCODE."
}
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Compress-Archive -Path $BundleDir -DestinationPath $ZipPath -CompressionLevel Optimal
$Hash = (Get-FileHash $ZipPath -Algorithm SHA256).Hash

Write-Host "Windows bundle created at: $BundleDir\$BundleDisplayName.exe"
Write-Host "Shareable ZIP created at: $ZipPath"
Write-Host "SHA256: $Hash"
Write-Host "Runtime variant: $RuntimeVariant"
Write-Host "The bundled ProAimCLI.exe helper is used internally for live detector logs."
Write-Host "PyInstaller builds for the current OS; run this helper on Windows to create the .exe."
