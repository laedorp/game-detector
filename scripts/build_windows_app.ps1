param(
    [ValidateSet("cuda", "directml")]
    [string]$RuntimeVariant = "cuda"
)

$ErrorActionPreference = "Stop"

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

& $ProjectPython -c "import tkinter; tkinter.Tcl()"
if ($LASTEXITCODE -ne 0) {
    throw "Tkinter is missing from this Python installation. Install a standard python.org build with Tcl/Tk support."
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
