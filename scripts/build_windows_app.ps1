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

Push-Location $ProjectDir
try {
    & $ProjectPython -m PyInstaller --noconfirm --clean "packaging\game_detector.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Windows bundle created at: $ProjectDir\dist\GameDetector\GameDetector.exe"
Write-Host "The bundled GameDetectorCLI.exe helper is used internally for live detector logs."
Write-Host "PyInstaller builds for the current OS; run this helper on Windows to create the .exe."
