[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)]
    [ValidateSet("amd_rx_6950_xt", "nvidia_rtx_5060_laptop")]
    [string]$GpuRole,
    [Parameter(Mandatory = $true)][string]$QualifiedGpu,
    [Parameter(Mandatory = $true)][ValidateRange(0, 2147483647)][int]$AdapterIndex,
    [Parameter(Mandatory = $true)][string]$ExpectedObserverName,
    [Parameter(Mandatory = $true)][string]$ExpectedTypedConfirmation,
    [Parameter(Mandatory = $true)][string]$GitHubActor,
    [Parameter(Mandatory = $true)][string]$GitHubRunId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

Add-Type -AssemblyName Microsoft.VisualBasic
Add-Type -AssemblyName System.Windows.Forms

function Read-RequiredObservation {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][string]$Title
    )
    $Value = [Microsoft.VisualBasic.Interaction]::InputBox($Prompt, $Title, "")
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Title was cancelled or left blank. DirectML qualification is incomplete."
    }
    if ($Value.Length -gt 240 -or $Value.IndexOfAny([char[]]"`r`n`0") -ge 0) {
        throw "$Title must be one line of at most 240 characters."
    }
    return $Value.Trim()
}

function Require-YesObservation {
    param([Parameter(Mandatory = $true)][string]$Question)
    $Result = [System.Windows.Forms.MessageBox]::Show(
        $Question,
        "ProAim physical DirectML qualification",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Warning,
        [System.Windows.Forms.MessageBoxDefaultButton]::Button2
    )
    if ($Result -ne [System.Windows.Forms.DialogResult]::Yes) {
        throw "A required physical DirectML observation was not confirmed."
    }
}

$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "Refusing to overwrite physical observation record: $ResolvedOutput"
}
$Parent = Split-Path -Parent $ResolvedOutput
if (-not (Test-Path -LiteralPath $Parent -PathType Container)) {
    throw "Physical observation output parent does not exist: $Parent"
}

[System.Windows.Forms.MessageBox]::Show(
    "The DirectML benchmark and both bounded live passes have finished. Complete this form only if you personally watched ProAimCLI.exe use '$QualifiedGpu' in Task Manager during all three passes and the GPU/engine column agreed with the automated adapter-LUID telemetry.",
    "Complete ProAim physical DirectML observation",
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Information
) | Out-Null

$ObserverName = Read-RequiredObservation -Title "Observer name" -Prompt "Enter your full name:"
if ($ObserverName -cne $ExpectedObserverName) {
    throw "Local observer name does not exactly match the workflow dispatch record."
}
$TaskManagerEngine = Read-RequiredObservation `
    -Title "Task Manager GPU/engine" `
    -Prompt "Enter the exact Task Manager GPU number and engine text observed for ProAimCLI.exe:"

Require-YesObservation -Question "Did you observe ProAimCLI.exe on '$QualifiedGpu' during the release-default DirectML benchmark?"
Require-YesObservation -Question "Did you observe ProAimCLI.exe on '$QualifiedGpu' during the live no-preview DirectML pass?"
Require-YesObservation -Question "Did you observe ProAimCLI.exe on '$QualifiedGpu' during the live preview-15 DirectML pass?"

$TypedConfirmation = Read-RequiredObservation `
    -Title "Exact physical confirmation" `
    -Prompt "Retype exactly:`r`n$ExpectedTypedConfirmation"
if ($TypedConfirmation -cne $ExpectedTypedConfirmation) {
    throw "Local physical confirmation text does not match exactly."
}

$Record = [ordered]@{
    schema_version = 1
    status = "completed_after_automated_directml_runs"
    completed = $true
    repository = $Repository
    tag = $Tag
    github_actor = $GitHubActor
    github_run_id = $GitHubRunId
    observer_name = $ObserverName
    observed_at_utc = [DateTime]::UtcNow.ToString("o")
    gpu_role = $GpuRole
    physical_gpu_name = $QualifiedGpu
    directml_adapter_index = $AdapterIndex
    task_manager_gpu_engine = $TaskManagerEngine
    typed_confirmation = $TypedConfirmation
    observations = [ordered]@{
        release_default_benchmark = $true
        live_no_preview = $true
        live_preview_15 = $true
        automated_luid_telemetry_agreed = $true
    }
    completion_method = "interactive Windows desktop form after automated DirectML runs"
}
$Json = $Record | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($ResolvedOutput, $Json + [Environment]::NewLine, $Utf8NoBom)
