[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$QualifiedGpu,
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
        throw "$Title was cancelled or left blank. Physical qualification is incomplete."
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
        "ProAim physical CUDA qualification",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Warning,
        [System.Windows.Forms.MessageBoxDefaultButton]::Button2
    )
    if ($Result -ne [System.Windows.Forms.DialogResult]::Yes) {
        throw "A required physical observation was not confirmed."
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
    "The automated CUDA benchmark and both live passes have finished. Complete this form only if you personally watched ProAimCLI.exe use the named physical GPU in Task Manager during all three passes.",
    "Complete ProAim physical CUDA observation",
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Information
) | Out-Null

$ObserverName = Read-RequiredObservation -Title "Observer name" -Prompt "Enter your full name:"
if ($ObserverName -ne $ExpectedObserverName) {
    throw "Local observer name does not exactly match the workflow dispatch record."
}
$TaskManagerEngine = Read-RequiredObservation `
    -Title "Task Manager GPU/engine" `
    -Prompt "Enter the exact Task Manager GPU number and engine text observed for ProAimCLI.exe:"

Require-YesObservation -Question "Did you observe ProAimCLI.exe on '$QualifiedGpu' during the release-default CUDA benchmark?"
Require-YesObservation -Question "Did you observe ProAimCLI.exe on '$QualifiedGpu' during the live no-preview CUDA pass?"
Require-YesObservation -Question "Did you observe ProAimCLI.exe on '$QualifiedGpu' during the live preview-15 CUDA pass?"

$TypedConfirmation = Read-RequiredObservation `
    -Title "Exact physical confirmation" `
    -Prompt "Retype exactly:`r`n$ExpectedTypedConfirmation"
if ($TypedConfirmation -cne $ExpectedTypedConfirmation) {
    throw "Local physical confirmation text does not match exactly."
}

$Record = [ordered]@{
    schema_version = 1
    status = "completed_after_automated_gpu_runs"
    completed = $true
    tag = $Tag
    github_actor = $GitHubActor
    github_run_id = $GitHubRunId
    observer_name = $ObserverName
    observed_at_utc = [DateTime]::UtcNow.ToString("o")
    physical_gpu_name = $QualifiedGpu
    task_manager_gpu_engine = $TaskManagerEngine
    typed_confirmation = $TypedConfirmation
    observations = [ordered]@{
        release_default_benchmark = $true
        live_no_preview = $true
        live_preview_15 = $true
    }
    completion_method = "interactive Windows desktop form after automated GPU runs"
}
$Json = $Record | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($ResolvedOutput, $Json + [Environment]::NewLine, $Utf8NoBom)
