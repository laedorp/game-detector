[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [string]$StopFile,

    [ValidateRange(250, 10000)]
    [int]$IntervalMilliseconds = 500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Convert-RequiredNumber {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $Parsed = 0.0
    if (-not [double]::TryParse(
        $Value.Trim(),
        [System.Globalization.NumberStyles]::Float,
        $Invariant,
        [ref]$Parsed
    )) {
        throw "nvidia-smi returned an invalid $Description value: '$Value'."
    }
    return $Parsed
}

function Write-JsonLine {
    param(
        [Parameter(Mandatory = $true)]$Writer,
        [Parameter(Mandatory = $true)]$Record
    )
    $Writer.WriteLine(($Record | ConvertTo-Json -Compress -Depth 8))
    $Writer.Flush()
}

$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$ResolvedStop = [System.IO.Path]::GetFullPath($StopFile)
$OutputParent = Split-Path -Parent $ResolvedOutput
if (-not (Test-Path -LiteralPath $OutputParent -PathType Container)) {
    throw "Telemetry output parent does not exist: $OutputParent"
}
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "Refusing to overwrite telemetry output: $ResolvedOutput"
}
if (Test-Path -LiteralPath $ResolvedStop) {
    throw "Telemetry stop file must not already exist: $ResolvedStop"
}
if ($null -eq (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi.exe is required for physical CUDA qualification."
}

$Writer = New-Object System.IO.StreamWriter($ResolvedOutput, $false, $Utf8NoBom)
try {
    do {
        $CapturedAt = [DateTime]::UtcNow.ToString("o")
        $GpuLines = @(& nvidia-smi.exe `
            --query-gpu=timestamp,index,name,uuid,driver_version,compute_cap,utilization.gpu,memory.used,memory.total `
            --format=csv,noheader,nounits)
        if ($LASTEXITCODE -ne 0 -or $GpuLines.Count -lt 1) {
            throw "nvidia-smi GPU telemetry query failed with exit code $LASTEXITCODE."
        }
        $GpuRows = @($GpuLines | ConvertFrom-Csv -Header @(
            "nvidia_timestamp",
            "gpu_index",
            "gpu_name",
            "gpu_uuid",
            "driver_version",
            "compute_capability",
            "utilization_gpu_percent",
            "memory_used_mib",
            "memory_total_mib"
        ))
        foreach ($Row in $GpuRows) {
            Write-JsonLine -Writer $Writer -Record ([ordered]@{
                schema_version = 1
                kind = "gpu"
                captured_at_utc = $CapturedAt
                nvidia_timestamp = ([string]$Row.nvidia_timestamp).Trim()
                gpu_index = [int](Convert-RequiredNumber -Value ([string]$Row.gpu_index) -Description "GPU index")
                gpu_name = ([string]$Row.gpu_name).Trim()
                gpu_uuid = ([string]$Row.gpu_uuid).Trim()
                driver_version = ([string]$Row.driver_version).Trim()
                compute_capability = ([string]$Row.compute_capability).Trim()
                utilization_gpu_percent = Convert-RequiredNumber -Value ([string]$Row.utilization_gpu_percent) -Description "GPU utilization"
                memory_used_mib = Convert-RequiredNumber -Value ([string]$Row.memory_used_mib) -Description "used memory"
                memory_total_mib = Convert-RequiredNumber -Value ([string]$Row.memory_total_mib) -Description "total memory"
            })
        }

        $ProcessLines = @(& nvidia-smi.exe `
            --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory `
            --format=csv,noheader,nounits)
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi compute-process query failed with exit code $LASTEXITCODE."
        }
        if ($ProcessLines.Count -gt 0) {
            $ProcessRows = @($ProcessLines | ConvertFrom-Csv -Header @(
                "gpu_uuid",
                "pid",
                "process_name",
                "used_gpu_memory_mib"
            ))
            foreach ($Row in $ProcessRows) {
                $ProcessMemoryText = ([string]$Row.used_gpu_memory_mib).Trim()
                $ProcessMemorySupported = $ProcessMemoryText -notin @("", "N/A", "[N/A]", "Not Supported")
                $ProcessMemory = if ($ProcessMemorySupported) {
                    Convert-RequiredNumber -Value $ProcessMemoryText -Description "process GPU memory"
                } else { $null }
                Write-JsonLine -Writer $Writer -Record ([ordered]@{
                    schema_version = 1
                    kind = "compute_process"
                    captured_at_utc = $CapturedAt
                    gpu_uuid = ([string]$Row.gpu_uuid).Trim()
                    pid = [int](Convert-RequiredNumber -Value ([string]$Row.pid) -Description "process ID")
                    # A full process path can expose user/profile names.  The
                    # executable basename is sufficient for correlation.
                    process_name = [System.IO.Path]::GetFileName(([string]$Row.process_name).Trim())
                    used_gpu_memory_mib = $ProcessMemory
                    used_gpu_memory_supported = $ProcessMemorySupported
                })
            }
        }
        if (-not (Test-Path -LiteralPath $ResolvedStop -PathType Leaf)) {
            Start-Sleep -Milliseconds $IntervalMilliseconds
        }
    } while (-not (Test-Path -LiteralPath $ResolvedStop -PathType Leaf))
} finally {
    $Writer.Dispose()
}
