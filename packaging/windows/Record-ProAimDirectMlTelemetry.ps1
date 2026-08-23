[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [string]$StopFile,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 2147483647)]
    [int]$AdapterIndex,

    [Parameter(Mandatory = $true)]
    [ValidateSet("amd_rx_6950_xt", "nvidia_rtx_5060_laptop")]
    [string]$GpuRole,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedProductName,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedExecutablePath,

    [ValidateRange(250, 10000)]
    [int]$IntervalMilliseconds = 500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Normalize-ProductName {
    param([Parameter(Mandatory = $true)][string]$Value)
    return (($Value.Trim() -replace '\s+', ' ').ToUpperInvariant())
}

function Convert-RequiredCounterNumber {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Description
    )
    try { $Number = [double]$Value } catch { throw "$Description must be numeric." }
    if ([double]::IsNaN($Number) -or [double]::IsInfinity($Number) -or $Number -lt 0.0) {
        throw "$Description must be a finite non-negative number."
    }
    return $Number
}

function Convert-AdapterLuid {
    param([Parameter(Mandatory = $true)]$Value)
    try { $Unsigned = [uint64]$Value } catch { throw "DirectX registry AdapterLuid is invalid." }
    $Low = [uint32]($Unsigned -band [uint64]0xffffffff)
    $High = [uint32](($Unsigned -shr 32) -band [uint64]0xffffffff)
    return "0x{0:x8}_0x{1:x8}" -f $High, $Low
}

function Write-JsonLine {
    param(
        [Parameter(Mandatory = $true)]$Writer,
        [Parameter(Mandatory = $true)]$Record
    )
    $Writer.WriteLine(($Record | ConvertTo-Json -Compress -Depth 10))
    $Writer.Flush()
}

$ExpectedNames = @{
    amd_rx_6950_xt = "AMD Radeon RX 6950 XT"
    nvidia_rtx_5060_laptop = "NVIDIA GeForce RTX 5060 Laptop GPU"
}
$ExpectedVendors = @{
    amd_rx_6950_xt = "0x1002"
    nvidia_rtx_5060_laptop = "0x10de"
}
$CanonicalExpected = [string]$ExpectedNames[$GpuRole]
if ((Normalize-ProductName -Value $ExpectedProductName) -cne (Normalize-ProductName -Value $CanonicalExpected)) {
    throw "ExpectedProductName must equal the documented product for $GpuRole: $CanonicalExpected"
}

$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$ResolvedStop = [System.IO.Path]::GetFullPath($StopFile)
$ResolvedExecutable = [System.IO.Path]::GetFullPath($ExpectedExecutablePath)
if (-not (Test-Path -LiteralPath $ResolvedExecutable -PathType Leaf)) {
    throw "Expected ProAimCLI executable does not exist: $ResolvedExecutable"
}
$OutputParent = Split-Path -Parent $ResolvedOutput
if (-not (Test-Path -LiteralPath $OutputParent -PathType Container)) {
    throw "Telemetry output parent does not exist: $OutputParent"
}
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "Refusing to overwrite DirectML telemetry output: $ResolvedOutput"
}
if (Test-Path -LiteralPath $ResolvedStop) {
    throw "Telemetry stop file must not already exist: $ResolvedStop"
}

# Windows exposes the same adapter LUID in the DirectX registry inventory and
# in every GPU Engine performance-counter instance.  Product-name matching is
# intentionally exact after only whitespace/case normalization.  Multiple
# matching adapters are ambiguous and therefore fail closed.
$DirectXRoot = "HKLM:\SOFTWARE\Microsoft\DirectX"
if (-not (Test-Path -LiteralPath $DirectXRoot -PathType Container)) {
    throw "Windows DirectX adapter inventory is unavailable."
}
$Inventory = @()
foreach ($Key in @(Get-ChildItem -LiteralPath $DirectXRoot -ErrorAction Stop)) {
    $Record = Get-ItemProperty -LiteralPath $Key.PSPath -ErrorAction SilentlyContinue
    if ($null -eq $Record -or [string]::IsNullOrWhiteSpace([string]$Record.Description) -or $null -eq $Record.AdapterLuid) {
        continue
    }
    $VendorId = if ($null -ne $Record.VendorId) { "0x{0:x4}" -f [uint32]$Record.VendorId } else { "" }
    $DeviceId = if ($null -ne $Record.DeviceId) { "0x{0:x4}" -f [uint32]$Record.DeviceId } else { "" }
    $Inventory += [ordered]@{
        description = ([string]$Record.Description).Trim()
        normalized_description = Normalize-ProductName -Value ([string]$Record.Description)
        adapter_luid = Convert-AdapterLuid -Value $Record.AdapterLuid
        vendor_id = $VendorId.ToLowerInvariant()
        device_id = $DeviceId.ToLowerInvariant()
        driver_version = ([string]$Record.DriverVersion).Trim()
    }
}
$Matches = @($Inventory | Where-Object {
    $_.normalized_description -ceq (Normalize-ProductName -Value $CanonicalExpected)
})
if ($Matches.Count -ne 1) {
    throw "DirectX inventory must contain exactly one '$CanonicalExpected' adapter; found $($Matches.Count)."
}
$Selected = $Matches[0]
if ([string]$Selected.vendor_id -cne [string]$ExpectedVendors[$GpuRole]) {
    throw "The selected adapter has vendor ID '$($Selected.vendor_id)', expected '$($ExpectedVendors[$GpuRole])'."
}
$ExpectedLuid = [string]$Selected.adapter_luid
$CounterPath = '\GPU Engine(*)\Utilization Percentage'
$null = Get-Counter -Counter $CounterPath -MaxSamples 1 -ErrorAction Stop

$Writer = New-Object System.IO.StreamWriter($ResolvedOutput, $false, $Utf8NoBom)
try {
    Write-JsonLine -Writer $Writer -Record ([ordered]@{
        schema_version = 1
        kind = "adapter_inventory"
        captured_at_utc = [DateTime]::UtcNow.ToString("o")
        gpu_role = $GpuRole
        directml_adapter_index = $AdapterIndex
        product_name = $CanonicalExpected
        normalized_product_name = Normalize-ProductName -Value $CanonicalExpected
        adapter_luid = $ExpectedLuid
        vendor_id = [string]$Selected.vendor_id
        device_id = [string]$Selected.device_id
        driver_version = [string]$Selected.driver_version
        exact_product_match_count = $Matches.Count
        telemetry_interval_milliseconds = $IntervalMilliseconds
    })
    do {
        $CapturedAt = [DateTime]::UtcNow.ToString("o")
        $Processes = @(
            Get-Process -Name "ProAimCLI" -ErrorAction SilentlyContinue |
                Where-Object {
                    try {
                        [string]::Equals(
                            [System.IO.Path]::GetFullPath($_.Path),
                            $ResolvedExecutable,
                            [System.StringComparison]::OrdinalIgnoreCase
                        )
                    } catch { $false }
                }
        )
        $ProcessIds = @{}
        foreach ($Process in $Processes) { $ProcessIds[[int]$Process.Id] = $true }
        $Counter = Get-Counter -Counter $CounterPath -MaxSamples 1 -ErrorAction Stop
        foreach ($Sample in @($Counter.CounterSamples)) {
            $Instance = [string]$Sample.InstanceName
            $Match = [regex]::Match(
                $Instance,
                '^pid_(?<pid>[0-9]+)_luid_(?<luid>0x[0-9a-f]+_0x[0-9a-f]+)_phys_(?<physical>[0-9]+)_eng_(?<engine>[0-9]+)_engtype_(?<type>.+)$',
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
            )
            if (-not $Match.Success) { continue }
            $ProcessIdValue = [int]$Match.Groups['pid'].Value
            $LuidValue = $Match.Groups['luid'].Value.ToLowerInvariant()
            if ($LuidValue -cne $ExpectedLuid.ToLowerInvariant() -or -not $ProcessIds.ContainsKey($ProcessIdValue)) {
                continue
            }
            Write-JsonLine -Writer $Writer -Record ([ordered]@{
                schema_version = 1
                kind = "proaim_gpu_engine"
                captured_at_utc = $CapturedAt
                pid = $ProcessIdValue
                process_name = "ProAimCLI.exe"
                adapter_luid = $LuidValue
                physical_adapter = [int]$Match.Groups['physical'].Value
                engine_index = [int]$Match.Groups['engine'].Value
                engine_type = $Match.Groups['type'].Value
                utilization_percent = Convert-RequiredCounterNumber -Value $Sample.CookedValue -Description "GPU Engine utilization"
            })
        }
        if (-not (Test-Path -LiteralPath $ResolvedStop -PathType Leaf)) {
            Start-Sleep -Milliseconds $IntervalMilliseconds
        }
    } while (-not (Test-Path -LiteralPath $ResolvedStop -PathType Leaf))
} finally {
    $Writer.Dispose()
}
