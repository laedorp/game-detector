[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("DirectML", "CUDA")]
    [string]$Provider,

    [ValidateRange(-1, 2147483647)]
    [int]$AdapterIndex = -1,

    [string]$ExpectedAdapterName = "",

    [string]$EvidenceDirectory = "",

    [switch]$RunLive,

    [ValidateRange(0, 64)]
    [int]$ScreenMonitor = 1,

    [ValidateRange(1.0, 240.0)]
    [double]$ScreenFps = 60.0,

    [ValidateRange(2, 100000)]
    [int]$MaxFrames = 1000,

    [ValidateRange(1.0, 600.0)]
    [double]$MaxSeconds = 60.0,

    [ValidateRange(1, 512)]
    [int]$Samples = 32,

    [ValidateRange(0, 1000)]
    [int]$Warmup = 30,

    [ValidateRange(1, 10000)]
    [int]$Iterations = 100,

    [ValidateRange(1, 100)]
    [int]$Repeats = 3,

    [ValidateRange(-1, 16384)]
    [int]$DetailCropSize = -1,

    [string]$BundleArchivePath = "",

    [switch]$SkipReadyPrompt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# PowerShell 7 turns a non-zero native exit into a terminating error. Windows
# PowerShell 5.1 ignores this variable, so Invoke-FrozenCli also checks
# LASTEXITCODE explicitly. Both paths therefore fail closed.
$PSNativeCommandUseErrorActionPreference = $true
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$PolicyBenchmarkMinSamples = 32
$PolicyBenchmarkMinIterations = 100
$PolicyBenchmarkMinRepeats = 3
$PolicyBenchmarkMaxP95InferenceMs = 35.0
$PolicyLiveMinProcessedFrames = 120
$PolicyLiveMinElapsedFps = 20.0
$PolicyLiveMinUpdateFps = 20.0
$PolicyLiveMaxP95ObservedPipelineMs = 50.0
$PolicyLiveMaxP95FreshnessLatencyMs = 50.0
$PolicyLiveMaxSeconds = 60.0

function Get-FinitePolicyNumber {
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

function Assert-CloseTiming {
    param(
        [Parameter(Mandatory = $true)][double]$Actual,
        [Parameter(Mandatory = $true)][double]$Expected,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $Tolerance = [Math]::Max(0.1, [Math]::Abs($Expected) * 0.01)
    if ([Math]::Abs($Actual - $Expected) -gt $Tolerance) {
        throw "$Description is internally inconsistent."
    }
}

function Assert-TimingSummary {
    param(
        [Parameter(Mandatory = $true)]$Timing,
        [Parameter(Mandatory = $true)][int]$ExpectedSamples,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-StrictJsonInteger $Timing.samples) -or [int]$Timing.samples -ne $ExpectedSamples) {
        throw "$Description has the wrong sample count."
    }
    $Values = [ordered]@{}
    foreach ($MetricName in @("mean", "p50", "median", "p95", "p99", "min", "max", "stdev")) {
        $Values[$MetricName] = Get-FinitePolicyNumber -Value $Timing.$MetricName -Description "$Description $MetricName"
    }
    if (
        [double]$Values.min -gt [double]$Values.p50 -or
        [double]$Values.p50 -gt [double]$Values.p95 -or
        [double]$Values.p95 -gt [double]$Values.p99 -or
        [double]$Values.p99 -gt [double]$Values.max -or
        [double]$Values.mean -lt [double]$Values.min -or
        [double]$Values.mean -gt [double]$Values.max -or
        [Math]::Abs([double]$Values.p50 - [double]$Values.median) -gt 0.000000001
    ) {
        throw "$Description is internally inconsistent."
    }
    return $Values
}

function Test-StrictJsonInteger {
    param($Value)
    if ($null -eq $Value -or $Value -is [bool]) {
        return $false
    }
    return (
        $Value -is [sbyte] -or $Value -is [byte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    )
}

function Normalize-AdapterName {
    param([Parameter(Mandatory = $true)][string]$Value)
    return (($Value.Trim() -replace '\s+', ' ').ToUpperInvariant())
}

function Write-Utf8File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Content
    )
    [System.IO.File]::WriteAllText($Path, $Content, $Utf8NoBom)
}

function Resolve-ExistingFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "$Description path must not be empty."
    }
    $Item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($Item.PSIsContainer) {
        throw "$Description must be a file: $($Item.FullName)"
    }
    return $Item.FullName
}

function Resolve-BundleContractFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string]$BundleRoot
    )
    if (
        [string]::IsNullOrWhiteSpace($RelativePath) -or
        $RelativePath.Contains("\") -or
        -not $RelativePath.StartsWith("_internal/", [System.StringComparison]::Ordinal) -or
        [System.IO.Path]::IsPathRooted($RelativePath) -or
        @($RelativePath.Split('/') | Where-Object { $_ -in @("", ".", "..") }).Count -ne 0
    ) {
        throw "$Description path is not canonical bundle-relative POSIX: $RelativePath"
    }
    $Root = [System.IO.Path]::GetFullPath($BundleRoot).TrimEnd('\', '/')
    $Candidate = [System.IO.Path]::GetFullPath(
        (Join-Path $Root ($RelativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)))
    )
    $Prefix = $Root + [System.IO.Path]::DirectorySeparatorChar
    if (-not $Candidate.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description escaped the bundle root."
    }
    return Resolve-ExistingFile -Path $Candidate -Description $Description
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-ArtifactRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BundleRoot
    )
    $Item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $FullPath = [System.IO.Path]::GetFullPath($Item.FullName)
    $RootPrefix = [System.IO.Path]::GetFullPath($BundleRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $InsideBundle = $FullPath.StartsWith(
        $RootPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    $RecordedPath = $Item.Name
    if ($InsideBundle) {
        $RecordedPath = $FullPath.Substring($RootPrefix.Length).Replace('\', '/')
    }
    return [ordered]@{
        role = $Role
        path = $RecordedPath
        location = if ($InsideBundle) { "bundle" } else { "external" }
        size_bytes = [int64]$Item.Length
        sha256 = Get-Sha256 -Path $FullPath
    }
}

function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    try {
        return (Get-Content -LiteralPath $Path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop)
    } catch {
        throw "$Description did not contain valid JSON: $($_.Exception.Message)"
    }
}

function Get-ErrorTail {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ""
    }
    return ((Get-Content -LiteralPath $Path -Tail 20 -ErrorAction SilentlyContinue) -join [Environment]::NewLine)
}

function Invoke-FrozenCli {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )
    Write-Host "Running $Name..."
    $StartedAtUtc = [DateTime]::UtcNow.ToString("o")
    $OutputLines = @()
    $NativeFailure = $null
    $LASTEXITCODE = 0
    try {
        # Argument splatting keeps each user-controlled path as one native
        # argument. Do not replace this with dynamic command evaluation: that
        # would turn a model path into executable syntax.
        $OutputLines = @(& $CliPath @Arguments 2> $StderrPath)
    } catch {
        $NativeFailure = $_
    }
    [System.IO.File]::WriteAllLines(
        $StdoutPath,
        [string[]]$OutputLines,
        $Utf8NoBom
    )
    if (-not (Test-Path -LiteralPath $StderrPath -PathType Leaf)) {
        Write-Utf8File -Path $StderrPath -Content ""
    }
    $ExitCode = [int]$LASTEXITCODE
    $CompletedAtUtc = [DateTime]::UtcNow.ToString("o")
    if ($null -ne $NativeFailure -or $ExitCode -ne 0) {
        $Details = Get-ErrorTail -Path $StderrPath
        if ([string]::IsNullOrWhiteSpace($Details) -and $null -ne $NativeFailure) {
            $Details = $NativeFailure.Exception.Message
        }
        throw "$Name failed with exit code $ExitCode. $Details"
    }
    if (-not (Test-Path -LiteralPath $StdoutPath -PathType Leaf)) {
        throw "$Name did not create its stdout record."
    }
    return [ordered]@{
        name = $Name
        arguments = @($Arguments)
        exit_code = $ExitCode
        started_at_utc = $StartedAtUtc
        completed_at_utc = $CompletedAtUtc
        stdout = [ordered]@{
            file = [System.IO.Path]::GetFileName($StdoutPath)
            sha256 = Get-Sha256 -Path $StdoutPath
        }
        stderr = [ordered]@{
            file = [System.IO.Path]::GetFileName($StderrPath)
            sha256 = Get-Sha256 -Path $StderrPath
        }
    }
}

function Assert-ProviderSummary {
    param(
        [Parameter(Mandatory = $true)]$Summary,
        [Parameter(Mandatory = $true)][string]$Context
    )
    if ([string]$Summary.requested_provider -ne $ExpectedProvider) {
        throw "$Context requested provider '$($Summary.requested_provider)', expected '$ExpectedProvider'."
    }
    $ActiveProviders = @($Summary.active_providers | ForEach-Object { [string]$_ })
    if (-not ($ActiveProviders -contains $ExpectedProvider)) {
        throw "$Context did not activate $ExpectedProvider. Active providers: $($ActiveProviders -join ', ')."
    }
    if (-not [bool]$Summary.require_full_provider) {
        throw "$Context did not record require_full_provider=true."
    }
    if (-not [bool]$Summary.configured_session_options.disable_cpu_ep_fallback) {
        throw "$Context did not record configured_session_options.disable_cpu_ep_fallback=true."
    }
    if (-not [bool]$Summary.runtime_ep_fail_fallback_disabled) {
        throw "$Context did not disable ONNX Runtime EPFail fallback."
    }
    if ([string]$Summary.provider_options_status -ne "ok") {
        throw "$Context could not report CUDA provider options."
    }
    if ($Provider -eq "CUDA" -and [string]$Summary.provider_options.CUDAExecutionProvider.device_id -ne "0") {
        throw "$Context did not bind CUDA device_id=0 on the dedicated single-GPU runner."
    }
    if ($Provider -eq "DirectML") {
        $ConfiguredIndex = [string]$Summary.provider_option_overrides.DmlExecutionProvider.device_id
        if ($ConfiguredIndex -ne [string]$AdapterIndex) {
            throw "$Context bound DirectML device_id '$ConfiguredIndex', expected '$AdapterIndex'."
        }
    }
}

function Invoke-ModelBenchmark {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string]$LabelsPath,
        [Parameter(Mandatory = $true)][string]$InferenceSize,
        [Parameter(Mandatory = $true)][string]$ExpectedModelHash,
        [Parameter(Mandatory = $true)][string]$ExpectedLabelsHash,
        [Parameter(Mandatory = $true)][int[]]$ExpectedShape
    )
    $StdoutPath = Join-Path $StageDirectory ("benchmark-{0}.json" -f $Key)
    $StderrPath = Join-Path $StageDirectory ("benchmark-{0}.stderr.txt" -f $Key)
    $Arguments = @(
        "--benchmark-models",
        "--backend", "onnxruntime",
        "--model", $ModelPath,
        "--name", $Key,
        "--precision", "FP32",
        "--labels", $LabelsPath,
        "--inference-size", $InferenceSize,
        "--device", $Device,
        "--synthetic",
        "--require-full-provider",
        "--samples", [string]$Samples,
        "--warmup", [string]$Warmup,
        "--iterations", [string]$Iterations,
        "--repeats", [string]$Repeats
    )
    $RunRecord = Invoke-FrozenCli -Name "model benchmark ($Key)" -Arguments $Arguments -StdoutPath $StdoutPath -StderrPath $StderrPath
    $Report = Read-JsonFile -Path $StdoutPath -Description "Model benchmark ($Key)"
    if ($Report.PSObject.Properties.Name -contains "error") {
        throw "Model benchmark ($Key) reported an error: $($Report.error)"
    }
    if (-not [bool]$Report.methodology.require_full_provider) {
        throw "Model benchmark ($Key) did not use the full-provider gate."
    }
    if ([string]$Report.methodology.backend -ne "onnxruntime" -or [string]$Report.methodology.requested_device -ne $Device) {
        throw "Model benchmark ($Key) recorded the wrong requested device."
    }
    $Models = @($Report.models)
    if ($Models.Count -ne 1) {
        throw "Model benchmark ($Key) must report exactly one model."
    }
    Assert-ProviderSummary -Summary $Models[0].runtime -Context "Model benchmark ($Key)"
    if ($Samples -ne $PolicyBenchmarkMinSamples -or $Warmup -ne 30 -or $Iterations -ne $PolicyBenchmarkMinIterations -or $Repeats -ne $PolicyBenchmarkMinRepeats) {
        throw "Model benchmark ($Key) dimensions differ from repository qualification policy."
    }
    if ([int]$Report.methodology.warmup_per_model -ne $Warmup -or [int]$Report.methodology.iterations_per_repeat -ne $Iterations -or [int]$Report.methodology.repeats -ne $Repeats) {
        throw "Model benchmark ($Key) reported methodology that differs from the requested bounds."
    }
    if (
        [string]$Report.input.kind -ne "synthetic" -or
        [string]$Report.input.generator -ne "numpy.default_rng(seed=0), uint8 720x1280 BGR" -or
        [int]$Report.input.count -ne $Samples
    ) {
        throw "Model benchmark ($Key) reported a different synthetic input workload."
    }
    $BenchmarkGenerated = [DateTimeOffset]::Parse([string]$Report.generated_at_utc).UtcDateTime
    $BenchmarkStarted = [DateTimeOffset]::Parse([string]$RunRecord.started_at_utc).UtcDateTime
    $BenchmarkCompleted = [DateTimeOffset]::Parse([string]$RunRecord.completed_at_utc).UtcDateTime
    if ($BenchmarkGenerated -lt $BenchmarkStarted -or $BenchmarkGenerated -gt $BenchmarkCompleted) {
        throw "Model benchmark ($Key) generation time is outside its frozen CLI run."
    }
    $ExpectedTimedSamples = $Iterations * $Repeats
    $AggregateSummaries = @{}
    foreach ($TimingName in @("preprocess", "inference", "postprocess", "pipeline")) {
        $AggregateSummaries[$TimingName] = Assert-TimingSummary `
            -Timing $Models[0].timing_ms.$TimingName `
            -ExpectedSamples $ExpectedTimedSamples `
            -Description "Model benchmark ($Key) $TimingName timing"
    }
    $ExpectedPipelineMean =
        [double]$Models[0].timing_ms.preprocess.mean +
        [double]$Models[0].timing_ms.inference.mean +
        [double]$Models[0].timing_ms.postprocess.mean
    Assert-CloseTiming `
        -Actual ([double]$Models[0].timing_ms.pipeline.mean) `
        -Expected $ExpectedPipelineMean `
        -Description "Model benchmark ($Key) mean pipeline/component timing"
    $RepeatRecords = @($Models[0].repeats)
    if ($RepeatRecords.Count -ne $Repeats) {
        throw "Model benchmark ($Key) omitted exact repeat-level timing evidence."
    }
    $RepeatSummaries = @()
    for ($RepeatIndex = 0; $RepeatIndex -lt $RepeatRecords.Count; $RepeatIndex++) {
        $RepeatNumber = $RepeatIndex + 1
        $RepeatRecord = $RepeatRecords[$RepeatIndex]
        if (-not (Test-StrictJsonInteger $RepeatRecord.repeat) -or [int]$RepeatRecord.repeat -ne $RepeatNumber) {
            throw "Model benchmark ($Key) repeat identities are incomplete or out of order."
        }
        $Current = @{}
        foreach ($TimingName in @("preprocess", "inference", "postprocess", "pipeline")) {
            $Current[$TimingName] = Assert-TimingSummary `
                -Timing $RepeatRecord.timing_ms.$TimingName `
                -ExpectedSamples $Iterations `
                -Description "Model benchmark ($Key) repeat $RepeatNumber $TimingName timing"
        }
        $RepeatPipelineMean =
            [double]$Current.preprocess.mean +
            [double]$Current.inference.mean +
            [double]$Current.postprocess.mean
        Assert-CloseTiming -Actual ([double]$Current.pipeline.mean) -Expected $RepeatPipelineMean -Description "Model benchmark ($Key) repeat $RepeatNumber mean pipeline/component timing"
        if ([double]$Current.inference.p95 -gt $PolicyBenchmarkMaxP95InferenceMs) {
            throw "Model benchmark ($Key) repeat $RepeatNumber p95 inference latency exceeds release policy."
        }
        $RepeatSummaries += ,$Current
    }
    foreach ($TimingName in @("preprocess", "inference", "postprocess", "pipeline")) {
        $RepeatMean = [double](($RepeatSummaries | ForEach-Object { [double]$_[$TimingName].mean } | Measure-Object -Average).Average)
        Assert-CloseTiming -Actual ([double]$AggregateSummaries[$TimingName].mean) -Expected $RepeatMean -Description "Model benchmark ($Key) aggregate/repeat $TimingName mean"
        $RepeatMinimum = [double](($RepeatSummaries | ForEach-Object { [double]$_[$TimingName].min } | Measure-Object -Minimum).Minimum)
        $RepeatMaximum = [double](($RepeatSummaries | ForEach-Object { [double]$_[$TimingName].max } | Measure-Object -Maximum).Maximum)
        if ([double]$AggregateSummaries[$TimingName].min -ne $RepeatMinimum -or [double]$AggregateSummaries[$TimingName].max -ne $RepeatMaximum) {
            throw "Model benchmark ($Key) aggregate/repeat $TimingName extrema differ."
        }
    }
    $PipelineMean = [double]$AggregateSummaries.pipeline.mean
    $PipelineFps = Get-FinitePolicyNumber -Value $Models[0].pipeline_fps_from_mean -Description "Model benchmark ($Key) pipeline FPS"
    Assert-CloseTiming -Actual $PipelineFps -Expected (1000.0 / $PipelineMean) -Description "Model benchmark ($Key) pipeline FPS"
    if ((Get-FinitePolicyNumber -Value $Models[0].timing_ms.inference.p95 -Description "Model benchmark ($Key) p95 inference") -gt $PolicyBenchmarkMaxP95InferenceMs) {
        throw "Model benchmark ($Key) p95 inference latency exceeds release policy."
    }
    if (@($Models[0].input_shape_hw).Count -ne 2 -or [int]$Models[0].input_shape_hw[0] -ne $ExpectedShape[0] -or [int]$Models[0].input_shape_hw[1] -ne $ExpectedShape[1]) {
        throw "Model benchmark ($Key) reported the wrong input shape."
    }
    $Files = @($Models[0].artifact.files)
    if (
        $Files.Count -ne 1 -or
        [string]$Files[0].sha256 -ne $ExpectedModelHash -or
        -not [string]::Equals(
            [System.IO.Path]::GetFullPath([string]$Files[0].resolved_path),
            [System.IO.Path]::GetFullPath($ModelPath),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Model benchmark ($Key) did not fingerprint the exact selected ONNX artifact."
    }
    $LabelFiles = @($Models[0].labels_artifact.files)
    if (
        $LabelFiles.Count -ne 1 -or
        [string]$LabelFiles[0].sha256 -ne $ExpectedLabelsHash -or
        -not [string]::Equals(
            [System.IO.Path]::GetFullPath([string]$LabelFiles[0].resolved_path),
            [System.IO.Path]::GetFullPath($LabelsPath),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Model benchmark ($Key) did not fingerprint the exact labels artifact."
    }
    return $RunRecord
}

function Invoke-LivePass {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][bool]$PreviewEnabled,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string]$LabelsPath,
        [Parameter(Mandatory = $true)][string]$InferenceSize,
        [Parameter(Mandatory = $true)][string]$ExpectedModelHash,
        [Parameter(Mandatory = $true)][string]$ExpectedLabelsHash,
        [Parameter(Mandatory = $true)][int[]]$ExpectedShape
    )
    $MetricsPath = Join-Path $StageDirectory ("live-{0}.json" -f $Key)
    $StdoutPath = Join-Path $StageDirectory ("live-{0}.stdout.txt" -f $Key)
    $StderrPath = Join-Path $StageDirectory ("live-{0}.stderr.txt" -f $Key)
    $ScreenFpsText = $ScreenFps.ToString("0.################", [System.Globalization.CultureInfo]::InvariantCulture)
    $MaxSecondsText = $MaxSeconds.ToString("0.################", [System.Globalization.CultureInfo]::InvariantCulture)
    $Arguments = @(
        "--cli",
        "--source", "screen",
        "--screen-monitor", [string]$ScreenMonitor,
        "--screen-fps", $ScreenFpsText,
        "--backend", "onnxruntime",
        "--device", $Device,
        "--require-full-provider",
        "--model", $ModelPath,
        "--labels", $LabelsPath,
        "--inference-size", $InferenceSize,
        "--stats-window", [string]$MaxFrames,
        "--max-frames", [string]$MaxFrames,
        "--max-seconds", $MaxSecondsText,
        "--metrics-json", $MetricsPath
    )
    if ($PreviewEnabled) {
        $Arguments += @("--preview-fps", "15")
    } else {
        $Arguments += "--no-preview"
    }
    if ($DetailCropSize -gt 0) {
        $Arguments += @("--detail-crop-size", [string]$DetailCropSize)
    }
    $RunRecord = Invoke-FrozenCli -Name "live pipeline ($Key)" -Arguments $Arguments -StdoutPath $StdoutPath -StderrPath $StderrPath
    $Report = Read-JsonFile -Path $MetricsPath -Description "Live pipeline ($Key) report"
    Assert-ProviderSummary -Summary $Report.detector_runtime -Context "Live pipeline ($Key)"
    if (-not [bool]$Report.config.require_full_provider) {
        throw "Live pipeline ($Key) did not record the full-provider gate."
    }
    if (
        [string]$Report.config.backend -ne "onnxruntime" -or
        [string]$Report.config.device -ne $Device -or
        [string]$Report.config.source.kind -ne "screen" -or
        $null -ne $Report.config.source.value -or
        $null -ne $Report.config.capture.screen_region -or
        [int]$Report.config.capture.screen_monitor -ne $ScreenMonitor -or
        [double]$Report.config.capture.screen_fps -ne $ScreenFps -or
        $null -ne $Report.config.inference.crop_size
    ) {
        throw "Live pipeline ($Key) recorded a different full-screen workload."
    }
    $ExpectedDetailCrop = if ($DetailCropSize -gt 0) { $DetailCropSize } else { $null }
    if ($Report.config.inference.detail_crop_size -ne $ExpectedDetailCrop) {
        throw "Live pipeline ($Key) recorded a different detail workload."
    }
    if ([string]$Report.model_artifact.sha256 -ne $ExpectedModelHash) {
        throw "Live pipeline ($Key) did not use the exact selected ONNX artifact."
    }
    if ([string]$Report.labels_artifact.sha256 -ne $ExpectedLabelsHash) {
        throw "Live pipeline ($Key) did not use the exact selected labels artifact."
    }
    if (@($Report.config.inference.shape_hw).Count -ne 2 -or [int]$Report.config.inference.shape_hw[0] -ne $ExpectedShape[0] -or [int]$Report.config.inference.shape_hw[1] -ne $ExpectedShape[1]) {
        throw "Live pipeline ($Key) recorded the wrong input shape."
    }
    if (
        [bool]$Report.preview.enabled -ne $PreviewEnabled -or
        [bool]$Report.config.preview.enabled -ne $PreviewEnabled
    ) {
        throw "Live pipeline ($Key) recorded the wrong preview state."
    }
    $ProcessedFrames = [int64]$Report.pipeline.processed_frames
    if ($PreviewEnabled) {
        if ([double]$Report.preview.fps_limit -ne 15.0 -or [double]$Report.config.preview.fps_limit -ne 15.0 -or [string]$Report.preview.mode -eq "disabled") {
            throw "Live pipeline ($Key) did not run preview at exactly 15 FPS."
        }
        $PreviewStats = $Report.preview.stats
        if (
            @($PreviewStats.PSObject.Properties).Count -ne 3 -or
            -not (Test-StrictJsonInteger $PreviewStats.submitted_frames) -or
            -not (Test-StrictJsonInteger $PreviewStats.displayed_frames) -or
            -not (Test-StrictJsonInteger $PreviewStats.replaced_frames) -or
            [int64]$PreviewStats.displayed_frames -le 0 -or
            [int64]$PreviewStats.displayed_frames -gt [int64]$PreviewStats.submitted_frames -or
            [int64]$PreviewStats.submitted_frames -gt $ProcessedFrames -or
            [int64]$PreviewStats.replaced_frames -lt 0
        ) {
            throw "Live pipeline ($Key) recorded impossible preview activity."
        }
    } elseif ([string]$Report.preview.mode -ne "disabled" -or @($Report.preview.stats.PSObject.Properties).Count -ne 0) {
        throw "Live pipeline ($Key) recorded unexpected preview activity."
    }
    $ReportStarted = [DateTimeOffset]::Parse([string]$Report.started_utc).UtcDateTime
    $ReportCompleted = [DateTimeOffset]::Parse([string]$Report.completed_utc).UtcDateTime
    $RunStarted = [DateTimeOffset]::Parse([string]$RunRecord.started_at_utc).UtcDateTime
    $RunCompleted = [DateTimeOffset]::Parse([string]$RunRecord.completed_at_utc).UtcDateTime
    if ($ReportCompleted -lt $ReportStarted -or $ReportStarted -lt $RunStarted -or $ReportCompleted -gt $RunCompleted) {
        throw "Live pipeline ($Key) report timestamps are outside its frozen CLI run."
    }
    $ExpectedDetailEnabled = $DetailCropSize -gt 0
    if ([bool]$Report.detail_pass.enabled -ne $ExpectedDetailEnabled) {
        throw "Live pipeline ($Key) recorded the wrong detail-pass state."
    }
    if ($ExpectedDetailEnabled -and [int]$Report.detail_pass.requested_crop_size -ne $DetailCropSize) {
        throw "Live pipeline ($Key) recorded the wrong detail crop size."
    }
    if ($ExpectedDetailEnabled) {
        $Plan = $Report.detail_pass.last_plan
        if (
            [string]$Report.detail_pass.crop_policy -ne "centered_model_aspect_roi" -or
            $null -eq $Plan -or
            [string]$Plan.crop_policy -ne "centered_model_aspect_roi" -or
            [int]$Plan.requested_crop_size -ne $DetailCropSize -or
            [int]$Plan.model_height -ne $ExpectedShape[0] -or
            [int]$Plan.model_width -ne $ExpectedShape[1]
        ) {
            throw "Live pipeline ($Key) omitted its exact model-aspect detail ROI."
        }
        $SourceWidth = [int]$Plan.source_width
        $SourceHeight = [int]$Plan.source_height
        $ModelWidth = [int]$Plan.model_width
        $ModelHeight = [int]$Plan.model_height
        if ($SourceWidth -le 0 -or $SourceHeight -le 0) {
            throw "Live pipeline ($Key) detail ROI has invalid source geometry."
        }
        $ExpectedRequestedHeight = [Math]::Max(1, [int][Math]::Round($DetailCropSize * $ModelHeight / [double]$ModelWidth))
        function Get-GreatestCommonDivisor {
            param([int]$First, [int]$Second)
            while ($Second -ne 0) {
                $Remainder = $First % $Second
                $First = $Second
                $Second = $Remainder
            }
            return [Math]::Abs($First)
        }
        $Common = Get-GreatestCommonDivisor -First $ModelWidth -Second $ModelHeight
        $AspectWidth = [int]($ModelWidth / $Common)
        $AspectHeight = [int]($ModelHeight / $Common)
        $AspectUnits = [Math]::Min(
            [int][Math]::Floor($DetailCropSize / [double]$AspectWidth),
            [Math]::Min(
                [int][Math]::Floor($SourceWidth / [double]$AspectWidth),
                [int][Math]::Floor($SourceHeight / [double]$AspectHeight)
            )
        )
        if ($AspectUnits -gt 0) {
            $ExpectedWidth = $AspectUnits * $AspectWidth
            $ExpectedHeight = $AspectUnits * $AspectHeight
        } else {
            $ExpectedWidth = [Math]::Min($DetailCropSize, $SourceWidth)
            $ExpectedHeight = [Math]::Min($SourceHeight, [Math]::Max(1, [int][Math]::Round($ExpectedWidth * $ModelHeight / [double]$ModelWidth)))
        }
        $ExpectedCropX = [int][Math]::Floor(($SourceWidth - $ExpectedWidth) / 2.0)
        $ExpectedCropY = [int][Math]::Floor(($SourceHeight - $ExpectedHeight) / 2.0)
        if (
            [int]$Plan.requested_crop_height -ne $ExpectedRequestedHeight -or
            [int]$Plan.applied_crop_width -ne $ExpectedWidth -or
            [int]$Plan.applied_crop_height -ne $ExpectedHeight -or
            [int]$Plan.crop_x -ne $ExpectedCropX -or
            [int]$Plan.crop_y -ne $ExpectedCropY
        ) {
            throw "Live pipeline ($Key) detail ROI width/height/aspect geometry is inconsistent."
        }
        $ExpectedFullScale = [Math]::Min($ModelWidth / [double]$SourceWidth, $ModelHeight / [double]$SourceHeight)
        $ExpectedDetailScale = [Math]::Min($ModelWidth / [double]$ExpectedWidth, $ModelHeight / [double]$ExpectedHeight)
        Assert-CloseTiming -Actual ([double]$Plan.full_frame_scale) -Expected $ExpectedFullScale -Description "Live pipeline ($Key) detail full-frame scale"
        Assert-CloseTiming -Actual ([double]$Plan.detail_scale) -Expected $ExpectedDetailScale -Description "Live pipeline ($Key) detail ROI scale"
        Assert-CloseTiming -Actual ([double]$Plan.effective_linear_magnification) -Expected ($ExpectedDetailScale / $ExpectedFullScale) -Description "Live pipeline ($Key) detail magnification"
    } elseif ($null -ne $Report.detail_pass.last_plan) {
        throw "Live pipeline ($Key) recorded a detail ROI while BUILD-INFO disables it."
    }
    if ($ProcessedFrames -lt $PolicyLiveMinProcessedFrames -or [int64]$Report.pipeline.rolling_sample_count -lt $PolicyLiveMinProcessedFrames) {
        throw "Live pipeline ($Key) processed no frames or too few frames/timing samples for release policy."
    }
    $ElapsedSeconds = Get-FinitePolicyNumber -Value $Report.pipeline.elapsed_seconds -Description "Live pipeline ($Key) elapsed seconds"
    if ($ElapsedSeconds -lt 2.0) {
        throw "Live pipeline ($Key) measured too little elapsed time."
    }
    $ReportDuration = ($ReportCompleted - $ReportStarted).TotalSeconds
    if ([Math]::Abs($ReportDuration - $ElapsedSeconds) -gt 2.0) {
        throw "Live pipeline ($Key) elapsed time is internally inconsistent."
    }
    $ExpectedElapsedFps = [double]$ProcessedFrames / $ElapsedSeconds
    $ActualElapsedFps = Get-FinitePolicyNumber -Value $Report.pipeline.elapsed_fps -Description "Live pipeline ($Key) elapsed FPS"
    if ([Math]::Abs($ExpectedElapsedFps - $ActualElapsedFps) -gt [Math]::Max(0.5, $ExpectedElapsedFps * 0.02)) {
        throw "Live pipeline ($Key) elapsed FPS is internally inconsistent."
    }
    if ([int]$Report.pipeline.rolling_sample_count -ne [Math]::Min([int]$ProcessedFrames, [int]$Report.config.stats_window)) {
        throw "Live pipeline ($Key) rolling sample count is internally inconsistent."
    }
    if ((Get-FinitePolicyNumber -Value $Report.pipeline.elapsed_fps -Description "Live pipeline ($Key) elapsed FPS") -lt $PolicyLiveMinElapsedFps) {
        throw "Live pipeline ($Key) elapsed FPS is below release policy."
    }
    if ((Get-FinitePolicyNumber -Value $Report.pipeline.update_fps -Description "Live pipeline ($Key) update FPS") -lt $PolicyLiveMinUpdateFps) {
        throw "Live pipeline ($Key) update FPS is below release policy."
    }
    if ((Get-FinitePolicyNumber -Value $Report.pipeline.timings.p95.observed_pipeline_ms -Description "Live pipeline ($Key) p95 pipeline latency") -gt $PolicyLiveMaxP95ObservedPipelineMs) {
        throw "Live pipeline ($Key) p95 pipeline latency exceeds release policy."
    }
    if ((Get-FinitePolicyNumber -Value $Report.pipeline.timings.p95.freshness_latency_ms -Description "Live pipeline ($Key) p95 freshness latency") -gt $PolicyLiveMaxP95FreshnessLatencyMs) {
        throw "Live pipeline ($Key) p95 freshness latency exceeds release policy."
    }
    $ExpectedTimingFields = @("capture_ms", "queue_age_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "detail_preprocess_ms", "detail_inference_ms", "detail_postprocess_ms", "control_ms", "processing_ms", "freshness_latency_ms", "observed_pipeline_ms", "draw_ms", "preview_service_ms")
    foreach ($Percentile in @("mean", "p50", "p95", "p99")) {
        foreach ($TimingField in $ExpectedTimingFields) {
            $null = Get-FinitePolicyNumber -Value $Report.pipeline.timings.$Percentile.$TimingField -Description "Live pipeline ($Key) $Percentile $TimingField"
        }
    }
    $Mean = $Report.pipeline.timings.mean
    $ExpectedProcessingMean =
        [double]$Mean.preprocess_ms +
        [double]$Mean.inference_ms +
        [double]$Mean.postprocess_ms +
        [double]$Mean.detail_preprocess_ms +
        [double]$Mean.detail_inference_ms +
        [double]$Mean.detail_postprocess_ms +
        [double]$Mean.control_ms
    Assert-CloseTiming -Actual ([double]$Mean.processing_ms) -Expected $ExpectedProcessingMean -Description "Live pipeline ($Key) mean processing/component timing"
    Assert-CloseTiming -Actual ([double]$Mean.freshness_latency_ms) -Expected ([double]$Mean.queue_age_ms + [double]$Mean.processing_ms) -Description "Live pipeline ($Key) mean freshness timing"
    Assert-CloseTiming -Actual ([double]$Mean.observed_pipeline_ms) -Expected ([double]$Mean.capture_ms + [double]$Mean.freshness_latency_ms) -Description "Live pipeline ($Key) mean observed-pipeline timing"
    if ($null -eq $ExpectedDetailCrop) {
        if ([bool]$Report.detail_pass.enabled -or $null -ne $Report.detail_pass.requested_crop_size) {
            throw "Live pipeline ($Key) unexpectedly enabled the detail pass."
        }
        foreach ($Percentile in @("mean", "p50", "p95", "p99")) {
            foreach ($TimingField in @("detail_preprocess_ms", "detail_inference_ms", "detail_postprocess_ms")) {
                if ([double]$Report.pipeline.timings.$Percentile.$TimingField -ne 0.0) {
                    throw "Live pipeline ($Key) recorded detail timing while detail was disabled."
                }
            }
        }
    } elseif (-not [bool]$Report.detail_pass.enabled -or [int]$Report.detail_pass.requested_crop_size -ne $ExpectedDetailCrop) {
        throw "Live pipeline ($Key) did not record its requested detail workload."
    }
    if ($ExpectedDetailEnabled -and [double]$Mean.detail_inference_ms -le 0.0) {
        throw "Live pipeline ($Key) did not time the BUILD-INFO detail inference."
    }
    if ($PreviewEnabled) {
        if ([double]$Mean.preview_service_ms -le 0.0) {
            throw "Live pipeline ($Key) recorded no preview service work."
        }
    } else {
        foreach ($Percentile in @("mean", "p50", "p95", "p99")) {
            if ([double]$Report.pipeline.timings.$Percentile.preview_service_ms -ne 0.0) {
                throw "Live pipeline ($Key) recorded preview timing while preview was disabled."
            }
        }
    }
    if ((Get-FinitePolicyNumber -Value $Report.termination.requested_max_seconds -Description "Live pipeline ($Key) maximum seconds") -gt $PolicyLiveMaxSeconds) {
        throw "Live pipeline ($Key) maximum duration exceeds release policy."
    }
    if ([double]$Report.termination.requested_max_seconds -ne $PolicyLiveMaxSeconds -or [int]$Report.termination.requested_max_frames -ne 1000 -or [int]$Report.config.stats_window -ne 1000) {
        throw "Live pipeline ($Key) bounds differ from repository release policy."
    }
    $TerminationReason = [string]$Report.termination.reason
    if ($TerminationReason -notin @("max_frames", "max_seconds")) {
        throw "Live pipeline ($Key) ended unexpectedly: $TerminationReason."
    }
    if ($TerminationReason -eq "max_frames" -and $ProcessedFrames -ne [int64]$Report.termination.requested_max_frames) {
        throw "Live pipeline ($Key) did not reach its claimed frame bound."
    }
    if ($TerminationReason -eq "max_seconds" -and [Math]::Abs($ElapsedSeconds - [double]$Report.termination.requested_max_seconds) -gt 2.0) {
        throw "Live pipeline ($Key) did not reach its claimed time bound."
    }
    if ($ProcessedFrames -gt [int64]$Report.termination.requested_max_frames -or $ElapsedSeconds -gt ([double]$Report.termination.requested_max_seconds + 2.0)) {
        throw "Live pipeline ($Key) exceeded its configured bound."
    }
    if ([string]$Report.source.backend -ne "dxcam-dxgi") {
        throw "Live pipeline ($Key) did not use DXcam/DXGI; reported '$($Report.source.backend)'."
    }
    if ($null -ne $Report.source.fallback_reason -and -not [string]::IsNullOrWhiteSpace([string]$Report.source.fallback_reason)) {
        throw "Live pipeline ($Key) reported a capture fallback: $($Report.source.fallback_reason)"
    }
    if ([int64]$Report.capture.read_failures -ne 0) {
        throw "Live pipeline ($Key) reported capture read failures."
    }
    if ($Provider -eq "DirectML") {
        if ($null -eq $Report.directml_adapter) {
            throw "Live pipeline ($Key) omitted its DirectML adapter record."
        }
        if ([int]$Report.directml_adapter.effective_index -ne $AdapterIndex) {
            throw "Live pipeline ($Key) recorded the wrong DirectML adapter index."
        }
        if ([bool]$Report.directml_adapter.requested_provider_mismatch) {
            throw "Live pipeline ($Key) reported a DirectML provider-index mismatch."
        }
        if (
            [string]$Report.directml_adapter.enumeration_status -ne "matched_dxgi_adapter" -or
            $null -eq $Report.directml_adapter.descriptor -or
            [int]$Report.directml_adapter.descriptor.index -ne $AdapterIndex -or
            [string]::IsNullOrWhiteSpace([string]$Report.directml_adapter.descriptor.name)
        ) {
            throw "Live pipeline ($Key) did not bind a concrete DXGI adapter descriptor."
        }
        if (
            -not [string]::IsNullOrWhiteSpace($ExpectedAdapterName) -and
            (Normalize-AdapterName -Value ([string]$Report.directml_adapter.descriptor.name)) -cne
                (Normalize-AdapterName -Value $ExpectedAdapterName)
        ) {
            throw "Live pipeline ($Key) DXGI descriptor does not match -ExpectedAdapterName."
        }
    }
    $RunRecord.metrics = [ordered]@{
        file = [System.IO.Path]::GetFileName($MetricsPath)
        sha256 = Get-Sha256 -Path $MetricsPath
    }
    return $RunRecord
}

$BundleRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$CliPath = Resolve-ExistingFile -Path (Join-Path $BundleRoot "ProAimCLI.exe") -Description "Frozen CLI"
$BuildInfoPath = Resolve-ExistingFile -Path (Join-Path $BundleRoot "BUILD-INFO.json") -Description "Bundle BUILD-INFO"
$DependencyManifestPath = Resolve-ExistingFile -Path (Join-Path $BundleRoot "DEPENDENCY-MANIFEST.json") -Description "Bundle dependency manifest"
$HelperPath = Resolve-ExistingFile -Path $PSCommandPath -Description "GPU qualification helper"

if ($Provider -eq "DirectML") {
    if ($AdapterIndex -lt 0) {
        throw "DirectML qualification requires -AdapterIndex N from ProAim's Scan hardware result."
    }
    $Device = "DIRECTML:$AdapterIndex"
    $ExpectedProvider = "DmlExecutionProvider"
    $ExpectedRuntimeVariant = "directml"
} else {
    if ($AdapterIndex -ne -1) {
        throw "-AdapterIndex applies only to DirectML. CUDA selects CUDAExecutionProvider."
    }
    $Device = "CUDA"
    $ExpectedProvider = "CUDAExecutionProvider"
    $ExpectedRuntimeVariant = "cuda"
}
if ($Provider -eq "CUDA" -and -not [string]::IsNullOrWhiteSpace($ExpectedAdapterName)) {
    throw "-ExpectedAdapterName applies only to DirectML."
}

$BuildInfo = Read-JsonFile -Path $BuildInfoPath -Description "Bundle BUILD-INFO"
if ([string]$BuildInfo.application -ne "ProAim") {
    throw "BUILD-INFO.json does not identify a ProAim bundle."
}
if ([string]$BuildInfo.runtime_variant -ne $ExpectedRuntimeVariant) {
    throw "This is a '$($BuildInfo.runtime_variant)' bundle, but -Provider $Provider requires the '$ExpectedRuntimeVariant' bundle."
}
if ([int]$BuildInfo.schema -ne 2) {
    throw "BUILD-INFO.json uses an unsupported schema."
}
$ReleaseDefault = $BuildInfo.release_default_model
if ($null -eq $ReleaseDefault -or [string]::IsNullOrWhiteSpace([string]$ReleaseDefault.preset)) {
    throw "BUILD-INFO.json omits its release-default model contract."
}
$ContractDetailCrop = $ReleaseDefault.detail_crop_size_source_pixels
if (
    -not (Test-StrictJsonInteger $ContractDetailCrop) -or
    [int]$ContractDetailCrop -lt 0 -or
    [int]$ContractDetailCrop -gt 16384
) {
    throw "BUILD-INFO.json release-default detail_crop_size_source_pixels is invalid."
}
if ($DetailCropSize -ge 0 -and $DetailCropSize -ne [int]$ContractDetailCrop) {
    throw "-DetailCropSize cannot override the release-default BUILD-INFO contract."
}
$DetailCropSize = [int]$ContractDetailCrop
$ReleaseDefaultShape = @($ReleaseDefault.input_shape_hw)
if ($ReleaseDefaultShape.Count -ne 2) {
    throw "BUILD-INFO.json release-default input_shape_hw must contain [height,width]."
}
foreach ($Dimension in $ReleaseDefaultShape) {
    if ([int]$Dimension -lt 32 -or [int]$Dimension -gt 4096 -or ([int]$Dimension % 32) -ne 0) {
        throw "BUILD-INFO.json release-default shape has an invalid YOLO dimension."
    }
}
$ReleaseDefaultModel = Resolve-BundleContractFile -RelativePath ([string]$ReleaseDefault.model_path) -Description "Release-default ONNX model" -BundleRoot $BundleRoot
$ReleaseDefaultLabels = Resolve-BundleContractFile -RelativePath ([string]$ReleaseDefault.labels_path) -Description "Release-default labels" -BundleRoot $BundleRoot
if ([System.IO.Path]::GetExtension($ReleaseDefaultModel).ToLowerInvariant() -ne ".onnx") {
    throw "Release-default model contract must select an ONNX graph."
}
$ReleaseDefaultModelHash = Get-Sha256 -Path $ReleaseDefaultModel
$ReleaseDefaultLabelsHash = Get-Sha256 -Path $ReleaseDefaultLabels
if ([string]$ReleaseDefault.model_sha256 -ne $ReleaseDefaultModelHash -or [string]$ReleaseDefault.labels_sha256 -ne $ReleaseDefaultLabelsHash) {
    throw "Release-default model or labels hash differs from BUILD-INFO.json."
}
$ReleaseDefaultInferenceSize = if ([int]$ReleaseDefaultShape[0] -eq [int]$ReleaseDefaultShape[1]) {
    [string][int]$ReleaseDefaultShape[0]
} else {
    "{0}x{1}" -f [int]$ReleaseDefaultShape[0], [int]$ReleaseDefaultShape[1]
}
if ([string]$BuildInfo.dependency_manifest.path -ne "DEPENDENCY-MANIFEST.json") {
    throw "BUILD-INFO.json does not bind the adjacent dependency manifest."
}
$DependencyManifestHash = Get-Sha256 -Path $DependencyManifestPath
if ([string]$BuildInfo.dependency_manifest.sha256 -ne $DependencyManifestHash) {
    throw "DEPENDENCY-MANIFEST.json does not match the SHA-256 recorded in BUILD-INFO.json."
}
$DependencyManifest = Read-JsonFile -Path $DependencyManifestPath -Description "Bundle dependency manifest"
$ExpectedLockProfile = if ($Provider -eq "DirectML") { "windows-directml-py313" } else { "windows-cuda-py313" }
if (
    [int]$DependencyManifest.schema_version -ne 1 -or
    [string]$DependencyManifest.application -ne "ProAim" -or
    [string]$DependencyManifest.runtime_variant -ne $ExpectedRuntimeVariant -or
    [string]$DependencyManifest.lock_profile -ne $ExpectedLockProfile -or
    -not [bool]$DependencyManifest.artifact_hash_contract.enforced_before_install
) {
    throw "DEPENDENCY-MANIFEST.json does not identify the required hash-locked $ExpectedLockProfile environment."
}
$DependencyDistributions = @($DependencyManifest.distributions)
if (
    $DependencyDistributions.Count -le 0 -or
    -not (Test-StrictJsonInteger -Value $BuildInfo.dependency_manifest.distribution_count) -or
    [int64]$BuildInfo.dependency_manifest.distribution_count -ne $DependencyDistributions.Count
) {
    throw "DEPENDENCY-MANIFEST.json has an invalid dependency distribution count."
}
$ExpectedInstalledFileKeys = @(
    "aggregate_sha256",
    "record_document_sha256",
    "record_entry_count",
    "record_sha256_entries_verified",
    "total_size_bytes",
    "unhashed_record_entries"
)
foreach ($Distribution in $DependencyDistributions) {
    if ($null -eq $Distribution -or $null -eq $Distribution.installed_files) {
        throw "DEPENDENCY-MANIFEST.json distribution lacks installed_files verification."
    }
    $InstalledFiles = $Distribution.installed_files
    $ActualInstalledFileKeys = @($InstalledFiles.PSObject.Properties.Name)
    if (
        $ActualInstalledFileKeys.Count -ne $ExpectedInstalledFileKeys.Count -or
        @(Compare-Object -ReferenceObject $ExpectedInstalledFileKeys -DifferenceObject $ActualInstalledFileKeys).Count -ne 0
    ) {
        throw "DEPENDENCY-MANIFEST.json distribution has an incomplete installed_files record."
    }
    $EntryCount = $InstalledFiles.record_entry_count
    $VerifiedCount = $InstalledFiles.record_sha256_entries_verified
    $UnhashedCount = $InstalledFiles.unhashed_record_entries
    $TotalSize = $InstalledFiles.total_size_bytes
    if (
        -not (Test-StrictJsonInteger -Value $EntryCount) -or [int64]$EntryCount -lt 2 -or
        -not (Test-StrictJsonInteger -Value $VerifiedCount) -or [int64]$VerifiedCount -ne ([int64]$EntryCount - 1) -or
        -not (Test-StrictJsonInteger -Value $UnhashedCount) -or [int64]$UnhashedCount -ne 1 -or
        -not (Test-StrictJsonInteger -Value $TotalSize) -or [int64]$TotalSize -le 0
    ) {
        throw "DEPENDENCY-MANIFEST.json distribution has invalid installed RECORD counts."
    }
    if (
        $InstalledFiles.aggregate_sha256 -isnot [string] -or
        [string]$InstalledFiles.aggregate_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
        $InstalledFiles.record_document_sha256 -isnot [string] -or
        [string]$InstalledFiles.record_document_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
        $Distribution.installed_record_sha256 -isnot [string] -or
        [string]$Distribution.installed_record_sha256 -cnotmatch '^[0-9a-f]{64}$'
    ) {
        throw "DEPENDENCY-MANIFEST.json distribution has an invalid installed-file SHA-256."
    }
    if ([string]$InstalledFiles.record_document_sha256 -cne [string]$Distribution.installed_record_sha256) {
        throw "DEPENDENCY-MANIFEST.json distribution installed RECORD digests disagree."
    }
}

$BundleArchive = $null
if (-not [string]::IsNullOrWhiteSpace($BundleArchivePath)) {
    $BundleArchive = Resolve-ExistingFile -Path $BundleArchivePath -Description "Original bundle archive"
    if ([System.IO.Path]::GetExtension($BundleArchive).ToLowerInvariant() -ne ".zip") {
        throw "-BundleArchivePath must select the original ZIP archive."
    }
}

if ([string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
    $Timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $EvidenceDirectory = "ProAim-GPU-Evidence-$Timestamp"
}
if (-not [System.IO.Path]::IsPathRooted($EvidenceDirectory)) {
    $EvidenceDirectory = Join-Path $BundleRoot $EvidenceDirectory
}
$FinalDirectory = [System.IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $FinalDirectory) {
    throw "Refusing to overwrite existing evidence path: $FinalDirectory"
}
$EvidenceParent = Split-Path -Parent $FinalDirectory
if ([string]::IsNullOrWhiteSpace($EvidenceParent) -or -not (Test-Path -LiteralPath $EvidenceParent -PathType Container)) {
    throw "Evidence parent directory must already exist: $EvidenceParent"
}
$LeafName = Split-Path -Leaf $FinalDirectory
if ([string]::IsNullOrWhiteSpace($LeafName)) {
    throw "Evidence directory must have a non-empty final name."
}
$StageDirectory = Join-Path $EvidenceParent (".{0}.partial-{1}" -f $LeafName, [Guid]::NewGuid().ToString("N"))
$StageCreated = $false

$TaskManagerInstructions = @"
ProAim physical-GPU confirmation (required)
===========================================

The helper can prove that ONNX Runtime accepted $ExpectedProvider with CPU graph-node fallback disabled.
It cannot prove which physical GPU did the work.

Before continuing:
1. Open Task Manager (Ctrl+Shift+Esc), choose More details if shown, and open
   the Performance tab.
2. Identify the intended discrete GPU by its full product name. On a hybrid
   laptop, do not assume that the highest Task Manager GPU number is correct.
3. In Details, enable the GPU and GPU engine columns and locate ProAimCLI.exe.
4. During every benchmark, verify sustained activity on the intended physical
   GPU's compute engine. During optional live runs, also verify it while both
   no-preview and preview-15 modes run.
5. Record the physical GPU name, Task Manager GPU/engine text, tester, and UTC
   time in TASK-MANAGER-CONFIRMATION.txt after this helper succeeds. Attach a
   screenshot if release policy permits it.

DirectML note: -AdapterIndex is the DXGI index shown by ProAim Scan hardware;
it is not necessarily the same number Task Manager displays. CUDA does not use
a DirectML adapter index. Provider activation alone is not physical-GPU proof.
"@

try {
    [System.IO.Directory]::CreateDirectory($StageDirectory) | Out-Null
    $StageCreated = $true

    Write-Utf8File -Path (Join-Path $StageDirectory "TASK-MANAGER-INSTRUCTIONS.txt") -Content ($TaskManagerInstructions.Trim() + [Environment]::NewLine)
    $ConfirmationTemplate = @"
Manual confirmation is intentionally not collected or inferred by the helper.

Tester:
UTC date/time:
Physical GPU full name:
Task Manager GPU number and engine:
Observed during release-default model benchmark: YES / NO
Observed during live no-preview (if run): YES / NO / N/A
Observed during live preview-15 (if run): YES / NO / N/A
Screenshot filename (optional):
Notes:
"@
    Write-Utf8File -Path (Join-Path $StageDirectory "TASK-MANAGER-CONFIRMATION.txt") -Content ($ConfirmationTemplate.Trim() + [Environment]::NewLine)
    [System.IO.File]::Copy($BuildInfoPath, (Join-Path $StageDirectory "bundle-BUILD-INFO.json"), $false)
    [System.IO.File]::Copy($DependencyManifestPath, (Join-Path $StageDirectory "bundle-DEPENDENCY-MANIFEST.json"), $false)

    Write-Host ""
    Write-Host $TaskManagerInstructions
    Write-Host ""
    if (-not $SkipReadyPrompt) {
        Read-Host "Open Task Manager now, then press Enter to start" | Out-Null
    } else {
        Write-Warning "Ready prompt skipped; physical-GPU confirmation remains pending."
    }

    $ArtifactRecords = @(
        (New-ArtifactRecord -Role "frozen_cli" -Path $CliPath -BundleRoot $BundleRoot),
        (New-ArtifactRecord -Role "build_info" -Path $BuildInfoPath -BundleRoot $BundleRoot),
        (New-ArtifactRecord -Role "dependency_manifest" -Path $DependencyManifestPath -BundleRoot $BundleRoot),
        (New-ArtifactRecord -Role "qualification_helper" -Path $HelperPath -BundleRoot $BundleRoot),
        (New-ArtifactRecord -Role "release_default_model" -Path $ReleaseDefaultModel -BundleRoot $BundleRoot),
        (New-ArtifactRecord -Role "release_default_labels" -Path $ReleaseDefaultLabels -BundleRoot $BundleRoot)
    )
    $GuiPath = Join-Path $BundleRoot "ProAim.exe"
    if (Test-Path -LiteralPath $GuiPath -PathType Leaf) {
        $ArtifactRecords += New-ArtifactRecord -Role "frozen_gui" -Path $GuiPath -BundleRoot $BundleRoot
    }
    if ($null -ne $BundleArchive) {
        $ArtifactRecords += New-ArtifactRecord -Role "original_bundle_archive" -Path $BundleArchive -BundleRoot $BundleRoot
    }

    $RuntimeStdout = Join-Path $StageDirectory "runtime-info.json"
    $RuntimeStderr = Join-Path $StageDirectory "runtime-info.stderr.txt"
    $RunRecords = @(
        (Invoke-FrozenCli -Name "frozen runtime info" -Arguments @("--runtime-info") -StdoutPath $RuntimeStdout -StderrPath $RuntimeStderr)
    )
    $RuntimeInfo = Read-JsonFile -Path $RuntimeStdout -Description "Frozen runtime info"
    if (-not [bool]$RuntimeInfo.frozen) {
        throw "Runtime info did not identify a frozen executable."
    }
    $AvailableProviders = @($RuntimeInfo.onnxruntime_providers | ForEach-Object { [string]$_ })
    if (-not ($AvailableProviders -contains $ExpectedProvider)) {
        throw "Frozen runtime does not expose $ExpectedProvider. Available: $($AvailableProviders -join ', ')."
    }

    $RunRecords += Invoke-ModelBenchmark -Key "release-default" -ModelPath $ReleaseDefaultModel -LabelsPath $ReleaseDefaultLabels -InferenceSize $ReleaseDefaultInferenceSize -ExpectedModelHash $ReleaseDefaultModelHash -ExpectedLabelsHash $ReleaseDefaultLabelsHash -ExpectedShape $ReleaseDefaultShape

    if ($RunLive) {
        $LiveModel = $ReleaseDefaultModel
        $LiveLabels = $ReleaseDefaultLabels
        $LiveSize = $ReleaseDefaultInferenceSize
        $LiveHash = $ReleaseDefaultModelHash
        $LiveKey = "release-default"
        $RunRecords += Invoke-LivePass -Key "$LiveKey-no-preview" -PreviewEnabled $false -ModelPath $LiveModel -LabelsPath $LiveLabels -InferenceSize $LiveSize -ExpectedModelHash $LiveHash -ExpectedLabelsHash $ReleaseDefaultLabelsHash -ExpectedShape $ReleaseDefaultShape
        $RunRecords += Invoke-LivePass -Key "$LiveKey-preview-15" -PreviewEnabled $true -ModelPath $LiveModel -LabelsPath $LiveLabels -InferenceSize $LiveSize -ExpectedModelHash $LiveHash -ExpectedLabelsHash $ReleaseDefaultLabelsHash -ExpectedShape $ReleaseDefaultShape
    }

    $EvidenceFiles = @()
    foreach ($File in Get-ChildItem -LiteralPath $StageDirectory -File | Sort-Object Name) {
        # This template is expected to change when the tester records the
        # independent observation. It is intentionally outside the immutable
        # software-evidence hashes below.
        if ($File.Name -eq "TASK-MANAGER-CONFIRMATION.txt") {
            continue
        }
        $EvidenceFiles += [ordered]@{
            file = $File.Name
            size_bytes = [int64]$File.Length
            sha256 = Get-Sha256 -Path $File.FullName
        }
    }
    $Manifest = [ordered]@{
        schema_version = 1
        generated_at_utc = [DateTime]::UtcNow.ToString("o")
        status = "software_checks_passed_physical_gpu_confirmation_pending"
        qualified = $false
        qualification_limit = "Task Manager physical-GPU activity must be confirmed and recorded separately."
        provider = [ordered]@{
            selection = $Provider
            requested_device = $Device
            expected_execution_provider = $ExpectedProvider
            directml_adapter_index = if ($Provider -eq "DirectML") { $AdapterIndex } else { $null }
        }
        bundle_build_info = $BuildInfo
        benchmark_bounds = [ordered]@{
            samples = $Samples
            warmup = $Warmup
            iterations = $Iterations
            repeats = $Repeats
        }
        live_bounds = [ordered]@{
            enabled = [bool]$RunLive
            selected_model = "release-default"
            release_default_model = $ReleaseDefault
            screen_monitor = $ScreenMonitor
            screen_fps = $ScreenFps
            max_frames = $MaxFrames
            max_seconds = $MaxSeconds
            detail_crop_size = if ($DetailCropSize -gt 0) { $DetailCropSize } else { $null }
            modes = if ($RunLive) { @("no-preview", "preview-15") } else { @() }
        }
        input_artifacts = @($ArtifactRecords)
        runs = @($RunRecords)
        evidence_files = @($EvidenceFiles)
        manual_confirmation = [ordered]@{
            required = $true
            completed_by_helper = $false
            instructions = "TASK-MANAGER-INSTRUCTIONS.txt"
            template = "TASK-MANAGER-CONFIRMATION.txt"
        }
    }
    $ManifestJson = $Manifest | ConvertTo-Json -Depth 20
    Write-Utf8File -Path (Join-Path $StageDirectory "qualification-manifest.json") -Content ($ManifestJson + [Environment]::NewLine)

    if (Test-Path -LiteralPath $FinalDirectory) {
        throw "Evidence destination appeared during the run; refusing to overwrite it: $FinalDirectory"
    }
    [System.IO.Directory]::Move($StageDirectory, $FinalDirectory)
    $StageCreated = $false
} catch {
    if ($StageCreated -and (Test-Path -LiteralPath $StageDirectory -PathType Container)) {
        Remove-Item -LiteralPath $StageDirectory -Recurse -Force
    }
    throw
}

Write-Host ""
Write-Host "Software qualification evidence written atomically to: $FinalDirectory"
Write-Warning "This is not a physical-GPU qualification until TASK-MANAGER-CONFIRMATION.txt is completed."
