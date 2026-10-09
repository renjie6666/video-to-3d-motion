param(
    [string]$Sequence = "",
    [string]$Annotations = "",
    [string]$OutputDirectory = "",
    [string]$Pose2DConfig = "",
    [ValidateSet("strict", "highest_score")][string]$PersonSelection = "highest_score",
    [ValidateRange(0, 2147483647)][int]$MaxFrames = 0,
    [ValidateRange(1, 2147483647)][int]$ComparisonEveryNFrames = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project .venv is missing. Run tools\setup_local_gpu.ps1 first."
}
if (-not $Sequence) {
    $Sequence = Join-Path $projectRoot "datasets\s_01_act_02_subact_01_ca_01"
}
if (-not $Annotations) {
    $Annotations = Join-Path $projectRoot "datasets\h36m_train.pkl"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot "results"
}
if (-not $Pose2DConfig) {
    $Pose2DConfig = Join-Path $projectRoot "configs\pose2d.yaml"
}
$Sequence = [IO.Path]::GetFullPath($Sequence)
$Annotations = [IO.Path]::GetFullPath($Annotations)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$Pose2DConfig = [IO.Path]::GetFullPath($Pose2DConfig)
if (-not (Test-Path -LiteralPath $Sequence -PathType Container)) {
    throw "Image sequence does not exist: $Sequence"
}
if (-not (Test-Path -LiteralPath $Annotations -PathType Leaf)) {
    throw "Annotation pickle does not exist: $Annotations"
}
$runArguments = @(
    "-m", "video_to_3d_motion.pose2d.experiments", $Sequence,
    "--media-config", (Join-Path $projectRoot "configs\media_decode.yaml"),
    "--pose2d-config", $Pose2DConfig,
    "--output-root", $OutputDirectory,
    "--annotations", $Annotations,
    "--comparison-every-n-frames", $ComparisonEveryNFrames
)
$runArguments += @("--person-selection", $PersonSelection)
if ($MaxFrames -gt 0) {
    $runArguments += @("--max-frames", $MaxFrames)
}
& $pythonPath @runArguments
if ($LASTEXITCODE -ne 0) {
    throw "Pose2D experiment failed; see the experiment CSV and run log."
}
Write-Host "Results: $OutputDirectory"
