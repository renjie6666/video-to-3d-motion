param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$InputSummary,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python environment not found: $pythonPath"
}
if (-not $Config) {
    $Config = Join-Path $projectRoot "configs/pose_adapter.yaml"
}
& $pythonPath -m video_to_3d_motion.pose_adapter.cli --input $InputPath --input-summary $InputSummary --config $Config --output-dir $OutputDir
exit $LASTEXITCODE
