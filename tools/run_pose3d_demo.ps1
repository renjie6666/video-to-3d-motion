param(
    [ValidateSet('baseline','full')][string]$Mode = 'full',
    [ValidateSet('none','ema')][string]$Smoothing = 'none',
    [string]$OutputDir,
    [switch]$NoFlip
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runName = 'rtmpose_l_h36m_2026-10-10-09-52-29'
$sequence = 'S9_Directions_1.54138969'
$runRoot = Join-Path $projectRoot "results/s9_generalization_directions_1_54138969/runs/$runName"
if (-not $OutputDir) {
    $experiment = 'posemamba_s9_' + $Mode + '_' + $Smoothing + '_' + (Get-Date -Format 'yyyy-MM-dd_HH-mm-ss-fff')
    $OutputDir = Join-Path $projectRoot "results/$experiment"
}
$parameters = @{
    InputPath = Join-Path $runRoot ($runName + '_pose2d_frames.jsonl')
    InputSummary = Join-Path $runRoot ($runName + '_summary.csv')
    OutputDir = $OutputDir
    Mode = $Mode
    Smoothing = $Smoothing
    Annotations = Join-Path $projectRoot 'datasets/Human3.6M/h36m_annot/h36m/annot/valid.h5'
    AnnotationNames = Join-Path $projectRoot 'datasets/Human3.6M/h36m_annot/h36m/annot/valid_images.txt'
    Sequence = $sequence
    ImageDir = Join-Path $projectRoot "results/s9_generalization_directions_1_54138969/input/$sequence"
}
if ($NoFlip) { $parameters.NoFlip = $true }
& (Join-Path $PSScriptRoot 'run_pose3d.ps1') @parameters
exit $LASTEXITCODE
