param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$InputSummary,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [ValidateSet('baseline','full')][string]$Mode = 'baseline',
    [ValidateSet('none','ema')][string]$Smoothing = 'none',
    [double]$TauSeconds = 0.04,
    [string]$Config,
    [string]$Annotations,
    [string]$AnnotationNames,
    [string]$Sequence,
    [string]$ImageDir,
    [string]$ImagePrefix,
    [switch]$NoFlip
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not $Config) { $Config = Join-Path $projectRoot 'configs/pose3d.yaml' }
$arguments = @('-B','-m','video_to_3d_motion.pose3d.cli','--input',$InputPath,
               '--input-summary',$InputSummary,'--config',$Config,'--output-dir',$OutputDir,
               '--mode',$Mode,'--smoothing',$Smoothing,'--tau-seconds',[string]$TauSeconds)
if ($Annotations) { $arguments += @('--annotations',$Annotations) }
if ($AnnotationNames) { $arguments += @('--annotation-names',$AnnotationNames) }
if ($Sequence) { $arguments += @('--sequence',$Sequence) }
if ($ImageDir) { $arguments += @('--image-dir',$ImageDir) }
if ($ImagePrefix) { $arguments += @('--image-prefix',$ImagePrefix) }
if ($NoFlip) { $arguments += '--no-flip' }
& $python @arguments
exit $LASTEXITCODE
