param([switch]$DownloadAssets, [string]$CondaExe = '')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$sources = @(
    @{ Url = 'https://github.com/OpenMotionLab/MotionGPT.git'; Path = 'work/motiongpt_source'; Commit = '001aaca8d0ee218fc17f8265d11ac124044fe42f' },
    @{ Url = 'https://github.com/ou524u/MotionCritic.git'; Path = 'work/motioncritic_source'; Commit = '71e9f52e67df4d9924607afd6ac4e0d4dc0dd27e' }
)
foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source.Path)) {
        git clone $source.Url $source.Path
        if ($LASTEXITCODE -ne 0) { throw 'Source clone failed' }
        git -C $source.Path checkout --detach $source.Commit
        if ($LASTEXITCODE -ne 0) { throw 'Source checkout failed' }
    }
    $actual = git -C $source.Path rev-parse HEAD
    if ($actual.Trim() -ne $source.Commit) { throw "Unexpected source revision at $($source.Path)" }
}
$condaRoot = Join-Path $repoRoot 'work/miniconda3'
if (-not $CondaExe) { $CondaExe = Join-Path $condaRoot 'Scripts/conda.exe' }
if (-not (Test-Path -LiteralPath $CondaExe)) {
    & (Join-Path $PSScriptRoot 'install_miniconda.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Miniconda installation failed' }
}
$environmentRoot = Join-Path $condaRoot 'envs/motiongpt'
$interpreter = Join-Path $environmentRoot 'python.exe'
if (-not (Test-Path -LiteralPath $interpreter)) {
    # Explicitly override installer-default channels. This matches the
    # conda-forge-only environment.yml without touching global Conda settings.
    if (Test-Path -LiteralPath 'configs/motiongpt_stage1/conda-win64-explicit.txt') {
        & $CondaExe create --prefix $environmentRoot --yes --override-channels --channel conda-forge --file configs/motiongpt_stage1/conda-win64-explicit.txt
    } else {
        & $CondaExe create --prefix $environmentRoot --yes --override-channels --channel conda-forge python=3.10 pip=25.0 setuptools=69.5 wheel=0.45
    }
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
uv pip install --python $interpreter --cache-dir .uv-cache --no-config pip==25.0.1 setuptools==69.5.1 wheel==0.45.1
if ($LASTEXITCODE -ne 0) { throw 'Build dependencies installation failed' }
$requirements = 'configs/motiongpt_stage1/requirements.txt'
if (Test-Path -LiteralPath 'configs/motiongpt_stage1/requirements-win64.lock.txt') {
    $requirements = 'configs/motiongpt_stage1/requirements-win64.lock.txt'
}
uv pip install --python $interpreter --cache-dir .uv-cache --no-config --no-build-isolation -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'MotionGPT dependencies installation failed' }
if ($DownloadAssets) {
    & $interpreter tools/prepare_motiongpt_stage1.py model critic t5 evaluator smpl example
    if ($LASTEXITCODE -ne 0) { throw 'Asset download failed' }
    & $interpreter tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml extract
    if ($LASTEXITCODE -ne 0) { throw 'Asset extraction failed' }
}
& $interpreter tools/run_motiongpt_stage1.py --config configs/motiongpt_stage1/config.yaml check
exit $LASTEXITCODE
