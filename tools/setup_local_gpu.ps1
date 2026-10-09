<#
Set up the Windows S1-02 environment from the project root or any directory.
Requires uv on PATH. Downloads Python and packages into the project's .venv.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gpuPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$uvExecutable = (Get-Command uv -ErrorAction Stop).Source
$previousCache = $env:UV_CACHE_DIR
$previousPythonDirectory = $env:UV_PYTHON_INSTALL_DIR
$previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousHttpTimeout = $env:UV_HTTP_TIMEOUT

function Invoke-Uv {
    param([string[]]$UvArguments)
    & $uvExecutable @UvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python-gpu'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv'
    if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = '300' }
    if (-not (Test-Path -LiteralPath $gpuPython)) {
        Invoke-Uv -UvArguments @('venv', '.venv', '--python', '3.10', '--managed-python')
    }
    & $gpuPython -c 'import sys; assert sys.version_info[:2] == (3, 10), "This environment requires Python 3.10"'
    if ($LASTEXITCODE -ne 0) { throw 'Invalid Python environment' }

    $constraints = Join-Path $projectRoot 'configs\gpu-windows-constraints.txt'
    Invoke-Uv -UvArguments @('pip', 'install', '--python', $gpuPython, '-c', $constraints,
        'numpy==1.26.4', 'pip', 'setuptools==80.9.0', 'wheel')
    Invoke-Uv -UvArguments @('pip', 'install', '--python', $gpuPython, '-c', $constraints,
        'torch==2.1.2', 'torchvision==0.16.2', '--index-url', 'https://download.pytorch.org/whl/cu118')
    Invoke-Uv -UvArguments @('pip', 'install', '--python', $gpuPython, '--no-deps',
        'https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl')
    Invoke-Uv -UvArguments @('sync', '--locked', '--extra', 'pose2d', '--python', $gpuPython,
        '--no-build-isolation-package', 'chumpy')
    Invoke-Uv -UvArguments @('pip', 'check', '--python', $gpuPython)
    & $gpuPython tools/check_local_gpu.py --require-pose2d --output results/local_gpu_check.json
    if ($LASTEXITCODE -ne 0) { throw 'GPU environment verification failed; see results/local_gpu_check.json' }
    Write-Host 'SUCCESS: use .venv\Scripts\python.exe for S1-02.'
} finally {
    $env:UV_CACHE_DIR = $previousCache
    $env:UV_PYTHON_INSTALL_DIR = $previousPythonDirectory
    $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
    $env:UV_HTTP_TIMEOUT = $previousHttpTimeout
    Pop-Location
}
