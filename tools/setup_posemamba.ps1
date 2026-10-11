$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
# Use the already provisioned GPU Torch; do not replace the 2D stack.
& uv pip install --python $python 'einops==0.8.2' 'h5py>=3.11,<4' 'Pillow>=9,<12'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -B (Join-Path $PSScriptRoot 'download_posemamba.py')
exit $LASTEXITCODE
