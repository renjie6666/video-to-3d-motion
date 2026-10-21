$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$installRoot = Join-Path $repoRoot 'work/miniconda3'
$installer = Join-Path $repoRoot 'work/Miniconda3-py310_26.9.3-0-Windows-x86_64.exe'
$expectedHash = '4716fd947314668123e9b1a4aa011a6bff629bf1ec0614fb72df50997608a1e3'
$condaExe = Join-Path $installRoot 'Scripts/conda.exe'
if (Test-Path -LiteralPath $condaExe) {
    & $condaExe --version
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $installer)) {
    Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-py310_26.9.3-0-Windows-x86_64.exe' -OutFile $installer
}
$actualHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw 'Miniconda installer SHA256 mismatch' }
$process = Start-Process -FilePath $installer -ArgumentList @('/S', '/InstallationType=JustMe', '/AddToPath=0', '/RegisterPython=0', "/D=$installRoot") -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -ne 0) { throw "Miniconda installer exited with $($process.ExitCode)" }
& $condaExe --version
if ($LASTEXITCODE -ne 0) { throw 'Miniconda installation verification failed' }
