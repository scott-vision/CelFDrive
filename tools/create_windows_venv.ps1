<#
.SYNOPSIS
Create an isolated Windows CelFDrive virtual environment.

.EXAMPLE
.\tools\create_windows_venv.ps1 -Device cpu
.\tools\create_windows_venv.ps1 -Device gpu -CudaWheel cu118
#>
[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu")]
    [string]$Device = "cpu",

    [ValidateSet("cu118", "cu126", "cu128")]
    [string]$CudaWheel = "cu118",

    [string]$VenvPath
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = ".venv-celfdrive-$Device"
}
$targetVenv = Join-Path $repositoryRoot $VenvPath

if (Test-Path -LiteralPath $targetVenv) {
    throw "Virtual environment already exists: $targetVenv. Remove it deliberately or choose -VenvPath."
}

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pythonLauncher) {
    & $pythonLauncher.Source "-3.11" "-m" "venv" $targetVenv
} else {
    throw "Python 3.11 is required. Install it with the Python launcher ('py -3.11') before creating a CelFDrive venv."
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $targetVenv)) {
    throw "Could not create the virtual environment with Python 3.11. Install it with the Python launcher ('py -3.11') and retry."
}

$venvPython = Join-Path $targetVenv "Scripts\python.exe"
& $venvPython "-m" "pip" "install" "--upgrade" "pip"

if ($Device -eq "gpu") {
    & $venvPython "-m" "pip" "install" "torch" "torchvision" "torchaudio" "--index-url" "https://download.pytorch.org/whl/$CudaWheel"
} else {
    & $venvPython "-m" "pip" "install" "torch" "torchvision" "torchaudio" "--index-url" "https://download.pytorch.org/whl/cpu"
}

& $venvPython "-m" "pip" "install" "-r" (Join-Path $repositoryRoot "Environments\requirements-windows-venv.txt")
& $venvPython (Join-Path $repositoryRoot "tools\verify_slidebook_runtime.py") "--device" $Device

Write-Host "Activate with: $targetVenv\Scripts\Activate.ps1"
