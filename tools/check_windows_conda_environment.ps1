<#
.SYNOPSIS
Check that a Windows CelFDrive Conda environment is usable.

.DESCRIPTION
This script is read-only. It does not create, update, or remove Conda
environments or packages. Run it from Anaconda Prompt after activating the
new Anaconda installation.
#>
[CmdletBinding()]
param(
    [string]$CondaExecutable = "D:\Anaconda3\Scripts\conda.exe",
    [string]$EnvironmentName = "celfdrive-windows"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CondaExecutable)) {
    throw "Conda executable was not found at '$CondaExecutable'. Pass -CondaExecutable with the full path to conda.exe."
}

Write-Host "Conda: $(& $CondaExecutable --version)"
Write-Host "Base:  $(& $CondaExecutable info --base)"

$environmentPaths = ((& $CondaExecutable env list --json) | ConvertFrom-Json).envs
$environmentPath = $environmentPaths |
    Where-Object { (Split-Path -Leaf $_) -eq $EnvironmentName } |
    Select-Object -First 1

if ($null -eq $environmentPath) {
    throw "Environment '$EnvironmentName' is not registered with this Conda installation."
}

Write-Host "Environment: $environmentPath"
& $CondaExecutable run --name $EnvironmentName python -c "import sys; print(sys.executable); import cv2, pandas, tifffile, torch; print('imports: passed'); print(f'torch: {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}')"
if ($LASTEXITCODE -ne 0) {
    throw "Core CelFDrive imports failed in '$EnvironmentName'."
}

& $CondaExecutable run --name $EnvironmentName python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "The CelFDrive test suite failed in '$EnvironmentName'."
}

Write-Host "CelFDrive environment check passed."
