<#
.SYNOPSIS
Create a Windows CelFDrive Conda Forge environment with the selected device.

.DESCRIPTION
SAHI is installed after Conda resolves the native stack. Conda environment YAML
files pass pip entries through a requirements file, which cannot represent
pip's --no-deps install flag. Installing it here prevents pip from replacing
Conda Forge's Torch or OpenCV packages.

.EXAMPLE
.\tools\create_windows_conda_env.ps1 -Device gpu
.\tools\create_windows_conda_env.ps1 -Device cpu
#>
[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu")]
    [string]$Device = "gpu"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path $repositoryRoot "Environments\environment-$Device-windows.yml"
$environmentName = if ($Device -eq "gpu") { "celfdrive-windows" } else { "celfdrive-windows-cpu" }

& conda env create --file $environmentFile
if ($LASTEXITCODE -ne 0) {
    throw "Conda could not create '$environmentName' from '$environmentFile'."
}
& conda run --name $environmentName python (Join-Path $repositoryRoot "tools\install_sahi.py")
if ($LASTEXITCODE -ne 0) {
    throw "SAHI installation failed in '$environmentName'."
}
& conda run --name $environmentName python (Join-Path $repositoryRoot "tools\verify_slidebook_runtime.py") "--device" $Device
if ($LASTEXITCODE -ne 0) {
    throw "CelFDrive runtime verification failed for '$environmentName'."
}

Write-Host "Environment '$environmentName' is ready. Activate with: conda activate $environmentName"
