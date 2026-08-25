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
    [string]$Device = "gpu",

    [string]$CondaExecutable,

    [ValidateSet("libmamba", "rattler")]
    [string]$Solver = "rattler"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path $repositoryRoot "Environments\environment-$Device-windows.yml"
$environmentName = if ($Device -eq "gpu") { "celfdrive-windows" } else { "celfdrive-windows-cpu" }
$condaPath = $CondaExecutable
if ([string]::IsNullOrWhiteSpace($condaPath)) {
    $condaApplication = @(Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue)[0]
    if ($null -ne $condaApplication) {
        $condaPath = $condaApplication.Source
    }
}
if ([string]::IsNullOrWhiteSpace($condaPath) -or -not (Test-Path -LiteralPath $condaPath)) {
    throw "Conda executable was not found on PATH. Install Conda, then open a new PowerShell window and retry."
}
$condaPath = (Resolve-Path -LiteralPath $condaPath).Path
$condaVersion = (& $condaPath --version).Trim()
if ($LASTEXITCODE -ne 0 -or $condaVersion -notmatch '^conda (?<major>\d+)\.(?<minor>\d+)') {
    throw "Could not determine the Conda version from '$condaPath'."
}
if ([int]$Matches.major -lt 22 -or ([int]$Matches.major -eq 22 -and [int]$Matches.minor -lt 11)) {
    throw "Conda $condaVersion is unsupported for this Conda Forge CUDA environment. Upgrade Conda to 22.11 or newer, then retry."
}

& $condaPath env create --solver $Solver --file $environmentFile
$environmentRoots = ((& $condaPath config --show envs_dirs --json) | ConvertFrom-Json).envs_dirs
if ($LASTEXITCODE -ne 0 -or $null -eq $environmentRoots -or $environmentRoots.Count -eq 0) {
    throw "Could not determine the Conda environment directories for '$condaPath'."
}
$environmentPath = Join-Path $environmentRoots[0] $environmentName
if ($LASTEXITCODE -ne 0 -or $null -eq $environmentPath -or -not (Test-Path -LiteralPath $environmentPath)) {
    throw "Conda could not create '$environmentName' from '$environmentFile'."
}
$environmentPython = Join-Path $environmentPath "python.exe"
if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw "Conda created '$environmentName' without Python at '$environmentPython'."
}
& $environmentPython (Join-Path $repositoryRoot "tools\install_sahi.py")
if ($LASTEXITCODE -ne 0) {
    throw "SAHI installation failed in '$environmentName'."
}
& $environmentPython (Join-Path $repositoryRoot "tools\verify_slidebook_runtime.py") "--device" $Device
if ($LASTEXITCODE -ne 0) {
    throw "CelFDrive runtime verification failed for '$environmentName'."
}

Write-Host "Environment '$environmentName' is ready at '$environmentPath'."
Write-Host "Run commands with: & '$environmentPython' <command>"
