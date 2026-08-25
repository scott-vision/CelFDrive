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

    [string]$VenvPath,

    [string]$CondaExecutable,

    [string]$CondaEnvironmentPrefix
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = ".venv-celfdrive-$Device"
}
$targetVenv = if ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $VenvPath
} else {
    Join-Path $repositoryRoot $VenvPath
}

if (Test-Path -LiteralPath $targetVenv) {
    throw "Virtual environment already exists: $targetVenv. Remove it deliberately or choose -VenvPath."
}

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
$createdVenv = $false
if ($null -ne $pythonLauncher) {
    & $pythonLauncher.Source "-3.11" "-c" "import sys; assert sys.version_info[:2] == (3, 11)"
    if ($LASTEXITCODE -eq 0) {
        & $pythonLauncher.Source "-3.11" "-m" "venv" $targetVenv
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 could not create the virtual environment at '$targetVenv'."
        }
        $createdVenv = $true
    }
}

if (-not $createdVenv) {
    $condaPath = $CondaExecutable
    if ([string]::IsNullOrWhiteSpace($condaPath)) {
        $condaApplication = @(Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue)[0]
        if ($null -ne $condaApplication) {
            $condaPath = $condaApplication.Source
        }
    }
    if ([string]::IsNullOrWhiteSpace($condaPath) -or -not (Test-Path -LiteralPath $condaPath)) {
        throw "Python 3.11 is required. Install it with 'py -3.11', or create the celfdrive-windows Conda environment first so its Python 3.11 can bootstrap this venv."
    }
    $condaPath = (Resolve-Path -LiteralPath $condaPath).Path
    if ([string]::IsNullOrWhiteSpace($CondaEnvironmentPrefix)) {
        $environmentRoots = ((& $condaPath config --show envs_dirs --json) | ConvertFrom-Json).envs_dirs
        $CondaEnvironmentPrefix = Join-Path $environmentRoots[0] "celfdrive-windows"
    }
    if (-not (Test-Path -LiteralPath $CondaEnvironmentPrefix)) {
        throw "The Conda bootstrap environment was not found at '$CondaEnvironmentPrefix'. Create it first or pass -CondaEnvironmentPrefix."
    }
    $bootstrapPython = Join-Path $CondaEnvironmentPrefix "python.exe"
    if (-not (Test-Path -LiteralPath $bootstrapPython)) {
        throw "The Conda bootstrap Python was not found at '$bootstrapPython'."
    }
    & $bootstrapPython "-c" "import sys; assert sys.version_info[:2] == (3, 11)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 is unavailable. Install it with 'py -3.11', or create the celfdrive-windows Conda environment before creating a CelFDrive venv."
    }
    & $bootstrapPython "-m" "venv" $targetVenv
    if ($LASTEXITCODE -ne 0) {
        throw "The celfdrive-windows Python 3.11 environment could not create the virtual environment at '$targetVenv'."
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $targetVenv)) {
    throw "Could not create the virtual environment at '$targetVenv'."
}

$venvPython = Join-Path $targetVenv "Scripts\python.exe"
& $venvPython "-m" "pip" "install" "--no-cache-dir" "--upgrade" "pip"
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip in '$targetVenv'."
}

if ($Device -eq "gpu") {
    & $venvPython "-m" "pip" "install" "--no-cache-dir" "torch" "torchvision" "torchaudio" "--index-url" "https://download.pytorch.org/whl/$CudaWheel"
} else {
    & $venvPython "-m" "pip" "install" "--no-cache-dir" "torch" "torchvision" "torchaudio" "--index-url" "https://download.pytorch.org/whl/cpu"
}
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the $Device PyTorch build in '$targetVenv'."
}

& $venvPython "-m" "pip" "install" "--no-cache-dir" "-r" (Join-Path $repositoryRoot "Environments\requirements-windows-venv.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install CelFDrive requirements in '$targetVenv'."
}
& $venvPython (Join-Path $repositoryRoot "tools\verify_slidebook_runtime.py") "--device" $Device
if ($LASTEXITCODE -ne 0) {
    throw "CelFDrive runtime verification failed for '$targetVenv'."
}

Write-Host "Activate with: $targetVenv\Scripts\Activate.ps1"
