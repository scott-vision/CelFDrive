<#
.SYNOPSIS
Export an exact package lock file for a created CelFDrive environment.

.DESCRIPTION
The YAML files in Environments describe a compatible environment, not an exact
one. Archived releases need an exact one. Run this on the machine where the
environment is known to work; it writes Environments\<name>.lock.txt, which
recreates that environment byte-for-byte on the same platform with:

    conda create --name <name> --file Environments\<name>.lock.txt

Lock files are platform-specific. Export one per platform you support.

.EXAMPLE
.\tools\export_environment_lock.ps1 -Name celfdrive-windows
.\tools\export_environment_lock.ps1 -Name celfdrive-analysis
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Name
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $repositoryRoot "Environments\$Name.lock.txt"

& conda list --explicit --md5 --name $Name | Set-Content -Path $lockPath -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    throw "Conda could not list packages for '$Name'. Create the environment first."
}

Write-Host "Wrote $lockPath"
Write-Host "Recreate with: conda create --name $Name --file Environments\$Name.lock.txt"
