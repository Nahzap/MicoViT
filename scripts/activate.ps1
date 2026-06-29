# Activador del entorno virtual del proyecto MicorizaeVision (PowerShell).
# Uso:
#   cd f:\MicorizaeVision
#   .\scripts\activate.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$activator = Join-Path $root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $activator)) {
    Write-Host "[micorizae] .venv no encontrado. Creándolo..." -ForegroundColor Yellow
    python -m venv (Join-Path $root ".venv")
}

. $activator
Write-Host "[micorizae] entorno activo en $root" -ForegroundColor Green
Write-Host "[micorizae] python: $(python --version)"
