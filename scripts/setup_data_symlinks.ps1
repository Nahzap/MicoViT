# scripts/setup_data_symlinks.ps1
#
# Crea una estructura organizada de enlaces simbolicos hacia las imagenes
# originales y sus anotaciones, SIN duplicar bytes en disco.
#
# Resultado (ejemplo):
#     data_links/
#       AM/
#         train/
#           10E_2L_E.jpg  -> ..\..\..\Data\am\am\train\10E_2L_E_Default_Extended.jpg
#           10E_2L_E.csv  -> ..\..\..\Data\am\am\train\10E_2L_E_..._cnn_1_annotations.csv
#         test/
#           ...
#       ERM/
#         train/ ...
#         test/  ...
#
# Requisitos:
#   - Ejecutar PowerShell como administrador, O
#   - Tener habilitado "Developer Mode" en Windows (Settings > For developers).
#
# Uso:
#     cd f:\MicorizaeVision
#     .\scripts\setup_data_symlinks.ps1
#     .\scripts\setup_data_symlinks.ps1 -Clean   # elimina y recrea
#
param(
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$linksRoot = Join-Path $root "data_links"
$manifestPath = Join-Path $root "manifests\manifest_images.csv"

if (-not (Test-Path $manifestPath)) {
    throw "No existe $manifestPath. Corre 'python -m micorizae ingest' primero."
}

if ($Clean -and (Test-Path $linksRoot)) {
    Write-Host "[symlinks] Limpiando $linksRoot..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $linksRoot
}

if (-not (Test-Path $linksRoot)) {
    New-Item -ItemType Directory -Path $linksRoot | Out-Null
}

$rows = Import-Csv -Path $manifestPath
$total = $rows.Count
$created = 0
$skipped = 0

Write-Host "[symlinks] Procesando $total filas del manifest..." -ForegroundColor Cyan

foreach ($row in $rows) {
    if ($row.status -ne 'ok') { continue }

    $lineage = $row.lineage
    $split = $row.split
    $stem = $row.image_stem
    $imgRel = $row.image_path -replace '/', '\'
    $annRel = $row.annotation_path -replace '/', '\'

    $imgSrc = Join-Path $root $imgRel
    $annSrc = Join-Path $root $annRel
    $imgExt = [System.IO.Path]::GetExtension($imgSrc)

    $targetDir = Join-Path $linksRoot (Join-Path $lineage $split)
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    $imgLink = Join-Path $targetDir ($stem + $imgExt)
    $annLink = Join-Path $targetDir ($stem + ".csv")

    foreach ($pair in @(@($imgSrc, $imgLink), @($annSrc, $annLink))) {
        $src, $lnk = $pair
        if (-not (Test-Path $src)) {
            Write-Host "[symlinks] WARN: source missing $src" -ForegroundColor Yellow
            continue
        }
        if (Test-Path $lnk) {
            $skipped++
            continue
        }
        try {
            New-Item -ItemType SymbolicLink -Path $lnk -Target $src -ErrorAction Stop | Out-Null
            $created++
        } catch {
            Write-Host "[symlinks] ERROR creando $lnk : $_" -ForegroundColor Red
            Write-Host "[symlinks] Solucion: ejecuta como Admin o habilita Developer Mode." -ForegroundColor Red
            throw
        }
    }
}

Write-Host ""
Write-Host "[symlinks] OK -> $linksRoot" -ForegroundColor Green
Write-Host "  enlaces creados: $created"
Write-Host "  ya existian   : $skipped"
$sizeMB = 0
if (Test-Path $linksRoot) {
    $sizeMB = (Get-ChildItem $linksRoot -Recurse -Force | Measure-Object Length -Sum).Sum / 1MB
}
Write-Host ("  tamanio total : {0:N3} MB (deberia ser ~0)" -f $sizeMB)
