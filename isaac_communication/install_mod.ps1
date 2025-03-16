# Install The Binding of Isaac GameStateReader Mod
# This script installs the mod files to the correct location

Write-Host "Isaac Game State Reader Mod Installer" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Default Isaac mods directory
$defaultModsDir = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods"
$modsDir = Read-Host "Enter Isaac mods directory (default: $defaultModsDir)"

if ([string]::IsNullOrWhiteSpace($modsDir)) {
    $modsDir = $defaultModsDir
}

# Validate the mods directory
if (!(Test-Path $modsDir)) {
    Write-Host "Error: Directory $modsDir does not exist." -ForegroundColor Red
    exit 1
}

# Create mod directory if it doesn't exist
$modDir = Join-Path $modsDir "IsaacGameStateReader"
if (!(Test-Path $modDir)) {
    Write-Host "Creating mod directory: $modDir" -ForegroundColor Yellow
    New-Item -Path $modDir -ItemType Directory -Force | Out-Null
}

# Create output directory for logs and temporary files
$outputDir = Join-Path $modDir "output"
if (!(Test-Path $outputDir)) {
    Write-Host "Creating output directory: $outputDir" -ForegroundColor Yellow
    New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
}

# Copy mod files
$rootDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$modSourceDir = Join-Path $rootDir "isaac_mod"

Write-Host "Copying mod files from $modSourceDir to $modDir" -ForegroundColor Yellow
Copy-Item -Path (Join-Path $modSourceDir "main.lua") -Destination (Join-Path $modDir "main.lua") -Force
Copy-Item -Path (Join-Path $modSourceDir "metadata.xml") -Destination (Join-Path $modDir "metadata.xml") -Force

Write-Host "Mod installation complete!" -ForegroundColor Green
Write-Host "To use the mod:" -ForegroundColor Yellow
Write-Host "1. Launch Isaac with the --luadebug option in Steam" -ForegroundColor Yellow
Write-Host "2. Run the restart_isaac_with_reader.ps1 script to start both the game and reader" -ForegroundColor Yellow 