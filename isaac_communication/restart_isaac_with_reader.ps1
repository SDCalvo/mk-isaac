# Restart Isaac and Launch Game State Reader Script
# This script stops The Binding of Isaac if it's running, 
# restarts it with debug mode, and launches the reader script

Write-Host "Isaac Game State Reader Launcher" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Get the script's directory path
$scriptDir = Split-Path -Parent $PSCommandPath
$rootDir = Split-Path -Parent $scriptDir

# 1. Stop Isaac if it's running
Write-Host "Checking if Isaac is running..." -ForegroundColor Yellow
$isaacProcesses = Get-Process | Where-Object { $_.Name -like "*isaac*" }

if ($isaacProcesses) {
    Write-Host "Found Isaac processes. Stopping them..." -ForegroundColor Yellow
    $isaacProcesses | ForEach-Object {
        Write-Host "  Stopping $($_.Name) (ID: $($_.Id))"
        $_ | Stop-Process -Force
    }
    Write-Host "All Isaac processes stopped." -ForegroundColor Green
} else {
    Write-Host "No Isaac processes found." -ForegroundColor Green
}

# 2. Wait a moment to ensure processes are fully stopped
Write-Host "Waiting for processes to fully terminate..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 3. Check if any Python reader processes are running and stop them
$pythonProcesses = Get-Process | Where-Object { $_.Name -eq "python" }
if ($pythonProcesses) {
    Write-Host "Found Python processes. Checking if they're related to the reader..." -ForegroundColor Yellow
    $pythonProcesses | ForEach-Object {
        # Try to get command line to check if it's our reader script
        try {
            $cmdLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
            if ($cmdLine -like "*isaac_game_state_reader.py*") {
                Write-Host "  Stopping Python reader process (ID: $($_.Id))"
                $_ | Stop-Process -Force
            }
        } catch {
            # If we can't get command line, better to be safe
            Write-Host "  Stopping Python process (ID: $($_.Id))"
            $_ | Stop-Process -Force
        }
    }
    Write-Host "Python processes checked and stopped if needed." -ForegroundColor Green
}

# 4. Clean up temporary pipe files
Write-Host "Cleaning up temporary files..." -ForegroundColor Yellow
$tempDir = [System.IO.Path]::GetTempPath()
$inputPipe = Join-Path $tempDir "isaac_input_pipe.txt"
$outputPipe = Join-Path $tempDir "isaac_output_pipe.txt"

if (Test-Path $inputPipe) {
    Remove-Item $inputPipe -Force
    Write-Host "  Removed $inputPipe"
}

if (Test-Path $outputPipe) {
    Remove-Item $outputPipe -Force
    Write-Host "  Removed $outputPipe"
}

# 5. Create output directory if it doesn't exist
$modOutputDir = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader/output"
if (-not (Test-Path $modOutputDir)) {
    Write-Host "Creating mod output directory..." -ForegroundColor Yellow
    New-Item -Path $modOutputDir -ItemType Directory -Force | Out-Null
    Write-Host "Created $modOutputDir" -ForegroundColor Green
}

# 6. Start Isaac with --luadebug option
Write-Host "Starting Isaac with --luadebug option..." -ForegroundColor Yellow
Start-Process "steam://run/250900//--luadebug"

# 7. Wait for Isaac to start
Write-Host "Waiting for Isaac to start (15 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# 8. Check if Isaac is running
$isaacRunning = Get-Process | Where-Object { $_.Name -like "*isaac*" }
if (-not $isaacRunning) {
    Write-Host "WARNING: Isaac doesn't appear to be running yet. Waiting longer..." -ForegroundColor Red
    Start-Sleep -Seconds 10
}

# 9. Start the reader script
Write-Host "Starting Isaac Game State Reader script..." -ForegroundColor Yellow
$pythonExe = "python"
$readerScriptPath = Join-Path $scriptDir "isaac_game_state_reader.py"

# Start in a new window so this script can finish
Start-Process $pythonExe -ArgumentList $readerScriptPath -NoNewWindow

Write-Host "Done! The reader script is now running in the background." -ForegroundColor Green
Write-Host "NOTE: If you see connection errors, you may need to restart Isaac manually." -ForegroundColor Yellow 