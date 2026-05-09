param (
    [string]$Stage = "both"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Invoking WSL training pipeline..." -ForegroundColor Cyan

# Replace backslashes with forward slashes, drop the drive letter, and format for WSL
# Example: D:\Projects\Emotion Detection -> /mnt/d/Projects/Emotion Detection
$drive = $scriptDir.Substring(0, 1).ToLower()
$pathOpts = $scriptDir.Substring(2).Replace('\', '/')
$wslPath = "/mnt/$drive$pathOpts"

# Make sure train_wsl.sh has Unix line endings and execute it
wsl bash -c "cd '$wslPath' && sed -i 's/\r`$//' train_wsl.sh && bash train_wsl.sh --stage $Stage"
