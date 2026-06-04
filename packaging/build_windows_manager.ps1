$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".\streamlit_app.py")) {
  throw "Could not find the project folder containing streamlit_app.py."
}

if (-not (Test-Path ".\Logo\logo_aim4lab.png")) {
  throw "Could not find AIM4LAB logo: .\Logo\logo_aim4lab.png"
}

$CondaExe = "$env:UserProfile\miniconda3\Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
  throw "Could not find conda.exe: $CondaExe"
}

& $CondaExe run -n local_customgui_windows python -m pip install pyinstaller
& $CondaExe run -n local_customgui_windows python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name LocalCustomGUI-Manager `
  --add-data "Logo\logo_aim4lab.png;Logo" `
  .\packaging\windows_manager_launcher.py

Write-Host ""
Write-Host "Manager EXE created: $ProjectRoot\dist\LocalCustomGUI-Manager.exe"
