$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".\streamlit_app.py")) {
  throw "Could not find the project folder containing streamlit_app.py."
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
  --console `
  --name LocalCustomGUI-Delete `
  .\packaging\windows_delete_launcher.py

Write-Host ""
Write-Host "Delete EXE created: $ProjectRoot\dist\LocalCustomGUI-Delete.exe"
