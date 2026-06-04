$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".\streamlit_app.py")) {
  throw "streamlit_app.py가 있는 프로젝트 폴더를 찾지 못했습니다."
}

$CondaExe = "$env:UserProfile\miniconda3\Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
  throw "conda.exe를 찾지 못했습니다: $CondaExe"
}

& $CondaExe run -n local_customgui_windows python -m pip install pyinstaller
& $CondaExe run -n local_customgui_windows python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --console `
  --name LocalCustomGUI-Windows `
  .\packaging\windows_launcher.py

Write-Host ""
Write-Host "EXE 생성 완료: $ProjectRoot\dist\LocalCustomGUI-Windows.exe"
