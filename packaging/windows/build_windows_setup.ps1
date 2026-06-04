$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $FilePath $($Arguments -join ' ')"
  }
}

if (-not (Test-Path ".\streamlit_app.py")) {
  throw "Could not find the project folder containing streamlit_app.py."
}

if (-not (Test-Path ".\Icon\aim4lab_app_icon.ico")) {
  throw "Could not find AIM4LAB app icon: .\Icon\aim4lab_app_icon.ico"
}

$CondaExe = "$env:UserProfile\miniconda3\Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
  throw "Could not find conda.exe: $CondaExe"
}

$IconPath = "$ProjectRoot\Icon\aim4lab_app_icon.ico"

Invoke-Checked $CondaExe @("run", "-n", "local_customgui_windows", "python", "-m", "pip", "install", "pyinstaller")
Invoke-Checked $CondaExe @(
  "run", "-n", "local_customgui_windows", "python", "-m", "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onefile",
  "--console",
  "--workpath", ".\packaging\windows\build",
  "--specpath", ".\packaging\windows\specs",
  "--distpath", ".\dist",
  "--name", "LocalCustomGUI-Setup",
  "--icon", $IconPath,
  ".\packaging\windows\windows_setup_launcher.py"
)

Write-Host ""
Write-Host "Setup EXE created: $ProjectRoot\dist\LocalCustomGUI-Setup.exe"
