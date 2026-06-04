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

function Stop-ExistingExe {
  param([Parameter(Mandatory = $true)][string]$ProcessName)

  $processes = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
  if ($processes) {
    Write-Host "Stopping existing $ProcessName process before rebuild..."
    $processes | Stop-Process -Force
    Start-Sleep -Seconds 1
  }
}

if (-not (Test-Path ".\streamlit_app.py")) {
  throw "Could not find the project folder containing streamlit_app.py."
}

if (-not (Test-Path ".\Logo\logo_aim4lab.png")) {
  throw "Could not find AIM4LAB logo: .\Logo\logo_aim4lab.png"
}

if (-not (Test-Path ".\Icon\aim4lab_app_icon.ico")) {
  throw "Could not find AIM4LAB app icon: .\Icon\aim4lab_app_icon.ico"
}

if (-not (Test-Path ".\Icon\aim4lab_app_icon.png")) {
  throw "Could not find AIM4LAB app icon PNG: .\Icon\aim4lab_app_icon.png"
}

$CondaExe = "$env:UserProfile\miniconda3\Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
  throw "Could not find conda.exe: $CondaExe"
}

$LogoData = "$ProjectRoot\Logo\logo_aim4lab.png;Logo"
$IconData = "$ProjectRoot\Icon\aim4lab_app_icon.png;Icon"
$IconPath = "$ProjectRoot\Icon\aim4lab_app_icon.ico"

Stop-ExistingExe "LocalCustomGUI-Manager"
Invoke-Checked $CondaExe @("run", "-n", "local_customgui_windows", "python", "-m", "pip", "install", "pyinstaller", "pystray", "pillow")
Invoke-Checked $CondaExe @(
  "run", "-n", "local_customgui_windows", "python", "-m", "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onefile",
  "--windowed",
  "--workpath", ".\packaging\windows\build",
  "--specpath", ".\packaging\windows\specs",
  "--distpath", ".",
  "--name", "LocalCustomGUI-Manager",
  "--icon", $IconPath,
  "--hidden-import", "tkinter.messagebox",
  "--hidden-import", "pystray._win32",
  "--add-data", $LogoData,
  "--add-data", $IconData,
  ".\packaging\windows\windows_manager_launcher.py"
)

Write-Host ""
Write-Host "Manager EXE created: $ProjectRoot\LocalCustomGUI-Manager.exe"
