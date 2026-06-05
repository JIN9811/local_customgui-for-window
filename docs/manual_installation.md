# 수동 설치 및 개발자 명령

이 문서는 README에서 분리한 수동 설치, 실행, 빌드, 검증 명령 모음입니다. 일반 사용자는 `LocalCustomGUI-Manager.exe`를 사용하는 것을 권장합니다.

## Windows PowerShell 수동 설치

아래 블록은 `README.md`와 `streamlit_app.py`가 있는 프로젝트 폴더에서 PowerShell을 열고 그대로 붙여넣어 실행합니다. Miniconda 환경 `local_customgui_windows`를 만들고 프로그램을 설치한 뒤 Streamlit 서버를 실행합니다.

```powershell
$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$EnvName = "local_customgui_windows"
$TotalRamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$OllamaModel = if ($TotalRamGB -ge 31) { "gemma4:e4b" } else { "gemma4:e2b" }
Write-Host "Detected RAM: $TotalRamGB GB"
Write-Host "Selected Ollama model: $OllamaModel"
Write-Host "Recommendation: 16GB RAM -> gemma4:e2b, 32GB-class+ RAM -> gemma4:e4b"

function Invoke-Checked {
  $FilePath = $args[0]
  $Arguments = @($args | Select-Object -Skip 1)
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "명령 실행 실패: $FilePath $($Arguments -join ' ')"
  }
}

if (-not (Test-Path ".\streamlit_app.py")) {
  throw "streamlit_app.py가 있는 프로젝트 폴더에서 실행하세요."
}

if (Get-Command winget -ErrorAction SilentlyContinue) {
  if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    winget install -e --id Anaconda.Miniconda3 --accept-package-agreements --accept-source-agreements
  }
  if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  }
} else {
  Write-Warning "winget이 없어 Miniconda/Ollama 자동 설치를 건너뜁니다. Miniconda를 설치한 뒤 다시 실행하세요."
}

$CondaExe = $null
$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($CondaCommand) {
  $CondaExe = $CondaCommand.Source
}
if (-not $CondaExe) {
  $CondaCandidates = @(
    "$env:UserProfile\miniconda3\Scripts\conda.exe",
    "$env:LocalAppData\miniconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\Scripts\conda.exe",
    "$env:ProgramFiles\Miniconda3\Scripts\conda.exe"
  )
  $CondaExe = $CondaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $CondaExe) {
  throw "conda.exe를 찾지 못했습니다. 새 PowerShell을 열어 같은 명령을 다시 실행하거나 Miniconda를 설치하세요."
}

& $CondaExe run -n $EnvName python --version | Out-Null
if ($LASTEXITCODE -ne 0) {
  Invoke-Checked $CondaExe create -y -n $EnvName --override-channels -c conda-forge python=3.11 pip
}

Invoke-Checked $CondaExe run -n $EnvName python -m pip install -U pip
Invoke-Checked $CondaExe run -n $EnvName python -m pip install -r requirements.txt
Invoke-Checked $CondaExe run -n $EnvName python -m pip install -r requirements-pycaret.txt
Invoke-Checked $CondaExe run -n $EnvName python -m pip install -e .

if (-not (Test-Path ".\.env")) {
  Copy-Item ".\.env.example" ".\.env"
}
$EnvContent = Get-Content ".\.env" -ErrorAction SilentlyContinue
if ($EnvContent -match "^OLLAMA_MODEL=") {
  $EnvContent = $EnvContent | ForEach-Object { if ($_ -match "^OLLAMA_MODEL=") { "OLLAMA_MODEL=$OllamaModel" } else { $_ } }
} else {
  $EnvContent = @($EnvContent) + "OLLAMA_MODEL=$OllamaModel"
}
$EnvContent | Set-Content ".\.env" -Encoding UTF8

$ConfigPath = ".\config.json"
$Config = if (Test-Path $ConfigPath) { Get-Content $ConfigPath -Raw | ConvertFrom-Json } else { "{}" | ConvertFrom-Json }
$Config | Add-Member -Force -MemberType NoteProperty -Name "default_backend" -Value "ollama"
if (-not $Config.PSObject.Properties["ollama"]) {
  $Config | Add-Member -Force -MemberType NoteProperty -Name "ollama" -Value ([pscustomobject]@{})
}
$Config.ollama | Add-Member -Force -MemberType NoteProperty -Name "base_url" -Value "http://127.0.0.1:11434"
$Config.ollama | Add-Member -Force -MemberType NoteProperty -Name "model" -Value $OllamaModel
if (-not $Config.ollama.PSObject.Properties["num_ctx"]) {
  $Config.ollama | Add-Member -Force -MemberType NoteProperty -Name "num_ctx" -Value 16384
}
$Config | ConvertTo-Json -Depth 12 | Set-Content $ConfigPath -Encoding UTF8

$OllamaExe = $null
$OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if ($OllamaCommand) {
  $OllamaExe = $OllamaCommand.Source
}
if (-not $OllamaExe) {
  $OllamaCandidates = @(
    "$env:LocalAppData\Programs\Ollama\ollama.exe",
    "$env:ProgramFiles\Ollama\ollama.exe"
  )
  $OllamaExe = $OllamaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if ($OllamaExe) {
  & $OllamaExe list | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
  }
  Invoke-Checked $OllamaExe pull $OllamaModel
} else {
  Write-Warning "Ollama 실행 파일을 찾지 못했습니다. 앱은 실행되지만 LLM 연결에는 Ollama 설치와 모델 다운로드가 필요합니다."
}

$StreamlitConfigDir = Join-Path $env:UserProfile ".streamlit"
New-Item -ItemType Directory -Force -Path $StreamlitConfigDir | Out-Null
@"
[server]
headless = true
showEmailPrompt = false

[browser]
gatherUsageStats = false
"@ | Set-Content -LiteralPath (Join-Path $StreamlitConfigDir "config.toml") -Encoding UTF8

$env:STREAMLIT_SERVER_HEADLESS = "true"
$env:STREAMLIT_SERVER_SHOW_EMAIL_PROMPT = "false"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

Write-Host ""
Write-Host "설치 완료. Streamlit을 실행합니다: http://127.0.0.1:8791"
Invoke-Checked $CondaExe run -n $EnvName python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8791 --server.headless true --server.showEmailPrompt false --browser.gatherUsageStats false
```

## Windows 수동 실행

설치가 끝난 뒤에는 아래 명령만 실행하면 됩니다.

```powershell
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_HEADLESS = "true"
$env:STREAMLIT_SERVER_SHOW_EMAIL_PROMPT = "false"
& "$env:UserProfile\miniconda3\Scripts\conda.exe" run -n local_customgui_windows python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8791 --server.headless true --server.showEmailPrompt false --browser.gatherUsageStats false
```

접속 주소:

```text
http://127.0.0.1:8791
```

## Windows 수동 중지

Manager EXE를 사용하는 경우 `Run` 탭의 `Stop App` 또는 트레이 메뉴의 `Quit Server`를 누르면 됩니다. 이 동작은 Streamlit 프로세스 트리, 같은 프로젝트의 백그라운드 Streamlit 프로세스, Ollama에 로딩된 AIM4LAB 모델을 함께 정리합니다.

PowerShell에서 수동 실행한 서버를 직접 정리해야 할 때는 아래 명령을 사용합니다.

```powershell
$ProjectRoot = (Get-Location).Path.ToLowerInvariant()
$EnvName = "local_customgui_windows"

Get-CimInstance Win32_Process | Where-Object {
  $name = ([string]$_.Name).ToLowerInvariant()
  $runtimeProcess = @("python.exe", "pythonw.exe", "conda.exe", "streamlit.exe") -contains $name
  $text = (([string]$_.CommandLine) + " " + ([string]$_.ExecutablePath)).ToLowerInvariant()
  $appMatch = $text.Contains("streamlit_app.py") -and $text.Contains($ProjectRoot)
  $envMatch = $text.Contains($EnvName) -and $text.Contains("streamlit")
  $runtimeProcess -and ($appMatch -or $envMatch)
} | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

$OllamaExe = "$env:LocalAppData\Programs\Ollama\ollama.exe"
if (Test-Path $OllamaExe) {
  & $OllamaExe stop gemma4:e2b 2>$null
  & $OllamaExe stop gemma4:e4b 2>$null
  & $OllamaExe ps
}
```

## Manager EXE 빌드

설치, 실행, 삭제를 한 창에서 처리하는 Manager EXE를 빌드합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build_windows_manager.ps1
```

생성 파일:

```text
LocalCustomGUI-Manager.exe
```

Manager EXE는 AIM4LAB 로고와 아이콘을 포함하며 `Install`, `Run`, `Uninstall` 탭을 제공합니다. `Stop App`은 Streamlit만 닫는 버튼이 아니라, 앱 런타임 전체를 정리하는 버튼입니다.

콘솔 설치 런처를 직접 사용할 때는 모델 옵션을 반복하거나 `both`를 지정할 수 있습니다.

```powershell
# RAM 기준 추천 모델 1개 자동 선택
python packaging\windows\windows_setup_launcher.py

# e2b와 e4b를 둘 다 다운로드하고 기본 모델은 RAM 추천값으로 설정
python packaging\windows\windows_setup_launcher.py --ollama-model both

# 명시적으로 두 모델 선택
python packaging\windows\windows_setup_launcher.py --ollama-model gemma4:e2b --ollama-model gemma4:e4b
```

## Windows 프로그램 추가/제거 등록

설치 완료 시 Manager가 현재 사용자 기준으로 아래 항목을 등록합니다.

```text
HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\AIM4LAB_LocalCustomGUI
```

수동 등록:

```powershell
.\LocalCustomGUI-Manager.exe --register-uninstall
```

수동 등록 해제:

```powershell
.\LocalCustomGUI-Manager.exe --unregister-uninstall
```

Windows 설정에서 `AIM4LAB LocalCustomGUI` 제거를 누르면 아래 명령이 실행되어 Manager의 Uninstall 탭으로 이동합니다.

```text
LocalCustomGUI-Manager.exe --uninstall
```

## Ollama / vLLM 설정

기본 `.env` 값:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
# RAM 16GB 권장
OLLAMA_MODEL=gemma4:e2b
# RAM 32GB 이상 권장
# OLLAMA_MODEL=gemma4:e4b
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=local-model
VLLM_API_KEY=EMPTY
```

Ollama 모델 수동 다운로드:

```powershell
# RAM 16GB 권장
ollama pull gemma4:e2b

# RAM 32GB 이상 권장
ollama pull gemma4:e4b
```

두 모델을 모두 준비하려면 위 두 명령을 모두 실행합니다. 앱 기본 모델은 `.env`의 `OLLAMA_MODEL`과 `config.json`의 `ollama.model` 값으로 결정됩니다.

Ollama Desktop에서 Context length는 16k 근처로 맞추는 것을 권장합니다. Streamlit 앱의 기본 `Context Length`도 `16384`입니다.

vLLM은 OpenAI-compatible endpoint가 필요합니다.

```text
http://127.0.0.1:8000/v1/chat/completions
http://127.0.0.1:8000/v1/models
```

## Linux/macOS 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

실행:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8791
```

## CLI 예시

PowerShell:

```powershell
conda run -n local_customgui_windows python -m hd_serving.train_classification --input data\raw\class_extracted.xlsx
conda run -n local_customgui_windows python -m hd_serving.train_regression --input data\raw\Reg_extracted.xlsx
conda run -n local_customgui_windows python -m hd_serving.inference --task classification --model latest --input data\raw\class_extracted.xlsx --output predictions_classification.xlsx
conda run -n local_customgui_windows python -m hd_serving.inference --task regression --model latest --input data\raw\Reg_extracted.xlsx --output predictions_regression.xlsx
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m hd_serving.train_classification --input data/raw/class_extracted.xlsx
python -m hd_serving.train_regression --input data/raw/Reg_extracted.xlsx
python -m hd_serving.inference --task classification --model latest --input data/raw/class_extracted.xlsx --output predictions_classification.xlsx
python -m hd_serving.inference --task regression --model latest --input data/raw/Reg_extracted.xlsx --output predictions_regression.xlsx
```

## 모델 artifact 구조

```text
models/
  classification/
    latest/
      model.joblib
      schema.json
      metrics.json
      model_card.md
    YYYYMMDD_HHMMSS/
      ...
  regression/
    latest/
      ...
```

`latest`는 가능하면 symlink로 관리하고, 실패하면 directory copy로 관리합니다.

## 검증

PowerShell:

```powershell
conda run -n local_customgui_windows python -m py_compile app.py streamlit_app.py src\hd_serving\__init__.py src\hd_serving\artifacts.py src\hd_serving\constants.py src\hd_serving\data_loader.py src\hd_serving\explanation.py src\hd_serving\inference.py src\hd_serving\llm_client.py src\hd_serving\nemoclaw_vllm_runtime.py src\hd_serving\orchestrator.py src\hd_serving\preprocessing.py src\hd_serving\pycaret_bridge.py src\hd_serving\pycaret_worker.py src\hd_serving\schema.py src\hd_serving\tools.py src\hd_serving\training.py src\hd_serving\train_classification.py src\hd_serving\train_regression.py
conda run -n local_customgui_windows python -m pytest
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m py_compile app.py streamlit_app.py src/hd_serving/*.py
node --check static/app.js
bash -n scripts/run.sh scripts/run_streamlit.sh
pytest
```
