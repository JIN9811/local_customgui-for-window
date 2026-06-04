# 형상/시험 변수 기반 고압차단기 성능 예측

이 프로그램은 `serving/HD_LLM.ipynb`의 수치형 classification/regression 로직을 재현 가능한 Python package와 Streamlit GUI로 분리한 로컬 데이터 분석 앱이다.

핵심 원칙:

- ML 예측은 저장된 model artifact가 수행한다.
- LLM은 tool 호출과 결과 설명만 담당한다.
- Excel 전체 row를 LLM prompt에 넣지 않는다.
- schema 검증 없이 prediction하지 않는다.
- notebook은 참고 자료이며 runtime에서 직접 실행하지 않는다.
- PPT 과거 성능값을 하드코딩하지 않고 현재 데이터로 다시 계산한다.

## 파일 역할

- `serving/HD_LLM.ipynb`: 원본 PyCaret notebook reference.
- `serving/class_extracted.xlsx`: classification 예제 데이터.
- `serving/Reg_extracted.xlsx`: regression 예제 데이터.
- `serving/HD현대일렉트릭_과제중간발표.pptx`: 프로젝트 개요 reference.
- `streamlit_app.py`: Streamlit GUI entrypoint.
- `src/hd_serving/`: preprocessing, schema, training, inference, tools, LLM router package.
- `src/hd_serving/pycaret_worker.py`: Python 3.11 PyCaret notebook-parity worker.
- `models/`: 저장된 model artifact.
- `data/raw/`: 예제 Excel 복사본.
- `docs/`: 프로젝트 요약과 notebook source notes.
- `app.py`: 기존 dependency-free Ollama/vLLM browser chat bridge. Legacy로 유지.

## Windows PowerShell 빠른 설치

아래 블록은 **README.md와 `streamlit_app.py`가 있는 프로젝트 폴더에서 PowerShell을 열고 그대로 붙여넣어 실행**한다. Miniconda 환경 `local_customgui_windows`를 만들고 그 안에 프로그램을 설치한 뒤 실행한다.

```powershell
$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$EnvName = "local_customgui_windows"

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
  Invoke-Checked $OllamaExe pull gemma4:e2b
} else {
  Write-Warning "Ollama 실행 파일을 찾지 못했습니다. 앱은 실행되지만 LLM을 쓰려면 Ollama 설치와 모델 다운로드가 필요합니다."
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

다음부터는 설치 과정을 반복하지 않고 아래만 실행하면 된다.

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

## PyCaret 확인

Windows 빠른 설치는 PyCaret을 같은 Miniconda 환경에 설치하고 기본 학습 엔진으로 사용한다. 설치 확인은 아래처럼 한다.

```powershell
conda run -n local_customgui_windows python -c "import pycaret; print(pycaret.__version__)"
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

## Ollama / vLLM 설정

`.env.example` 기준:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=local-model
VLLM_API_KEY=EMPTY
```

Ollama 예시:

```bash
ollama serve
ollama pull gemma4:e2b
```

Docker Ollama를 쓰는 경우 컨테이너가 `11434:11434`로 노출되어 있으면 GUI의 Base URL은 그대로 `http://127.0.0.1:11434`를 사용한다. GUI에서 Base URL 아래 `Connect`를 누르면 현재 Ollama의 `/api/tags` 모델 목록을 읽어 `Model` 선택 목록에 표시한다.

vLLM은 OpenAI-compatible endpoint가 필요하다:

```text
http://127.0.0.1:8000/v1/chat/completions
http://127.0.0.1:8000/v1/models
```

Windows에서 Ollama를 사용하려면 Ollama 앱을 실행한 뒤 모델을 받는다.

```powershell
ollama pull gemma4:e2b
```

vLLM은 OpenAI-compatible endpoint가 필요하며, 기본 config는 일반 로컬 endpoint를 사용한다.

## 사용 흐름

1. 메인 채팅 화면에서 `.xlsx` 또는 `.xls` 파일 업로드.
2. 업로드 즉시 Ollama tool-calling agent가 데이터 검증, 데이터 유형 판별, 모델 학습, schema 검증, batch prediction tool 호출을 결정하고 실행.
3. 각 단계 결과는 채팅 메시지의 접기/펼치기 블록으로 표시하며, `LLM Reasoning`은 Ollama가 실제 반환한 `thinking`만 표시하고 deterministic 실행 내역은 `Execution trace`로 분리.
4. 예측 결과 메시지를 펼치면 preview와 CSV/XLSX 다운로드 버튼을 제공.
5. 이후 “방금 올린 엑셀 요약해줘”, “분류 모델 다시 학습해줘”, “예측 결과 설명해줘”, “성능 요약해줘”처럼 요청하면 LLM workflow가 deterministic tool을 호출해 답변.
6. 모델 artifact 적용, metrics/schema 확인, 삭제는 업로드 영역 아래 `Model Management` 패널에서 수행.

## Notebook parity

Classification:

- `Result == 1.0 -> label 1`, else `0`.
- target: `label`.
- feature 제외: `Result`, `TRVmax[kV]`.
- `dropna()` 수행 및 dropped row 수 기록.
- runtime: PyCaret `setup(..., session_id=0, train_size=0.9)` 후 `compare_models()`.

Regression:

- target: `TRVmax[kV]`.
- feature 제외: `Time`, `Result`, `CZM`, `Test`, `TRVmax[kV]`.
- `dropna()` 수행 및 dropped row 수 기록.
- runtime: PyCaret `setup(..., session_id=0, train_size=0.9)` 후 `compare_models()`.

## CLI

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

## Artifact 구조

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

`latest`는 symlink를 우선 사용하고, 실패하면 directory copy로 관리한다.

## Schema mismatch 처리

Prediction은 저장된 `schema.json`의 `features` 목록을 기준으로 수행한다.

- missing feature가 있으면 예측하지 않는다.
- extra column은 무시할 수 있지만, ignored/extra column 목록을 사용자에게 보여준다.
- target/ignored column은 prediction input에 있어도 feature로 쓰지 않는다.

## LLM 안전 원칙

- LLM은 예측값, 확률, metric, explanation 값을 임의 생성하지 않는다.
- 데이터 요약/검증/학습/예측/설명은 deterministic tool 결과를 기반으로 한다.
- LLM output을 Python 코드로 실행하지 않는다.
- shell command tool은 제공하지 않는다.
- 로컬 LLM 실패 시 cloud fallback을 자동 사용하지 않는다.

## Legacy Browser GUI

기존 간단 채팅 GUI도 유지된다.

```powershell
conda run -n local_customgui_windows python app.py --host 127.0.0.1 --port 8790 --open
```

접속:

```text
http://127.0.0.1:8790
```

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
bash -n run.sh run_streamlit.sh
pytest
```
