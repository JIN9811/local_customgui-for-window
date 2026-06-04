# Codex Instructions

This folder is a standalone HD Hyundai Electric local ML serving app. Treat it as independent from `/home/jin/autonomous_researcher`.

## Work Rules

1. Keep training, preprocessing, schema validation, inference, explanations, and deterministic tools under `src/hd_serving/`.
2. Do not import modules from `autonomous_researcher`.
3. Preserve both local LLM backend paths:
   - Ollama: `POST {base_url}/api/chat`
   - vLLM: `POST {base_url}/chat/completions`
4. Never use the LLM as a predictor. Prediction must come from saved ML artifacts.
5. Preserve notebook parity:
   - classification: `Result == 1.0 -> label 1`, else `0`; drop `Result`, `TRVmax[kV]`.
   - regression: target `TRVmax[kV]`; drop `Time`, `Result`, `CZM`, `Test`, `TRVmax[kV]`.
6. Do not execute `serving/HD_LLM.ipynb` at runtime.
7. Do not hardcode PPT performance values. Metrics must come from current training runs.
8. If changing Streamlit UI, keep `streamlit_app.py`, README, requirements, and tests synchronized.
9. If changing the legacy dependency-free chat GUI, keep `app.py`, `index.html`, `static/app.js`, and `static/style.css` synchronized.

## Validation

Run these checks after edits:

```bash
cd /home/jin/local_customgui
source .venv/bin/activate
python -m py_compile app.py streamlit_app.py src/hd_serving/*.py
node --check static/app.js
bash -n scripts/run.sh scripts/run_streamlit.sh
pytest
```

Smoke train/inference when touching ML code:

```bash
python -m hd_serving.train_classification --input data/raw/class_extracted.xlsx
python -m hd_serving.train_regression --input data/raw/Reg_extracted.xlsx
python -m hd_serving.inference --task classification --model latest --input data/raw/class_extracted.xlsx --output /tmp/hd_class_predictions.xlsx
```

Only run an actual LLM chat call when Ollama or vLLM is already serving a model.

## File Roles

- `streamlit_app.py`: HD Excel upload, validation, training, prediction, model management, and LLM-tool chat GUI.
- `src/hd_serving/`: deterministic ML package and tool layer.
- `models/`: saved joblib/schema/metrics/model_card artifacts.
- `data/raw/`: local example Excel files.
- `docs/`: project summary and notebook source notes.
- `scripts/`: Linux/macOS launch scripts.
- `app.py`: legacy Python HTTP server and Ollama/vLLM bridge.
- `index.html`, `static/app.js`, `static/style.css`: legacy browser chat GUI.
- `config.json`: local backend/model/system-prompt defaults.
