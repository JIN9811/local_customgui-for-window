# Source Notes

## Notebook Logic

`serving/HD_LLM.ipynb` used PyCaret for model exploration. Runtime serving now executes the same PyCaret-style workflow through a local Python 3.11 worker environment because PyCaret 3.3.x does not support Python 3.12.

### Classification

- Read `class_extracted.xlsx`.
- Drop rows with missing values.
- Create `label = 1` only when `Result == 1.0`; otherwise `label = 0`.
- Drop `Result` and `TRVmax[kV]` from features.
- Train/test split follows `train_size=0.9`, `session_id/random_state=0` by default.

### Regression

- Read `Reg_extracted.xlsx`.
- Drop rows with missing values.
- Target is `TRVmax[kV]`.
- Drop `Time`, `Result`, `CZM`, `Test`, and `TRVmax[kV]` from features.
- Train/test split follows `train_size=0.9`, `session_id/random_state=0` by default.

## Runtime Decision

The default runtime engine is PyCaret notebook parity:

- Streamlit server runs in the main `.venv`.
- PyCaret training runs through conda env `local_customgui_pycaret`.
- Classification follows `dropna -> label -> drop Result/TRVmax[kV] -> setup -> compare_models`.
- Regression follows `dropna -> drop Time/Result/CZM/Test -> move first column to end -> setup -> compare_models`.
- The selected PyCaret estimator is stored as a joblib artifact with schema, metrics, compare table, and model card.

Set `HD_SERVING_TRAIN_ENGINE=sklearn` only when a lightweight fallback trainer is needed.
