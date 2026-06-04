"""Batch inference and CLI for saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import re

from .artifacts import load_artifact
from .preprocessing import encode_categorical_features, prepare_prediction_features
from .schema import validate_columns_against_schema


def prediction_column_for_target(target: str) -> str:
    if target == "TRVmax[kV]":
        return "predicted_TRVmax[kV]"
    safe = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", str(target or "target")).strip("_")
    return f"predicted_{safe or 'target'}"


def predict_batch(df: pd.DataFrame, *, task: str, model_id: str = "latest", model_root: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact = load_artifact(task, model_id, model_root)
    schema = artifact["schema"]
    validation = validate_columns_against_schema(list(df.columns), schema)
    if not validation["ok"]:
        raise ValueError(f"Schema mismatch: missing columns {validation['missing_columns']}")
    X, prep = prepare_prediction_features(
        df,
        schema["features"],
        numeric_only=not (schema.get("engine") == "pycaret" or schema.get("domain") == "motor_noise"),
    )
    categorical_levels = schema.get("categorical_levels") if isinstance(schema.get("categorical_levels"), dict) else {}
    if categorical_levels:
        X, encoded_levels = encode_categorical_features(X, categorical_levels)
        prep["categorical_levels"] = encoded_levels
    if schema.get("domain") == "motor_noise":
        X = X.apply(pd.to_numeric, errors="raise")
    model_features = schema.get("model_features")
    if isinstance(model_features, list) and len(model_features) == len(X.columns):
        X = X.copy()
        X.columns = [str(col) for col in model_features]
    model = artifact["model"]
    output = df.copy()
    if task == "classification":
        pred = model.predict(X).astype(int)
        output["prediction"] = pred
        output["prediction_label"] = ["차단성공 Class 1" if int(value) == 1 else "차단실패 Class 0" for value in pred]
        try:
            proba = model.predict_proba(X)
            output["probability_failure"] = proba[:, 0]
            output["probability_success"] = proba[:, 1]
        except Exception:
            output["probability_failure"] = pd.NA
            output["probability_success"] = pd.NA
    elif task == "regression":
        target = str(schema.get("target") or "TRVmax[kV]")
        output[prediction_column_for_target(target)] = model.predict(X)
    else:
        raise ValueError(f"Unsupported task: {task}")
    summary = {
        "ok": True,
        "task": task,
        "model_id": model_id,
        "rows": int(len(output)),
        "ignored_or_extra_columns": prep["extra_columns"],
        "schema_validation": validation,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HD serving batch inference.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--model", default="latest", help="Model id or artifact directory name under models/<task>")
    parser.add_argument("--input", required=True, help="Input Excel file")
    parser.add_argument("--output", required=True, help="Output .xlsx or .csv")
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    result, summary = predict_batch(df, task=args.task, model_id=args.model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        result.to_csv(out, index=False)
    else:
        result.to_excel(out, index=False)
    print(summary)


if __name__ == "__main__":
    main()
