"""Dataset type inference and schema validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import (
    CLASSIFICATION_IGNORED_COLUMNS,
    MOTOR_NOISE_DIRECTION_COLUMNS,
    MOTOR_NOISE_IGNORED_COLUMNS,
    MOTOR_NOISE_TARGET,
    REGRESSION_IGNORED_COLUMNS,
)


def infer_dataset_type(columns: list[str]) -> str:
    colset = set(columns)
    has_sound_mean = MOTOR_NOISE_TARGET in colset
    has_sound_directions = any(col in colset for col in MOTOR_NOISE_DIRECTION_COLUMNS)
    if has_sound_mean and has_sound_directions:
        return "motor_noise_regression_training"
    has_result = "Result" in colset
    has_trv = "TRVmax[kV]" in colset
    has_reg_meta = {"Time", "CZM", "Test"}.issubset(colset)
    if has_result and has_trv and has_reg_meta:
        return "regression_training"
    if has_result and has_trv:
        return "classification_training"
    return "prediction_input_or_unknown"


def target_distribution(df: pd.DataFrame, dataset_type: str) -> dict[str, Any]:
    if dataset_type == "classification_training" and "Result" in df.columns:
        label = (df["Result"] == 1.0).astype(int)
        return {
            "result_distribution": {str(k): int(v) for k, v in df["Result"].value_counts(dropna=False).sort_index().items()},
            "label_distribution": {str(k): int(v) for k, v in label.value_counts(dropna=False).sort_index().items()},
            "positive_class_rule": "Result == 1.0 -> 1 else 0",
        }
    if dataset_type == "regression_training" and "TRVmax[kV]" in df.columns:
        target = pd.to_numeric(df["TRVmax[kV]"], errors="coerce")
        return {"target": "TRVmax[kV]", "target_summary": target.describe().to_dict()}
    if dataset_type == "motor_noise_regression_training" and MOTOR_NOISE_TARGET in df.columns:
        target = pd.to_numeric(df[MOTOR_NOISE_TARGET], errors="coerce")
        directions = {
            col: pd.to_numeric(df[col], errors="coerce").describe().to_dict()
            for col in MOTOR_NOISE_DIRECTION_COLUMNS
            if col in df.columns
        }
        return {
            "target": MOTOR_NOISE_TARGET,
            "target_summary": target.describe().to_dict(),
            "excluded_sound_direction_summary": directions,
        }
    return {}


def validate_columns_against_schema(columns: list[str], schema: dict[str, Any]) -> dict[str, Any]:
    features = [str(col) for col in schema.get("features", [])]
    ignored = [str(col) for col in schema.get("ignored_columns", [])]
    colset = set(columns)
    missing = [col for col in features if col not in colset]
    extra = [col for col in columns if col not in set(features) and col not in set(ignored)]
    ignored_present = [col for col in ignored if col in colset]
    return {
        "ok": not missing,
        "features": features,
        "missing_columns": missing,
        "extra_columns": extra,
        "ignored_columns": ignored,
        "ignored_present": ignored_present,
        "ordered_feature_columns": [col for col in features if col in colset],
    }


def expected_ignored_columns(task: str) -> list[str]:
    if task == "classification":
        return CLASSIFICATION_IGNORED_COLUMNS
    if task == "regression":
        return REGRESSION_IGNORED_COLUMNS
    if task == "motor_noise_regression":
        return MOTOR_NOISE_IGNORED_COLUMNS
    return []
