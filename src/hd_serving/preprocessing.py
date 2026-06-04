"""Notebook-equivalent preprocessing logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    CLASSIFICATION_IGNORED_COLUMNS,
    CLASSIFICATION_SOURCE_TARGET,
    MOTOR_NOISE_DIRECTION_COLUMNS,
    MOTOR_NOISE_IGNORED_COLUMNS,
    MOTOR_NOISE_TARGET,
    REGRESSION_IGNORED_COLUMNS,
    REGRESSION_TARGET,
)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _ensure_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    converted = X.copy()
    bad_columns: list[str] = []
    for col in converted.columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
        if converted[col].isna().any():
            bad_columns.append(str(col))
    if bad_columns:
        raise ValueError(f"Non-numeric or missing feature values detected: {bad_columns[:20]}")
    return converted


def is_motor_noise_dataframe(df: pd.DataFrame) -> bool:
    columns = set(map(str, df.columns))
    return MOTOR_NOISE_TARGET in columns and any(col in columns for col in MOTOR_NOISE_DIRECTION_COLUMNS)


def motor_noise_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [str(col) for col in df.columns]
    sound_indexes = [idx for idx, col in enumerate(columns) if col in MOTOR_NOISE_IGNORED_COLUMNS or col.lower().startswith("sound_")]
    if sound_indexes:
        return columns[: min(sound_indexes)]
    return [col for col in columns if col not in MOTOR_NOISE_IGNORED_COLUMNS]


def design_model_names(df: pd.DataFrame) -> list[str]:
    if "DESIGN_MODEL_NO__" not in df.columns:
        return []
    names: list[str] = []
    for value in df["DESIGN_MODEL_NO__"].dropna().astype(str).tolist():
        item = value.strip()
        if item and item not in names:
            names.append(item)
    return names


def encode_categorical_features(
    X: pd.DataFrame,
    categorical_levels: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    encoded = X.copy()
    levels: dict[str, list[Any]] = {}
    configured = categorical_levels or {}
    for col in encoded.columns:
        if col in configured:
            categories = [str(item) for item in configured[col]]
        elif not pd.api.types.is_numeric_dtype(encoded[col]):
            categories = sorted(str(item) for item in encoded[col].dropna().unique().tolist())
        else:
            continue
        levels[str(col)] = categories
        encoded[col] = pd.Categorical(encoded[col].astype(str), categories=categories).codes.astype(float)
    return encoded, levels


def prepare_classification_training_df(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Replicate the classification notebook preprocessing."""
    _require_columns(df, CLASSIFICATION_IGNORED_COLUMNS)
    original_rows = len(df)
    clean = df.dropna().reset_index(drop=True)
    dropped_rows = original_rows - len(clean)
    y = np.where(clean[CLASSIFICATION_SOURCE_TARGET] == 1.0, 1, 0).astype(int)
    X = clean.drop(columns=CLASSIFICATION_IGNORED_COLUMNS, errors="raise")
    X = _ensure_numeric_features(X)
    return X, y, {
        "dropped_rows": int(dropped_rows),
        "original_rows": int(original_rows),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
    }


def prepare_regression_training_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Replicate the regression notebook preprocessing."""
    if is_motor_noise_dataframe(df):
        _require_columns(df, [MOTOR_NOISE_TARGET])
        original_rows = len(df)
        feature_columns = motor_noise_feature_columns(df)
        clean = df.dropna(subset=[*feature_columns, MOTOR_NOISE_TARGET]).reset_index(drop=True)
        dropped_rows = original_rows - len(clean)
        y = pd.to_numeric(clean[MOTOR_NOISE_TARGET], errors="raise")
        X = clean[feature_columns].copy()
        X, categorical_levels = encode_categorical_features(X)
        X = _ensure_numeric_features(X)
        return X, y, {
            "domain": "motor_noise",
            "display_name": "회전기 소음 예측 모델",
            "design_model_column": "DESIGN_MODEL_NO__",
            "design_model_names": design_model_names(clean),
            "dropped_rows": int(dropped_rows),
            "original_rows": int(original_rows),
            "target": MOTOR_NOISE_TARGET,
            "target_summary": y.describe().to_dict(),
            "ignored_columns": [col for col in MOTOR_NOISE_IGNORED_COLUMNS if col in df.columns],
            "feature_columns": feature_columns,
            "categorical_levels": categorical_levels,
        }

    _require_columns(df, REGRESSION_IGNORED_COLUMNS)
    original_rows = len(df)
    clean = df.dropna().reset_index(drop=True)
    dropped_rows = original_rows - len(clean)
    y = pd.to_numeric(clean[REGRESSION_TARGET], errors="raise")
    X = clean.drop(columns=REGRESSION_IGNORED_COLUMNS, errors="raise")
    X = _ensure_numeric_features(X)
    return X, y, {
        "dropped_rows": int(dropped_rows),
        "original_rows": int(original_rows),
        "target_summary": y.describe().to_dict(),
    }


def prepare_prediction_features(df: pd.DataFrame, features: list[str], *, numeric_only: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort prediction input by schema feature order and reject missing/NaN features."""
    missing = [col for col in features if col not in df.columns]
    extra = [col for col in df.columns if col not in features]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[features].copy()
    if numeric_only:
        X = _ensure_numeric_features(X)
    elif X.isna().any().any():
        bad_columns = [str(col) for col in X.columns if X[col].isna().any()]
        raise ValueError(f"Missing feature values detected: {bad_columns[:20]}")
    return X, {"missing_columns": missing, "extra_columns": extra, "feature_order": features}
