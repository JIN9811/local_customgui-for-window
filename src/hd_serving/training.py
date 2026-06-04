"""Training pipelines for classification and regression."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

from .artifacts import save_artifact
from .constants import (
    CLASSIFICATION_IGNORED_COLUMNS,
    CLASSIFICATION_SOURCE_TARGET,
    CLASSIFICATION_TARGET,
    REGRESSION_IGNORED_COLUMNS,
    REGRESSION_TARGET,
)
from .preprocessing import prepare_classification_training_df, prepare_regression_training_df
from .pycaret_bridge import train_pycaret_model


DEFAULT_TRAIN_ENGINE = "pycaret"


def _safe_train_test_split(X: pd.DataFrame, y: Any, *, train_size: float, random_state: int, stratify: Any | None = None):
    test_size = max(1.0 - float(train_size), 0.05)
    if len(X) < 4:
        raise ValueError("At least 4 rows are required for train/test split.")
    try:
        return train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _classification_candidates(random_state: int) -> list[tuple[str, Any]]:
    return [
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=random_state)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=160, random_state=random_state, class_weight="balanced")),
        ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=180, random_state=random_state, class_weight="balanced")),
        ("LogisticRegression", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ("SVC", SVC(probability=True, class_weight="balanced", random_state=random_state)),
    ]


def _regression_candidates(random_state: int) -> list[tuple[str, Any]]:
    return [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=160, random_state=random_state)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=180, random_state=random_state)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=random_state)),
        ("Ridge", Ridge()),
        ("SVR", SVR()),
    ]


def _classification_metrics(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).astype(int).tolist(),
    }
    try:
        proba = model.predict_proba(X_test)[:, 1]
        if len(set(y_test)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def _regression_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
    }


def train_classification_model(
    df: pd.DataFrame,
    *,
    source_filename: str = "uploaded.xlsx",
    train_size: float = 0.9,
    random_state: int = 0,
    model_root: Path | None = None,
) -> dict[str, Any]:
    engine = str(os.environ.get("HD_SERVING_TRAIN_ENGINE", DEFAULT_TRAIN_ENGINE)).lower()
    if engine == "pycaret":
        return train_pycaret_model(
            df,
            task="classification",
            source_filename=source_filename,
            train_size=train_size,
            random_state=random_state,
            model_root=model_root,
        )

    X, y, prep = prepare_classification_training_df(df)
    stratify = y if len(set(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = _safe_train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=stratify)
    candidates = []
    best: tuple[str, Any, dict[str, Any]] | None = None
    for name, model in _classification_candidates(random_state):
        try:
            model.fit(X_train, y_train)
            holdout = _classification_metrics(model, X_test, y_test)
            row = {"model": name, "holdout": holdout}
            candidates.append(row)
            if best is None or holdout["accuracy"] > best[2]["accuracy"]:
                best = (name, model, holdout)
        except Exception as exc:
            candidates.append({"model": name, "error": str(exc)})
    if best is None:
        raise RuntimeError("No classification candidate model could be trained.")
    best_name, best_model, holdout = best
    schema = {
        "task": "classification",
        "target": CLASSIFICATION_TARGET,
        "source_target_column": CLASSIFICATION_SOURCE_TARGET,
        "positive_class_rule": "Result == 1.0 -> 1 else 0",
        "features": list(X.columns),
        "ignored_columns": CLASSIFICATION_IGNORED_COLUMNS,
        "dropna": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "train_size": train_size,
        "model_name": best_name,
    }
    metrics = {
        "task": "classification",
        "engine": "sklearn",
        "notebook_parity": False,
        "best_model": best_name,
        "selection_metric": "accuracy",
        "holdout": holdout,
        "candidates": candidates,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_distribution": prep["label_distribution"],
        "dropped_rows": prep["dropped_rows"],
    }
    artifact = save_artifact(task="classification", estimator=best_model, schema=schema, metrics=metrics, source_filename=source_filename, model_root=model_root)
    return {**artifact, "metrics": metrics, "schema": schema}


def train_regression_model(
    df: pd.DataFrame,
    *,
    source_filename: str = "uploaded.xlsx",
    train_size: float = 0.9,
    random_state: int = 0,
    model_root: Path | None = None,
) -> dict[str, Any]:
    engine = str(os.environ.get("HD_SERVING_TRAIN_ENGINE", DEFAULT_TRAIN_ENGINE)).lower()
    if engine == "pycaret":
        return train_pycaret_model(
            df,
            task="regression",
            source_filename=source_filename,
            train_size=train_size,
            random_state=random_state,
            model_root=model_root,
        )

    X, y, prep = prepare_regression_training_df(df)
    X_train, X_test, y_train, y_test = _safe_train_test_split(X, y, train_size=train_size, random_state=random_state)
    candidates = []
    best: tuple[str, Any, dict[str, Any]] | None = None
    for name, model in _regression_candidates(random_state):
        try:
            model.fit(X_train, y_train)
            holdout = _regression_metrics(model, X_test, y_test)
            row = {"model": name, "holdout": holdout}
            candidates.append(row)
            if best is None or holdout["r2"] > best[2]["r2"]:
                best = (name, model, holdout)
        except Exception as exc:
            candidates.append({"model": name, "error": str(exc)})
    if best is None:
        raise RuntimeError("No regression candidate model could be trained.")
    best_name, best_model, holdout = best
    schema = {
        "task": "regression",
        "domain": prep.get("domain", "circuit_breaker"),
        "display_name": prep.get("display_name", "형상/시험 변수 기반 고압차단기 성능 예측"),
        "design_model_column": prep.get("design_model_column"),
        "design_model_names": prep.get("design_model_names", []),
        "target": prep.get("target", REGRESSION_TARGET),
        "features": list(X.columns),
        "ignored_columns": prep.get("ignored_columns", REGRESSION_IGNORED_COLUMNS),
        "categorical_levels": prep.get("categorical_levels", {}),
        "dropna": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "train_size": train_size,
        "model_name": best_name,
    }
    metrics = {
        "task": "regression",
        "domain": prep.get("domain", "circuit_breaker"),
        "display_name": prep.get("display_name", "형상/시험 변수 기반 고압차단기 성능 예측"),
        "design_model_column": prep.get("design_model_column"),
        "design_model_names": prep.get("design_model_names", []),
        "engine": "sklearn",
        "notebook_parity": False,
        "best_model": best_name,
        "selection_metric": "r2",
        "holdout": holdout,
        "candidates": candidates,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_summary": prep["target_summary"],
        "dropped_rows": prep["dropped_rows"],
    }
    artifact = save_artifact(task="regression", estimator=best_model, schema=schema, metrics=metrics, source_filename=source_filename, model_root=model_root)
    return {**artifact, "metrics": metrics, "schema": schema}
