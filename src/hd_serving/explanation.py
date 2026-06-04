"""Model explanation helpers with deterministic fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance

from .artifacts import load_artifact
from .preprocessing import prepare_prediction_features


def _top(values: list[tuple[str, float]], limit: int = 15) -> list[dict[str, Any]]:
    return [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(values, key=lambda item: abs(item[1]), reverse=True)[:limit]
    ]


def explain_model_or_prediction(
    *,
    task: str,
    model_id: str = "latest",
    df: pd.DataFrame | None = None,
    row_index: int | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    artifact = load_artifact(task, model_id, model_root)
    model = artifact["model"]
    schema = artifact["schema"]
    features = schema.get("features", [])

    if hasattr(model, "feature_importances_"):
        return {
            "ok": True,
            "task": task,
            "method": "feature_importances_",
            "top_features": _top(list(zip(features, model.feature_importances_))),
            "notes": "SHAP unavailable or not requested; used model.feature_importances_.",
        }
    if hasattr(model, "coef_"):
        coef = model.coef_
        if hasattr(coef, "ravel"):
            coef = coef.ravel()
        return {
            "ok": True,
            "task": task,
            "method": "coef_",
            "top_features": _top(list(zip(features, coef))),
            "notes": "Used linear model coefficients.",
        }
    if df is not None and len(df) > 1:
        X, _ = prepare_prediction_features(df, features)
        if row_index is not None:
            X = X.iloc[[int(row_index)]]
        scoring = "accuracy" if task == "classification" else "r2"
        try:
            # Uses model predictions as pseudo-target only to expose relative sensitivity.
            y = model.predict(X)
            result = permutation_importance(model, X, y, n_repeats=3, random_state=0, scoring=scoring)
            return {
                "ok": True,
                "task": task,
                "method": "permutation_importance_pseudo_target",
                "top_features": _top(list(zip(features, result.importances_mean))),
                "notes": "Used permutation importance fallback on current data and model predictions.",
            }
        except Exception as exc:
            return {"ok": False, "task": task, "failure_code": "EXPLANATION_UNAVAILABLE", "error": str(exc)}
    return {"ok": False, "task": task, "failure_code": "EXPLANATION_UNAVAILABLE", "error": "No supported explanation method."}
