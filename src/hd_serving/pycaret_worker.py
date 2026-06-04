"""PyCaret notebook-parity worker.

This module is intended to run under Python 3.11 because PyCaret 3.3.x does
not support Python 3.12.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pycaret
from PIL import Image

matplotlib.use("Agg", force=True)

from .artifacts import save_artifact, save_json
from .constants import (
    CLASSIFICATION_IGNORED_COLUMNS,
    CLASSIFICATION_SOURCE_TARGET,
    CLASSIFICATION_TARGET,
    MOTOR_NOISE_DIRECTION_COLUMNS,
    MOTOR_NOISE_IGNORED_COLUMNS,
    MOTOR_NOISE_TARGET,
    REGRESSION_IGNORED_COLUMNS,
    REGRESSION_TARGET,
)
from .preprocessing import design_model_names, encode_categorical_features

JSON_MARKER = "__HD_SERVING_JSON__"
FIGURE_DPI = 300
FIGURE_WIDTH_PX = 1654
FIGURE_HEIGHT_PX = 1100
FIGURE_SCALE = 1.55

matplotlib.rcParams.update(
    {
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _frame_records(df: pd.DataFrame, limit: int | None = 20) -> list[dict[str, Any]]:
    frame = df if limit is None else df.head(limit)
    safe = frame.replace({pd.NA: None}).where(pd.notna(frame), None)
    return _safe_json(safe.to_dict(orient="records"))


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _classification_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(df, CLASSIFICATION_IGNORED_COLUMNS)
    original_rows = len(df)
    clean = df.dropna().reset_index(drop=True)
    model_df = clean.copy()
    model_df[CLASSIFICATION_TARGET] = np.where(model_df[CLASSIFICATION_SOURCE_TARGET] == 1.0, 1, 0).astype(int)
    model_df = model_df.drop(CLASSIFICATION_IGNORED_COLUMNS, axis=1)
    y = model_df[CLASSIFICATION_TARGET]
    return model_df, {
        "original_rows": int(original_rows),
        "dropped_rows": int(original_rows - len(clean)),
        "features": [str(col) for col in model_df.columns if col != CLASSIFICATION_TARGET],
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }


def _regression_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [str(col) for col in df.columns]
    if MOTOR_NOISE_TARGET in set(columns) and any(col in set(columns) for col in MOTOR_NOISE_DIRECTION_COLUMNS):
        original_rows = len(df)
        sound_indexes = [idx for idx, col in enumerate(columns) if col in MOTOR_NOISE_IGNORED_COLUMNS or col.lower().startswith("sound_")]
        feature_columns = columns[: min(sound_indexes)] if sound_indexes else [col for col in columns if col not in MOTOR_NOISE_IGNORED_COLUMNS]
        clean = df.dropna(subset=[*feature_columns, MOTOR_NOISE_TARGET]).reset_index(drop=True)
        model_df = clean[[*feature_columns, MOTOR_NOISE_TARGET]].copy()
        encoded_features, categorical_levels = encode_categorical_features(model_df[feature_columns])
        model_df[feature_columns] = encoded_features
        target = pd.to_numeric(model_df[MOTOR_NOISE_TARGET], errors="raise")
        model_df[MOTOR_NOISE_TARGET] = target
        return model_df, {
            "domain": "motor_noise",
            "display_name": "회전기 소음 예측 모델",
            "design_model_column": "DESIGN_MODEL_NO__",
            "design_model_names": design_model_names(clean),
            "original_rows": int(original_rows),
            "dropped_rows": int(original_rows - len(clean)),
            "features": [str(col) for col in feature_columns],
            "target": MOTOR_NOISE_TARGET,
            "ignored_columns": [col for col in MOTOR_NOISE_IGNORED_COLUMNS if col in df.columns],
            "categorical_levels": categorical_levels,
            "target_summary": _safe_json(target.describe().to_dict()),
            "notebook_workflow": "motor_noise_regression: dropna -> use columns before sound_* as features -> target sound_mean -> setup -> compare_models",
        }

    _require_columns(df, REGRESSION_IGNORED_COLUMNS)
    original_rows = len(df)
    clean = df.dropna().reset_index(drop=True)
    model_df = clean.drop(["Time", "Result", "CZM", "Test"], axis=1)
    first_col = model_df.columns[0]
    model_df = model_df[model_df.columns[1:].tolist() + [first_col]]
    target = pd.to_numeric(model_df[REGRESSION_TARGET], errors="raise")
    return model_df, {
        "domain": "circuit_breaker",
        "display_name": "형상/시험 변수 기반 고압차단기 성능 예측",
        "original_rows": int(original_rows),
        "dropped_rows": int(original_rows - len(clean)),
        "features": [str(col) for col in model_df.columns if col != REGRESSION_TARGET],
        "target": REGRESSION_TARGET,
        "ignored_columns": REGRESSION_IGNORED_COLUMNS,
        "target_summary": _safe_json(target.describe().to_dict()),
        "notebook_workflow": "regression: dropna -> drop Time/Result/CZM/Test -> move first column to end -> setup -> compare_models",
    }


def _best_row(table: pd.DataFrame) -> dict[str, Any]:
    if table is None or table.empty:
        return {}
    row = table.head(1).reset_index(drop=False).iloc[0].to_dict()
    return _safe_json(row)


def _model_features(model: Any, fallback: list[str]) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return fallback
    return [str(name) for name in list(names)]


def _plot_title(plot_id: str) -> str:
    titles = {
        "auc": "PyCaret AUC Curve",
        "pr": "PyCaret Precision-Recall Curve",
        "confusion_matrix": "PyCaret Confusion Matrix",
        "class_report": "PyCaret Classification Report",
        "error": "PyCaret Prediction Error",
        "residuals": "PyCaret Residuals Plot",
        "feature": "PyCaret Feature Importance",
    }
    return titles.get(plot_id, f"PyCaret {plot_id}")


def _capture_new_plot_file(tmp_dir: Path, returned: str | None, before: set[Path]) -> Path | None:
    candidates: list[Path] = []
    if returned:
        returned_path = Path(returned)
        if not returned_path.is_absolute():
            returned_path = tmp_dir / returned_path
        candidates.append(returned_path)
    after = {path for path in tmp_dir.rglob("*") if path.is_file()}
    candidates.extend(sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _save_publication_png(source: Path, dest: Path) -> tuple[int, int]:
    image = Image.open(source).convert("RGBA")
    background = Image.new("RGBA", image.size, "WHITE")
    background.alpha_composite(image)
    image = background.convert("RGB")

    image.thumbnail((FIGURE_WIDTH_PX, FIGURE_HEIGHT_PX), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, FIGURE_HEIGHT_PX), "WHITE")
    left = (FIGURE_WIDTH_PX - image.width) // 2
    top = (FIGURE_HEIGHT_PX - image.height) // 2
    canvas.paste(image, (left, top))
    canvas.save(dest, format="PNG", dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)
    return canvas.size


def _save_pycaret_figures(plot_model_fn: Any, estimator: Any, *, plots: list[str], artifact_dir: Path) -> list[dict[str, Any]]:
    figure_dir = artifact_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hd_pycaret_figures_") as tmp_name:
        tmp_dir = Path(tmp_name)
        cwd = Path.cwd()
        os.chdir(tmp_dir)
        try:
            for index, plot_id in enumerate(plots, start=1):
                before = {path for path in tmp_dir.rglob("*") if path.is_file()}
                try:
                    returned = plot_model_fn(estimator, plot=plot_id, scale=FIGURE_SCALE, save=True, verbose=False)
                    source = _capture_new_plot_file(tmp_dir, returned, before)
                    if source is None:
                        raise RuntimeError("plot_model did not return or create a plot file")
                    dest = figure_dir / f"{index:02d}_{plot_id}.png"
                    width_px, height_px = _save_publication_png(source, dest)
                    figures.append(
                        {
                            "ok": True,
                            "plot": plot_id,
                            "title": _plot_title(plot_id),
                            "path": str(dest),
                            "filename": dest.name,
                            "width_px": width_px,
                            "height_px": height_px,
                            "dpi": FIGURE_DPI,
                            "paper_size_in": [round(width_px / FIGURE_DPI, 3), round(height_px / FIGURE_DPI, 3)],
                            "figure_standard": "publication_double_column_canvas",
                        }
                    )
                except Exception as exc:
                    figures.append({"ok": False, "plot": plot_id, "title": _plot_title(plot_id), "error": str(exc)})
                finally:
                    for path in tmp_dir.rglob("*"):
                        if path.is_file():
                            try:
                                path.unlink()
                            except OSError:
                                pass
        finally:
            os.chdir(cwd)
    return figures


def _update_artifact_metrics(artifact: dict[str, Any], metrics: dict[str, Any]) -> None:
    metrics_path = Path(str(artifact["metrics_path"]))
    save_json(metrics_path, metrics)
    latest_metrics = Path(str(artifact["latest_dir"])) / "metrics.json"
    try:
        if latest_metrics.resolve() != metrics_path.resolve():
            save_json(latest_metrics, metrics)
    except FileNotFoundError:
        save_json(latest_metrics, metrics)


def _classification_validation_outputs(holdout_predictions: pd.DataFrame, holdout_table: pd.DataFrame) -> dict[str, Any]:
    frame = holdout_predictions.copy()
    if CLASSIFICATION_TARGET not in frame.columns or "prediction_label" not in frame.columns:
        return {
            "validation_metrics": _frame_records(holdout_table, limit=5),
            "validation_predictions": _frame_records(frame, limit=None),
        }

    actual = pd.to_numeric(frame[CLASSIFICATION_TARGET], errors="coerce").astype("Int64")
    predicted = pd.to_numeric(frame["prediction_label"], errors="coerce").astype("Int64")
    labels = sorted({int(v) for v in actual.dropna().unique().tolist() + predicted.dropna().unique().tolist()})
    matrix = pd.crosstab(actual, predicted, rownames=["actual"], colnames=["predicted"], dropna=False)
    if labels:
        matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    frame["correct"] = actual == predicted
    frame["actual_label"] = actual
    frame["predicted_label"] = predicted

    return {
        "validation_metrics": _frame_records(holdout_table, limit=5),
        "validation_confusion_matrix": {
            "labels": labels,
            "rows": [
                {
                    "actual": int(actual_label),
                    **{f"predicted_{int(predicted_label)}": int(matrix.loc[actual_label, predicted_label]) for predicted_label in matrix.columns},
                }
                for actual_label in matrix.index
            ],
        },
        "validation_predictions": _frame_records(frame, limit=None),
    }


def _regression_validation_outputs(holdout_predictions: pd.DataFrame, holdout_table: pd.DataFrame, *, target: str = REGRESSION_TARGET) -> dict[str, Any]:
    frame = holdout_predictions.copy()
    if target in frame.columns and "prediction_label" in frame.columns:
        actual = pd.to_numeric(frame[target], errors="coerce")
        predicted = pd.to_numeric(frame["prediction_label"], errors="coerce")
        frame["residual"] = predicted - actual
        frame["absolute_error"] = (predicted - actual).abs()
        residual_summary = _safe_json(
            {
                "mean_residual": frame["residual"].mean(),
                "median_residual": frame["residual"].median(),
                "max_absolute_error": frame["absolute_error"].max(),
                "mean_absolute_error_from_validation_rows": frame["absolute_error"].mean(),
            }
        )
    else:
        residual_summary = {}
    return {
        "validation_metrics": _frame_records(holdout_table, limit=5),
        "validation_residual_summary": residual_summary,
        "validation_predictions": _frame_records(frame, limit=None),
    }


def train_classification(df: pd.DataFrame, *, source_filename: str, train_size: float, random_state: int, model_root: Path) -> dict[str, Any]:
    from pycaret.classification import compare_models, plot_model, predict_model, pull, setup

    model_df, prep = _classification_frame(df)
    setup(data=model_df, target=CLASSIFICATION_TARGET, session_id=random_state, train_size=train_size, html=False, verbose=False)
    best_model = compare_models()
    compare_table = pull()
    holdout_predictions = predict_model(best_model)
    holdout_table = pull()
    best = _best_row(compare_table)
    validation_outputs = _classification_validation_outputs(holdout_predictions, holdout_table)
    schema = {
        "task": "classification",
        "engine": "pycaret",
        "notebook_parity": True,
        "target": CLASSIFICATION_TARGET,
        "source_target_column": CLASSIFICATION_SOURCE_TARGET,
        "positive_class_rule": "Result == 1.0 -> 1 else 0",
        "features": prep["features"],
        "model_features": _model_features(best_model, prep["features"]),
        "ignored_columns": CLASSIFICATION_IGNORED_COLUMNS,
        "dropna": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "train_size": train_size,
        "model_name": str(best.get("Model") or type(best_model).__name__),
        "pycaret_version": pycaret.__version__,
        "notebook_workflow": "classification: dropna -> label -> drop Result/TRVmax[kV] -> setup -> compare_models",
    }
    metrics = {
        "task": "classification",
        "engine": "pycaret",
        "notebook_parity": True,
        "best_model": schema["model_name"],
        "selection_metric": "PyCaret compare_models default",
        "holdout": best,
        "compare_models": _frame_records(compare_table, limit=30),
        "predict_model_metrics": _frame_records(holdout_table, limit=5),
        "holdout_prediction_preview": _frame_records(holdout_predictions, limit=5),
        "validation_split": {
            "train_size": float(train_size),
            "validation_size": round(1.0 - float(train_size), 6),
            "validation_rows": int(len(holdout_predictions)),
        },
        **validation_outputs,
        "n_rows_after_dropna": int(len(model_df)),
        "class_distribution": prep["class_distribution"],
        "dropped_rows": prep["dropped_rows"],
    }
    artifact = save_artifact(
        task="classification",
        estimator=best_model,
        schema=schema,
        metrics=metrics,
        source_filename=source_filename,
        model_root=model_root,
    )
    metrics["pycaret_figures"] = _save_pycaret_figures(
        plot_model,
        best_model,
        plots=["confusion_matrix", "class_report", "auc", "pr", "error", "feature"],
        artifact_dir=Path(str(artifact["artifact_dir"])),
    )
    _update_artifact_metrics(artifact, metrics)
    return {**artifact, "metrics": metrics, "schema": schema}


def train_regression(df: pd.DataFrame, *, source_filename: str, train_size: float, random_state: int, model_root: Path) -> dict[str, Any]:
    from pycaret.regression import compare_models, plot_model, predict_model, pull, setup

    model_df, prep = _regression_frame(df)
    target = str(prep.get("target") or REGRESSION_TARGET)
    setup(data=model_df, target=target, session_id=random_state, train_size=train_size, html=False, verbose=False)
    best_model = compare_models()
    compare_table = pull()
    holdout_predictions = predict_model(best_model)
    holdout_table = pull()
    best = _best_row(compare_table)
    validation_outputs = _regression_validation_outputs(holdout_predictions, holdout_table, target=target)
    schema = {
        "task": "regression",
        "domain": prep.get("domain", "circuit_breaker"),
        "display_name": prep.get("display_name", "형상/시험 변수 기반 고압차단기 성능 예측"),
        "design_model_column": prep.get("design_model_column"),
        "design_model_names": prep.get("design_model_names", []),
        "engine": "pycaret",
        "notebook_parity": True,
        "target": target,
        "features": prep["features"],
        "model_features": _model_features(best_model, prep["features"]),
        "ignored_columns": prep.get("ignored_columns", REGRESSION_IGNORED_COLUMNS),
        "categorical_levels": prep.get("categorical_levels", {}),
        "dropna": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "train_size": train_size,
        "model_name": str(best.get("Model") or type(best_model).__name__),
        "pycaret_version": pycaret.__version__,
        "notebook_workflow": prep.get("notebook_workflow", "regression: dropna -> setup -> compare_models"),
    }
    metrics = {
        "task": "regression",
        "domain": prep.get("domain", "circuit_breaker"),
        "display_name": prep.get("display_name", "형상/시험 변수 기반 고압차단기 성능 예측"),
        "design_model_column": prep.get("design_model_column"),
        "design_model_names": prep.get("design_model_names", []),
        "engine": "pycaret",
        "notebook_parity": True,
        "best_model": schema["model_name"],
        "selection_metric": "PyCaret compare_models default",
        "holdout": best,
        "compare_models": _frame_records(compare_table, limit=30),
        "predict_model_metrics": _frame_records(holdout_table, limit=5),
        "holdout_prediction_preview": _frame_records(holdout_predictions, limit=5),
        "validation_split": {
            "train_size": float(train_size),
            "validation_size": round(1.0 - float(train_size), 6),
            "validation_rows": int(len(holdout_predictions)),
        },
        **validation_outputs,
        "n_rows_after_dropna": int(len(model_df)),
        "target_summary": prep["target_summary"],
        "dropped_rows": prep["dropped_rows"],
    }
    artifact = save_artifact(
        task="regression",
        estimator=best_model,
        schema=schema,
        metrics=metrics,
        source_filename=source_filename,
        model_root=model_root,
    )
    metrics["pycaret_figures"] = _save_pycaret_figures(
        plot_model,
        best_model,
        plots=["residuals", "error", "feature"],
        artifact_dir=Path(str(artifact["artifact_dir"])),
    )
    _update_artifact_metrics(artifact, metrics)
    return {**artifact, "metrics": metrics, "schema": schema}


def main() -> int:
    parser = argparse.ArgumentParser(description="HD Serving PyCaret worker")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--task", choices=["classification", "regression"], required=True)
    train.add_argument("--input", required=True)
    train.add_argument("--source-filename", default="uploaded.xlsx")
    train.add_argument("--train-size", type=float, default=0.9)
    train.add_argument("--random-state", type=int, default=0)
    train.add_argument("--model-root", required=True)
    args = parser.parse_args()

    if args.command == "train":
        df = pd.read_excel(args.input)
        root = Path(args.model_root)
        if args.task == "classification":
            result = train_classification(df, source_filename=args.source_filename, train_size=args.train_size, random_state=args.random_state, model_root=root)
        else:
            result = train_regression(df, source_filename=args.source_filename, train_size=args.train_size, random_state=args.random_state, model_root=root)
        print(JSON_MARKER + json.dumps(_safe_json(result), ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
