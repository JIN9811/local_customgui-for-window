"""Deterministic tool layer used by Streamlit and the local LLM router."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .artifacts import list_artifacts, load_artifact
from .data_loader import DatasetRecord, dataframe_summary
from .explanation import explain_model_or_prediction
from .inference import predict_batch
from .schema import infer_dataset_type, target_distribution, validate_columns_against_schema
from .training import train_classification_model, train_regression_model


@dataclass
class ToolContext:
    datasets: dict[str, DatasetRecord] = field(default_factory=dict)
    active_dataset_id: str | None = None
    results: dict[str, pd.DataFrame] = field(default_factory=dict)
    model_root: Path | None = None

    def active_dataset(self, dataset_id: str | None = None) -> DatasetRecord:
        key = dataset_id or self.active_dataset_id
        if not key or key not in self.datasets:
            raise ValueError("활성 업로드 데이터가 없습니다.")
        return self.datasets[key]


def ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **payload}


def fail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "failure_code": code, "message": message, **extra}


def get_uploaded_data_summary(ctx: ToolContext, dataset_id: str | None = None) -> dict[str, Any]:
    try:
        record = ctx.active_dataset(dataset_id)
        summary = dataframe_summary(record.dataframe, max_preview_rows=5)
        return ok(
            {
                "tool": "get_uploaded_data_summary",
                "dataset_id": record.dataset_id,
                "filename": record.filename,
                "dataset_type": record.dataset_type,
                "summary": summary,
                "target_distribution": target_distribution(record.dataframe, record.dataset_type),
            }
        )
    except Exception as exc:
        return fail("DATASET_UNAVAILABLE", str(exc))


def infer_uploaded_dataset_type(ctx: ToolContext, dataset_id: str | None = None) -> dict[str, Any]:
    try:
        record = ctx.active_dataset(dataset_id)
        inferred = infer_dataset_type(list(record.dataframe.columns))
        return ok({"tool": "infer_uploaded_dataset_type", "dataset_id": record.dataset_id, "inferred_type": inferred})
    except Exception as exc:
        return fail("DATASET_UNAVAILABLE", str(exc))


def validate_schema(ctx: ToolContext, *, task: str, dataset_id: str | None = None, model_id: str = "latest") -> dict[str, Any]:
    try:
        record = ctx.active_dataset(dataset_id)
        artifact = load_artifact(task, model_id, ctx.model_root)
        validation = validate_columns_against_schema(list(record.dataframe.columns), artifact["schema"])
        return ok({"tool": f"validate_{task}_schema", "dataset_id": record.dataset_id, "model_id": model_id, **validation})
    except Exception as exc:
        return fail("SCHEMA_VALIDATION_FAILED", str(exc))


def train_tool(ctx: ToolContext, *, task: str, dataset_id: str | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    try:
        record = ctx.active_dataset(dataset_id)
        if task == "classification":
            result = train_classification_model(
                record.dataframe,
                source_filename=record.filename,
                train_size=float(options.get("train_size", 0.9)),
                random_state=int(options.get("random_state", 0)),
                model_root=ctx.model_root,
            )
        elif task == "regression":
            result = train_regression_model(
                record.dataframe,
                source_filename=record.filename,
                train_size=float(options.get("train_size", 0.9)),
                random_state=int(options.get("random_state", 0)),
                model_root=ctx.model_root,
            )
        else:
            return fail("BAD_TASK", f"Unsupported task: {task}")
        return ok({"tool": f"train_{task}_model", **result})
    except Exception as exc:
        return fail("TRAINING_FAILED", str(exc), task=task)


def predict_tool(ctx: ToolContext, *, task: str, dataset_id: str | None = None, model_id: str = "latest") -> dict[str, Any]:
    try:
        record = ctx.active_dataset(dataset_id)
        result_df, summary = predict_batch(record.dataframe, task=task, model_id=model_id, model_root=ctx.model_root)
        result_id = f"pred-{uuid4().hex[:10]}"
        ctx.results[result_id] = result_df
        preview = result_df.head(5).where(pd.notna(result_df.head(5)), None).to_dict(orient="records")
        return ok(
            {
                "tool": "prediction.batch",
                "task": task,
                "dataset_id": record.dataset_id,
                "model_id": model_id,
                "result_id": result_id,
                "summary": summary,
                "preview": preview,
            }
        )
    except Exception as exc:
        return fail("PREDICTION_FAILED", str(exc), task=task)


def explain_tool(ctx: ToolContext, *, task: str, model_id: str = "latest", dataset_id: str | None = None, row_index: int | None = None) -> dict[str, Any]:
    try:
        df = None
        if dataset_id or ctx.active_dataset_id:
            df = ctx.active_dataset(dataset_id).dataframe
        result = explain_model_or_prediction(task=task, model_id=model_id, df=df, row_index=row_index, model_root=ctx.model_root)
        return ok({"tool": "explain_model_or_prediction", **result})
    except Exception as exc:
        return fail("EXPLANATION_FAILED", str(exc), task=task)


def get_model_metrics(ctx: ToolContext, *, task: str, model_id: str = "latest") -> dict[str, Any]:
    try:
        artifact = load_artifact(task, model_id, ctx.model_root)
        return ok(
            {
                "tool": "get_model_metrics",
                "task": task,
                "model_id": model_id,
                "metrics": artifact["metrics"],
                "schema": artifact["schema"],
                "model_card": artifact["model_card"][:4000],
            }
        )
    except Exception as exc:
        return fail("MODEL_METRICS_UNAVAILABLE", str(exc), task=task)


def get_prediction_result_summary(ctx: ToolContext, *, result_id: str) -> dict[str, Any]:
    if result_id not in ctx.results:
        return fail("RESULT_NOT_FOUND", f"Prediction result not found: {result_id}")
    df = ctx.results[result_id]
    prediction_cols = [col for col in df.columns if str(col).startswith("prediction") or str(col).startswith("probability") or str(col).startswith("predicted_")]
    return ok(
        {
            "tool": "get_prediction_result_summary",
            "result_id": result_id,
            "rows": int(len(df)),
            "prediction_columns": prediction_cols,
            "preview": df[prediction_cols].head(10).where(pd.notna(df[prediction_cols].head(10)), None).to_dict(orient="records")
            if prediction_cols
            else [],
        }
    )


def available_models(ctx: ToolContext, task: str) -> list[dict[str, Any]]:
    return list_artifacts(task, ctx.model_root)
