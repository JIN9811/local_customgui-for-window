"""Keyword-safe local tool router for the LLM chat tab."""

from __future__ import annotations

import json
from typing import Any

from .tools import (
    ToolContext,
    explain_tool,
    get_model_metrics,
    get_prediction_result_summary,
    get_uploaded_data_summary,
    infer_uploaded_dataset_type,
    predict_tool,
    train_tool,
    validate_schema,
)


def _task_from_text(text: str, selected_task: str) -> str:
    lower = text.lower()
    if "회귀" in text or "regression" in lower or "trv" in lower:
        return "regression"
    if "분류" in text or "classification" in lower or "성공" in text or "실패" in text:
        return "classification"
    return selected_task


def route_tool_request(
    text: str,
    *,
    ctx: ToolContext,
    selected_task: str = "classification",
    model_id: str = "latest",
    latest_result_id: str | None = None,
) -> dict[str, Any] | None:
    normalized = text.lower().replace(" ", "")
    task = _task_from_text(text, selected_task)

    if any(token in normalized for token in ["요약", "summary", "컬럼", "column", "미리보기"]):
        return get_uploaded_data_summary(ctx)
    if any(token in normalized for token in ["분류용인지", "회귀용인지", "데이터유형", "infer", "판별"]):
        return infer_uploaded_dataset_type(ctx)
    if any(token in normalized for token in ["스키마", "schema", "검증", "누락", "missing"]):
        return validate_schema(ctx, task=task, model_id=model_id)
    if any(token in normalized for token in ["학습", "train", "모델만들", "모델생성"]):
        return train_tool(ctx, task=task)
    if any(token in normalized for token in ["예측", "predict", "inference", "추론"]):
        return predict_tool(ctx, task=task, model_id=model_id)
    if any(token in normalized for token in ["설명", "explain", "중요", "featureimportance", "shap"]):
        if "결과" in normalized and latest_result_id:
            return get_prediction_result_summary(ctx, result_id=latest_result_id)
        return explain_tool(ctx, task=task, model_id=model_id)
    if any(token in normalized for token in ["metric", "성능", "정확도", "r2", "auc"]):
        return get_model_metrics(ctx, task=task, model_id=model_id)
    return None


def tool_result_to_korean(tool_result: dict[str, Any]) -> str:
    if not tool_result:
        return "처리 결과가 없습니다."
    if not tool_result.get("ok"):
        return f"요청 처리 실패: {tool_result.get('message') or tool_result.get('error') or tool_result.get('failure_code')}"
    tool = str(tool_result.get("tool", ""))
    task = str(tool_result.get("task") or "")
    if tool == "prediction.batch" or tool.startswith("predict_"):
        label = f"{task} 예측 결과" if task else "예측 결과"
    elif tool.startswith("train_"):
        label = "모델 학습 결과"
    elif tool == "validate_schema":
        label = "입력 변수 검증 결과"
    else:
        label = "처리 결과"
    lines = [f"{label}입니다."]
    if "dataset_id" in tool_result:
        lines.append(f"- dataset_id: `{tool_result['dataset_id']}`")
    if "model_id" in tool_result:
        lines.append(f"- model_id: `{tool_result['model_id']}`")
    if "result_id" in tool_result:
        lines.append(f"- result_id: `{tool_result['result_id']}`")
    if "summary" in tool_result:
        summary = tool_result["summary"]
        if isinstance(summary, dict):
            rows = summary.get("rows")
            cols = summary.get("columns")
            if rows is not None:
                lines.append(f"- rows: {rows}")
            if cols is not None:
                lines.append(f"- columns: {cols}")
            if summary.get("column_list"):
                lines.append(f"- columns preview: {', '.join(map(str, summary['column_list'][:12]))}")
    if "metrics" in tool_result:
        metrics = tool_result["metrics"]
        lines.append(f"- best_model: {metrics.get('best_model')}")
        lines.append(f"- holdout: `{json.dumps(metrics.get('holdout', {}), ensure_ascii=False)}`")
    if "top_features" in tool_result:
        lines.append("- top_features:")
        for row in tool_result["top_features"][:8]:
            lines.append(f"  - {row.get('feature')}: {row.get('importance')}")
    if "preview" in tool_result:
        lines.append("- 미리보기와 전체 다운로드는 결과 메시지를 펼치면 확인할 수 있습니다.")
    return "\n".join(lines)
