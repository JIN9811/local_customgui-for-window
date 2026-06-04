#!/usr/bin/env python3
"""HD Hyundai Electric local LLM based ML serving Streamlit app."""

from __future__ import annotations

import base64
import json
import hashlib
import html
import mimetypes
import re
import sys
import threading
import time
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import altair as alt
except ImportError:  # pragma: no cover - Streamlit normally installs Altair.
    alt = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties, findfont
    from PIL import Image
except ImportError:  # pragma: no cover - PDF export is optional at runtime.
    plt = None
    PdfPages = None
    FontProperties = None
    findfont = None
    Image = None

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ICON_DIR = ROOT / "Icon"
LOGO_DIR = ROOT / "Logo"
APP_ICON_PATH = ICON_DIR / "hd_hyundai_streamlit_avatar.svg"
USER_ICON_PATH = ICON_DIR / "user_person_gear_navy.svg"
AIM4LAB_LOGO_PATH = LOGO_DIR / "logo_aim4lab.png"
STATE_DIR = ROOT / "state"
UI_STATE_PATH = STATE_DIR / "ui_state.json"
PERSISTED_DATASETS_DIR = STATE_DIR / "datasets"
PERSISTED_RESULTS_DIR = STATE_DIR / "prediction_results"
REPORTS_DIR = STATE_DIR / "reports"

from app import check_health, deep_merge, http_get_json, http_post_json, load_config, save_config  # noqa: E402
from app import call_ollama as call_ollama_raw  # noqa: E402
from hd_serving.artifacts import delete_artifact, list_artifacts, load_artifact  # noqa: E402
from hd_serving.constants import DATASET_TYPES, MODEL_ROOT  # noqa: E402
from hd_serving.data_loader import dataframe_summary, make_dataset_record, missing_summary, read_excel_file  # noqa: E402
from hd_serving.inference import predict_batch  # noqa: E402
from hd_serving.llm_client import call_local_llm, stream_local_llm  # noqa: E402
from hd_serving.nemoclaw_vllm_runtime import NemoClawVLLMRuntime, default_nemoclaw_vllm_config  # noqa: E402
from hd_serving.orchestrator import route_tool_request, tool_result_to_korean  # noqa: E402
from hd_serving.schema import infer_dataset_type, target_distribution, validate_columns_against_schema  # noqa: E402
from hd_serving.tools import (  # noqa: E402
    ToolContext,
    available_models,
    get_model_metrics,
    get_prediction_result_summary,
    get_uploaded_data_summary,
    infer_uploaded_dataset_type,
    predict_tool,
    train_tool,
    validate_schema,
)

BACKENDS = ("ollama", "vllm")
TASKS = ("classification", "regression")
MODEL_LOAD_LOCK = threading.Lock()
MODEL_LOAD_THREADS: set[str] = set()
MODEL_LOAD_STATUS: dict[str, dict[str, Any]] = {}
WORKFLOW_ACTIONS = {
    "data_summary",
    "infer_dataset_type",
    "validate_schema",
    "train_model",
    "predict_batch",
    "explain",
    "model_metrics",
    "prediction_summary",
    "none",
}


def app_icon_value() -> Path | str:
    return APP_ICON_PATH if APP_ICON_PATH.exists() else "⚡"


def hd_electric_logo_path() -> Path | None:
    matches = sorted(LOGO_DIR.glob("HD Hyundai Electric*Negative*.png"))
    return matches[0] if matches else None


def image_data_uri(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def chat_avatar_value(role: str) -> Path | None:
    normalized = str(role or "").lower()
    if normalized in {"user", "human"} and USER_ICON_PATH.exists():
        return USER_ICON_PATH
    if normalized in {"assistant", "ai"} and APP_ICON_PATH.exists():
        return APP_ICON_PATH
    return None


def chat_message(role: str):
    normalized = str(role or "assistant").lower()
    name = normalized if normalized in {"user", "assistant", "ai", "human"} else "assistant"
    return st.chat_message(name, avatar=chat_avatar_value(name))

OLLAMA_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_uploaded_data_summary",
            "description": "현재 활성 Excel 데이터셋의 행/열/컬럼/타깃 분포 요약을 조회합니다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "infer_uploaded_dataset_type",
            "description": "현재 활성 Excel 데이터셋이 classification/regression/prediction 입력 중 무엇인지 판별합니다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": "notebook-parity workflow로 classification 또는 regression 모델을 학습합니다.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string", "enum": ["classification", "regression"]}},
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_schema",
            "description": "현재 활성 데이터셋 컬럼이 저장된 모델 스키마와 맞는지 검증합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": ["classification", "regression"]},
                    "model_id": {"type": "string"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_batch",
            "description": "저장된 모델로 현재 활성 Excel 데이터셋의 batch prediction을 수행합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": ["classification", "regression"]},
                    "model_id": {"type": "string"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "저장된 모델의 metric/model card/schema를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": ["classification", "regression"]},
                    "model_id": {"type": "string"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prediction_result_summary",
            "description": "가장 최근 또는 지정된 prediction result의 예측 컬럼 preview를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"result_id": {"type": "string"}},
                "required": [],
            },
        },
    },
]


def init_state() -> None:
    st.session_state.setdefault("config", load_config())
    restore_persistent_ui_state()
    st.session_state.setdefault("datasets", {})
    st.session_state.setdefault("active_dataset_id", None)
    st.session_state.setdefault("prediction_results", {})
    st.session_state.setdefault("latest_result_id", None)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("health_result", None)
    st.session_state.setdefault("selected_task", "classification")
    st.session_state.setdefault("selected_model_id", "latest")
    st.session_state.setdefault("llm_model_options", {})
    st.session_state.setdefault("llm_model_connect_status", None)
    st.session_state.setdefault("processed_upload_fingerprint", None)
    st.session_state.setdefault("workflow_completed_dataset_ids", set())
    st.session_state.setdefault("pending_workflow_dataset_id", None)
    st.session_state.setdefault("pending_workflow_source_message", None)
    st.session_state.setdefault("pending_workflow_force", False)
    st.session_state.setdefault("pending_workflow_stage", None)
    st.session_state.setdefault("pending_workflow_train_result", None)
    st.session_state.setdefault("pending_workflow_task", None)
    st.session_state.setdefault("pending_workflow_model_id", None)
    st.session_state.setdefault("pending_prediction_form_task", None)
    st.session_state.setdefault("pending_prediction_form_model_id", "latest")
    st.session_state.setdefault("auto_connect_keys", set())
    st.session_state.setdefault("auto_model_warmup_keys", set())
    st.session_state.setdefault("upload_panel_expanded", True)
    st.session_state.setdefault("upload_panel_auto_collapsed_once", False)
    st.session_state.setdefault("upload_panel_collapse_requested", False)
    st.session_state.setdefault("confirm_chat_clear", False)
    if sanitize_persisted_chat_messages():
        persist_ui_state()


def _sanitize_internal_tool_text(text: str) -> str:
    replacements = {
        "`predict_regression_batch` 실행 결과입니다.": "regression 예측 결과입니다.",
        "`predict_classification_batch` 실행 결과입니다.": "classification 예측 결과입니다.",
        "preview는 화면의 tool JSON/detail에서 확인하세요. 전체 dataframe은 prompt에 넣지 않았습니다.": "미리보기와 전체 다운로드는 결과 메시지를 펼치면 확인할 수 있습니다.",
        "Tool 실행 실패:": "요청 처리 실패:",
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _sanitize_tool_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    changed = False
    tool = str(payload.get("tool") or "")
    if tool == "predict_regression_batch":
        payload["tool"] = "prediction.batch"
        payload["task"] = "regression"
        changed = True
    elif tool == "predict_classification_batch":
        payload["tool"] = "prediction.batch"
        payload["task"] = "classification"
        changed = True
    return changed


def sanitize_persisted_chat_messages() -> bool:
    messages = st.session_state.get("chat_messages", [])
    if not isinstance(messages, list):
        return False
    changed = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _sanitize_internal_tool_text(content)
            if sanitized != content:
                msg["content"] = sanitized
                changed = True
        if _sanitize_tool_payload(msg.get("tool_result")):
            changed = True
        tool_results = msg.get("tool_results")
        if isinstance(tool_results, list):
            for payload in tool_results:
                if _sanitize_tool_payload(payload):
                    changed = True
    return changed


def _persistable_chat_messages() -> list[dict[str, Any]]:
    messages = []
    for msg in st.session_state.get("chat_messages", []):
        if isinstance(msg, dict):
            messages.append(_jsonable(msg))
    return messages


def _safe_state_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe[:160] or "item"


def restore_persistent_ui_state() -> None:
    if st.session_state.get("_persistent_ui_state_restored"):
        return
    st.session_state["_persistent_ui_state_restored"] = True
    if not UI_STATE_PATH.exists():
        return
    try:
        payload = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    for key in (
        "chat_messages",
        "active_dataset_id",
        "latest_result_id",
        "selected_task",
        "selected_model_id",
        "pending_prediction_form_task",
        "pending_prediction_form_model_id",
        "upload_panel_expanded",
        "upload_panel_auto_collapsed_once",
    ):
        if key in payload:
            st.session_state[key] = payload[key]

    completed = payload.get("workflow_completed_dataset_ids")
    if isinstance(completed, list):
        st.session_state["workflow_completed_dataset_ids"] = set(str(item) for item in completed)

    datasets: dict[str, Any] = {}
    for dataset_id in payload.get("dataset_ids", []) if isinstance(payload.get("dataset_ids"), list) else []:
        path = PERSISTED_DATASETS_DIR / f"{_safe_state_filename(str(dataset_id))}.pkl"
        if not path.exists():
            continue
        try:
            datasets[str(dataset_id)] = pd.read_pickle(path)
        except Exception:
            continue
    if datasets:
        st.session_state["datasets"] = datasets

    results: dict[str, pd.DataFrame] = {}
    for result_id in payload.get("prediction_result_ids", []) if isinstance(payload.get("prediction_result_ids"), list) else []:
        path = PERSISTED_RESULTS_DIR / f"{_safe_state_filename(str(result_id))}.pkl"
        if not path.exists():
            continue
        try:
            results[str(result_id)] = pd.read_pickle(path)
        except Exception:
            continue
    if results:
        st.session_state["prediction_results"] = results


def persist_ui_state() -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        PERSISTED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        PERSISTED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        datasets = st.session_state.get("datasets", {})
        results = st.session_state.get("prediction_results", {})
        dataset_ids = []
        if isinstance(datasets, dict):
            for dataset_id, record in datasets.items():
                dataset_ids.append(str(dataset_id))
                pd.to_pickle(record, PERSISTED_DATASETS_DIR / f"{_safe_state_filename(str(dataset_id))}.pkl")
        result_ids = []
        if isinstance(results, dict):
            for result_id, df in results.items():
                if isinstance(df, pd.DataFrame):
                    result_ids.append(str(result_id))
                    df.to_pickle(PERSISTED_RESULTS_DIR / f"{_safe_state_filename(str(result_id))}.pkl")
        completed = st.session_state.get("workflow_completed_dataset_ids", set())
        payload = {
            "chat_messages": _persistable_chat_messages(),
            "active_dataset_id": st.session_state.get("active_dataset_id"),
            "latest_result_id": st.session_state.get("latest_result_id"),
            "selected_task": st.session_state.get("selected_task"),
            "selected_model_id": st.session_state.get("selected_model_id"),
            "pending_prediction_form_task": st.session_state.get("pending_prediction_form_task"),
            "pending_prediction_form_model_id": st.session_state.get("pending_prediction_form_model_id"),
            "upload_panel_expanded": st.session_state.get("upload_panel_expanded", True),
            "upload_panel_auto_collapsed_once": st.session_state.get("upload_panel_auto_collapsed_once", False),
            "workflow_completed_dataset_ids": sorted(str(item) for item in completed) if isinstance(completed, set) else [],
            "dataset_ids": dataset_ids,
            "prediction_result_ids": result_ids,
        }
        UI_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def clear_persistent_chat_state() -> None:
    st.session_state.chat_messages = []
    st.session_state.prediction_results = {}
    st.session_state.latest_result_id = None
    st.session_state.pending_prediction_form_task = None
    st.session_state.pending_prediction_form_model_id = "latest"
    st.session_state.pending_workflow_dataset_id = None
    st.session_state.pending_workflow_source_message = None
    st.session_state.pending_workflow_force = False
    clear_pending_workflow_runtime_state()
    try:
        if UI_STATE_PATH.exists():
            UI_STATE_PATH.unlink()
        if PERSISTED_RESULTS_DIR.exists():
            for path in PERSISTED_RESULTS_DIR.glob("*.pkl"):
                path.unlink()
    except Exception:
        pass


def clear_pending_workflow_runtime_state() -> None:
    st.session_state.pending_workflow_stage = None
    st.session_state.pending_workflow_train_result = None
    st.session_state.pending_workflow_task = None
    st.session_state.pending_workflow_model_id = None


def schedule_post_training_workflow_steps(*, train_result: dict[str, Any], task: str, model_id: str) -> None:
    st.session_state.pending_workflow_stage = "schema"
    st.session_state.pending_workflow_train_result = train_result
    st.session_state.pending_workflow_task = task
    st.session_state.pending_workflow_model_id = model_id


def render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          color-scheme: dark light;
          --hd-ui-text: #e5e7eb;
          --hd-ui-muted: #cbd5e1;
          --hd-ui-subtle: #aeb4bd;
          --hd-ui-panel: #0f172a;
          --hd-ui-panel-soft: #111827;
          --hd-ui-border: rgba(226, 232, 240, .18);
          --hd-ui-table-head: rgba(226, 232, 240, .08);
          --hd-ui-code-bg: rgba(226, 232, 240, .08);
          --hd-ui-code-text: #f8fafc;
          --hd-upload-text: #e5e7eb;
          --hd-info-bg: rgba(56, 189, 248, .10);
          --hd-info-border: rgba(56, 189, 248, .28);
          --hd-reasoning-text: #aeb4bd;
          --hd-reasoning-bg: rgba(226, 232, 240, .06);
          --hd-reasoning-border: rgba(148, 163, 184, .22);
          --hd-reasoning-spinner: rgba(203, 213, 225, .55);
          --hd-avatar-bg: #0f2a44;
          --hd-icon-color: #e5e7eb;
        }
        @media (prefers-color-scheme: light) {
          :root {
            --hd-ui-text: #0f172a;
            --hd-ui-muted: #475569;
            --hd-ui-subtle: #64748b;
            --hd-ui-panel: #ffffff;
            --hd-ui-panel-soft: #f8fafc;
            --hd-ui-border: rgba(15, 23, 42, .14);
            --hd-ui-table-head: rgba(15, 23, 42, .045);
            --hd-ui-code-bg: rgba(15, 23, 42, .045);
            --hd-ui-code-text: #111827;
            --hd-upload-text: #334155;
            --hd-info-bg: rgba(14, 116, 144, .08);
            --hd-info-border: rgba(14, 116, 144, .22);
            --hd-reasoning-text: #64748b;
            --hd-reasoning-bg: rgba(15, 23, 42, .035);
            --hd-reasoning-border: rgba(100, 116, 139, .24);
            --hd-reasoning-spinner: rgba(100, 116, 139, .48);
            --hd-avatar-bg: #e8eef8;
            --hd-icon-color: #1e293b;
          }
        }
        [data-theme="light"],
        [data-baseweb-theme="light"] {
          --hd-ui-text: #0f172a;
          --hd-ui-muted: #475569;
          --hd-ui-subtle: #64748b;
          --hd-ui-panel: #ffffff;
          --hd-ui-panel-soft: #f8fafc;
          --hd-ui-border: rgba(15, 23, 42, .14);
          --hd-ui-table-head: rgba(15, 23, 42, .045);
          --hd-ui-code-bg: rgba(15, 23, 42, .045);
          --hd-ui-code-text: #111827;
          --hd-upload-text: #334155;
          --hd-info-bg: rgba(14, 116, 144, .08);
          --hd-info-border: rgba(14, 116, 144, .22);
          --hd-reasoning-text: #64748b;
          --hd-reasoning-bg: rgba(15, 23, 42, .035);
          --hd-reasoning-border: rgba(100, 116, 139, .24);
          --hd-reasoning-spinner: rgba(100, 116, 139, .48);
          --hd-avatar-bg: #e8eef8;
          --hd-icon-color: #1e293b;
        }
        [data-theme="dark"],
        [data-baseweb-theme="dark"] {
          --hd-ui-text: #e5e7eb;
          --hd-ui-muted: #cbd5e1;
          --hd-ui-subtle: #aeb4bd;
          --hd-ui-panel: #0f172a;
          --hd-ui-panel-soft: #111827;
          --hd-ui-border: rgba(226, 232, 240, .18);
          --hd-ui-table-head: rgba(226, 232, 240, .08);
          --hd-ui-code-bg: rgba(226, 232, 240, .08);
          --hd-ui-code-text: #f8fafc;
          --hd-upload-text: #e5e7eb;
          --hd-info-bg: rgba(56, 189, 248, .10);
          --hd-info-border: rgba(56, 189, 248, .28);
          --hd-reasoning-text: #aeb4bd;
          --hd-reasoning-bg: rgba(226, 232, 240, .06);
          --hd-reasoning-border: rgba(148, 163, 184, .22);
          --hd-reasoning-spinner: rgba(203, 213, 225, .55);
          --hd-avatar-bg: #0f2a44;
          --hd-icon-color: #e5e7eb;
        }
        .stApp, [data-testid="stAppViewContainer"] { color: var(--hd-ui-text); }
        .block-container { max-width: 1320px; padding-top: 2.8rem; padding-bottom: 4rem; }
        .app-hero { margin: .65rem 0 1.1rem; padding: 1.1rem 1.25rem 1.25rem; border-radius: 24px; background: linear-gradient(135deg, #071c35 0%, #0b2c55 58%, #123965 100%); color: #fff; box-shadow: 0 22px 60px rgba(7,28,53,.18); }
        .app-hero-logo-row { display: flex; align-items: center; margin-bottom: .9rem; min-height: 42px; }
        .app-hero-logo { max-height: 42px; width: auto; object-fit: contain; display: block; }
        .app-title-row { display: flex; align-items: center; gap: .85rem; flex-wrap: wrap; }
        .app-title-row h1 { margin: 0; color: #fff; font-size: clamp(2rem, 3.7vw, 3.25rem); line-height: 1.03; letter-spacing: -.045em; font-weight: 820; }
        .app-byline { display: inline-flex; align-items: center; gap: .5rem; transform: translateY(.12rem); }
        .app-byline span { color: rgba(255,255,255,.72); font-size: clamp(1rem, 1.85vw, 1.55rem); line-height: 1; font-weight: 650; letter-spacing: -.02em; }
        .app-aim-logo { height: clamp(1.25rem, 2.05vw, 1.8rem); width: auto; object-fit: contain; display: block; background: rgba(255,255,255,.92); border-radius: 9px; padding: .16rem .32rem; }
        .app-aim-lab { color: #fff; font-size: clamp(1rem, 1.85vw, 1.55rem); line-height: 1; font-weight: 850; letter-spacing: -.03em; margin-left: -.25rem; }
        .app-hero-caption { margin: .75rem 0 0; color: rgba(255,255,255,.74); font-size: .96rem; }
        .chat-clear-anchor + div button { min-height: 3.25rem; }
        .chat-clear-anchor + div button p { white-space: nowrap; }
        .training-upload-title { color: var(--hd-upload-text); font-size: 1.08rem; font-weight: 820; margin: .2rem 0 .45rem; }
        .training-upload-box [data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] { color: var(--hd-upload-text); }
        .training-upload-box [data-testid="stFileUploader"] label,
        .training-upload-box [data-testid="stFileUploader"] label p,
        .training-upload-box [data-testid="stFileUploader"] small,
        .training-upload-box [data-testid="stFileUploader"] span,
        .training-upload-box [data-testid="stFileUploader"] section,
        .training-upload-box [data-testid="stFileUploader"] div,
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] label p,
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] div { color: var(--hd-upload-text) !important; }
        .training-upload-box [data-testid="stFileUploader"] label p,
        div[data-testid="stFileUploader"] label p { font-size: 1.04rem; font-weight: 820; }
        .active-dataset-banner { margin: .15rem 0 .75rem; padding: .72rem .85rem; border-radius: 14px; background: var(--hd-info-bg); border: 1px solid var(--hd-info-border); color: var(--hd-ui-text); font-size: 1.01rem; font-weight: 680; }
        @media (max-width: 720px) {
          .app-hero { padding: .95rem 1rem 1.05rem; border-radius: 18px; }
          .app-hero-logo { max-height: 34px; }
          .app-byline { width: 100%; }
        }
        .hd-note { border: 1px solid var(--hd-ui-border); border-radius: 10px; padding: .8rem 1rem; background: var(--hd-ui-panel-soft); color: var(--hd-ui-text); }
        .hd-small { color: var(--hd-ui-muted); font-size: .86rem; }
        .hd-reasoning { margin: .15rem 0 .55rem; }
        .hd-reasoning summary { color: var(--hd-reasoning-text); font-size: .82rem; cursor: pointer; list-style-position: inside; }
        .hd-reasoning-box { margin-top: .35rem; max-height: 11.5rem; overflow-y: auto; overscroll-behavior: contain; border: 1px solid var(--hd-reasoning-border); border-radius: 12px; background: var(--hd-reasoning-bg); padding: .65rem .75rem; scroll-behavior: smooth; }
        .hd-reasoning-box:focus { outline: 1px solid rgba(56, 189, 248, .55); outline-offset: 2px; }
        .hd-reasoning-content { color: var(--hd-reasoning-text); font-size: .78rem; white-space: pre-wrap; line-height: 1.44; background: transparent; border: 0; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
        .hd-reasoning-box::-webkit-scrollbar { width: 8px; }
        .hd-reasoning-box::-webkit-scrollbar-track { background: transparent; }
        .hd-reasoning-box::-webkit-scrollbar-thumb { background: rgba(148,163,184,.42); border-radius: 99px; }
        .hd-live-reasoning { color: var(--hd-reasoning-text); font-size: .82rem; display: inline-flex; align-items: center; gap: .45rem; }
        .hd-reasoning-dot { width: .75rem; height: .75rem; border: 2px solid var(--hd-reasoning-spinner); border-top-color: var(--hd-reasoning-text); border-radius: 50%; display: inline-block; animation: hd-spin .8s linear infinite; }
        .report-model-id { margin: .2rem 0 .65rem; color: var(--hd-ui-muted); font-size: .82rem; line-height: 1.38; font-weight: 400; }
        .report-model-id-label { display: block; font-weight: 400; }
        .report-model-id-value { display: block; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .78rem; font-weight: 400; word-break: break-all; overflow-wrap: anywhere; white-space: normal; }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] { color: var(--hd-ui-text) !important; border-color: var(--hd-ui-border) !important; }
        [data-testid="stDataFrame"] div,
        [data-testid="stDataFrame"] span,
        [data-testid="stTable"] * { color: var(--hd-ui-text) !important; }
        [data-testid="stTable"] table { background: var(--hd-ui-panel) !important; }
        [data-testid="stTable"] th { background: var(--hd-ui-table-head) !important; color: var(--hd-ui-text) !important; border-color: var(--hd-ui-border) !important; }
        [data-testid="stTable"] td { border-color: var(--hd-ui-border) !important; color: var(--hd-ui-text) !important; }
        [data-testid="stMetric"],
        [data-testid="stMetric"] * { color: var(--hd-ui-text) !important; }
        [data-testid="stJson"] pre,
        pre,
        code { color: var(--hd-ui-code-text); background-color: var(--hd-ui-code-bg); }
        [data-testid="stChatMessageAvatar"] { background: var(--hd-avatar-bg) !important; border: 1px solid var(--hd-ui-border) !important; color: var(--hd-icon-color) !important; }
        [data-testid="stChatMessageAvatar"] svg,
        [data-testid="stIconMaterial"],
        [data-testid="stIconMaterial"] svg { color: var(--hd-icon-color) !important; fill: currentColor !important; }
        @keyframes hd-spin { to { transform: rotate(360deg); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def scroll_to_chat_bottom() -> None:
    marker_key = f"{len(st.session_state.chat_messages)}-{st.session_state.latest_result_id or ''}-{st.session_state.pending_workflow_dataset_id or ''}"
    if st.session_state.get("last_scroll_marker") == marker_key:
        return
    st.session_state.last_scroll_marker = marker_key
    components.html(
        """
        <script>
        const scroll = () => {
          const doc = window.parent.document;
          const target = doc.getElementById("chat-bottom-anchor");
          if (target) target.scrollIntoView({behavior: "smooth", block: "end"});
        };
        setTimeout(scroll, 80);
        setTimeout(scroll, 400);
        </script>
        """,
        height=0,
    )


def backend_defaults(config: dict[str, Any], backend: str) -> dict[str, Any]:
    defaults = {
        "ollama": {"base_url": "http://127.0.0.1:11434", "model": "gemma4:e2b"},
        "vllm": {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "api_key": "EMPTY",
            "context_window": 8192,
            "max_context_tokens": 8192,
        },
    }
    configured = config.get(backend, {}) if isinstance(config.get(backend), dict) else {}
    return {**defaults[backend], **configured}


def model_options_key(backend: str, base_url: str) -> str:
    return f"{backend}:{base_url.strip().rstrip('/')}"


def update_llm_model_options(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    result = check_health(config, runtime)
    models = [str(model) for model in result.get("models", []) if model]
    st.session_state.llm_model_options[model_options_key(str(result.get("backend") or runtime["backend"]), str(result.get("base_url") or runtime["base_url"]))] = models
    st.session_state.llm_model_connect_status = result
    st.session_state.health_result = result
    return result


def connect_once(config: dict[str, Any], runtime: dict[str, Any]) -> None:
    keys = st.session_state.auto_connect_keys
    if not isinstance(keys, set):
        keys = set(keys)
        st.session_state.auto_connect_keys = keys
    key = model_options_key(str(runtime.get("backend") or ""), str(runtime.get("base_url") or ""))
    if key in keys:
        return
    keys.add(key)
    try:
        update_llm_model_options(config, runtime)
    except Exception as exc:
        st.session_state.llm_model_connect_status = {"ok": False, "error": str(exc)}
        st.session_state.health_result = {"ok": False, "error": str(exc)}


def selected_model_key(runtime: dict[str, Any]) -> str:
    backend = str(runtime.get("backend") or "").strip().lower()
    base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
    model = str(runtime.get("model") or "").strip()
    return f"{backend}:{base_url}:{model}"


def set_model_load_status(key: str, payload: dict[str, Any]) -> None:
    with MODEL_LOAD_LOCK:
        MODEL_LOAD_STATUS[key] = {"updated_at": round(time.time(), 3), **payload}


def get_model_load_status(key: str) -> dict[str, Any] | None:
    with MODEL_LOAD_LOCK:
        value = MODEL_LOAD_STATUS.get(key)
        return dict(value) if isinstance(value, dict) else None


def managed_vllm_runtime(config: dict[str, Any]) -> NemoClawVLLMRuntime:
    vllm_cfg = backend_defaults(config, "vllm")
    raw_runtime = vllm_cfg.get("nemoclaw_k8s")
    runtime_cfg = raw_runtime if isinstance(raw_runtime, dict) else default_nemoclaw_vllm_config()
    return NemoClawVLLMRuntime.from_config(runtime_cfg)


def connect_managed_vllm_model(config: dict[str, Any], model: str) -> dict[str, Any]:
    runtime = managed_vllm_runtime(config)
    result = runtime.load_model(model)
    return {
        "ok": bool(result.get("loaded")),
        "backend": "vllm",
        "model": model,
        "state": "connected" if result.get("loaded") else "loading",
        "base_url": result.get("base_url") or runtime.base_url_for_model(model),
        "result": result,
    }


def disconnect_managed_vllm_model(config: dict[str, Any], model: str) -> dict[str, Any]:
    runtime = managed_vllm_runtime(config)
    result = runtime.unload_model(model)
    return {
        "ok": bool(result.get("unloaded")),
        "backend": "vllm",
        "model": model,
        "state": "disconnected",
        "result": result,
    }


def load_selected_llm_model(runtime: dict[str, Any], *, source: str = "manual") -> dict[str, Any]:
    backend = str(runtime.get("backend") or "").lower()
    base_url = str(runtime.get("base_url") or "").rstrip("/")
    model = str(runtime.get("model") or "").strip()
    timeout = float(runtime.get("timeout_sec") or st.session_state.config.get("timeout_sec") or 240)
    if not base_url or not model:
        raise RuntimeError("Base URL과 Model을 먼저 지정해야 합니다.")

    started = time.time()
    if backend == "ollama":
        raw = http_post_json(
            f"{base_url}/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply OK only."}],
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 2},
            },
            timeout=timeout,
        )
        message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
        return {
            "ok": True,
            "backend": backend,
            "model": model,
            "state": "loaded",
            "source": source,
            "elapsed_sec": round(time.time() - started, 3),
            "probe": str(message.get("content") or raw.get("response") or "").strip(),
        }

    if backend == "vllm":
        headers = {"Authorization": f"Bearer {runtime.get('api_key')}"} if runtime.get("api_key") else {}
        raw = http_post_json(
            f"{base_url}/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply OK only."}],
                "temperature": 0,
                "max_tokens": 2,
                "stream": False,
            },
            headers=headers,
            timeout=timeout,
        )
        return {
            "ok": True,
            "backend": backend,
            "model": model,
            "state": "reachable",
            "source": source,
            "elapsed_sec": round(time.time() - started, 3),
            "probe": "chat/completions OK",
            "raw_model": raw.get("model"),
        }

    raise RuntimeError(f"지원하지 않는 backend입니다: {backend}")


def unload_selected_llm_model(runtime: dict[str, Any]) -> dict[str, Any]:
    backend = str(runtime.get("backend") or "").lower()
    base_url = str(runtime.get("base_url") or "").rstrip("/")
    model = str(runtime.get("model") or "").strip()
    timeout = float(runtime.get("timeout_sec") or st.session_state.config.get("timeout_sec") or 240)
    if not base_url or not model:
        raise RuntimeError("Base URL과 Model을 먼저 지정해야 합니다.")
    if backend != "ollama":
        return {
            "ok": False,
            "backend": backend,
            "model": model,
            "state": "unsupported",
            "message": "OpenAI-compatible vLLM API에는 범용 unload endpoint가 없습니다.",
        }
    started = time.time()
    raw = http_post_json(f"{base_url}/api/generate", {"model": model, "keep_alive": 0}, timeout=timeout)
    return {
        "ok": True,
        "backend": backend,
        "model": model,
        "state": "unloaded",
        "elapsed_sec": round(time.time() - started, 3),
        "raw_done": raw.get("done"),
    }


def start_model_warmup_once(runtime: dict[str, Any]) -> None:
    keys = st.session_state.auto_model_warmup_keys
    if not isinstance(keys, set):
        keys = set(keys)
        st.session_state.auto_model_warmup_keys = keys
    key = selected_model_key(runtime)
    if not key or key in keys:
        return
    keys.add(key)
    set_model_load_status(key, {"ok": None, "state": "loading", "source": "server_start", "message": "background probe started"})

    def worker() -> None:
        try:
            result = load_selected_llm_model(dict(runtime), source="server_start")
            set_model_load_status(key, result)
        except Exception as exc:
            set_model_load_status(key, {"ok": False, "state": "error", "source": "server_start", "error": str(exc)})
        finally:
            with MODEL_LOAD_LOCK:
                MODEL_LOAD_THREADS.discard(key)

    with MODEL_LOAD_LOCK:
        if key in MODEL_LOAD_THREADS:
            return
        MODEL_LOAD_THREADS.add(key)
    threading.Thread(target=worker, daemon=True, name=f"llm-warmup-{hashlib.sha1(key.encode()).hexdigest()[:8]}").start()


def sync_ollama_loaded_model(runtime: dict[str, Any]) -> None:
    if str(runtime.get("backend") or "").lower() != "ollama":
        return
    base_url = str(runtime.get("base_url") or "").rstrip("/")
    selected_model = str(runtime.get("model") or "").strip()
    if not base_url or not selected_model:
        return
    try:
        data = http_get_json(f"{base_url}/api/ps", timeout=3)
        loaded = data.get("models", []) if isinstance(data.get("models"), list) else []
        for item in loaded:
            if not isinstance(item, dict):
                continue
            loaded_name = str(item.get("name") or item.get("model") or "").strip()
            if loaded_name and loaded_name != selected_model:
                http_post_json(f"{base_url}/api/generate", {"model": loaded_name, "keep_alive": 0}, timeout=10)
    except Exception as exc:
        st.session_state.llm_model_connect_status = {"ok": False, "error": f"Ollama loaded-model sync failed: {exc}"}


def tool_context() -> ToolContext:
    return ToolContext(
        datasets=st.session_state.datasets,
        active_dataset_id=st.session_state.active_dataset_id,
        results=st.session_state.prediction_results,
        model_root=MODEL_ROOT,
    )


def artifact_ids(task: str) -> list[str]:
    ids = [str(row["model_id"]) for row in list_artifacts(task, MODEL_ROOT) if row.get("has_model") and row.get("model_id") != "latest"]
    return ids or ["latest"]


def active_dataset():
    dataset_id = st.session_state.active_dataset_id
    datasets = st.session_state.datasets
    if not dataset_id or dataset_id not in datasets:
        return None
    return datasets[dataset_id]


def render_sidebar() -> dict[str, Any]:
    config = st.session_state.config
    st.sidebar.title("Runtime")
    st.sidebar.subheader("Local LLM")
    default_backend = str(config.get("default_backend") or "ollama")
    backend = st.sidebar.radio("Backend", BACKENDS, index=BACKENDS.index(default_backend) if default_backend in BACKENDS else 0, horizontal=True)
    backend_cfg = backend_defaults(config, backend)
    if backend == "vllm":
        base_url = st.sidebar.text_input("Base URL", value=str(backend_cfg.get("base_url") or "http://127.0.0.1:8000/v1"))
        api_key = st.sidebar.text_input("API Key", value=str(backend_cfg.get("api_key") or "EMPTY"), type="password")
    else:
        base_url = st.sidebar.text_input("Base URL", value=str(backend_cfg.get("base_url") or ""))
        api_key = ""
    connect_runtime = {
        "backend": backend,
        "base_url": base_url.strip(),
        "api_key": api_key if backend == "vllm" else "",
    }
    if backend != "vllm":
        connect_once(config, connect_runtime)
    configured_model = str(backend_cfg.get("model") or "")
    options_key = model_options_key(backend, base_url)
    connected_model_options = list(st.session_state.llm_model_options.get(options_key, []))
    model_options = list(connected_model_options)
    if not model_options and configured_model:
        model_options.insert(0, configured_model)
    if not model_options:
        model_options = [str(backend_cfg.get("model") or "")]
    selected_index = model_options.index(configured_model) if configured_model in model_options else 0
    model = st.sidebar.selectbox("Model", model_options, index=selected_index)
    connect_status = st.session_state.llm_model_connect_status
    if connect_status and connect_status.get("ok") and connected_model_options:
        st.sidebar.caption(f"Connected: {len(connected_model_options)} model(s)")
    elif connect_status and connect_status.get("ok") is False:
        st.sidebar.caption(f"Connect failed: {connect_status.get('error')}")
    else:
        st.sidebar.caption("Ollama와 vLLM은 Health로 endpoint 상태와 모델 목록을 확인합니다.")
    timeout_sec = st.sidebar.number_input("Timeout Sec", min_value=10, max_value=3600, value=int(config.get("timeout_sec", 240)), step=10)
    model_runtime = {
        "backend": backend,
        "base_url": base_url.strip(),
        "model": model.strip(),
        "api_key": api_key,
        "timeout_sec": int(timeout_sec),
    }
    sync_ollama_loaded_model(model_runtime)

    st.sidebar.caption("Model control")
    load_col, unload_col = st.sidebar.columns(2)
    model_key = selected_model_key(model_runtime)
    if load_col.button("Loading", use_container_width=True):
        set_model_load_status(model_key, {"ok": None, "state": "loading", "source": "manual", "message": "manual probe started"})
        try:
            result = load_selected_llm_model(model_runtime, source="manual")
            set_model_load_status(model_key, result)
            st.session_state.health_result = {"ok": True, "backend": backend, "model": model, "message": "model probe OK"}
        except Exception as exc:
            set_model_load_status(model_key, {"ok": False, "state": "error", "source": "manual", "error": str(exc)})
            st.session_state.health_result = {"ok": False, "error": str(exc)}
    if unload_col.button("Unloading", use_container_width=True):
        try:
            result = unload_selected_llm_model(model_runtime)
            set_model_load_status(model_key, result)
            st.session_state.health_result = {"ok": bool(result.get("ok")), "backend": backend, "model": model, "message": result.get("message") or result.get("state")}
        except Exception as exc:
            set_model_load_status(model_key, {"ok": False, "state": "error", "source": "manual_unload", "error": str(exc)})
            st.session_state.health_result = {"ok": False, "error": str(exc)}
    model_status = get_model_load_status(model_key)
    if model_status:
        if model_status.get("ok") is True:
            st.sidebar.success(f"{model_status.get('state')}: {model_status.get('elapsed_sec')} sec")
        elif model_status.get("ok") is False:
            st.sidebar.error(model_status.get("error") or model_status.get("message") or model_status.get("state"))
        else:
            st.sidebar.info("Loading probe is running in background.")

    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, float(config.get("temperature", 0.3)), 0.05)
    system_prompt = st.sidebar.text_area("System Prompt", value=str(config.get("system_prompt") or ""), height=120)
    runtime = {
        **model_runtime,
        "temperature": float(temperature),
        "system_prompt": system_prompt,
    }
    runtime["max_tokens"] = int(config.get("max_tokens", 2048))
    if backend == "vllm":
        if backend_cfg.get("context_window"):
            runtime["context_window"] = int(backend_cfg["context_window"])
        if backend_cfg.get("max_context_tokens"):
            runtime["max_context_tokens"] = int(backend_cfg["max_context_tokens"])
        template_kwargs = backend_cfg.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            runtime["chat_template_kwargs"] = template_kwargs

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Save LLM", use_container_width=True):
        next_cfg = {
            "default_backend": backend,
            "timeout_sec": int(timeout_sec),
            "temperature": float(temperature),
            "system_prompt": system_prompt,
            backend: {"base_url": runtime["base_url"], "model": runtime["model"]},
        }
        if backend == "vllm":
            next_cfg["vllm"]["api_key"] = api_key or "EMPTY"
        st.session_state.config = deep_merge(config, next_cfg)
        save_config(st.session_state.config)
        st.sidebar.success("config.json 저장 완료")
    if col2.button("Health" if backend != "vllm" else "Status", use_container_width=True):
        try:
            st.session_state.health_result = update_llm_model_options(config, runtime)
        except Exception as exc:
            st.session_state.health_result = {"ok": False, "error": str(exc)}
    if st.session_state.health_result:
        if st.session_state.health_result.get("ok"):
            st.sidebar.success(f"{st.session_state.health_result.get('backend')} OK")
        else:
            st.sidebar.error(st.session_state.health_result.get("error"))
    return runtime


def tab_upload() -> None:
    st.header("1. Excel 업로드")
    uploaded = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx", "xls"])
    if uploaded and st.button("업로드 데이터 등록", type="primary"):
        try:
            df = read_excel_file(uploaded)
            inferred = infer_dataset_type(list(df.columns))
            record = make_dataset_record(uploaded.name, df, inferred)
            st.session_state.datasets[record.dataset_id] = record
            st.session_state.active_dataset_id = record.dataset_id
            st.success(f"등록 완료: {record.dataset_id}")
        except Exception as exc:
            st.error(f"Excel 로딩 실패: {exc}")
    record = active_dataset()
    if not record:
        st.info("현재 활성 데이터셋이 없습니다.")
        return
    df = record.dataframe
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Inferred Type", record.inferred_type)
    st.caption(f"파일: {record.filename} / 업로드: {record.uploaded_at}")
    st.write("Columns")
    st.code(", ".join(map(str, df.columns)))
    st.dataframe(df.head(50), use_container_width=True)


def tab_validation() -> None:
    st.header("2. 데이터 검증")
    record = active_dataset()
    if not record:
        st.info("Excel 업로드 후 사용할 수 있습니다.")
        return
    df = record.dataframe
    current_type = record.manual_type or record.inferred_type
    index = DATASET_TYPES.index(current_type) if current_type in DATASET_TYPES else len(DATASET_TYPES) - 1
    record.manual_type = st.selectbox("Dataset Type Override", DATASET_TYPES, index=index)
    st.write("기본 요약")
    st.json(dataframe_summary(df, max_preview_rows=5), expanded=False)
    st.write("결측치 Summary")
    st.dataframe(missing_summary(df), use_container_width=True)
    dist = target_distribution(df, record.dataset_type)
    if dist:
        st.write("Target Distribution / Summary")
        st.json(dist)
    task = st.session_state.selected_task
    model_id = st.session_state.selected_model_id
    try:
        artifact = load_artifact(task, model_id, MODEL_ROOT)
        validation = validate_columns_against_schema(list(df.columns), artifact["schema"])
        st.write(f"저장된 `{task}` schema 비교: `{model_id}`")
        st.json(validation)
    except Exception as exc:
        st.warning(f"저장된 {task} 모델을 아직 로드할 수 없습니다: {exc}")


def tab_training() -> None:
    st.header("3. 모델 학습")
    record = active_dataset()
    if not record:
        st.info("Excel 업로드 후 학습할 수 있습니다.")
        return
    task = st.radio("Training Task", TASKS, index=TASKS.index(st.session_state.selected_task), horizontal=True)
    train_size = st.slider("Train Size", 0.5, 0.95, 0.9, 0.05)
    random_state = st.number_input("Random State / Session ID", min_value=0, max_value=999999, value=0, step=1)
    st.caption("Classification은 `Result == 1.0 -> label 1`, else `0`; feature에서 `Result`, `TRVmax[kV]`를 제거합니다.")
    if st.button("Train Model", type="primary"):
        with st.spinner("모델 후보를 학습하고 holdout metric을 계산 중입니다..."):
            result = train_tool(tool_context(), task=task, options={"train_size": float(train_size), "random_state": int(random_state)})
        if not result.get("ok"):
            st.error(result.get("message"))
            st.json(result)
        else:
            st.success(f"{task} 학습 완료: {result.get('model_id')}")
            st.json(result.get("metrics", {}))


def _download_widget_key(basename: str, kind: str, df: pd.DataFrame) -> str:
    counter = int(st.session_state.get("_download_widget_counter", 0)) + 1
    st.session_state["_download_widget_counter"] = counter
    signature = f"{basename}:{kind}:{counter}:{df.shape}:{','.join(map(str, df.columns))}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"download_{kind}_{counter}_{digest}"


def _result_downloads(df: pd.DataFrame, basename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8-sig")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="prediction")
    col1, col2 = st.columns(2)
    col1.download_button(
        "CSV 다운로드",
        data=csv,
        file_name=f"{basename}.csv",
        mime="text/csv",
        use_container_width=True,
        key=_download_widget_key(basename, "csv", df),
    )
    col2.download_button(
        "Excel 다운로드",
        data=buffer.getvalue(),
        file_name=f"{basename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=_download_widget_key(basename, "xlsx", df),
    )


def tab_prediction() -> None:
    st.header("4. 모델 예측")
    record = active_dataset()
    if not record:
        st.info("Excel 업로드 후 예측할 수 있습니다.")
        return
    task = st.radio("Prediction Task", TASKS, index=TASKS.index(st.session_state.selected_task), horizontal=True, key="prediction_task")
    model_id = st.selectbox("Saved Model", artifact_ids(task), key=f"predict_model_{task}")
    try:
        artifact = load_artifact(task, model_id, MODEL_ROOT)
        validation = validate_columns_against_schema(list(record.dataframe.columns), artifact["schema"])
        if not validation["ok"]:
            st.error(f"Schema mismatch. Missing columns: {validation['missing_columns']}")
            st.json(validation)
            return
        st.success("Schema OK. 예측 가능합니다.")
        st.json({"ignored_present": validation["ignored_present"], "extra_columns": validation["extra_columns"]})
    except Exception as exc:
        st.error(f"저장 모델 로드 실패: {exc}")
        return
    row_index = st.number_input("Single-row preview index", min_value=0, max_value=max(len(record.dataframe) - 1, 0), value=0, step=1)
    st.dataframe(record.dataframe.iloc[[int(row_index)]], use_container_width=True)
    if st.button("Batch Prediction 실행", type="primary"):
        try:
            result_df, summary = predict_batch(record.dataframe, task=task, model_id=model_id, model_root=MODEL_ROOT)
            result_id = f"ui-{task}-{len(st.session_state.prediction_results) + 1}"
            st.session_state.prediction_results[result_id] = result_df
            st.session_state.latest_result_id = result_id
            st.success(f"예측 완료: {result_id}")
            st.json(summary)
            st.dataframe(result_df.head(100), use_container_width=True)
            _result_downloads(result_df, result_id)
        except Exception as exc:
            st.error(f"예측 실패: {exc}")
    if st.session_state.latest_result_id:
        df = st.session_state.prediction_results.get(st.session_state.latest_result_id)
        if df is not None:
            st.write(f"Latest result: `{st.session_state.latest_result_id}`")
            st.dataframe(df.head(100), use_container_width=True)
            _result_downloads(df, st.session_state.latest_result_id)


def _chat_context_prompt() -> str:
    record = active_dataset()
    dataset_context = "No active dataset."
    if record:
        dataset_context = json.dumps(
            {
                "dataset_id": record.dataset_id,
                "filename": record.filename,
                "dataset_type": record.dataset_type,
                "shape": list(record.dataframe.shape),
                "columns": list(map(str, record.dataframe.columns)),
            },
            ensure_ascii=False,
        )
    return (
        "You are a high-voltage circuit breaker performance prediction workspace embedded in the HD Hyundai Electric analysis environment. "
        "Your role is to help users prepare training data, understand validation results, and apply saved prediction models. "
        "You must not invent predictions, probabilities, metrics, or SHAP values. "
        "Prediction/training/schema/data questions must use deterministic tool results. "
        "Answer in Korean.\n"
        f"active_context={dataset_context}\n"
        f"selected_task={st.session_state.selected_task}, selected_model_id={st.session_state.selected_model_id}\n"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_json_dumps(payload: Any, *, limit: int = 12000) -> str:
    text = json.dumps(_jsonable(payload), ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[:limit] + "\n...<truncated>"
    return text


def _tool_call_name_and_args(call: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(call, dict):
        return "", {}
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "").strip()
    args = function.get("arguments") or call.get("arguments") or {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def execute_ollama_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    ctx = tool_context()
    inferred_task = None
    try:
        inferred_task = task_for_dataset_type(ctx.active_dataset().dataset_type)
    except Exception:
        inferred_task = None
    task = str(args.get("task") or inferred_task or st.session_state.selected_task or "classification")
    if task not in TASKS:
        task = st.session_state.selected_task if st.session_state.selected_task in TASKS else "classification"
    model_id = str(args.get("model_id") or st.session_state.selected_model_id or "latest")

    if name == "get_uploaded_data_summary":
        return get_uploaded_data_summary(ctx)
    if name == "infer_uploaded_dataset_type":
        result = infer_uploaded_dataset_type(ctx)
        inferred = str(result.get("inferred_type") or "")
        inferred_task = task_for_dataset_type(inferred)
        if result.get("ok") and inferred_task:
            st.session_state.selected_task = inferred_task
        return result
    if name == "train_model":
        result = train_tool(ctx, task=task)
        if result.get("ok"):
            st.session_state.selected_task = task
            if result.get("model_id"):
                st.session_state.selected_model_id = str(result["model_id"])
        return result
    if name == "validate_schema":
        return validate_schema(ctx, task=task, model_id=model_id)
    if name == "predict_batch":
        result = predict_tool(ctx, task=task, model_id=model_id)
        if result.get("ok") and result.get("result_id"):
            st.session_state.latest_result_id = str(result["result_id"])
        return result
    if name == "get_model_metrics":
        return get_model_metrics(ctx, task=task, model_id=model_id)
    if name == "get_prediction_result_summary":
        result_id = str(args.get("result_id") or st.session_state.latest_result_id or "")
        if not result_id:
            return {"ok": False, "failure_code": "RESULT_ID_REQUIRED", "message": "최근 prediction result가 없습니다."}
        return get_prediction_result_summary(ctx, result_id=result_id)
    return {"ok": False, "failure_code": "UNKNOWN_TOOL", "message": f"지원하지 않는 tool입니다: {name}"}


def user_facing_step_label(name: str) -> str:
    labels = {
        "get_uploaded_data_summary": "데이터 구성을 확인하고 있습니다.",
        "infer_uploaded_dataset_type": "분석 목적을 판별하고 있습니다.",
        "train_model": "예측 모델을 학습하고 있습니다.",
        "validate_schema": "입력 변수 구성을 검증하고 있습니다.",
        "predict_batch": "예측 결과를 계산하고 있습니다.",
        "get_model_metrics": "모델 성능을 정리하고 있습니다.",
        "get_prediction_result_summary": "예측 결과를 요약하고 있습니다.",
    }
    return labels.get(name, "분석을 진행하고 있습니다.")


def compact_tool_result_for_llm(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    metrics = compact.get("metrics")
    if isinstance(metrics, dict):
        metrics = dict(metrics)
        for key in ("compare_models", "holdout_prediction_preview"):
            value = metrics.get(key)
            if isinstance(value, list) and len(value) > 5:
                metrics[key] = value[:5]
        compact["metrics"] = metrics
    preview = compact.get("preview")
    if isinstance(preview, list) and len(preview) > 5:
        compact["preview"] = preview[:5]
    return compact


def ollama_agent_system_prompt() -> str:
    return (
        _chat_context_prompt()
        + "\n"
        + (
            "You are the conversational interface for a high-voltage circuit breaker performance prediction workspace. "
            "Be professional, concise, and user-friendly. "
            "Use internal tools when data summary, type inference, training, validation, prediction, or metric lookup is needed. "
            "Do not mention tool names, function names, implementation details, routing, or workflow internals to the user. "
            "For newly uploaded training datasets, internally perform data check, task identification, model training, validation, and prediction when appropriate. "
            "Never fabricate metrics or predictions. After tools finish, summarize concrete tool results in Korean. "
            "If an internal step fails, explain what the user can do next in plain Korean."
        )
    )


def ollama_tool_agent(
    runtime: dict[str, Any],
    user_prompt: str,
    *,
    status: Any = None,
    max_rounds: int = 8,
) -> dict[str, Any]:
    request_id = f"tool-llm-{time.strftime('%Y%m%dT%H%M%S')}-{hashlib.sha1(str(time.time()).encode('utf-8')).hexdigest()[:8]}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"[request_id={request_id}]\n{user_prompt}"}]
    tool_results: list[dict[str, Any]] = []
    thinking_parts: list[str] = []
    trace: list[str] = [
        f"request_id={request_id}",
        f"backend=ollama",
        f"model={runtime.get('model') or 'unknown'}",
        "LLM tool-agent request started.",
    ]
    final_content = ""

    for round_index in range(max_rounds):
        if status is not None:
            workflow_status_write(status, f"분석 단계 {round_index + 1} 진행 중...")
        response = call_ollama_raw(
            st.session_state.config,
            {
                **runtime,
                "messages": messages,
                "system_prompt": ollama_agent_system_prompt() + "\n" + str(runtime.get("system_prompt") or ""),
                "tools": OLLAMA_TOOL_DEFS,
                "think": True,
            },
        )
        if response.get("reasoning"):
            thinking_parts.append(str(response["reasoning"]))
        tool_calls = response.get("tool_calls") if isinstance(response.get("tool_calls"), list) else []
        final_content = str(response.get("content") or "")
        trace.append(
            f"round={round_index + 1}, model={response.get('model')}, "
            f"tool_calls={len(tool_calls)}, elapsed_sec={response.get('elapsed_sec')}"
        )
        messages.append(
            {
                "role": "assistant",
                "content": final_content,
                "thinking": str(response.get("reasoning") or ""),
                "tool_calls": tool_calls,
            }
        )
        if not tool_calls:
            model_reasoning = "\n\n".join(part for part in thinking_parts if part)
            reasoning_text = (
                "\n".join([*trace, "Backend returned explicit reasoning/thinking content."]) + "\n\n--- model reasoning ---\n" + model_reasoning
                if model_reasoning
                else "\n".join([*trace, "Backend did not expose a reasoning/thinking field for this tool-agent request."])
            )
            return {
                "ok": True,
                "content": final_content,
                "reasoning": reasoning_text,
                "trace": "\n".join(trace),
                "tool_results": tool_results,
            }
        for call in tool_calls:
            name, args = _tool_call_name_and_args(call)
            if status is not None:
                workflow_status_write(status, user_facing_step_label(name))
            result = execute_ollama_tool_call(name, args)
            tool_results.append(result)
            trace.append(f"tool={name}, args={_safe_json_dumps(args, limit=1200)}, ok={result.get('ok')}")
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": _safe_json_dumps(compact_tool_result_for_llm(result)),
                }
            )

    return {
        "ok": False,
        "content": final_content or "분석을 마무리하지 못했습니다. 입력 데이터와 모델 설정을 확인한 뒤 다시 시도해주세요.",
        "reasoning": (
            "\n".join([*trace, "Backend returned explicit reasoning/thinking content."])
            + "\n\n--- model reasoning ---\n"
            + "\n\n".join(part for part in thinking_parts if part)
        )
        if thinking_parts
        else "\n".join([*trace, "Backend did not expose a reasoning/thinking field for this tool-agent request."]),
        "trace": "\n".join(trace),
        "tool_results": tool_results,
    }


def should_use_tool_agent(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    tool_tokens = [
        "데이터",
        "엑셀",
        "파일",
        "업로드",
        "요약",
        "컬럼",
        "column",
        "summary",
        "학습",
        "train",
        "모델",
        "model",
        "예측",
        "predict",
        "추론",
        "inference",
        "성능",
        "metric",
        "auc",
        "r2",
        "정확도",
        "스키마",
        "schema",
        "검증",
        "분류",
        "classification",
        "회귀",
        "regression",
        "결과",
        "result",
        "feature",
        "중요도",
        "shap",
    ]
    return any(token in normalized for token in tool_tokens)


def tab_llm(runtime: dict[str, Any]) -> None:
    st.header("5. 대화")
    st.caption("데이터 분석 요청과 일반 질문을 같은 대화창에서 처리합니다.")
    for msg in st.session_state.chat_messages:
        with chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))
            if msg.get("tool_result"):
                with st.expander("원본 결과", expanded=False):
                    st.json(msg["tool_result"])
            if msg.get("reasoning"):
                with st.expander("Reasoning", expanded=False):
                    st.code(msg["reasoning"])
    prompt = st.chat_input("예: 방금 올린 엑셀 요약해줘 / 분류 모델 학습해줘 / 저장된 분류 모델로 예측해줘")
    if not prompt:
        return
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with chat_message("user"):
        st.markdown(prompt)
    with chat_message("assistant"):
        tool_result = route_tool_request(
            prompt,
            ctx=tool_context(),
            selected_task=st.session_state.selected_task,
            model_id=st.session_state.selected_model_id,
            latest_result_id=st.session_state.latest_result_id,
        )
        if tool_result is not None:
            answer = tool_result_to_korean(tool_result)
            st.markdown(answer)
            with st.expander("원본 결과", expanded=False):
                st.json(tool_result)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer, "tool_result": tool_result})
            return
        try:
            with st.status("Calling local LLM...", expanded=False):
                messages = llm_messages_from_history()
                result = call_local_llm(runtime, messages, _chat_context_prompt() + "\n" + str(runtime.get("system_prompt") or ""))
            answer = result.get("content") or ""
            st.markdown(answer)
            if result.get("reasoning"):
                with st.expander("Reasoning", expanded=False):
                    st.code(result["reasoning"])
            st.session_state.chat_messages.append({"role": "assistant", "content": answer, "reasoning": result.get("reasoning")})
        except Exception as exc:
            answer = f"응답 생성 중 문제가 발생했습니다: {exc}"
            st.error(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


def tab_model_management() -> None:
    st.header("6. 모델 관리")
    for task in TASKS:
        st.subheader(task)
        rows = list_artifacts(task, MODEL_ROOT)
        if not rows:
            st.info(f"{task} 저장 모델이 없습니다.")
            continue
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        model_id = st.selectbox(f"{task} model detail", [row["model_id"] for row in rows], key=f"mgmt_{task}")
        try:
            artifact = load_artifact(task, model_id, MODEL_ROOT)
            c1, c2 = st.columns(2)
            with c1:
                st.write("Metrics")
                st.json(artifact["metrics"])
            with c2:
                st.write("Schema")
                st.json(artifact["schema"])
            with st.expander("Model Card", expanded=False):
                st.markdown(artifact["model_card"])
        except Exception as exc:
            st.warning(f"저장 모델 상세 로드 실패: {exc}")


def ensure_intro_message(runtime: dict[str, Any] | None = None) -> None:
    if st.session_state.chat_messages:
        return
    append_chat_message(
        "assistant",
        (
            "고압차단기 성능 예측을 시작하려면 학습용 Excel 데이터셋을 업로드해주세요.\n\n"
            "데이터가 등록되면 변수 구성과 목표값을 확인하고, 적절한 예측 모델을 학습한 뒤 검증 성능과 예측 결과를 정리합니다. "
            "이후에는 모델 결과, 입력 변수, 예측값에 대해 자연어로 질문할 수 있습니다."
        ),
        title="분석 안내",
        expanded=True,
    )


def append_chat_message(
    role: str,
    content: str,
    *,
    title: str | None = None,
    reasoning: str | None = None,
    trace: str | None = None,
    tool_result: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    result_id: str | None = None,
    show_validation: bool = False,
    pycaret_view: str | None = None,
    report_markdown: str | None = None,
    report_path: str | None = None,
    report_pdf_path: str | None = None,
    expanded: bool = True,
) -> None:
    st.session_state.chat_messages.append(
        {
            "role": role,
            "title": title or ("사용자" if role == "user" else "Assistant"),
            "content": content,
            "reasoning": reasoning or "",
            "trace": trace or "",
            "tool_result": tool_result,
            "tool_results": tool_results or [],
            "result_id": result_id,
            "show_validation": show_validation,
            "pycaret_view": pycaret_view or "",
            "report_markdown": report_markdown or "",
            "report_path": report_path or "",
            "report_pdf_path": report_pdf_path or "",
            "expanded": expanded,
        }
    )
    persist_ui_state()


def reasoning_details_markup(reasoning: str, *, live: bool = False, expanded: bool = False, box_id: str | None = None) -> str:
    open_attr = " open" if expanded else ""
    label = (
        '<span class="hd-live-reasoning"><span class="hd-reasoning-dot"></span>reasoning...</span>'
        if live
        else "reasoning"
    )
    safe_box_id = html.escape(box_id or f"hd-reasoning-{hashlib.sha1((reasoning or '').encode('utf-8')).hexdigest()[:12]}", quote=True)
    body = html.escape(reasoning or "모델의 reasoning 스트림을 기다리는 중입니다.")
    return (
        f'<details class="hd-reasoning"{open_attr}>'
        f"<summary>{label}</summary>"
        f'<div id="{safe_box_id}" class="hd-reasoning-box" tabindex="0" data-autofocus-reasoning="true">'
        f'<pre class="hd-reasoning-content">{body}</pre>'
        "</div>"
        "</details>"
    )


def scroll_reasoning_box(box_id: str) -> None:
    components.html(
        f"""
        <script>
        const boxId = {json.dumps(box_id)};
        const isVisible = (el) => {{
          const doc = window.parent.document;
          const rect = el.getBoundingClientRect();
          const height = window.parent.innerHeight || doc.documentElement.clientHeight || 0;
          return rect.top >= 0 && rect.bottom <= height;
        }};
        const scrollBox = () => {{
          const doc = window.parent.document;
          const el = doc.getElementById(boxId);
          if (!el) return;
          el.scrollTop = el.scrollHeight;
          try {{ el.focus({{preventScroll: true}}); }} catch (_) {{ try {{ el.focus(); }} catch (__) {{}} }}
          if (!isVisible(el)) {{
            try {{ el.scrollIntoView({{behavior: "smooth", block: "nearest", inline: "nearest"}}); }}
            catch (_) {{ el.scrollIntoView(false); }}
          }}
        }};
        setTimeout(scrollBox, 0);
        setTimeout(scrollBox, 80);
        setTimeout(scrollBox, 220);
        </script>
        """,
        height=0,
    )


def render_live_reasoning_indicator(reasoning: str | None = None, *, expanded: bool = True) -> None:
    box_id = f"hd-live-reasoning-{hashlib.sha1(str(time.time()).encode('utf-8')).hexdigest()[:12]}"
    st.markdown(
        reasoning_details_markup(
            reasoning or "모델 응답 스트림을 수신하는 중입니다. 백엔드가 reasoning/thinking 필드를 제공하면 여기에 실시간으로 표시됩니다.",
            live=True,
            expanded=expanded,
            box_id=box_id,
        ),
        unsafe_allow_html=True,
    )
    scroll_reasoning_box(box_id)


def render_reasoning_placeholder(target: Any, reasoning: str, *, live: bool, expanded: bool = True, box_id: str | None = None) -> None:
    resolved_box_id = box_id or f"hd-reasoning-{hashlib.sha1((reasoning or '').encode('utf-8')).hexdigest()[:12]}"
    target.empty()
    with target.container():
        st.markdown(reasoning_details_markup(reasoning, live=live, expanded=expanded, box_id=resolved_box_id), unsafe_allow_html=True)
        if live or expanded:
            scroll_reasoning_box(resolved_box_id)


def _compact_text_for_llm_context(text: Any, *, max_chars: int = 520) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _compact_chat_message_for_llm(msg: dict[str, Any], *, max_chars: int = 900) -> str:
    role = str(msg.get("role") or "")
    title = str(msg.get("title") or "").strip()
    label = f"[{title}] " if title else ""

    tool_result = msg.get("tool_result")
    if isinstance(tool_result, dict):
        tool = str(tool_result.get("tool") or "")
        task = str(tool_result.get("task") or "")
        model_id = str(tool_result.get("model_id") or "")
        result_id = str(tool_result.get("result_id") or "")
        if tool == "model.report_source":
            details = "성능 분석 자료가 생성됨"
        else:
            details = "내부 분석 결과가 생성됨"
        extras = ", ".join(part for part in (f"task={task}" if task else "", f"model={model_id}" if model_id else "", f"result={result_id}" if result_id else "") if part)
        return f"{label}{details}" + (f" ({extras})" if extras else "")

    if msg.get("tool_results"):
        return f"{label}복수 내부 분석 단계가 실행됨"

    content = _compact_text_for_llm_context(msg.get("content"), max_chars=max_chars)
    if not content:
        return ""
    return f"{label}{content}" if role != "user" else content


def _llm_state_summary_lines() -> list[str]:
    lines: list[str] = []
    record = active_dataset()
    if record:
        try:
            shape = list(record.dataframe.shape)
        except Exception:
            shape = []
        lines.append(
            "활성 학습 데이터셋: "
            f"{getattr(record, 'filename', 'unknown')} "
            f"(type={getattr(record, 'dataset_type', 'unknown')}, shape={shape})"
        )
    else:
        lines.append("활성 학습 데이터셋: 없음")
    lines.append(f"현재 선택 task/model: {st.session_state.selected_task} / {st.session_state.selected_model_id}")
    if st.session_state.get("latest_result_id"):
        lines.append(f"최근 예측 결과 ID: {st.session_state.latest_result_id}")
    return lines


def _prior_chat_summary_for_llm(prior_messages: list[dict[str, Any]], *, max_items: int = 8) -> str:
    lines = ["참고용 요약 컨텍스트입니다. 아래 내용은 이전 대화 전문이 아니라 상태와 핵심 이력만 압축한 것입니다."]
    lines.extend(f"- {line}" for line in _llm_state_summary_lines())

    compact_items: list[str] = []
    for msg in prior_messages[-max_items:]:
        if msg.get("role") not in {"user", "assistant"}:
            continue
        compact = _compact_chat_message_for_llm(msg, max_chars=360)
        if compact:
            compact_items.append(f"{msg.get('role')}: {compact}")
    if compact_items:
        lines.append("- 이전 대화 핵심 이력:")
        lines.extend(f"  - {item}" for item in compact_items)
    return "\n".join(lines)


def llm_messages_from_history(*, recent_turns: int = 4) -> list[dict[str, str]]:
    """Use compact context for normal chat only.

    Analysis, report, prediction, and training paths pass their own isolated
    payloads. Normal chat keeps the current request plus a few recent turns,
    with older content reduced to a short state summary.
    """
    chat_messages = [
        msg
        for msg in st.session_state.chat_messages
        if isinstance(msg, dict) and msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    recent_count = max(2, int(recent_turns) * 2)
    recent = chat_messages[-recent_count:]
    prior = chat_messages[: max(0, len(chat_messages) - len(recent))]

    messages: list[dict[str, str]] = [
        {"role": "user", "content": _prior_chat_summary_for_llm(prior)}
    ]
    for index, msg in enumerate(recent):
        role = str(msg.get("role") or "user")
        is_current_user_request = index == len(recent) - 1 and role == "user"
        content = _compact_chat_message_for_llm(msg, max_chars=2600 if is_current_user_request else 900)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def estimate_llm_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative mixed Korean/JSON estimate. It intentionally overestimates
    # a bit so small-context vLLM deployments do not reject the request.
    return max(1, int(len(str(text)) / 2) + 1)


def estimate_llm_messages_tokens(messages: list[dict[str, str]], system_prompt: str) -> int:
    total = estimate_llm_tokens(system_prompt)
    for message in messages:
        total += 8
        total += estimate_llm_tokens(str(message.get("role") or ""))
        total += estimate_llm_tokens(str(message.get("content") or ""))
    return total + 16


def llm_context_limit(runtime: dict[str, Any]) -> int:
    for key in ("context_window", "context_length", "max_context_tokens", "num_ctx"):
        value = runtime.get(key)
        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
    backend = str(runtime.get("backend") or "").lower()
    return 8192 if backend == "vllm" else 8192


def trim_llm_content_to_budget(content: str, token_budget: int) -> str:
    text = str(content or "")
    if estimate_llm_tokens(text) <= token_budget:
        return text
    char_budget = max(240, int(token_budget * 1.6))
    if len(text) <= char_budget:
        return text
    head = max(120, char_budget // 2)
    tail = max(120, char_budget - head)
    return text[:head].rstrip() + "\n\n...[context trimmed]...\n\n" + text[-tail:].lstrip()


def fit_llm_messages_to_context(
    messages: list[dict[str, str]],
    system_prompt: str,
    *,
    context_limit: int,
    reserved_output_tokens: int,
) -> tuple[list[dict[str, str]], list[str]]:
    notes: list[str] = []
    overhead = estimate_llm_tokens(system_prompt) + 96
    available = max(128, context_limit - overhead - reserved_output_tokens)
    fitted_reversed: list[dict[str, str]] = []
    used = 0

    for message in reversed(messages):
        content = str(message.get("content") or "")
        role = str(message.get("role") or "user")
        cost = estimate_llm_tokens(content) + estimate_llm_tokens(role) + 8
        if used + cost <= available:
            fitted_reversed.append({"role": role, "content": content})
            used += cost
            continue
        remaining = available - used - estimate_llm_tokens(role) - 8
        if not fitted_reversed and remaining > 96:
            fitted_reversed.append({"role": role, "content": trim_llm_content_to_budget(content, remaining)})
            used = available
            notes.append("current message was trimmed to fit the model context window.")
        else:
            notes.append("older chat history was dropped to fit the model context window.")
        break

    fitted = list(reversed(fitted_reversed)) or [{"role": "user", "content": "요청 내용을 처리해주세요."}]
    if len(fitted) != len(messages):
        notes.append(f"message_count adjusted: {len(messages)} -> {len(fitted)}")
    return fitted, notes


def apply_llm_context_budget(
    runtime: dict[str, Any],
    messages: list[dict[str, str]],
    system_prompt: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    scoped = dict(runtime)
    notes: list[str] = []
    context_limit = llm_context_limit(scoped)
    backend = str(scoped.get("backend") or "").lower()
    raw_max_tokens = scoped.get("max_tokens")
    explicit_max_tokens = raw_max_tokens not in (None, "", "auto", "default")
    if explicit_max_tokens:
        try:
            requested_max = int(raw_max_tokens)
        except (TypeError, ValueError):
            requested_max = 1024
    elif backend == "vllm":
        requested_max = max(512, min(4096, context_limit // 2))
        notes.append(f"vllm auto output budget from context window: {requested_max}")
    else:
        try:
            requested_max = int(st.session_state.config.get("max_tokens") or 2048)
        except (TypeError, ValueError):
            requested_max = 2048

    hard_cap = max(256, context_limit // 2 if backend == "vllm" and not explicit_max_tokens else context_limit // 3)
    planned_max = min(requested_max, hard_cap)
    if planned_max != requested_max:
        notes.append(f"max_tokens capped for context window: {requested_max} -> {planned_max}")

    fitted_messages, fit_notes = fit_llm_messages_to_context(
        messages,
        system_prompt,
        context_limit=context_limit,
        reserved_output_tokens=planned_max,
    )
    notes.extend(fit_notes)
    input_estimate = estimate_llm_messages_tokens(fitted_messages, system_prompt)
    allowed_output = max(64, context_limit - input_estimate - 96)
    final_max = max(64, min(planned_max, allowed_output))
    if final_max != planned_max:
        notes.append(f"max_tokens adjusted by estimated input size: {planned_max} -> {final_max}")

    scoped["max_tokens"] = int(final_max)
    notes.append(f"context_budget: limit={context_limit}, input_estimate={input_estimate}, max_tokens={final_max}")
    return scoped, fitted_messages, notes


def stream_llm_answer_to_chat(
    runtime: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    title: str = "응답",
    trace: str = "일반 대화 응답입니다.",
    tool_result: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    result_id: str | None = None,
    show_validation: bool = False,
    pycaret_view: str | None = None,
    fallback_content: str | None = None,
    save_report: bool = False,
    report_task: str | None = None,
    report_model_id: str | None = None,
    report_payload: dict[str, Any] | None = None,
    prediction_payload: dict[str, Any] | None = None,
) -> None:
    request_id = f"llm-{time.strftime('%Y%m%dT%H%M%S')}-{hashlib.sha1(str(time.time()).encode('utf-8')).hexdigest()[:8]}"
    system_prompt = _chat_context_prompt() + "\n" + str(runtime.get("system_prompt") or "")
    system_prompt += (
        f"\nrequest_id={request_id}. Treat this as a fresh LLM request. "
        "Do not reuse or refer to a previous response unless the user explicitly asks for prior context."
    )
    stream_runtime = dict(runtime)
    backend = str(stream_runtime.get("backend") or "").lower()
    if backend == "ollama":
        stream_runtime.setdefault("think", True)
    elif backend == "vllm":
        system_prompt += (
            "\nFor non-trivial user requests, use the model reasoning/thinking channel before the final answer "
            "when the backend supports it. Keep the final answer concise and user-facing."
        )
    stream_runtime, messages, budget_notes = apply_llm_context_budget(stream_runtime, list(messages), system_prompt)
    reasoning_slot = st.empty()
    answer_slot = st.empty()
    reasoning_box_id = f"hd-reasoning-live-{hashlib.sha1(str(time.time()).encode('utf-8')).hexdigest()[:12]}"
    render_reasoning_placeholder(
        reasoning_slot,
        "모델 응답 스트림을 수신하는 중입니다. 백엔드가 reasoning/thinking 필드를 제공하면 여기에 실시간으로 표시됩니다.",
        live=True,
        expanded=True,
        box_id=reasoning_box_id,
    )
    final_content = ""
    final_reasoning = ""
    llm_trace = [
        f"request_id: {request_id}",
        f"backend: {backend or 'unknown'}",
        f"model: {stream_runtime.get('model') or 'unknown'}",
        *budget_notes,
        "LLM stream request started.",
    ]
    used_fallback = False
    last_update = 0.0

    try:
        for chunk in stream_local_llm(stream_runtime, messages, system_prompt):
            final_content = str(chunk.get("content") or final_content)
            final_reasoning = str(chunk.get("reasoning") or final_reasoning)
            if chunk.get("done"):
                llm_trace.append(f"LLM stream finished in {chunk.get('elapsed_sec')} sec.")
            now = time.time()
            if final_reasoning and (now - last_update > 0.12 or chunk.get("done")):
                render_reasoning_placeholder(reasoning_slot, final_reasoning, live=not bool(chunk.get("done")), expanded=True, box_id=reasoning_box_id)
                last_update = now
            elif not final_reasoning and (now - last_update > 0.5 or chunk.get("done")):
                render_reasoning_placeholder(
                    reasoning_slot,
                    "모델 응답 스트림을 수신하는 중입니다.",
                    live=not bool(chunk.get("done")),
                    expanded=True,
                    box_id=reasoning_box_id,
                )
                last_update = now
            if final_content:
                answer_slot.markdown(final_content)
    except Exception as exc:
        llm_trace.append(f"LLM stream failed: {exc}")
        if fallback_content:
            used_fallback = True
            final_content = fallback_content
            answer_slot.markdown(final_content)
            st.warning(f"LLM 응답 생성에 실패해 기본 요약으로 대체했습니다: {exc}")
        else:
            append_chat_message("assistant", f"응답 생성 중 문제가 발생했습니다.\n\n- error: `{exc}`", title="응답 실패", expanded=True)
            st.error(f"응답 생성 중 문제가 발생했습니다: {exc}")
            return

    if not final_content:
        llm_trace.append("LLM returned an empty final answer.")
        if fallback_content:
            used_fallback = True
        final_content = fallback_content or "응답 본문이 비어 있습니다. 모델 연결 상태와 선택 모델을 확인해주세요."
        answer_slot.warning(final_content)
    if used_fallback:
        final_content = (
            "LLM 응답 생성이 완료되지 않아 계산된 산출물 기반 요약을 표시합니다.\n\n"
            f"{final_content}"
        )
        answer_slot.markdown(final_content)
    if final_reasoning:
        llm_trace.append("Backend returned explicit reasoning/thinking content.")
        final_reasoning = "\n".join(llm_trace) + "\n\n--- model reasoning ---\n" + final_reasoning
        render_reasoning_placeholder(reasoning_slot, final_reasoning, live=False, expanded=False, box_id=reasoning_box_id)
    else:
        llm_trace.append("Backend did not expose a reasoning/thinking field for this request.")
        if used_fallback:
            llm_trace.append("Deterministic fallback content was used after the LLM failure/empty response.")
        final_reasoning = "\n".join(llm_trace)
        render_reasoning_placeholder(reasoning_slot, final_reasoning, live=False, expanded=False, box_id=reasoning_box_id)
    if save_report and report_payload:
        final_content = ensure_report_pdf_order_content(final_content, report_payload, prediction_payload)
        answer_slot.markdown(final_content)
    report_markdown = ""
    report_path = ""
    report_pdf_path = ""
    if save_report:
        report_markdown = final_content
        try:
            saved = save_report_markdown(report_task or "model", report_model_id or "latest", final_content)
            report_path = str(saved)
        except Exception:
            report_path = ""
        if report_payload:
            try:
                saved_pdf = save_report_pdf(
                    report_task or str(report_payload.get("task") or "model"),
                    report_model_id or str(report_payload.get("model_id") or "latest"),
                    final_content,
                    report_payload,
                    prediction_payload,
                )
                report_pdf_path = str(saved_pdf) if saved_pdf else ""
            except Exception:
                report_pdf_path = ""
    append_chat_message(
        "assistant",
        final_content,
        title=title,
        reasoning=final_reasoning,
        trace=trace,
        tool_result=tool_result,
        tool_results=tool_results,
        result_id=result_id,
        show_validation=show_validation,
        pycaret_view=pycaret_view,
        report_markdown=report_markdown,
        report_path=report_path,
        report_pdf_path=report_pdf_path,
        expanded=True,
    )


def uploaded_fingerprint(uploaded: Any) -> tuple[str, bytes]:
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    return f"{uploaded.name}:{len(data)}:{digest}", data


def task_for_dataset_type(dataset_type: str) -> str | None:
    if dataset_type == "classification_training":
        return "classification"
    if dataset_type in {"regression_training", "motor_noise_regression_training"}:
        return "regression"
    return None


def format_holdout(metrics: dict[str, Any]) -> str:
    holdout = metrics.get("holdout", {})
    if not isinstance(holdout, dict) or not holdout:
        return "`{}`"
    return "`" + json.dumps(holdout, ensure_ascii=False) + "`"


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _metric_value(row: dict[str, Any], names: list[str]) -> float | None:
    lower_map = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name in row:
            value = _as_float(row.get(name))
            if value is not None:
                return value
        value = _as_float(lower_map.get(name.lower()))
        if value is not None:
            return value
    return None


def _artifact_metadata(task: str, model_id: str) -> dict[str, Any]:
    directory = (MODEL_ROOT / task / model_id).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Model artifact not found: {directory}")
    metrics_path = directory / "metrics.json"
    schema_path = directory / "schema.json"
    model_card_path = directory / "model_card.md"
    return {
        "task": task,
        "model_id": model_id,
        "directory": str(directory),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
        "schema": json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {},
        "model_card": model_card_path.read_text(encoding="utf-8") if model_card_path.exists() else "",
    }


def resolve_concrete_model_id(task: str, model_id: str) -> str:
    if model_id != "latest":
        return model_id
    try:
        directory = (MODEL_ROOT / task / model_id).resolve()
    except OSError:
        return model_id
    return directory.name if directory.name != "latest" else model_id


def design_model_names_from_metadata(schema: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    raw = schema.get("design_model_names") or metrics.get("design_model_names") or []
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if str(item).strip()]
    levels = schema.get("categorical_levels")
    if isinstance(levels, dict):
        names = levels.get("DESIGN_MODEL_NO__")
        if isinstance(names, list) and names:
            return [str(item) for item in names if str(item).strip()]
    return []


def _artifact_comparison_row(task: str, model_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
    holdout = metrics.get("holdout") if isinstance(metrics.get("holdout"), dict) else {}
    validation_metrics = metrics.get("validation_metrics") or metrics.get("predict_model_metrics") or []
    validation_row = validation_metrics[0] if validation_metrics and isinstance(validation_metrics[0], dict) else {}
    metric_row = {**holdout, **validation_row}
    split = metrics.get("validation_split") if isinstance(metrics.get("validation_split"), dict) else {}
    design_names = design_model_names_from_metadata(schema, metrics)
    design_text = ", ".join(design_names)
    common = {
        "model_id": model_id,
        "excel_design_model": design_text,
        "best_model": metrics.get("best_model") or metric_row.get("Model"),
        "features": len(schema.get("features") or []),
        "validation_rows": split.get("validation_rows"),
        "train_rows": metrics.get("n_train"),
        "test_rows": metrics.get("n_test"),
        "rows_after_dropna": metrics.get("n_rows_after_dropna"),
    }
    if task == "classification":
        common.update(
            {
                "Accuracy": _metric_value(metric_row, ["Accuracy", "accuracy"]),
                "AUC": _metric_value(metric_row, ["AUC", "roc_auc"]),
                "Recall": _metric_value(metric_row, ["Recall", "recall"]),
                "Precision": _metric_value(metric_row, ["Prec.", "Precision", "precision"]),
                "F1": _metric_value(metric_row, ["F1", "f1"]),
                "MCC": _metric_value(metric_row, ["MCC", "mcc"]),
            }
        )
    else:
        common.update(
            {
                "MAE": _metric_value(metric_row, ["MAE", "mae"]),
                "RMSE": _metric_value(metric_row, ["RMSE", "rmse"]),
                "R2": _metric_value(metric_row, ["R2", "r2"]),
                "MAPE": _metric_value(metric_row, ["MAPE", "mape"]),
                "RMSLE": _metric_value(metric_row, ["RMSLE", "rmsle"]),
            }
        )
    return {key: value for key, value in common.items() if value is not None}


def _sort_artifact_comparison(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if task == "classification":
        return sorted(rows, key=lambda row: (_as_float(row.get("F1")) or _as_float(row.get("AUC")) or _as_float(row.get("Accuracy")) or -1), reverse=True)
    return sorted(rows, key=lambda row: (_as_float(row.get("R2")) if _as_float(row.get("R2")) is not None else -999), reverse=True)


def _diagnostic_findings(task: str, payload: dict[str, Any]) -> list[str]:
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    candidates = payload.get("pycaret_candidate_comparison") if isinstance(payload.get("pycaret_candidate_comparison"), list) else []
    findings: list[str] = []
    validation_rows = _as_float(selected.get("validation_rows"))
    if validation_rows is not None and validation_rows < 30:
        findings.append("검증 세트 row 수가 30개 미만이어서 정량 지표의 변동성이 클 수 있습니다.")
    elif validation_rows is not None:
        findings.append(f"검증 세트는 {int(validation_rows)}개 row 기준으로 평가되었습니다.")

    if task == "classification":
        f1 = _as_float(selected.get("F1"))
        auc = _as_float(selected.get("AUC"))
        accuracy = _as_float(selected.get("Accuracy"))
        if f1 is not None:
            if f1 >= 0.85:
                findings.append("F1 기준 분류 성능은 높은 편입니다.")
            elif f1 >= 0.70:
                findings.append("F1 기준 분류 성능은 중간 수준이며, class별 오류 확인이 필요합니다.")
            else:
                findings.append("F1 기준 분류 성능이 낮아 feature/label 품질 또는 모델 후보 재검토가 필요합니다.")
        if accuracy is not None and f1 is not None and abs(accuracy - f1) >= 0.15:
            findings.append("Accuracy와 F1 차이가 커서 class imbalance 또는 특정 class 오분류 가능성을 확인해야 합니다.")
        if auc is not None and auc < 0.70:
            findings.append("AUC가 0.70 미만이면 decision threshold 조정이나 feature 개선이 필요할 수 있습니다.")
    else:
        r2 = _as_float(selected.get("R2"))
        rmse = _as_float(selected.get("RMSE"))
        mape = _as_float(selected.get("MAPE"))
        residual = payload.get("residual_summary") if isinstance(payload.get("residual_summary"), dict) else {}
        max_abs = _as_float(residual.get("max_absolute_error"))
        mae = _as_float(selected.get("MAE")) or _as_float(residual.get("mean_absolute_error_from_validation_rows"))
        if r2 is not None:
            if r2 >= 0.70:
                findings.append("R2 기준 회귀 설명력은 높은 편입니다.")
            elif r2 >= 0.40:
                findings.append("R2 기준 회귀 설명력은 중간 수준이며, 설계 변수 비선형성과 이상치를 추가 확인해야 합니다.")
            else:
                findings.append("R2 기준 회귀 설명력이 낮아 feature coverage, 데이터 분포, 모델 후보 재검토가 필요합니다.")
        if mape is not None and mape >= 0.20:
            findings.append("MAPE가 20% 이상이므로 절대 오차뿐 아니라 상대 오차 관점의 개선이 필요합니다.")
        if max_abs is not None and mae is not None and max_abs >= max(mae * 2.0, mae + 1e-9):
            findings.append("최대 절대오차가 평균 절대오차보다 크게 튀는 row가 있어 outlier 또는 특정 운전조건 오차를 점검해야 합니다.")
        if rmse is not None and mae is not None and rmse > mae * 1.35:
            findings.append("RMSE가 MAE보다 상대적으로 커서 큰 오차 샘플이 성능을 끌어내릴 가능성이 있습니다.")

    if len(candidates) >= 2:
        first = candidates[0]
        second = candidates[1]
        if task == "classification":
            first_score = _metric_value(first, ["F1", "AUC", "Accuracy"])
            second_score = _metric_value(second, ["F1", "AUC", "Accuracy"])
            if first_score is not None and second_score is not None and abs(first_score - second_score) <= 0.03:
                findings.append("상위 후보 모델 간 성능 차이가 작아 해석성/추론 속도/안정성 기준으로 최종 모델을 선택할 여지가 있습니다.")
        else:
            first_r2 = _metric_value(first, ["R2"])
            second_r2 = _metric_value(second, ["R2"])
            if first_r2 is not None and second_r2 is not None and abs(first_r2 - second_r2) <= 0.03:
                findings.append("상위 회귀 후보 간 R2 차이가 작아 RMSE/MAE와 해석성 기준을 함께 봐야 합니다.")
    return findings or ["현재 저장된 정량 지표 기준으로 진단 가능한 특이사항은 제한적입니다."]


def build_model_diagnostic_payload(task: str, model_id: str) -> dict[str, Any]:
    concrete_model_id = resolve_concrete_model_id(task, model_id)
    metadata = _artifact_metadata(task, concrete_model_id)
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
    selected_domain = str(schema.get("domain") or metrics.get("domain") or "circuit_breaker")
    design_names = design_model_names_from_metadata(schema, metrics)
    design_column = schema.get("design_model_column") or metrics.get("design_model_column") or ("DESIGN_MODEL_NO__" if design_names else None)
    selected = _artifact_comparison_row(task, concrete_model_id, metadata)
    artifact_rows: list[dict[str, Any]] = []
    for row in list_artifacts(task, MODEL_ROOT):
        artifact_id = str(row.get("model_id") or "")
        if not artifact_id or artifact_id == "latest" or not row.get("has_metrics"):
            continue
        try:
            row_metadata = _artifact_metadata(task, artifact_id)
            row_metrics = row_metadata.get("metrics") if isinstance(row_metadata.get("metrics"), dict) else {}
            row_schema = row_metadata.get("schema") if isinstance(row_metadata.get("schema"), dict) else {}
            row_domain = str(row_schema.get("domain") or row_metrics.get("domain") or "circuit_breaker")
            if row_domain != selected_domain:
                continue
            artifact_rows.append(_artifact_comparison_row(task, artifact_id, row_metadata))
        except Exception:
            continue
    artifact_rows = _sort_artifact_comparison(task, artifact_rows)
    compare_rows = metrics.get("compare_models") if isinstance(metrics.get("compare_models"), list) else []
    split = metrics.get("validation_split") if isinstance(metrics.get("validation_split"), dict) else {}
    payload = {
        "ok": True,
        "tool": "model.performance_diagnostics",
        "task": task,
        "domain": selected_domain,
        "display_name": schema.get("display_name") or metrics.get("display_name") or "형상/시험 변수 기반 고압차단기 성능 예측",
        "design_model_column": design_column,
        "design_model_names": design_names,
        "model_id": concrete_model_id,
        "requested_model_id": model_id,
        "artifact_dir": metadata.get("directory"),
        "selected_model": selected,
        "schema": {
            "target": schema.get("target"),
            "domain": selected_domain,
            "display_name": schema.get("display_name") or metrics.get("display_name") or "형상/시험 변수 기반 고압차단기 성능 예측",
            "design_model_column": design_column,
            "design_model_names": design_names,
            "feature_count": len(schema.get("features") or []),
            "ignored_columns": schema.get("ignored_columns") or [],
            "engine": schema.get("engine") or metrics.get("engine"),
        },
        "validation_split": split,
        "validation_metrics": metrics.get("validation_metrics") or metrics.get("predict_model_metrics") or [],
        "confusion_matrix": metrics.get("validation_confusion_matrix"),
        "residual_summary": metrics.get("validation_residual_summary"),
        "artifact_comparison": artifact_rows[:10],
        "pycaret_candidate_comparison": compare_rows[:10],
        "figures": [
            {
                "title": clean_figure_title(figure.get("title") or figure.get("plot")),
                "path": figure.get("path"),
            }
            for figure in metrics.get("pycaret_figures", [])
            if isinstance(figure, dict) and figure.get("ok") and figure.get("path")
        ],
    }
    payload["findings"] = _diagnostic_findings(task, payload)
    return payload


def compact_model_payload_for_llm(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": payload.get("task"),
        "domain": payload.get("domain"),
        "display_name": payload.get("display_name"),
        "design_model_column": payload.get("design_model_column"),
        "design_model_names": payload.get("design_model_names"),
        "model_id": payload.get("model_id"),
        "selected_model": payload.get("selected_model"),
        "schema": payload.get("schema"),
        "validation_split": payload.get("validation_split"),
        "validation_metrics": payload.get("validation_metrics"),
        "confusion_matrix": payload.get("confusion_matrix"),
        "residual_summary": payload.get("residual_summary"),
        "artifact_comparison": (payload.get("artifact_comparison") or [])[:6],
        "pycaret_candidate_comparison": (payload.get("pycaret_candidate_comparison") or [])[:8],
        "generated_figures": [
            {"title": clean_figure_title(figure.get("title")), "path": figure.get("path")}
            for figure in (payload.get("figures") or [])[:8]
            if isinstance(figure, dict)
        ],
        "findings": payload.get("findings"),
    }


def deterministic_diagnostic_summary(payload: dict[str, Any]) -> str:
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    lines = [
        "선택 모델의 저장된 검증 지표를 기준으로 성능 진단/비교를 정리했습니다.",
        f"- task: `{payload.get('task')}`",
        f"- model_id: `{payload.get('model_id')}`",
        f"- best_model: `{selected.get('best_model')}`",
    ]
    metric_keys = ["Accuracy", "AUC", "F1", "MAE", "RMSE", "R2", "MAPE"]
    metric_text = ", ".join(f"{key}={selected[key]}" for key in metric_keys if selected.get(key) is not None)
    if metric_text:
        lines.append(f"- 주요 지표: `{metric_text}`")
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    if findings:
        lines.append("\n진단 포인트:")
        lines.extend(f"- {item}" for item in findings)
    lines.append("\n아래 상세 영역에서 저장 모델 비교표와 후보 모델 비교표를 확인할 수 있습니다.")
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int = 8) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows[:max_rows]:
        body.append("| " + " | ".join(str(_compact_table_value(row.get(col, ""), max_chars=28)).replace("|", "/") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _compact_table_value(value: Any, *, max_chars: int = 36) -> Any:
    parsed = _as_float(value)
    if parsed is not None:
        if abs(parsed) >= 1000:
            return f"{parsed:.1f}"
        return f"{parsed:.4g}"
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def clean_figure_title(value: Any) -> str:
    title = str(value or "검증 시각화").strip()
    title = re.sub(r"\bPyCaret\b\s*", "", title, flags=re.IGNORECASE).strip()
    title = title.replace("Plot", "그래프").replace("plot", "그래프")
    title = re.sub(r"\s+", " ", title).strip()
    return title or "검증 시각화"


def _pdf_cell_value(value: Any, *, max_chars: int = 24) -> str:
    text = str(_compact_table_value(value, max_chars=max_chars)).replace("_", " ")
    if len(text) <= 12:
        return text
    wrapped = textwrap.wrap(text, width=12, break_long_words=True)
    return "\n".join(wrapped[:2])


def regression_prediction_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if str(col).startswith("predicted_") and not str(col).startswith("predicted_probability")
    ]


def actual_column_for_prediction(df: pd.DataFrame, pred_col: str) -> str | None:
    if pred_col == "predicted_TRVmax[kV]" and "TRVmax[kV]" in df.columns:
        return "TRVmax[kV]"
    candidate = pred_col.removeprefix("predicted_")
    return candidate if candidate in df.columns else None


def prediction_with_regression_errors(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    pred_cols = regression_prediction_columns(frame)
    pred_col = pred_cols[0] if pred_cols else ""
    actual_col = actual_column_for_prediction(frame, pred_col) if pred_col else None
    if not pred_col or not actual_col:
        return frame
    actual = pd.to_numeric(frame[actual_col], errors="coerce")
    predicted = pd.to_numeric(frame[pred_col], errors="coerce")
    frame["residual"] = predicted - actual
    frame["absolute_error"] = frame["residual"].abs()
    return frame


def prediction_ranked_by_error_if_available(df: pd.DataFrame) -> pd.DataFrame:
    frame = prediction_with_regression_errors(df)
    if "absolute_error" not in frame.columns:
        return frame
    return frame.sort_values("absolute_error", ascending=True, na_position="last", kind="mergesort")


def prediction_report_preview_frame(df: pd.DataFrame, *, max_rows: int = 10) -> pd.DataFrame:
    ranked = prediction_ranked_by_error_if_available(df)
    pred_cols = prediction_result_columns(ranked)
    regression_cols = regression_prediction_columns(df)
    regression_actuals = [actual_column_for_prediction(df, col) for col in regression_cols]
    priority = [col for col in ("Result", "TRVmax[kV]", "sound_mean") if col in df.columns]
    for actual_col in regression_actuals:
        if actual_col and actual_col not in priority:
            priority.append(actual_col)
    for pred_col in regression_cols:
        if pred_col not in priority:
            priority.append(pred_col)
    for error_col in ("absolute_error", "residual"):
        if error_col in ranked.columns and error_col not in priority:
            priority.append(error_col)
    if "prediction_label" in df.columns and "prediction_label" not in priority:
        priority.append("prediction_label")
    for col in pred_cols:
        if col not in priority and len(priority) < 8:
            priority.append(col)
    if not priority:
        priority = list(ranked.columns[: min(8, len(ranked.columns))])
    preview = ranked.loc[:, priority].head(max_rows).copy()
    source_rows = []
    for index in preview.index:
        source_rows.append(index + 1 if isinstance(index, int) else str(index))
    preview.insert(0, "row", source_rows)
    return preview


def prediction_source_row_label(index: Any) -> int | str:
    return index + 1 if isinstance(index, int) else str(index)


def prediction_report_kind(df: pd.DataFrame) -> str:
    if regression_prediction_columns(df):
        return "regression"
    if "prediction_label" in df.columns or "prediction" in df.columns:
        return "classification"
    return "unknown"


def prediction_report_domain(df: pd.DataFrame) -> str:
    columns = {str(col) for col in df.columns}
    regression_cols = set(regression_prediction_columns(df))
    if "predicted_sound_mean" in regression_cols or "sound_mean" in columns:
        return "motor_noise_regression"
    if "predicted_TRVmax[kV]" in regression_cols or "TRVmax[kV]" in columns:
        return "trvmax_regression"
    if prediction_report_kind(df) == "classification":
        return "classification"
    if regression_cols:
        return "generic_regression"
    return "unknown"


def prediction_report_type_label(value: str | None) -> str:
    labels = {
        "trvmax_regression": "TRVmax 회귀 예측",
        "motor_noise_regression": "회전기 소음 회귀 예측",
        "generic_regression": "회귀 예측",
        "classification": "분류 예측",
        "unknown": "미확인 예측",
    }
    return labels.get(str(value or "unknown"), "미확인 예측")


def prediction_numeric_statistics(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        min_index = values.idxmin()
        max_index = values.idxmax()
        rows.append(
            {
                "column": column,
                "count": int(values.count()),
                "min": float(values.min()),
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q3": float(values.quantile(0.75)),
                "max": float(values.max()),
                "std": float(values.std()) if len(values) > 1 else 0.0,
                "min_row": prediction_source_row_label(min_index),
                "max_row": prediction_source_row_label(max_index),
            }
        )
    return rows


def prediction_classification_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    label_column = "prediction_label" if "prediction_label" in df.columns else "prediction" if "prediction" in df.columns else ""
    if not label_column:
        return []
    counts = df[label_column].value_counts(dropna=False).to_dict()
    total = max(len(df), 1)
    return [
        {"label": str(label), "count": int(count), "ratio_pct": round(float(count) / total * 100.0, 3)}
        for label, count in counts.items()
    ]


def prediction_report_analysis_lines(prediction_payload: dict[str, Any] | None) -> list[str]:
    if not prediction_payload:
        return []
    rows = prediction_payload.get("rows")
    result_id = prediction_payload.get("result_id")
    kind = str(prediction_payload.get("prediction_kind") or "unknown")
    type_label = str(prediction_payload.get("prediction_type_label") or prediction_report_type_label(prediction_payload.get("prediction_domain")))
    lines = [f"Result ID `{result_id}`의 예측용 파일 {rows}개 row를 `{type_label}`로 분석했습니다."]
    numeric_stats = prediction_payload.get("numeric_statistics") if isinstance(prediction_payload.get("numeric_statistics"), list) else []
    class_summary = prediction_payload.get("classification_summary") if isinstance(prediction_payload.get("classification_summary"), list) else []
    probability_stats = prediction_payload.get("probability_statistics") if isinstance(prediction_payload.get("probability_statistics"), list) else []
    if kind == "regression" and numeric_stats:
        main = numeric_stats[0]
        target_name = str(main.get("column") or "회귀 예측값").replace("predicted_", "")
        lines.append(
            f"{target_name} 예측 범위는 "
            f"{_compact_table_value(main.get('min'), max_chars=12)} ~ {_compact_table_value(main.get('max'), max_chars=12)}이며, "
            f"중앙값은 {_compact_table_value(main.get('median'), max_chars=12)}, 평균은 {_compact_table_value(main.get('mean'), max_chars=12)}입니다."
        )
        lines.append(
            f"최솟값은 원본 row {main.get('min_row')}, 최댓값은 원본 row {main.get('max_row')}에서 발생했습니다."
        )
    elif kind == "classification" and class_summary:
        top = class_summary[0]
        lines.append(f"가장 많이 예측된 class는 `{top.get('label')}`이며 {top.get('count')}개 row, {top.get('ratio_pct')}%입니다.")
        if probability_stats:
            prob = probability_stats[0]
            lines.append(
                f"{prob.get('column')} 확률 범위는 {_compact_table_value(prob.get('min'), max_chars=12)} ~ "
                f"{_compact_table_value(prob.get('max'), max_chars=12)}, 평균은 {_compact_table_value(prob.get('mean'), max_chars=12)}입니다."
            )
    else:
        prediction_cols = ", ".join(map(str, prediction_payload.get("prediction_columns") or [])) or "N/A"
        lines.append(f"예측 결과 컬럼은 {prediction_cols}입니다.")
    lines.append("아래 preview는 원본 데이터 row 기준으로 예측 결과를 확인하기 위한 표입니다.")
    return lines


def prediction_report_opinion_lines(prediction_payload: dict[str, Any] | None) -> list[str]:
    if not prediction_payload:
        return []
    numeric_stats = prediction_payload.get("numeric_statistics") if isinstance(prediction_payload.get("numeric_statistics"), list) else []
    class_summary = prediction_payload.get("classification_summary") if isinstance(prediction_payload.get("classification_summary"), list) else []
    probability_stats = prediction_payload.get("probability_statistics") if isinstance(prediction_payload.get("probability_statistics"), list) else []
    if numeric_stats:
        main = numeric_stats[0]
        mean = _as_float(main.get("mean")) or 0.0
        std = _as_float(main.get("std")) or 0.0
        min_value = _as_float(main.get("min"))
        max_value = _as_float(main.get("max"))
        variation = abs(std / mean) if mean else 0.0
        spread_text = "예측값 분산이 큰 편이므로 상/하위 row의 입력 변수 조건을 추가 확인하는 것이 좋습니다." if variation >= 0.20 else "예측값 분산은 상대적으로 제한적이며, 대표값 중심의 경향 해석이 가능합니다."
        return [
            f"예측 분포는 평균 {_compact_table_value(mean, max_chars=12)}와 표준편차 {_compact_table_value(std, max_chars=12)} 기준으로 검토했습니다.",
            f"최솟값 {_compact_table_value(min_value, max_chars=12)}와 최댓값 {_compact_table_value(max_value, max_chars=12)} row는 설계 변수 차이를 비교할 우선 후보입니다.",
            spread_text,
        ]
    if class_summary:
        top = class_summary[0]
        ratio = _as_float(top.get("ratio_pct")) or 0.0
        balance_text = "특정 class 편중이 강하므로 실제 적용 전 class imbalance와 threshold 민감도를 확인해야 합니다." if ratio >= 80.0 else "예측 class 분포가 한쪽으로만 치우치지는 않아 class별 오류 가능성을 함께 검토할 수 있습니다."
        lines = [
            f"최빈 예측 class는 {top.get('label')}이며 전체의 {_compact_table_value(ratio, max_chars=12)}%입니다.",
            balance_text,
        ]
        if probability_stats:
            prob = probability_stats[0]
            lines.append(
                f"{prob.get('column')} 평균 확률은 {_compact_table_value(prob.get('mean'), max_chars=12)}로, 낮은 확신도 row는 별도 검토 대상입니다."
            )
        return lines
    return ["예측 결과 컬럼은 확인되었지만 정량 해석을 위한 수치/분포 정보가 제한적입니다."]


def prediction_report_section_markdown(prediction_payload: dict[str, Any] | None) -> str:
    if not prediction_payload:
        return ""
    preview = prediction_payload.get("preview") if isinstance(prediction_payload.get("preview"), list) else []
    preview_columns = list(preview[0].keys()) if preview else []
    numeric_stats = prediction_payload.get("numeric_statistics") if isinstance(prediction_payload.get("numeric_statistics"), list) else []
    class_summary = prediction_payload.get("classification_summary") if isinstance(prediction_payload.get("classification_summary"), list) else []
    probability_stats = prediction_payload.get("probability_statistics") if isinstance(prediction_payload.get("probability_statistics"), list) else []
    numeric_columns = ["column", "count", "min", "q1", "median", "mean", "q3", "max", "std", "min_row", "max_row"]
    probability_columns = ["column", "count", "min", "median", "mean", "max", "std"]
    lines = "\n".join(f"- {line}" for line in prediction_report_analysis_lines(prediction_payload))
    opinions = "\n".join(f"- {line}" for line in prediction_report_opinion_lines(prediction_payload))
    stats_block = ""
    if numeric_stats:
        stats_block += "\n### 6.2 회귀 예측 통계\n" + (_markdown_table(numeric_stats, numeric_columns, max_rows=4) or "- 회귀 통계가 없습니다.") + "\n"
    if class_summary:
        stats_block += "\n### 6.2 분류 결과 분포\n" + (_markdown_table(class_summary, ["label", "count", "ratio_pct"], max_rows=8) or "- 분류 분포가 없습니다.") + "\n"
    if probability_stats:
        stats_block += "\n### 6.3 확률 컬럼 범위\n" + (_markdown_table(probability_stats, probability_columns, max_rows=6) or "- 확률 통계가 없습니다.") + "\n"
    preview_title = "6.4 예측 결과 미리보기" if probability_stats or class_summary or numeric_stats else "6.2 예측 결과 미리보기"
    return f"""### 6.1 분석 개요
- Result ID: `{prediction_payload.get('result_id')}`
- 예측 row 수: `{prediction_payload.get('rows')}`
- 예측 유형: `{prediction_payload.get('prediction_kind')}`
- 예측 구분: `{prediction_payload.get('prediction_type_label') or prediction_report_type_label(prediction_payload.get('prediction_domain'))}`
- 예측 결과 컬럼: `{prediction_payload.get('prediction_columns')}`

### 핵심 해석
{lines or "- 해석 가능한 예측 요약이 제한적입니다."}

### 해석 의견
{opinions or "- 예측 결과 해석 의견이 제한적입니다."}
{stats_block}
### {preview_title}
{_markdown_table(preview, preview_columns, max_rows=10) if preview else "- 예측 preview가 없습니다."}
"""


def build_prediction_report_payload(result_id: str | None = None) -> dict[str, Any] | None:
    target_result_id = result_id or st.session_state.get("latest_result_id")
    if not target_result_id:
        return None
    df = st.session_state.get("prediction_results", {}).get(target_result_id)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    preview_df = prediction_report_preview_frame(df, max_rows=10)
    prediction_columns = prediction_result_columns(df)
    probability_columns = [col for col in prediction_columns if str(col).startswith("probability")]
    regression_columns = regression_prediction_columns(df)
    payload: dict[str, Any] = {
        "result_id": target_result_id,
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "prediction_kind": prediction_report_kind(df),
        "prediction_domain": prediction_report_domain(df),
        "prediction_columns": prediction_columns,
        "preview_sort": "absolute_error_ascending" if "absolute_error" in prediction_ranked_by_error_if_available(df).columns else "source_row_order",
        "summary_lines": prediction_result_summary_lines(df),
        "preview": preview_df.where(pd.notna(preview_df), None).to_dict(orient="records"),
        "numeric_statistics": prediction_numeric_statistics(df, regression_columns),
        "probability_statistics": prediction_numeric_statistics(df, probability_columns),
        "classification_summary": prediction_classification_summary(df),
    }
    payload["prediction_type_label"] = prediction_report_type_label(payload.get("prediction_domain"))
    minmax = prediction_minmax_frame(df)
    if not minmax.empty:
        payload["minmax"] = minmax.where(pd.notna(minmax), None).to_dict(orient="records")
    if "prediction_label" in df.columns:
        payload["classification_counts"] = {str(k): int(v) for k, v in df["prediction_label"].value_counts(dropna=False).to_dict().items()}
    elif "prediction" in df.columns:
        payload["classification_counts"] = {str(k): int(v) for k, v in df["prediction"].value_counts(dropna=False).to_dict().items()}
    payload["analysis_lines"] = prediction_report_analysis_lines(payload)
    return payload


def report_conclusion_text(payload: dict[str, Any], prediction_payload: dict[str, Any] | None = None) -> str:
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    task = str(payload.get("task") or "")
    best_model = selected.get("best_model") or "선택 모델"
    if task == "classification":
        primary_metric = _metric_value(selected, ["F1", "AUC", "Accuracy"])
        metric_name = "F1/AUC/Accuracy"
        metric_text = f"{metric_name} 기준 주요 성능은 {_compact_table_value(primary_metric, max_chars=12)}입니다." if primary_metric is not None else "분류 주요 성능 지표는 저장 결과 기준으로 확인했습니다."
    else:
        r2 = _metric_value(selected, ["R2"])
        rmse = _metric_value(selected, ["RMSE"])
        metric_text = f"R2는 {_compact_table_value(r2, max_chars=12)}, RMSE는 {_compact_table_value(rmse, max_chars=12)}입니다." if r2 is not None or rmse is not None else "회귀 주요 성능 지표는 저장 결과 기준으로 확인했습니다."
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    risk_text = findings[0] if findings else "운영 적용 전 별도 데이터 기반 재검증이 필요합니다."
    candidate_opinions = candidate_comparison_opinion_lines(task, payload) if payload else []
    prediction_analysis = prediction_report_analysis_lines(prediction_payload) if prediction_payload else []
    prediction_opinions = prediction_report_opinion_lines(prediction_payload) if prediction_payload else []
    prediction_text = (
        f"최신 예측용 파일 `{prediction_payload.get('result_id')}`의 {prediction_payload.get('rows')}개 row 결과를 함께 반영했습니다."
        if prediction_payload
        else "현재 세션에는 예측용 파일 결과가 없어 저장 모델의 검증 성능과 후보 모델 비교 결과를 중심으로 결론을 작성했습니다."
    )
    lines = [
        "### 종합 판단",
        f"- `{best_model}`은 현재 저장된 검증 데이터 기준으로 선택된 모델이며, {metric_text}",
        f"- {prediction_text}",
        "",
        "### 성능 및 모델 비교",
        f"- 후보 모델 비교 관점에서는 {candidate_opinions[0] if candidate_opinions else '비교 가능한 후보 모델 지표가 제한적입니다.'}",
    ]
    if len(candidate_opinions) > 1:
        lines.append(f"- {candidate_opinions[1]}")
    if findings:
        lines.append(f"- 주요 진단 포인트는 {risk_text}")

    lines.extend(["", "### 예측용 파일 반영"])
    if prediction_payload:
        if prediction_analysis:
            lines.append(f"- {prediction_analysis[0]}")
        if len(prediction_analysis) > 1:
            lines.append(f"- {prediction_analysis[1]}")
        if prediction_opinions:
            lines.append(f"- {prediction_opinions[0]}")
    else:
        lines.append("- 별도 예측용 파일 결과가 없으므로, 실제 배치 예측 분포나 row별 오차 평가는 포함하지 않았습니다.")
        lines.append("- 다음 분석 자료에서는 예측용 Excel 결과를 함께 넣으면 모델 검증 결과와 실제 적용 데이터의 분포 차이를 같이 판단할 수 있습니다.")

    lines.extend(
        [
            "",
            "### 운영 적용 시 유의사항",
            f"- 결론적으로 본 모델은 현재 데이터 범위 내 예측 지원에 사용할 수 있으나, {risk_text}",
            "- 신규 설계 조건이나 기존 데이터 범위를 벗어난 입력에 대해서는 예측 신뢰도가 낮아질 수 있으므로 별도 검증이 필요합니다.",
            "",
            "### 후속 작업",
            "- 신규 시험 데이터 또는 별도 holdout set으로 재검증하고, 오차가 큰 row의 입력 변수 조건을 우선 점검해야 합니다.",
            "- 후보 모델 간 성능 차이가 작다면 성능 지표뿐 아니라 해석성, 추론 속도, 재현성을 함께 기준으로 최종 운용 모델을 선택해야 합니다.",
        ]
    )
    return "\n".join(lines)


def ensure_report_conclusion(content: str, payload: dict[str, Any] | None, prediction_payload: dict[str, Any] | None) -> str:
    if not payload:
        return content
    conclusion_number = "7" if prediction_payload else "6"
    structured = f"## {conclusion_number}. 결론\n{report_conclusion_text(payload, prediction_payload)}"
    pattern = re.compile(r"(?ims)^##\s*(?:\d+\.\s*)?결론\s*$.*?(?=^##\s+|\Z)")
    match = pattern.search(content or "")
    if not match:
        return (content or "").rstrip() + "\n\n" + structured
    existing = match.group(0)
    if "### 종합 판단" in existing and "### 후속 작업" in existing:
        return content
    return (content[: match.start()].rstrip() + "\n\n" + structured + "\n\n" + content[match.end() :].lstrip()).rstrip()


def deterministic_report_markdown(payload: dict[str, Any], prediction_payload: dict[str, Any] | None = None) -> str:
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    split = payload.get("validation_split") if isinstance(payload.get("validation_split"), dict) else {}
    task = str(payload.get("task") or "")
    family_title = model_family_title(payload)
    family_purpose = model_family_purpose(payload)
    design_names = payload.get("design_model_names") if isinstance(payload.get("design_model_names"), list) else []
    design_text = ", ".join(map(str, design_names)) if design_names else "N/A"
    metric_columns = (
        ["excel_design_model", "model_id", "best_model", "Accuracy", "AUC", "F1"]
        if task == "classification"
        else ["excel_design_model", "model_id", "best_model", "MAE", "RMSE", "R2", "MAPE"]
    )
    candidate_columns = ["Model", "Accuracy", "AUC", "F1"] if task == "classification" else ["Model", "MAE", "RMSE", "R2", "MAPE"]
    artifact_table = _markdown_table(payload.get("artifact_comparison") or [], metric_columns, max_rows=6)
    candidate_table = _markdown_table(payload.get("pycaret_candidate_comparison") or [], candidate_columns, max_rows=8)
    candidate_opinions = "\n".join(f"- {line}" for line in candidate_comparison_opinion_lines(task, payload))
    findings = "\n".join(f"- {item}" for item in payload.get("findings", []))
    prediction_section = ""
    if prediction_payload:
        prediction_section = prediction_report_section_markdown(prediction_payload)
    conclusion = report_conclusion_text(payload, prediction_payload)
    prediction_block = f"""
## 6. 예측용 파일 분석 ({prediction_payload.get('prediction_type_label') or prediction_report_type_label(prediction_payload.get('prediction_domain'))})
{prediction_section}
""" if prediction_payload else ""
    conclusion_number = "7" if prediction_payload else "6"
    return f"""# {family_title} 분석

## 1. 요약
- 분석 목적: {family_purpose}의 검증 결과 정리
- Excel 설계모델명: `{design_text}`
- 선택 모델: `{selected.get('best_model')}`
- 모델 ID: `{payload.get('model_id')}`
- 분석 유형: `{task}`

## 2. 데이터 및 검증 조건
- Excel 설계모델명: `{design_text}`
- Target: `{schema.get('target')}`
- 입력 변수 수: `{schema.get('feature_count')}`
- 검증 방식: 학습 데이터 90%, 검증 데이터 10%
- 검증 row 수: `{split.get('validation_rows')}`
- 제외 컬럼: `{schema.get('ignored_columns')}`

## 3. 모델 성능 평가
{_markdown_table([selected], metric_columns, max_rows=1) or "- 저장된 성능 지표가 부족합니다."}

### 진단 포인트
{findings or "- 자동 진단 포인트가 없습니다."}

## 4. 후보 모델 비교
### 저장 모델 비교
{artifact_table or "- 비교 가능한 저장 모델이 부족합니다."}

### 자동 후보 모델 비교
{candidate_table or "- 후보 모델 비교표가 없습니다."}

### 해석 의견
{candidate_opinions or "- 후보 모델 비교 해석 의견이 제한적입니다."}

## 5. 검증 시각화 해석
- PDF와 채팅 미리보기의 검증 시각화 항목에서 그림별 캡션과 해석을 제공합니다.

{prediction_block}

## {conclusion_number}. 결론
{conclusion}
"""


def _top_report_section_number(line: str) -> int | None:
    match = re.match(r"^\s*##\s+(\d+)(?:\.\d+)?[.)]?\s+", str(line or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _split_numbered_report_sections(content: str) -> tuple[list[str], dict[int, list[str]], list[int]]:
    preamble: list[str] = []
    sections: dict[int, list[str]] = {}
    order: list[int] = []
    current_no: int | None = None
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_no, current_lines
        if current_no is not None and current_lines:
            sections[current_no] = current_lines[:]
            if current_no not in order:
                order.append(current_no)
        current_no = None
        current_lines = []

    for line in str(content or "").splitlines():
        section_no = _top_report_section_number(line)
        if section_no is not None:
            flush_current()
            current_no = section_no
            current_lines = [line]
            continue
        if current_no is None:
            preamble.append(line)
        else:
            current_lines.append(line)
    flush_current()
    return preamble, sections, order


def report_order_numbers(prediction_payload: dict[str, Any] | None = None) -> list[int]:
    return [1, 2, 3, 4, 5, 6, 7] if prediction_payload else [1, 2, 3, 4, 5, 6]


def ensure_report_pdf_order_content(
    content: str,
    payload: dict[str, Any] | None,
    prediction_payload: dict[str, Any] | None,
) -> str:
    if not payload:
        return content
    prediction_payload = prediction_payload if isinstance(prediction_payload, dict) else None
    base = ensure_report_conclusion(content or "", payload, prediction_payload)
    fallback = deterministic_report_markdown(payload, prediction_payload)
    base_preamble, base_sections, base_order = _split_numbered_report_sections(base)
    fallback_preamble, fallback_sections, _fallback_order = _split_numbered_report_sections(fallback)
    required_order = report_order_numbers(prediction_payload)

    preamble = "\n".join(line for line in base_preamble if str(line).strip()).strip()
    if not preamble:
        preamble = "\n".join(line for line in fallback_preamble if str(line).strip()).strip()

    blocks: list[str] = []
    if preamble:
        blocks.append(preamble)
    for section_no in required_order:
        block_lines = base_sections.get(section_no)
        if not block_lines or len("\n".join(block_lines).strip()) < 24:
            block_lines = fallback_sections.get(section_no)
        if block_lines:
            blocks.append("\n".join(block_lines).strip())

    extra_sections = [section_no for section_no in base_order if section_no not in required_order]
    for section_no in extra_sections:
        block_lines = base_sections.get(section_no)
        if block_lines:
            blocks.append("\n".join(block_lines).strip())

    return "\n\n".join(block for block in blocks if block).strip()


def save_report_markdown(task: str, model_id: str, content: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_state_filename(model_id)
    path = REPORTS_DIR / f"{timestamp}_{task}_{safe_model}_report.md"
    path.write_text(content, encoding="utf-8")
    return path


def _pdf_font(path_hint: str = "regular", size: float = 10) -> Any:
    if FontProperties is None:
        return None
    candidates = (
        ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]
        if path_hint == "bold"
        else ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return FontProperties(fname=str(path), size=size)
    if findfont is not None:
        try:
            return FontProperties(fname=findfont("DejaVu Sans"), size=size)
        except Exception:
            return FontProperties(size=size)
    return FontProperties(size=size)


def _pdf_text(fig: Any, x: float, y: float, text: str, *, size: float = 10, bold: bool = False, color: str = "#111827", ha: str = "left") -> None:
    font = _pdf_font("bold" if bold else "regular", size)
    fig.text(x, y, text, ha=ha, va="top", color=color, fontproperties=font)


def _pdf_center_lines(fig: Any, lines: list[str], *, x: float = 0.5, y: float = 0.15, line_height: float = 0.022, size: float = 8.5, color: str = "#111827") -> None:
    cursor = y
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=82, break_long_words=False) or [""]:
            _pdf_text(fig, x, cursor, wrapped, size=size, color=color, ha="center")
            cursor -= line_height


def _pdf_bullet_lines(fig: Any, lines: list[str], *, x: float = 0.12, y: float = 0.15, width: int = 108, line_height: float = 0.020, size: float = 8.3, color: str = "#374151") -> float:
    cursor = y
    for line in lines:
        wrapped_lines = textwrap.wrap(str(line), width=width, break_long_words=False) or [""]
        for index, wrapped in enumerate(wrapped_lines):
            prefix = "- " if index == 0 else "  "
            _pdf_text(fig, x, cursor, prefix + wrapped, size=size, color=color, ha="left")
            cursor -= line_height
    return cursor


def _pdf_footer(fig: Any, page_no: int, total_pages: int) -> None:
    _pdf_text(fig, 0.50, 0.035, f"{page_no} / {total_pages}", size=8.5, color="#6b7280", ha="center")


def _pdf_image(fig: Any, path: Path | None, rect: list[float]) -> None:
    if Image is None or path is None or not path.exists():
        return
    try:
        ax = fig.add_axes(rect)
        ax.imshow(Image.open(path))
        ax.axis("off")
    except Exception:
        return


def report_section_titles(prediction_payload: dict[str, Any] | None = None) -> list[str]:
    sections = [
        "1. 요약",
        "2. 데이터 및 검증 조건",
        "3. 모델 성능 평가",
        "4. 후보 모델 비교",
        "5. 검증 시각화 해석",
    ]
    if prediction_payload:
        type_label = prediction_payload.get("prediction_type_label") or prediction_report_type_label(prediction_payload.get("prediction_domain"))
        sections.append(f"6. 예측용 파일 분석 ({type_label})")
        sections.append("7. 결론")
    else:
        sections.append("6. 결론")
    return sections


def _pdf_add_cover(pdf: Any, *, task: str, model_id: str, payload: dict[str, Any], page_no: int, total_pages: int) -> None:
    if plt is None:
        return
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor("white")
    panel = fig.add_axes([0.10, 0.52, 0.80, 0.33])
    panel.axis("off")
    panel.add_patch(plt.Rectangle((0, 0), 1, 1, transform=panel.transAxes, facecolor="#0f2a44", edgecolor="none"))
    _pdf_image(fig, hd_electric_logo_path(), [0.18, 0.76, 0.28, 0.055])
    _pdf_text(fig, 0.50, 0.705, model_family_title(payload), size=17, bold=True, color="#ffffff", ha="center")
    _pdf_text(fig, 0.50, 0.655, "Report", size=25, bold=True, color="#ffffff", ha="center")
    _pdf_text(fig, 0.50, 0.603, f"모델 번호: {model_id}", size=9, color="#dbeafe", ha="center")
    _pdf_text(fig, 0.50, 0.575, f"분석 유형: {task} / 선택 모델: {selected.get('best_model') or 'N/A'}", size=8.5, color="#dbeafe", ha="center")
    target = schema.get("target")
    if target:
        _pdf_text(fig, 0.50, 0.550, f"Target: {target}", size=8.5, color="#dbeafe", ha="center")
    _pdf_text(fig, 0.45, 0.457, "by", size=9, color="#4b5563", ha="right")
    _pdf_image(fig, AIM4LAB_LOGO_PATH if AIM4LAB_LOGO_PATH.exists() else None, [0.465, 0.432, 0.080, 0.045])
    _pdf_text(fig, 0.555, 0.458, "Lab", size=11, bold=True, color="#111827", ha="left")
    _pdf_text(fig, 0.50, 0.205, time.strftime("%Y-%m-%d"), size=13, color="#374151", ha="center")
    _pdf_footer(fig, page_no, total_pages)
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_add_toc(pdf: Any, sections: list[str], *, page_no: int, total_pages: int) -> None:
    if plt is None:
        return
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    _pdf_text(fig, 0.08, 0.92, "목차", size=20, bold=True, color="#0f2a44")
    cursor = 0.84
    for section in sections:
        _pdf_text(fig, 0.12, cursor, section, size=12, bold=True, color="#111827")
        cursor -= 0.055
    _pdf_footer(fig, page_no, total_pages)
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_lines(text: str, *, width: int = 88) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        wrapped = textwrap.wrap(stripped, width=width, break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def _pdf_write_lines(
    fig: Any,
    lines: list[str],
    *,
    x: float = 0.08,
    y: float = 0.90,
    line_height: float = 0.026,
    size: float = 10,
    width: int = 92,
    bottom: float = 0.08,
) -> float:
    cursor = y
    for line in lines:
        if cursor < bottom:
            break
        stripped = line.strip()
        if stripped.startswith("# "):
            for wrapped in textwrap.wrap(stripped[2:], width=72, break_long_words=False) or [""]:
                if cursor < bottom:
                    break
                _pdf_text(fig, x, cursor, wrapped, size=18, bold=True, color="#0f2a44")
                cursor -= line_height * 1.55
        elif stripped.startswith("## "):
            for wrapped in textwrap.wrap(stripped[3:], width=78, break_long_words=False) or [""]:
                if cursor < bottom:
                    break
                _pdf_text(fig, x, cursor, wrapped, size=13, bold=True, color="#0f2a44")
                cursor -= line_height * 1.25
        elif stripped.startswith("### "):
            for wrapped in textwrap.wrap(stripped[4:], width=86, break_long_words=False) or [""]:
                if cursor < bottom:
                    break
                _pdf_text(fig, x, cursor, wrapped, size=11, bold=True, color="#1f3b57")
                cursor -= line_height * 1.1
        elif stripped:
            continuation_prefix = "  " if stripped.startswith("- ") else ""
            wrapped_lines = textwrap.wrap(stripped, width=width, break_long_words=False, replace_whitespace=False) or [stripped]
            for index, wrapped in enumerate(wrapped_lines):
                if cursor < bottom:
                    break
                _pdf_text(fig, x, cursor, (continuation_prefix + wrapped) if index else wrapped, size=size)
                cursor -= line_height
        else:
            cursor -= line_height * 0.65
    return cursor


def _pdf_table(ax: Any, rows: list[dict[str, Any]], columns: list[str], *, max_rows: int = 8, title: str = "") -> None:
    ax.axis("off")
    if title:
        ax.set_title(title, fontproperties=_pdf_font("bold", 11), loc="left", pad=8, color="#0f2a44")
    if not rows:
        ax.text(0.02, 0.82, "표시할 데이터가 없습니다.", fontproperties=_pdf_font("regular", 10), transform=ax.transAxes)
        return
    table_rows = [[_pdf_cell_value(row.get(col)) for col in columns] for row in rows[:max_rows]]
    col_widths = [0.95 / max(len(columns), 1)] * len(columns)
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1.0, 1.6)
    for (row, _col), cell in table.get_celld().items():
        cell.get_text().set_fontproperties(_pdf_font("bold" if row == 0 else "regular", 6.8))
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_facecolor("#e8f0f8")
            cell.set_edgecolor("#aab7c4")
        else:
            cell.set_edgecolor("#d6dde5")


def _candidate_score(task: str, row: dict[str, Any]) -> tuple[str, float | None, bool]:
    if task == "classification":
        for metric in ("F1", "AUC", "Accuracy"):
            value = _metric_value(row, [metric])
            if value is not None:
                return metric, value, True
        return "score", None, True
    for metric in ("R2", "RMSE", "MAE", "MAPE"):
        value = _metric_value(row, [metric])
        if value is not None:
            return metric, value, metric == "R2"
    return "score", None, True


def _candidate_score_rows(task: str, rows: list[dict[str, Any]], *, limit: int = 10) -> tuple[str, bool, list[dict[str, Any]]]:
    scored = []
    metric_name = ""
    higher_is_better = True
    for row in rows[:limit]:
        metric_name, value, higher_is_better = _candidate_score(task, row)
        if value is None:
            continue
        scored.append({"model": str(row.get("Model") or row.get("best_model") or "model"), "score": value})
    scored = sorted(scored, key=lambda item: item["score"], reverse=higher_is_better)
    return metric_name, higher_is_better, scored


def candidate_comparison_opinion_lines(task: str, payload: dict[str, Any]) -> list[str]:
    candidates = payload.get("pycaret_candidate_comparison") if isinstance(payload.get("pycaret_candidate_comparison"), list) else []
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    metric_name, higher_is_better, scored_rows = _candidate_score_rows(task, candidates, limit=10)
    if not scored_rows:
        return ["후보 모델 비교 지표가 없어 저장 모델의 단일 검증 성능 중심으로 해석해야 합니다."]
    top = scored_rows[0]
    selected_name = str(selected.get("best_model") or "선택 모델")
    direction = "높을수록 유리한" if higher_is_better else "낮을수록 유리한"
    lines = [
        f"{metric_name}는 {direction} 지표이며, 후보 중 `{top['model']}`의 값이 {_compact_table_value(top['score'], max_chars=12)}로 가장 우수합니다.",
        f"현재 선택 모델은 `{selected_name}`이며, 후보 비교 결과와 저장 모델 검증 지표를 함께 기준으로 판단해야 합니다.",
    ]
    if len(scored_rows) >= 2:
        second = scored_rows[1]
        gap = abs(float(top["score"]) - float(second["score"]))
        if gap <= 0.03:
            lines.append("상위 후보 간 지표 차이가 작아 성능뿐 아니라 해석성, 추론 안정성, 재현성을 함께 검토하는 것이 적절합니다.")
        else:
            lines.append(f"1위와 2위 후보의 {metric_name} 차이는 {_compact_table_value(gap, max_chars=12)}로, 후보 우선순위가 비교적 분명합니다.")
    return lines


def _chart_label(value: str, *, width: int = 15, max_lines: int = 2) -> str:
    text = str(value).replace("_", " ").strip()
    wrapped = textwrap.wrap(text, width=width, break_long_words=False)
    if not wrapped:
        return text[:width]
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][: max(1, width - 1)] + "..."
    return "\n".join(wrapped)


def _pdf_candidate_histogram(ax: Any, task: str, rows: list[dict[str, Any]]) -> None:
    metric_name, _higher_is_better, scored_rows = _candidate_score_rows(task, rows, limit=8)
    scored = [(row["model"], row["score"]) for row in scored_rows]
    if not scored:
        ax.text(0.05, 0.55, "후보 모델 비교 지표가 없습니다.", fontproperties=_pdf_font("regular", 9), transform=ax.transAxes)
        ax.axis("off")
        return
    names = [_chart_label(item[0], width=14, max_lines=2) for item in scored]
    values = [item[1] for item in scored]
    y_pos = list(range(len(names)))
    bars = ax.barh(y_pos, values, color="#2f6f9f")
    ax.set_yticks(y_pos, labels=names, fontproperties=_pdf_font("regular", 6.0))
    ax.invert_yaxis()
    ax.set_xlabel(metric_name, fontproperties=_pdf_font("regular", 7.5), labelpad=5)
    ax.grid(axis="x", color="#d6dde5", linestyle="--", linewidth=0.7)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.tick_params(axis="y", labelsize=6.0, pad=2)
    if values:
        max_value = max(values)
        min_value = min(values)
        span = max(abs(max_value), abs(min_value), 1e-6)
        ax.set_xlim(left=min(0, min_value) - span * 0.08, right=max(0, max_value) + span * 0.25)
    for bar, value in zip(bars, values):
        left, right = ax.get_xlim()
        offset = (right - left) * 0.012
        label_x = value + offset if value >= 0 else value - offset
        label_x = min(max(label_x, left + offset), right - offset)
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:.4g}", va="center", ha="left" if value >= 0 else "right", fontproperties=_pdf_font("regular", 6.3), clip_on=True)


def figure_interpretation_lines(task: str, figure: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    title = clean_figure_title(figure.get("title") or figure.get("plot") or Path(str(figure.get("path") or "figure")).stem)
    key = (title + " " + str(figure.get("path") or "")).lower()
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    residual = payload.get("residual_summary") if isinstance(payload.get("residual_summary"), dict) else {}
    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    split = payload.get("validation_split") if isinstance(payload.get("validation_split"), dict) else {}
    validation_rows = split.get("validation_rows") or selected.get("validation_rows")
    feature_count = schema.get("feature_count")
    target = schema.get("target") or "target"
    best_model = selected.get("best_model") or "선택 모델"
    regression_metrics = ", ".join(
        f"{metric}={_compact_table_value(selected.get(metric), max_chars=12)}"
        for metric in ("R2", "RMSE", "MAE", "MAPE")
        if selected.get(metric) is not None
    )
    classification_metrics = ", ".join(
        f"{metric}={_compact_table_value(selected.get(metric), max_chars=12)}"
        for metric in ("F1", "AUC", "Accuracy")
        if selected.get(metric) is not None
    )
    if "feature" in key:
        return [
            f"{best_model}이 {target} 예측에 사용한 입력 변수 {feature_count or 'N/A'}개 중 상대 영향도가 큰 항목을 확인합니다.",
            f"검증 row {validation_rows or 'N/A'}개 기준의 성능 지표와 함께 보면 feature 우선순위를 더 안정적으로 해석할 수 있습니다.",
            "중요도는 인과관계를 의미하지 않으므로 물리적 해석과 추가 검증을 함께 적용해야 합니다.",
        ]
    if "residual" in key:
        return [
            f"잔차 평균은 {_compact_table_value(residual.get('mean_residual', 'N/A'), max_chars=12)}, 중앙값은 {_compact_table_value(residual.get('median_residual', 'N/A'), max_chars=12)}입니다.",
            f"최대 절대오차는 {_compact_table_value(residual.get('max_absolute_error', 'N/A'), max_chars=12)}이며, 검증 row {validation_rows or 'N/A'}개에서 큰 오차 샘플 존재 여부를 봅니다.",
            "잔차가 0 주변에 모일수록 체계적 편향이 작고, 긴 꼬리나 한쪽 치우침은 특정 조건의 outlier 점검 신호입니다.",
        ]
    if "error" in key or "prediction" in key:
        return [
            f"검증 세트 예측 결과를 실제값과 비교합니다. 저장 지표는 {regression_metrics or classification_metrics or '제한적입니다'}.",
            f"{target} 기준으로 대각선 또는 낮은 오차 영역에 가까울수록 실제값과 예측값의 일치도가 높습니다.",
            "크게 벗어난 샘플은 데이터 품질, feature coverage, 비선형 조건 누락 여부를 확인하는 후보입니다.",
        ]
    if "auc" in key or "pr" in key or "confusion" in key or "class" in key:
        return [
            f"분류 검증 지표는 {classification_metrics or '저장 결과가 제한적'}이며, threshold 기반 class 판별 성능을 확인합니다.",
            "정확도만으로 보이지 않는 false positive/false negative 균형을 confusion matrix 또는 PR/AUC 흐름으로 함께 평가합니다.",
            "운영 기준에서는 비용이 큰 오류 유형을 기준으로 threshold 또는 모델 선택을 조정할 수 있습니다.",
        ]
    return [
        f"{title} 그림은 {best_model}의 검증 동작을 해석하기 위한 자료이며, 검증 row는 {validation_rows or 'N/A'}개입니다.",
        f"정량 지표({regression_metrics or classification_metrics or '제한적'})와 함께 보면 모델의 강점과 취약 조건을 판단할 수 있습니다.",
        "해석에는 저장된 검증 데이터 결과와 자동 생성된 검증 그림만 사용했습니다.",
    ]


def figure_caption(index: int, figure: dict[str, Any]) -> str:
    title = clean_figure_title(figure.get("title") or figure.get("plot") or Path(str(figure.get("path") or "figure")).stem)
    return f"<fig.{index} {title}>"


def _pdf_add_pycaret_figures(
    pdf: Any,
    figures: list[dict[str, Any]],
    *,
    task: str,
    payload: dict[str, Any],
    section_title: str = "5. 검증 시각화 해석",
    start_page_no: int = 1,
    total_pages: int = 1,
) -> int:
    if plt is None or Image is None:
        return 0
    valid = [figure for figure in figures if isinstance(figure, dict) and figure.get("path") and Path(str(figure.get("path"))).exists()]
    for offset in range(0, len(valid), 2):
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        _pdf_text(fig, 0.08, 0.95, section_title, size=16, bold=True, color="#0f2a44")
        row = valid[offset : offset + 2]
        positions = [(0.08, 0.64, 0.84, 0.24), (0.08, 0.27, 0.84, 0.24)]
        caption_y = [0.59, 0.22]
        for local_index, (item, pos) in enumerate(zip(row, positions), start=1):
            fig_index = offset + local_index
            ax = fig.add_axes(pos)
            try:
                image = Image.open(str(item.get("path")))
                ax.imshow(image)
                ax.axis("off")
                _pdf_text(fig, 0.50, caption_y[local_index - 1], figure_caption(fig_index, item), size=10, bold=True, color="#1f3b57", ha="center")
                interpretation = figure_interpretation_lines(task, item, payload)
                _pdf_bullet_lines(fig, interpretation[:3], x=0.12, y=caption_y[local_index - 1] - 0.030, width=108, line_height=0.020, size=8.3, color="#374151")
            except Exception:
                ax.text(0.05, 0.5, f"Figure 로드 실패: {item.get('path')}", fontproperties=_pdf_font("regular", 9), transform=ax.transAxes)
                ax.axis("off")
        _pdf_footer(fig, start_page_no + offset // 2, total_pages)
        pdf.savefig(fig)
        plt.close(fig)
    return (len(valid) + 1) // 2


def _prediction_columns_text(prediction_payload: dict[str, Any]) -> str:
    columns = prediction_payload.get("prediction_columns")
    if isinstance(columns, list):
        visible = [str(item) for item in columns[:8]]
        suffix = f" 외 {len(columns) - len(visible)}개" if len(columns) > len(visible) else ""
        return ", ".join(visible) + suffix
    return _compact_table_value(columns, max_chars=96)


def save_report_pdf(
    task: str,
    model_id: str,
    markdown_content: str,
    payload: dict[str, Any],
    prediction_payload: dict[str, Any] | None = None,
) -> Path | None:
    if plt is None or PdfPages is None:
        return None
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORTS_DIR / f"{timestamp}_{task}_{_safe_state_filename(model_id)}_report.pdf"
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
    split = payload.get("validation_split") if isinstance(payload.get("validation_split"), dict) else {}
    sections = report_section_titles(prediction_payload)
    conclusion_number = "7" if prediction_payload else "6"
    figure_items = payload.get("figures") if isinstance(payload.get("figures"), list) else []
    valid_figure_count = sum(1 for figure in figure_items if isinstance(figure, dict) and figure.get("path") and Path(str(figure.get("path"))).exists())
    figure_pages = (valid_figure_count + 1) // 2
    prediction_preview = prediction_payload.get("preview") if isinstance(prediction_payload, dict) and isinstance(prediction_payload.get("preview"), list) else []
    prediction_pages = (2 if prediction_preview else 1) if prediction_payload else 0
    total_pages = 2 + 1 + 1 + figure_pages + prediction_pages + 1
    page_no = 1
    family_purpose = model_family_purpose(payload)
    design_names = payload.get("design_model_names") if isinstance(payload.get("design_model_names"), list) else []
    design_text = ", ".join(map(str, design_names)) if design_names else "N/A"
    with PdfPages(pdf_path) as pdf:
        _pdf_add_cover(pdf, task=task, model_id=model_id, payload=payload, page_no=page_no, total_pages=total_pages)
        page_no += 1
        _pdf_add_toc(pdf, sections, page_no=page_no, total_pages=total_pages)
        page_no += 1

        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        summary_lines = [
            "## 1. 요약",
            f"- 분석 목적: {family_purpose}의 검증 결과 정리",
            f"- Excel 설계모델명: {design_text}",
            f"- Task: {task}",
            f"- Model ID: {model_id}",
            f"- 선택 모델: {selected.get('best_model')}",
            "",
            "## 2. 데이터 및 검증 조건",
            f"- Excel 설계모델명: {design_text}",
            f"- Target: {schema.get('target')}",
            f"- 입력 변수 수: {schema.get('feature_count')}",
            f"- 검증 방식: 학습 데이터 90%, 검증 데이터 10%",
            f"- 검증 row 수: {split.get('validation_rows')}",
            f"- 제외 컬럼: {schema.get('ignored_columns')}",
            "",
            "## 3. 모델 성능 평가",
            "- " + ", ".join(f"{key}: {selected[key]}" for key in ("Accuracy", "AUC", "F1", "MAE", "RMSE", "R2", "MAPE") if selected.get(key) is not None),
            "",
            "### 진단 포인트",
            *[f"- {item}" for item in payload.get("findings", [])],
        ]
        _pdf_write_lines(fig, summary_lines, y=0.94, size=10)
        _pdf_footer(fig, page_no, total_pages)
        pdf.savefig(fig)
        plt.close(fig)
        page_no += 1

        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        _pdf_text(fig, 0.06, 0.95, "4. 후보 모델 비교", size=16, bold=True, color="#0f2a44")
        artifact_cols = ["model_id", "best_model", "Accuracy", "AUC", "F1"] if task == "classification" else ["model_id", "best_model", "MAE", "RMSE", "R2", "MAPE"]
        candidate_cols = ["Model", "Accuracy", "AUC", "F1"] if task == "classification" else ["Model", "MAE", "RMSE", "R2", "MAPE"]
        ax_table = fig.add_axes([0.06, 0.62, 0.88, 0.28])
        _pdf_table(ax_table, payload.get("artifact_comparison") or [], artifact_cols, max_rows=6, title="4.1 저장 모델 성능 비교")
        ax_candidate_table = fig.add_axes([0.06, 0.43, 0.88, 0.16])
        _pdf_table(ax_candidate_table, payload.get("pycaret_candidate_comparison") or [], candidate_cols, max_rows=5, title="4.2 자동 후보 모델 비교표")
        _pdf_text(fig, 0.06, 0.345, "4.3 후보 모델 비교 히스토그램", size=11, bold=True, color="#0f2a44")
        ax_hist = fig.add_axes([0.29, 0.145, 0.58, 0.175])
        _pdf_candidate_histogram(ax_hist, task, payload.get("pycaret_candidate_comparison") or [])
        _pdf_text(fig, 0.06, 0.115, "4.4 해석 의견", size=10.5, bold=True, color="#0f2a44")
        _pdf_bullet_lines(fig, candidate_comparison_opinion_lines(task, payload), x=0.08, y=0.092, width=112, line_height=0.017, size=7.8, color="#374151")
        _pdf_footer(fig, page_no, total_pages)
        pdf.savefig(fig)
        plt.close(fig)
        page_no += 1

        page_no += _pdf_add_pycaret_figures(
            pdf,
            payload.get("figures") or [],
            task=task,
            payload=payload,
            section_title="5. 검증 시각화 해석",
            start_page_no=page_no,
            total_pages=total_pages,
        )

        if prediction_payload:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
            type_label = prediction_payload.get("prediction_type_label") or prediction_report_type_label(prediction_payload.get("prediction_domain"))
            _pdf_text(fig, 0.08, 0.95, f"6. 예측용 파일 분석 ({type_label})", size=16, bold=True, color="#0f2a44")
            prediction_lines = [
                f"- Result ID: {prediction_payload.get('result_id')}",
                f"- 예측 row 수: {prediction_payload.get('rows')}",
                f"- 예측 유형: {prediction_payload.get('prediction_kind')}",
                f"- 예측 구분: {prediction_payload.get('prediction_type_label') or prediction_report_type_label(prediction_payload.get('prediction_domain'))}",
                f"- 예측 결과 컬럼: {_prediction_columns_text(prediction_payload)}",
                "",
                "### 핵심 해석",
                *[f"- {line}" for line in prediction_report_analysis_lines(prediction_payload)],
                "",
                "### 해석 의견",
                *[f"- {line}" for line in prediction_report_opinion_lines(prediction_payload)],
            ]
            _pdf_write_lines(fig, prediction_lines, y=0.89, size=8.2, line_height=0.018, width=104, bottom=0.47)
            numeric_stats = prediction_payload.get("numeric_statistics") if isinstance(prediction_payload.get("numeric_statistics"), list) else []
            class_summary = prediction_payload.get("classification_summary") if isinstance(prediction_payload.get("classification_summary"), list) else []
            probability_stats = prediction_payload.get("probability_statistics") if isinstance(prediction_payload.get("probability_statistics"), list) else []
            if numeric_stats:
                ax_stats = fig.add_axes([0.04, 0.28, 0.92, 0.15])
                _pdf_table(ax_stats, numeric_stats, ["column", "count", "min", "median", "mean", "max", "std", "min_row", "max_row"], max_rows=3, title="예측 통계")
            elif class_summary:
                ax_stats = fig.add_axes([0.08, 0.31, 0.84, 0.12])
                _pdf_table(ax_stats, class_summary, ["label", "count", "ratio_pct"], max_rows=6, title="분류 결과 분포")
            if probability_stats:
                ax_prob = fig.add_axes([0.04, 0.13, 0.92, 0.12])
                _pdf_table(ax_prob, probability_stats, ["column", "min", "median", "mean", "max", "std"], max_rows=4, title="확률 컬럼 범위")
            _pdf_footer(fig, page_no, total_pages)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1
            preview = prediction_preview
            if preview:
                fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
                _pdf_text(fig, 0.08, 0.95, "6.1 예측 결과 미리보기", size=16, bold=True, color="#0f2a44")
                _pdf_write_lines(
                    fig,
                    [
                        "- 화면에는 대표 row만 표시합니다. 전체 예측 결과는 함께 저장된 다운로드 파일에서 확인합니다.",
                        "- 회귀 결과는 실제값 대비 오차가 작은 순서, 분류 결과는 원본 row 기준 미리보기로 정리합니다.",
                    ],
                    y=0.89,
                    size=8.4,
                    line_height=0.020,
                    width=102,
                    bottom=0.78,
                )
                columns = list(preview[0].keys()) if preview else []
                ax_pred = fig.add_axes([0.04, 0.16, 0.92, 0.58])
                _pdf_table(ax_pred, preview, columns, max_rows=10, title="예측 결과 미리보기")
                _pdf_footer(fig, page_no, total_pages)
                pdf.savefig(fig)
                plt.close(fig)
                page_no += 1

        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        _pdf_text(fig, 0.08, 0.95, f"{conclusion_number}. 결론", size=16, bold=True, color="#0f2a44")
        conclusion_lines = report_conclusion_text(payload, prediction_payload).splitlines()
        _pdf_write_lines(fig, conclusion_lines, y=0.88, size=8.8, line_height=0.020, width=102, bottom=0.10)
        _pdf_footer(fig, page_no, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


def _action_runtime(runtime: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
    scoped = dict(runtime)
    try:
        current_max = int(scoped.get("max_tokens") or max_tokens)
    except (TypeError, ValueError):
        current_max = max_tokens
    scoped["max_tokens"] = max(256, min(current_max, max_tokens))
    return scoped


def model_family_title(payload: dict[str, Any] | None = None) -> str:
    if isinstance(payload, dict):
        design_names = payload.get("design_model_names")
        if not isinstance(design_names, list):
            schema_for_names = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
            design_names = schema_for_names.get("design_model_names") if isinstance(schema_for_names.get("design_model_names"), list) else []
        design_suffix = f" ({', '.join(map(str, design_names))})" if design_names else ""
        display_name = str(payload.get("display_name") or "")
        if display_name:
            return display_name + design_suffix
        schema = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
        display_name = str(schema.get("display_name") or "")
        if display_name:
            return display_name + design_suffix
        if payload.get("domain") == "motor_noise" or schema.get("domain") == "motor_noise":
            return "회전기 소음 예측 모델" + design_suffix
    return "형상/시험 변수 기반 고압차단기 성능 예측"


def model_family_purpose(payload: dict[str, Any] | None = None) -> str:
    if isinstance(payload, dict) and payload.get("domain") == "motor_noise":
        names = payload.get("design_model_names")
        name_text = f" ({', '.join(map(str, names))})" if isinstance(names, list) and names else ""
        return f"회전기 설계모델{name_text}의 설계 변수 기반 소음 평균값(sound_mean) 예측 모델"
    return "형상/시험 변수 기반 고압차단기 성능 예측 모델"


def run_model_diagnostics_to_chat(runtime: dict[str, Any], task: str, model_id: str) -> None:
    try:
        payload = build_model_diagnostic_payload(task, model_id)
    except Exception as exc:
        append_chat_message(
            "assistant",
            f"모델 성능 비교에 실패했습니다.\n\n- error: `{exc}`",
            title="모델 성능 비교 실패",
            expanded=True,
        )
        st.error(f"모델 성능 비교 실패: {exc}")
        return

    compact_payload = compact_model_payload_for_llm(payload)
    family_title = model_family_title(payload)
    family_purpose = model_family_purpose(payload)
    prompt = (
        f"다음 JSON은 `{family_title}` 저장 산출물에서 계산한 성능 비교 자료입니다.\n"
        f"분석 대상은 `{family_purpose}`입니다.\n"
        "JSON의 design_model_names는 Excel 원본의 설계모델명이며, 회전기 소음 모델 보고/비교에서는 반드시 명시하세요.\n"
        "이 값만 근거로 한국어로 사용자 친화적인 모델 성능 비교를 작성하세요.\n"
        "필수 구성: 1) 결론, 2) 핵심 성능, 3) 저장 모델 비교, 4) 후보 모델 비교, 5) 리스크/주의사항, 6) 다음 액션.\n"
        "각 항목은 최대 2개 bullet, bullet 하나는 한 문장으로 끝내세요.\n"
        "문장을 중간에 끊지 말고 마지막 문장까지 완결하세요. 수치를 새로 만들지 말고 JSON에 없는 값은 없다고 말하세요.\n\n"
        f"```json\n{_safe_json_dumps(compact_payload, limit=4200)}\n```"
    )
    diagnostic_runtime = _action_runtime(runtime, max_tokens=1300)
    try:
        diagnostic_runtime["max_tokens"] = max(int(diagnostic_runtime.get("max_tokens") or 0), 900)
    except (TypeError, ValueError):
        diagnostic_runtime["max_tokens"] = 900
    stream_llm_answer_to_chat(
        diagnostic_runtime,
        [{"role": "user", "content": prompt}],
        title=f"{family_title} 성능 비교",
        trace="stored model comparison payload -> local LLM summary.",
        tool_result=payload,
        fallback_content=deterministic_diagnostic_summary(payload),
    )


def run_model_report_to_chat(runtime: dict[str, Any], task: str, model_id: str) -> None:
    try:
        payload = build_model_diagnostic_payload(task, model_id)
    except Exception as exc:
        append_chat_message(
            "assistant",
            f"성능 분석 자료 생성에 실패했습니다.\n\n- error: `{exc}`",
            title="성능 분석 자료 생성 실패",
            expanded=True,
        )
        st.error(f"성능 분석 자료 생성 실패: {exc}")
        return

    prediction_payload = build_prediction_report_payload()
    compact_payload = compact_model_payload_for_llm(payload)
    family_title = model_family_title(payload)
    family_purpose = model_family_purpose(payload)
    if prediction_payload:
        compact_payload["latest_prediction_result"] = {
            "result_id": prediction_payload.get("result_id"),
            "rows": prediction_payload.get("rows"),
            "prediction_kind": prediction_payload.get("prediction_kind"),
            "prediction_domain": prediction_payload.get("prediction_domain"),
            "prediction_type_label": prediction_payload.get("prediction_type_label"),
            "summary_lines": prediction_payload.get("summary_lines"),
            "analysis_lines": prediction_payload.get("analysis_lines"),
            "opinion_lines": prediction_report_opinion_lines(prediction_payload),
            "numeric_statistics": prediction_payload.get("numeric_statistics"),
            "classification_summary": prediction_payload.get("classification_summary"),
            "probability_statistics": prediction_payload.get("probability_statistics"),
            "preview": (prediction_payload.get("preview") or [])[:5],
        }
    compact_payload["candidate_opinion_lines"] = candidate_comparison_opinion_lines(task, payload)
    fallback = deterministic_report_markdown(payload, prediction_payload)
    prompt = (
        f"다음 JSON은 `{family_title}`의 저장 결과 요약입니다.\n"
        f"분석 대상은 `{family_purpose}`입니다.\n"
        "JSON의 design_model_names는 Excel 원본의 설계모델명이며, 회전기 소음 모델 보고서에서는 반드시 요약과 데이터 조건에 명시하세요.\n"
        "이 값만 근거로 바로 공유 가능한 한국어 Markdown 성능 분석 자료를 작성하세요.\n"
        "각 장 제목에는 1. 요약, 2. 데이터 및 검증 조건처럼 번호를 붙이세요.\n"
        "필수 장을 생략하지 말고 PDF 순서와 동일하게 출력하세요: 1. 요약, 2. 데이터 및 검증 조건, 3. 모델 성능 평가, "
        "4. 후보 모델 비교, 5. 검증 시각화 해석, 6. 예측용 파일 분석(제공된 경우), 7. 결론.\n"
        "예측용 파일 분석은 prediction_type_label을 기준으로 TRVmax 회귀 예측, 회전기 소음 회귀 예측, 분류 예측을 명확히 구분하세요.\n"
        "후보 모델 비교와 예측용 파일 분석에는 JSON 수치에 근거한 해석 의견을 각각 2-3줄 포함하세요.\n"
        "결론은 종합 판단, 성능 및 모델 비교, 예측용 파일 반영, 운영 적용 시 유의사항, 후속 작업 관점으로 짜임새 있게 작성하세요.\n"
        "본문만 출력하고, tool 이름이나 구현 세부사항은 쓰지 마세요. JSON에 없는 수치는 만들지 마세요.\n\n"
        f"```json\n{_safe_json_dumps(compact_payload, limit=7200)}\n```"
    )
    report_runtime = _action_runtime(runtime, max_tokens=2600)
    try:
        report_runtime["max_tokens"] = max(int(report_runtime.get("max_tokens") or 0), 2200)
    except (TypeError, ValueError):
        report_runtime["max_tokens"] = 2200
    stream_llm_answer_to_chat(
        report_runtime,
        [{"role": "user", "content": prompt}],
        title=f"{family_title} 성능 분석 자료 생성",
        trace="stored model comparison payload -> local LLM analysis generation.",
        tool_result={**payload, "tool": "model.report_source", "prediction_report": prediction_payload},
        fallback_content=fallback,
        save_report=True,
        report_task=task,
        report_model_id=model_id,
        report_payload=payload,
        prediction_payload=prediction_payload,
    )


def is_model_diagnostic_request(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    return any(token in normalized for token in ("성능진단", "모델진단", "모델비교", "성능비교", "diagnostic", "diagnosis", "comparemodel", "modelcompare"))


def is_report_request(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    return any(token in normalized for token in ("보고서", "리포트", "report", "보고서생성", "리포트생성"))


def workflow_validation_message(record: Any) -> tuple[str, dict[str, Any], str]:
    summary = dataframe_summary(record.dataframe, max_preview_rows=5)
    missing = missing_summary(record.dataframe)
    missing_top = missing[missing["missing_count"] > 0].head(8).to_dict(orient="records")
    target = target_distribution(record.dataframe, record.dataset_type)
    tool_result = {
        "ok": True,
        "tool": "workflow.data_validation_and_type_inference",
        "dataset_id": record.dataset_id,
        "filename": record.filename,
        "dataset_type": record.dataset_type,
        "summary": summary,
        "missing_top": missing_top,
        "target_distribution": target,
    }
    lines = [
        "데이터셋을 등록하고 기본 검증을 완료했습니다.",
        f"- dataset_id: `{record.dataset_id}`",
        f"- 파일: `{record.filename}`",
        f"- shape: `{summary['rows']} rows x {summary['columns']} columns`",
        f"- 판별된 데이터 유형: `{record.dataset_type}`",
    ]
    if missing_top:
        lines.append("- 결측치 상위 컬럼: " + ", ".join(f"{row['column']}={row['missing_count']}" for row in missing_top))
    else:
        lines.append("- 결측치: 주요 결측치 없음")
    if target:
        lines.append("- target/distribution 정보는 detail JSON에 기록했습니다.")
    reasoning = (
        "실행 trace: read_excel_file -> make_dataset_record -> infer_dataset_type -> "
        "dataframe_summary -> missing_summary -> target_distribution."
    )
    return "\n".join(lines), tool_result, reasoning


def workflow_training_message(result: dict[str, Any], task: str) -> tuple[str, str]:
    if not result.get("ok"):
        return f"{task} 모델 학습에 실패했습니다.\n\n- 원인: {result.get('message') or result.get('failure_code')}", (
            f"실행 trace: train_tool(task={task}) 호출 후 실패 응답을 받았습니다."
        )
    metrics = result.get("metrics", {})
    engine = result.get("schema", {}).get("engine") or metrics.get("engine") or "unknown"
    lines = [
        f"{task} 모델 학습을 완료했습니다.",
        f"- engine: `{engine}`",
        f"- model_id: `{result.get('model_id')}`",
        f"- best_model: `{metrics.get('best_model')}`",
        f"- selection_metric: `{metrics.get('selection_metric')}`",
        f"- holdout: {format_holdout(metrics)}",
    ]
    if metrics.get("n_train") is not None or metrics.get("n_test") is not None:
        lines.append(f"- train/test rows: `{metrics.get('n_train')}` / `{metrics.get('n_test')}`")
    if metrics.get("n_rows_after_dropna") is not None:
        lines.append(f"- rows after dropna: `{metrics.get('n_rows_after_dropna')}`")
    if metrics.get("compare_models"):
        lines.append("- 후보 모델 비교 결과 표가 아래에 표시됩니다.")
    if metrics.get("dropped_rows") is not None:
        lines.append(f"- preprocessing dropped_rows: `{metrics.get('dropped_rows')}`")
    reasoning = (
        f"실행 trace: train_tool(task={task}) -> model training worker -> notebook 기준 preprocessing/dropna -> "
        "setup -> compare_models -> best model 저장 -> latest 갱신."
    )
    return "\n".join(lines), reasoning


def workflow_validation_prediction_message(result: dict[str, Any], task: str) -> tuple[str, str]:
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    split = metrics.get("validation_split") if isinstance(metrics.get("validation_split"), dict) else {}
    rows = split.get("validation_rows") or len(metrics.get("validation_predictions") or [])
    lines = [
        "10% 검증 세트에 대한 예측 결과를 정리했습니다.",
        "- split: `train 90% / validation 10%`",
        f"- task: `{task}`",
        f"- validation rows: `{rows}`",
        "- 아래 표는 학습에 쓰지 않은 10% validation set에 대한 실제값/예측값 결과입니다.",
    ]
    if task == "classification":
        lines.append("- classification은 `actual_label`, `predicted_label`, `correct` 컬럼으로 맞고 틀린 row를 확인합니다.")
    elif task == "regression":
        lines.append("- regression은 `residual`, `absolute_error` 컬럼으로 row별 오차를 확인합니다.")
    reasoning = "실행 trace: training setup(train_size=0.9) 내부 holdout split -> predict_model(best_model) -> validation row 예측 결과 저장."
    return "\n".join(lines), reasoning


def workflow_validation_metrics_message(result: dict[str, Any], task: str) -> tuple[str, str]:
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    split = metrics.get("validation_split") if isinstance(metrics.get("validation_split"), dict) else {}
    rows = split.get("validation_rows") or len(metrics.get("validation_predictions") or [])
    lines = [
        "10% validation set 기준 정량지표를 계산했습니다.",
        f"- validation rows: `{rows}`",
        "- 모든 값은 저장된 검증 세트 예측 결과에서 가져왔습니다.",
    ]
    if task == "classification":
        lines.append("- 주요 지표: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC")
        if metrics.get("validation_confusion_matrix"):
            lines.append("- confusion matrix를 함께 표시합니다.")
    elif task == "regression":
        lines.append("- 주요 지표: MAE, MSE, RMSE, R2, RMSLE, MAPE")
        if metrics.get("validation_residual_summary"):
            lines.append("- residual/absolute error 요약을 함께 표시합니다.")
    reasoning = "실행 trace: predict_model(best_model) -> metric table -> validation confusion/residual summary 산출."
    return "\n".join(lines), reasoning


def workflow_status_write(status: Any, text: str, *, label: str | None = None) -> None:
    if status is None:
        return
    try:
        if label:
            status.update(label=label, state="running")
        status.write(text)
    except Exception:
        pass


def workflow_progress(status: Any, value: float, text: str = "") -> None:
    if status is None:
        return
    try:
        if hasattr(status, "progress"):
            status.progress(value, text)
    except Exception:
        pass


def train_tool_with_live_progress(ctx: ToolContext, task: str, status: Any) -> dict[str, Any]:
    """Run blocking model training while the chat UI keeps advancing progress."""
    result_box: dict[str, Any] = {"done": False, "result": None}

    def worker() -> None:
        try:
            result_box["result"] = train_tool(ctx, task=task)
        except Exception as exc:
            result_box["result"] = {
                "ok": False,
                "failure_code": "TRAINING_FAILED",
                "message": str(exc),
                "task": task,
            }
        finally:
            result_box["done"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started_at = time.time()
    last_percent = -1

    while not result_box["done"]:
        elapsed = time.time() - started_at
        # The training backend does not expose stable per-model callbacks here, so use an
        # asymptotic estimate and reserve the final jump for the real result.
        progress_value = min(0.86, 0.52 + (1.0 - pow(0.92, elapsed)) * 0.34)
        percent = int(progress_value * 100)
        if percent != last_percent:
            workflow_progress(status, progress_value, f"2/5 모델 학습 중... {percent}%")
            last_percent = percent
        time.sleep(0.5)

    thread.join(timeout=0)
    result = result_box.get("result")
    if isinstance(result, dict):
        return result
    return {
        "ok": False,
        "failure_code": "TRAINING_FAILED",
        "message": "Training finished without a result payload.",
        "task": task,
    }


def workflow_prediction_message(result: dict[str, Any], task: str) -> tuple[str, str]:
    if not result.get("ok"):
        return f"{task} batch prediction에 실패했습니다.\n\n- 원인: {result.get('message') or result.get('failure_code')}", (
            f"실행 trace: predict_tool(task={task}) 호출 후 실패 응답을 받았습니다."
        )
    summary = result.get("summary", {})
    lines = [
        f"{task} batch prediction을 완료했습니다.",
        f"- result_id: `{result.get('result_id')}`",
        f"- model_id: `{result.get('model_id')}`",
        f"- rows: `{summary.get('rows')}`",
        "- 예측 preview와 다운로드 버튼은 이 메시지를 펼치면 표시됩니다.",
    ]
    reasoning = "실행 trace: validate_columns_against_schema -> prepare_prediction_features -> saved model predict -> result dataframe 저장."
    return "\n".join(lines), reasoning


def prediction_form_task_from_text(text: str) -> str | None:
    normalized = text.lower().replace(" ", "")
    if not normalized:
        return None
    if any(token in normalized for token in ("분류예측", "classificationpredict", "classificationprediction", "classprediction")):
        return "classification"
    if any(token in normalized for token in ("회귀예측", "trvmax예측", "trv예측", "regressionpredict", "regressionprediction")):
        return "regression"
    return None


def prediction_input_request_task_from_text(text: str) -> str | None:
    normalized = text.lower().replace(" ", "")
    if not normalized:
        return None
    explicit = prediction_form_task_from_text(text)
    if explicit:
        return explicit
    prediction_tokens = ("예측요청", "예측입력", "예측폼", "입력폼", "변수입력", "predictionrequest", "predictionform")
    if any(token in normalized for token in prediction_tokens):
        hinted = prediction_hint_task_from_text(text)
        if hinted:
            return hinted
        selected = str(st.session_state.get("selected_task") or "")
        return selected if selected in TASKS else "classification"
    generic_prediction_only = normalized in {
        "예측",
        "예측해줘",
        "예측해주세요",
        "예측하고싶어",
        "예측하고싶습니다",
        "predict",
        "prediction",
    }
    if generic_prediction_only:
        hinted = prediction_hint_task_from_text(text)
        if hinted:
            return hinted
        selected = str(st.session_state.get("selected_task") or "")
        return selected if selected in TASKS else "classification"
    return None


def prediction_hint_task_from_text(text: str) -> str | None:
    normalized = text.lower().replace(" ", "")
    if not normalized:
        return None
    if any(token in normalized for token in ("회귀", "regression", "trvmax", "trv")):
        return "regression"
    if any(token in normalized for token in ("분류", "classification", "classify", "class")):
        return "classification"
    return None


def model_id_for_prediction_task(task: str) -> str:
    selected = str(st.session_state.get("selected_model_id", "latest") or "latest")
    if st.session_state.get("selected_task") == task and selected in artifact_ids(task):
        return selected
    return "latest"


def prediction_schema_validation(df: pd.DataFrame, task: str, model_id: str) -> dict[str, Any]:
    artifact = load_artifact(task, model_id, MODEL_ROOT)
    validation = validate_columns_against_schema(list(df.columns), artifact["schema"])
    return {
        "task": task,
        "model_id": model_id,
        "schema": artifact["schema"],
        "validation": validation,
    }


def infer_prediction_task_for_dataframe(df: pd.DataFrame, *, prompt: str = "") -> tuple[str, str, str]:
    hinted = prediction_hint_task_from_text(prompt)
    if hinted:
        return hinted, model_id_for_prediction_task(hinted), "요청 문구에서 예측 유형을 확인했습니다."

    dataset_type = infer_dataset_type(list(df.columns))
    dataset_task = task_for_dataset_type(dataset_type)
    if dataset_task:
        return dataset_task, model_id_for_prediction_task(dataset_task), f"파일 컬럼 구성으로 `{dataset_type}`을 판별했습니다."

    candidates: list[dict[str, Any]] = []
    for task in TASKS:
        model_id = model_id_for_prediction_task(task)
        try:
            check = prediction_schema_validation(df, task, model_id)
            validation = check["validation"]
            candidates.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "ok": bool(validation.get("ok")),
                    "missing": len(validation.get("missing_columns") or []),
                }
            )
        except Exception:
            continue
    ok_candidates = [item for item in candidates if item["ok"]]
    if len(ok_candidates) == 1:
        item = ok_candidates[0]
        return str(item["task"]), str(item["model_id"]), "파일 컬럼이 저장 모델 스키마와 일치하는 유형을 선택했습니다."
    if len(ok_candidates) > 1:
        preferred = st.session_state.get("selected_task") if st.session_state.get("selected_task") in TASKS else ok_candidates[0]["task"]
        for item in ok_candidates:
            if item["task"] == preferred:
                return str(item["task"]), str(item["model_id"]), "분류/회귀 스키마가 모두 맞아 현재 선택된 모델 유형을 사용했습니다."
        item = ok_candidates[0]
        return str(item["task"]), str(item["model_id"]), "분류/회귀 스키마가 모두 맞아 첫 번째 사용 가능 유형을 선택했습니다."

    if candidates:
        best = sorted(candidates, key=lambda item: item["missing"])[0]
        return str(best["task"]), str(best["model_id"]), "완전 일치 스키마가 없어 누락 변수가 가장 적은 유형으로 검증을 시도합니다."
    fallback = st.session_state.get("selected_task") if st.session_state.get("selected_task") in TASKS else "classification"
    return fallback, "latest", "현재 선택된 모델 유형을 사용합니다."


def prediction_result_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if str(col).startswith("prediction") or str(col).startswith("probability") or str(col).startswith("predicted_")
    ]


def prediction_result_summary_lines(df: pd.DataFrame) -> list[str]:
    lines = [f"rows: `{len(df)}`"]
    regression_cols = regression_prediction_columns(df)
    if regression_cols:
        pred_col = regression_cols[0]
        ranked = prediction_with_regression_errors(df)
        values = pd.to_numeric(ranked[pred_col], errors="coerce").dropna()
        if values.empty:
            return lines
        if len(values) == 1:
            lines.append(f"{pred_col}: `{values.iloc[0]:.6g}`")
        else:
            lines.append(f"{pred_col} min/max: `{values.min():.6g}` / `{values.max():.6g}`")
        if "absolute_error" in ranked.columns:
            errors = pd.to_numeric(ranked["absolute_error"], errors="coerce").dropna()
            if not errors.empty:
                lines.append(f"absolute_error min/median/max: `{errors.min():.6g}` / `{errors.median():.6g}` / `{errors.max():.6g}`")
        return lines

    if "prediction_label" in df.columns:
        counts = df["prediction_label"].value_counts(dropna=False).to_dict()
        lines.append("classification counts: `" + json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False) + "`")
    elif "prediction" in df.columns:
        counts = df["prediction"].value_counts(dropna=False).to_dict()
        lines.append("classification counts: `" + json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False) + "`")
    return lines


def prediction_minmax_frame(df: pd.DataFrame) -> pd.DataFrame:
    regression_cols = regression_prediction_columns(df)
    if not regression_cols:
        return pd.DataFrame()
    rows = []
    for column in regression_cols:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            rows.append({"column": column, "min": float(values.min()), "max": float(values.max())})
    return pd.DataFrame(rows)


def prediction_row_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    ranked = prediction_ranked_by_error_if_available(df)
    base_cols = [col for col in ("Result", "TRVmax[kV]", "sound_mean") if col in ranked.columns]
    pred_cols = prediction_result_columns(ranked)
    error_cols = [col for col in ("absolute_error", "residual") if col in ranked.columns]
    cols = base_cols + [col for col in pred_cols if col not in base_cols] + [col for col in error_cols if col not in base_cols and col not in pred_cols]
    if not cols:
        return pd.DataFrame()
    summary = ranked.loc[:, cols].copy()
    source_rows = []
    for index in summary.index:
        if isinstance(index, int):
            source_rows.append(index + 1)
        else:
            source_rows.append(str(index))
    summary.insert(0, "row", source_rows)
    return summary


def prediction_summary_message(result_df: pd.DataFrame, summary: dict[str, Any], *, task: str, model_id: str, source_name: str) -> str:
    rows = int(summary.get("rows") or len(result_df))
    lines = [
        f"`{source_name}` 입력에 대한 {task} 예측을 완료했습니다.",
        f"- model_id: `{model_id}`",
        f"- rows: `{rows}`",
    ]
    pred_cols = prediction_result_columns(result_df)
    if rows == 1 and pred_cols:
        first = result_df.iloc[0]
        regression_cols = regression_prediction_columns(result_df)
        if task == "regression" and regression_cols:
            pred_col = regression_cols[0]
            lines.append(f"- {pred_col}: `{first[pred_col]}`")
        elif task == "classification":
            label = first.get("prediction_label", first.get("prediction", ""))
            lines.append(f"- classification result: `{label}`")
            if "probability_success" in result_df.columns:
                lines.append(f"- probability_success: `{first.get('probability_success')}`")
            if "probability_failure" in result_df.columns:
                lines.append(f"- probability_failure: `{first.get('probability_failure')}`")
    else:
        lines.append("- 예측 preview와 다운로드 버튼은 이 메시지를 펼치면 표시됩니다.")
    return "\n".join(lines)


def store_prediction_result(result_df: pd.DataFrame, summary: dict[str, Any], *, task: str, model_id: str, source_name: str) -> str:
    result_id = f"chat-{task}-{len(st.session_state.prediction_results) + 1}"
    st.session_state.prediction_results[result_id] = result_df
    st.session_state.latest_result_id = result_id
    st.session_state.selected_task = task
    st.session_state.selected_model_id = model_id
    st.session_state.pending_prediction_form_task = None
    st.session_state.pending_prediction_form_model_id = "latest"
    preview = result_df.head(5).where(pd.notna(result_df.head(5)), None).to_dict(orient="records")
    tool_result = {
        "ok": True,
        "tool": "prediction.batch",
        "task": task,
        "source": source_name,
        "model_id": model_id,
        "result_id": result_id,
        "summary": summary,
        "preview": preview,
    }
    append_chat_message(
        "assistant",
        prediction_summary_message(result_df, summary, task=task, model_id=model_id, source_name=source_name),
        title="예측 결과",
        trace="입력 변수 검증 후 저장 모델로 예측하고 결과표를 저장했습니다.",
        tool_result=tool_result,
        result_id=result_id,
        expanded=True,
    )
    return result_id


def run_prediction_dataframe(df: pd.DataFrame, *, task: str, model_id: str, source_name: str) -> str:
    result_df, summary = predict_batch(df, task=task, model_id=model_id, model_root=MODEL_ROOT)
    return store_prediction_result(result_df, summary, task=task, model_id=model_id, source_name=source_name)


def feature_default_values(features: list[str]) -> dict[str, float]:
    defaults = {feature: 0.0 for feature in features}
    record = active_dataset()
    if record is None:
        return defaults
    for feature in features:
        if feature not in record.dataframe.columns:
            continue
        values = pd.to_numeric(record.dataframe[feature], errors="coerce").dropna()
        if not values.empty:
            defaults[feature] = float(values.median())
    return defaults


def render_prediction_input_panel() -> None:
    task = st.session_state.get("pending_prediction_form_task")
    if task not in TASKS:
        return

    with st.container(border=True):
        st.subheader("예측 입력")
        st.caption("저장된 모델의 입력 변수 스키마를 기준으로 Excel batch 예측 또는 1-row 직접 입력 예측을 실행합니다.")
        task_options = list(TASKS)
        task = st.radio("Prediction Task", task_options, index=task_options.index(task), horizontal=True, key="manual_prediction_task")
        st.session_state.pending_prediction_form_task = task

        ids = artifact_ids(task)
        preferred = str(st.session_state.get("pending_prediction_form_model_id") or model_id_for_prediction_task(task))
        model_index = ids.index(preferred) if preferred in ids else 0
        model_id = st.selectbox("Model Artifact", ids, index=model_index, key=f"manual_prediction_model_{task}")
        st.session_state.pending_prediction_form_model_id = model_id

        try:
            artifact = load_artifact(task, model_id, MODEL_ROOT)
            features = [str(col) for col in artifact["schema"].get("features", [])]
        except Exception as exc:
            st.error(f"저장 모델을 로드할 수 없습니다: {exc}")
            return

        st.write(f"필요 입력 변수: `{len(features)}`개")
        uploaded = st.file_uploader("예측용 Excel 업로드", type=["xlsx", "xls"], key=f"manual_prediction_upload_{task}_{model_id}")
        if uploaded is not None:
            if st.button("Excel 예측 실행 →", type="primary", use_container_width=True, key=f"manual_prediction_upload_run_{task}_{model_id}"):
                try:
                    df = read_excel_file(BytesIO(uploaded.getvalue()))
                    append_chat_message("user", f"`{uploaded.name}` 파일로 {task} 예측을 요청했습니다.", title="Excel 예측", expanded=True)
                    run_prediction_dataframe(df, task=task, model_id=model_id, source_name=uploaded.name)
                    st.rerun()
                except Exception as exc:
                    append_chat_message("assistant", f"Excel 예측에 실패했습니다.\n\n- error: `{exc}`", title="예측 실패", expanded=True)
                    st.rerun()

        defaults = feature_default_values(features)
        with st.expander("직접 입력", expanded=False):
            with st.form(f"manual_prediction_form_{task}_{model_id}"):
                values: dict[str, float] = {}
                cols = st.columns(4)
                for index, feature in enumerate(features):
                    key = f"manual_pred_{task}_{model_id}_{hashlib.sha1(feature.encode('utf-8')).hexdigest()[:12]}"
                    values[feature] = cols[index % 4].number_input(
                        feature,
                        value=float(defaults.get(feature, 0.0)),
                        format="%.8g",
                        key=key,
                    )
                submitted = st.form_submit_button("→ 예측 실행", type="primary", use_container_width=True)
            if submitted:
                try:
                    input_df = pd.DataFrame([values], columns=features)
                    append_chat_message("user", f"직접 입력한 변수로 {task} 예측을 요청했습니다.", title="직접 입력 예측", expanded=True)
                    run_prediction_dataframe(input_df, task=task, model_id=model_id, source_name="manual_input")
                    st.rerun()
                except Exception as exc:
                    append_chat_message("assistant", f"직접 입력 예측에 실패했습니다.\n\n- error: `{exc}`", title="예측 실패", expanded=True)
                    st.rerun()


def split_chat_input(value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return "", []
    if isinstance(value, str):
        return value, []
    text = ""
    files: list[Any] = []
    if isinstance(value, dict):
        text = str(value.get("text") or "")
        raw_files = value.get("files") or []
    else:
        text = str(getattr(value, "text", "") or "")
        raw_files = getattr(value, "files", []) or []
    if isinstance(raw_files, (list, tuple)):
        files = list(raw_files)
    elif raw_files:
        files = [raw_files]
    return text, files


def is_training_upload_request(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    return any(
        token in normalized
        for token in (
            "학습",
            "훈련",
            "모델학습",
            "재학습",
            "트레이닝",
            "training",
            "train",
            "fitmodel",
        )
    )


def handle_chat_uploads(prompt: str, files: list[Any]) -> None:
    names = ", ".join(str(getattr(file, "name", "uploaded.xlsx")) for file in files)
    request_text = prompt.strip() or "첨부한 Excel 파일로 예측해주세요."

    if is_training_upload_request(prompt):
        append_chat_message("user", f"{request_text}\n\n첨부 파일: `{names}`", title="학습용 Excel 업로드", expanded=True)
        if len(files) > 1:
            append_chat_message(
                "assistant",
                "학습 workflow는 한 번에 하나의 학습용 Excel 파일을 기준으로 실행합니다. 첫 번째 파일로 학습을 시작하고, 나머지는 필요할 때 다시 업로드해주세요.",
                title="학습 파일 확인",
                expanded=True,
            )
        uploaded = files[0]
        st.session_state.processed_upload_fingerprint = None
        register_uploaded_dataset(uploaded)
        st.session_state.pending_workflow_source_message = None
        persist_ui_state()
        return

    append_chat_message("user", f"{request_text}\n\n첨부 파일: `{names}`", title="Excel 예측 요청", expanded=True)

    for uploaded in files:
        filename = str(getattr(uploaded, "name", "uploaded.xlsx"))
        try:
            data = uploaded.getvalue()
            df = read_excel_file(BytesIO(data))
            task, model_id, reason = infer_prediction_task_for_dataframe(df, prompt=prompt)
            result_df, summary = predict_batch(df, task=task, model_id=model_id, model_root=MODEL_ROOT)
            result_id = store_prediction_result(result_df, summary, task=task, model_id=model_id, source_name=filename)
            st.session_state.chat_messages[-1]["content"] += f"\n\n- 유형 판별: {reason}\n- result_id: `{result_id}`"
            persist_ui_state()
        except Exception as exc:
            append_chat_message(
                "assistant",
                f"`{filename}` 예측에 실패했습니다.\n\n- error: `{exc}`\n\n필요하면 `분류 예측` 또는 `회귀 예측`이라고 입력해서 변수 입력 폼으로 한 row씩 확인할 수 있습니다.",
                title="예측 실패",
                expanded=True,
            )


def run_record_workflow(
    record: Any,
    *,
    runtime: dict[str, Any] | None = None,
    source_message: str | None = None,
    force: bool = False,
    status: Any = None,
) -> None:
    completed = st.session_state.workflow_completed_dataset_ids
    if not isinstance(completed, set):
        completed = set(completed)
        st.session_state.workflow_completed_dataset_ids = completed
    if not force and record.dataset_id in completed:
        workflow_status_write(status, "이미 완료된 분석입니다.")
        try:
            status.update(label="분석 완료됨", state="complete")
        except Exception:
            pass
        return
    if force and record.dataset_id in completed:
        completed.remove(record.dataset_id)

    if source_message:
        append_chat_message("user", source_message, title="데이터셋 업로드", expanded=True)

    workflow_status_write(status, "1. 데이터 검증 및 유형 판별 실행 중...", label="1/5 데이터 검증 중")
    content, tool_result, reasoning = workflow_validation_message(record)
    append_chat_message("assistant", content, title="1. 데이터 검증 및 유형 판별", trace=reasoning, tool_result=tool_result, expanded=True)
    workflow_status_write(status, "1. 데이터 검증 및 유형 판별 완료")

    task = task_for_dataset_type(record.dataset_type)
    if task is None:
        append_chat_message(
            "assistant",
            (
                "이 데이터셋은 학습용 target 구성이 명확하지 않아 자동 학습을 중단했습니다.\n\n"
                "- classification 학습에는 `Result`, `TRVmax[kV]` 컬럼이 필요합니다.\n"
                "- regression 학습에는 `Time`, `CZM`, `Test`, `Result`, `TRVmax[kV]` 기준 구성이 필요합니다.\n"
                "- 회전기 소음 회귀 학습에는 `sound_mean` target과 `sound_front/rear/right/left` 측정 컬럼이 필요합니다.\n"
                "- 기존 모델로 예측하려면 채팅에 `저장된 분류 모델로 예측해줘`처럼 요청하세요."
            ),
            title="2. 자동 학습 보류",
            trace="데이터 유형이 prediction_input_or_unknown으로 판별되어 target 기반 학습 tool을 호출하지 않았습니다.",
            tool_result=tool_result,
            expanded=True,
        )
        completed.add(record.dataset_id)
        workflow_status_write(status, "학습용 목표값 구성이 명확하지 않아 자동 분석을 보류했습니다.", label="분석 보류")
        try:
            status.update(label="분석 보류", state="complete")
        except Exception:
            pass
        return

    st.session_state.selected_task = task
    workflow_status_write(status, f"2. {task} 예측 모델 학습을 시작합니다.", label="2/5 모델 학습 중")
    workflow_progress(status, 0.30, "2/5 모델 학습 준비 중")
    workflow_status_write(status, "학습 데이터 전처리 및 학습 엔진 초기화 중...")
    workflow_progress(status, 0.38, "2/5 전처리 및 setup")
    workflow_status_write(status, "후보 모델 비교와 holdout 검증을 실행 중입니다. 데이터 크기에 따라 시간이 걸릴 수 있습니다.")
    workflow_progress(status, 0.52, "2/5 후보 모델 비교 중")
    train_result = train_tool_with_live_progress(tool_context(), task, status)
    workflow_progress(status, 0.88, "2/5 학습 결과 정리 중")
    if train_result.get("ok") and train_result.get("model_id"):
        st.session_state.selected_model_id = str(train_result["model_id"])
    content, reasoning = workflow_training_message(train_result, task)
    append_chat_message("assistant", content, title="2. 모델 학습", trace=reasoning, tool_result=train_result, expanded=True)
    if not train_result.get("ok"):
        completed.add(record.dataset_id)
        workflow_status_write(status, "2. 모델 학습 실패", label="분석 실패")
        try:
            status.update(label="분석 실패", state="error")
        except Exception:
            pass
        return
    workflow_progress(status, 1.0, "2/5 모델 학습 완료")
    workflow_status_write(status, f"2. 모델 학습 완료: `{train_result.get('metrics', {}).get('best_model')}`")

    model_id = str(train_result.get("model_id") or "latest")
    schedule_post_training_workflow_steps(train_result=train_result, task=task, model_id=model_id)
    workflow_status_write(status, "3. 입력 변수 검증 단계로 이동합니다.", label="3/5 변수 검증 준비")
    return


def run_post_training_workflow_step(record: Any, *, status: Any = None) -> bool:
    completed = st.session_state.workflow_completed_dataset_ids
    if not isinstance(completed, set):
        completed = set(completed)
        st.session_state.workflow_completed_dataset_ids = completed

    train_result = st.session_state.get("pending_workflow_train_result")
    if not isinstance(train_result, dict):
        clear_pending_workflow_runtime_state()
        completed.add(record.dataset_id)
        append_chat_message(
            "assistant",
            "학습 결과 상태를 찾을 수 없어 후속 검증 단계를 중단했습니다. 학습 workflow를 다시 실행해주세요.",
            title="후속 단계 중단",
            expanded=True,
        )
        return True

    task = str(st.session_state.get("pending_workflow_task") or train_result.get("task") or st.session_state.get("selected_task") or "classification")
    if task not in TASKS:
        task = "classification"
    model_id = str(st.session_state.get("pending_workflow_model_id") or train_result.get("model_id") or "latest")
    stage = str(st.session_state.get("pending_workflow_stage") or "")

    if stage == "schema":
        workflow_status_write(status, "입력 변수 구성을 검증하고 있습니다.", label="3/5 변수 검증 중")
        workflow_progress(status, 0.08, "3/5 입력 변수 검증 준비")
        time.sleep(0.25)
        schema_result = validate_schema(tool_context(), task=task, model_id=model_id)
        workflow_progress(status, 1.0, "3/5 입력 변수 검증 완료")
        append_chat_message(
            "assistant",
            (
                "입력 변수 구성을 확인했습니다.\n\n"
                f"- 검증 결과: `{schema_result.get('ok')}`\n"
                f"- 누락 변수: `{schema_result.get('missing_columns')}`\n"
                f"- 예측에서 제외되는 변수: `{schema_result.get('ignored_present')}`"
            ),
            title="3. 입력 변수 검증",
            trace="학습 직후 생성된 저장 모델 기준으로 입력 변수 구성을 확인했습니다.",
            tool_result=schema_result,
            expanded=True,
        )
        if not schema_result.get("ok"):
            completed.add(record.dataset_id)
            clear_pending_workflow_runtime_state()
            workflow_status_write(status, "3. 입력 변수 검증 실패", label="분석 실패")
            try:
                status.update(label="분석 실패", state="error")
            except Exception:
                pass
            return True
        st.session_state.pending_workflow_stage = "validation_predictions"
        workflow_status_write(status, "4. 검증 세트 예측 단계로 이동합니다.", label="4/5 검증 세트 예측 준비")
        return False

    if stage == "validation_predictions":
        workflow_status_write(status, "10% validation set 예측 결과를 정리하고 있습니다.", label="4/5 검증 세트 예측 중")
        workflow_progress(status, 0.08, "4/5 검증 세트 예측 결과 준비")
        time.sleep(0.25)
        content, reasoning = workflow_validation_prediction_message(train_result, task)
        workflow_progress(status, 1.0, "4/5 검증 세트 예측 결과 완료")
        append_chat_message(
            "assistant",
            content,
            title="4. 10% 검증 세트 예측 결과",
            trace=reasoning,
            tool_result=train_result,
            show_validation=True,
            pycaret_view="validation_predictions",
            expanded=True,
        )
        st.session_state.pending_workflow_stage = "validation_metrics"
        workflow_status_write(status, "5. 정량지표 정리 단계로 이동합니다.", label="5/5 정량지표 준비")
        return False

    if stage == "validation_metrics":
        workflow_status_write(status, "검증 세트 정량지표를 정리하고 있습니다.", label="5/5 정량지표 정리 중")
        workflow_progress(status, 0.08, "5/5 검증 지표 및 그림 준비")
        time.sleep(0.25)
        content, reasoning = workflow_validation_metrics_message(train_result, task)
        workflow_progress(status, 1.0, "5/5 정량지표 정리 완료")
        append_chat_message(
            "assistant",
            content,
            title="5. 검증 세트 정량지표",
            trace=reasoning,
            tool_result=train_result,
            show_validation=True,
            pycaret_view="validation_metrics",
            expanded=True,
        )
        completed.add(record.dataset_id)
        clear_pending_workflow_runtime_state()
        workflow_status_write(status, "5. 검증 세트 정량지표 정리 완료", label="분석 완료")
        try:
            status.update(label="분석 완료", state="complete")
        except Exception:
            pass
        return True

    clear_pending_workflow_runtime_state()
    completed.add(record.dataset_id)
    workflow_status_write(status, "후속 workflow 상태가 유효하지 않아 종료했습니다.", label="분석 종료")
    return True


def register_uploaded_dataset(uploaded: Any) -> None:
    fingerprint, data = uploaded_fingerprint(uploaded)
    if st.session_state.processed_upload_fingerprint == fingerprint:
        return
    st.session_state.processed_upload_fingerprint = fingerprint
    try:
        df = read_excel_file(BytesIO(data))
        inferred = infer_dataset_type(list(df.columns))
        record = make_dataset_record(uploaded.name, df, inferred)
        st.session_state.datasets[record.dataset_id] = record
        st.session_state.active_dataset_id = record.dataset_id
    except Exception as exc:
        append_chat_message("assistant", f"Excel 로딩에 실패했습니다.\n\n- error: `{exc}`", title="업로드 실패", expanded=True)
        return

    st.session_state.pending_workflow_dataset_id = record.dataset_id
    st.session_state.pending_workflow_source_message = f"`{uploaded.name}` 파일을 업로드했습니다."
    st.session_state.pending_workflow_force = True
    clear_pending_workflow_runtime_state()
    if not bool(st.session_state.get("upload_panel_auto_collapsed_once")):
        st.session_state.upload_panel_expanded = False
        st.session_state.upload_panel_auto_collapsed_once = True
        st.session_state.upload_panel_collapse_requested = True
    persist_ui_state()


def run_dataset_workflow(uploaded: Any, *, runtime: dict[str, Any] | None = None, status: Any = None) -> None:
    register_uploaded_dataset(uploaded)
    record = active_dataset()
    if record is not None:
        run_record_workflow(record, runtime=runtime, source_message=f"`{record.filename}` 파일을 업로드했습니다.", force=True, status=status)


def render_pending_workflow(runtime: dict[str, Any]) -> bool:
    dataset_id = st.session_state.pending_workflow_dataset_id
    if not dataset_id:
        return False
    record = st.session_state.datasets.get(dataset_id)
    if record is None:
        st.session_state.pending_workflow_dataset_id = None
        st.session_state.pending_workflow_source_message = None
        st.session_state.pending_workflow_force = False
        return False

    class ChatProgress:
        def __init__(self) -> None:
            self.label = "응답을 준비하고 있습니다."
            self.lines: list[str] = []
            self.placeholder = st.empty()
            self.progress_placeholder = st.empty()
            self.render()

        def update(self, label: str | None = None, state: str | None = None) -> None:
            if label:
                self.label = label
            if state == "complete":
                self.label = "분석 완료"
            elif state == "error":
                self.label = "확인 필요"
            self.render()

        def write(self, text: str) -> None:
            self.lines.append(str(text))
            self.render()

        def progress(self, value: float, text: str = "") -> None:
            safe_value = min(max(float(value), 0.0), 1.0)
            label = text or f"{int(safe_value * 100)}%"
            self.progress_placeholder.progress(safe_value, text=label)

        def render(self) -> None:
            body = [f"**{self.label}**"]
            body.extend(f"- {line}" for line in self.lines[-10:])
            self.placeholder.markdown("\n".join(body))

    with chat_message("assistant"):
        progress = ChatProgress()
        progress.write(f"Dataset: `{record.filename}`")
        progress.write(f"Model: `{runtime.get('model')}`")
        if st.session_state.get("pending_workflow_stage"):
            finished = run_post_training_workflow_step(record, status=progress)
        else:
            run_record_workflow(
                record,
                runtime=runtime,
                source_message=st.session_state.pending_workflow_source_message,
                force=bool(st.session_state.pending_workflow_force),
                status=progress,
            )
            finished = not bool(st.session_state.get("pending_workflow_stage"))

    if finished:
        st.session_state.pending_workflow_dataset_id = None
        st.session_state.pending_workflow_source_message = None
        st.session_state.pending_workflow_force = False
        clear_pending_workflow_runtime_state()
    persist_ui_state()
    st.rerun()
    return True


def render_result_dataframe(result_id: str) -> None:
    df = st.session_state.prediction_results.get(result_id)
    if df is None:
        st.info(f"`{result_id}` result dataframe이 현재 세션에 없습니다.")
        return
    st.dataframe(df.head(100), use_container_width=True)
    _result_downloads(df, result_id)
    summary_lines = prediction_result_summary_lines(df)
    if summary_lines:
        st.markdown("**예측 결과 요약**")
        st.caption(f"예측 구분: {prediction_report_type_label(prediction_report_domain(df))}")
        st.markdown("\n".join(f"- {line}" for line in summary_lines))
    minmax = prediction_minmax_frame(df)
    if not minmax.empty:
        st.markdown("**예측값 Min/Max**")
        st.dataframe(minmax, use_container_width=True, hide_index=True)
    row_summary = prediction_row_summary_frame(df) if regression_prediction_columns(df) else pd.DataFrame()
    if not row_summary.empty:
        ranked_by_error = "absolute_error" in row_summary.columns
        title = "**Row별 예측 요약 상위 10개**" + (" *(absolute_error 작은 순)*" if ranked_by_error else "")
        st.markdown(title)
        st.dataframe(row_summary.head(10), use_container_width=True, hide_index=True)
        if len(row_summary) > 10:
            if ranked_by_error:
                st.caption(f"화면에는 실제값 대비 예측 오차가 작은 상위 10개 row만 표시합니다. 전체 {len(row_summary)}개 row는 위 다운로드 파일에 포함됩니다.")
            else:
                st.caption(f"화면에는 상위 10개 row만 표시합니다. 전체 {len(row_summary)}개 row는 위 다운로드 파일에 포함됩니다.")


def metric_long_frame(rows: list[dict[str, Any]], metric_names: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["metric", "value"])
    first = rows[0]
    values = []
    for metric in metric_names:
        if metric not in first:
            continue
        value = pd.to_numeric(pd.Series([first.get(metric)]), errors="coerce").iloc[0]
        if pd.notna(value):
            values.append({"metric": metric, "value": float(value)})
    return pd.DataFrame(values)


def render_metric_bar_chart(rows: list[dict[str, Any]], metric_names: list[str], *, title: str, y_title: str = "Value") -> None:
    chart_df = metric_long_frame(rows, metric_names)
    if chart_df.empty:
        return
    st.markdown(f"**{title}**")
    if alt is None:
        st.bar_chart(chart_df.set_index("metric")["value"], use_container_width=True)
        return
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("metric:N", title=None, sort=None),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("metric:N", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=[alt.Tooltip("metric:N", title="Metric"), alt.Tooltip("value:Q", title="Value", format=".4f")],
        )
        .properties(height=260)
    )
    text = chart.mark_text(dy=-8, color="#1f2937").encode(text=alt.Text("value:Q", format=".3f"))
    st.altair_chart(chart + text, use_container_width=True)


def render_confusion_matrix_chart(confusion: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for row in confusion.get("rows") or []:
        actual = row.get("actual")
        for key, value in row.items():
            if not str(key).startswith("predicted_"):
                continue
            rows.append(
                {
                    "actual": str(actual),
                    "predicted": str(key).removeprefix("predicted_"),
                    "count": int(value or 0),
                }
            )
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        return
    st.markdown("**Confusion Matrix Heatmap**")
    if alt is None:
        st.bar_chart(chart_df.pivot(index="actual", columns="predicted", values="count").fillna(0), use_container_width=True)
        return
    heatmap = (
        alt.Chart(chart_df)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("actual:N", title="Actual"),
            color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("actual:N", title="Actual"),
                alt.Tooltip("predicted:N", title="Predicted"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=300)
    )
    labels = heatmap.mark_text(fontSize=18, fontWeight="bold").encode(text=alt.Text("count:Q"), color=alt.value("#111827"))
    st.altair_chart(heatmap + labels, use_container_width=True)


def render_regression_validation_charts(metrics: dict[str, Any]) -> None:
    validation_metrics = metrics.get("validation_metrics") or metrics.get("predict_model_metrics") or []
    render_metric_bar_chart(validation_metrics, ["MAE", "RMSE", "RMSLE", "MAPE"], title="Regression Error Metrics", y_title="Error")
    render_metric_bar_chart(validation_metrics, ["MSE"], title="Mean Squared Error", y_title="MSE")
    render_metric_bar_chart(validation_metrics, ["R2"], title="R2 Score", y_title="R2")

    validation_predictions = metrics.get("validation_predictions") or []
    if not validation_predictions:
        return
    pred_df = pd.DataFrame(validation_predictions)
    actual_candidates = [col for col in ("TRVmax[kV]", "sound_mean") if col in pred_df.columns]
    actual_col = actual_candidates[0] if actual_candidates else ""
    if alt is not None and actual_col and "prediction_label" in pred_df.columns:
        st.markdown("**Actual vs Predicted**")
        scatter = (
            alt.Chart(pred_df)
            .mark_circle(size=70, opacity=0.72)
            .encode(
                x=alt.X(f"{actual_col}:Q", title=f"Actual {actual_col}"),
                y=alt.Y("prediction_label:Q", title=f"Predicted {actual_col}"),
                tooltip=[
                    alt.Tooltip(f"{actual_col}:Q", title="Actual", format=".4f"),
                    alt.Tooltip("prediction_label:Q", title="Predicted", format=".4f"),
                    alt.Tooltip("absolute_error:Q", title="Abs. Error", format=".4f"),
                ],
                color=alt.Color("absolute_error:Q", title="Abs. Error", scale=alt.Scale(scheme="orangered")),
            )
            .properties(height=320)
        )
        st.altair_chart(scatter, use_container_width=True)
    if alt is not None and "residual" in pred_df.columns:
        st.markdown("**Residual Distribution**")
        hist = (
            alt.Chart(pred_df)
            .mark_bar()
            .encode(
                x=alt.X("residual:Q", bin=alt.Bin(maxbins=18), title="Residual = Predicted - Actual"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count():Q", title="Count")],
                color=alt.value("#2563eb"),
            )
            .properties(height=260)
        )
        st.altair_chart(hist, use_container_width=True)


def render_pycaret_figure_gallery(metrics: dict[str, Any]) -> bool:
    figures = metrics.get("pycaret_figures") if isinstance(metrics.get("pycaret_figures"), list) else []
    ok_figures = [figure for figure in figures if isinstance(figure, dict) and figure.get("ok") and figure.get("path")]
    if not ok_figures:
        st.warning("이 모델 산출물에는 검증 그림이 없습니다. workflow를 다시 실행하면 검증 시각화 결과가 생성됩니다.")
        return False

    st.markdown("**검증 시각화**")
    for offset in range(0, len(ok_figures), 2):
        cols = st.columns(2)
        row_figures = ok_figures[offset : offset + 2]
        for col, figure in zip(cols, row_figures):
            path = Path(str(figure.get("path")))
            with col:
                if not path.exists():
                    st.warning(f"Figure file not found: `{path}`")
                    continue
                st.image(str(path), caption=clean_figure_title(figure.get("title") or figure.get("plot") or path.name), use_container_width=True)
                st.download_button(
                    "PNG 다운로드",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="image/png",
                    key=f"download_pycaret_figure_{hashlib.sha1(str(path).encode()).hexdigest()}",
                    use_container_width=True,
                )
        if len(row_figures) == 1:
            cols[1].empty()

    failed = [figure for figure in figures if isinstance(figure, dict) and not figure.get("ok")]
    if failed:
        with st.expander("생성되지 않은 검증 그림", expanded=False):
            st.json(failed, expanded=False)
    return True


def render_model_diagnostic_payload(payload: dict[str, Any]) -> None:
    if payload.get("tool") not in {"model.performance_diagnostics", "model.report_source"}:
        return
    selected = payload.get("selected_model") if isinstance(payload.get("selected_model"), dict) else {}
    st.markdown("**진단 기준 모델**")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Task", str(payload.get("task") or "-"))
    metric_cols[1].metric("Best Model", str(selected.get("best_model") or "-"))
    if str(payload.get("task")) == "classification":
        metric_cols[2].metric("F1", selected.get("F1", "-"))
        metric_cols[3].metric("AUC", selected.get("AUC", "-"))
    else:
        metric_cols[2].metric("R2", selected.get("R2", "-"))
        metric_cols[3].metric("RMSE", selected.get("RMSE", "-"))

    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    if findings:
        st.markdown("**진단 포인트**")
        st.markdown("\n".join(f"- {item}" for item in findings))

    comparison = payload.get("artifact_comparison") if isinstance(payload.get("artifact_comparison"), list) else []
    if comparison:
        st.markdown("**저장 모델 성능 비교**")
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    candidates = payload.get("pycaret_candidate_comparison") if isinstance(payload.get("pycaret_candidate_comparison"), list) else []
    if candidates:
        st.markdown("**자동 후보 모델 비교**")
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)


def render_candidate_histogram(task: str, rows: list[dict[str, Any]]) -> None:
    metric_name, _higher_is_better, scored_rows = _candidate_score_rows(task, rows, limit=8)
    if not scored_rows:
        st.info("후보 모델 비교 히스토그램에 표시할 지표가 없습니다.")
        return
    chart_df = pd.DataFrame(scored_rows)
    chart_df["model"] = chart_df["model"].map(lambda value: str(value)[:34])
    st.markdown("**후보 모델 비교 히스토그램**")
    if alt is None:
        st.bar_chart(chart_df.set_index("model")["score"], use_container_width=True)
        return
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("model:N", title=None, sort="-x"),
            x=alt.X("score:Q", title=metric_name or "score"),
            color=alt.value("#2f6f9f"),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("score:Q", title=metric_name or "Score", format=".4f")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def compact_dataframe_for_display(rows: list[dict[str, Any]], *, max_rows: int = 10) -> pd.DataFrame:
    display_rows: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        display_rows.append({str(key): _compact_table_value(value, max_chars=34) for key, value in row.items()})
    return pd.DataFrame(display_rows)


def render_report_candidate_visuals(payload: dict[str, Any]) -> None:
    task = str(payload.get("task") or "")
    artifact_rows = payload.get("artifact_comparison") if isinstance(payload.get("artifact_comparison"), list) else []
    if artifact_rows:
        st.markdown("**저장 모델 비교표**")
        st.dataframe(compact_dataframe_for_display(artifact_rows, max_rows=6), use_container_width=True, hide_index=True)

    candidates = payload.get("pycaret_candidate_comparison") if isinstance(payload.get("pycaret_candidate_comparison"), list) else []
    if candidates:
        st.markdown("**자동 후보 모델 비교표**")
        st.dataframe(compact_dataframe_for_display(candidates, max_rows=8), use_container_width=True, hide_index=True)
        render_candidate_histogram(task, candidates)
        st.markdown("**후보 모델 비교 해석 의견**")
        st.markdown("\n".join(f"- {line}" for line in candidate_comparison_opinion_lines(task, payload)))


def render_report_validation_figures(payload: dict[str, Any]) -> None:
    task = str(payload.get("task") or "")
    figures = payload.get("figures") if isinstance(payload.get("figures"), list) else []
    ok_figures = [figure for figure in figures if isinstance(figure, dict) and figure.get("path") and Path(str(figure.get("path"))).exists()]
    if ok_figures:
        st.markdown("**검증 시각화 및 해석**")
        for offset in range(0, len(ok_figures), 2):
            cols = st.columns(2)
            for local_index, (col, figure) in enumerate(zip(cols, ok_figures[offset : offset + 2]), start=1):
                fig_index = offset + local_index
                path = Path(str(figure.get("path")))
                with col:
                    st.image(str(path), use_container_width=True)
                    st.markdown(f"**{figure_caption(fig_index, figure)}**")
                    st.markdown("\n".join(f"- {line}" for line in figure_interpretation_lines(task, figure, payload)[:3]))
            if len(ok_figures[offset : offset + 2]) == 1:
                cols[1].empty()


def render_report_prediction_visuals(payload: dict[str, Any]) -> None:
    prediction_payload = payload.get("prediction_report") if isinstance(payload.get("prediction_report"), dict) else None
    if prediction_payload:
        st.markdown("**예측용 파일 분석 요약**")
        st.caption(f"예측 구분: {prediction_payload.get('prediction_type_label') or prediction_report_type_label(prediction_payload.get('prediction_domain'))}")
        st.markdown("\n".join(f"- {line}" for line in prediction_report_analysis_lines(prediction_payload)))
        st.markdown("**예측 결과 해석 의견**")
        st.markdown("\n".join(f"- {line}" for line in prediction_report_opinion_lines(prediction_payload)))
        numeric_stats = prediction_payload.get("numeric_statistics") if isinstance(prediction_payload.get("numeric_statistics"), list) else []
        class_summary = prediction_payload.get("classification_summary") if isinstance(prediction_payload.get("classification_summary"), list) else []
        probability_stats = prediction_payload.get("probability_statistics") if isinstance(prediction_payload.get("probability_statistics"), list) else []
        if numeric_stats:
            st.dataframe(compact_dataframe_for_display(numeric_stats, max_rows=4), use_container_width=True, hide_index=True)
        if class_summary:
            st.dataframe(compact_dataframe_for_display(class_summary, max_rows=8), use_container_width=True, hide_index=True)
        if probability_stats:
            st.dataframe(compact_dataframe_for_display(probability_stats, max_rows=6), use_container_width=True, hide_index=True)
        preview = prediction_payload.get("preview") if isinstance(prediction_payload.get("preview"), list) else []
        if preview:
            st.dataframe(compact_dataframe_for_display(preview, max_rows=10), use_container_width=True, hide_index=True)


def render_report_visual_preview(msg: dict[str, Any]) -> None:
    payload = msg.get("tool_result")
    if not isinstance(payload, dict) or payload.get("tool") != "model.report_source":
        return
    st.markdown("**시각자료 미리보기**")
    render_report_candidate_visuals(payload)
    render_report_validation_figures(payload)
    render_report_prediction_visuals(payload)


def _report_section_number(line: str) -> int | None:
    match = re.match(r"^\s*#{1,4}\s*(\d+)(?:\.\d+)?[.)]?\s+", str(line or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def render_report_chat_content(content: str, payload: dict[str, Any] | None = None) -> None:
    report_source = isinstance(payload, dict) and payload.get("tool") == "model.report_source"
    prediction_payload = payload.get("prediction_report") if report_source and isinstance(payload.get("prediction_report"), dict) else None
    if report_source:
        content = ensure_report_pdf_order_content(content, payload, prediction_payload)
    model_id_pattern = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?(모델\s*ID|Model\s*ID)(?:\*\*)?\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)

    def render_markdown_block(block: str) -> None:
        buffer: list[str] = []

        def flush_buffer() -> None:
            if buffer:
                st.markdown("\n".join(buffer))
                buffer.clear()

        for line in str(block or "").splitlines():
            match = model_id_pattern.match(line)
            if match:
                flush_buffer()
                label = html.escape(match.group(1).replace(" ", " "))
                value = str(match.group(2)).strip().strip("`").strip("*").strip()
                st.markdown(
                    (
                        '<div class="report-model-id">'
                        f'<span class="report-model-id-label">{label}</span>'
                        f'<span class="report-model-id-value">{html.escape(value)}</span>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
                continue
            buffer.append(line)
        flush_buffer()

    preamble, sections, section_order = _split_numbered_report_sections(content)
    if not sections:
        render_markdown_block(content)
        return

    preamble_text = "\n".join(preamble).strip()
    if preamble_text:
        render_markdown_block(preamble_text)

    rendered_sections: set[int] = set()
    ordered_numbers = report_order_numbers(prediction_payload) if report_source else section_order
    for section_no in ordered_numbers:
        block_lines = sections.get(section_no)
        if not block_lines:
            continue
        render_markdown_block("\n".join(block_lines))
        rendered_sections.add(section_no)
        if report_source and isinstance(payload, dict):
            if section_no == 4:
                render_report_candidate_visuals(payload)
            elif section_no == 5:
                render_report_validation_figures(payload)
            elif section_no == 6 and prediction_payload:
                render_report_prediction_visuals(payload)

    for section_no in section_order:
        if section_no in rendered_sections:
            continue
        block_lines = sections.get(section_no)
        if block_lines:
            render_markdown_block("\n".join(block_lines))


def render_report_download(msg: dict[str, Any], index: int) -> None:
    report_markdown = str(msg.get("report_markdown") or "")
    report_pdf_path = str(msg.get("report_pdf_path") or "")
    if not report_markdown and not report_pdf_path:
        return
    cols = st.columns(2)
    if report_pdf_path and Path(report_pdf_path).exists():
        pdf_bytes = Path(report_pdf_path).read_bytes()
        cols[0].download_button(
            "PDF 다운로드",
            data=pdf_bytes,
            file_name=Path(report_pdf_path).name,
            mime="application/pdf",
            key=f"download_report_pdf_{index}_{hashlib.sha1(pdf_bytes).hexdigest()[:12]}",
            use_container_width=True,
        )
    elif report_pdf_path:
        cols[0].warning(f"PDF 파일을 찾을 수 없습니다: `{report_pdf_path}`")
    if report_markdown:
        report_path = str(msg.get("report_path") or "")
        file_name = Path(report_path).name if report_path else f"model_report_{index + 1}.md"
        cols[1].download_button(
            "Markdown 다운로드",
            data=report_markdown.encode("utf-8"),
            file_name=file_name,
            mime="text/markdown",
            key=f"download_report_md_{index}_{hashlib.sha1(report_markdown.encode('utf-8')).hexdigest()[:12]}",
            use_container_width=True,
        )
    saved_paths = [path for path in (report_pdf_path, str(msg.get("report_path") or "")) if path]
    if saved_paths:
        st.caption("저장 경로: " + " / ".join(f"`{path}`" for path in saved_paths))


def render_pycaret_outputs(tool_result: dict[str, Any], *, show_validation: bool = False, pycaret_view: str | None = None) -> None:
    metrics = tool_result.get("metrics") if isinstance(tool_result, dict) else None
    if not isinstance(metrics, dict) or metrics.get("engine") != "pycaret":
        return
    view = pycaret_view or ("validation_metrics" if show_validation else "training")
    compare_rows = metrics.get("compare_models")
    if view in {"training", "full"} and compare_rows:
        st.markdown("**후보 모델 비교 결과**")
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
    if view == "training":
        return

    split = metrics.get("validation_split") or {}
    if view in {"validation_predictions", "validation_metrics", "full"} and split:
        c1, c2, c3 = st.columns(3)
        c1.metric("학습 비율", split.get("train_size"))
        c2.metric("검증 비율", split.get("validation_size"))
        c3.metric("검증 row 수", split.get("validation_rows"))

    validation_predictions = metrics.get("validation_predictions")
    if view in {"validation_predictions", "full"} and validation_predictions:
        st.markdown("**4. 10% 검증 세트 예측 결과**")
        validation_df = pd.DataFrame(validation_predictions)
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        _result_downloads(validation_df, f"{metrics.get('task', 'model')}_validation_set")

    if view not in {"validation_metrics", "full"}:
        return

    st.markdown("**5. 검증 세트 정량 지표**")
    render_pycaret_figure_gallery(metrics)

    validation_metrics = metrics.get("validation_metrics") or metrics.get("predict_model_metrics")
    if validation_metrics:
        with st.expander("검증 metric raw values", expanded=False):
            st.dataframe(pd.DataFrame(validation_metrics), use_container_width=True, hide_index=True)

    confusion = metrics.get("validation_confusion_matrix")
    if isinstance(confusion, dict) and confusion.get("rows"):
        with st.expander("Confusion matrix raw values", expanded=False):
            st.dataframe(pd.DataFrame(confusion["rows"]), use_container_width=True, hide_index=True)

    residual_summary = metrics.get("validation_residual_summary")
    if isinstance(residual_summary, dict) and residual_summary:
        with st.expander("Regression error raw values", expanded=False):
            st.json(residual_summary, expanded=False)


def render_chat_history() -> None:
    for index, msg in enumerate(st.session_state.chat_messages):
        role = msg.get("role", "assistant")
        with chat_message(role if role in {"user", "assistant"} else "assistant"):
            if role == "user":
                if msg.get("report_markdown") or msg.get("report_pdf_path"):
                    render_report_chat_content(str(msg.get("content", "")))
                else:
                    st.markdown(msg.get("content", ""))
                continue
            title = msg.get("title") or "Assistant"
            expanded = bool(msg.get("expanded", True))
            with st.expander(title, expanded=expanded):
                reasoning_payload = str(msg.get("reasoning") or "")
                if reasoning_payload:
                    st.markdown(
                        reasoning_details_markup(
                            reasoning_payload,
                            live=False,
                            expanded=False,
                            box_id=f"hd-reasoning-history-{index}",
                        ),
                        unsafe_allow_html=True,
                    )
                report_payload = msg.get("tool_result") if isinstance(msg.get("tool_result"), dict) and msg["tool_result"].get("tool") == "model.report_source" else None
                if report_payload:
                    render_report_chat_content(str(msg.get("content", "")), report_payload)
                else:
                    st.markdown(msg.get("content", ""))
                if msg.get("tool_result") is not None:
                    if isinstance(msg["tool_result"], dict):
                        if msg["tool_result"].get("tool") == "model.report_source":
                            pass
                        else:
                            render_model_diagnostic_payload(msg["tool_result"])
                    render_pycaret_outputs(
                        msg["tool_result"],
                        show_validation=bool(msg.get("show_validation")),
                        pycaret_view=str(msg.get("pycaret_view") or "") or None,
                    )
                for tool_index, tool_result in enumerate(msg.get("tool_results") or []):
                    with st.expander(f"세부 결과 {tool_index + 1}", expanded=False):
                        if isinstance(tool_result, dict):
                            render_model_diagnostic_payload(tool_result)
                            render_pycaret_outputs(
                                tool_result,
                                show_validation=bool(msg.get("show_validation")),
                                pycaret_view=str(msg.get("pycaret_view") or "") or None,
                            )
                            result_id = tool_result.get("result_id")
                            if result_id:
                                render_result_dataframe(str(result_id))
                            st.json(tool_result)
                result_id = msg.get("result_id")
                if result_id:
                    render_result_dataframe(str(result_id))
                render_report_download(msg, index)
                if msg.get("tool_result") is not None:
                    show_json = st.toggle("원본 결과 보기 / 닫기", value=False, key=f"tool_json_{index}")
                    if show_json:
                        st.json(msg["tool_result"])


def handle_chat_prompt(prompt: str, runtime: dict[str, Any]) -> None:
    form_task = prediction_input_request_task_from_text(prompt)
    if form_task:
        st.session_state.pending_prediction_form_task = form_task
        st.session_state.pending_prediction_form_model_id = model_id_for_prediction_task(form_task)
        append_chat_message("user", prompt, title="예측 입력 요청", expanded=True)
        append_chat_message(
            "assistant",
            (
                f"`{form_task}` 예측 입력 창을 열었습니다.\n\n"
                "- Excel 파일을 넣으면 여러 행을 한 번에 예측합니다.\n"
                "- 직접 입력 영역에 변수값을 넣으면 한 행 기준으로 예측합니다.\n"
                "- 예측값은 저장된 모델 스키마 검증을 통과한 뒤 계산됩니다."
            ),
            title="예측 입력",
            trace="사용자 요청에 따라 예측 입력 창을 준비했습니다.",
            expanded=True,
        )
        return

    append_chat_message("user", prompt, title="사용자 요청", expanded=True)
    if is_report_request(prompt):
        run_model_report_to_chat(runtime, st.session_state.selected_task, st.session_state.selected_model_id)
        return
    if is_model_diagnostic_request(prompt):
        run_model_diagnostics_to_chat(runtime, st.session_state.selected_task, st.session_state.selected_model_id)
        return

    tool_result = route_tool_request(
        prompt,
        ctx=tool_context(),
        selected_task=st.session_state.selected_task,
        model_id=st.session_state.selected_model_id,
        latest_result_id=st.session_state.latest_result_id,
    )
    if tool_result is not None:
        if tool_result.get("task") in TASKS:
            st.session_state.selected_task = str(tool_result["task"])
        if tool_result.get("result_id"):
            st.session_state.latest_result_id = str(tool_result["result_id"])
        if tool_result.get("model_id") and str(tool_result.get("tool", "")).startswith("train_"):
            st.session_state.selected_model_id = str(tool_result["model_id"])
        answer = tool_result_to_korean(tool_result)
        append_chat_message(
            "assistant",
            answer,
            title="처리 결과",
            trace="모델 backend와 무관하게 내부 tool router가 요청을 처리했습니다.",
            tool_result=tool_result,
            result_id=tool_result.get("result_id") if tool_result.get("ok") else None,
            expanded=True,
        )
        persist_ui_state()
        return

    if str(runtime.get("backend") or "").lower() == "ollama" and should_use_tool_agent(prompt):
        try:
            agent_result = ollama_tool_agent(runtime, prompt)
            append_chat_message(
                "assistant",
                agent_result.get("content") or "요청하신 분석을 완료하지 못했습니다. 입력 데이터와 모델 설정을 확인해주세요.",
                title="분석 결과",
                reasoning=agent_result.get("reasoning") or "",
                trace=agent_result.get("trace") or "",
                tool_results=agent_result.get("tool_results") or [],
                result_id=st.session_state.latest_result_id,
                expanded=True,
            )
        except Exception as exc:
            append_chat_message("assistant", f"요청 처리 중 문제가 발생했습니다.\n\n- error: `{exc}`", title="분석 실패", expanded=True)
        return

    if str(runtime.get("backend") or "").lower() == "ollama":
        messages = llm_messages_from_history()
        stream_llm_answer_to_chat(runtime, messages, title="응답", trace="일반 대화 응답입니다.")
        return

    messages = llm_messages_from_history()
    stream_llm_answer_to_chat(
        runtime,
        messages,
        title="응답",
        trace="일반 질의로 판정되어 선택된 local LLM endpoint에 chat completion을 요청했습니다.",
    )


def render_model_management_panel() -> None:
    with st.expander("Model Management", expanded=False):
        current_task = st.session_state.selected_task if st.session_state.selected_task in TASKS else TASKS[0]
        task = st.radio(
            "Task",
            TASKS,
            index=TASKS.index(current_task),
            horizontal=True,
        )
        st.session_state.selected_task = task
        persist_ui_state()
        rows = list_artifacts(task, MODEL_ROOT)
        if not rows:
            st.info(f"{task} 저장 모델이 없습니다.")
            return

        st.caption("`Apply`는 이후 채팅 요청에 사용할 task/model을 지정합니다. `X`는 선택한 저장 모델만 삭제합니다.")
        header = st.columns([0.26, 0.22, 0.14, 0.14, 0.12, 0.12])
        header[0].markdown("**Model ID**")
        header[1].markdown("**Path**")
        header[2].markdown("**Model**")
        header[3].markdown("**Metrics**")
        header[4].markdown("**Apply**")
        header[5].markdown("**Delete**")
        for row in rows:
            model_id = str(row["model_id"])
            cols = st.columns([0.26, 0.22, 0.14, 0.14, 0.12, 0.12], vertical_alignment="center")
            selected = task == st.session_state.selected_task and model_id == st.session_state.selected_model_id
            cols[0].markdown(f"{'**' if selected else ''}`{model_id}`{'**' if selected else ''}")
            cols[1].caption(str(row.get("path", "")))
            cols[2].write("ok" if row.get("has_model") else "missing")
            cols[3].write("ok" if row.get("has_metrics") else "missing")
            if cols[4].button("Apply", key=f"apply_model_{task}_{model_id}", use_container_width=True):
                st.session_state.selected_task = task
                st.session_state.selected_model_id = model_id
                append_chat_message(
                    "assistant",
                    f"현재 대화에 사용할 모델을 `{task}` / `{model_id}`로 설정했습니다.",
                    title="Model Applied",
                    trace="Model Management에서 선택한 저장 모델을 selected_task/selected_model_id에 반영했습니다.",
                    expanded=True,
                )
                st.rerun()
            if model_id == "latest":
                cols[5].caption("alias")
            elif cols[5].button("X", key=f"delete_model_{task}_{model_id}", use_container_width=True):
                try:
                    delete_artifact(task, model_id, MODEL_ROOT)
                    if st.session_state.selected_model_id == model_id:
                        ids = artifact_ids(task)
                        st.session_state.selected_model_id = ids[0] if ids else "latest"
                    persist_ui_state()
                    st.rerun()
                except Exception as exc:
                    st.error(f"삭제 실패: {exc}")

        available_ids = [str(row.get("model_id")) for row in rows]
        selected_id = str(st.session_state.selected_model_id or "latest")
        if selected_id not in available_ids:
            selected_id = available_ids[0]
            st.session_state.selected_model_id = selected_id
        with st.expander(f"Selected model detail: {task} / {selected_id}", expanded=False):
            try:
                artifact = load_artifact(task, selected_id, MODEL_ROOT)
                st.write("Metrics")
                st.json(artifact["metrics"], expanded=False)
                st.write("Schema")
                st.json(artifact["schema"], expanded=False)
            except Exception as exc:
                st.warning(f"저장 모델 상세 로드 실패: {exc}")


def render_dataset_selector() -> None:
    dataset_options = list(st.session_state.datasets.keys())
    if not dataset_options:
        return
    current = st.session_state.active_dataset_id if st.session_state.active_dataset_id in dataset_options else dataset_options[-1]
    selected_dataset = st.selectbox(
        "Active Dataset",
        dataset_options,
        index=dataset_options.index(current),
        format_func=lambda key: f"{key} · {st.session_state.datasets[key].filename}",
    )
    st.session_state.active_dataset_id = selected_dataset
    persist_ui_state()


def delete_dataset_record(dataset_id: str) -> None:
    datasets = st.session_state.datasets if isinstance(st.session_state.datasets, dict) else {}
    datasets.pop(dataset_id, None)
    completed = st.session_state.workflow_completed_dataset_ids
    if not isinstance(completed, set):
        completed = set(completed)
        st.session_state.workflow_completed_dataset_ids = completed
    completed.discard(dataset_id)
    if st.session_state.pending_workflow_dataset_id == dataset_id:
        st.session_state.pending_workflow_dataset_id = None
        st.session_state.pending_workflow_source_message = None
        st.session_state.pending_workflow_force = False
        clear_pending_workflow_runtime_state()
    if st.session_state.active_dataset_id == dataset_id:
        remaining = list(datasets.keys())
        st.session_state.active_dataset_id = remaining[-1] if remaining else None
    st.session_state.processed_upload_fingerprint = None
    try:
        path = PERSISTED_DATASETS_DIR / f"{_safe_state_filename(dataset_id)}.pkl"
        if path.exists():
            path.unlink()
    except Exception:
        pass
    persist_ui_state()


def render_dataset_management_panel() -> None:
    with st.expander("Training Dataset Management", expanded=False):
        datasets = st.session_state.datasets if isinstance(st.session_state.datasets, dict) else {}
        active_id = st.session_state.active_dataset_id
        if active_id and active_id in datasets:
            record = datasets[active_id]
            st.markdown(
                (
                    '<div class="active-dataset-banner">'
                    f'Active training dataset: <code>{html.escape(active_id)}</code> · {html.escape(str(getattr(record, "filename", "")))}'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="active-dataset-banner">Active training dataset: 없음</div>', unsafe_allow_html=True)
        if not datasets:
            st.info("등록된 학습용 dataset이 없습니다.")
            return
        st.caption("`Apply`는 활성 학습용 dataset을 바꾸고, `X`는 세션/저장 상태에서 해당 dataset을 제거합니다.")
        header = st.columns([0.25, 0.25, 0.16, 0.12, 0.11, 0.11])
        header[0].markdown("**Dataset ID**")
        header[1].markdown("**File**")
        header[2].markdown("**Type**")
        header[3].markdown("**Shape**")
        header[4].markdown("**Apply**")
        header[5].markdown("**Delete**")
        for dataset_id, record in list(datasets.items()):
            cols = st.columns([0.25, 0.25, 0.16, 0.12, 0.11, 0.11], vertical_alignment="center")
            selected = dataset_id == st.session_state.active_dataset_id
            cols[0].markdown(f"{'**' if selected else ''}`{dataset_id}`{'**' if selected else ''}")
            cols[1].caption(str(getattr(record, "filename", "")))
            cols[2].write(str(getattr(record, "dataset_type", getattr(record, "inferred_type", ""))))
            dataframe = getattr(record, "dataframe", None)
            shape = getattr(dataframe, "shape", None)
            cols[3].write(f"{shape[0]} x {shape[1]}" if shape else "-")
            if cols[4].button("Apply", key=f"apply_dataset_{dataset_id}", use_container_width=True):
                st.session_state.active_dataset_id = dataset_id
                persist_ui_state()
                st.rerun()
            if cols[5].button("X", key=f"delete_dataset_{dataset_id}", use_container_width=True):
                delete_dataset_record(dataset_id)
                st.rerun()

        if active_id and active_id in datasets:
            record = datasets[active_id]
            with st.expander(f"Active training dataset detail: {active_id}", expanded=False):
                st.write(
                    {
                        "filename": getattr(record, "filename", ""),
                        "dataset_type": getattr(record, "dataset_type", ""),
                        "uploaded_at": getattr(record, "uploaded_at", ""),
                    }
                )
                dataframe = getattr(record, "dataframe", None)
                if isinstance(dataframe, pd.DataFrame):
                    st.dataframe(dataframe.head(20), use_container_width=True)


def render_app_header(app_title: str) -> None:
    hd_logo = image_data_uri(hd_electric_logo_path())
    aim_logo = image_data_uri(AIM4LAB_LOGO_PATH)
    hd_img = f'<img class="app-hero-logo" src="{hd_logo}" alt="HD Hyundai Electric" />' if hd_logo else ""
    aim_img = f'<img class="app-aim-logo" src="{aim_logo}" alt="Aim4" /><strong class="app-aim-lab">Lab</strong>' if aim_logo else '<strong class="app-aim-lab">Aim4Lab</strong>'
    st.markdown(
        (
            '<section class="app-hero">'
            f'<div class="app-hero-logo-row">{hd_img}</div>'
            '<div class="app-title-row">'
            f"<h1>{html.escape(app_title)}</h1>"
            f'<div class="app-byline"><span>by</span>{aim_img}</div>'
            "</div>"
            '<p class="app-hero-caption">데이터 확인, 모델 학습, 성능 평가, 예측 결과 정리를 한 화면에서 진행합니다.</p>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    app_title = "형상/시험 변수 기반 고압차단기 성능 예측"
    st.set_page_config(page_title=app_title, page_icon=app_icon_value(), layout="wide")
    init_state()
    render_css()
    runtime = render_sidebar()
    render_app_header(app_title)
    st.caption("LLM 연결이 실패해도 학습용 Excel 업로드, 모델 학습/예측, 저장 모델 예측 기능은 계속 사용할 수 있습니다.")

    if st.session_state.get("upload_panel_collapse_requested"):
        st.session_state.upload_panel_expanded = False
        st.session_state.upload_panel_collapse_requested = False
    with st.expander("학습용 Excel 데이터셋 업로드 / 관리", expanded=bool(st.session_state.get("upload_panel_expanded", True))):
        st.markdown('<div class="training-upload-title">학습용 Excel 데이터셋 업로드</div>', unsafe_allow_html=True)
        st.markdown('<div class="training-upload-box">', unsafe_allow_html=True)
        uploaded = st.file_uploader("학습용 Excel 파일 선택", type=["xlsx", "xls"])
        st.markdown("</div>", unsafe_allow_html=True)
        render_dataset_management_panel()
        render_model_management_panel()
        if uploaded is not None:
            register_uploaded_dataset(uploaded)
        elif active_dataset() is not None and st.session_state.pending_workflow_dataset_id is None:
            completed = st.session_state.workflow_completed_dataset_ids
            if not isinstance(completed, set):
                completed = set(completed)
                st.session_state.workflow_completed_dataset_ids = completed
            record = active_dataset()
            if record is not None and record.dataset_id not in completed:
                st.session_state.pending_workflow_dataset_id = record.dataset_id
                st.session_state.pending_workflow_source_message = None
                st.session_state.pending_workflow_force = False
        if active_dataset() and st.button("현재 학습용 dataset workflow 다시 실행", use_container_width=True):
            append_chat_message("user", "현재 활성 학습용 dataset workflow를 다시 실행합니다.", title="학습 workflow 재실행", expanded=True)
            record = active_dataset()
            if record is not None:
                st.session_state.pending_workflow_dataset_id = record.dataset_id
                st.session_state.pending_workflow_source_message = None
                st.session_state.pending_workflow_force = True
                st.rerun()

    if not st.session_state.chat_messages:
        ensure_intro_message(runtime)
        st.rerun()

    render_chat_history()
    render_pending_workflow(runtime)
    render_prediction_input_panel()
    st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)
    scroll_to_chat_bottom()
    if st.session_state.confirm_chat_clear:
        st.warning("현재 채팅 내용과 예측 결과 표시를 삭제합니다. 계속할까요?")
        confirm_col, cancel_col, _ = st.columns([0.13, 0.13, 0.74])
        if confirm_col.button("Clear now", type="primary", use_container_width=True):
            st.session_state.confirm_chat_clear = False
            clear_persistent_chat_state()
            st.rerun()
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.confirm_chat_clear = False
            st.rerun()
    chat_input_col, clear_input_col = st.columns([0.91, 0.09], vertical_alignment="bottom")
    with chat_input_col:
        chat_value = st.chat_input(
            "질문, 예측 요청, 또는 예측용 Excel 파일을 입력하세요. 모델 성능 비교와 성능 분석 자료 생성도 요청할 수 있습니다.",
            accept_file=True,
            file_type=["xlsx", "xls"],
            key="main_chat_input",
        )
    with clear_input_col:
        st.markdown('<span class="chat-clear-anchor"></span>', unsafe_allow_html=True)
        if st.button("Chat Clear", key="chat_clear_bottom", help="현재 채팅을 삭제합니다.", use_container_width=True):
            st.session_state.confirm_chat_clear = True
            st.rerun()
    prompt, chat_files = split_chat_input(chat_value)
    if chat_files:
        with chat_message("user"):
            st.markdown((prompt.strip() or "첨부한 Excel 파일로 예측해주세요.") + "\n\n" + ", ".join(f"`{getattr(file, 'name', 'uploaded.xlsx')}`" for file in chat_files))
        handle_chat_uploads(prompt, chat_files)
        persist_ui_state()
        st.rerun()
    if prompt:
        with chat_message("user"):
            st.markdown(prompt)
        with chat_message("assistant"):
            handle_chat_prompt(prompt, runtime)
        persist_ui_state()
        st.rerun()


if __name__ == "__main__":
    main()
