"""Model artifact persistence helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .constants import MODEL_ROOT


def timestamp_id() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def task_model_root(task: str, model_root: Path | None = None) -> Path:
    return (model_root or MODEL_ROOT) / task


def model_dir(task: str, model_id: str = "latest", model_root: Path | None = None) -> Path:
    return task_model_root(task, model_root) / model_id


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_card(path: Path, *, task: str, source_filename: str, schema: dict[str, Any], metrics: dict[str, Any]) -> None:
    holdout = metrics.get("holdout", {})
    metric_lines = "\n".join(f"- {key}: {value}" for key, value in holdout.items())
    domain = str(schema.get("domain") or metrics.get("domain") or "circuit_breaker")
    display_name = str(schema.get("display_name") or metrics.get("display_name") or "형상/시험 변수 기반 고압차단기 성능 예측")
    design_names = schema.get("design_model_names") or metrics.get("design_model_names") or []
    design_text = ", ".join(map(str, design_names)) if isinstance(design_names, list) else str(design_names or "")
    if domain == "motor_noise":
        purpose = "HD현대일렉트릭 회전기 설계 변수 기반 소음 예측 회귀 모델 artifact입니다."
        regression_rule = "- Regression은 `sound_mean`을 target으로 사용하고, 4방향 sound 측정값은 입력 변수에서 제외합니다."
    else:
        purpose = f"HD현대일렉트릭 고압차단기 수치형 형상/시험 변수 기반 `{task}` 모델 artifact입니다."
        regression_rule = "- Regression은 `TRVmax[kV]`를 target으로 사용."
    content = f"""# Model Card - {task}

## 목적
{purpose}

## 데이터
- model family: `{display_name}`
- Excel 설계모델명: `{design_text or "N/A"}`
- source file: `{source_filename}`
- feature count: {len(schema.get("features", []))}
- target: `{schema.get("target")}`

## Preprocessing
- Notebook 기준 `dropna()` 수행.
- Classification은 `Result == 1.0 -> label 1`, 그 외 `0`.
{regression_rule}
- Ignored columns: {schema.get("ignored_columns", [])}

## Holdout Metrics
{metric_lines}

## 제한사항
- 데이터 버전과 preprocessing에 따라 성능이 달라질 수 있습니다.
- LLM은 예측값을 임의 생성하지 않고 저장된 model artifact와 deterministic tool 결과만 설명해야 합니다.
"""
    path.write_text(content, encoding="utf-8")


def publish_latest(source_dir: Path, latest_dir: Path) -> None:
    if latest_dir.exists():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    try:
        latest_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source_dir, latest_dir)


def _safe_artifact_path(task: str, model_id: str, model_root: Path | None = None) -> Path:
    if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
        raise ValueError(f"Unsafe model_id: {model_id}")
    root = task_model_root(task, model_root).resolve()
    return root / model_id


def _republish_latest(task: str, model_root: Path | None = None) -> str | None:
    root = task_model_root(task, model_root)
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    candidates = [
        path
        for path in sorted(root.iterdir(), reverse=True)
        if path.name != "latest" and path.is_dir() and (path / "model.joblib").exists()
    ] if root.exists() else []
    if not candidates:
        return None
    publish_latest(candidates[0], latest)
    return candidates[0].name


def delete_artifact(task: str, model_id: str, model_root: Path | None = None) -> dict[str, Any]:
    """Delete one concrete model artifact under models/<task>.

    The `latest` alias is intentionally not deletable here because it may be a
    symlink or copied alias rather than a concrete model run.
    """
    if model_id == "latest":
        raise ValueError("`latest` is an alias. Delete a concrete timestamp model instead.")
    root = task_model_root(task, model_root).resolve()
    path = _safe_artifact_path(task, model_id, model_root)
    if not root.exists():
        raise FileNotFoundError(f"Model root not found: {root}")
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    latest = root / "latest"
    target_resolved = path.resolve() if path.exists() else path
    latest_pointed_to_deleted = latest.is_symlink() and latest.resolve() == target_resolved

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        resolved = path.resolve()
        if resolved != path and root not in resolved.parents:
            raise ValueError(f"Refusing to delete artifact outside model root: {resolved}")
        shutil.rmtree(path)
    else:
        path.unlink()

    latest_model_id = None
    if latest_pointed_to_deleted or latest.exists() or latest.is_symlink():
        latest_model_id = _republish_latest(task, model_root)
    return {"ok": True, "task": task, "deleted_model_id": model_id, "latest_model_id": latest_model_id}


def save_artifact(
    *,
    task: str,
    estimator: Any,
    schema: dict[str, Any],
    metrics: dict[str, Any],
    source_filename: str,
    model_root: Path | None = None,
) -> dict[str, Any]:
    root = task_model_root(task, model_root)
    artifact_id = timestamp_id()
    out_dir = root / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, out_dir / "model.joblib")
    save_json(out_dir / "schema.json", schema)
    save_json(out_dir / "metrics.json", metrics)
    write_model_card(out_dir / "model_card.md", task=task, source_filename=source_filename, schema=schema, metrics=metrics)
    publish_latest(out_dir, root / "latest")
    return {
        "ok": True,
        "task": task,
        "model_id": artifact_id,
        "artifact_dir": str(out_dir),
        "latest_dir": str(root / "latest"),
        "model_path": str(out_dir / "model.joblib"),
        "schema_path": str(out_dir / "schema.json"),
        "metrics_path": str(out_dir / "metrics.json"),
        "model_card_path": str(out_dir / "model_card.md"),
    }


def load_artifact(task: str, model_id: str = "latest", model_root: Path | None = None) -> dict[str, Any]:
    directory = model_dir(task, model_id, model_root).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Model artifact not found: {directory}")
    return {
        "task": task,
        "model_id": model_id,
        "directory": directory,
        "model": joblib.load(directory / "model.joblib"),
        "schema": load_json(directory / "schema.json"),
        "metrics": load_json(directory / "metrics.json"),
        "model_card": (directory / "model_card.md").read_text(encoding="utf-8"),
    }


def list_artifacts(task: str, model_root: Path | None = None) -> list[dict[str, Any]]:
    root = task_model_root(task, model_root)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    concrete_resolved: set[Path] = set()
    for path in root.iterdir():
        if path.name == "latest" or not path.is_dir():
            continue
        if (path / "model.joblib").exists():
            try:
                concrete_resolved.add(path.resolve())
            except OSError:
                concrete_resolved.add(path)
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir() and not path.is_symlink():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        is_latest_alias = path.name == "latest"
        if is_latest_alias and resolved in concrete_resolved:
            continue
        metrics_path = path / "metrics.json"
        schema_path = path / "schema.json"
        rows.append(
            {
                "model_id": path.name,
                "path": str(path),
                "is_latest_alias": is_latest_alias,
                "resolved_model_id": resolved.name if is_latest_alias else path.name,
                "has_model": (path / "model.joblib").exists(),
                "has_metrics": metrics_path.exists(),
                "has_schema": schema_path.exists(),
            }
        )
    return rows
