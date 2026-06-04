"""Subprocess bridge for notebook-parity PyCaret training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import PROJECT_ROOT

JSON_MARKER = "__HD_SERVING_JSON__"
DEFAULT_PYCARET_ENV = ""


def _detect_conda_exe() -> str | None:
    configured = os.environ.get("LOCAL_CUSTOMGUI_CONDA_EXE") or os.environ.get("CONDA_EXE")
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("conda")
    if found:
        return found
    user_profile = os.environ.get("USERPROFILE")
    candidates = []
    if user_profile:
        candidates.extend(
            [
                Path(user_profile) / "miniconda3" / "Scripts" / "conda.exe",
                Path(user_profile) / "anaconda3" / "Scripts" / "conda.exe",
            ]
        )
    candidates.extend(
        [
            Path(os.environ.get("LOCALAPPDATA", "")) / "miniconda3" / "Scripts" / "conda.exe",
            Path(os.environ.get("PROGRAMDATA", "")) / "miniconda3" / "Scripts" / "conda.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Miniconda3" / "Scripts" / "conda.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(JSON_MARKER):
            payload = line[len(JSON_MARKER) :].strip()
            return json.loads(payload)
    raise RuntimeError(f"PyCaret worker did not emit result JSON. stdout_tail={stdout[-2000:]}")


def train_pycaret_model(
    df: pd.DataFrame,
    *,
    task: str,
    source_filename: str,
    train_size: float,
    random_state: int,
    model_root: Path | None,
) -> dict[str, Any]:
    conda_env = os.environ.get("LOCAL_CUSTOMGUI_PYCARET_ENV", DEFAULT_PYCARET_ENV)
    timeout = int(os.environ.get("LOCAL_CUSTOMGUI_PYCARET_TIMEOUT_SEC", "3600"))
    root = (model_root or (PROJECT_ROOT / "models")).resolve()
    with tempfile.TemporaryDirectory(prefix="hd_pycaret_") as tmp:
        input_path = Path(tmp) / "input.xlsx"
        df.to_excel(input_path, index=False)
        worker_args = [
            "-m",
            "hd_serving.pycaret_worker",
            "train",
            "--task",
            task,
            "--input",
            str(input_path),
            "--source-filename",
            source_filename,
            "--train-size",
            str(train_size),
            "--random-state",
            str(random_state),
            "--model-root",
            str(root),
        ]
        if conda_env:
            conda_exe = _detect_conda_exe()
            if not conda_exe:
                raise RuntimeError("LOCAL_CUSTOMGUI_PYCARET_ENV is set, but conda.exe could not be found.")
            cmd = [conda_exe, "run", "-n", conda_env, "python", *worker_args]
        else:
            cmd = [sys.executable, *worker_args]
        env = os.environ.copy()
        src = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}{os.pathsep}{env['PYTHONPATH']}"
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "PyCaret worker failed "
            f"(returncode={proc.returncode}). stdout_tail={proc.stdout[-2000:]} stderr_tail={proc.stderr[-4000:]}"
        )
    result = _json_from_stdout(proc.stdout)
    result["worker_stdout_tail"] = proc.stdout[-4000:]
    if proc.stderr.strip():
        result["worker_stderr_tail"] = proc.stderr[-4000:]
    return result
