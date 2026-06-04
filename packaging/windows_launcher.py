from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ENV_NAME = "local_customgui_windows"
PORT = "8791"


def say(message: str = "") -> None:
    print(message, flush=True)


def banner(title: str) -> None:
    line = "=" * 72
    say(line)
    say(title)
    say(line)


def wait_for_server(url: str, *, timeout_sec: int = 45) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    return False


def exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_project_root() -> Path:
    env_root = os.environ.get("LOCAL_CUSTOMGUI_PROJECT_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    cwd = Path.cwd()
    here = exe_dir()
    candidates.extend([cwd, here, here.parent, here.parent.parent])
    for candidate in candidates:
        if (candidate / "streamlit_app.py").exists():
            return candidate.resolve()
    raise RuntimeError(
        "streamlit_app.py를 찾지 못했습니다. EXE를 프로젝트 폴더 또는 dist 폴더에서 실행하거나 "
        "LOCAL_CUSTOMGUI_PROJECT_ROOT 환경변수에 프로젝트 폴더를 지정하세요."
    )


def find_conda_exe() -> Path:
    candidates: list[Path] = []
    for env_key in ("CONDA_EXE",):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value))
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.extend(
            [
                Path(user_profile) / "miniconda3" / "Scripts" / "conda.exe",
                Path(user_profile) / "anaconda3" / "Scripts" / "conda.exe",
            ]
        )
    for env_key, suffix in (
        ("LOCALAPPDATA", ("miniconda3", "Scripts", "conda.exe")),
        ("PROGRAMDATA", ("miniconda3", "Scripts", "conda.exe")),
        ("PROGRAMFILES", ("Miniconda3", "Scripts", "conda.exe")),
    ):
        base = os.environ.get(env_key)
        if base:
            candidates.append(Path(base).joinpath(*suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError("conda.exe를 찾지 못했습니다. Miniconda 설치를 먼저 완료하세요.")


def main() -> int:
    banner("LocalCustomGUI Windows Launcher")
    try:
        project_root = find_project_root()
        conda_exe = find_conda_exe()
    except Exception as exc:
        say(f"[ERROR] {exc}")
        input("Enter를 누르면 종료합니다...")
        return 1

    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_SHOW_EMAIL_PROMPT"] = "false"

    url = f"http://127.0.0.1:{PORT}"
    cmd = [
        str(conda_exe),
        "run",
        "-n",
        ENV_NAME,
        "python",
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        PORT,
        "--server.headless",
        "true",
        "--server.showEmailPrompt",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    say(f"[INFO] Project: {project_root}")
    say(f"[INFO] Conda: {conda_exe}")
    say(f"[RUNNING] Streamlit 서버를 시작합니다: {url}")
    proc = subprocess.Popen(cmd, cwd=str(project_root), env=env)
    if wait_for_server(url):
        say(f"[DONE] 서버 실행중: {url}")
    else:
        say(f"[WAIT] 서버 시작 확인이 늦어지고 있습니다. 잠시 뒤 브라우저에서 {url}을 확인하세요.")
    webbrowser.open(url)
    say("[INFO] 이 창을 닫으면 서버가 종료될 수 있습니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        say("")
        say("[STOPPING] 서버를 종료합니다...")
        proc.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
