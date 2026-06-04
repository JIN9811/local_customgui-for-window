from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ENV_NAME = "local_customgui_windows"
OLLAMA_MODEL = "gemma4:e2b"
PORT = "8791"


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
        if (candidate / "streamlit_app.py").exists() and (candidate / "requirements.txt").exists():
            return candidate.resolve()
    raise RuntimeError(
        "streamlit_app.py를 찾지 못했습니다. Setup EXE를 프로젝트 폴더 또는 dist 폴더에서 실행하거나 "
        "LOCAL_CUSTOMGUI_PROJECT_ROOT 환경변수에 프로젝트 폴더를 지정하세요."
    )


def find_conda_exe() -> Path | None:
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
    found = shutil.which("conda")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def find_ollama_exe() -> Path | None:
    candidates: list[Path] = []
    found = shutil.which("ollama")
    if found:
        candidates.append(Path(found))
    for env_key, suffix in (
        ("LOCALAPPDATA", ("Programs", "Ollama", "ollama.exe")),
        ("PROGRAMFILES", ("Ollama", "ollama.exe")),
    ):
        base = os.environ.get(env_key)
        if base:
            candidates.append(Path(base).joinpath(*suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def run_checked(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("")
    print("> " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env)
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"명령 실행 실패(returncode={return_code}): {' '.join(cmd)}")


def run_quiet(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return int(proc.returncode)


def install_winget(package_id: str) -> None:
    winget = shutil.which("winget")
    if not winget:
        raise RuntimeError("winget을 찾지 못했습니다. Miniconda/Ollama를 수동 설치한 뒤 다시 실행하세요.")
    run_checked(
        [
            winget,
            "install",
            "-e",
            "--id",
            package_id,
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    )


def ensure_conda() -> Path:
    conda_exe = find_conda_exe()
    if conda_exe:
        return conda_exe
    print("Miniconda가 없어 winget으로 설치를 시도합니다.")
    install_winget("Anaconda.Miniconda3")
    for _ in range(20):
        conda_exe = find_conda_exe()
        if conda_exe:
            return conda_exe
        time.sleep(3)
    raise RuntimeError("Miniconda 설치 후에도 conda.exe를 찾지 못했습니다. 새 PowerShell에서 다시 실행하세요.")


def ensure_ollama() -> Path | None:
    ollama_exe = find_ollama_exe()
    if ollama_exe:
        return ollama_exe
    print("Ollama가 없어 winget으로 설치를 시도합니다.")
    try:
        install_winget("Ollama.Ollama")
    except Exception as exc:
        print(f"[WARN] Ollama 자동 설치 실패: {exc}")
        return None
    for _ in range(20):
        ollama_exe = find_ollama_exe()
        if ollama_exe:
            return ollama_exe
        time.sleep(3)
    print("[WARN] Ollama 설치 후 실행 파일을 찾지 못했습니다. LLM 기능은 Ollama 설치 후 사용할 수 있습니다.")
    return None


def ensure_conda_env(conda_exe: Path) -> None:
    if run_quiet([str(conda_exe), "run", "-n", ENV_NAME, "python", "--version"]) == 0:
        print(f"Conda env exists: {ENV_NAME}")
        return
    run_checked([str(conda_exe), "create", "-y", "-n", ENV_NAME, "--override-channels", "-c", "conda-forge", "python=3.11", "pip"])


def install_python_deps(conda_exe: Path, project_root: Path) -> None:
    base = [str(conda_exe), "run", "-n", ENV_NAME, "python", "-m"]
    run_checked([*base, "pip", "install", "-U", "pip"], cwd=project_root)
    run_checked([*base, "pip", "install", "-r", "requirements.txt"], cwd=project_root)
    run_checked([*base, "pip", "install", "-r", "requirements-pycaret.txt"], cwd=project_root)
    run_checked([*base, "pip", "install", "-e", "."], cwd=project_root)


def ensure_env_file(project_root: Path) -> None:
    env_file = project_root / ".env"
    example = project_root / ".env.example"
    if not env_file.exists() and example.exists():
        env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_streamlit_config() -> None:
    user_profile = os.environ.get("USERPROFILE")
    if not user_profile:
        return
    config_dir = Path(user_profile) / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        "[server]\n"
        "headless = true\n"
        "showEmailPrompt = false\n\n"
        "[browser]\n"
        "gatherUsageStats = false\n",
        encoding="utf-8",
    )


def ensure_ollama_model(ollama_exe: Path | None) -> None:
    if not ollama_exe:
        print("[WARN] Ollama 실행 파일이 없어 모델 다운로드를 건너뜁니다.")
        return
    if run_quiet([str(ollama_exe), "list"]) != 0:
        subprocess.Popen([str(ollama_exe), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    run_checked([str(ollama_exe), "pull", OLLAMA_MODEL])


def launch_streamlit(conda_exe: Path, project_root: Path) -> int:
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
    print("")
    print(f"설치 완료. Streamlit을 실행합니다: {url}")
    proc = subprocess.Popen(cmd, cwd=str(project_root), env=env)
    time.sleep(4)
    webbrowser.open(url)
    return int(proc.wait())


def main() -> int:
    parser = argparse.ArgumentParser(description="Install and launch LocalCustomGUI on Windows.")
    parser.add_argument("--dry-run", action="store_true", help="Print detected paths without installing.")
    parser.add_argument("--no-launch", action="store_true", help="Install only; do not launch Streamlit.")
    parser.add_argument("--skip-model", action="store_true", help="Skip ollama pull.")
    args = parser.parse_args()

    try:
        project_root = find_project_root()
        print(f"Project: {project_root}")
        conda_exe = find_conda_exe()
        ollama_exe = find_ollama_exe()
        print(f"Conda: {conda_exe or 'not found'}")
        print(f"Ollama: {ollama_exe or 'not found'}")
        if args.dry_run:
            return 0

        conda_exe = ensure_conda()
        ollama_exe = ensure_ollama()
        ensure_conda_env(conda_exe)
        install_python_deps(conda_exe, project_root)
        ensure_env_file(project_root)
        ensure_streamlit_config()
        if not args.skip_model:
            ensure_ollama_model(ollama_exe)
        if args.no_launch:
            print("설치 완료. --no-launch 옵션으로 앱 실행은 건너뜁니다.")
            return 0
        return launch_streamlit(conda_exe, project_root)
    except Exception as exc:
        print("")
        print(f"[ERROR] {exc}")
        input("Enter를 누르면 종료합니다...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
