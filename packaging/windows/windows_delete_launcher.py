from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - bundled exe includes psutil, source mode may not.
    psutil = None


ENV_NAME = "local_customgui_windows"
OLLAMA_MODEL_E2B = "gemma4:e2b"
OLLAMA_MODEL_E4B = "gemma4:e4b"
OLLAMA_MODELS = (OLLAMA_MODEL_E2B, OLLAMA_MODEL_E4B)
OLLAMA_MODEL = OLLAMA_MODEL_E4B
CONDA_TOS_CHANNELS = (
    "https://repo.anaconda.com/pkgs/main",
    "https://repo.anaconda.com/pkgs/r",
    "https://repo.anaconda.com/pkgs/msys2",
)


def say(message: str = "") -> None:
    print(message, flush=True)


def banner(title: str) -> None:
    line = "=" * 72
    say(line)
    say(title)
    say(line)


def exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_project_root() -> Path | None:
    env_root = os.environ.get("LOCAL_CUSTOMGUI_PROJECT_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    cwd = Path.cwd()
    here = exe_dir()
    candidates.extend([cwd, here, here.parent, here.parent.parent])
    for candidate in candidates:
        if (candidate / "streamlit_app.py").exists() and (candidate / "README.md").exists():
            return candidate.resolve()
    return None


def find_conda_exe() -> Path | None:
    candidates: list[Path] = []
    value = os.environ.get("CONDA_EXE")
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


def find_powershell_exe() -> str:
    found = shutil.which("powershell.exe") or shutil.which("powershell")
    if found:
        return found
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidate = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if candidate.exists():
            return str(candidate)
    return "powershell.exe"


def run_quiet(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return int(proc.returncode), proc.stdout


def run_checked(cmd: list[str]) -> None:
    say("")
    say("[RUNNING] " + " ".join(cmd))
    proc = subprocess.Popen(cmd)
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed(returncode={return_code}): {' '.join(cmd)}")
    say("[DONE] Command completed")


def local_streamlit_processes(project_root: Path | None = None) -> list[object]:
    if psutil is None:
        return []
    root_text = ""
    if project_root:
        try:
            root_text = str(project_root.resolve()).lower()
        except OSError:
            root_text = str(project_root).lower()
    matches: list[object] = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "cwd"]):
        try:
            info = proc.as_dict(attrs=["pid", "name", "exe", "cmdline", "cwd"], ad_value="")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if info.get("pid") == current_pid:
            continue
        process_name = str(info.get("name") or "").lower()
        runtime_process = process_name in {"python.exe", "pythonw.exe", "conda.exe", "streamlit.exe"}
        if not runtime_process:
            continue
        cmdline = info.get("cmdline") or []
        text = " ".join(
            [
                str(info.get("name") or ""),
                str(info.get("exe") or ""),
                str(info.get("cwd") or ""),
                " ".join(str(part) for part in cmdline),
            ]
        ).lower()
        app_match = "streamlit_app.py" in text and (not root_text or root_text in text)
        env_match = ENV_NAME.lower() in text and "streamlit" in text
        if app_match or env_match:
            matches.append(proc)
    return matches


def stop_streamlit_processes_with_powershell(project_root: Path | None = None) -> None:
    env = os.environ.copy()
    env["LOCAL_CUSTOMGUI_PROCESS_ROOT"] = str(project_root or "")
    env["LOCAL_CUSTOMGUI_ENV_NAME"] = ENV_NAME
    script = r"""
$root = [string]$env:LOCAL_CUSTOMGUI_PROCESS_ROOT
$envName = [string]$env:LOCAL_CUSTOMGUI_ENV_NAME
$root = $root.ToLowerInvariant()
$envName = $envName.ToLowerInvariant()
$matched = 0
Get-CimInstance Win32_Process | ForEach-Object {
    $name = ([string]$_.Name).ToLowerInvariant()
    $runtimeProcess = @("python.exe", "pythonw.exe", "conda.exe", "streamlit.exe") -contains $name
    $cmd = [string]$_.CommandLine
    $exe = [string]$_.ExecutablePath
    $text = ($cmd + " " + $exe).ToLowerInvariant()
    $appMatch = $text.Contains("streamlit_app.py") -and (($root.Length -eq 0) -or $text.Contains($root))
    $envMatch = ($envName.Length -gt 0) -and $text.Contains($envName) -and $text.Contains("streamlit")
    if ($runtimeProcess -and ($appMatch -or $envMatch) -and ($_.ProcessId -ne $PID)) {
        $matched += 1
        Write-Output ("[STOPPING] Background Streamlit process PID " + $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
if ($matched -eq 0) {
    Write-Output "[OK] No background Streamlit server was found."
} else {
    Write-Output "[DONE] Background Streamlit cleanup completed."
}
"""
    proc = subprocess.run(
        [find_powershell_exe(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    output = proc.stdout.strip()
    if output:
        for line in output.splitlines():
            say(line.rstrip())
    if proc.returncode != 0:
        say("[WARN] PowerShell Streamlit cleanup did not complete cleanly.")


def stop_external_streamlit_processes(project_root: Path | None = None) -> None:
    if psutil is None:
        stop_streamlit_processes_with_powershell(project_root)
        return
    procs = local_streamlit_processes(project_root)
    if not procs:
        say("[OK] No background Streamlit server was found.")
        return
    for proc in procs:
        try:
            say(f"[STOPPING] Background Streamlit process PID {proc.pid}")
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _, alive = psutil.wait_procs(procs, timeout=8)
    for proc in alive:
        try:
            say(f"[KILL] Background Streamlit process PID {proc.pid}")
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if alive:
        psutil.wait_procs(alive, timeout=5)
    say("[DONE] Background Streamlit cleanup completed.")


def conda_env_exists(conda_exe: Path | None) -> bool:
    if not conda_exe:
        return False
    code, _ = run_quiet([str(conda_exe), "run", "-n", ENV_NAME, "python", "--version"])
    return code == 0


def normalize_ollama_model(model: str | None) -> str:
    value = str(model or "").strip()
    return value if value in OLLAMA_MODELS else OLLAMA_MODEL


def ollama_model_exists(ollama_exe: Path | None, model: str | None = None) -> bool:
    if not ollama_exe:
        return False
    code, output = run_quiet([str(ollama_exe), "list"])
    if code != 0:
        return False
    model_name = normalize_ollama_model(model).lower()
    for line in output.splitlines()[1:]:
        columns = line.split()
        if columns and columns[0].lower() == model_name:
            return True
    return False


def existing_ollama_models(ollama_exe: Path | None) -> list[str]:
    return [model for model in OLLAMA_MODELS if ollama_model_exists(ollama_exe, model)]


def stop_ollama_model(ollama_exe: Path | None, model: str) -> None:
    if not ollama_exe:
        return
    model = normalize_ollama_model(model)
    code, _ = run_quiet([str(ollama_exe), "stop", model])
    if code == 0:
        say(f"[DONE] Ollama model stopped: {model}")


def accept_conda_tos(conda_exe: Path) -> None:
    for channel in CONDA_TOS_CHANNELS:
        code, output = run_quiet([str(conda_exe), "tos", "accept", "--override-channels", "--channel", channel])
        if code == 0:
            say(f"[OK] Conda ToS accepted: {channel}")
            continue
        lowered = output.lower()
        if "invalid choice" in lowered or "no such command" in lowered:
            say("[INFO] Conda ToS command is unavailable; skipping.")
            return
        if "already accepted" in lowered:
            say(f"[OK] Conda ToS already accepted: {channel}")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def make_writable(func, path: str, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_path(path: Path, allowed_root: Path) -> None:
    if not path.exists() and not path.is_symlink():
        say(f"[SKIP] Missing: {path}")
        return
    if path.resolve() == allowed_root.resolve() or not is_under(path, allowed_root):
        raise RuntimeError(f"Refusing to delete outside allowed root: {path}")
    say(f"[DELETE] {path}")
    if path.is_symlink():
        path.unlink()
        return
    if path.is_file():
        path.chmod(stat.S_IWRITE)
        path.unlink()
        return
    shutil.rmtree(path, onerror=make_writable)


def runtime_targets(project_root: Path | None) -> list[Path]:
    if not project_root:
        return []
    targets = [
        project_root / "state",
        project_root / "logs.log",
        project_root / ".pytest_cache",
        project_root / "__pycache__",
    ]
    return [target for target in targets if target.exists() or target.is_symlink()]


def env_targets(project_root: Path | None) -> list[Path]:
    if not project_root:
        return []
    target = project_root / ".env"
    return [target] if target.exists() else []


def model_targets(project_root: Path | None) -> list[Path]:
    if not project_root:
        return []
    targets: list[Path] = []
    for task_name in ("classification", "regression"):
        task_dir = project_root / "models" / task_name
        if not task_dir.exists():
            continue
        for child in sorted(task_dir.iterdir()):
            targets.append(child)
    return targets


def streamlit_config_target() -> Path | None:
    user_profile = os.environ.get("USERPROFILE")
    if not user_profile:
        return None
    target = Path(user_profile) / ".streamlit" / "config.toml"
    return target if target.exists() else None


def delete_conda_env(conda_exe: Path | None) -> None:
    if not conda_exe:
        say("[SKIP] conda.exe was not found.")
        return
    if not conda_env_exists(conda_exe):
        say(f"[SKIP] Conda env is missing: {ENV_NAME}")
        return
    stop_external_streamlit_processes(find_project_root())
    accept_conda_tos(conda_exe)
    run_checked([str(conda_exe), "env", "remove", "-y", "-n", ENV_NAME])


def delete_ollama_models(ollama_exe: Path | None) -> None:
    if not ollama_exe:
        say("[SKIP] ollama.exe was not found.")
        return
    models = existing_ollama_models(ollama_exe)
    if not models:
        say(f"[SKIP] Ollama models are missing or Ollama is not running: {' / '.join(OLLAMA_MODELS)}")
        return
    for model in models:
        stop_ollama_model(ollama_exe, model)
        run_checked([str(ollama_exe), "rm", model])


def delete_project_paths(project_root: Path | None, targets: list[Path]) -> None:
    if not project_root:
        say("[SKIP] Project root was not found.")
        return
    if not targets:
        say("[SKIP] Nothing to delete.")
        return
    for target in targets:
        remove_path(target, project_root)


def delete_streamlit_config(target: Path | None) -> None:
    if not target:
        say("[SKIP] Streamlit user config was not found.")
        return
    allowed_root = target.parent
    remove_path(target, allowed_root)


def build_items(project_root: Path | None, conda_exe: Path | None, ollama_exe: Path | None) -> list[dict[str, object]]:
    runtime = runtime_targets(project_root)
    env_files = env_targets(project_root)
    models = model_targets(project_root)
    streamlit_config = streamlit_config_target()
    return [
        {
            "id": "1",
            "title": f"Conda env: {ENV_NAME}",
            "available": conda_env_exists(conda_exe),
            "preview": [f"conda env remove -n {ENV_NAME}"],
            "delete": lambda: delete_conda_env(conda_exe),
        },
        {
            "id": "2",
            "title": f"Ollama models: {' / '.join(OLLAMA_MODELS)}",
            "available": bool(existing_ollama_models(ollama_exe)),
            "preview": [f"ollama rm {model}" for model in OLLAMA_MODELS],
            "delete": lambda: delete_ollama_models(ollama_exe),
        },
        {
            "id": "3",
            "title": "Runtime state, logs, and caches",
            "available": bool(runtime),
            "preview": [str(path) for path in runtime],
            "delete": lambda: delete_project_paths(project_root, runtime),
        },
        {
            "id": "4",
            "title": "Local app config: .env",
            "available": bool(env_files),
            "preview": [str(path) for path in env_files],
            "delete": lambda: delete_project_paths(project_root, env_files),
        },
        {
            "id": "5",
            "title": "Model artifacts under models/classification and models/regression",
            "available": bool(models),
            "preview": [str(path) for path in models],
            "delete": lambda: delete_project_paths(project_root, models),
        },
        {
            "id": "6",
            "title": "Streamlit user config: %USERPROFILE%\\.streamlit\\config.toml",
            "available": streamlit_config is not None,
            "preview": [str(streamlit_config)] if streamlit_config else [],
            "delete": lambda: delete_streamlit_config(streamlit_config),
        },
    ]


def status_text(available: bool) -> str:
    return "available" if available else "missing"


def show_menu(items: list[dict[str, object]]) -> None:
    say("")
    say("Select items to delete:")
    for item in items:
        say(f"  {item['id']}. {item['title']} [{status_text(bool(item['available']))}]")
    say("")
    say("  R. Recommended uninstall cleanup (1, 2, 3, 4)")
    say("  A. All available items")
    say("  Q. Quit")
    say("")
    say("The project folder itself is never deleted automatically.")


def parse_selection(raw: str, items: list[dict[str, object]]) -> list[str]:
    normalized = raw.strip().lower()
    if not normalized or normalized in {"q", "quit", "exit"}:
        return []
    available_ids = [str(item["id"]) for item in items if bool(item["available"])]
    if normalized in {"a", "all"}:
        return available_ids
    if normalized in {"r", "recommended"}:
        recommended = {"1", "2", "3", "4"}
        return [item_id for item_id in available_ids if item_id in recommended]
    selected: list[str] = []
    known = {str(item["id"]) for item in items}
    for token in normalized.replace(" ", "").split(","):
        if not token:
            continue
        if token not in known:
            raise RuntimeError(f"Unknown selection: {token}")
        selected.append(token)
    return selected


def selected_items(items: list[dict[str, object]], ids: list[str]) -> list[dict[str, object]]:
    selected_set = set(ids)
    return [item for item in items if str(item["id"]) in selected_set and bool(item["available"])]


def show_preview(items: list[dict[str, object]]) -> None:
    say("")
    say("Delete preview:")
    for item in items:
        say(f"- {item['title']}")
        preview = list(item["preview"])  # type: ignore[arg-type]
        if not preview:
            say("    (nothing found)")
            continue
        for target in preview[:20]:
            say(f"    {target}")
        if len(preview) > 20:
            say(f"    ... and {len(preview) - 20} more")


def confirm_or_exit(yes: bool) -> bool:
    if yes:
        return True
    say("")
    say("Type DELETE to continue. Anything else will cancel.")
    return input("> ").strip() == "DELETE"


def pause_before_exit(args: argparse.Namespace) -> None:
    if args.select is None and not args.yes and not args.dry_run:
        input("Press Enter to exit...")


def main() -> int:
    banner("LocalCustomGUI Windows Delete Tool")
    parser = argparse.ArgumentParser(description="Delete selected LocalCustomGUI Windows install/runtime items.")
    parser.add_argument("--select", help="Comma-separated item ids, R/recommended, or A/all.")
    parser.add_argument("--yes", action="store_true", help="Skip DELETE confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting.")
    args = parser.parse_args()

    project_root = find_project_root()
    conda_exe = find_conda_exe()
    ollama_exe = find_ollama_exe()
    say(f"[INFO] Project: {project_root or 'not found'}")
    say(f"[INFO] Conda: {conda_exe or 'not found'}")
    say(f"[INFO] Ollama: {ollama_exe or 'not found'}")

    try:
        items = build_items(project_root, conda_exe, ollama_exe)
        show_menu(items)
        raw_selection = args.select if args.select is not None else input("Choice: ")
        ids = parse_selection(raw_selection, items)
        if not ids:
            say("[CANCELLED] Nothing selected.")
            pause_before_exit(args)
            return 0
        chosen = selected_items(items, ids)
        if not chosen:
            say("[CANCELLED] Selected items were missing.")
            pause_before_exit(args)
            return 0
        show_preview(chosen)
        if args.dry_run:
            say("")
            say("[DRY-RUN] No files or environments were deleted.")
            return 0
        if not confirm_or_exit(args.yes):
            say("[CANCELLED] Delete cancelled.")
            pause_before_exit(args)
            return 0
        for item in chosen:
            delete_func = item["delete"]
            delete_func()  # type: ignore[operator]
        say("")
        say("[DONE] Selected cleanup completed.")
        pause_before_exit(args)
        return 0
    except Exception as exc:
        say("")
        say(f"[ERROR] {exc}")
        pause_before_exit(args)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
