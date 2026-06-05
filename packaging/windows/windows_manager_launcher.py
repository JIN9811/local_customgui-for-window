from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox as tk_messagebox


ENV_NAME = "local_customgui_windows"
OLLAMA_MODEL_E2B = "gemma4:e2b"
OLLAMA_MODEL_E4B = "gemma4:e4b"
OLLAMA_MODELS = (OLLAMA_MODEL_E2B, OLLAMA_MODEL_E4B)
OLLAMA_MODEL = OLLAMA_MODEL_E4B
PORT = "8791"
TOTAL_INSTALL_STEPS = 9
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
GITHUB_URL = "https://github.com/JIN9811/local_customgui-for-window"
APP_DISPLAY_NAME = "AIM4LAB LocalCustomGUI"
APP_PUBLISHER = "AIM4LAB"
APP_VERSION = "0.1.0"
UNINSTALL_REGISTRY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\AIM4LAB_LocalCustomGUI"


def creation_flags() -> int:
    return CREATE_NO_WINDOW if os.name == "nt" else 0


def exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    frozen_bundle = getattr(sys, "_MEIPASS", None)
    if frozen_bundle:
        return Path(frozen_bundle)
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return bundle_dir().joinpath(*parts)


def quote_windows_arg(value: str) -> str:
    return '"' + value.replace('"', r'\"') + '"'


def normalize_ollama_model(model: str | None) -> str:
    value = str(model or "").strip()
    return value if value in OLLAMA_MODELS else OLLAMA_MODEL


def normalize_ollama_models(models: list[str] | tuple[str, ...] | None) -> list[str]:
    selected: list[str] = []
    for model in models or []:
        normalized = normalize_ollama_model(model)
        if normalized not in selected:
            selected.append(normalized)
    return selected or [recommended_ollama_model()]


def default_ollama_model_for_selection(models: list[str] | tuple[str, ...]) -> str:
    selected = normalize_ollama_models(list(models))
    recommended = recommended_ollama_model()
    if recommended in selected:
        return recommended
    if OLLAMA_MODEL_E4B in selected:
        return OLLAMA_MODEL_E4B
    return selected[0]


def installed_ram_gb() -> float | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return float(status.ullTotalPhys) / (1024**3)
        return None
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024**3)
        except (OSError, ValueError):
            return None
    return None


def recommended_ollama_model() -> str:
    ram_gb = installed_ram_gb()
    if ram_gb is not None and ram_gb < 31:
        return OLLAMA_MODEL_E2B
    return OLLAMA_MODEL_E4B


def ram_recommendation_text() -> str:
    selected = recommended_ollama_model()
    reason = "16 GB -> e2b. 32 GB-class or more -> e4b."
    return f"Recommended model: {selected}. {reason}"


def detected_ram_text() -> str:
    ram_gb = installed_ram_gb()
    if ram_gb is None:
        return "Detected RAM: unavailable"
    return f"Detected RAM: {ram_gb:.0f} GB"


def manager_exe_for_registration(project_root: Path | None) -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    if project_root:
        candidate = project_root / "LocalCustomGUI-Manager.exe"
        if candidate.exists():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def find_project_root() -> Path | None:
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
    return None


def require_project_root() -> Path:
    project_root = find_project_root()
    if project_root:
        return project_root
    raise RuntimeError(
        "Could not find streamlit_app.py. Run this EXE from the project folder or dist folder, "
        "or set LOCAL_CUSTOMGUI_PROJECT_ROOT to the project folder."
    )


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


def run_quiet(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=creation_flags(),
    )
    return int(proc.returncode), proc.stdout


def run_live(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log=print) -> None:
    log("")
    log("[RUNNING] " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed(returncode={return_code}): {' '.join(cmd)}")
    log("[DONE] Command completed")


def conda_env_exists(conda_exe: Path | None) -> bool:
    if not conda_exe:
        return False
    code, _ = run_quiet([str(conda_exe), "run", "-n", ENV_NAME, "python", "--version"])
    return code == 0


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


def wait_for_server(url: str, *, timeout_sec: int = 45) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    return False


def install_winget(package_id: str, *, log=print) -> None:
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { "
        "throw 'winget was not found. Install from Microsoft App Installer, then run this manager again.' "
        "}; "
        f"winget install -e --id '{package_id}' "
        "--accept-package-agreements --accept-source-agreements --disable-interactivity"
    )
    run_live([find_powershell_exe(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], log=log)


def ensure_conda(*, log=print) -> Path:
    conda_exe = find_conda_exe()
    if conda_exe:
        log(f"[OK] Miniconda found: {conda_exe}")
        return conda_exe
    log("[RUNNING] Miniconda was not found. Installing with winget.")
    install_winget("Anaconda.Miniconda3", log=log)
    for _ in range(20):
        conda_exe = find_conda_exe()
        if conda_exe:
            return conda_exe
        time.sleep(3)
    raise RuntimeError("Miniconda was installed, but conda.exe was not found. Reopen Windows and try again.")


def ensure_ollama(*, log=print) -> Path | None:
    ollama_exe = find_ollama_exe()
    if ollama_exe:
        log(f"[OK] Ollama found: {ollama_exe}")
        return ollama_exe
    log("[RUNNING] Ollama was not found. Installing with winget.")
    try:
        install_winget("Ollama.Ollama", log=log)
    except Exception as exc:
        log(f"[WARN] Ollama auto-install failed: {exc}")
        return None
    for _ in range(20):
        ollama_exe = find_ollama_exe()
        if ollama_exe:
            return ollama_exe
        time.sleep(3)
    log("[WARN] Ollama was installed, but ollama.exe was not found. LLM features may need manual setup.")
    return None


def ensure_conda_env(conda_exe: Path, *, log=print) -> None:
    if conda_env_exists(conda_exe):
        log(f"[OK] Conda env exists: {ENV_NAME}")
        return
    run_live(
        [
            str(conda_exe),
            "create",
            "-y",
            "-n",
            ENV_NAME,
            "--override-channels",
            "-c",
            "conda-forge",
            "python=3.11",
            "pip",
        ],
        log=log,
    )


def install_python_deps(conda_exe: Path, project_root: Path, *, log=print) -> None:
    base = [str(conda_exe), "run", "-n", ENV_NAME, "python", "-m"]
    run_live([*base, "pip", "install", "-U", "pip"], cwd=project_root, log=log)
    run_live([*base, "pip", "install", "-r", "requirements.txt"], cwd=project_root, log=log)
    run_live([*base, "pip", "install", "-r", "requirements-pycaret.txt"], cwd=project_root, log=log)
    run_live([*base, "pip", "install", "-e", "."], cwd=project_root, log=log)


def ensure_env_file(project_root: Path, *, log=print) -> None:
    env_file = project_root / ".env"
    example = project_root / ".env.example"
    if env_file.exists():
        log(f"[OK] .env exists: {env_file}")
        return
    if example.exists():
        env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"[DONE] Created .env from .env.example: {env_file}")
        return
    log("[WARN] .env.example was not found. Skipping .env creation.")


def set_env_ollama_model(project_root: Path, model: str, *, log=print) -> None:
    model = normalize_ollama_model(model)
    ensure_env_file(project_root, log=log)
    env_file = project_root / ".env"
    lines: list[str] = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
    found = False
    for index, line in enumerate(lines):
        if line.strip().startswith("OLLAMA_MODEL="):
            lines[index] = f"OLLAMA_MODEL={model}"
            found = True
            break
    if not found:
        lines.append(f"OLLAMA_MODEL={model}")
    env_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    log(f"[DONE] .env default Ollama model set: {model}")


def set_config_ollama_model(project_root: Path, model: str, *, log=print) -> None:
    model = normalize_ollama_model(model)
    config_path = project_root / "config.json"
    raw: dict[str, object] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception as exc:
            log(f"[WARN] Could not read config.json; rewriting minimal config: {exc}")
    raw["default_backend"] = "ollama"
    ollama_cfg = raw.get("ollama") if isinstance(raw.get("ollama"), dict) else {}
    ollama_cfg = dict(ollama_cfg)
    ollama_cfg.setdefault("base_url", "http://127.0.0.1:11434")
    ollama_cfg["model"] = model
    ollama_cfg.setdefault("num_ctx", 16384)
    raw["ollama"] = ollama_cfg
    config_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[DONE] config.json default Ollama model set: {model}")


def set_default_ollama_model(project_root: Path, model: str, *, log=print) -> None:
    set_env_ollama_model(project_root, model, log=log)
    set_config_ollama_model(project_root, model, log=log)


def ensure_streamlit_config(*, log=print) -> None:
    user_profile = os.environ.get("USERPROFILE")
    if not user_profile:
        log("[WARN] USERPROFILE was not found. Skipping Streamlit user config.")
        return
    config_dir = Path(user_profile) / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / "config.toml"
    target.write_text(
        "[server]\n"
        "headless = true\n"
        "showEmailPrompt = false\n\n"
        "[browser]\n"
        "gatherUsageStats = false\n",
        encoding="utf-8",
    )
    log(f"[DONE] Streamlit user config ready: {target}")


def ensure_ollama_model(ollama_exe: Path | None, model: str, *, log=print) -> None:
    model = normalize_ollama_model(model)
    if not ollama_exe:
        log("[WARN] ollama.exe was not found. Skipping model download.")
        return
    if ollama_model_exists(ollama_exe, model):
        log(f"[OK] Ollama model exists: {model}")
        return
    if run_quiet([str(ollama_exe), "list"])[0] != 0:
        log("[RUNNING] Starting Ollama server.")
        subprocess.Popen(
            [str(ollama_exe), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags(),
        )
        time.sleep(5)
    run_live([str(ollama_exe), "pull", model], log=log)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def make_writable(func, path: str, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_path(path: Path, allowed_root: Path, *, log=print) -> None:
    if not path.exists() and not path.is_symlink():
        log(f"[SKIP] Missing: {path}")
        return
    if path.resolve() == allowed_root.resolve() or not is_under(path, allowed_root):
        raise RuntimeError(f"Refusing to delete outside allowed root: {path}")
    log(f"[DELETE] {path}")
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


def windows_uninstall_entry_exists() -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_REGISTRY_KEY):
            return True
    except OSError:
        return False


def register_windows_uninstaller(project_root: Path | None, *, log=print) -> None:
    if os.name != "nt":
        log("[SKIP] Windows Apps uninstall registration is only available on Windows.")
        return
    try:
        import winreg

        install_root = project_root or find_project_root() or exe_dir()
        manager_exe = manager_exe_for_registration(project_root)
        uninstall_command = f"{quote_windows_arg(str(manager_exe))} --uninstall"
        estimated_size_kb = 0
        if manager_exe.exists():
            estimated_size_kb = max(1, int(manager_exe.stat().st_size / 1024))

        key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, UNINSTALL_REGISTRY_KEY, 0, winreg.KEY_SET_VALUE)
        with key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_DISPLAY_NAME)
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, APP_VERSION)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, APP_PUBLISHER)
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(install_root))
            winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, f"{str(manager_exe)},0")
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, uninstall_command)
            winreg.SetValueEx(key, "URLInfoAbout", 0, winreg.REG_SZ, GITHUB_URL)
            winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
            if estimated_size_kb:
                winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, estimated_size_kb)
        log(f"[DONE] Registered Windows Apps uninstall entry: {APP_DISPLAY_NAME}")
    except Exception as exc:
        log(f"[WARN] Could not register Windows Apps uninstall entry: {exc}")


def unregister_windows_uninstaller(*, log=print) -> None:
    if os.name != "nt":
        log("[SKIP] Windows Apps uninstall registration is only available on Windows.")
        return
    try:
        import winreg

        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_REGISTRY_KEY)
        log(f"[DONE] Removed Windows Apps uninstall entry: {APP_DISPLAY_NAME}")
    except FileNotFoundError:
        log(f"[SKIP] Windows Apps uninstall entry is missing: {APP_DISPLAY_NAME}")
    except OSError as exc:
        log(f"[WARN] Could not remove Windows Apps uninstall entry: {exc}")


class ManagerApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.tk = tk
        self.ttk = ttk
        self.messagebox = tk_messagebox
        self.root = tk.Tk()
        self.root.title("AIM4LAB LocalCustomGUI Manager")
        self.root.geometry("900x680")
        self.root.minsize(820, 600)
        self.root.configure(bg="#f4f6f8")
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.streamlit_proc: subprocess.Popen[str] | None = None
        self.hidden_to_tray = False
        self.tray_icon = None
        self.tray_thread: threading.Thread | None = None

        self._setup_style()
        self._build_layout(ScrolledText)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)
        self.root.after(250, self.refresh_delete_options)

    def _setup_style(self) -> None:
        style = self.ttk.Style()
        for theme in ("vista", "xpnative", "winnative", "clam"):
            try:
                if theme in style.theme_names():
                    style.theme_use(theme)
                    break
            except self.tk.TclError:
                continue
        style.configure("TFrame", background="#f4f6f8")
        style.configure("Header.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f4f6f8", foreground="#1f2937", font=("Segoe UI", 10))
        style.configure("HeaderTitle.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI", 18, "bold"))
        style.configure("HeaderSub.TLabel", background="#ffffff", foreground="#6b7280", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(14, 8))
        style.configure("TCheckbutton", background="#f4f6f8", foreground="#1f2937", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#f4f6f8", foreground="#374151")
        style.configure("TLabelframe.Label", background="#f4f6f8", foreground="#374151", font=("Segoe UI", 10, "bold"))
        style.configure("Horizontal.TProgressbar", troughcolor="#e5e7eb", background="#f97316", thickness=12)

    def _build_layout(self, scrolled_text_class) -> None:
        tk = self.tk
        ttk = self.ttk
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(22, 18))
        header.pack(fill="x")
        logo_path = resource_path("Logo", "logo_aim4lab.png")
        icon_path = resource_path("Icon", "aim4lab_app_icon.png")
        self.app_icon_image = None
        self.logo_image = None
        self.logo_display = None
        if icon_path.exists():
            try:
                self.app_icon_image = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(False, self.app_icon_image)
            except tk.TclError:
                pass
        if logo_path.exists():
            try:
                self.logo_image = tk.PhotoImage(file=str(logo_path))
                factor = max(1, self.logo_image.width() // 190)
                self.logo_display = self.logo_image.subsample(factor, factor)
                logo_label = tk.Label(header, image=self.logo_display, bg="#ffffff", borderwidth=0)
                logo_label.pack(side="left", padx=(0, 22))
            except tk.TclError:
                pass
        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text="LocalCustomGUI Manager", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_box,
            text="Install, run, and uninstall the Windows environment from one AIM4LAB wizard.",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        self.status_var = tk.StringVar(value="Ready")
        header_actions = ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side="right", anchor="n")
        ttk.Label(header_actions, textvariable=self.status_var, style="HeaderSub.TLabel").pack(anchor="e")
        ttk.Button(header_actions, text="GitHub", command=lambda: webbrowser.open(GITHUB_URL)).pack(anchor="e", pady=(8, 0))

        body = ttk.Frame(self.root, padding=(18, 16))
        body.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)

        self.install_tab = ttk.Frame(self.notebook, padding=18)
        self.run_tab = ttk.Frame(self.notebook, padding=18)
        self.uninstall_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.install_tab, text="Install")
        self.notebook.add(self.run_tab, text="Run")
        self.notebook.add(self.uninstall_tab, text="Uninstall")

        self._build_install_tab()
        self._build_run_tab()
        self._build_uninstall_tab()

        log_frame = ttk.LabelFrame(body, text="Progress Log", padding=10)
        log_frame.pack(fill="both", expand=False, pady=(14, 0))
        self.log_text = scrolled_text_class(log_frame, height=10, wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")
        self.log("AIM4LAB LocalCustomGUI Manager is ready.")

    def _build_install_tab(self) -> None:
        ttk = self.ttk
        tk = self.tk
        info = ttk.LabelFrame(self.install_tab, text="Install Options", padding=14)
        info.pack(fill="x")
        self.install_model_var = tk.BooleanVar(value=True)
        self.install_launch_var = tk.BooleanVar(value=True)
        recommended_model = recommended_ollama_model()
        self.install_ollama_model_vars = {
            OLLAMA_MODEL_E2B: tk.BooleanVar(value=recommended_model == OLLAMA_MODEL_E2B),
            OLLAMA_MODEL_E4B: tk.BooleanVar(value=recommended_model == OLLAMA_MODEL_E4B),
        }
        ttk.Label(info, text=ram_recommendation_text(), wraplength=820, foreground="#374151").pack(anchor="w", pady=(0, 2))
        ttk.Label(info, text=detected_ram_text(), font=("Segoe UI", 10, "bold"), foreground="#111827").pack(anchor="w", pady=(0, 8))
        model_box = ttk.Frame(info)
        model_box.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(
            model_box,
            text=f"{OLLAMA_MODEL_E2B} - recommended for 16 GB RAM PCs",
            variable=self.install_ollama_model_vars[OLLAMA_MODEL_E2B],
        ).pack(anchor="w", pady=1)
        ttk.Checkbutton(
            model_box,
            text=f"{OLLAMA_MODEL_E4B} - better quality, 32 GB RAM or more recommended",
            variable=self.install_ollama_model_vars[OLLAMA_MODEL_E4B],
        ).pack(anchor="w", pady=1)
        ttk.Label(
            info,
            text="You can select both models. If both are selected, the app default follows the RAM recommendation.",
            wraplength=820,
            foreground="#6b7280",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(info, text="Download or verify selected Ollama model(s)", variable=self.install_model_var).pack(
            anchor="w", pady=2
        )
        ttk.Checkbutton(info, text="Launch Streamlit after install", variable=self.install_launch_var).pack(anchor="w", pady=2)
        buttons = ttk.Frame(self.install_tab)
        buttons.pack(fill="x", pady=(16, 8))
        self.install_button = ttk.Button(buttons, text="Install / Repair", style="Accent.TButton", command=self.start_install)
        self.install_button.pack(side="left")
        ttk.Button(buttons, text="Open Project Folder", command=self.open_project_folder).pack(side="left", padx=(10, 0))
        self.progress_var = tk.DoubleVar(value=0)
        self.step_var = tk.StringVar(value="Waiting to start.")
        ttk.Label(self.install_tab, textvariable=self.step_var).pack(anchor="w", pady=(14, 4))
        self.progress = ttk.Progressbar(
            self.install_tab,
            variable=self.progress_var,
            maximum=TOTAL_INSTALL_STEPS,
            mode="determinate",
            style="Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x")
        note = (
            "The manager checks Miniconda and Ollama, creates the conda env, installs Python packages, "
            "prepares Streamlit settings, and can launch the app."
        )
        ttk.Label(self.install_tab, text=note, wraplength=780, foreground="#6b7280").pack(anchor="w", pady=(18, 0))

    def _build_run_tab(self) -> None:
        ttk = self.ttk
        self.url_var = self.tk.StringVar(value=f"http://127.0.0.1:{PORT}")
        row = ttk.Frame(self.run_tab)
        row.pack(fill="x")
        self.run_button = ttk.Button(row, text="Run App", style="Accent.TButton", command=self.start_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(row, text="Stop App", command=self.stop_streamlit)
        self.stop_button.pack(side="left", padx=(10, 0))
        ttk.Button(row, text="Open Browser", command=lambda: webbrowser.open(self.url_var.get())).pack(side="left", padx=(10, 0))
        ttk.Label(self.run_tab, text="Local URL").pack(anchor="w", pady=(20, 4))
        ttk.Label(self.run_tab, textvariable=self.url_var, font=("Segoe UI", 14, "bold"), foreground="#f97316").pack(anchor="w")
        note = "Use this tab after installation. The manager starts Streamlit from the local_customgui_windows conda environment."
        ttk.Label(self.run_tab, text=note, wraplength=780, foreground="#6b7280").pack(anchor="w", pady=(18, 0))

    def _build_uninstall_tab(self) -> None:
        ttk = self.ttk
        tk = self.tk
        self.delete_vars: dict[str, object] = {
            "conda_env": tk.BooleanVar(value=True),
            "ollama_model": tk.BooleanVar(value=True),
            "runtime": tk.BooleanVar(value=True),
            "env_file": tk.BooleanVar(value=True),
            "windows_uninstall_entry": tk.BooleanVar(value=True),
            "models": tk.BooleanVar(value=False),
            "streamlit_config": tk.BooleanVar(value=False),
        }
        self.delete_text_vars: dict[str, object] = {}
        box = ttk.LabelFrame(self.uninstall_tab, text="Uninstall Items", padding=14)
        box.pack(fill="x")
        for key, label in (
            ("conda_env", f"Conda env: {ENV_NAME}"),
            ("ollama_model", f"Ollama models: {' / '.join(OLLAMA_MODELS)}"),
            ("runtime", "Runtime state, logs, and caches"),
            ("env_file", "Local app config: .env"),
            ("windows_uninstall_entry", "Windows Apps uninstall entry"),
            ("models", "Model artifacts under models/classification and models/regression"),
            ("streamlit_config", "Streamlit user config in %USERPROFILE%\\.streamlit"),
        ):
            text_var = tk.StringVar(value=label)
            self.delete_text_vars[key] = text_var
            ttk.Checkbutton(box, textvariable=text_var, variable=self.delete_vars[key]).pack(anchor="w", pady=2)
        controls = ttk.Frame(self.uninstall_tab)
        controls.pack(fill="x", pady=(14, 8))
        ttk.Button(controls, text="Recommended", command=self.select_recommended_delete).pack(side="left")
        ttk.Button(controls, text="Select All", command=self.select_all_delete).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Clear", command=self.clear_delete_selection).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Refresh", command=self.refresh_delete_options).pack(side="left", padx=(8, 0))
        ttk.Label(self.uninstall_tab, text="Type DELETE to enable uninstall").pack(anchor="w", pady=(16, 4))
        self.delete_confirm_var = tk.StringVar()
        self.delete_entry = ttk.Entry(self.uninstall_tab, textvariable=self.delete_confirm_var, width=22)
        self.delete_entry.pack(anchor="w")
        self.uninstall_button = ttk.Button(
            self.uninstall_tab,
            text="Uninstall Selected",
            style="Accent.TButton",
            command=self.start_uninstall,
        )
        self.uninstall_button.pack(anchor="w", pady=(16, 0))
        note = (
            "Recommended cleanup removes the conda env, Ollama model, runtime state/log/cache, .env, and Windows Apps entry. "
            "The project folder itself is never deleted automatically."
        )
        ttk.Label(self.uninstall_tab, text=note, wraplength=780, foreground="#6b7280").pack(anchor="w", pady=(18, 0))

    def log(self, message: str = "") -> None:
        self.queue.put(("log", message))

    def set_status(self, message: str) -> None:
        self.queue.put(("status", message))

    def set_step(self, index: int, message: str) -> None:
        self.queue.put(("progress", (index, message)))

    def _poll_queue(self) -> None:
        while True:
            try:
                event, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                self.log_text.configure(state="normal")
                self.log_text.insert("end", str(payload) + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            elif event == "status":
                self.status_var.set(str(payload))
            elif event == "progress":
                index, message = payload  # type: ignore[misc]
                self.progress_var.set(float(index))
                self.step_var.set(str(message))
            elif event == "done":
                self.busy = False
                self._set_buttons_enabled(True)
                self.status_var.set(str(payload))
            elif event == "clear_delete_confirm":
                self.delete_confirm_var.set("")
            elif event == "refresh_delete":
                self.refresh_delete_options()
            elif event == "shutdown":
                self.shutdown_app()
        self.root.after(100, self._poll_queue)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (self.install_button, self.run_button, self.uninstall_button):
            button.configure(state=state)

    def start_task(self, name: str, target) -> None:
        if self.busy:
            self.messagebox.showinfo("Busy", "Another task is already running.")
            return
        self.busy = True
        self._set_buttons_enabled(False)
        self.set_status(name)

        def worker() -> None:
            try:
                target()
                self.queue.put(("done", "Ready"))
            except Exception as exc:
                self.log("")
                self.log(f"[ERROR] {exc}")
                self.queue.put(("done", "Error"))

        threading.Thread(target=worker, daemon=True).start()

    def start_install(self) -> None:
        download_model = bool(self.install_model_var.get())
        launch_after = bool(self.install_launch_var.get())
        ollama_models = [
            model
            for model in OLLAMA_MODELS
            if bool(self.install_ollama_model_vars[model].get())  # type: ignore[index,attr-defined]
        ]
        if not ollama_models:
            self.messagebox.showwarning("Model selection required", "Select at least one Ollama model.")
            return
        default_model = default_ollama_model_for_selection(ollama_models)
        self.start_task("Installing", lambda: self.install_flow(download_model, launch_after, ollama_models, default_model))

    def install_flow(self, download_model: bool, launch_after: bool, ollama_models: list[str], default_model: str) -> None:
        ollama_models = normalize_ollama_models(ollama_models)
        default_model = default_ollama_model_for_selection(ollama_models)
        self.log("")
        self.log("=== Install / Repair ===")
        self.log(f"[INFO] Selected Ollama models: {', '.join(ollama_models)}")
        self.log(f"[INFO] App default Ollama model: {default_model}")
        self.log(f"[INFO] {ram_recommendation_text()}")
        self.log(f"[INFO] {detected_ram_text()}")
        self.set_step(1, "Checking project and install paths")
        project_root = require_project_root()
        self.log(f"[INFO] Project: {project_root}")
        self.log(f"[INFO] Logo: {resource_path('Logo', 'logo_aim4lab.png')}")

        self.set_step(2, "Checking or installing Miniconda")
        conda_exe = ensure_conda(log=self.log)

        self.set_step(3, "Checking or installing Ollama")
        ollama_exe = ensure_ollama(log=self.log)

        self.set_step(4, f"Preparing conda env: {ENV_NAME}")
        ensure_conda_env(conda_exe, log=self.log)

        self.set_step(5, "Installing Python packages and PyCaret")
        install_python_deps(conda_exe, project_root, log=self.log)

        self.set_step(6, f"Preparing app config: {default_model}")
        set_default_ollama_model(project_root, default_model, log=self.log)

        self.set_step(7, "Preparing Streamlit user config")
        ensure_streamlit_config(log=self.log)
        register_windows_uninstaller(project_root, log=self.log)

        self.set_step(8, f"Preparing Ollama models: {', '.join(ollama_models)}")
        if download_model:
            for model in ollama_models:
                ensure_ollama_model(ollama_exe, model, log=self.log)
        else:
            self.log("[SKIP] Ollama model download was unchecked.")

        self.set_step(9, "Finishing")
        self.log("[DONE] Install / repair completed.")
        if launch_after:
            self.start_streamlit_process(project_root, conda_exe)

    def start_run(self) -> None:
        self.start_task("Starting app", self.run_flow)

    def run_flow(self) -> None:
        self.log("")
        self.log("=== Run App ===")
        project_root = require_project_root()
        conda_exe = find_conda_exe()
        if not conda_exe:
            raise RuntimeError("conda.exe was not found. Run Install first.")
        if not conda_env_exists(conda_exe):
            raise RuntimeError(f"Conda env was not found: {ENV_NAME}. Run Install first.")
        self.start_streamlit_process(project_root, conda_exe)

    def restart_server_from_tray(self) -> None:
        if self.busy:
            self.log("[WAIT] Another task is already running. Restart was skipped.")
            return
        self.start_task("Restarting server", self.restart_server_flow)

    def restart_server_flow(self) -> None:
        self.log("")
        self.log("=== Restart Server ===")
        self.stop_streamlit()
        self.run_flow()

    def start_streamlit_process(self, project_root: Path, conda_exe: Path) -> None:
        url = f"http://127.0.0.1:{PORT}"
        if self.streamlit_proc and self.streamlit_proc.poll() is None:
            self.log(f"[OK] Streamlit is already running: {url}")
            webbrowser.open(url)
            return
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_SERVER_SHOW_EMAIL_PROMPT"] = "false"
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
        self.log("[RUNNING] " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags(),
        )
        self.streamlit_proc = proc

        def reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log(line.rstrip())

        threading.Thread(target=reader, daemon=True).start()
        if wait_for_server(url):
            self.log(f"[DONE] Server is running: {url}")
        else:
            self.log(f"[WAIT] Server is still starting. Check your browser shortly: {url}")
        webbrowser.open(url)

    def stop_streamlit(self) -> None:
        proc = self.streamlit_proc
        if not proc or proc.poll() is not None:
            self.log("[OK] No Streamlit process is running from this manager.")
            self.streamlit_proc = None
            return
        self.log("[STOPPING] Streamlit process")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self.streamlit_proc = None
        self.log("[DONE] Streamlit process stopped.")

    def is_streamlit_running(self) -> bool:
        return self.streamlit_proc is not None and self.streamlit_proc.poll() is None

    def open_project_folder(self) -> None:
        project_root = find_project_root()
        if not project_root:
            self.messagebox.showwarning("Project not found", "Could not find the project folder.")
            return
        os.startfile(str(project_root))

    def delete_item_status(self) -> dict[str, bool]:
        project_root = find_project_root()
        conda_exe = find_conda_exe()
        ollama_exe = find_ollama_exe()
        return {
            "conda_env": conda_env_exists(conda_exe),
            "ollama_model": bool(existing_ollama_models(ollama_exe)),
            "runtime": bool(runtime_targets(project_root)),
            "env_file": bool(env_targets(project_root)),
            "windows_uninstall_entry": windows_uninstall_entry_exists(),
            "models": bool(model_targets(project_root)),
            "streamlit_config": streamlit_config_target() is not None,
        }

    def refresh_delete_options(self) -> None:
        labels = {
            "conda_env": f"Conda env: {ENV_NAME}",
            "ollama_model": f"Ollama models: {' / '.join(OLLAMA_MODELS)}",
            "runtime": "Runtime state, logs, and caches",
            "env_file": "Local app config: .env",
            "windows_uninstall_entry": "Windows Apps uninstall entry",
            "models": "Model artifacts under models/classification and models/regression",
            "streamlit_config": "Streamlit user config in %USERPROFILE%\\.streamlit",
        }
        try:
            statuses = self.delete_item_status()
        except Exception as exc:
            self.log(f"[WARN] Could not refresh uninstall status: {exc}")
            return
        for key, available in statuses.items():
            status = "available" if available else "missing"
            self.delete_text_vars[key].set(f"{labels[key]} [{status}]")  # type: ignore[index]

    def open_uninstall_page(self) -> None:
        self.notebook.select(self.uninstall_tab)
        self.select_recommended_delete()
        self.refresh_delete_options()
        self.status_var.set("Uninstall ready")
        self.root.after(250, self.delete_entry.focus_set)
        self.log("[INFO] Opened Uninstall page from Windows Apps uninstall action.")

    def select_recommended_delete(self) -> None:
        for key, var in self.delete_vars.items():
            var.set(key in {"conda_env", "ollama_model", "runtime", "env_file", "windows_uninstall_entry"})  # type: ignore[attr-defined]

    def select_all_delete(self) -> None:
        for var in self.delete_vars.values():
            var.set(True)  # type: ignore[attr-defined]

    def clear_delete_selection(self) -> None:
        for var in self.delete_vars.values():
            var.set(False)  # type: ignore[attr-defined]

    def selected_delete_keys(self) -> list[str]:
        return [key for key, var in self.delete_vars.items() if bool(var.get())]  # type: ignore[attr-defined]

    def delete_preview(self, keys: list[str]) -> list[str]:
        project_root = find_project_root()
        lines: list[str] = []
        if "conda_env" in keys:
            lines.append(f"conda env remove -n {ENV_NAME}")
        if "ollama_model" in keys:
            lines.extend(f"ollama rm {model}" for model in OLLAMA_MODELS)
        if "runtime" in keys:
            lines.extend(str(path) for path in runtime_targets(project_root))
        if "env_file" in keys:
            lines.extend(str(path) for path in env_targets(project_root))
        if "windows_uninstall_entry" in keys:
            lines.append("HKCU\\" + UNINSTALL_REGISTRY_KEY)
        if "models" in keys:
            lines.extend(str(path) for path in model_targets(project_root))
        if "streamlit_config" in keys:
            target = streamlit_config_target()
            if target:
                lines.append(str(target))
        return lines

    def start_uninstall(self) -> None:
        keys = self.selected_delete_keys()
        if not keys:
            self.messagebox.showinfo("No selection", "Select at least one uninstall item.")
            return
        if self.delete_confirm_var.get().strip() != "DELETE":
            self.messagebox.showwarning("Confirmation required", "Type DELETE before uninstalling selected items.")
            return
        preview = self.delete_preview(keys)
        self.log("")
        self.log("=== Uninstall Preview ===")
        for line in preview[:40]:
            self.log(line)
        if len(preview) > 40:
            self.log(f"... and {len(preview) - 40} more")
        if not self.messagebox.askyesno(
            "Confirm uninstall",
            "Selected items will be deleted. The project folder itself will not be deleted automatically.\n\nContinue?",
        ):
            self.log("[CANCELLED] Uninstall cancelled.")
            return
        self.start_task("Uninstalling", lambda: self.uninstall_flow(keys))

    def uninstall_flow(self, keys: list[str]) -> None:
        self.log("")
        self.log("=== Uninstall Selected ===")
        self.stop_streamlit()
        project_root = find_project_root()
        conda_exe = find_conda_exe()
        ollama_exe = find_ollama_exe()
        if "conda_env" in keys:
            if conda_exe and conda_env_exists(conda_exe):
                run_live([str(conda_exe), "env", "remove", "-y", "-n", ENV_NAME], log=self.log)
            else:
                self.log(f"[SKIP] Conda env is missing: {ENV_NAME}")
        if "ollama_model" in keys:
            removed_any = False
            for model in OLLAMA_MODELS:
                if ollama_exe and ollama_model_exists(ollama_exe, model):
                    run_live([str(ollama_exe), "rm", model], log=self.log)
                    removed_any = True
            if not removed_any:
                self.log(f"[SKIP] Ollama models are missing: {' / '.join(OLLAMA_MODELS)}")
        if project_root:
            if "runtime" in keys:
                for target in runtime_targets(project_root):
                    remove_path(target, project_root, log=self.log)
            if "env_file" in keys:
                for target in env_targets(project_root):
                    remove_path(target, project_root, log=self.log)
            if "models" in keys:
                for target in model_targets(project_root):
                    remove_path(target, project_root, log=self.log)
        elif any(key in keys for key in ("runtime", "env_file", "models")):
            self.log("[SKIP] Project root was not found.")
        if "streamlit_config" in keys:
            target = streamlit_config_target()
            if target:
                remove_path(target, target.parent, log=self.log)
            else:
                self.log("[SKIP] Streamlit user config is missing.")
        if "windows_uninstall_entry" in keys:
            unregister_windows_uninstaller(log=self.log)
        self.queue.put(("clear_delete_confirm", None))
        self.queue.put(("refresh_delete", None))
        self.log("[DONE] Uninstall completed.")

    def ensure_tray_icon(self) -> bool:
        if self.tray_icon is not None:
            return True
        try:
            import pystray
            from PIL import Image
        except Exception as exc:
            self.log(f"[WARN] System tray could not be initialized: {exc}")
            return False

        icon_path = resource_path("Icon", "aim4lab_app_icon.png")
        if not icon_path.exists():
            self.log(f"[WARN] Tray icon was not found: {icon_path}")
            return False

        def call_on_tk(callback):
            return lambda _icon, _item=None: self.root.after(0, callback)

        image = Image.open(icon_path)
        menu = pystray.Menu(
            pystray.MenuItem("Open GUI", call_on_tk(self.show_from_tray), default=True),
            pystray.MenuItem("Restart Server", call_on_tk(self.restart_server_from_tray)),
            pystray.MenuItem("Quit Server", call_on_tk(self.quit_server_from_tray)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Browser", lambda _icon, _item=None: webbrowser.open(f"http://127.0.0.1:{PORT}")),
        )
        self.tray_icon = pystray.Icon(
            "LocalCustomGUI",
            image,
            "AIM4LAB LocalCustomGUI Server",
            menu,
        )

        if hasattr(self.tray_icon, "run_detached"):
            self.tray_icon.run_detached()
        else:
            self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            self.tray_thread.start()
        return True

    def hide_to_tray(self) -> None:
        if not self.ensure_tray_icon():
            self.messagebox.showwarning(
                "Tray unavailable",
                "The server is still running, but the tray icon could not be created.",
            )
            return
        self.hidden_to_tray = True
        self.root.withdraw()
        self.log("[INFO] Manager hidden to system tray. Right-click the tray icon for server actions.")
        self.status_var.set("Running in tray")
        tray_icon = self.tray_icon
        if tray_icon is not None and hasattr(tray_icon, "notify"):
            try:
                tray_icon.notify(
                    "Server is still running. Right-click this icon to restart or quit.",
                    "AIM4LAB LocalCustomGUI",
                )
            except Exception:
                pass

    def show_from_tray(self) -> None:
        self.hidden_to_tray = False
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.status_var.set("Ready")

    def quit_server_from_tray(self) -> None:
        if self.busy:
            self.log("[WAIT] Another task is already running. Quit was skipped.")
            return

        def worker() -> None:
            try:
                self.log("")
                self.log("=== Quit Server ===")
                self.stop_streamlit()
            finally:
                self.queue.put(("shutdown", None))

        threading.Thread(target=worker, daemon=True).start()

    def stop_tray_icon(self) -> None:
        tray_icon = self.tray_icon
        self.tray_icon = None
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass

    def shutdown_app(self) -> None:
        self.stop_tray_icon()
        self.root.destroy()

    def _on_close(self) -> None:
        if self.is_streamlit_running():
            self.hide_to_tray()
            return
        self.shutdown_app()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def self_test() -> int:
    project_root = find_project_root()
    conda_exe = find_conda_exe()
    ollama_exe = find_ollama_exe()
    logo = resource_path("Logo", "logo_aim4lab.png")
    app_icon = resource_path("Icon", "aim4lab_app_icon.png")
    try:
        import pystray  # noqa: F401

        pystray_ok = True
    except Exception:
        pystray_ok = False
    print(f"project={project_root}")
    print(f"conda={conda_exe}")
    print(f"ollama={ollama_exe}")
    print(f"logo={logo} exists={logo.exists()}")
    print(f"app_icon={app_icon} exists={app_icon.exists()}")
    print(f"pystray={pystray_ok}")
    print(f"conda_env={conda_env_exists(conda_exe)}")
    for model in OLLAMA_MODELS:
        print(f"ollama_model_{model}={ollama_model_exists(ollama_exe, model)}")
    return 0 if project_root and logo.exists() and app_icon.exists() and pystray_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="AIM4LAB LocalCustomGUI Windows manager.")
    parser.add_argument("--self-test", action="store_true", help="Check paths and exit.")
    parser.add_argument("--uninstall", action="store_true", help="Open the Uninstall tab.")
    parser.add_argument("--register-uninstall", action="store_true", help="Register Windows Apps uninstall entry and exit.")
    parser.add_argument("--unregister-uninstall", action="store_true", help="Remove Windows Apps uninstall entry and exit.")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.register_uninstall:
        register_windows_uninstaller(find_project_root(), log=print)
        return 0
    if args.unregister_uninstall:
        unregister_windows_uninstaller(log=print)
        return 0
    app = ManagerApp()
    if args.uninstall:
        app.root.after(350, app.open_uninstall_page)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
