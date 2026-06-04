#!/usr/bin/env python3
"""
Standalone local LLM chat GUI.

This is intentionally independent from autonomous_researcher:
- Browser UI sends chat messages to this Python server.
- The Python server calls either Ollama /api/chat or vLLM OpenAI-compatible /v1/chat/completions.
- No external Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
ICON_DIR = ROOT / "Icon"
CONFIG_PATH = ROOT / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "default_backend": "ollama",
    "timeout_sec": 240,
    "temperature": 0.3,
    "max_tokens": 2048,
    "system_prompt": "You are a professional, user-friendly interface for high-voltage circuit breaker performance prediction. Answer in Korean unless the user asks otherwise.",
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "model": "llama3.1:8b",
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "local-model",
        "api_key": "EMPTY",
        "context_window": 8192,
        "max_context_tokens": 8192,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return deep_merge(DEFAULT_CONFIG, raw if isinstance(raw, dict) else {})


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def http_post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None, timeout: float = 240) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(url, data=json_bytes(payload), headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            if not data:
                return {}
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed for {url}: {exc}") from exc


def http_post_json_stream(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 240,
) -> Iterator[str]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(url, data=json_bytes(payload), headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    yield line
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed for {url}: {exc}") from exc


def http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 15) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            return json.loads(data.decode("utf-8")) if data else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed for {url}: {exc}") from exc


def normalize_messages(messages: Any, system_prompt: str) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    if system_prompt.strip():
        clean.append({"role": "system", "content": system_prompt.strip()})
    if not isinstance(messages, list):
        return clean
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role == "system" and system_prompt.strip():
            continue
        message: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            tool_name = str(item.get("tool_name") or item.get("name") or "").strip()
            if not tool_name or not content:
                continue
            message["tool_name"] = tool_name
        elif not content and not item.get("tool_calls"):
            continue
        if item.get("tool_calls"):
            message["tool_calls"] = item["tool_calls"]
        if item.get("thinking"):
            message["thinking"] = str(item["thinking"])
        clean.append(message)
    return clean


def split_reasoning(content: str, explicit_reasoning: str = "") -> tuple[str, str]:
    reasoning = explicit_reasoning.strip()
    text = content or ""

    think = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think:
        reasoning = reasoning or think.group(1).strip()
        text = (text[: think.start()] + text[think.end() :]).strip()

    channel = re.search(r"<\|channel\>\s*thought\s*(.*?)<channel\|>", text, flags=re.DOTALL)
    if channel:
        reasoning = reasoning or channel.group(1).strip()
        text = (text[: channel.start()] + text[channel.end() :]).strip()

    return text.strip(), reasoning


def split_stream_reasoning(content: str, explicit_reasoning: str = "") -> tuple[str, str]:
    text, reasoning = split_reasoning(content, explicit_reasoning)
    if reasoning:
        return text, reasoning

    raw = content or ""
    lower = raw.lower()
    start = lower.find("<think>")
    if start >= 0:
        end = lower.find("</think>", start)
        if end < 0:
            reasoning = raw[start + len("<think>") :].strip()
            text = raw[:start].strip()
    return text.strip(), reasoning.strip()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _estimate_llm_tokens(text: Any) -> int:
    if not text:
        return 0
    return max(1, int(len(str(text)) / 2) + 1)


def _estimate_request_input_tokens(req: dict[str, Any], system_prompt: str) -> int:
    total = _estimate_llm_tokens(system_prompt) + 16
    messages = req.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict):
                continue
            total += 8
            total += _estimate_llm_tokens(item.get("role") or "")
            total += _estimate_llm_tokens(item.get("content") or "")
    return total


def vllm_context_limit(config: dict[str, Any], req: dict[str, Any]) -> int:
    backend_cfg = dict(config.get("vllm", {}))
    for source in (req, backend_cfg):
        for key in ("max_context_tokens", "context_window", "context_length", "max_model_len"):
            parsed = _positive_int(source.get(key))
            if parsed:
                return parsed

    model = str(req.get("model") or backend_cfg.get("model") or "")
    k8s_cfg = backend_cfg.get("nemoclaw_k8s") if isinstance(backend_cfg.get("nemoclaw_k8s"), dict) else {}
    model_cfg = {}
    if isinstance(k8s_cfg, dict):
        models = k8s_cfg.get("models")
        if isinstance(models, dict) and isinstance(models.get(model), dict):
            model_cfg = models[model]
    for key in ("max_model_len", "max_context_tokens", "context_window"):
        parsed = _positive_int(model_cfg.get(key))
        if parsed:
            return parsed
    return 8192


def resolve_vllm_max_tokens(config: dict[str, Any], req: dict[str, Any]) -> int:
    explicit = req.get("max_tokens")
    requested: int | None = None
    if explicit not in (None, "", "auto", "default"):
        parsed = _positive_int(explicit)
        if parsed:
            requested = parsed
    context_limit = vllm_context_limit(config, req)
    if requested is None:
        requested = max(512, min(4096, context_limit // 2))
    system_prompt = str(req.get("system_prompt") or config.get("system_prompt") or "")
    input_estimate = _estimate_request_input_tokens(req, system_prompt)
    allowed = max(64, context_limit - input_estimate - 96)
    return max(64, min(requested, allowed))


def call_ollama(config: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    backend_cfg = dict(config.get("ollama", {}))
    base_url = str(req.get("base_url") or backend_cfg.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
    model = str(req.get("model") or backend_cfg.get("model") or "llama3.1:8b")
    timeout = float(req.get("timeout_sec") or config.get("timeout_sec") or 240)
    temperature = float(req.get("temperature") if req.get("temperature") is not None else config.get("temperature", 0.3))
    max_tokens = int(req.get("max_tokens") or config.get("max_tokens") or 2048)
    messages = normalize_messages(req.get("messages"), str(req.get("system_prompt") or config.get("system_prompt") or ""))
    started = time.time()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if "tools" in req:
        payload["tools"] = req["tools"]
    if "think" in req:
        payload["think"] = req["think"]
    if req.get("format") is not None:
        payload["format"] = req["format"]
    raw = http_post_json(f"{base_url}/api/chat", payload, timeout=timeout)
    message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
    content, reasoning = split_reasoning(
        str(message.get("content") or raw.get("response") or ""),
        str(message.get("thinking") or message.get("reasoning") or raw.get("thinking") or raw.get("reasoning") or ""),
    )
    return {
        "ok": True,
        "backend": "ollama",
        "model": model,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else [],
        "elapsed_sec": round(time.time() - started, 3),
        "raw_summary": {
            "done": raw.get("done"),
            "done_reason": raw.get("done_reason"),
            "total_duration": raw.get("total_duration"),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
        },
    }


def stream_ollama(config: dict[str, Any], req: dict[str, Any]) -> Iterator[dict[str, Any]]:
    backend_cfg = dict(config.get("ollama", {}))
    base_url = str(req.get("base_url") or backend_cfg.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
    model = str(req.get("model") or backend_cfg.get("model") or "llama3.1:8b")
    timeout = float(req.get("timeout_sec") or config.get("timeout_sec") or 240)
    temperature = float(req.get("temperature") if req.get("temperature") is not None else config.get("temperature", 0.3))
    max_tokens = int(req.get("max_tokens") or config.get("max_tokens") or 2048)
    messages = normalize_messages(req.get("messages"), str(req.get("system_prompt") or config.get("system_prompt") or ""))
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if "think" in req:
        payload["think"] = req["think"]
    if req.get("format") is not None:
        payload["format"] = req["format"]

    started = time.time()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    last_raw: dict[str, Any] = {}
    for line in http_post_json_stream(f"{base_url}/api/chat", payload, timeout=timeout):
        raw = json.loads(line)
        last_raw = raw if isinstance(raw, dict) else {}
        message = last_raw.get("message") if isinstance(last_raw.get("message"), dict) else {}
        delta_content = str(message.get("content") or last_raw.get("response") or "")
        delta_reasoning = str(
            message.get("thinking")
            or message.get("reasoning")
            or message.get("reasoning_content")
            or last_raw.get("thinking")
            or last_raw.get("reasoning")
            or ""
        )
        if delta_content:
            content_parts.append(delta_content)
        if delta_reasoning:
            reasoning_parts.append(delta_reasoning)
        content, reasoning = split_stream_reasoning("".join(content_parts), "".join(reasoning_parts))
        yield {
            "ok": True,
            "backend": "ollama",
            "model": model,
            "content": content,
            "reasoning": reasoning,
            "done": bool(last_raw.get("done")),
            "elapsed_sec": round(time.time() - started, 3),
            "raw_summary": {
                "done": last_raw.get("done"),
                "done_reason": last_raw.get("done_reason"),
                "total_duration": last_raw.get("total_duration"),
                "prompt_eval_count": last_raw.get("prompt_eval_count"),
                "eval_count": last_raw.get("eval_count"),
            },
        }


def call_vllm(config: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    backend_cfg = dict(config.get("vllm", {}))
    base_url = str(req.get("base_url") or backend_cfg.get("base_url") or "http://127.0.0.1:8000/v1").rstrip("/")
    model = str(req.get("model") or backend_cfg.get("model") or "local-model")
    api_key = str(req.get("api_key") if req.get("api_key") is not None else backend_cfg.get("api_key", "EMPTY"))
    timeout = float(req.get("timeout_sec") or config.get("timeout_sec") or 240)
    temperature = float(req.get("temperature") if req.get("temperature") is not None else config.get("temperature", 0.3))
    max_tokens = resolve_vllm_max_tokens(config, req)
    messages = normalize_messages(req.get("messages"), str(req.get("system_prompt") or config.get("system_prompt") or ""))
    chat_template_kwargs = req.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = backend_cfg.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.time()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "include_reasoning": True,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    raw = http_post_json(
        f"{base_url}/chat/completions",
        payload,
        headers=headers,
        timeout=timeout,
    )
    choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    explicit_reasoning = str(
        message.get("reasoning")
        or message.get("reasoning_content")
        or message.get("thinking")
        or raw.get("reasoning")
        or ""
    )
    content, reasoning = split_reasoning(str(message.get("content") or ""), explicit_reasoning)
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return {
        "ok": True,
        "backend": "vllm",
        "model": model,
        "content": content,
        "reasoning": reasoning,
        "elapsed_sec": round(time.time() - started, 3),
        "raw_summary": {
            "finish_reason": first.get("finish_reason"),
            "usage": usage,
        },
    }


def stream_vllm(config: dict[str, Any], req: dict[str, Any]) -> Iterator[dict[str, Any]]:
    backend_cfg = dict(config.get("vllm", {}))
    base_url = str(req.get("base_url") or backend_cfg.get("base_url") or "http://127.0.0.1:8000/v1").rstrip("/")
    model = str(req.get("model") or backend_cfg.get("model") or "local-model")
    api_key = str(req.get("api_key") if req.get("api_key") is not None else backend_cfg.get("api_key", "EMPTY"))
    timeout = float(req.get("timeout_sec") or config.get("timeout_sec") or 240)
    temperature = float(req.get("temperature") if req.get("temperature") is not None else config.get("temperature", 0.3))
    max_tokens = resolve_vllm_max_tokens(config, req)
    messages = normalize_messages(req.get("messages"), str(req.get("system_prompt") or config.get("system_prompt") or ""))
    chat_template_kwargs = req.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = backend_cfg.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.time()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "include_reasoning": True,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    for line in http_post_json_stream(f"{base_url}/chat/completions", payload, headers=headers, timeout=timeout):
        if line.startswith(":"):
            continue
        data = line[5:].strip() if line.startswith("data:") else line
        if not data:
            continue
        if data == "[DONE]":
            content, reasoning = split_stream_reasoning("".join(content_parts), "".join(reasoning_parts))
            yield {
                "ok": True,
                "backend": "vllm",
                "model": model,
                "content": content,
                "reasoning": reasoning,
                "done": True,
                "elapsed_sec": round(time.time() - started, 3),
                "raw_summary": {"finish_reason": finish_reason},
            }
            return
        raw = json.loads(data)
        choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        finish_reason = first.get("finish_reason") or finish_reason
        delta_content = str(delta.get("content") or "")
        delta_reasoning = str(delta.get("reasoning") or delta.get("reasoning_content") or delta.get("thinking") or "")
        if delta_content:
            content_parts.append(delta_content)
        if delta_reasoning:
            reasoning_parts.append(delta_reasoning)
        content, reasoning = split_stream_reasoning("".join(content_parts), "".join(reasoning_parts))
        yield {
            "ok": True,
            "backend": "vllm",
            "model": model,
            "content": content,
            "reasoning": reasoning,
            "done": False,
            "elapsed_sec": round(time.time() - started, 3),
            "raw_summary": {"finish_reason": finish_reason},
        }

    content, reasoning = split_stream_reasoning("".join(content_parts), "".join(reasoning_parts))
    yield {
        "ok": True,
        "backend": "vllm",
        "model": model,
        "content": content,
        "reasoning": reasoning,
        "done": True,
        "elapsed_sec": round(time.time() - started, 3),
        "raw_summary": {"finish_reason": finish_reason},
    }


def check_health(config: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    backend = str(req.get("backend") or config.get("default_backend") or "ollama").lower()
    if backend == "ollama":
        base_url = str(req.get("base_url") or config.get("ollama", {}).get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        data = http_get_json(f"{base_url}/api/tags", timeout=10)
        models = [item.get("name") for item in data.get("models", []) if isinstance(item, dict)]
        return {"ok": True, "backend": "ollama", "base_url": base_url, "models": models[:100]}
    if backend == "vllm":
        base_url = str(req.get("base_url") or config.get("vllm", {}).get("base_url") or "http://127.0.0.1:8000/v1").rstrip("/")
        api_key = str(req.get("api_key") if req.get("api_key") is not None else config.get("vllm", {}).get("api_key", "EMPTY"))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        data = http_get_json(f"{base_url}/models", headers=headers, timeout=10)
        models = data.get("data", []) if isinstance(data.get("data"), list) else []
        return {"ok": True, "backend": "vllm", "base_url": base_url, "models": [m.get("id") for m in models if isinstance(m, dict)]}
    return {"ok": False, "error": f"Unsupported backend: {backend}"}


class LocalCustomGUIHandler(BaseHTTPRequestHandler):
    server_version = "LocalCustomGUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.send_file(ROOT / "index.html")
            return
        if self.path == "/api/config":
            self.send_json({"ok": True, "config": load_config()})
            return
        if self.path.startswith("/static/"):
            requested = self.path.split("?", 1)[0].removeprefix("/static/")
            safe = Path(requested).name
            self.send_file(STATIC_DIR / safe)
            return
        if self.path.startswith("/Icon/"):
            requested = self.path.split("?", 1)[0].removeprefix("/Icon/")
            safe = Path(requested).name
            self.send_file(ICON_DIR / safe)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = read_json_body(self)
            config = load_config()
            if self.path == "/api/config":
                new_config = deep_merge(config, payload)
                save_config(new_config)
                self.send_json({"ok": True, "config": new_config})
                return
            if self.path == "/api/health":
                self.send_json(check_health(config, payload))
                return
            if self.path == "/api/chat":
                backend = str(payload.get("backend") or config.get("default_backend") or "ollama").lower()
                if backend == "ollama":
                    self.send_json(call_ollama(config, payload))
                    return
                if backend == "vllm":
                    self.send_json(call_vllm(config, payload))
                    return
                self.send_json({"ok": False, "error": f"Unsupported backend: {backend}"}, status=400)
                return
            self.send_error(404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone local LLM custom GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--open", action="store_true", help="Open the GUI in the default browser")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_config()
    server = ThreadingHTTPServer((args.host, args.port), LocalCustomGUIHandler)

    def shutdown(_signum: int, _frame: Any) -> None:
        print("\nStopping local_customgui...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    url = f"http://{args.host}:{args.port}"
    print(f"local_customgui running: {url}")
    print("Press Ctrl+C to stop.")
    if args.open and not args.no_open:
        webbrowser.open(url)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
