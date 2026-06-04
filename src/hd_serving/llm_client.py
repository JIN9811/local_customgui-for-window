"""Thin local LLM helper around the existing standalone bridge functions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def call_local_llm(runtime: dict[str, Any], messages: list[dict[str, str]], system_prompt: str) -> dict[str, Any]:
    """Reuse root app.py bridge without making hd_serving depend on Streamlit."""
    from app import call_ollama, call_vllm, load_config

    config = load_config()
    req = {**runtime, "messages": messages, "system_prompt": system_prompt}
    backend = str(runtime.get("backend") or config.get("default_backend") or "ollama").lower()
    if backend == "ollama":
        return call_ollama(config, req)
    if backend == "vllm":
        return call_vllm(config, req)
    raise ValueError(f"Unsupported LLM backend: {backend}")


def stream_local_llm(runtime: dict[str, Any], messages: list[dict[str, str]], system_prompt: str) -> Iterator[dict[str, Any]]:
    """Stream local model output when the selected backend supports token streaming."""
    from app import load_config, stream_ollama, stream_vllm

    config = load_config()
    req = {**runtime, "messages": messages, "system_prompt": system_prompt}
    backend = str(runtime.get("backend") or config.get("default_backend") or "ollama").lower()
    if backend == "ollama":
        yield from stream_ollama(config, req)
        return
    if backend == "vllm":
        yield from stream_vllm(config, req)
        return
    raise ValueError(f"Unsupported LLM backend: {backend}")
