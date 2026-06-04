"""Synchronous NemoClaw k3s vLLM runtime controls for the local custom GUI."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ManagedVLLMModel:
    deployment: str
    node_port: int
    persistent: bool = False
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    chat_template: str = ""
    reasoning_parser: str = ""
    default_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    kv_cache_memory_bytes: str = ""


def default_nemoclaw_vllm_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "cluster_container": "",
        "namespace": "",
        "node_host": "127.0.0.1",
        "startup_timeout_seconds": 1200,
        "models": {},
    }


class NemoClawVLLMRuntime:
    """Scale NemoClaw-hosted vLLM deployments from Streamlit button actions."""

    def __init__(
        self,
        *,
        enabled: bool,
        cluster_container: str,
        namespace: str,
        node_host: str = "auto",
        startup_timeout_s: float = 1200.0,
        models: dict[str, ManagedVLLMModel] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cluster_container = cluster_container
        self.namespace = namespace
        self.node_host = node_host.strip() or "auto"
        self.startup_timeout_s = float(startup_timeout_s)
        self.models = models or {}
        self._cached_node_host: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "NemoClawVLLMRuntime":
        cfg = {**default_nemoclaw_vllm_config(), **(cfg or {})}
        raw_models = cfg.get("models", {})
        models: dict[str, ManagedVLLMModel] = {}
        if isinstance(raw_models, dict):
            for alias, raw_item in raw_models.items():
                if not isinstance(raw_item, dict):
                    continue
                deployment = str(raw_item.get("deployment", "")).strip()
                node_port = raw_item.get("node_port")
                if not deployment or node_port is None:
                    continue
                depends_raw = raw_item.get("depends_on", [])
                depends_on = tuple(str(item).strip() for item in depends_raw if str(item).strip()) if isinstance(depends_raw, list) else ()
                models[str(alias)] = ManagedVLLMModel(
                    deployment=deployment,
                    node_port=int(node_port),
                    persistent=bool(raw_item.get("persistent", False)),
                    depends_on=depends_on,
                    chat_template=str(raw_item.get("chat_template") or "").strip(),
                    reasoning_parser=str(raw_item.get("reasoning_parser") or "").strip(),
                    default_chat_template_kwargs=(
                        raw_item.get("default_chat_template_kwargs")
                        if isinstance(raw_item.get("default_chat_template_kwargs"), dict)
                        else {}
                    ),
                    max_model_len=(
                        int(raw_item["max_model_len"])
                        if raw_item.get("max_model_len") not in (None, "")
                        else None
                    ),
                    gpu_memory_utilization=(
                        float(raw_item["gpu_memory_utilization"])
                        if raw_item.get("gpu_memory_utilization") not in (None, "")
                        else None
                    ),
                    kv_cache_memory_bytes=str(raw_item.get("kv_cache_memory_bytes") or "").strip(),
                )
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            cluster_container=str(cfg.get("cluster_container", "")),
            namespace=str(cfg.get("namespace", "")),
            node_host=str(cfg.get("node_host", "auto")),
            startup_timeout_s=float(cfg.get("startup_timeout_seconds", 1200)),
            models=models,
        )

    def base_url_for_model(self, model: str) -> str:
        managed = self._managed_model(model)
        node_host = self._resolve_node_host()
        return f"http://{node_host}:{managed.node_port}/v1"

    def load_model(self, model: str) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "model": model, "loaded": False}
        started = time.time()
        self.ensure_model(model)
        status = self.model_status(model)
        return {
            "enabled": True,
            "model": model,
            "loaded": bool(status.get("loaded")),
            "base_url": self.base_url_for_model(model),
            "elapsed_sec": round(time.time() - started, 3),
            "status": status,
        }

    def unload_model(self, model: str) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "model": model, "unloaded": False}
        started = time.time()
        managed = self._managed_model(model)
        self._scale_down_model(managed)
        return {
            "enabled": True,
            "model": model,
            "unloaded": True,
            "elapsed_sec": round(time.time() - started, 3),
            "status": self.model_status(model),
        }

    def ensure_model(self, model: str) -> None:
        managed = self._managed_model(model)
        self._scale_down_switchable_models_before(model)
        for dependency in managed.depends_on:
            if dependency in self.models:
                self.ensure_model(dependency)
        args_changed = self._ensure_deployment_runtime_args(managed)
        if args_changed and self._deployment_desired_replicas(managed.deployment) > 0:
            self._kubectl(
                "rollout",
                "status",
                f"deployment/{managed.deployment}",
                f"--timeout={int(self.startup_timeout_s)}s",
                timeout_s=self.startup_timeout_s + 30,
            )
            self._kubectl(
                "wait",
                "--for=condition=Available",
                f"deployment/{managed.deployment}",
                f"--timeout={int(self.startup_timeout_s)}s",
                timeout_s=self.startup_timeout_s + 30,
            )
        if self._deployment_available(managed.deployment):
            return
        if self._deployment_desired_replicas(managed.deployment) == 0:
            self._wait_for_deployment_pods_deleted(managed.deployment, timeout_s=300)
        self._kubectl("scale", "deployment", managed.deployment, "--replicas=1", timeout_s=60)
        self._kubectl(
            "wait",
            "--for=condition=Available",
            f"deployment/{managed.deployment}",
            f"--timeout={int(self.startup_timeout_s)}s",
            timeout_s=self.startup_timeout_s + 30,
        )

    def model_statuses(self) -> dict[str, Any]:
        items = []
        for model in self.models:
            items.append(self.model_status(model))
        return {"enabled": self.enabled, "models": items}

    def model_status(self, model: str) -> dict[str, Any]:
        managed = self._managed_model(model)
        status = self._deployment_status(managed.deployment)
        desired = int(status.get("desired_replicas", 0) or 0)
        available = int(status.get("available_replicas", 0) or 0)
        ready = int(status.get("ready_replicas", 0) or 0)
        state = "loaded" if available >= 1 else "loading" if desired >= 1 else "unloaded"
        return {
            "model": model,
            "deployment": managed.deployment,
            "node_port": managed.node_port,
            "persistent": managed.persistent,
            "chat_template": managed.chat_template,
            "reasoning_parser": managed.reasoning_parser,
            "default_chat_template_kwargs": managed.default_chat_template_kwargs,
            "max_model_len": managed.max_model_len,
            "gpu_memory_utilization": managed.gpu_memory_utilization,
            "kv_cache_memory_bytes": managed.kv_cache_memory_bytes,
            "desired_replicas": desired,
            "available_replicas": available,
            "ready_replicas": ready,
            "loaded": available >= 1,
            "state": state,
        }

    def _managed_model(self, model: str) -> ManagedVLLMModel:
        managed = self.models.get(model)
        if not self.enabled:
            raise RuntimeError("NemoClaw vLLM runtime is disabled.")
        if managed is None:
            raise ValueError(f"Unknown NemoClaw vLLM model: {model}")
        return managed

    def _scale_down_model(self, managed: ManagedVLLMModel) -> None:
        self._kubectl("scale", "deployment", managed.deployment, "--replicas=0", timeout_s=60)
        self._wait_for_deployment_pods_deleted(managed.deployment, timeout_s=300)

    def _scale_down_switchable_models_before(self, target_model: str) -> None:
        target = self.models.get(target_model)
        if target is None or target.persistent:
            return
        keep_models = {target_model, *target.depends_on}
        for model, managed in self.models.items():
            if model in keep_models or managed.persistent:
                continue
            if self._deployment_available(managed.deployment):
                self._scale_down_model(managed)

    def _deployment_desired_replicas(self, deployment: str) -> int:
        status = self._deployment_status(deployment)
        return int(status.get("desired_replicas", 0) or 0)

    def _deployment_available(self, deployment: str) -> bool:
        try:
            return int(self._deployment_status(deployment).get("available_replicas", 0) or 0) >= 1
        except Exception:
            return False

    def _deployment_status(self, deployment: str) -> dict[str, int]:
        payload = json.loads(self._kubectl("get", "deployment", deployment, "-o", "json", timeout_s=30))
        spec = payload.get("spec", {}) if isinstance(payload, dict) else {}
        status = payload.get("status", {}) if isinstance(payload, dict) else {}
        return {
            "desired_replicas": int(spec.get("replicas", 0) or 0),
            "available_replicas": int(status.get("availableReplicas", 0) or 0),
            "ready_replicas": int(status.get("readyReplicas", 0) or 0),
            "updated_replicas": int(status.get("updatedReplicas", 0) or 0),
            "unavailable_replicas": int(status.get("unavailableReplicas", 0) or 0),
        }

    def _deployment_args(self, deployment: str) -> list[str]:
        payload = json.loads(self._kubectl("get", "deployment", deployment, "-o", "json", timeout_s=30))
        spec = payload.get("spec", {}) if isinstance(payload, dict) else {}
        template = spec.get("template", {}) if isinstance(spec, dict) else {}
        pod_spec = template.get("spec", {}) if isinstance(template, dict) else {}
        containers = pod_spec.get("containers", []) if isinstance(pod_spec, dict) else []
        first = containers[0] if containers and isinstance(containers[0], dict) else {}
        args = first.get("args", []) if isinstance(first, dict) else []
        return [str(item) for item in args] if isinstance(args, list) else []

    def _without_arg_values(self, args: list[str], option_names: set[str], flag_names: set[str] | None = None) -> list[str]:
        flag_names = flag_names or set()
        filtered: list[str] = []
        skip_next = False
        for item in args:
            if skip_next:
                skip_next = False
                continue
            if item in flag_names:
                continue
            if item in option_names:
                skip_next = True
                continue
            filtered.append(item)
        return filtered

    def _ensure_deployment_runtime_args(self, managed: ManagedVLLMModel) -> bool:
        additions: list[str] = []
        if managed.chat_template:
            additions.extend(["--chat-template", managed.chat_template])
        if managed.reasoning_parser:
            additions.extend(["--reasoning-parser", managed.reasoning_parser])
        if managed.default_chat_template_kwargs:
            additions.extend(
                [
                    "--default-chat-template-kwargs",
                    json.dumps(managed.default_chat_template_kwargs, separators=(",", ":")),
                ]
            )
        if managed.max_model_len:
            additions.extend(["--max-model-len", str(managed.max_model_len)])
        if managed.gpu_memory_utilization:
            additions.extend(["--gpu-memory-utilization", f"{managed.gpu_memory_utilization:g}"])
        if managed.kv_cache_memory_bytes:
            additions.extend(["--kv-cache-memory-bytes", managed.kv_cache_memory_bytes])
        if not additions:
            return False

        current = self._deployment_args(managed.deployment)
        next_args = self._without_arg_values(
            current,
            {
                "--chat-template",
                "--reasoning-parser",
                "--default-chat-template-kwargs",
                "--max-model-len",
                "--gpu-memory-utilization",
                "--kv-cache-memory-bytes",
            },
            {"--enable-reasoning", "--no-enable-reasoning"},
        )
        next_args.extend(additions)
        if current == next_args:
            return False

        patch = json.dumps(
            [
                {
                    "op": "replace",
                    "path": "/spec/template/spec/containers/0/args",
                    "value": next_args,
                }
            ]
        )
        self._kubectl("patch", "deployment", managed.deployment, "--type=json", f"-p={patch}", timeout_s=60)
        return True

    def _deployment_label_selector(self, deployment: str) -> str:
        payload = json.loads(self._kubectl("get", "deployment", deployment, "-o", "json", timeout_s=30))
        spec = payload.get("spec", {}) if isinstance(payload, dict) else {}
        selector = spec.get("selector", {}) if isinstance(spec, dict) else {}
        labels = selector.get("matchLabels", {}) if isinstance(selector, dict) else {}
        if not isinstance(labels, dict) or not labels:
            return f"app={deployment}"
        return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))

    def _wait_for_deployment_pods_deleted(self, deployment: str, *, timeout_s: float) -> None:
        selector = self._deployment_label_selector(deployment)
        deadline = time.time() + timeout_s
        while True:
            payload = json.loads(self._kubectl("get", "pods", "-l", selector, "-o", "json", timeout_s=30))
            items = payload.get("items", []) if isinstance(payload, dict) else []
            if not items:
                return
            if time.time() >= deadline:
                names = [item.get("metadata", {}).get("name", "unknown") for item in items if isinstance(item, dict)]
                raise RuntimeError(f"Timed out waiting for {deployment} pods to terminate: {names}")
            time.sleep(2)

    def _resolve_node_host(self) -> str:
        if self.node_host.lower() != "auto":
            return self.node_host
        if self._cached_node_host:
            return self._cached_node_host
        node_host = self._run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                self.cluster_container,
            ],
            timeout_s=20,
        ).strip()
        if not node_host:
            raise RuntimeError(f"Could not resolve NemoClaw container IP: {self.cluster_container}")
        self._cached_node_host = node_host
        return node_host

    def _kubectl(self, *args: str, timeout_s: float) -> str:
        return self._run(["docker", "exec", self.cluster_container, "kubectl", "-n", self.namespace, *args], timeout_s=timeout_s)

    @staticmethod
    def _run(cmd: list[str], *, timeout_s: float) -> str:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
            raise RuntimeError(f"command failed: {' '.join(cmd)} :: {detail}")
        return result.stdout
