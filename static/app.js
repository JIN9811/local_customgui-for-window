const chatLog = document.getElementById("chat-log");
const messageInput = document.getElementById("message-input");
const btnSend = document.getElementById("btn-send");
const btnClear = document.getElementById("btn-clear");
const btnConnect = document.getElementById("btn-connect");
const btnHealth = document.getElementById("btn-health");
const btnSaveConfig = document.getElementById("btn-save-config");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");

const backendInput = document.getElementById("backend-input");
const baseUrlInput = document.getElementById("base-url-input");
const modelInput = document.getElementById("model-input");
const apiKeyInput = document.getElementById("api-key-input");
const temperatureInput = document.getElementById("temperature-input");
const maxTokensInput = document.getElementById("max-tokens-input");
const systemPromptInput = document.getElementById("system-prompt-input");
const assistantIconUrl = "/Icon/hd_hyundai_streamlit_avatar.svg";
const userIconUrl = "/Icon/user_person_gear_navy.svg";

const defaultBackendConfig = {
  ollama: { base_url: "http://127.0.0.1:11434", model: "gemma4:e2b", api_key: "" },
  vllm: { base_url: "http://127.0.0.1:8000/v1", model: "local-model", api_key: "EMPTY" },
};

let config = {};
let messages = [];
let busy = false;

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(label, state = "idle") {
  statusLabel.textContent = label;
  statusDot.className = `dot ${state}`;
}

function setBusy(value) {
  busy = value;
  btnSend.disabled = value;
  btnConnect.disabled = value;
  btnHealth.disabled = value;
  btnSaveConfig.disabled = value;
  setStatus(value ? "REASONING" : "READY", value ? "busy" : "idle");
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || data.detail || `HTTP ${response.status}`);
  }
  return data;
}

function activeBackendConfig() {
  const backend = backendInput.value || "ollama";
  return {
    backend,
    base_url: baseUrlInput.value.trim() || defaultBackendConfig[backend].base_url,
    model: modelInput.value.trim() || defaultBackendConfig[backend].model,
    api_key: apiKeyInput.value,
    temperature: Number(temperatureInput.value || 0.3),
    max_tokens: Number(maxTokensInput.value || 2048),
    system_prompt: systemPromptInput.value,
  };
}

function setModelOptions(models, selectedModel = "", { preserveSelected = true } = {}) {
  const selected = String(selectedModel || "").trim();
  const unique = [];
  for (const item of models || []) {
    const model = String(item || "").trim();
    if (model && !unique.includes(model)) unique.push(model);
  }
  if (preserveSelected && selected && !unique.includes(selected)) unique.unshift(selected);
  if (!unique.length) {
    const backend = backendInput.value || "ollama";
    unique.push(defaultBackendConfig[backend].model);
  }
  modelInput.innerHTML = "";
  for (const model of unique) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelInput.appendChild(option);
  }
  modelInput.value = selected && unique.includes(selected) ? selected : unique[0];
}

function applyConfig(nextConfig) {
  config = nextConfig || {};
  const backend = config.default_backend || "ollama";
  backendInput.value = backend;
  temperatureInput.value = Number(config.temperature ?? 0.3);
  maxTokensInput.value = Number(config.max_tokens ?? 2048);
  systemPromptInput.value = config.system_prompt || "";
  applyBackendFields();
}

function applyBackendFields() {
  const backend = backendInput.value || "ollama";
  const backendConfig = { ...defaultBackendConfig[backend], ...(config[backend] || {}) };
  baseUrlInput.value = backendConfig.base_url || defaultBackendConfig[backend].base_url;
  setModelOptions([], backendConfig.model || defaultBackendConfig[backend].model);
  apiKeyInput.value = backend === "vllm" ? (backendConfig.api_key || "EMPTY") : "";
  apiKeyInput.disabled = backend !== "vllm";
}

function renderReasoning(reasoning, pending = false) {
  if (pending) {
    return `
      <div class="reasoning-pending">
        <span class="spinner"></span>
        <span>reasoning</span>
      </div>
    `;
  }
  if (!reasoning) return "";
  return `
    <details class="reasoning-block">
      <summary><span class="reasoning-dot"></span>reasoning 보기 / 닫기</summary>
      <pre>${escapeHtml(reasoning)}</pre>
    </details>
  `;
}

function renderAvatar(role) {
  if (role === "assistant") return `<img class="message-avatar" src="${assistantIconUrl}" alt="Assistant" />`;
  if (role === "user") return `<img class="message-avatar" src="${userIconUrl}" alt="User" />`;
  return "";
}

function renderMessages() {
  if (!messages.length) {
    chatLog.innerHTML = `
      <article class="message assistant">
        ${renderAvatar("assistant")}
        <small>Assistant</small>
        <div class="content">Ollama 또는 vLLM endpoint를 선택하고 메시지를 보내세요. Settings는 <code>config.json</code>에 저장됩니다.</div>
      </article>
    `;
    return;
  }
  chatLog.innerHTML = messages.map((message) => {
    const role = message.role === "user" ? "user" : message.role === "system" ? "system" : "assistant";
    const label = role === "user" ? "You" : role === "system" ? "System" : `${message.backend || "Assistant"}${message.model ? ` · ${message.model}` : ""}`;
    const content = message.pending ? "응답을 준비하고 있습니다." : escapeHtml(message.content || "").replaceAll("\n", "<br />");
    return `
      <article class="message ${role}">
        ${renderAvatar(role)}
        <small>${escapeHtml(label)}</small>
        ${renderReasoning(message.reasoning, message.pending)}
        <div class="content">${content}</div>
        ${message.elapsed_sec ? `<div class="meta">${message.elapsed_sec}s</div>` : ""}
      </article>
    `;
  }).join("");
  chatLog.scrollTop = chatLog.scrollHeight;
}

function apiMessages() {
  return messages
    .filter((message) => !message.pending && (message.role === "user" || message.role === "assistant"))
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: message.content,
    }));
}

async function sendMessage() {
  const content = messageInput.value.trim();
  if (!content || busy) return;
  messageInput.value = "";
  const runtime = activeBackendConfig();
  messages.push({ role: "user", content });
  const pending = { role: "assistant", pending: true, backend: runtime.backend, model: runtime.model };
  messages.push(pending);
  renderMessages();
  setBusy(true);
  try {
    const data = await apiJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        ...runtime,
        messages: apiMessages().slice(0, -1),
      }),
    });
    Object.assign(pending, {
      pending: false,
      content: data.content || "",
      reasoning: data.reasoning || "",
      backend: data.backend,
      model: data.model,
      elapsed_sec: data.elapsed_sec,
      raw_summary: data.raw_summary,
    });
    setStatus("READY", "idle");
  } catch (err) {
    Object.assign(pending, {
      pending: false,
      role: "system",
      content: `LLM call failed: ${err.message}`,
    });
    setStatus("ERROR", "warn");
  } finally {
    setBusy(false);
    renderMessages();
  }
}

async function loadConfig() {
  const data = await apiJson("/api/config");
  applyConfig(data.config || {});
  renderMessages();
}

async function saveConfig() {
  const runtime = activeBackendConfig();
  const backend = runtime.backend;
  const next = {
    default_backend: backend,
    temperature: runtime.temperature,
    max_tokens: runtime.max_tokens,
    system_prompt: runtime.system_prompt,
    [backend]: {
      base_url: runtime.base_url,
      model: runtime.model,
    },
  };
  if (backend === "vllm") {
    next.vllm.api_key = runtime.api_key || "EMPTY";
  }
  const data = await apiJson("/api/config", {
    method: "POST",
    body: JSON.stringify(next),
  });
  applyConfig(data.config || {});
  messages.push({ role: "system", content: "Config saved." });
  renderMessages();
}

async function connectModels({ pushMessage = true } = {}) {
  const runtime = activeBackendConfig();
  setBusy(true);
  try {
    const data = await apiJson("/api/health", {
      method: "POST",
      body: JSON.stringify(runtime),
    });
    setModelOptions(Array.isArray(data.models) ? data.models : [], runtime.model, { preserveSelected: false });
    const models = Array.isArray(data.models) && data.models.length ? `\nmodels: ${data.models.slice(0, 12).join(", ")}` : "";
    if (pushMessage) {
      messages.push({ role: "system", content: `${data.backend} connected: ${data.base_url}${models}` });
    }
    setStatus("READY", "idle");
    return data;
  } catch (err) {
    if (pushMessage) {
      messages.push({ role: "system", content: `Connect failed: ${err.message}` });
    }
    setStatus("ERROR", "warn");
    throw err;
  } finally {
    setBusy(false);
    renderMessages();
  }
}

async function healthCheck() {
  return connectModels({ pushMessage: true });
}

backendInput.addEventListener("change", applyBackendFields);
btnConnect.addEventListener("click", () => connectModels({ pushMessage: true }).catch(() => {}));
btnSend.addEventListener("click", sendMessage);
btnClear.addEventListener("click", () => {
  messages = [];
  renderMessages();
});
btnHealth.addEventListener("click", healthCheck);
btnSaveConfig.addEventListener("click", () => saveConfig().catch((err) => {
  messages.push({ role: "system", content: `Config save failed: ${err.message}` });
  renderMessages();
}));

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

loadConfig().catch((err) => {
  setStatus("ERROR", "warn");
  messages.push({ role: "system", content: `Config load failed: ${err.message}` });
  renderMessages();
});
