"""LLM provider abstraction for AI Studio's two independent features (Chat and
Agent) — each keeps its own provider config (local Ollama model + CPU/GPU run
mode, or a cloud OpenAI-compatible endpoint) so, e.g., a small fast local
model can drive routine Chat while a stronger model is reserved for Agent
tasks, or vice versa. Mirrors the local/cloud + gpu_mode pattern already used
by meeting-intelligence/app/extraction.py.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
PROVIDER_FILE = BASE_DIR / "provider_config_studio.json"

_DEFAULT_SECTION = {
    "llm_mode": "local",  # "local" | "cloud"
    "llm_local": {"ollama_model": "llama3.1:8b", "gpu_mode": "auto"},  # gpu_mode: auto | cpu | gpu
    "llm_cloud": {"base_url": "https://api.openai.com/v1", "api_key": "", "model": ""},
}

DEFAULT_PROVIDER_CONFIG = {
    "chat": json.loads(json.dumps(_DEFAULT_SECTION)),
    "agent": json.loads(json.dumps(_DEFAULT_SECTION)),
}


def load_provider_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_PROVIDER_CONFIG))
    if PROVIDER_FILE.exists():
        try:
            saved = json.loads(PROVIDER_FILE.read_text(encoding="utf-8"))
            for section in ("chat", "agent"):
                s = saved.get(section, {})
                cfg[section]["llm_mode"] = s.get("llm_mode", cfg[section]["llm_mode"])
                cfg[section]["llm_local"].update(s.get("llm_local", {}))
                cfg[section]["llm_cloud"].update(s.get("llm_cloud", {}))
        except Exception:
            pass
    return cfg


def save_provider_config(cfg: dict):
    PROVIDER_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _ollama_base() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").split("/api/")[0]


async def list_ollama_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_ollama_base()}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


_pull_state: dict[str, dict] = {}  # model -> {"status", "percent", "done", "error"}


async def _run_pull(model: str):
    _pull_state[model] = {"status": "starting download…", "percent": 0, "done": False, "error": None}
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{_ollama_base()}/api/pull", json={"name": model, "stream": True}
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except ValueError:
                        continue
                    if evt.get("error"):
                        _pull_state[model] = {"status": evt["error"], "percent": 0, "done": True, "error": evt["error"]}
                        return
                    total, completed = evt.get("total") or 0, evt.get("completed") or 0
                    percent = round((completed / total) * 100) if total else _pull_state[model]["percent"]
                    _pull_state[model] = {"status": evt.get("status", ""), "percent": percent, "done": False, "error": None}
        _pull_state[model] = {"status": "success", "percent": 100, "done": True, "error": None}
    except Exception as e:
        _pull_state[model] = {"status": str(e), "percent": _pull_state[model].get("percent", 0), "done": True, "error": str(e)}


def start_pull(model: str) -> bool:
    """Kicks off `ollama pull <model>` in the background. Returns False if a
    pull for this exact model is already in flight (caller should treat that
    as 'already downloading', not an error)."""
    existing = _pull_state.get(model)
    if existing and not existing.get("done"):
        return False
    asyncio.create_task(_run_pull(model))
    return True


def get_pull_status(model: str) -> dict:
    return _pull_state.get(model, {"status": "not started", "percent": 0, "done": True, "error": None})


async def get_loaded_models() -> list[dict]:
    """Mirrors `ollama ps` — which locally-loaded models are currently
    resident and whether each is running on CPU, GPU, or a split."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_ollama_base()}/api/ps")
            resp.raise_for_status()
            out = []
            for m in resp.json().get("models", []):
                size = m.get("size", 0) or 0
                size_vram = m.get("size_vram", 0) or 0
                gpu_pct = round((size_vram / size) * 100) if size else 0
                out.append({"name": m.get("name") or m.get("model"), "gpu_percent": gpu_pct})
            return out
    except Exception:
        return []


async def unload_model(model: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(f"{_ollama_base()}/api/generate", json={"model": model, "keep_alive": 0})


async def model_supports_vision(model: str) -> bool:
    """Ollama reports per-model capabilities — only models listing "vision"
    can actually look at an attached image; everything else silently ignores
    the images field, which would look like the model hallucinating about a
    picture it never saw."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{_ollama_base()}/api/show", json={"model": model})
            resp.raise_for_status()
            return "vision" in (resp.json().get("capabilities") or [])
    except Exception:
        return False


def _gpu_options(model: str, gpu_mode: str) -> dict:
    options = {}
    if gpu_mode == "cpu" or (gpu_mode == "auto" and "gemma" in model.lower()):
        options["num_gpu"] = 0
    elif gpu_mode == "gpu":
        options["num_gpu"] = 999
    return options


# Attached images travel through the app as data: URIs (what the browser and
# the OpenAI-compatible APIs both use); Ollama instead wants bare base64, so
# each backend gets the messages reshaped to its own convention here.
def _strip_data_uri(uri: str) -> str:
    return uri.split(",", 1)[1] if uri.startswith("data:") else uri


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        msg = {"role": m["role"], "content": m.get("content", "")}
        if m.get("images"):
            msg["images"] = [_strip_data_uri(i) for i in m["images"]]
        out.append(msg)
    return out


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if not m.get("images"):
            out.append({"role": m["role"], "content": m.get("content", "")})
            continue
        parts = [{"type": "text", "text": m.get("content", "")}]
        parts += [{"type": "image_url", "image_url": {"url": i}} for i in m["images"]]
        out.append({"role": m["role"], "content": parts})
    return out


async def generate_local_chat(messages: list[dict], model: str, gpu_mode: str = "auto",
                               json_mode: bool = False, num_predict: int = 2048) -> str:
    """Multi-turn chat via Ollama's /api/chat (keeps role structure, unlike
    /api/generate's flat prompt string)."""
    options = {"temperature": 0.3, "num_predict": num_predict, **_gpu_options(model, gpu_mode)}
    payload = {"model": model, "messages": _to_ollama_messages(messages), "stream": False, "options": options}
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{_ollama_base()}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()


async def generate_cloud_chat(messages: list[dict], cfg: dict, json_mode: bool = False,
                               max_tokens: int = 2048) -> str:
    """Generic OpenAI-compatible chat-completions call — works with OpenAI,
    Azure OpenAI, Groq, OpenRouter, Together.ai, etc. by pointing base_url."""
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": cfg.get("model", ""),
        "messages": _to_openai_messages(messages),
        "stream": False,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def list_cloud_models(base_url: str, api_key: str) -> list[str]:
    """Model ids an OpenAI-compatible provider currently serves. Some providers
    (NVIDIA NIM among them) serve this list without a key at all."""
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("Base URL required")
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        resp.raise_for_status()
        return sorted(m.get("id", "") for m in resp.json().get("data", []) if m.get("id"))


async def generate_reply(section: str, messages: list[dict], json_mode: bool = False) -> str:
    """Dispatches Chat- or Agent-section messages to whichever provider is
    currently configured for that section."""
    cfg = load_provider_config()[section]
    if cfg["llm_mode"] == "cloud":
        return await generate_cloud_chat(messages, cfg["llm_cloud"], json_mode=json_mode)
    local = cfg["llm_local"]
    return await generate_local_chat(
        messages, local.get("ollama_model", "llama3.1:8b"),
        gpu_mode=local.get("gpu_mode", "auto"), json_mode=json_mode,
    )
