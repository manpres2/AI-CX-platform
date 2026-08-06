"""LLM-provider-agnostic task/decision extraction from a meeting transcript.

Independent from the two voice bots' LLM configuration on purpose: meeting
extraction is an admin-triggered batch operation (not a live conversational
turn), so it can afford a different, larger model, and the user wants to pick
and change that model from this app's own dashboard without touching either
bot's config. Local mode still talks to the same shared Ollama server the
bots use, just with its own model selection; cloud mode is a generic
OpenAI-compatible chat-completions call, same shape the bots already use.
"""

import json
import os
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
PROVIDER_FILE = BASE_DIR / "provider_config_meet.json"

DEFAULT_PROVIDER_CONFIG = {
    "llm_mode": "local",   # "local" | "cloud"
    "llm_local": {"ollama_model": "gemma4:e4b", "gpu_mode": "auto"},  # gpu_mode: "auto" | "cpu" | "gpu"
    "llm_cloud": {"base_url": "https://api.openai.com/v1", "api_key": "", "model": ""},
}


def load_provider_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_PROVIDER_CONFIG))  # deep copy of defaults
    if PROVIDER_FILE.exists():
        try:
            saved = json.loads(PROVIDER_FILE.read_text(encoding="utf-8"))
            cfg["llm_mode"] = saved.get("llm_mode", cfg["llm_mode"])
            cfg["llm_local"].update(saved.get("llm_local", {}))
            cfg["llm_cloud"].update(saved.get("llm_cloud", {}))
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


async def generate_local_reply(prompt: str, model: str, gpu_mode: str = "auto", json_mode: bool = False) -> str:
    # Ollama defaults to a small runtime context window regardless of what the
    # model actually supports, silently truncating the prompt (and mangling
    # JSON output) once a transcript exceeds it — size num_ctx to the actual
    # prompt instead of leaving it at Ollama's default for long meetings.
    num_ctx = min(max(len(prompt) // 3 + 2048, 4096), 65536)
    options = {"temperature": 0.1, "num_ctx": num_ctx, "num_predict": 4096}
    if gpu_mode == "cpu" or (gpu_mode == "auto" and "gemma" in model.lower()):
        # Gemma is kept off the GPU by default here — the 3060 Ti's 8GB VRAM is
        # too tight to reliably fit it alongside whatever else is using the
        # card, and a forced CPU run is more predictable than an unpredictable
        # GPU/CPU split. Admins can override this per-model from LLM Settings.
        options["num_gpu"] = 0
    elif gpu_mode == "gpu":
        # Force max GPU offload (all layers) rather than leaving it to
        # Ollama's automatic VRAM-fit heuristic.
        options["num_gpu"] = 999
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{_ollama_base()}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


async def get_loaded_models() -> list[dict]:
    """Mirrors `ollama ps` — which locally-loaded models are currently
    resident and whether they're running on CPU, GPU, or a split of both."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_ollama_base()}/api/ps")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            out = []
            for m in models:
                size = m.get("size", 0) or 0
                size_vram = m.get("size_vram", 0) or 0
                gpu_pct = round((size_vram / size) * 100) if size else 0
                out.append({"name": m.get("name") or m.get("model"), "gpu_percent": gpu_pct})
            return out
    except Exception:
        return []


async def unload_model(model: str):
    """Forces Ollama to drop `model` from memory immediately (keep_alive=0)
    instead of waiting for its idle timeout, so a changed CPU/GPU setting
    takes effect on the very next request rather than whenever it next
    naturally reloads."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(f"{_ollama_base()}/api/generate", json={"model": model, "keep_alive": 0})


async def generate_cloud_reply(prompt: str, cfg: dict, json_mode: bool = False) -> str:
    """Generic OpenAI-compatible chat-completions call — works with OpenAI, Azure
    OpenAI, Groq, OpenRouter, Together.ai, etc. by pointing base_url at them."""
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": cfg.get("model", ""),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
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


async def generate_extraction_reply(prompt: str, json_mode: bool = False) -> str:
    """Dispatches to whichever provider is currently configured for this app."""
    cfg = load_provider_config()
    if cfg["llm_mode"] == "cloud":
        return await generate_cloud_reply(prompt, cfg["llm_cloud"], json_mode=json_mode)
    return await generate_local_reply(
        prompt,
        cfg["llm_local"].get("ollama_model", "gemma4:e4b"),
        gpu_mode=cfg["llm_local"].get("gpu_mode", "auto"),
        json_mode=json_mode,
    )


EXTRACTION_PROMPT = """You are analyzing a meeting transcript to extract action items and decisions.
Read the transcript below (format: [MM:SS] Speaker: text).

For each action item, set "owner" to the person who actually COMMITTED to doing it — someone
who agreed to it themselves ("I'll do it", "I can take that", "leave it with me") or who was
assigned it by name and accepted ("Bob, can you handle the deck?" / "Sure, I'll get it done").
Do NOT set "owner" to someone who is merely mentioned, discussed, or referenced in the task
without themselves agreeing to act on it. If no one clearly commits, use null — do not guess.
{roster_line}Use the speaker name EXACTLY as it appears before the colon in the transcript —
never paraphrase, abbreviate, or invent a name that isn't one of the transcript's speakers.

Example:
[00:12] Alice: We need someone to update the pricing page before Friday.
[00:15] Bob: I can take that one.
-> {{"task": "Update the pricing page", "owner": "Bob", "deadline": null, "priority": "medium"}}
(Not "Alice" — she raised the task but never agreed to do it herself.)

Return ONLY a JSON object with this exact shape, no other text:
{{
  "tasks": [{{"task": "...", "owner": "name or null", "deadline": "YYYY-MM-DD or null", "priority": "low|medium|high"}}],
  "decisions": [{{"decision": "...", "timestamp_secs": 0}}]
}}
If a field is unknown, use null. If there are no tasks or decisions, return empty lists.

TRANSCRIPT:
{transcript_text}
"""


async def extract_tasks_decisions(transcript_text: str, speakers: list[str] | None = None) -> dict:
    roster_line = f"Known speakers in this transcript: {', '.join(speakers)}.\n" if speakers else ""
    prompt = EXTRACTION_PROMPT.format(transcript_text=transcript_text, roster_line=roster_line)
    try:
        raw = await generate_extraction_reply(prompt, json_mode=True)
    except Exception as e:
        return {"tasks": [], "decisions": [], "_parse_error": f"LLM call failed: {e}"}
    try:
        data = json.loads(raw)
        data.setdefault("tasks", [])
        data.setdefault("decisions", [])
        return data
    except Exception:
        return {"tasks": [], "decisions": [], "_parse_error": raw[:500]}
