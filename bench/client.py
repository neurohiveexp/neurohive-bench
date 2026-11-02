"""Minimal client for OpenAI-compatible /chat/completions endpoints (OpenAI, OpenRouter, DeepSeek,
Ollama, llama.cpp server, vLLM, LM Studio and others). Standard library only."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class APIError(RuntimeError):
    pass


def chat(base_url: str, api_key: str | None, model: str, prompt: str,
         temperature: float | None = 0.0, max_tokens: int | None = None,
         timeout: int = 600, retries: int = 2) -> dict:
    """Send one user message. Returns {text, usage, latency_s, raw_model}."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode("utf-8")

    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            latency = time.monotonic() - started
            choice = payload["choices"][0]["message"]
            text = choice.get("content") or ""
            usage = payload.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            return {
                "text": text,
                "latency_s": round(latency, 3),
                "raw_model": payload.get("model", model),
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cached_tokens": details.get("cached_tokens", 0) or 0,
                },
            }
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            last_error = APIError(f"HTTP {e.code}: {msg}")
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last_error = APIError(f"{type(e).__name__}: {e}")
        time.sleep(5 * (attempt + 1))
    raise last_error
