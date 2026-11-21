"""Owner: S4. Single LLM entry point so providers are swappable in one place.

llm_complete(prompt, system="") -> str     raises LLMError on any failure (caller falls back to a template)
llm_available() -> bool                    True if a provider is configured (no network call)

.env settings:
  LLM_PROVIDER = none | anthropic | vertex | groq | ollama
  LLM_MODEL    = model id (Claude default: claude-opus-5-5; groq default: openai/gpt-oss-120b)
  LLM_EFFORT   = low | medium | high      (Claude effort / gpt-oss reasoning effort, default low)
  LLM_TIMEOUT  = seconds per call (default 30)
  groq:       GROQ_API_KEY (free key from console.groq.com). Free tier for gpt-oss-120b: 30 req/min, 1K req/day,
              8K tokens/min, 200K tokens/day. On HTTP 429 the call waits out `retry-after` if it is at most
              LLM_RATE_WAIT seconds (default 10), else it fails and the draft falls back to the template.
  anthropic:  ANTHROPIC_API_KEY
  vertex:     Claude on Google Cloud Vertex AI. No API key: sign in once with
              `gcloud auth application-default login`. ANTHROPIC_VERTEX_PROJECT_ID (else the gcloud
              default project), CLOUD_ML_REGION (default global), LLM_FALLBACK_MODEL (default
              claude-opus-4-8, must also be enabled in Model Garden; empty disables).
"""
from __future__ import annotations

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for tests
    pass

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_VERTEX_FALLBACK = "claude-opus-4-8"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# models that support refusal fallbacks (server-side on the Claude API, client-side middleware on Vertex)
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2"}

_clients: dict = {}
last_rate_wait_s = 0.0  # seconds the last llm_complete() call spent waiting out rate limits (for latency reports)


class LLMError(RuntimeError):
    pass


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "none").strip().lower()


def _timeout() -> float:
    return float(os.getenv("LLM_TIMEOUT", "30"))


def _model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_CLAUDE_MODEL


def default_model() -> str:
    """Model used when LLM_MODEL is empty, for the configured provider."""
    return {"groq": DEFAULT_GROQ_MODEL, "ollama": DEFAULT_OLLAMA_MODEL}.get(_provider(), DEFAULT_CLAUDE_MODEL)


def llm_available() -> bool:
    p = _provider()
    if p == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if p == "groq":
        return bool(os.getenv("GROQ_API_KEY", "").strip())
    return p in ("vertex", "ollama")  # vertex auth is checked on the first call (falls back if missing)


def _import_anthropic():
    try:
        import anthropic
        return anthropic
    except Exception as e:  # not installed, or blocked by OS policy
        raise LLMError(f"anthropic SDK unavailable: {e}") from e


def _get_client(provider: str):
    if provider in _clients:
        return _clients[provider]
    anthropic = _import_anthropic()
    if provider == "anthropic":
        client = anthropic.Anthropic(timeout=_timeout(), max_retries=1)
    else:  # vertex: Google Application Default Credentials, no API key
        fallback = os.getenv("LLM_FALLBACK_MODEL", DEFAULT_VERTEX_FALLBACK).strip()
        middleware = ([anthropic.BetaRefusalFallbackMiddleware([{"model": fallback}])]
                      if fallback and _model() in _FALLBACK_MODELS else None)
        client = anthropic.AnthropicVertex(region=os.getenv("CLOUD_ML_REGION") or "global",
                                           timeout=_timeout(), max_retries=1, middleware=middleware)
    _clients[provider] = client
    return client


def _claude(provider: str, prompt: str, system: str) -> str:
    anthropic = _import_anthropic()
    client = _get_client(provider)
    model = _model()
    kwargs = dict(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": os.getenv("LLM_EFFORT", "low")},
    )
    if system:
        kwargs["system"] = system
    if provider == "anthropic" and model in _FALLBACK_MODELS:  # server-side refusal fallback
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    try:
        # beta endpoint: carries the fallback params, and the Vertex refusal middleware hooks into it
        resp = client.beta.messages.create(**kwargs)
    except anthropic.APIError as e:
        raise LLMError(f"Claude API error ({provider}): {e}") from e
    except Exception as e:  # e.g. google.auth DefaultCredentialsError when gcloud login was skipped
        hint = " (run: gcloud auth application-default login)" if provider == "vertex" else ""
        raise LLMError(f"{type(e).__name__}: {e}{hint}") from e
    if resp.stop_reason == "refusal":
        raise LLMError("model declined the request")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise LLMError(f"empty response (stop_reason={resp.stop_reason})")
    return text


def _groq(prompt: str, system: str) -> str:
    global last_rate_wait_s
    import requests
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMError("GROQ_API_KEY is not set in .env")
    body = {
        "model": os.getenv("LLM_MODEL") or DEFAULT_GROQ_MODEL,
        # Groq's guidance for reasoning models: no system prompt, all instructions in the user message
        "messages": [{"role": "user", "content": f"{system}\n\n{prompt}" if system else prompt}],
        "reasoning_effort": os.getenv("LLM_EFFORT", "low"),
        "include_reasoning": False,  # the model's reasoning must never reach the reply
        "temperature": 0.6,
        "top_p": 0.95,
        "max_completion_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1500")),
    }
    max_wait = float(os.getenv("LLM_RATE_WAIT", "10"))
    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL, json=body, timeout=_timeout(),
                              headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException as e:
            raise LLMError(f"Groq connection error: {type(e).__name__}") from e
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after") or 2)
            if attempt < 2 and wait <= max_wait:
                last_rate_wait_s += wait
                time.sleep(wait)
                continue
            raise LLMError(f"Groq rate limit (HTTP 429, retry after {wait:.0f}s); "
                           "free tier: 8K tokens/min, 200K tokens/day")
        if r.status_code >= 400:
            raise LLMError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
        break
    try:
        choice = r.json()["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected Groq response: {type(e).__name__}") from e
    if not text:
        raise LLMError(f"empty response from Groq (finish_reason={choice.get('finish_reason')})")
    return text


def _ollama(prompt: str, system: str) -> str:
    import requests
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        r = requests.post(url, timeout=_timeout(), json={
            "model": os.getenv("LLM_MODEL") or DEFAULT_OLLAMA_MODEL,
            "messages": msgs, "stream": False, "options": {"temperature": 0.2}})
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
    except Exception as e:
        raise LLMError(f"Ollama error: {e}") from e
    if not text:
        raise LLMError("empty response from Ollama")
    return text


def llm_complete(prompt: str, system: str = "") -> str:
    global last_rate_wait_s
    last_rate_wait_s = 0.0
    p = _provider()
    if p in ("anthropic", "vertex"):
        return _claude(p, prompt, system)
    if p == "groq":
        return _groq(prompt, system)
    if p == "ollama":
        return _ollama(prompt, system)
    if p == "none":
        raise LLMError("LLM_PROVIDER not configured")
    raise LLMError(f"unsupported LLM_PROVIDER '{p}' (use anthropic, vertex, groq or ollama)")
