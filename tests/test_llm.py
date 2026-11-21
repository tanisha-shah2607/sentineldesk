"""S4: LLM provider switch. No network: the SDK client and HTTP calls are faked."""
import json
from types import SimpleNamespace

import pytest
import requests

import src.llm as llm


class FakeAPIError(Exception):
    pass


def fake_setup(monkeypatch, provider, model="claude-opus-5-5", reply="Hello [TKT-00001]", stop="end_turn"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=reply)])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setattr(llm, "_import_anthropic", lambda: SimpleNamespace(APIError=FakeAPIError))
    monkeypatch.setattr(llm, "_get_client", lambda p: client)
    return calls


def test_vertex_never_sends_server_side_fallbacks(monkeypatch):
    calls = fake_setup(monkeypatch, "vertex")
    assert llm.llm_complete("prompt", "system") == "Hello [TKT-00001]"
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "system"
    assert "fallbacks" not in kw and "betas" not in kw   # Vertex rejects server-side fallbacks


def test_anthropic_opus_uses_server_side_fallback(monkeypatch):
    calls = fake_setup(monkeypatch, "anthropic")
    llm.llm_complete("prompt")
    assert calls[0]["fallbacks"] == "default" and "system" not in calls[0]


def test_anthropic_haiku_has_no_fallback(monkeypatch):
    calls = fake_setup(monkeypatch, "anthropic", model="claude-haiku-5-5")
    llm.llm_complete("prompt")
    assert "fallbacks" not in calls[0]


def test_refusal_raises(monkeypatch):
    fake_setup(monkeypatch, "vertex", stop="refusal")
    with pytest.raises(llm.LLMError):
        llm.llm_complete("prompt")


def test_auth_error_becomes_llm_error_with_hint(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("Could not automatically determine credentials")
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setattr(llm, "_import_anthropic", lambda: SimpleNamespace(APIError=FakeAPIError))
    monkeypatch.setattr(llm, "_get_client", lambda p: SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=boom))))
    with pytest.raises(llm.LLMError, match="application-default login"):
        llm.llm_complete("prompt")


def test_provider_availability(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for provider, expected in [("none", False), ("anthropic", False), ("groq", False), ("vertex", True),
                               ("ollama", True)]:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert llm.llm_available() is expected
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert llm.llm_available() is True


def test_unconfigured_and_unknown_providers_raise(monkeypatch):
    for provider in ("none", "gemini"):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        with pytest.raises(llm.LLMError):
            llm.llm_complete("prompt")


# ---- Groq (gpt-oss-120b) -----------------------------------------------------------------------

class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


OK_PAYLOAD = {"choices": [{"message": {"content": "Hello [TKT-00001]", "reasoning": "internal thoughts"},
                           "finish_reason": "stop"}]}


def groq_env(monkeypatch, key="test-key"):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", key)
    for k in ("LLM_MODEL", "LLM_EFFORT", "LLM_RATE_WAIT"):
        monkeypatch.delenv(k, raising=False)


def test_groq_request_shape_and_no_reasoning_leak(monkeypatch):
    groq_env(monkeypatch)
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return FakeResp(200, OK_PAYLOAD)

    monkeypatch.setattr(requests, "post", post)
    assert llm.default_model() == "openai/gpt-oss-120b"
    assert llm.llm_complete("PROMPT", "SYSTEM") == "Hello [TKT-00001]"   # the reasoning field is never returned
    url, body, headers = calls[0]
    assert url == llm.GROQ_URL and headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "openai/gpt-oss-120b" and body["include_reasoning"] is False
    assert body["reasoning_effort"] == "low" and "reasoning_format" not in body
    assert [m["role"] for m in body["messages"]] == ["user"]               # Groq: no system prompt
    assert body["messages"][0]["content"].startswith("SYSTEM") and "PROMPT" in body["messages"][0]["content"]


def test_groq_waits_out_short_rate_limit(monkeypatch):
    groq_env(monkeypatch)
    responses = [FakeResp(429, headers={"retry-after": "1"}), FakeResp(200, OK_PAYLOAD)]
    monkeypatch.setattr(requests, "post", lambda *a, **k: responses.pop(0))
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    assert llm.llm_complete("p") == "Hello [TKT-00001]"
    assert slept == [1.0] and llm.last_rate_wait_s == 1.0


def test_groq_long_rate_limit_fails_fast(monkeypatch):
    groq_env(monkeypatch)
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResp(429, headers={"retry-after": "45"})

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(llm.LLMError, match="rate limit"):
        llm.llm_complete("p")
    assert len(calls) == 1   # longer than LLM_RATE_WAIT (10s): no waiting during a live demo


def test_groq_missing_key_http_error_and_empty_reply(monkeypatch):
    groq_env(monkeypatch, key="")
    assert llm.llm_available() is False
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        llm.llm_complete("p")
    groq_env(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(401, {"error": {"message": "Invalid API Key"}}))
    with pytest.raises(llm.LLMError, match="401"):
        llm.llm_complete("p")
    empty = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(200, empty))
    with pytest.raises(llm.LLMError, match="length"):
        llm.llm_complete("p")
