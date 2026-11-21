"""S4: LLM provider switch. No network: the SDK client is faked."""
from types import SimpleNamespace

import pytest

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
    for provider, expected in [("none", False), ("anthropic", False), ("vertex", True), ("ollama", True)]:
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
