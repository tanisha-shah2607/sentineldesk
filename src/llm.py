"""Owner: S4. Single LLM entry point so providers are swappable in one place."""
import os

from dotenv import load_dotenv

load_dotenv()


def llm_complete(prompt: str, system: str = "") -> str:
    provider = os.getenv("LLM_PROVIDER", "none")
    if provider == "none":
        raise RuntimeError("LLM_PROVIDER not configured")
    # TODO(S4): implement anthropic / openai / ollama branches
    raise NotImplementedError(provider)
