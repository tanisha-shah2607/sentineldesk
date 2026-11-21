"""S4: tests never call a live LLM, even when .env configures one (that would spend the team's
free-tier quota and make tests slow and flaky). src.llm is imported first so it loads .env once;
the overrides below then win for the whole test session. Tests that need a provider set it themselves."""
import os

import src.llm  # noqa: F401  (loads .env)

os.environ["LLM_PROVIDER"] = "none"
for _k in ("LLM_MODEL", "LLM_EFFORT", "LLM_RATE_WAIT", "LLM_MAX_OUTPUT_TOKENS", "LLM_MAX_PRECEDENTS",
           "LLM_MAX_ANSWER_CHARS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(_k, None)
