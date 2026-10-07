"""Runtime configuration, resolved once from environment variables.

No secret is read at import time; the API key is only touched when the real
OpenAI client is constructed. A local ``.env`` is loaded automatically if
present (via python-dotenv) but is never required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    retriever: str = os.getenv("RETRIEVER", "lexical")  # "lexical" | "embedding"
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "8"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    use_mock: bool = os.getenv("USE_MOCK", "").strip().lower() in _TRUTHY

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


settings = Settings()
