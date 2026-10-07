"""Real LLM backend: OpenAI Chat Completions with function calling."""
from __future__ import annotations

from typing import Any

from ..config import settings
from .base import LLMResponse, ToolCall


class OpenAIClient:
    """Thin adapter translating our LLMResponse to/from the OpenAI SDK.

    The API key is read from the ``OPENAI_API_KEY`` environment variable by the
    SDK itself — it is never passed around in code or logged.
    """

    def __init__(self, model: str | None = None, temperature: float | None = None):
        from openai import OpenAI  # lazy import so the mock path needs no SDK/key

        self._client = OpenAI()
        self._model = model or settings.model
        self._temperature = settings.temperature if temperature is None else temperature

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self._temperature,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(tool_calls=calls, content=msg.content)
