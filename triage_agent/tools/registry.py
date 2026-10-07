"""A tiny tool framework.

Each tool declares a name, a description, JSON-Schema parameters, and a ``run``
implementation. The registry exports the OpenAI function-calling schema and
dispatches calls by name. Adding a capability is: subclass ``Tool``, implement
``run``, register it — the agent loop never changes.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema describing the arguments

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool and return a JSON-serialisable result."""

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def dispatch(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Run a tool by name; always return a JSON string for the model.

        Tool failures are returned to the model as an ``error`` field rather than
        raised, so one bad call cannot crash the whole triage run.
        """
        if name not in self._tools:
            return json.dumps({"error": f"unknown tool '{name}'"})
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            result = self._tools[name].run(**args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model intentionally
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, default=str)
