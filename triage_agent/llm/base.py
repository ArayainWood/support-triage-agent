"""Provider-agnostic LLM interface used by the agent loop.

The agent only depends on these small data classes and the ``LLMClient``
protocol, so swapping OpenAI for a mock (or a future provider) never touches
orchestration logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A model's request to invoke one tool."""

    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class LLMResponse:
    """One assistant turn: zero or more tool calls, and/or prose content."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str | None = None


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
    ) -> LLMResponse:
        """Run one completion given the running transcript and tool schemas.

        ``tool_choice`` is "auto" normally, or a forced function spec when the
        agent needs to compel a specific tool (used to force the final submit).
        """
        ...
