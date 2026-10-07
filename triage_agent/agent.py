"""The triage agent: an evidence-gathering loop that ends in one decision.

    system + ticket                      -> model
    while the model requests tools:         run them, feed results back
    when the model calls submit_triage:     validate -> TriageResult

``submit_triage`` is a terminal tool typed by the TriageResult schema, so the
model is forced to return a complete, structured decision rather than prose.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import settings
from .llm.base import LLMClient, ToolCall
from .prompts import SYSTEM_PROMPT
from .schemas import Ticket, TriageResult, triage_json_schema
from .tools.registry import Tool, ToolRegistry

SUBMIT_TOOL_NAME = "submit_triage"


def _clamp01(value: Any) -> Any:
    """Clamp a value to [0, 1] when it is numeric; otherwise leave it unchanged.

    A real model occasionally emits an out-of-range score (e.g. a 0-100 relevance).
    Clamping here keeps one stray number from failing validation of the whole result.
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return value


def _submit_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_TOOL_NAME,
            "description": (
                "Submit the final triage decision. Call this EXACTLY ONCE, only "
                "after gathering the evidence you need."
            ),
            "parameters": triage_json_schema(),
        },
    }


class TriageAgent:
    def __init__(self, llm: LLMClient, tools: list[Tool]):
        self._llm = llm
        self._registry = ToolRegistry(tools)
        self._tool_schemas = self._registry.schemas() + [_submit_tool_schema()]

    def run(self, ticket: Ticket) -> TriageResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_ticket(ticket)},
        ]

        for _ in range(settings.max_agent_steps):
            response = self._llm.complete(messages, self._tool_schemas)

            if not response.tool_calls:
                # Model replied in prose; nudge it toward the terminal tool.
                messages.append({"role": "assistant", "content": response.content or ""})
                messages.append(
                    {"role": "user", "content": f"Please finish by calling {SUBMIT_TOOL_NAME}."}
                )
                continue

            messages.append(self._assistant_tool_turn(response.tool_calls))

            for call in response.tool_calls:
                if call.name == SUBMIT_TOOL_NAME:
                    return self._finalize(ticket, call.arguments)
                result = self._registry.dispatch(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result,
                    }
                )

        raise RuntimeError(
            f"Agent did not submit a decision within {settings.max_agent_steps} steps."
        )

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _assistant_tool_turn(tool_calls: list[ToolCall]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in tool_calls
            ],
        }

    def _finalize(self, ticket: Ticket, arguments: str) -> TriageResult:
        try:
            data = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"submit_triage returned invalid JSON: {exc}") from exc
        data.setdefault("ticket_id", ticket.id)
        # Defensively clamp scores so one out-of-range value does not fail the ticket.
        if "confidence" in data:
            data["confidence"] = _clamp01(data["confidence"])
        for ref in data.get("kb_references") or []:
            if isinstance(ref, dict) and "relevance" in ref:
                ref["relevance"] = _clamp01(ref["relevance"])
        try:
            return TriageResult.model_validate(data)
        except ValidationError as exc:
            raise RuntimeError(f"submit_triage returned invalid data: {exc}") from exc

    @staticmethod
    def _format_ticket(ticket: Ticket) -> str:
        c = ticket.customer
        lines = [
            "Triage this support ticket.",
            "",
            f"ticket_id: {ticket.id}",
            f"customer_id: {c.customer_id}",
            f"plan: {c.plan}",
        ]
        if c.region:
            lines.append(f"region: {c.region}")
        if c.seats:
            lines.append(f"seats: {c.seats}")
        if c.tenure:
            lines.append(f"tenure: {c.tenure}")
        if c.notes:
            lines.append(f"notes: {c.notes}")
        lines += ["", "Conversation (chronological):", ticket.transcript()]
        return "\n".join(lines)
