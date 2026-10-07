"""Typed data contracts shared by the LLM, the tools and the CLI.

Keeping every shape in one place makes the boundary between components explicit
and gives us one source of truth to validate the model's output against.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Urgency(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Action(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ROUTE_TO_SPECIALIST = "route_to_specialist"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class Sentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class CustomerInfo(BaseModel):
    """Account context that travels with the ticket."""

    customer_id: str
    plan: str
    region: Optional[str] = None
    tenure: Optional[str] = None
    seats: Optional[int] = None
    notes: Optional[str] = None


class Ticket(BaseModel):
    """A support ticket: customer context plus the messages in time order."""

    id: str
    customer: CustomerInfo
    messages: list[str] = Field(..., description="Messages, oldest first.")

    def transcript(self) -> str:
        return "\n".join(f"[msg {i + 1}] {m}" for i, m in enumerate(self.messages))


class KBReference(BaseModel):
    doc_id: str
    title: str
    relevance: float = Field(..., ge=0.0, le=1.0)


class TriageResult(BaseModel):
    """The agent's final, structured decision for one ticket."""

    ticket_id: str
    urgency: Urgency
    urgency_reason: str = Field(..., description="One sentence justifying the urgency.")
    product_area: str = Field(
        ..., description="Product/feature area, free-form (e.g. 'Payments & subscriptions')."
    )
    issue_type: str = Field(
        ..., description="Issue category, free-form (e.g. billing, outage, bug, feature_request)."
    )
    sentiment: Sentiment
    sentiment_trend: str = Field(
        ..., description="How sentiment moves across the thread, free-form (e.g. escalating, stable, calm)."
    )
    key_facts: list[str] = Field(default_factory=list, description="Salient extracted facts.")
    kb_references: list[KBReference] = Field(default_factory=list)
    recommended_action: Action
    routing_target: Optional[str] = Field(
        None, description="Team/queue to route to, if any (e.g. 'Billing specialist')."
    )
    suggested_response: str = Field(
        ..., description="Draft reply for the customer, in the customer's own language."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool


def triage_json_schema() -> dict[str, Any]:
    """Flat JSON Schema for the terminal ``submit_triage`` tool.

    Hand-written (enums inlined, no ``$ref``) so it is accepted uniformly by the
    OpenAI function-calling API across versions; the result is still validated
    against :class:`TriageResult` afterwards.
    """
    return {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "urgency": {"type": "string", "enum": [u.value for u in Urgency]},
            "urgency_reason": {"type": "string"},
            "product_area": {"type": "string"},
            "issue_type": {"type": "string"},
            "sentiment": {"type": "string", "enum": [s.value for s in Sentiment]},
            "sentiment_trend": {"type": "string"},
            "key_facts": {"type": "array", "items": {"type": "string"}},
            "kb_references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "title": {"type": "string"},
                        "relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["doc_id", "title", "relevance"],
                },
            },
            "recommended_action": {"type": "string", "enum": [a.value for a in Action]},
            "routing_target": {"type": ["string", "null"]},
            "suggested_response": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "needs_human_review": {"type": "boolean"},
        },
        "required": [
            "ticket_id",
            "urgency",
            "urgency_reason",
            "product_area",
            "issue_type",
            "sentiment",
            "sentiment_trend",
            "recommended_action",
            "suggested_response",
            "confidence",
            "needs_human_review",
        ],
    }
