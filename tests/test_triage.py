"""End-to-end tests over the three sample tickets using the offline mock backend.

These run with no API key and no network, so they are safe for CI and for a
reviewer to run immediately after `pip install -r requirements.txt`.
"""
from __future__ import annotations

import re

import pytest

from triage_agent.agent import TriageAgent
from triage_agent.cli import load_tickets
from triage_agent.llm.mock_client import MockLLMClient
from triage_agent.schemas import (
    Action,
    Sentiment,
    TriageResult,
    Urgency,
    triage_json_schema,
)
from triage_agent.tools import default_tools
from triage_agent.tools.customer_history import CustomerHistoryTool
from triage_agent.tools.knowledge_base import KnowledgeBaseTool
from triage_agent.tools.service_status import ServiceStatusTool

_THAI = re.compile(r"[฀-๿]")


@pytest.fixture
def agent() -> TriageAgent:
    return TriageAgent(MockLLMClient(), default_tools())


@pytest.fixture
def tickets():
    return {t.id: t for t in load_tickets()}


def test_all_tickets_produce_valid_results(agent, tickets):
    for ticket in tickets.values():
        result = agent.run(ticket)
        assert isinstance(result, TriageResult)
        assert result.ticket_id == ticket.id
        assert result.suggested_response.strip()
        assert 0.0 <= result.confidence <= 1.0


def test_t1_billing_is_critical_and_escalated(agent, tickets):
    result = agent.run(tickets["t1"])
    assert result.urgency is Urgency.CRITICAL
    assert result.recommended_action is Action.ESCALATE_TO_HUMAN
    assert result.sentiment is Sentiment.ANGRY
    assert result.needs_human_review is True


def test_t2_outage_is_critical_and_replies_in_thai(agent, tickets):
    result = agent.run(tickets["t2"])
    assert result.urgency is Urgency.CRITICAL
    assert result.issue_type == "outage"
    # The customer wrote in Thai, so the draft reply must be in Thai.
    assert _THAI.search(result.suggested_response)
    # The status-reconciliation trap must fire: the agent detects the Asia incident
    # even though the public status page says "operational".
    assert "regional incident" in result.urgency_reason.lower() or any(
        "degraded" in fact.lower() or "operational" in fact.lower()
        for fact in result.key_facts
    )


def test_t3_feature_request_is_low_and_auto_respond(agent, tickets):
    result = agent.run(tickets["t3"])
    assert result.urgency is Urgency.LOW
    assert result.recommended_action is Action.AUTO_RESPOND
    assert result.sentiment is Sentiment.POSITIVE


def test_knowledge_base_returns_relevant_docs():
    kb = KnowledgeBaseTool()
    hits = kb.run(query="duplicate pending charges refund", top_k=3)["results"]
    assert hits, "expected at least one KB hit"
    assert any(h["doc_id"] in {"KB-102", "KB-401"} for h in hits)


def test_service_status_flags_asia_incident():
    status = ServiceStatusTool().run(region="asia")
    assert status["status"] == "degraded"
    assert status["public_status_page"] == "operational"


def test_unknown_customer_is_handled_gracefully():
    record = CustomerHistoryTool().run(customer_id="does_not_exist")
    assert record["found"] is False


def test_submit_schema_stays_in_sync_with_model():
    """Guard against drift between the pydantic model and the hand-written
    function schema (the two sources of truth for the submit_triage contract)."""
    schema = triage_json_schema()
    model_fields = set(TriageResult.model_fields.keys())
    assert set(schema["properties"].keys()) == model_fields
    required_model = {
        name for name, field in TriageResult.model_fields.items() if field.is_required()
    }
    assert set(schema["required"]) == required_model
