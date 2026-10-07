"""Deterministic, offline LLM backend.

It emulates the tool-calling behaviour of a real model closely enough to exercise
the FULL agent loop with no API key and no network. This powers the test-suite
and lets anyone run the agent before wiring up credentials
(``python main.py --mock``).

It is a test double built from simple keyword heuristics — deliberately NOT a
second implementation of the product's intelligence.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import LLMResponse, ToolCall

_THAI = re.compile(r"[฀-๿]")
_CUSTOMER_ID = re.compile(r"customer_id:\s*(\S+)")
_TICKET_ID = re.compile(r"ticket_id:\s*(\S+)")


def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


class MockLLMClient:
    """Drives the agent loop deterministically for offline runs and tests."""

    _counter = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        gathered = {m.get("name") for m in messages if m.get("role") == "tool"}
        ticket_text = self._ticket_text(messages)
        customer = self._customer_record(messages)

        # Phase 1 — gather evidence if we have not already.
        if "lookup_customer_history" not in gathered:
            calls = [
                self._call(
                    "lookup_customer_history",
                    {"customer_id": self._field(messages, _CUSTOMER_ID, "unknown")},
                ),
                self._call(
                    "search_knowledge_base",
                    {"query": self._kb_query(ticket_text), "top_k": 3},
                ),
            ]
            if _has(ticket_text, "500", "error", "down", "can't access",
                    "cannot access", "เข้าไม่ได้", "outage", "operational"):
                region = (customer or {}).get("region") or "global"
                calls.append(
                    self._call("check_service_status", {"region": self._region_key(region)})
                )
            return LLMResponse(tool_calls=calls)

        # Phase 2 — decide.
        decision = self._decide(ticket_text, customer, messages)
        return LLMResponse(tool_calls=[self._call("submit_triage", decision)])

    # -- loop plumbing -------------------------------------------------------
    def _call(self, name: str, args: dict[str, Any]) -> ToolCall:
        MockLLMClient._counter += 1
        return ToolCall(
            id=f"mock_{MockLLMClient._counter}",
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        )

    @staticmethod
    def _field(messages, pattern, default) -> str:
        text = MockLLMClient._user_text(messages)
        m = pattern.search(text)
        return m.group(1) if m else default

    @staticmethod
    def _user_text(messages) -> str:
        return "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        )

    @staticmethod
    def _ticket_text(messages) -> str:
        text = MockLLMClient._user_text(messages)
        marker = "Conversation (chronological):"
        return text.split(marker, 1)[1] if marker in text else text

    @staticmethod
    def _tool_payloads(messages, name) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m.get("role") == "tool" and m.get("name") == name:
                try:
                    out.append(json.loads(m.get("content") or "{}"))
                except json.JSONDecodeError:
                    pass
        return out

    def _customer_record(self, messages) -> dict[str, Any] | None:
        payloads = self._tool_payloads(messages, "lookup_customer_history")
        return payloads[0] if payloads else None

    def _kb_refs(self, messages) -> list[dict[str, Any]]:
        payloads = self._tool_payloads(messages, "search_knowledge_base")
        if not payloads:
            return []
        refs = []
        for r in payloads[0].get("results", [])[:3]:
            refs.append(
                {"doc_id": r["doc_id"], "title": r["title"],
                 "relevance": float(r.get("relevance", 0.0))}
            )
        return refs

    # -- decision heuristics -------------------------------------------------
    @staticmethod
    def _kb_query(text: str) -> str:
        if _has(text, "500", "error", "เข้าไม่ได้", "operational"):
            return "error 500 cannot access outage regional incident"
        if _has(text, "charge", "payment", "refund", "upgrade", "card"):
            return "payment failed duplicate charges refund upgrade to pro"
        if _has(text, "dark mode", "appearance", "theme"):
            return "dark mode appearance settings theme"
        return text.strip()[:120]

    @staticmethod
    def _region_key(region: str) -> str:
        low = (region or "").lower()
        if "asia" in low or "thai" in low or "ap-" in low:
            return "asia"
        if "eu" in low:
            return "europe"
        if "us" in low or "america" in low:
            return "us"
        return "global"

    def _decide(self, text, customer, messages) -> dict[str, Any]:
        thai = bool(_THAI.search(text))
        plan = (customer or {}).get("plan", "unknown")
        kb = self._kb_refs(messages)
        status = self._tool_payloads(messages, "check_service_status")
        degraded = any(s.get("status") == "degraded" for s in status)

        outage = _has(text, "500", "error", "เข้าไม่ได้", "can't access",
                      "cannot access", "operational")
        billing = _has(text, "charge", "payment", "refund", "upgrade", "card", "dispute")
        feature = _has(text, "dark mode", "feature", "wondering", "no rush",
                       "would be cool", "schedule")

        if outage:
            urgency, action = "critical", "escalate_to_human"
            issue, area = "outage", "Platform availability"
            target = "Infrastructure / On-call"
            sentiment, trend = "frustrated", "escalating"
        elif billing:
            urgency, action = "critical", "escalate_to_human"
            issue, area = "billing", "Payments & subscriptions"
            target = "Billing specialist"
            sentiment, trend = "angry", "escalating"
        elif feature:
            urgency, action = "low", "auto_respond"
            issue, area = "feature_request", "Appearance / Settings"
            target = None
            sentiment, trend = "positive", "calm"
        else:
            urgency, action = "medium", "route_to_specialist"
            issue, area = "general", "General support"
            target = "Tier-1 support"
            sentiment, trend = "neutral", "stable"

        return {
            "ticket_id": self._field(messages, _TICKET_ID, "unknown"),
            "urgency": urgency,
            "urgency_reason": self._reason(urgency, issue, plan, degraded),
            "product_area": area,
            "issue_type": issue,
            "sentiment": sentiment,
            "sentiment_trend": trend,
            "key_facts": self._facts(issue, customer, degraded),
            "kb_references": kb,
            "recommended_action": action,
            "routing_target": target,
            "suggested_response": self._draft(thai, issue, degraded),
            "confidence": 0.7,
            "needs_human_review": urgency in ("critical", "high"),
        }

    @staticmethod
    def _reason(urgency, issue, plan, degraded) -> str:
        if issue == "outage":
            base = "Multiple users blocked by server errors"
            return base + (" and an internal check confirms a regional incident." if degraded
                           else "; needs infrastructure confirmation.")
        if issue == "billing":
            return "Customer charged multiple times without receiving the paid service and is threatening a dispute."
        if issue == "feature_request":
            return "Low-risk product question with positive tone and no urgency."
        return f"Standard {issue} request on the {plan} plan."

    @staticmethod
    def _facts(issue, customer, degraded) -> list[str]:
        facts = []
        if customer and customer.get("plan"):
            facts.append(f"Plan: {customer['plan']}" + (
                f", {customer['seats']} seats" if customer.get("seats") else ""))
        if issue == "outage":
            facts.append("User reports HTTP 500 across multiple browsers/devices.")
            if degraded:
                facts.append("Internal status: region degraded despite public page showing operational.")
        if issue == "billing":
            facts.append("Repeated charge attempts; service still not delivered.")
        return facts

    @staticmethod
    def _draft(thai, issue, degraded) -> str:
        if thai:
            if issue == "outage":
                return ("เราตรวจสอบแล้วพบว่ามีปัญหาในภูมิภาคเอเชีย ทีมวิศวกรกำลังเร่งแก้ไขอยู่ "
                        "ขออภัยในความไม่สะดวกอย่างยิ่งครับ เราได้ยกระดับเคสนี้เป็นเรื่องด่วนสูงสุด "
                        "และจะอัปเดตความคืบหน้าให้ทราบโดยเร็วที่สุด")
            return "ขอบคุณที่แจ้งเข้ามาครับ เรากำลังตรวจสอบและจะรีบติดต่อกลับโดยเร็วที่สุด"
        if issue == "billing":
            return ("I'm so sorry for the trouble. I can see multiple charge attempts and have "
                    "escalated this to our billing team as urgent. The pending charges have not "
                    "been captured and we will ensure any duplicates are reversed. We'll follow up shortly.")
        if issue == "feature_request":
            return ("Thanks for reaching out! A dark-mode toggle is rolling out gradually and may "
                    "not be on your account yet. I've noted your interest, plus your idea about "
                    "scheduling theme changes — I'll pass it to the product team.")
        return "Thanks for contacting us — we're looking into this and will get back to you shortly."
