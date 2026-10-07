"""Command-line entry point.

    python main.py                 # triage every sample ticket
    python main.py --ticket t2     # triage one ticket by id
    python main.py --mock          # offline deterministic backend (no API key)
    python main.py --json          # raw JSON instead of a formatted report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DATA_DIR, settings
from .schemas import Ticket, TriageResult
from .tools import default_tools


def _force_utf8_output() -> None:
    """Print UTF-8 regardless of the OS console codepage.

    Windows consoles often default to a legacy codepage (e.g. cp874/cp1252) that
    cannot encode Thai text or box-drawing characters. Reconfiguring to UTF-8
    keeps the agent's output readable on every platform.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - older/odd streams
            pass


def build_agent(use_mock: bool):
    from .agent import TriageAgent

    tools = default_tools(use_mock)
    if use_mock:
        from .llm.mock_client import MockLLMClient

        llm = MockLLMClient()
    else:
        from .llm.openai_client import OpenAIClient

        llm = OpenAIClient()
    return TriageAgent(llm, tools)


def load_tickets(path: Path | None = None) -> list[Ticket]:
    path = path or (DATA_DIR / "sample_tickets.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Ticket.model_validate(t) for t in raw]


def _print_report(result: TriageResult) -> None:
    bar = "─" * 70
    print(bar)
    print(
        f"TICKET {result.ticket_id}   URGENCY={result.urgency.value.upper()}"
        f"   ACTION={result.recommended_action.value}"
    )
    print(bar)
    print(f"  Issue        : {result.issue_type}  ({result.product_area})")
    print(f"  Sentiment    : {result.sentiment.value}  (trend: {result.sentiment_trend})")
    print(f"  Why urgent   : {result.urgency_reason}")
    if result.routing_target:
        print(f"  Route to     : {result.routing_target}")
    print(
        f"  Confidence   : {result.confidence:.0%}"
        f"   human review: {'yes' if result.needs_human_review else 'no'}"
    )
    if result.key_facts:
        print("  Key facts    :")
        for fact in result.key_facts:
            print(f"      - {fact}")
    if result.kb_references:
        print("  KB matches   :")
        for ref in result.kb_references:
            print(f"      - [{ref.doc_id}] {ref.title}  (rel {ref.relevance:.2f})")
    print("  Suggested reply:")
    for line in (result.suggested_response.splitlines() or [""]):
        print(f"      {line}")
    print()


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = argparse.ArgumentParser(description="Support Ticket Triage Agent")
    parser.add_argument("--ticket", help="Triage only the ticket with this id.")
    parser.add_argument(
        "--mock", action="store_true", help="Use the offline deterministic backend."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print raw JSON instead of a report."
    )
    args = parser.parse_args(argv)

    use_mock = args.mock or settings.use_mock
    if not use_mock and not settings.has_api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "  - Put it in a .env file (copy .env.example), or\n"
            "  - run with --mock to use the offline backend.",
            file=sys.stderr,
        )
        return 2

    tickets = load_tickets()
    if args.ticket:
        tickets = [t for t in tickets if t.id == args.ticket]
        if not tickets:
            print(f"No ticket with id '{args.ticket}'.", file=sys.stderr)
            return 1

    agent = build_agent(use_mock)
    backend = "mock" if use_mock else f"openai:{settings.model}"
    # Status banner goes to stderr so --json output on stdout stays pure JSON.
    print(f"Triage Agent  |  backend={backend}  |  tickets={len(tickets)}\n", file=sys.stderr)

    json_outputs: list[dict] = []
    failures = 0
    for ticket in tickets:
        try:
            result = agent.run(ticket)
        except Exception as exc:  # one bad ticket must not sink the whole batch
            failures += 1
            print(f"[{ticket.id}] triage failed: {exc}", file=sys.stderr)
            if args.json:
                json_outputs.append({"ticket_id": ticket.id, "error": str(exc)})
            continue
        if args.json:
            json_outputs.append(result.model_dump())
        else:
            _print_report(result)

    if args.json:
        print(json.dumps(json_outputs, ensure_ascii=False, indent=2))
    return 1 if failures else 0
