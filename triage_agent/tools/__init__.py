"""Agent tools: customer history, knowledge-base search, and service status."""
from __future__ import annotations

from .registry import Tool


def default_tools(use_mock: bool = False) -> list[Tool]:
    """The agent's standard tool set — the single place tools are registered.

    Add a capability here (one line) and it is picked up by both the CLI and the
    tests. ``use_mock=True`` forces offline retrieval so ``--mock`` never makes a
    network call.
    """
    from .customer_history import CustomerHistoryTool
    from .knowledge_base import KnowledgeBaseTool
    from .service_status import ServiceStatusTool

    return [
        CustomerHistoryTool(),
        KnowledgeBaseTool(offline=use_mock),
        ServiceStatusTool(),
    ]
